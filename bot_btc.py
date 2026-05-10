import ccxt
import time
import os
import telebot
from dotenv import load_dotenv
from collections import deque
import numpy as np

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---

LEVERAGE = 10 # đòn bẩy
DEFAULT_TRADE_AMOUNT = 5 # vốn vào lệnh
INITIAL_BALANCE = 24.20 # tổng vốn
CHECK_INTERVAL = 5 # quét giá
WARMUP_PERIOD = 300 # tích dữ liệu giá
VOL_WINDOW_SIZE = 1800 # thời gian tính volume
COOLDOWN_PERIOD = 300 # thời gian khóa coi sau khi trây xong
VOL_DIFF_THRESHOLD = 1.00 # chênh lệch %
CONFIRMATION_TIME = 60 # thời gian xác nhận tín hiệu
PRICE_SURGE_THRESHOLD = 0.002 # mức tăng giá tối thiểu
STATUS_REPORT_INTERVAL = 1200 # thời gian gửi báo cáo
FEE_RATE = 0.0005 # 0.05% phí

# --- THÔNG TIN TELEGRAM ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

exchange = ccxt.okx({
    'apiKey': os.getenv("OKX_API_KEY"),
    'secret': os.getenv("OKX_SECRET_KEY"),
    'password': os.getenv("OKX_PASSPHRASE"),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'
    }
})
markets = exchange.load_markets()
# Lấy toàn bộ Futures USDT đang hoạt động
SYMBOLS = [
    symbol for symbol, market in markets.items()
    if market.get('swap')
    and market.get('quote') == 'USDT'
    and market.get('active') == True
]
print(f"Tổng coin futures quét: {len(SYMBOLS)}", flush=True)
for s in SYMBOLS[:20]:
    print(f"✅ {s}", flush=True)


bot = telebot.TeleBot(TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

def send_telegram(message):
    print(message)
    if bot and TELEGRAM_CHAT_ID:
        try:
            bot.send_message(TELEGRAM_CHAT_ID, message, parse_mode='Markdown')
        except Exception as e:
            print(f"Lỗi gửi Telegram: {e}")

class TradingBot:
    def __init__(self):
        self.balance = INITIAL_BALANCE
        self.current_position = None
        self.active_symbol = None
        self.entry_price = 0
        self.amount_coin = 0
        self.current_trade_amount = 0
        
        self.coins = {}
        for symbol in SYMBOLS:
            self.coins[symbol] = {
                'vol_trades': deque(),
                'last_trade_id': None,
                'price_history': deque(maxlen=310),
                'pending_side': None,
                'trigger_price': 0,
                'trigger_time': 0,
                'trigger_vol_diff': 0, 
                'last_close_time': 0,
                'total_buy_30p': 0.0,
                'total_sell_30p': 0.0,
                'last_open': 0,
                'last_close': 0,
            }
        
        self.start_time = time.time()
        self.last_status_time = time.time()
        self.is_warmed_up = False

    def update_coin_data(self, symbol):
        try:
            c = self.coins[symbol]
            current_time = time.time()
            trades = exchange.fetch_trades(symbol, limit=50) 
            new_trades = []
            if c['last_trade_id'] is None:
                new_trades = trades
            else:
                for trade in reversed(trades):
                    if trade['id'] == c['last_trade_id']:
                        break
                    new_trades.insert(0, trade)
            if new_trades:
                for t in new_trades:
                    c['vol_trades'].append((t['timestamp'] / 1000, t['side'], t['amount']))
                c['last_trade_id'] = new_trades[-1]['id']

            cutoff = current_time - VOL_WINDOW_SIZE
            while c['vol_trades'] and c['vol_trades'][0][0] < cutoff:
                c['vol_trades'].popleft()

            c['total_buy_30p'] = sum(t[2] for t in c['vol_trades'] if t[1] == 'buy')
            c['total_sell_30p'] = sum(t[2] for t in c['vol_trades'] if t[1] == 'sell')

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=60)
            closes = [x[4] for x in ohlcv]
            c['price_history'].clear()
            c['price_history'].extend(closes)
            current_price = closes[-1]
            last_candle = ohlcv[-1]
            c['last_open'] = last_candle[1]
            c['last_close'] = last_candle[4]
            return current_price


        except Exception as e:
            print(f"Lỗi cập nhật {symbol}: {e}")
            return None

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        if len(prices) < period:
            return None, None, None

        closes = np.array(list(prices)[-period:])

        middle = np.mean(closes)
        std = np.std(closes)

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def is_valid_bb_zone(self, side, current_price, upper, middle, lower):

        if upper is None or middle is None or lower is None:
            return False

        # SELL gần dải trên
        if side == 'sell':
            upper_zone = middle + (upper - middle) * 0.75
            return current_price >= upper_zone

        # BUY gần dải dưới
        elif side == 'buy':
            lower_zone = lower + (middle - lower) * 0.25
            return current_price <= lower_zone

        return False

    def run(self):
        send_telegram(f"🚀 *Bé nhà đã dậy*\n- đang nạp dữ liệu")
        
        while True:
            current_time = time.time()
            if not self.is_warmed_up:
                if current_time - self.start_time >= WARMUP_PERIOD:
                    self.is_warmed_up = True
                    send_telegram("✅ *Nạp dữ liệu xong!* Bắt đầu săn tìm cơ hội.")
                else:
                    for sym in SYMBOLS:
                        self.update_coin_data(sym)
                        time.sleep(0.01)
                    continue

            # --- TRƯỜNG HỢP 1: ĐI SĂN TÍN HIỆU ---
            if self.active_symbol is None:
                for symbol in SYMBOLS:
                    current_price = self.update_coin_data(symbol)
                    if current_price is None: continue
                    
                    c = self.coins[symbol]
                    price_3p_ago = c['price_history'][-3] if len(c['price_history']) >= 3 else c['price_history'][0]
                    buy_diff = (c['total_buy_30p'] - c['total_sell_30p']) / c['total_sell_30p'] if c['total_sell_30p'] > 0 else 1.0
                    sell_diff = (c['total_sell_30p'] - c['total_buy_30p']) / c['total_buy_30p'] if c['total_buy_30p'] > 0 else 1.0

                    if current_time - c['last_close_time'] >= COOLDOWN_PERIOD:
                        if c['pending_side'] is None:
                            if buy_diff > VOL_DIFF_THRESHOLD and current_price > price_3p_ago:
                                c['pending_side'] = 'sell_trigger'
                                c['trigger_price'] = current_price
                                c['trigger_time'] = current_time
                                c['trigger_vol_diff'] = buy_diff
                                print(f"🔍 [{symbol}] Chờ SHORT đảo chiều...")
                            elif sell_diff > VOL_DIFF_THRESHOLD and current_price < price_3p_ago:
                                c['pending_side'] = 'buy_trigger'
                                c['trigger_price'] = current_price
                                c['trigger_time'] = current_time
                                c['trigger_vol_diff'] = sell_diff
                                print(f"🔍 [{symbol}] Chờ LONG đảo chiều...")
                        else:
                            elapsed = current_time - c['trigger_time']
                            price_change = (current_price - c['trigger_price']) / c['trigger_price']
                            
                            if c['pending_side'] == 'sell_trigger':
                                if current_price < c['trigger_price'] * 0.999:
                                    c['pending_side'] = None

                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= CONFIRMATION_TIME * 3:
                                        print(f"⌛ [{symbol}] SELL timeout")
                                        c['pending_side'] = None
                                        continue
                                    if price_change >= PRICE_SURGE_THRESHOLD and buy_diff > c['trigger_vol_diff']:
                                        upper, middle, lower = self.calculate_bollinger_bands(c['price_history'])

                                        is_green_candle = c['last_close'] > c['last_open'] * 1.003

                                        valid_bb = self.is_valid_bb_zone(
                                            'sell',
                                            current_price,
                                            upper,
                                            middle,
                                            lower
                                        )

                                        if not valid_bb:
                                            print(f"⌛ [{symbol}] SELL chờ chạm BB trên...")
                                            continue

                                        if not is_green_candle:
                                            print(f"❌ [{symbol}] SELL bỏ qua - nến chưa xanh")
                                            c['pending_side'] = None
                                            continue
                                        self.open_position(symbol, 'sell', current_price, buy_diff)
                                        break
                                    else:
                                        c['pending_side'] = None
                            
                            elif c['pending_side'] == 'buy_trigger':
                                if current_price > c['trigger_price'] * 1.001:
                                    c['pending_side'] = None
                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= CONFIRMATION_TIME * 3:
                                        print(f"⌛ [{symbol}] BUY timeout")
                                        c['pending_side'] = None
                                        continue
                                    if abs(price_change) >= PRICE_SURGE_THRESHOLD and sell_diff > c['trigger_vol_diff']:
                                        upper, middle, lower = self.calculate_bollinger_bands(c['price_history'])

                                        is_red_candle = c['last_close'] < c['last_open'] * 0.997

                                        valid_bb = self.is_valid_bb_zone(
                                            'buy',
                                            current_price,
                                            upper,
                                            middle,
                                            lower
                                        )

                                        if not valid_bb:
                                            print(f"⌛ [{symbol}] BUY chờ chạm BB dưới...")
                                            continue

                                        if not is_red_candle:
                                            print(f"❌ [{symbol}] BUY bỏ qua - nến chưa đỏ")
                                            c['pending_side'] = None
                                            continue
                                        self.open_position(symbol, 'buy', current_price, sell_diff)
                                        break
                                    else:
                                        c['pending_side'] = None
                    time.sleep(0.01)

            # --- TRƯỜNG HỢP 2: ĐANG GIỮ LỆNH (CHỈ TP/SL) ---
            else:
                symbol = self.active_symbol
                current_price = self.update_coin_data(symbol)
                if current_price:
                    if self.current_position == 'buy':
                        raw_pnl = (current_price - self.entry_price) * self.amount_coin
                    else:
                        raw_pnl = (self.entry_price - current_price) * self.amount_coin
                    
                    target_profit = (self.current_trade_amount * 0.02) + 1.0 # 2$ lãi + 1$ phí
                    
                    if raw_pnl >= target_profit:
                        self.close_position(current_price, "Chốt lời (TP) lãi ròng 2%")
                    elif raw_pnl <= -self.current_trade_amount:
                        self.close_position(current_price, "Cháy tài khoản (SL 100%)")

            if current_time - self.last_status_time >= STATUS_REPORT_INTERVAL:
                self.send_multi_report()
                self.last_status_time = current_time
            time.sleep(CHECK_INTERVAL)

    def open_position(self, symbol, side, price, vol_diff):
        exchange.set_leverage(
            LEVERAGE,
            symbol,
            params={"mgnMode": "cross"}
        )

        self.current_trade_amount = min(self.balance, DEFAULT_TRADE_AMOUNT)
        entry_fee = (self.current_trade_amount * LEVERAGE) * FEE_RATE



        amount = exchange.amount_to_precision(
            symbol,
            (self.current_trade_amount * LEVERAGE) / price
        )

        self.amount_coin = float(amount)

        exchange.set_leverage(
            LEVERAGE,
            symbol,
            params={"mgnMode": "cross"}
        )

        try:
            print(f"symbol={symbol}")
            print(f"amount={self.amount_coin}")
            print(f"price={price}")
            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=self.amount_coin,
                params={
                    "tdMode": "cross",
                    "posSide": "long" if side == "buy" else "short"
                }
            )
            self.balance -= entry_fee
            print(f"✅ Đã mở lệnh thật: {symbol}")
            self.active_symbol = symbol
            self.current_position = side
            self.entry_price = price

        except Exception as e:
            print(f"❌ Lỗi mở lệnh: {e}")
            send_telegram(f"❌ Lỗi mở lệnh {symbol}:\n`{e}`")
            return
        
        emoji = "🔴" if side == 'sell' else "🟢"
        msg = (
            f"{emoji} *VÀO LỆNH {side.upper()} ({symbol})*\n"
            f"💰 Giá: `{price:,.4f}`\n"
            f"📊 Vol chênh lệch: `+{vol_diff*100:.1f}%` 🔥\n"
            f"💸 Phí mở lệnh: `$0.50` (Đã trừ)\n"
            f"💵 Ký quỹ: `${self.current_trade_amount:,.2f}`"
        )
        send_telegram(msg)
        for s in SYMBOLS: self.coins[s]['pending_side'] = None

    def close_position(self, price, reason):
        symbol = self.active_symbol

        # Xác định chiều đóng lệnh
        close_side = 'sell' if self.current_position == 'buy' else 'buy'

        # Đóng lệnh thật trên OKX
        try:
            exchange.create_market_order(
                symbol,
                close_side,
                self.amount_coin,
                params={
                    "tdMode": "cross",
                    "reduceOnly": True,
                    "posSide": "long" if self.current_position == "buy" else "short"
                }
            )
            print(f"✅ Đã đóng lệnh thật: {symbol}")

        except Exception as e:
            print(f"❌ Lỗi đóng lệnh thật: {e}")
            send_telegram(f"❌ Lỗi đóng lệnh thật {symbol}:\n`{e}`")
            return

        # Tính PNL giả lập để báo cáo Telegram
        if self.current_position == 'buy':
            raw_pnl = (price - self.entry_price) * self.amount_coin
        else:
            raw_pnl = (self.entry_price - price) * self.amount_coin

        exit_fee = (self.current_trade_amount * LEVERAGE) * FEE_RATE

        # Trừ tổng phí vào + ra
        real_net_profit = raw_pnl - 1.0

        self.balance += (raw_pnl - exit_fee)

        self.coins[symbol]['last_close_time'] = time.time()

        status = "LÃI ✅" if real_net_profit > 0 else "LỖ ❌"

        msg = (
            f"⚠️ *ĐÓNG LỆNH {symbol}*\n"
            f"📝 Lý do: {reason}\n"
            f"🏁 Lợi nhuận thô: `{raw_pnl:,.2f}$`\n"
            f"💸 Tổng phí (vào+ra): `$1.00`\n"
            f"💰 Lãi ròng thực tế: `{real_net_profit:,.2f}$` ({status})\n"
            f"🏦 Số dư cuối: `${self.balance:,.2f}$`"
        )

        send_telegram(msg)

        # Reset position
        self.active_symbol = None
        self.current_position = None

    def send_multi_report(self):
        msg = f"📊 *GIÁM SÁT HỆ THỐNG*\n📍 {'Đang trade: ' + self.active_symbol if self.active_symbol else 'Đang săn tín hiệu đảo chiều...'}\n🏦 Vốn: `${self.balance:,.2f}$`"
        send_telegram(msg)

if __name__ == "__main__":
    bot_trading = TradingBot()
    try:
        bot_trading.run()
    except KeyboardInterrupt:
        send_telegram("🛑 *Bot đã dừng.*")
