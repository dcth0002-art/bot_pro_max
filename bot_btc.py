import ccxt
import time
import os
import telebot
from dotenv import load_dotenv
from collections import deque
import numpy as np
import traceback

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---

LEVERAGE = 20 # đòn bẩy
DEFAULT_TRADE_AMOUNT = 10 # vốn vào lệnh
INITIAL_BALANCE = 433.27 # tổng vốn
CHECK_INTERVAL = 5 # quét giá
WARMUP_PERIOD = 300 # tích dữ liệu giá
VOL_WINDOW_SIZE = 1000 # thời gian tính volume
COOLDOWN_PERIOD = 600 # thời gian khóa coi sau khi trây xong
VOL_DIFF_THRESHOLD = 1.00 # chênh lệch %
CONFIRMATION_TIME = 60 # thời gian xác nhận tín hiệu
PRICE_SURGE_THRESHOLD = 0.002 # mức tăng giá tối thiểu
STATUS_REPORT_INTERVAL = 1200 # thời gian gửi báo cáo
FEE_RATE = 0.0005 # 0.05% phí
MAX_POSITIONS = 5 # số lệnh tối đa cùng lúc
MAX_DCA = 4

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
        self.positions = []
        self.current_max_positions = MAX_POSITIONS
        self.active_dca_symbol = None
        self.bot_paused = False
        
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

                'waiting_bb': False,
                'bb_wait_candle': 0,
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

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=60)
            closes = [x[4] for x in ohlcv]
            c['price_history'].clear()
            c['price_history'].extend(closes)
            current_price = closes[-1]
            last_candle = ohlcv[-1]
            c['last_candle_time'] = last_candle[0]
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

    def has_open_position(self, symbol):
        for pos in self.positions:
            if pos['symbol'] == symbol:
                    return True
        return False

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

    def is_boll_expanding_smooth(
        self,
        prices,
        period=20,
        lookback=4,
        min_percent=190 #% chênh lệch để so sánh BB cũ với hiện tại
    ):

        prices = list(prices)

        # cần đủ dữ liệu
        if len(prices) < (period * 2) + lookback + 5:
            return False

        def get_bb_width(data):

            middle = np.mean(data)
            std = np.std(data)

            upper = middle + (std * 2)
            lower = middle - (std * 2)

            return upper - lower

        # ===== BB hiện tại =====
        current_width = get_bb_width(
            prices[-period:]
        )

        # ===== lấy BB lớn nhất trong 20 nến cũ =====

        old_widths = []

        # bắt đầu sau vùng lookback
        start_offset = lookback + 1

        for i in range(start_offset, start_offset + period):

            sample = prices[
                -period - i:
                -i
            ]

            width = get_bb_width(sample)

            old_widths.append(width)

        max_old_width = max(old_widths)

        # ===== tính % =====
        bb_percent = (
            current_width / max_old_width
        ) * 100

        # ===== điều kiện % =====
        if bb_percent < min_percent:
            return False

        # ===== kiểm tra mở rộng liên tục =====

        widths = []

        for i in range(lookback + 1):

            offset = i

            sample = prices[
                -period - offset:
                -offset if offset != 0 else None
            ]

            width = get_bb_width(sample)

            widths.append(width)

        # phải mở rộng dần
        for i in range(len(widths) - 1):

            if widths[i] <= widths[i + 1]:
                return False

        return True

    def run(self):
        send_telegram(f"🚀 *Bé nhà đã dậy*\n- đang nạp dữ liệu")
        
        while True:

            if self.bot_paused:

                time.sleep(10)

                continue

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
            if len(self.positions) < self.current_max_positions:
                for symbol in SYMBOLS:
                    current_price = self.update_coin_data(symbol)
                    if current_price is None: continue
                    
                    c = self.coins[symbol]
                    # Coin đã có lệnh rồi thì bỏ qua
                    if self.has_open_position(symbol):
                        continue
                    if len(c['price_history']) < 3:
                        continue

                    price_3p_ago = c['price_history'][-3]
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
                                    c['waiting_bb'] = False
                                    c['bb_wait_candle'] = 0

                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= CONFIRMATION_TIME * 3:
                                        print(f"⌛ [{symbol}] SELL timeout")

                                        c['pending_side'] = None
                                        c['waiting_bb'] = False
                                        c['bb_wait_candle'] = 0
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

                                        bb_expand = self.is_boll_expanding_smooth(
                                            c['price_history']
                                        )

                                        # ===== BB chưa đạt =====
                                        if not bb_expand:

                                            # chưa vào trạng thái chờ
                                            if not c['waiting_bb']:

                                                c['waiting_bb'] = True

                                                # lưu cây nến hiện tại
                                                c['bb_wait_candle'] = c['last_candle_time']

                                                print(f"⌛ [{symbol}] SELL chờ BB mở rộng thêm...")
                                                continue

                                            # đã sang nến mới mà BB vẫn fail
                                            elif c['last_candle_time'] != c['bb_wait_candle']:

                                                print(f"❌ [{symbol}] SELL hủy - nến đóng nhưng BB chưa đạt")

                                                c['waiting_bb'] = False
                                                c['pending_side'] = None

                                                continue

                                            # vẫn đang trong cùng 1 nến
                                            else:

                                                print(f"⌛ [{symbol}] SELL đang đợi nến đóng...")
                                                continue

                                        # ===== BB đạt =====
                                        c['waiting_bb'] = False
                                        c['bb_wait_candle'] = 0

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
                                    c['waiting_bb'] = False
                                    c['bb_wait_candle'] = 0
                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= CONFIRMATION_TIME * 3:
                                        print(f"⌛ [{symbol}] BUY timeout")

                                        c['pending_side'] = None
                                        c['waiting_bb'] = False
                                        c['bb_wait_candle'] = 0
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

                                        bb_expand = self.is_boll_expanding_smooth(
                                            c['price_history']
                                        )

                                        # ===== BB chưa đạt =====
                                        if not bb_expand:

                                            # chưa vào trạng thái chờ
                                            if not c['waiting_bb']:

                                                c['waiting_bb'] = True

                                                # lưu cây nến hiện tại
                                                c['bb_wait_candle'] = c['last_candle_time']

                                                print(f"⌛ [{symbol}] BUY chờ BB mở rộng thêm...")
                                                continue

                                            # đã sang nến mới mà BB vẫn fail
                                            elif c['last_candle_time'] != c['bb_wait_candle']:

                                                print(f"❌ [{symbol}] BUY hủy - nến đóng nhưng BB chưa đạt")

                                                c['waiting_bb'] = False
                                                c['pending_side'] = None

                                                continue

                                            # vẫn đang trong cùng 1 nến
                                            else:

                                                print(f"⌛ [{symbol}] BUY đang đợi nến đóng...")
                                                continue

                                        # ===== BB đạt =====
                                        c['waiting_bb'] = False

                                        if not is_red_candle:
                                            print(f"❌ [{symbol}] BUY bỏ qua - nến chưa đỏ")
                                            c['pending_side'] = None
                                            continue
                                        self.open_position(symbol, 'buy', current_price, sell_diff)
                                        break
                                    else:
                                        c['pending_side'] = None
                    time.sleep(0.01)

            # --- QUẢN LÝ NHIỀU LỆNH ---
            for pos in self.positions[:]:

                symbol = pos['symbol']

                current_price = self.update_coin_data(symbol)

                if current_price:

                    positions = exchange.fetch_positions([symbol])

                    if positions:
                        unrealized_pnl = float(
                            positions[0].get('unrealizedPnl') or 0
                        )
                    else:
                        unrealized_pnl = 0

                    first_price = pos['first_entry_price']

                    if pos['side'] == 'buy':

                        loss_percent = (
                            (first_price - current_price)
                            / first_price
                        ) * 100

                    else:

                        loss_percent = (
                            (current_price - first_price)
                            / first_price
                        ) * 100

                    next_dca_level = (
                        pos['dca_count'] + 1
                    ) * (100 / pos['leverage'])

                    if (
                        loss_percent >= next_dca_level
                        and not pos['waiting_dca']
                        and pos['dca_count'] < MAX_DCA
                    ):

                        pos['waiting_dca'] = True

                        send_telegram(
                            f"⚠️ {symbol} đạt ngưỡng DCA {pos['dca_count']+1}, chờ DCA"
                        )

                    exit_fee = (pos['trade_amount'] * pos['leverage']) * FEE_RATE

                    total_fee = pos['entry_fee'] + exit_fee

                    if pos['is_dca_position']:

                        target_profit = total_fee

                    else:

                        target_profit = (
                            (pos['trade_amount'] * pos['leverage']) * 0.01
                        ) + total_fee

                    if unrealized_pnl >= target_profit:

                        self.close_position(
                            pos,
                            current_price,
                            "Chốt lời (TP) 1%"
                        )

                    #elif unrealized_pnl <= -self.current_trade_amount:
                        #self.close_position(current_price, "Cháy tài khoản (SL 100%)")

            for pos in self.positions:

                if not pos['waiting_dca']:
                    continue

                if self.active_dca_symbol:
                    continue

                if len(self.positions) >= self.current_max_positions:
                    continue

                self.execute_dca(pos)

                break

            if current_time - self.last_status_time >= STATUS_REPORT_INTERVAL:
                self.send_multi_report()
                self.last_status_time = current_time
            time.sleep(CHECK_INTERVAL)

    def open_position(self, symbol, side, price, vol_diff):

        # ===== TỰ ĐỘNG CHỌN ĐÒN BẨY =====
        current_leverage = LEVERAGE

        try:

            exchange.set_leverage(
                current_leverage,
                symbol,
                params={"mgnMode": "cross"}
            )

            print(f"✅ {symbol} dùng leverage x{current_leverage}")

        except Exception:

            print(f"⚠️ {symbol} không hỗ trợ x{LEVERAGE}, thử x10...")

            try:

                current_leverage = 10

                exchange.set_leverage(
                    current_leverage,
                    symbol,
                    params={"mgnMode": "cross"}
                )

                print(f"✅ {symbol} chuyển sang leverage x10")

            except Exception as e:

                print(f"❌ Không set được leverage cho {symbol}: {e}")

                send_telegram(
                    f"❌ {symbol} không hỗ trợ leverage phù hợp"
                )

                return

        trade_amount = min(self.balance, DEFAULT_TRADE_AMOUNT)

        entry_fee = (
            trade_amount *
            current_leverage
        ) * FEE_RATE

        market = exchange.market(symbol)

        contract_size = float(
            market.get("contractSize") or 1
        )

        position_value = (
            trade_amount *
            current_leverage
        )

        amount = (
            position_value /
            (price * contract_size)
        )

        amount = exchange.amount_to_precision(
            symbol,
            amount
        )

        amount_coin = float(amount)

        try:

            print(f"symbol={symbol}")
            print(f"amount={amount_coin}")
            print(f"price={price}")

            order = exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount_coin,
                params={
                    "tdMode": "cross",
                    "posSide": "long"
                    if side == "buy"
                    else "short"
                }
            )

            self.balance -= entry_fee

            print(f"✅ Đã mở lệnh thật: {symbol}")

            self.positions.append({
                'symbol': symbol,
                'side': side,

                'entry_price': price,
                'first_entry_price': price,

                'amount_coin': amount_coin,
                'trade_amount': trade_amount,
                'entry_fee': entry_fee,
                'leverage': current_leverage,

                'dca_count': 0,
                'waiting_dca': False,
                'is_dca_position': False
            })

        except Exception as e:

            print(f"❌ Lỗi mở lệnh: {e}")

            send_telegram(
                f"❌ Lỗi mở lệnh {symbol}:\n`{e}`"
            )

            return

        emoji = "🔴" if side == "sell" else "🟢"

        msg = (
            f"{emoji} *VÀO LỆNH {side.upper()} ({symbol})*\n"
            f"💰 Giá: `{price:,.4f}`\n"
            f"📊 Vol chênh lệch: `+{vol_diff*100:.1f}%` 🔥\n"
            f"💸 Phí mở lệnh: `${entry_fee:.4f}`\n"
            f"💵 Ký quỹ: `${trade_amount:,.2f}`"
        )

        send_telegram(msg)

        for s in SYMBOLS:
            self.coins[s]['pending_side'] = None




    def execute_dca(self, pos):

        symbol = pos['symbol']

        trade_amount = DEFAULT_TRADE_AMOUNT

        try:

            ticker = exchange.fetch_ticker(symbol)

            current_price = ticker['last']

            market = exchange.market(symbol)

            contract_size = float(
                market.get("contractSize") or 1
            )

            position_value = (
                trade_amount *
                pos['leverage']
            )

            amount = (
                position_value /
                (current_price * contract_size)
            )

            amount = exchange.amount_to_precision(
                symbol,
                amount
            )

            exchange.create_order(
                symbol=symbol,
                type='market',
                side=pos['side'],
                amount=float(amount),
                params={
                    "tdMode": "cross",
                    "posSide": "long"
                    if pos['side'] == "buy"
                    else "short"
                }
            )
            time.sleep(2)
            positions = exchange.fetch_positions([symbol])

            if positions:

                pos['entry_price'] = float(
                    positions[0]['entryPrice']
                )

                pos['amount_coin'] = float(
                    positions[0]['contracts']
                )

                print(
                    f"✅ Giá trung bình mới: {pos['entry_price']}"
                )

                print(
                    f"✅ Khối lượng mới: {pos['amount_coin']}"
                )

            pos['trade_amount'] += trade_amount

            pos['entry_fee'] += (
                trade_amount *
                pos['leverage']
            ) * FEE_RATE

            pos['dca_count'] += 1

            pos['waiting_dca'] = False

            pos['is_dca_position'] = True

            self.current_max_positions -= 1

            self.active_dca_symbol = symbol

            # mở khóa sau khi DCA hoàn tất
            self.active_dca_symbol = None

            send_telegram(
                f"📉 DCA lần {pos['dca_count']} cho {symbol}\n"
                f"📦 Slot còn: {self.current_max_positions}"
            )

            if pos['dca_count'] >= MAX_DCA:

                self.bot_paused = True

                send_telegram(
                    f"🚨 {symbol} đã DCA tối đa {MAX_DCA} lần\n"
                    f"🚨 Bot tạm dừng để xử lý thủ công"
                )

        except Exception as e:

            print(f"DCA lỗi: {e}")




    def close_position(self, pos, price, reason):
        symbol = pos['symbol']

        # Xác định chiều đóng lệnh
        close_side = 'sell' if pos['side'] == 'buy' else 'buy'

        # Lấy PNL trước khi đóng lệnh
        positions = exchange.fetch_positions([symbol])

        if positions:

            raw_pnl = float(
                positions[0].get('unrealizedPnl') or 0
            )

        else:

            raw_pnl = 0

        # Đóng lệnh thật trên OKX
        try:
            exchange.create_market_order(
                symbol,
                close_side,
                pos['amount_coin'],
                params={
                    "tdMode": "cross",
                    "reduceOnly": True,
                    "posSide": "long" if pos['side'] == "buy" else "short"
                }
            )

            print(f"✅ Đã đóng lệnh thật: {symbol}")

        except Exception as e:

            print(f"❌ Lỗi đóng lệnh thật: {e}")

            send_telegram(
                f"❌ Lỗi đóng lệnh thật {symbol}:\n`{e}`"
            )

            return

        exit_fee = (pos['trade_amount'] * pos['leverage']) * FEE_RATE

        total_fee = pos['entry_fee'] + exit_fee

        # Trừ đúng tổng phí thật
        real_net_profit = raw_pnl - total_fee

        self.balance += (raw_pnl - exit_fee)

        self.coins[symbol]['last_close_time'] = time.time()

        status = "LÃI ✅" if real_net_profit > 0 else "LỖ ❌"

        msg = (
            f"✅ *ĐÓNG LỆNH {symbol}*\n"
            f"📝 Lý do: {reason}\n"
            f"💸 Tổng phí (vào+ra): `${total_fee:.4f}`\n"
            f"💰  ({status})\n"
        )

        send_telegram(msg)
        if pos['dca_count'] > 0:

            self.current_max_positions = MAX_POSITIONS

            self.active_dca_symbol = None

            self.bot_paused = False

        # Reset position
        self.positions.remove(pos)

    def send_multi_report(self):

        if self.positions:

            symbols = [
                pos['symbol']
                for pos in self.positions
            ]

            status_text = ", ".join(symbols)

        else:

            status_text = "Đang săn tín hiệu đảo chiều..."

        msg = (
            f"📊 *GIÁM SÁT HỆ THỐNG*\n"
            f"📍 {status_text}\n"
            f"🏦 Vốn: `${self.balance:,.2f}$`\n"
            f"📦 Số lệnh: `{len(self.positions)}/{MAX_POSITIONS}`"
        )

        send_telegram(msg)

if __name__ == "__main__":

    while True:

        try:

            bot_trading = TradingBot()

            bot_trading.run()

        except Exception as e:

            error_text = traceback.format_exc()

            print(error_text)

            send_telegram(
                f"💥 Crash toàn bot:\n```{error_text[:3500]}```"
            )

            time.sleep(10)
