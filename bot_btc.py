import ccxt
import time
import os
import telebot
from dotenv import load_dotenv
from collections import deque
import numpy as np
import traceback
import threading

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---

LEVERAGE = 20 # đòn bẩy
DEFAULT_TRADE_AMOUNT = 5 # vốn vào lệnh
INITIAL_BALANCE = 422.25 # tổng vốn
CHECK_INTERVAL = 5 # quét giá
WARMUP_PERIOD = 300 # tích dữ liệu giá
VOL_WINDOW_SIZE = 1000 # thời gian tính volume
COOLDOWN_PERIOD = 600 # thời gian khóa coi sau khi trây xong
VOL_DIFF_THRESHOLD = 1.00 # chênh lệch %
CONFIRMATION_TIME = 60 # thời gian xác nhận tín hiệu
PRICE_SURGE_THRESHOLD = 0.002 # mức tăng giá tối thiểu
STATUS_REPORT_INTERVAL = 1800 # thời gian gửi báo cáo
FEE_RATE = 0.0005 # 0.05% phí
MAX_POSITIONS = 10 # số lệnh tối đa cùng lúc
MAX_DCA = 10
MARGIN_MODE = "isolated" # isolated = Cô lập trên OKX, cross = Chéo

# --- BB DYNAMIC MIN_PERCENT THEO KHUNG 1M ---
BB_1M_CANDLE_THRESHOLD = 12 # nếu khung 1M có dưới 12 nến thì dùng min_percent cao hơn
BB_MIN_PERCENT_LOW_HISTORY = 200 # coin mới / ít dữ liệu 1M
BB_MIN_PERCENT_ENOUGH_HISTORY = 130 # coin có trên hoặc bằng 12 nến 1M
BB_1M_CACHE_SECONDS = 6 * 60 * 60 # cache 6 tiếng để tránh gọi API quá nhiều

# --- TRAILING TP / CẮT BỚT LỆNH ÂM ---
TRAILING_TP_CALLBACK_RATIO = 0.30 # tụt 30% phần lời dùng làm khoảng kéo SL dương
TRAILING_TP_MIN_GAP = 0.20 # tối thiểu cho giá thở, đơn vị USDT
LOSS_CUT_TRIGGER_PERCENT = 50 # lệnh âm trên 50% ký quỹ thì xét cắt bớt
LOSS_CUT_PROFIT_USAGE = 0.25 # dùng tối đa 25% tiền lời TP để cắt lỗ, TP 2$ => cắt khoảng 0.5$
MIN_PARTIAL_CLOSE_AMOUNT_RATIO = 0.05 # không đóng từng phần quá nhỏ dưới 5% khối lượng lệnh

# --- BƠM LẠI LỆNH ĐÃ BỊ CẮT NHỎ ---
REBUILD_TRIGGER_TRADE_AMOUNT = 1 # khi ký quỹ lệnh còn <= 2$ thì xét bơm lại
REBUILD_ADD_AMOUNT = 3 # số tiền ký quỹ thêm vào lệnh nhỏ để kéo lại giá trung bình
REBUILD_MIN_LOSS_PERCENT = 70 # chỉ bơm lại nếu lệnh nhỏ vẫn đang âm ít nhất 70% ký quỹ

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
ACTIVE_TRADING_BOT = None
TELEGRAM_POLLING_STARTED = False

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
        self.next_position_id = 1
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
        self.bb_1m_cache = {}

        # D/T Telegram: D = dừng săn lệnh mới, T = tiếp tục săn lệnh mới.
        # Các lệnh đang mở vẫn được quản lý TP/DCA/cắt lỗ bình thường.
        self.search_paused = False

        global ACTIVE_TRADING_BOT
        ACTIVE_TRADING_BOT = self

        self.setup_telegram_commands()

    def count_root_positions(self):
        # Chỉ đếm lệnh gốc để giới hạn slot săn lệnh mới.
        # Lệnh DCA riêng không chiếm slot, nên full 5/5 vẫn DCA được.
        return sum(
            1 for pos in self.positions
            if not pos.get('is_dca_position')
        )

    def setup_telegram_commands(self):
        global TELEGRAM_POLLING_STARTED

        if not bot:
            return

        if TELEGRAM_POLLING_STARTED:
            return

        @bot.message_handler(func=lambda message: True)
        def handle_telegram_command(message):
            try:
                if TELEGRAM_CHAT_ID and str(message.chat.id) != str(TELEGRAM_CHAT_ID):
                    return

                text = (message.text or '').strip().upper()
                active_bot = ACTIVE_TRADING_BOT

                if active_bot is None:
                    return

                if text == 'D':
                    active_bot.search_paused = True
                    send_telegram(
                        "DỖI KO TÌM LỆNH NỮA"
                    )

                elif text == 'T':
                    active_bot.search_paused = False
                    send_telegram(
                        "NỂ LẮM MỚI LÀM TIẾP ĐẤY"
                    )

            except Exception as e:
                print(f"Lỗi xử lý lệnh Telegram: {e}")

        def polling_worker():
            while True:
                try:
                    bot.infinity_polling(
                        timeout=20,
                        long_polling_timeout=20,
                        skip_pending=True
                    )
                except Exception as e:
                    print(f"Telegram polling lỗi: {e}")
                    time.sleep(5)

        threading.Thread(
            target=polling_worker,
            daemon=True
        ).start()

        TELEGRAM_POLLING_STARTED = True

    def make_position_id(self):
        position_id = self.next_position_id
        self.next_position_id += 1
        return position_id

    def refresh_root_dca_count(self, root_id):
        # DCA reset: nếu các lệnh DCA riêng đã TP/đóng hết,
        # lệnh gốc sẽ quay về DCA0 để lần sau có thể mở lại DCA1.
        root_pos = None

        for p in self.positions:
            if p.get('position_id') == root_id and not p.get('is_dca_position'):
                root_pos = p
                break

        if not root_pos:
            return

        active_dca_numbers = [
            p.get('dca_number', 0)
            for p in self.positions
            if p.get('root_id') == root_id and p.get('is_dca_position')
        ]

        new_dca_count = max(active_dca_numbers) if active_dca_numbers else 0

        old_dca_count = root_pos.get('dca_count', 0)

        root_pos['dca_count'] = new_dca_count
        root_pos['waiting_dca'] = False

        if new_dca_count == 0 and old_dca_count != 0:
            send_telegram(
                f"🔄 {root_pos['symbol']} DCA riêng đã đóng hết, reset về DCA0"
            )


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

    def calculate_virtual_pnl(self, pos, current_price):

        market = exchange.market(pos['symbol'])

        contract_size = float(
            market.get("contractSize") or 1
        )

        entry_price = pos['entry_price']
        amount_coin = pos['amount_coin']

        if pos['side'] == 'buy':
            pnl = (
                current_price - entry_price
            ) * amount_coin * contract_size
        else:
            pnl = (
                entry_price - current_price
            ) * amount_coin * contract_size

        return pnl

    def get_dynamic_bb_min_percent(self, symbol):

        # Kiểm tra số nến khung 1M của coin.
        # Coin có lịch sử 1M ngắn (< 12 nến) thì yêu cầu BB mở rộng mạnh hơn.
        now = time.time()

        cached = self.bb_1m_cache.get(symbol)

        if cached and now - cached['time'] < BB_1M_CACHE_SECONDS:
            return cached['min_percent']

        try:

            monthly_ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe='1M',
                limit=20
            )

            candle_count = len(monthly_ohlcv)

            if candle_count < BB_1M_CANDLE_THRESHOLD:
                min_percent = BB_MIN_PERCENT_LOW_HISTORY
            else:
                min_percent = BB_MIN_PERCENT_ENOUGH_HISTORY

            self.bb_1m_cache[symbol] = {
                'time': now,
                'candle_count': candle_count,
                'min_percent': min_percent
            }

            print(
                f"📊 [{symbol}] 1M có {candle_count} nến => BB min_percent={min_percent}"
            )

            return min_percent

        except Exception as e:

            print(
                f"⚠️ Không kiểm tra được nến 1M của {symbol}: {e}"
            )

            # Nếu lỗi API thì dùng mức an toàn hơn.
            return BB_MIN_PERCENT_LOW_HISTORY

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
            # Chỉ slot lệnh gốc mới giới hạn tìm lệnh mới.
            # Lệnh DCA riêng không tính vào slot này.
            if (not self.search_paused) and self.count_root_positions() < self.current_max_positions:
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

                                        bb_min_percent = self.get_dynamic_bb_min_percent(symbol)

                                        bb_expand = self.is_boll_expanding_smooth(
                                            c['price_history'],
                                            min_percent=bb_min_percent
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

                                        bb_min_percent = self.get_dynamic_bb_min_percent(symbol)

                                        bb_expand = self.is_boll_expanding_smooth(
                                            c['price_history'],
                                            min_percent=bb_min_percent
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

                    unrealized_pnl = self.calculate_virtual_pnl(
                        pos,
                        current_price
                    )

                    # Chỉ lệnh gốc mới được kích hoạt các lần DCA.
                    # Các lệnh DCA là lệnh riêng, TP riêng, không tự mở DCA tiếp.
                    if not pos.get('is_dca_position'):

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
                                f"⚠️ {symbol} đạt ngưỡng DCA {pos['dca_count']+1}, chờ mở lệnh DCA riêng"
                            )

                    exit_fee = (pos['trade_amount'] * pos['leverage']) * FEE_RATE

                    total_fee = pos['entry_fee'] + exit_fee

                    target_profit = (
                        (pos['trade_amount'] * pos['leverage']) * 0.01
                    ) + total_fee

                    # ===== TRAILING TP KIỂU SL DƯƠNG =====
                    # Vẫn giữ mốc TP cũ. Khi đạt TP thì chưa đóng ngay,
                    # mà bật trailing để lợi nhuận còn tăng thì tiếp tục giữ.
                    if not pos.get('tp_trailing_active'):

                        if unrealized_pnl >= target_profit:

                            pos['tp_trailing_active'] = True
                            pos['tp_peak_pnl'] = unrealized_pnl
                            pos['tp_trailing_stop_pnl'] = target_profit

                            send_telegram(
                                f"🟢 {symbol} đã vào vùng TP, bật trailing TP\n"
                                f"🎯 TP gốc: `${target_profit:.4f}`\n"
                                f"📈 Lời hiện tại: `${unrealized_pnl:.4f}`"
                            )

                    else:

                        if unrealized_pnl > pos.get('tp_peak_pnl', 0):

                            pos['tp_peak_pnl'] = unrealized_pnl

                            trail_gap = max(
                                target_profit * TRAILING_TP_CALLBACK_RATIO,
                                TRAILING_TP_MIN_GAP
                            )

                            pos['tp_trailing_stop_pnl'] = max(
                                target_profit,
                                pos['tp_peak_pnl'] - trail_gap
                            )

                            print(
                                f"📈 {symbol} trailing TP peak={pos['tp_peak_pnl']:.4f}, "
                                f"stop={pos['tp_trailing_stop_pnl']:.4f}"
                            )

                        if unrealized_pnl <= pos.get('tp_trailing_stop_pnl', target_profit):

                            closed_profit = self.close_position(
                                pos,
                                current_price,
                                "Trailing TP - lợi nhuận yếu đi nên chốt"
                            )

                            if closed_profit and closed_profit > 0:

                                self.reduce_biggest_loser_after_tp(
                                    closed_profit
                                )

                    #elif unrealized_pnl <= -self.current_trade_amount:
                        #self.close_position(current_price, "Cháy tài khoản (SL 100%)")

            # Sau khi các lệnh âm bị cắt nhỏ, nếu lệnh còn khoảng 2$
            # thì bơm thêm tiền để kéo lại giá trung bình cho lệnh đó.
            for pos in self.positions[:]:

                self.rebuild_small_loser_position(pos)

            for pos in self.positions:

                if not pos['waiting_dca']:
                    continue

                if self.active_dca_symbol:
                    continue

                # DCA riêng không bị chặn bởi slot 5/5 lệnh gốc.
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
                params={"mgnMode": MARGIN_MODE}
            )

            print(f"✅ {symbol} dùng leverage x{current_leverage}")

        except Exception:

            print(f"⚠️ {symbol} không hỗ trợ x{LEVERAGE}, thử x10...")

            try:

                current_leverage = 10

                exchange.set_leverage(
                    current_leverage,
                    symbol,
                    params={"mgnMode": MARGIN_MODE}
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
                    "tdMode": MARGIN_MODE,
                    "posSide": "long"
                    if side == "buy"
                    else "short"
                }
            )

            self.balance -= entry_fee

            print(f"✅ Đã mở lệnh thật: {symbol}")

            position_id = self.make_position_id()

            self.positions.append({
                'position_id': position_id,
                'root_id': position_id,
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
                'is_dca_position': False,

                'tp_trailing_active': False,
                'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0,

                'rebuild_count': 0
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

        trade_amount = min(self.balance, DEFAULT_TRADE_AMOUNT)

        dca_number = pos['dca_count'] + 1

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

            amount_coin = float(amount)

            entry_fee = (
                trade_amount *
                pos['leverage']
            ) * FEE_RATE

            # Mở thêm 1 order cùng chiều, nhưng bot lưu thành 1 vị thế riêng.
            # Lưu ý: trên OKX cùng symbol + cùng posSide vẫn gộp ngoài sàn,
            # còn bot sẽ quản lý TP/đóng từng phần bằng reduceOnly.
            exchange.create_order(
                symbol=symbol,
                type='market',
                side=pos['side'],
                amount=amount_coin,
                params={
                    "tdMode": MARGIN_MODE,
                    "posSide": "long"
                    if pos['side'] == "buy"
                    else "short"
                }
            )

            self.balance -= entry_fee

            dca_position_id = self.make_position_id()
            root_id = pos.get('root_id', pos.get('position_id'))

            self.positions.append({
                'position_id': dca_position_id,
                'root_id': root_id,
                'symbol': symbol,
                'side': pos['side'],

                'entry_price': current_price,
                'first_entry_price': current_price,

                'amount_coin': amount_coin,
                'trade_amount': trade_amount,
                'entry_fee': entry_fee,
                'leverage': pos['leverage'],

                'dca_count': 0,
                'waiting_dca': False,
                'is_dca_position': True,
                'dca_number': dca_number,
                'parent_entry_price': pos['first_entry_price'],

                'tp_trailing_active': False,
                'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0,

                'rebuild_count': 0
            })

            # Lệnh gốc chỉ ghi nhận đã mở DCA lần mấy,
            # KHÔNG sửa entry_price, KHÔNG cộng trade_amount,
            # KHÔNG kéo giá trung bình nữa.
            pos['dca_count'] = dca_number
            pos['waiting_dca'] = False

            # DCA riêng đã được tính bằng len(self.positions).
            # Không trừ current_max_positions nữa, nếu trừ sẽ làm bot bị chặn DCA sớm
            # và không mở đủ MAX_DCA như mong muốn.

            send_telegram(
                f"📉 Đã mở LỆNH DCA RIÊNG lần {dca_number} cho {symbol}\n"
                f"💰 Giá DCA: `{current_price}`\n"
                f"💵 Ký quỹ: `${trade_amount:,.2f}`\n"
                f"📦 Slot lệnh gốc: {self.count_root_positions()}/{self.current_max_positions}\n"
                f"📌 Tổng lệnh bot đang quản lý: {len(self.positions)}"
            )

            if pos['dca_count'] >= MAX_DCA:

                self.bot_paused = True

                send_telegram(
                    f"🚨 {symbol} đã mở đủ {MAX_DCA} lệnh DCA riêng\n"
                    f"🚨 Bot tạm dừng để xử lý thủ công"
                )

        except Exception as e:

            pos['waiting_dca'] = False

            print(f"DCA lỗi: {e}")

            send_telegram(
                f"❌ Lỗi mở lệnh DCA riêng {symbol}:\n`{e}`"
            )




    def rebuild_small_loser_position(self, pos):

        # Khi một lệnh bị các lần TP cắt bớt đến mức còn nhỏ,
        # bot sẽ thêm tiền vào chính lệnh đó để kéo giá trung bình mới.
        # Ví dụ lệnh ban đầu 10$, bị cắt còn khoảng 2$,
        # bot thêm 6$ => lệnh còn khoảng 8$ và có entry_price trung bình mới.
        if pos.get('trade_amount', 0) > REBUILD_TRIGGER_TRADE_AMOUNT:
            return False

        if pos.get('rebuilding'):
            return False

        if self.balance < REBUILD_ADD_AMOUNT:
            print(
                f"⌛ {pos['symbol']} còn nhỏ nhưng balance không đủ để bơm lại"
            )
            return False

        symbol = pos['symbol']

        current_price = self.update_coin_data(symbol)

        if not current_price:
            return False

        current_pnl = self.calculate_virtual_pnl(
            pos,
            current_price
        )

        loss_percent = (
            abs(current_pnl) / pos['trade_amount']
        ) * 100 if pos['trade_amount'] > 0 else 0

        # Chỉ bơm lại lệnh đang âm, tránh lệnh nhỏ nhưng đang gần hòa/lãi cũng bị bơm.
        if current_pnl >= 0 or loss_percent < REBUILD_MIN_LOSS_PERCENT:
            return False

        pos['rebuilding'] = True

        try:

            trade_amount = REBUILD_ADD_AMOUNT

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

            add_amount_coin = float(amount)

            if add_amount_coin <= 0:
                pos['rebuilding'] = False
                return False

            exchange.create_order(
                symbol=symbol,
                type='market',
                side=pos['side'],
                amount=add_amount_coin,
                params={
                    "tdMode": MARGIN_MODE,
                    "posSide": "long"
                    if pos['side'] == "buy"
                    else "short"
                }
            )

            old_amount_coin = pos['amount_coin']
            old_entry_price = pos['entry_price']

            new_amount_coin = old_amount_coin + add_amount_coin

            # Giá trung bình riêng trong bot cho phần lệnh này.
            new_entry_price = (
                (old_entry_price * old_amount_coin) +
                (current_price * add_amount_coin)
            ) / new_amount_coin

            entry_fee = (
                trade_amount *
                pos['leverage']
            ) * FEE_RATE

            pos['entry_price'] = new_entry_price
            pos['first_entry_price'] = new_entry_price
            pos['amount_coin'] = new_amount_coin
            pos['trade_amount'] += trade_amount
            pos['entry_fee'] += entry_fee
            pos['rebuild_count'] = pos.get('rebuild_count', 0) + 1

            # Sau khi kéo giá trung bình thì reset trailing TP để tính lại từ đầu.
            pos['tp_trailing_active'] = False
            pos['tp_peak_pnl'] = 0
            pos['tp_trailing_stop_pnl'] = 0

            self.balance -= entry_fee

            send_telegram(
                f"🧱 *BƠM LẠI LỆNH NHỎ*\n"
                f"📍 `{symbol}`\n"
                f"📉 Trước khi bơm đang âm: `{loss_percent:.1f}%` ký quỹ\n"
                f"💵 Ký quỹ cũ còn: `${pos['trade_amount'] - trade_amount:.4f}`\n"
                f"➕ Thêm ký quỹ: `${trade_amount:.2f}`\n"
                f"📊 Ký quỹ mới: `${pos['trade_amount']:.4f}`\n"
                f"🎯 Giá trung bình mới: `{new_entry_price}`"
            )

            pos['rebuilding'] = False

            return True

        except Exception as e:

            pos['rebuilding'] = False

            print(f"❌ Lỗi bơm lại lệnh nhỏ {symbol}: {e}")

            send_telegram(
                f"❌ Lỗi bơm lại lệnh nhỏ {symbol}:\n`{e}`"
            )

            return False


    def reduce_biggest_loser_after_tp(self, tp_profit):

        # Dùng một phần tiền lời vừa TP để cắt bớt lệnh âm nặng,
        # mục tiêu là giảm khối lượng/ký quỹ của lệnh đang âm mà tổng vẫn còn lời.
        loss_budget = tp_profit * LOSS_CUT_PROFIT_USAGE

        if loss_budget <= 0:
            return

        biggest_loser = None
        biggest_loser_price = None
        biggest_loss = 0
        biggest_loss_percent = 0

        for p in self.positions:

            current_price = self.update_coin_data(p['symbol'])

            if not current_price:
                continue

            pnl = self.calculate_virtual_pnl(
                p,
                current_price
            )

            loss_percent = (
                abs(pnl) / p['trade_amount']
            ) * 100 if p['trade_amount'] > 0 else 0

            if (
                pnl < 0
                and loss_percent >= LOSS_CUT_TRIGGER_PERCENT
                and abs(pnl) > biggest_loss
            ):

                biggest_loser = p
                biggest_loser_price = current_price
                biggest_loss = abs(pnl)
                biggest_loss_percent = loss_percent

        if not biggest_loser:
            return

        symbol = biggest_loser['symbol']
        close_side = 'sell' if biggest_loser['side'] == 'buy' else 'buy'

        # Tính cả phí đóng ước tính vào ngân sách cắt lỗ.
        full_exit_fee = (
            biggest_loser['trade_amount'] *
            biggest_loser['leverage']
        ) * FEE_RATE

        full_loss_with_fee = biggest_loss + full_exit_fee

        if full_loss_with_fee <= 0:
            return

        close_ratio = loss_budget / full_loss_with_fee

        close_ratio = min(close_ratio, 0.90)

        if close_ratio < MIN_PARTIAL_CLOSE_AMOUNT_RATIO:

            print(
                f"⌛ {symbol} có lệnh âm {biggest_loss_percent:.1f}% nhưng phần cắt quá nhỏ, bỏ qua"
            )

            return

        close_amount = biggest_loser['amount_coin'] * close_ratio

        close_amount = exchange.amount_to_precision(
            symbol,
            close_amount
        )

        close_amount = float(close_amount)

        if close_amount <= 0:
            return

        try:

            exchange.create_market_order(
                symbol,
                close_side,
                close_amount,
                params={
                    "tdMode": MARGIN_MODE,
                    "reduceOnly": True,
                    "posSide": "long" if biggest_loser['side'] == "buy" else "short"
                }
            )

        except Exception as e:

            print(f"❌ Lỗi cắt bớt lệnh âm: {e}")

            send_telegram(
                f"❌ Lỗi cắt bớt lệnh âm {symbol}:\n`{e}`"
            )

            return

        partial_loss = biggest_loss * close_ratio
        partial_exit_fee = full_exit_fee * close_ratio
        estimated_net_loss = partial_loss + partial_exit_fee

        # Cập nhật position ảo của bot sau khi đóng một phần.
        remain_ratio = 1 - close_ratio

        biggest_loser['amount_coin'] *= remain_ratio
        biggest_loser['trade_amount'] *= remain_ratio
        biggest_loser['entry_fee'] *= remain_ratio

        # Cắt bớt xong thì lệnh này không còn dùng trailing cũ nữa.
        biggest_loser['tp_trailing_active'] = False
        biggest_loser['tp_peak_pnl'] = 0
        biggest_loser['tp_trailing_stop_pnl'] = 0

        self.balance -= estimated_net_loss

        send_telegram(
            f"🧯 *CẮT BỚT LỆNH ÂM SAU TP*\n"
            f"📍 Lệnh âm: `{symbol}`\n"
            f"📉 Đang âm khoảng: `{biggest_loss_percent:.1f}%` ký quỹ\n"
            f"💰 Lời TP vừa đóng: `${tp_profit:.4f}`\n"
            f"✂️ Cắt lỗ một phần ước tính: `${estimated_net_loss:.4f}`\n"
            f"📦 Đã đóng khoảng: `{close_ratio*100:.1f}%` khối lượng lệnh âm\n"
            f"📊 Ký quỹ còn lại: `${biggest_loser['trade_amount']:.4f}`"
        )

        if biggest_loser['trade_amount'] <= REBUILD_TRIGGER_TRADE_AMOUNT:

            self.rebuild_small_loser_position(biggest_loser)


    def close_position(self, pos, price, reason):
        symbol = pos['symbol']

        # Xác định chiều đóng lệnh
        close_side = 'sell' if pos['side'] == 'buy' else 'buy'

        # PNL riêng của vị thế bot đang đóng
        raw_pnl = self.calculate_virtual_pnl(
            pos,
            price
        )

        # Đóng lệnh thật trên OKX
        try:
            exchange.create_market_order(
                symbol,
                close_side,
                pos['amount_coin'],
                params={
                    "tdMode": MARGIN_MODE,
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

            return 0

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

        was_dca_position = pos.get('is_dca_position')
        root_id = pos.get('root_id')

        # Reset position
        self.positions.remove(pos)

        if was_dca_position and root_id:
            self.refresh_root_dca_count(root_id)

        # Nếu đã đóng hết các lệnh DCA/gốc của symbol này thì mở lại slot bình thường
        if not any(p['symbol'] == symbol for p in self.positions):

            self.current_max_positions = MAX_POSITIONS

            self.active_dca_symbol = None

            self.bot_paused = False

        return real_net_profit

    def send_multi_report(self):

        if self.positions:

            symbols = [
                pos['symbol']
                for pos in self.positions
            ]

            status_text = ", ".join(symbols)

        else:

            status_text = "Đang săn tín hiệu đảo chiều..."

        search_status = "⏸ Đang dừng săn lệnh mới" if self.search_paused else "▶️ Đang săn lệnh mới"

        msg = (
            f"📊 *GIÁM SÁT HỆ THỐNG*\n"
            f"📍 {status_text}\n"
            f"{search_status}\n"
            f"🏦 Vốn: `${self.balance:,.2f}$`\n"
            f"📦 Slot lệnh gốc: `{self.count_root_positions()}/{MAX_POSITIONS}`\n"
            f"📌 Tổng lệnh bot đang quản lý: `{len(self.positions)}`"
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
