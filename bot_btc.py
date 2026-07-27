import ccxt
import time
import os
import telebot
from dotenv import load_dotenv
from collections import deque
from functools import wraps
import numpy as np
import traceback
import threading

# Load biến môi trường
load_dotenv()

# --- CẤU HÌNH ---

LEVERAGE = 20 # đòn bẩy
DEFAULT_TRADE_AMOUNT = 10 # vốn vào lệnh
INITIAL_BALANCE = 500 # tổng vốn
CHECK_INTERVAL = 5 # quét giá
WARMUP_PERIOD = 300 # tích dữ liệu giá
VOL_WINDOW_SIZE = 1000 # cửa sổ tính volume theo chiến lược gốc
OHLCV_CACHE_SECONDS = 30 # nến 1h không cần tải lại cho mỗi lần đọc giá
REVERSAL_5M_CACHE_SECONDS = 5 # chỉ tải nến 5m cho coin đang chờ đảo chiều
COOLDOWN_PERIOD = 11 # thời gian khóa coi sau khi trây xong
VOL_DIFF_THRESHOLD = 1.00 # chênh lệch %
CONFIRMATION_TIME = 11 # thời gian tối thiểu xác nhận cú đẩy
SIGNAL_TIMEOUT = 59 * 60 # tối đa 59 phút để đủ surge + volume + Bollinger
PRICE_SURGE_THRESHOLD = 0.002 # mức tăng giá tối thiểu
REVERSAL_ENTRY_CALLBACK = 0.004 # hồi ngược 0.4% từ đỉnh/đáy mới vào lệnh
REVERSAL_WAIT_TIMEOUT = 15 * 60 # tối đa 15 phút chờ đảo chiều thật
STATUS_REPORT_INTERVAL = 1800 # thời gian gửi báo cáo
FEE_RATE = 0.0005 # 0.05% phí
MAX_POSITIONS = 10 # số lệnh tối đa cùng lúc
MAX_DCA = 0 # chiến lược mới: bỏ DCA1/DCA2, đi thẳng từ ROOT sang RESCUE
TP_NET_PROFIT_USD = 2.0
RESCUE_ENABLED = True
RESCUE_PROFIT_USD = 2.0
RESCUE_SOURCE_SLICES = 3
RESCUE_MULTIPLIER = 3.0
RESCUE_AVG_TP_MIN_NET_USD = 1.0
RESCUE_EXTERNAL_HELP_FROM_NUMBER = 3
MARGIN_MODE = "cross" # isolated = Cô lập trên OKX, cross = Chéo

# --- KIỂM TRA MỞ/ĐÓNG LỆNH THẬT QUA TELEGRAM ---
TEST_SYMBOL = "BTC/USDT:USDT"
TEST_TRADE_AMOUNT_USD = 2.0
TEST_WAIT_SECONDS = 2.0

# --- DANH SÁCH ĐEN COIN ---
# Chỉ lưu trong RAM; bot/Railway restart thì danh sách tự reset.

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
MIN_PARTIAL_CLOSE_AMOUNT_RATIO = 0.10 # không đóng từng phần quá nhỏ dưới 5% khối lượng lệnh

# --- QUỸ CẮT LỖ TÍCH LŨY ---
LOSS_BANK_ENABLED = True
LOSS_BANK_MIN_CUT_RATIO = MIN_PARTIAL_CLOSE_AMOUNT_RATIO
LOSS_BANK_MAX_CUT_RATIO = 0.25  # mỗi lần quỹ đủ, cắt tối đa 25% một lệnh âm
LOSS_BANK_NOTIFY_COOLDOWN = 900

# --- ĐỒNG BỘ OKX + GLOBAL TP TOÀN COIN ---
OKX_SYNC_ENABLED = True
OKX_SYNC_INTERVAL = 15
OKX_SYNC_REL_TOLERANCE = 0.03  # cho phép lệch 3% do precision/khớp lệnh
OKX_SYNC_ABS_TOLERANCE = 1e-8
OKX_SYNC_NOTIFY_COOLDOWN = 900
GLOBAL_TP_ENABLED = True
GLOBAL_TP_MIN_NET_USD = 2.0
GLOBAL_TP_MARGIN_RATIO = 0.10  # hoặc tối thiểu 10% tổng ký quỹ coin
GLOBAL_TP_MIN_ROE_PERCENT = 10.0

# --- BƠM LẠI LỆNH ĐÃ BỊ CẮT NHỎ ---
REBUILD_TRIGGER_TRADE_AMOUNT = 2 # khi ký quỹ lệnh còn <= 2$ thì xét bơm lại
REBUILD_ADD_AMOUNT = 6 # số tiền ký quỹ thêm vào lệnh nhỏ để kéo lại giá trung bình
REBUILD_MIN_LOSS_PERCENT = 70 # chỉ bơm lại nếu lệnh nhỏ vẫn đang âm ít nhất 70% ký quỹ

# --- TỰ DỌN LỆNH QUÁ NHỎ ---
TINY_POSITION_VALUE_USDT = 0 # nếu giá trị vị thế còn <= 0.20 USDT thì bot tự đóng/xóa khỏi quản lý
TINY_POSITION_TRADE_AMOUNT = 0 # nếu ký quỹ ảo còn <= 0.05$ thì bot tự đóng/xóa khỏi quản lý

# --- TP CHỜ ĐỦ LỜI ĐỂ CẮT LỆNH ÂM ---
WAIT_TP_UNTIL_CAN_CUT_LOSER = False # TP đạt rồi nhưng nếu có lệnh âm mà chưa đủ tiền cắt tối thiểu thì chờ thêm
TP_WAIT_NOTIFY_COOLDOWN = 1200 # báo Telegram tối đa 1 lần mỗi 5 phút cho mỗi lệnh đang chờ TP

# --- PHỤC HỒI KÝ QUỸ SAU KHI CẮT LỆNH ÂM ---
RESTORE_CUT_POSITION_TO_ORIGINAL = True # sau khi dùng lời TP cắt lỗ, dùng phần lời còn lại bơm lệnh đó về mức ban đầu
RESTORE_TARGET_TRADE_AMOUNT = DEFAULT_TRADE_AMOUNT # lệnh gốc/DCA đều phục hồi tối đa về 5$
MIN_RESTORE_ADD_AMOUNT = 0.10 # nếu phần cần bơm lại dưới 0.10$ thì bỏ qua để tránh order quá nhỏ

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

def synchronized_trading(method):
    """Không cho Telegram và vòng quản lý chính gửi order chồng lên nhau."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.trading_lock:
            return method(self, *args, **kwargs)
    return wrapper

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
        self.trading_lock = threading.RLock()
        # Cấu hình runtime có thể đổi qua Telegram.
        self.default_trade_amount = float(DEFAULT_TRADE_AMOUNT)
        self.max_positions = int(MAX_POSITIONS)
        self.current_max_positions = self.max_positions

        # Danh sách đen chỉ lưu trong RAM.
        # Khi bot hoặc Railway restart, danh sách sẽ tự trở về trống.
        self.blacklist = set()

        # Quỹ cắt lỗ chỉ tích lũy từ phần lợi nhuận TP đã đóng.
        self.loss_bank = 0.0
        self.last_loss_bank_notify = 0.0
        # Trạng thái Rescue theo từng coin + chiều.
        self.rescue_chains = {}
        self.active_rescue_closing = False
        # Chi tiết lần đóng gần nhất dùng để đối chiếu PnL ảo với PnL theo
        # giá trung bình gộp thật của OKX.
        self.last_close_accounting = {}

        # Khối lượng thật còn trên OKX nhưng bot đã chủ động quên khỏi một lệnh ảo
        # sau TP khớp một phần hoặc lệnh XR. Khối lượng này không chiếm slot DCA.
        # Khi lệnh ảo cuối cùng của coin/chiều đóng, bot sẽ đóng luôn toàn bộ phần này.
        self.ignored_residuals = {}

        # Ảnh chụp vị thế gộp thật trên OKX. Không ghi đè giá lệnh riêng của bot.
        self.okx_position_snapshot = {}
        self.last_okx_sync_time = 0.0
        self.last_sync_notify = {}

        self.active_dca_symbol = None
        self.bot_paused = False
        self.test_order_running = False
        self.recovery_session = None
        self.startup_position_check_done = False
        
        self.coins = {}
        for symbol in SYMBOLS:
            self.coins[symbol] = {
                'vol_trades': deque(),
                'last_trade_id': None,
                'last_trade_timestamp': None,
                'price_history': deque(maxlen=310),
                'last_ohlcv_fetch_time': 0.0,
                'last_market_price': 0.0,
                'pending_side': None,
                'trigger_price': 0,
                'trigger_time': 0,
                'trigger_vol_diff': 0, 
                'waiting_reversal': False,
                'reversal_extreme_price': 0,
                'reversal_start_time': 0,
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
        self.reversal_5m_cache = {}

        # D/T Telegram: D = dừng săn lệnh mới, T = tiếp tục săn lệnh mới.
        # Các lệnh đang mở vẫn được quản lý TP/DCA/cắt lỗ bình thường.
        self.search_paused = False

        # Chế độ XEM: bot không đặt, đóng, DCA, TP, cắt lỗ hay bơm lệnh.
        # Bot chỉ kiểm tra những coin đang được quản lý và nhận biết khi
        # người dùng đã đóng HẾT vị thế đó trực tiếp trên OKX.
        self.view_mode = False
        self.last_view_sync_time = 0.0
        self.last_view_partial_notify = {}

        global ACTIVE_TRADING_BOT
        ACTIVE_TRADING_BOT = self

        self.setup_telegram_commands()

    def count_root_positions(self):
        # Chỉ đếm lệnh gốc để giới hạn slot săn lệnh mới.
        # Lệnh DCA riêng không chiếm slot, nên full 5/5 vẫn DCA được.
        return sum(
            1 for pos in self.positions
            if not pos.get('is_dca_position') and not pos.get('is_rescue_position')
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

                if text in ('CĐ XEM', 'CD XEM'):
                    active_bot.enable_view_mode()

                elif text in ('CĐ CHẠY', 'CĐ CHAY', 'CD CHẠY', 'CD CHAY'):
                    active_bot.disable_view_mode()

                elif text in ('KHOI PHUC', 'KHÔI PHỤC'):
                    active_bot.start_okx_recovery()

                elif text in ('HUY KHOI PHUC', 'HỦY KHÔI PHỤC'):
                    active_bot.cancel_okx_recovery()

                elif active_bot.recovery_session is not None:
                    active_bot.handle_okx_recovery_answer(text)

                elif text == 'TEST':
                    threading.Thread(
                        target=active_bot.run_real_tp_test,
                        daemon=True
                    ).start()

                elif text == 'D':
                    active_bot.search_paused = True
                    send_telegram(
                        "DỖI KO TÌM LỆNH NỮA"
                    )

                elif text == 'T':
                    active_bot.search_paused = False
                    send_telegram(
                        "NỂ LẮM MỚI LÀM TIẾP ĐẤY"
                    )

                elif text.startswith('Đ ') or text.startswith('DONG '):
                    coin_text = text.split(maxsplit=1)[1] if ' ' in text else ''
                    active_bot.close_symbol_by_telegram(coin_text)

                elif text.startswith('XR '):
                    active_bot.forget_virtual_position_by_telegram(text[3:].strip())

                elif text.startswith('X '):
                    coin_text = text.split(maxsplit=1)[1] if ' ' in text else ''
                    active_bot.add_coin_to_blacklist(coin_text)

                elif text.startswith('GX ') or text.startswith('GOX '):
                    coin_text = text.split(maxsplit=1)[1] if ' ' in text else ''
                    active_bot.remove_coin_from_blacklist(coin_text)

                elif text in ('DSX', 'BLACKLIST'):
                    active_bot.send_blacklist_info()

                elif text.startswith('VON '):
                    try:
                        value = float(text.split(maxsplit=1)[1].replace(',', '.'))
                        if value <= 0:
                            raise ValueError
                        old_value = active_bot.default_trade_amount
                        active_bot.default_trade_amount = value
                        send_telegram(
                            f"💵 Đã đổi vốn mỗi lệnh: `${old_value:g}` → `${value:g}`\n"
                            f"📌 Chỉ áp dụng cho lệnh gốc/DCA mở từ bây giờ."
                        )
                    except Exception:
                        send_telegram("⚠️ Cú pháp đúng: `VON 10`")

                elif text.startswith('LENH '):
                    try:
                        value = int(float(text.split(maxsplit=1)[1].replace(',', '.')))
                        if value <= 0:
                            raise ValueError
                        old_value = active_bot.max_positions
                        active_bot.max_positions = value
                        active_bot.current_max_positions = value
                        send_telegram(
                            f"📦 Đã đổi số slot lệnh gốc: `{old_value}` → `{value}`\n"
                            f"📊 Hiện đang có: `{active_bot.count_root_positions()}/{value}`"
                        )
                    except Exception:
                        send_telegram("⚠️ Cú pháp đúng: `LENH 10`")

                elif text in ('QUY', 'BANK'):
                    loser, _, loss_usd, loss_percent = active_bot.find_biggest_loser_for_cut()
                    loser_text = (
                        f"`{loser['symbol']}` âm khoảng `{loss_percent:.1f}%` / `${loss_usd:.4f}`"
                        if loser else "Không có lệnh riêng nào vượt ngưỡng cắt"
                    )
                    send_telegram(
                        f"🏦 *QUỸ CẮT LỖ*\n"
                        f"💰 Số dư quỹ: `${active_bot.loss_bank:.4f}`\n"
                        f"🎯 Ưu tiên hiện tại: {loser_text}\n"
                        "✅ Không dùng cơ chế khóa coin do lệch dữ liệu OKX."
                    )

                elif text:
                    # Nhắn tên coin, ví dụ TURBO / WIF / TURBOUSDT,
                    # bot sẽ gửi chi tiết lệnh gốc + các lệnh DCA đang quản lý.
                    active_bot.send_symbol_position_info(text)

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

    def normalize_symbol_query(self, text):
        query = (text or '').strip().upper()
        query = query.replace('/USDT:USDT', '')
        query = query.replace('/USDT', '')
        query = query.replace('USDT', '')
        query = query.replace('/', '')
        query = query.replace(':', '')
        return query

    def symbol_base_name(self, symbol):
        return symbol.split('/')[0].upper()

    def _recovery_role_name(self, index):
        if index == 0:
            return 'LỆNH GỐC'
        return f'DCA{index}'

    def _validate_recovery_group_prices(self, group):
        """Kiểm tra các giá riêng có tái tạo gần đúng giá trung bình OKX không."""
        prices = group.get('entry_prices') or []
        if len(prices) != group.get('slice_count'):
            return False, None
        try:
            market = exchange.market(group['symbol'])
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            leverage = self._safe_float(group.get('leverage'), LEVERAGE)
            raw_weights = [
                (self.default_trade_amount * leverage) / (price * contract_size)
                for price in prices
            ]
            weight_total = sum(raw_weights)
            if weight_total <= 0:
                return False, None
            amounts = [
                group['contracts'] * weight / weight_total
                for weight in raw_weights
            ]
            recovered_avg = (
                sum(price * amount for price, amount in zip(prices, amounts))
                / group['contracts']
            )
            okx_avg = self._safe_float(group.get('okx_entry_price'))
            if okx_avg <= 0:
                return False, recovered_avg
            avg_error = abs(recovered_avg - okx_avg) / okx_avg
            return avg_error <= OKX_SYNC_REL_TOLERANCE, recovered_avg
        except Exception:
            return False, None

    def _auto_complete_recovery_prices(self, group, root_price):
        """
        Suy ngược giá DCA từ giá gộp OKX với giả định mỗi lát dùng cùng vốn.
        - 2 lát: nghiệm DCA1 là duy nhất.
        - 3 lát: lấy DCA1 tại đúng ngưỡng chiến lược rồi giải DCA2.
        """
        slice_count = int(group.get('slice_count', 1))
        okx_avg = self._safe_float(group.get('okx_entry_price'))
        leverage = self._safe_float(group.get('leverage'), LEVERAGE)
        if root_price <= 0 or okx_avg <= 0 or leverage <= 0:
            raise ValueError("Thiếu giá gốc, giá trung bình OKX hoặc leverage.")
        if slice_count == 1:
            return [root_price]

        inverse_target = slice_count / okx_avg
        if slice_count == 2:
            inverse_dca1 = inverse_target - (1.0 / root_price)
            if inverse_dca1 <= 0:
                raise ValueError("Không giải được giá DCA1 dương từ giá gốc này.")
            prices = [root_price, 1.0 / inverse_dca1]
            if (
                group['side'] == 'buy' and prices[1] >= prices[0]
            ) or (
                group['side'] == 'sell' and prices[1] <= prices[0]
            ):
                raise ValueError("Giá DCA1 suy ra không đúng chiều DCA của vị thế.")
            return prices

        if slice_count == 3:
            dca_step = 1.0 / leverage
            if group['side'] == 'buy':
                dca1_price = root_price * (1.0 - dca_step)
            else:
                dca1_price = root_price * (1.0 + dca_step)
            inverse_dca2 = (
                inverse_target
                - (1.0 / root_price)
                - (1.0 / dca1_price)
            )
            if dca1_price <= 0 or inverse_dca2 <= 0:
                raise ValueError("Không giải được giá DCA2 dương từ giá gốc này.")
            prices = [root_price, dca1_price, 1.0 / inverse_dca2]
            if (
                group['side'] == 'buy' and not (prices[2] < prices[1] < prices[0])
            ) or (
                group['side'] == 'sell' and not (prices[2] > prices[1] > prices[0])
            ):
                raise ValueError("Giá DCA2 suy ra không đúng thứ tự giá của chuỗi DCA.")
            return prices

        raise ValueError("Chỉ hỗ trợ tự tính tối đa LỆNH GỐC + DCA1 + DCA2.")

    def guard_unmanaged_okx_positions_on_startup(self):
        """Không săn lệnh mới nếu OKX đã có vị thế nhưng RAM bot đang trống."""
        if self.startup_position_check_done:
            return True
        try:
            raw_positions = exchange.fetch_positions()
        except Exception as e:
            self.view_mode = True
            self.search_paused = True
            send_telegram(
                f"⚠️ Không kiểm tra được vị thế OKX lúc khởi động:\n`{e}`\n"
                "👁 Bot tự khóa ở CHẾ ĐỘ XEM. Hãy kiểm tra kết nối rồi nhắn `KHOI PHUC`."
            )
            return False

        self.startup_position_check_done = True
        active = [
            item for item in (raw_positions or [])
            if abs(self._safe_float(item.get('contracts'))) > OKX_SYNC_ABS_TOLERANCE
        ]
        if active and not self.positions:
            self.view_mode = True
            self.search_paused = True
            send_telegram(
                "🛡 *PHÁT HIỆN VỊ THẾ OKX KHI BOT VỪA KHỞI ĐỘNG*\n"
                f"📦 Có `{len(active)}` vị thế coin/chiều nhưng bộ nhớ lệnh riêng đang trống.\n"
                "👁 Bot đã tự khóa ở CHẾ ĐỘ XEM và không săn lệnh mới.\n"
                "➡️ Nhắn `KHOI PHUC` để dựng lại một ROOT theo giá trung bình OKX."
            )
        return True

    def _send_next_recovery_question(self):
        session = self.recovery_session
        if not session:
            return
        groups = session['groups']
        group_index = session['group_index']
        if group_index >= len(groups):
            self._finish_okx_recovery()
            return

        group = groups[group_index]
        price_index = len(group['entry_prices'])
        if price_index >= group['slice_count']:
            session['group_index'] += 1
            self._send_next_recovery_question()
            return

        send_telegram(
            "🧩 *KHÔI PHỤC GIÁ VÀO RIÊNG*\n"
            f"📍 `{group['symbol']}` - `{group['side'].upper()}`\n"
            f"🏦 Giá trung bình OKX: `{group['okx_entry_price']}`\n"
            f"📦 Tổng contract OKX: `{group['contracts']}`\n"
            f"💵 Ký quỹ ước tính: `${group['estimated_margin']:.4f}`\n"
            f"🧱 Bot nhận diện: `{group['slice_count']}` lệnh riêng\n\n"
            "➡️ Chỉ cần gửi *GIÁ VÀO CỦA LỆNH GỐC* dưới dạng số.\n"
            "Bot sẽ tự tính DCA1/DCA2 để giá gộp khớp OKX.\n"
            "Ví dụ: `0.012345`\n"
            "Hủy toàn bộ bằng: `HUY KHOI PHUC`"
        )

    @synchronized_trading
    def start_okx_recovery(self):
        if self.recovery_session is not None:
            send_telegram(
                "⏳ Một phiên `KHOI PHUC` đang chờ nhập giá. "
                "Tiếp tục trả lời hoặc nhắn `HUY KHOI PHUC`."
            )
            return False
        if self.positions:
            send_telegram(
                "⛔ Bot đang có dữ liệu lệnh riêng trong bộ nhớ nên không được khôi phục chồng lên.\n"
                "📌 Chỉ dùng `KHOI PHUC` ngay sau khi bot mới khởi động và bộ nhớ đang trống."
            )
            return False

        # Khôi phục luôn bắt đầu ở chế độ chỉ xem. Không order nào được phép chạy
        # cho đến khi người dùng kiểm tra xong và chủ động nhắn CĐ CHẠY.
        self.view_mode = True
        self.search_paused = True
        try:
            raw_positions = exchange.fetch_positions()
        except Exception as e:
            send_telegram(
                f"❌ Không đọc được vị thế OKX để khôi phục:\n`{e}`\n"
                "✅ Bot vẫn ở CHẾ ĐỘ XEM; chưa tạo dữ liệu lệnh nào."
            )
            return False

        groups = []
        unsupported = []
        for item in raw_positions or []:
            symbol = item.get('symbol')
            side = self._okx_side_from_position(item)
            contracts = abs(self._safe_float(item.get('contracts')))
            if not symbol or side is None or contracts <= OKX_SYNC_ABS_TOLERANCE:
                continue
            market = markets.get(symbol) or {}
            if not market.get('swap') or market.get('quote') != 'USDT':
                continue

            info = item.get('info') or {}
            leverage = self._safe_float(
                item.get('leverage'),
                self._safe_float(info.get('lever'), LEVERAGE)
            )
            if leverage <= 0:
                leverage = LEVERAGE
            entry_price = self._safe_float(
                item.get('entryPrice'),
                self._safe_float(info.get('avgPx'))
            )
            mark_price = self._safe_float(
                item.get('markPrice'),
                self._safe_float(info.get('markPx'), entry_price)
            )
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            # Đếm lát theo notional tại giá vào để biến động mark price không làm
            # một lệnh 10$ bị nhận nhầm thành 2 lệnh.
            entry_notional = contracts * entry_price * contract_size
            estimated_margin = entry_notional / leverage if leverage > 0 else 0.0
            if estimated_margin <= 0:
                notional = abs(self._safe_float(
                    item.get('notional'),
                    self._safe_float(info.get('notionalUsd'))
                ))
                estimated_margin = notional / leverage if leverage > 0 else 0.0
            if estimated_margin <= 0:
                estimated_margin = abs(self._safe_float(
                    item.get('initialMargin'), self._safe_float(info.get('margin'))
                ))

            # Chiến lược mới không dựng lại DCA/RESCUE ảo. Toàn bộ vị thế thật
            # được tiếp quản như một ROOT duy nhất tại giá trung bình OKX.
            slice_count = 1

            groups.append({
                'symbol': symbol,
                'side': side,
                'contracts': contracts,
                'leverage': leverage,
                'okx_entry_price': entry_price,
                'mark_price': mark_price,
                'estimated_margin': estimated_margin,
                'slice_count': slice_count,
                'entry_prices': [],
            })

        if not groups:
            send_telegram(
                "ℹ️ OKX không có vị thế swap USDT nào để khôi phục.\n"
                "👁 Bot vẫn đang ở CHẾ ĐỘ XEM."
            )
            return False

        # Vị thế chỉ có một lệnh có thể dùng trực tiếp giá trung bình OKX.
        for group in groups:
            if group['slice_count'] == 1:
                if group['okx_entry_price'] <= 0:
                    send_telegram(
                        f"⛔ `{group['symbol']}` không có giá trung bình OKX hợp lệ."
                    )
                    return False
                group['entry_prices'].append(group['okx_entry_price'])

        self.recovery_session = {
            'groups': groups,
            'group_index': 0,
            'started_at': time.time(),
        }
        send_telegram(
            "🔎 *BẮT ĐẦU KHÔI PHỤC TỪ OKX*\n"
            f"📦 Phát hiện `{len(groups)}` vị thế coin/chiều.\n"
            "🧱 Mỗi vị thế sẽ được dựng thành một ROOT duy nhất tại giá trung bình OKX.\n"
            "👁 Bot đã khóa ở CHẾ ĐỘ XEM và chưa gửi bất kỳ order nào."
        )
        self._send_next_recovery_question()
        return True

    @synchronized_trading
    def handle_okx_recovery_answer(self, text):
        session = self.recovery_session
        if session is None:
            return False
        try:
            price = float(str(text).strip().replace(',', '.'))
            if price <= 0:
                raise ValueError
        except Exception:
            send_telegram(
                "⚠️ Giá không hợp lệ. Chỉ gửi một số dương, ví dụ: `0.012345`.\n"
                "Hủy bằng: `HUY KHOI PHUC`"
            )
            return False

        group = session['groups'][session['group_index']]
        try:
            group['entry_prices'] = self._auto_complete_recovery_prices(
                group, price
            )
        except Exception as e:
            group['entry_prices'].clear()
            send_telegram(
                f"⚠️ Không thể tự tính giá DCA từ giá gốc `{price}`:\n`{e}`\n"
                "🔄 Hãy kiểm tra và nhập lại giá LỆNH GỐC."
            )
            self._send_next_recovery_question()
            return False

        send_telegram(
            f"✅ Đã ghi giá LỆNH GỐC của `{group['symbol']}`: `{price}`"
        )
        if group['slice_count'] > 1:
            calculated = "\n".join(
                f"• {self._recovery_role_name(index)}: `{entry_price}`"
                for index, entry_price in enumerate(group['entry_prices'][1:], start=1)
            )
            send_telegram(
                "🧮 *GIÁ DCA BOT TỰ TÍNH*\n"
                f"📍 `{group['symbol']}` - `{group['side'].upper()}`\n"
                f"{calculated}\n"
                f"🏦 Mục tiêu giá gộp OKX: `{group['okx_entry_price']}`"
            )
        if len(group['entry_prices']) >= group['slice_count']:
            valid_prices, recovered_avg = self._validate_recovery_group_prices(group)
            if not valid_prices:
                okx_avg = self._safe_float(group.get('okx_entry_price'))
                error_percent = (
                    abs(self._safe_float(recovered_avg) - okx_avg) / okx_avg * 100
                    if recovered_avg is not None and okx_avg > 0 else None
                )
                group['entry_prices'].clear()
                send_telegram(
                    "⚠️ *GIÁ KHÔI PHỤC CHƯA KHỚP OKX*\n"
                    f"📍 `{group['symbol']}` - `{group['side'].upper()}`\n"
                    f"🏦 Giá trung bình OKX: `{okx_avg}`\n"
                    f"🧮 Giá gộp từ các giá vừa nhập: "
                    f"`{recovered_avg if recovered_avg is not None else 'không tính được'}`\n"
                    f"📉 Độ lệch: "
                    f"`{f'{error_percent:.2f}%' if error_percent is not None else 'không xác định'}`\n"
                    f"📏 Mức cho phép: `{OKX_SYNC_REL_TOLERANCE*100:.0f}%`\n"
                    "🔄 Bot sẽ hỏi lại riêng coin này từ LỆNH GỐC; "
                    "không cần nhắn `KHOI PHUC` lại."
                )
                self._send_next_recovery_question()
                return False
            session['group_index'] += 1
        self._send_next_recovery_question()
        return True

    @synchronized_trading
    def cancel_okx_recovery(self):
        if self.recovery_session is None:
            send_telegram("ℹ️ Không có phiên khôi phục nào đang chạy.")
            return False
        self.recovery_session = None
        send_telegram(
            "🛑 Đã hủy khôi phục. Không có dữ liệu lệnh nào được tạo.\n"
            "👁 Bot vẫn ở CHẾ ĐỘ XEM."
        )
        return True

    def _finish_okx_recovery(self):
        session = self.recovery_session
        if not session:
            return False

        recovered = []
        try:
            for group in session['groups']:
                prices = group['entry_prices']
                if len(prices) != group['slice_count'] or any(p <= 0 for p in prices):
                    raise RuntimeError(f"Thiếu giá vào cho {group['symbol']}")

                market = exchange.market(group['symbol'])
                contract_size = self._safe_float(market.get('contractSize'), 1.0)
                leverage = group['leverage']
                # Mỗi lát bắt đầu với cùng vốn chuẩn. Scale contract để tổng cuối
                # khớp tuyệt đối với vị thế thật OKX.
                raw_weights = [
                    (self.default_trade_amount * leverage) / (price * contract_size)
                    for price in prices
                ]
                weight_total = sum(raw_weights)
                if weight_total <= 0:
                    raise RuntimeError(f"Không phân bổ được contract cho {group['symbol']}")
                amounts = [
                    group['contracts'] * weight / weight_total
                    for weight in raw_weights
                ]
                recovered_avg = (
                    sum(price * amount for price, amount in zip(prices, amounts))
                    / group['contracts']
                )
                okx_avg = group['okx_entry_price']
                if okx_avg > 0:
                    avg_error = abs(recovered_avg - okx_avg) / okx_avg
                    if avg_error > OKX_SYNC_REL_TOLERANCE:
                        raise RuntimeError(
                            f"Giá đã nhập cho {group['symbol']} tạo giá gộp "
                            f"{recovered_avg}, lệch quá {OKX_SYNC_REL_TOLERANCE*100:.0f}% "
                            f"so với OKX {okx_avg}. Hãy KHOI PHUC lại và kiểm tra giá."
                        )

                root_id = self.make_position_id()
                root_price = prices[0]
                for index, (price, amount) in enumerate(zip(prices, amounts)):
                    amount = self._safe_float(amount)
                    trade_amount = self.margin_from_fill(
                        group['symbol'], amount, price, leverage,
                        self.default_trade_amount
                    )
                    entry_fee = trade_amount * leverage * FEE_RATE
                    position_id = root_id if index == 0 else self.make_position_id()
                    pos = {
                        'position_id': position_id,
                        'root_id': root_id,
                        'symbol': group['symbol'],
                        'side': group['side'],
                        'entry_price': price,
                        'first_entry_price': root_price if index == 0 else price,
                        'amount_coin': amount,
                        'trade_amount': trade_amount,
                        'original_trade_amount': trade_amount,
                        'entry_fee': entry_fee,
                        'leverage': leverage,
                        'dca_count': group['slice_count'] - 1 if index == 0 else 0,
                        'waiting_dca': False,
                        'is_dca_position': index > 0,
                        'is_rescue_position': False,
                        'chain_order': index,
                        'tp_trailing_active': False,
                        'tp_peak_pnl': 0,
                        'tp_trailing_stop_pnl': 0,
                        'rebuild_count': 0,
                        'fills': [],
                        'recovered_from_okx': True,
                        'recovery_debt': 0.0,
                    }
                    if index > 0:
                        pos['dca_number'] = index
                        pos['parent_entry_price'] = root_price
                    recovered.append(pos)

            for group in session['groups']:
                recovered_amount = sum(
                    self._safe_float(p.get('amount_coin'))
                    for p in recovered
                    if p['symbol'] == group['symbol'] and p['side'] == group['side']
                )
                tolerance = max(
                    OKX_SYNC_ABS_TOLERANCE,
                    group['contracts'] * OKX_SYNC_REL_TOLERANCE
                )
                if abs(recovered_amount - group['contracts']) > tolerance:
                    raise RuntimeError(
                        f"Tổng contract khôi phục lệch OKX cho {group['symbol']}: "
                        f"{recovered_amount} != {group['contracts']}"
                    )

            self.positions.extend(recovered)
            self.recovery_session = None
            # CĐ CHẠY sau khi người dùng kiểm tra sẽ tiếp tục cả quản lý lệnh
            # lẫn săn tín hiệu mới, trừ khi họ chủ động nhắn D sau đó.
            self.search_paused = False
            self.refresh_bot_paused_state()
            lines = [
                "✅ *KHÔI PHỤC HOÀN TẤT TRONG CHẾ ĐỘ XEM*",
                f"📚 Đã dựng lại `{len(recovered)}` ROOT theo giá trung bình OKX.",
            ]
            for group in session['groups']:
                labels = ", ".join(
                    f"{self._recovery_role_name(i)}=`{price}`"
                    for i, price in enumerate(group['entry_prices'])
                )
                lines.append(
                    f"📍 `{group['symbol']}` `{group['side'].upper()}`: {labels}"
                )
            lines.extend([
                "👁 Bot vẫn chưa được phép đặt/đóng order.",
                "📌 Hãy nhắn tên từng coin để kiểm tra.",
                "▶️ Chỉ khi mọi thứ đúng mới nhắn: `CĐ CHẠY`",
            ])
            send_telegram("\n".join(lines)[:3900])
            return True
        except Exception as e:
            self.recovery_session = None
            send_telegram(
                f"❌ Khôi phục thất bại:\n`{e}`\n"
                "✅ Chưa thêm dữ liệu lệnh nào; bot vẫn ở CHẾ ĐỘ XEM."
            )
            return False

    def is_symbol_blacklisted(self, symbol):
        return self.symbol_base_name(symbol) in self.blacklist

    def add_coin_to_blacklist(self, text):
        coin = self.normalize_symbol_query(text)
        if not coin:
            send_telegram("⚠️ Cú pháp đúng: `X BTC`")
            return False

        matching_markets = [s for s in SYMBOLS if self.symbol_base_name(s) == coin]
        if not matching_markets:
            send_telegram(f"⚠️ Không tìm thấy coin futures USDT `{coin}` trên danh sách market của bot.")
            return False

        open_positions = [p for p in self.positions if self.symbol_base_name(p['symbol']) == coin]
        if open_positions:
            send_telegram(
                f"⛔ *KHÔNG THỂ THÊM VÀO DANH SÁCH ĐEN*\n"
                f"📍 `{coin}` đang có `{len(open_positions)}` lệnh bot quản lý.\n"
                f"📌 Hãy đóng coin trước bằng: `Đ {coin}`"
            )
            return False

        if coin in self.blacklist:
            send_telegram(f"ℹ️ `{coin}` đã có trong danh sách đen.")
            return True

        self.blacklist.add(coin)
        for symbol in matching_markets:
            coin_state = self.coins.get(symbol)
            if coin_state:
                self.reset_coin_signal(coin_state)

        send_telegram(
            f"🚫 *ĐÃ THÊM VÀO DANH SÁCH ĐEN TẠM THỜI*\n"
            f"📍 `{coin}`\n"
            f"✅ Bot sẽ bỏ qua coin này dù đạt đủ điều kiện vào lệnh.\n"
            f"🧠 Danh sách chỉ lưu trong RAM và sẽ tự xóa khi bot restart."
        )
        return True

    def remove_coin_from_blacklist(self, text):
        coin = self.normalize_symbol_query(text)
        if not coin:
            send_telegram("⚠️ Cú pháp đúng: `GX BTC`")
            return False
        if coin not in self.blacklist:
            send_telegram(f"ℹ️ `{coin}` không có trong danh sách đen.")
            return True
        self.blacklist.remove(coin)
        send_telegram(
            f"✅ Đã gỡ `{coin}` khỏi danh sách đen tạm thời. "
            f"Bot có thể giao dịch lại coin này."
        )
        return True

    def send_blacklist_info(self):
        if not self.blacklist:
            send_telegram("📋 *DANH SÁCH ĐEN TẠM THỜI*\nDanh sách hiện đang trống.\n🧠 Bot restart thì danh sách tự xóa.")
            return
        coins = ', '.join(f'`{coin}`' for coin in sorted(self.blacklist))
        send_telegram(
            f"📋 *DANH SÁCH ĐEN TẠM THỜI ({len(self.blacklist)} COIN)*\n"
            f"{coins}\n\n"
            f"➕ Thêm: `X BTC`\n"
            f"➖ Gỡ: `GX BTC`\n"
            f"🧠 Bot restart thì danh sách tự xóa."
        )

    def _remember_ignored_residual(self, pos, amount_coin):
        """Ghi nhớ khối lượng thật còn trên OKX nhưng không còn chiếm một slot lệnh ảo."""
        amount_coin = abs(self._safe_float(amount_coin))
        if amount_coin <= 0:
            return
        key = self.get_position_key(pos['symbol'], pos['side'])
        self.ignored_residuals[key] = self._safe_float(
            self.ignored_residuals.get(key)
        ) + amount_coin

    def _clear_ignored_residual(self, key):
        self.ignored_residuals.pop(key, None)

    @synchronized_trading
    def forget_virtual_position_by_telegram(self, text):
        """XR CRDO DCA1: quên riêng lệnh ảo, không gửi lệnh đóng lên OKX."""
        parts = (text or '').strip().upper().split()
        if len(parts) < 2:
            send_telegram(
                "⚠️ Cú pháp đúng:\n"
                "`XR CRDO DCA1`\n"
                "`XR CRDO DCA2`\n"
                "`XR CRDO RESCUE1`"
            )
            return False

        coin = self.normalize_symbol_query(parts[0])
        role = ''.join(parts[1:]).replace(' ', '')
        target = None

        for pos in self.positions:
            if self.symbol_base_name(pos['symbol']) != coin:
                continue
            if role in ('GOC', 'ROOT', 'LENHGOC'):
                matched = not pos.get('is_dca_position') and not pos.get('is_rescue_position')
            elif role.startswith('DCA'):
                try:
                    number = int(role.replace('DCA', ''))
                except Exception:
                    number = -1
                matched = pos.get('is_dca_position') and int(pos.get('dca_number', -1)) == number
            elif role.startswith('RESCUE'):
                try:
                    number = int(role.replace('RESCUE', ''))
                except Exception:
                    number = -1
                matched = pos.get('is_rescue_position') and int(pos.get('rescue_number', -1)) == number
            else:
                matched = False

            if matched:
                target = pos
                break

        if target is None:
            available = []
            for p in self.positions:
                if self.symbol_base_name(p['symbol']) != coin:
                    continue
                if p.get('is_rescue_position'):
                    available.append(f"RESCUE{p.get('rescue_number', '?')}")
                elif p.get('is_dca_position'):
                    available.append(f"DCA{p.get('dca_number', '?')}")
                else:
                    available.append('ROOT')
            available_text = ', '.join(available) if available else 'không có lệnh nào'
            send_telegram(
                f"🔎 Không tìm thấy `{coin} {role}` trong bộ nhớ bot.\n"
                f"📋 Các vai trò hiện có: `{available_text}`"
            )
            return False

        same_key_positions = [
            p for p in self.positions
            if p['symbol'] == target['symbol'] and p['side'] == target['side']
        ]
        if len(same_key_positions) <= 1:
            send_telegram(
                f"⛔ Không thể XR `{coin} {role}` vì đây là lệnh ảo cuối cùng của coin/chiều.\n"
                f"📌 Hãy dùng `Đ {coin}` để đóng toàn bộ vị thế thật."
            )
            return False

        amount = self._safe_float(target.get('amount_coin'))
        key = self.get_position_key(target['symbol'], target['side'])
        self._remember_ignored_residual(target, amount)
        self.remove_position_from_memory(target)

        # Xóa một mắt xích giữa chuỗi thì dịch các lệnh phía sau xuống ngay:
        # DCA2 -> DCA1, Rescue1 -> DCA2...
        if any(
            p['symbol'] == target['symbol'] and p['side'] == target['side']
            for p in self.positions
        ):
            self._promote_rescue_chain(key)

        send_telegram(
            f"🧹 *ĐÃ QUÊN LỆNH ẢO KHỎI BOT*\n"
            f"📍 `{coin} {role}`\n"
            f"📦 Khối lượng thật còn trên OKX: `{amount}`\n"
            f"✅ Lệnh này không còn chiếm slot DCA.\n"
            f"🔒 Bot sẽ cộng phần dư này khi đồng bộ và đóng sạch khi lệnh cuối cùng của coin đóng."
        )
        return True


    def _find_market_symbol(self, preferred_symbol):
        if preferred_symbol in markets:
            return preferred_symbol
        preferred_base = str(preferred_symbol).split('/')[0].upper()
        for symbol, market in markets.items():
            if (
                market.get('swap')
                and market.get('quote') == 'USDT'
                and str(market.get('base') or '').upper() == preferred_base
                and market.get('active') is True
            ):
                return symbol
        return None

    @synchronized_trading
    def run_real_tp_test(self):
        """Mở BTC thật rồi gọi đúng close_position như thể lệnh vừa đạt TP."""
        if self.test_order_running:
            send_telegram("⏳ Một bài `TEST` khác đang chạy.")
            return False
        if self.view_mode:
            send_telegram("👁 Hãy nhắn `CĐ CHẠY` trước khi chạy `TEST`.")
            return False

        self.test_order_running = True
        test_pos = None
        try:
            symbol = self._find_market_symbol(TEST_SYMBOL)
            if not symbol:
                raise RuntimeError("Không tìm thấy BTC USDT perpetual swap.")

            if any(p.get('symbol') == symbol for p in self.positions):
                send_telegram(
                    "⛔ *KHÔNG CHẠY TEST*\n"
                    f"Bot đang quản lý vị thế `{symbol}` nên lệnh TEST có thể bị gộp."
                )
                return False

            before = self.fetch_okx_position_snapshot_for_symbols([symbol])
            if any(
                key[0] == symbol and abs(self._safe_float(item.get('contracts'))) > OKX_SYNC_ABS_TOLERANCE
                for key, item in (before or {}).items()
            ):
                send_telegram(
                    "⛔ *KHÔNG CHẠY TEST*\n"
                    f"OKX đang có vị thế thật `{symbol}`."
                )
                return False

            ticker = exchange.fetch_ticker(symbol)
            open_price = self._safe_float(ticker.get('last'))
            if open_price <= 0:
                raise RuntimeError("Không lấy được giá BTC.")

            trade_amount = min(
                self._safe_float(TEST_TRADE_AMOUNT_USD),
                self._safe_float(self.balance)
            )
            if trade_amount <= 0:
                raise RuntimeError("Số dư ảo không đủ cho TEST.")

            send_telegram(
                "🧪 *BẮT ĐẦU TEST TP THẬT*\n"
                f"📍 `{symbol}`\n"
                f"💵 Ký quỹ test: `${trade_amount:.2f}`\n"
                "⚠️ Đây là lệnh thật, có phí và trượt giá."
            )

            order, actual_leverage, requested_amount, estimated_fee = (
                self.create_entry_order_with_leverage_fallback(
                    symbol=symbol,
                    side='buy',
                    price=open_price,
                    trade_amount=trade_amount,
                    preferred_leverage=LEVERAGE
                )
            )
            fill = self.resolve_order_fill(order, symbol, open_price, requested_amount)
            fill_price = self._safe_float(fill.get('price'), open_price)
            amount_coin = self._safe_float(fill.get('amount'), requested_amount)
            trade_amount = self.margin_from_fill(
                symbol, amount_coin, fill_price, actual_leverage, trade_amount
            )
            entry_fee = self._safe_float(fill.get('fee'))
            if entry_fee <= 0:
                entry_fee = self._safe_float(estimated_fee)
            if amount_coin <= 0:
                raise RuntimeError("Không đọc được khối lượng khớp của lệnh TEST.")

            self.balance -= entry_fee
            position_id = self.make_position_id()
            test_pos = {
                'position_id': position_id,
                'root_id': position_id,
                'symbol': symbol,
                'side': 'buy',
                'entry_price': fill_price,
                'first_entry_price': fill_price,
                'amount_coin': amount_coin,
                'trade_amount': trade_amount,
                'original_trade_amount': trade_amount,
                'entry_fee': entry_fee,
                'leverage': actual_leverage,
                'dca_count': 0,
                'waiting_dca': False,
                'is_dca_position': False,
                'is_rescue_position': False,
                'is_test_position': True,
                'chain_order': 0,
                'tp_trailing_active': False,
                'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0,
                'rebuild_count': 0,
                'fills': []
            }
            self.add_fill_event(
                test_pos, 'TEST_OPEN', fill, trade_amount,
                'Mở BTC thật để kiểm tra đường TP'
            )
            self.positions.append(test_pos)

            send_telegram(
                "✅ *TEST MỞ LỆNH OK*\n"
                f"💰 Giá: `{fill_price}`\n"
                f"📦 Khối lượng: `{amount_coin}`\n"
                f"⚙️ Đòn bẩy: `x{actual_leverage}`\n"
                f"⏳ Sau `{TEST_WAIT_SECONDS:g}` giây bot sẽ giả lập đạt TP."
            )

            time.sleep(max(0.0, self._safe_float(TEST_WAIT_SECONDS)))
            close_price = self._safe_float(
                exchange.fetch_ticker(symbol).get('last'),
                fill_price
            )

            realized = self.close_position(
                test_pos,
                close_price,
                "TEST giả lập đã đạt TP ròng 2 USD"
            )

            time.sleep(1.0)
            final_snapshot = self.fetch_okx_position_snapshot_for_symbols([symbol])
            remaining = sum(
                abs(self._safe_float(item.get('contracts')))
                for key, item in (final_snapshot or {}).items()
                if key[0] == symbol
            )
            still_in_bot = test_pos in self.positions

            # Nếu vị thế thật đã bằng 0 thì TEST phải coi là đóng thành công.
            # Dọn ngay dữ liệu ảo còn sót thay vì chờ vòng đồng bộ kế tiếp.
            if remaining <= OKX_SYNC_ABS_TOLERANCE and still_in_bot:
                self.remove_position_from_memory(test_pos)
                self._clear_ignored_residual(self.get_position_key(symbol, 'buy'))
                self.rescue_chains.pop(self.get_position_key(symbol, 'buy'), None)
                still_in_bot = False

            if not still_in_bot and remaining <= OKX_SYNC_ABS_TOLERANCE:
                send_telegram(
                    "✅ *TEST TP THÀNH CÔNG*\n"
                    "✅ Mở lệnh thật: OK\n"
                    "✅ Gọi đúng hàm TP: OK\n"
                    "✅ Đóng sạch trên OKX: OK\n"
                    f"💰 Lời/lỗ bot đọc được: `${self._safe_float(realized):.6f}`\n"
                    "📌 Nếu hiện 0 thì OKX đã đóng sạch nhưng phản hồi fill không đầy đủ; TEST vẫn đạt."
                )
                return True

            send_telegram(
                "❌ *TEST TP CHƯA ĐÓNG SẠCH*\n"
                f"🤖 Còn trong bot: `{'Có' if still_in_bot else 'Không'}`\n"
                f"🏦 Khối lượng còn trên OKX: `{remaining}`\n"
                "📌 Không chạy TEST lần nữa; kiểm tra OKX và dùng `Đ BTC` nếu cần."
            )
            return False

        except Exception as e:
            send_telegram(
                "❌ *TEST MỞ/ĐÓNG THẤT BẠI*\n"
                f"`{e}`\n"
                "📌 Kiểm tra OKX xem lệnh BTC TEST có còn mở không."
            )
            return False
        finally:
            self.test_order_running = False

    @synchronized_trading
    def close_symbol_by_telegram(self, text):
        if self.view_mode:
            send_telegram(
                "👁 *BOT ĐANG Ở CHẾ ĐỘ XEM*\n"
                "Bot không được phép tự đóng lệnh trong chế độ này.\n"
                "📌 Hãy đóng toàn bộ coin trực tiếp trên OKX; bot sẽ tự nhận biết và xóa khỏi bộ nhớ."
            )
            return False

        coin = self.normalize_symbol_query(text)
        if not coin:
            send_telegram("⚠️ Cú pháp đúng: `Đ BTC`")
            return False

        groups = {}
        for pos in self.positions:
            if self.symbol_base_name(pos['symbol']) == coin:
                groups.setdefault((pos['symbol'], pos['side']), []).append(pos)

        if not groups:
            send_telegram(f"🔎 Bot không quản lý lệnh nào của `{coin}`, không có gì để đóng.")
            return False

        send_telegram(
            f"🛑 *NHẬN LỆNH ĐÓNG COIN THỦ CÔNG*\n"
            f"📍 `{coin}`\n"
            f"📦 Số lệnh riêng chuẩn bị đóng: `{sum(len(g) for g in groups.values())}`"
        )

        all_ok = True
        for (symbol, side), group in list(groups.items()):
            snapshot = self.fetch_okx_position_snapshot_for_symbols([symbol])
            if snapshot is None:
                send_telegram(
                    f"❌ Không xác minh được vị thế OKX của `{symbol}`; "
                    "bot không đóng và không xóa dữ liệu lệnh."
                )
                all_ok = False
                continue
            okx_pos = snapshot.get((symbol, side)) if snapshot else None
            total_amount = (
                abs(self._safe_float(okx_pos.get('contracts')))
                if okx_pos else 0.0
            )
            if total_amount <= OKX_SYNC_ABS_TOLERANCE:
                self.forget_stale_group_when_okx_empty(
                    (symbol, side), group,
                    "Lệnh đóng Telegram kiểm tra và thấy OKX đã hết vị thế."
                )
                continue
            requested_amount = self._safe_float(exchange.amount_to_precision(symbol, total_amount))
            if requested_amount <= 0:
                send_telegram(f"❌ `{symbol}` có khối lượng đóng bằng 0 sau khi làm tròn.")
                all_ok = False
                continue

            close_side = 'sell' if side == 'buy' else 'buy'
            try:
                ticker = exchange.fetch_ticker(symbol)
                fallback_price = self._safe_float(ticker.get('last'))
                clid = self.make_client_order_id('MANUAL', group[0].get('position_id'))
                order = exchange.create_market_order(
                    symbol,
                    close_side,
                    requested_amount,
                    params={
                        'tdMode': MARGIN_MODE,
                        'reduceOnly': True,
                        'posSide': 'long' if side == 'buy' else 'short',
                        'clOrdId': clid,
                    }
                )
                fill = self.resolve_order_fill(order, symbol, fallback_price, requested_amount)
                fill_amount = min(self._safe_float(fill.get('amount'), requested_amount), total_amount)
                fill_price = self._safe_float(fill.get('price'), fallback_price)
                exit_fee = self._safe_float(fill.get('fee'))
                if exit_fee <= 0:
                    market = exchange.market(symbol)
                    contract_size = self._safe_float(market.get('contractSize'), 1.0)
                    exit_fee = fill_amount * fill_price * contract_size * FEE_RATE

                # Nếu OKX chỉ khớp một phần tổng vị thế (kể cả residual), giữ lại
                # một phần dữ liệu ảo để bot không bỏ quên khối lượng còn trên sàn.
                close_ratio = min(1.0, fill_amount / total_amount) if total_amount > 0 else 0.0
                raw_pnl = 0.0
                allocated_entry_fee = 0.0

                for pos in group[:]:
                    old_amount = self._safe_float(pos.get('amount_coin'))
                    allocated_amount = old_amount * close_ratio
                    if allocated_amount <= 0:
                        continue
                    raw_pnl += self.calculate_realized_pnl_from_fill(pos, fill_price, allocated_amount)
                    allocated_entry_fee += self._safe_float(pos.get('entry_fee')) * close_ratio
                    pos_fill = dict(fill)
                    pos_fill['amount'] = allocated_amount
                    self.add_fill_event(pos, 'MANUAL_CLOSE', pos_fill, -self._safe_float(pos.get('trade_amount')) * close_ratio, 'Đóng bằng Telegram')

                    if close_ratio >= 0.999999:
                        self.remove_position_from_memory(pos)
                    else:
                        pos['amount_coin'] = max(0.0, old_amount - allocated_amount)
                        pos['trade_amount'] *= (1.0 - close_ratio)
                        pos['entry_fee'] *= (1.0 - close_ratio)

                net_pnl = raw_pnl - allocated_entry_fee - exit_fee
                self.balance += raw_pnl - exit_fee
                self.coins[symbol]['last_close_time'] = time.time()
                if fill_amount + OKX_SYNC_ABS_TOLERANCE >= total_amount:
                    self._clear_ignored_residual((symbol, side))
                    self.rescue_chains.pop((symbol, side), None)

                send_telegram(
                    f"✅ *ĐÃ ĐÓNG {coin} BẰNG TELEGRAM*\n"
                    f"📍 `{symbol}` - `{side.upper()}`\n"
                    f"💰 Giá khớp OKX: `{fill_price}`\n"
                    f"📦 Khối lượng khớp: `{fill_amount}/{total_amount}`\n"
                    f"💸 Phí đóng: `${exit_fee:.6f}`\n"
                    f"💰 Lời/lỗ ròng ước tính: `${net_pnl:.4f}`"
                )

                if fill_amount + OKX_SYNC_ABS_TOLERANCE < total_amount:
                    all_ok = False
                    send_telegram(
                        f"⚠️ `{symbol}` chỉ khớp một phần. Bot giữ lại phần khối lượng chưa đóng để tiếp tục quản lý."
                    )

            except Exception as e:
                all_ok = False
                send_telegram(f"❌ Lỗi đóng `{symbol}` bằng Telegram:\n`{e}`")

        if all_ok:
            send_telegram(f"🔄 `{coin}` đã đóng toàn bộ và reset khỏi bộ nhớ bot.")
        return all_ok

    @synchronized_trading
    def enable_view_mode(self):
        if self.view_mode:
            send_telegram(
                "ℹ️ Bot đã ở *CHẾ ĐỘ XEM* rồi.\n"
                f"📌 Đang ghi nhớ `{len(self.positions)}` lệnh ảo."
            )
            return

        self.view_mode = True
        self.last_view_sync_time = 0.0

        # Hủy các tín hiệu vào lệnh đang chờ để chúng không chạy lại khi bật chế độ thường.
        for coin_state in self.coins.values():
            self.reset_coin_signal(coin_state)

        # Không giữ yêu cầu DCA đang xếp hàng. Khi quay lại chế độ chạy,
        # bot sẽ tự xét lại ngưỡng DCA theo giá hiện tại.
        for pos in self.positions:
            pos['waiting_dca'] = False

        send_telegram(
            "👁 *ĐÃ BẬT CHẾ ĐỘ XEM*\n"
            "⛔ Không tìm lệnh mới\n"
            "⛔ Không TP, DCA, Global TP, cắt lỗ, bơm hoặc phục hồi\n"
            "🧠 Vẫn giữ nguyên các lệnh trong bộ nhớ\n"
            "🔎 Chỉ kiểm tra các coin bot đang giữ; nếu bạn đóng HẾT trên OKX, bot sẽ tự xóa coin đó khỏi bộ nhớ\n"
            "▶️ Bật lại bằng: `CĐ CHẠY`"
        )

    @synchronized_trading
    def disable_view_mode(self):
        if self.recovery_session is not None:
            send_telegram(
                "⛔ Chưa thể bật CHẾ ĐỘ CHẠY vì `KHOI PHUC` chưa nhập xong.\n"
                "📌 Hãy trả lời đủ giá hoặc nhắn `HUY KHOI PHUC`."
            )
            return
        if not self.view_mode:
            send_telegram("ℹ️ Bot hiện không ở chế độ xem.")
            return

        # Đồng bộ lần cuối trước khi cho phép bot hoạt động lại.
        self.monitor_view_mode_full_closures(force=True)
        self.view_mode = False
        self.last_okx_sync_time = 0.0

        send_telegram(
            "▶️ *ĐÃ TẮT CHẾ ĐỘ XEM*\n"
            "Bot tiếp tục săn lệnh mới và quản lý các lệnh còn tồn tại.\n"
            f"📌 Số lệnh bot còn ghi nhớ: `{len(self.positions)}`"
        )

    def monitor_view_mode_full_closures(self, force=False):
        """Trong chế độ xem, chỉ nhận biết vị thế đã bị đóng HẾT trên OKX."""
        if not self.view_mode or not self.positions:
            return

        now = time.time()
        if not force and now - self.last_view_sync_time < OKX_SYNC_INTERVAL:
            return
        self.last_view_sync_time = now

        snapshot = self.fetch_okx_position_snapshot()
        if snapshot is None:
            return

        self.okx_position_snapshot = snapshot

        groups = {}
        for pos in self.positions:
            key = self.get_position_key(pos['symbol'], pos['side'])
            groups.setdefault(key, []).append(pos)

        for key, group in list(groups.items()):
            symbol, side = key
            bot_amount = sum(
                abs(self._safe_float(pos.get('amount_coin')))
                for pos in group
            )
            okx_pos = snapshot.get(key)
            okx_amount = (
                abs(self._safe_float(okx_pos.get('contracts')))
                if okx_pos else 0.0
            )

            tolerance = max(
                OKX_SYNC_ABS_TOLERANCE,
                bot_amount * OKX_SYNC_REL_TOLERANCE
            )

            # Chỉ xử lý trường hợp đóng HẾT.
            if okx_amount <= OKX_SYNC_ABS_TOLERANCE:
                virtual_count = len(group)

                for pos in group[:]:
                    self.remove_position_from_memory(pos)

                self.coins[symbol]['last_close_time'] = now

                send_telegram(
                    "✅ *PHÁT HIỆN ĐÃ ĐÓNG HẾT TRÊN OKX*\n"
                    f"📍 `{symbol}` - `{side.upper()}`\n"
                    f"📦 Đã xóa `{virtual_count}` lệnh ảo (gốc + DCA) khỏi bộ nhớ bot\n"
                    "🔄 Coin đã được reset hoàn toàn về DCA0\n"
                    "ℹ️ Bot không tự tính lời/lỗ vì lệnh được đóng thủ công ngoài bot."
                )
                continue

            # Nếu có đóng một phần, bot chỉ cảnh báo và tuyệt đối không sửa dữ liệu.
            if abs(bot_amount - okx_amount) > tolerance:
                last_notice = self.last_view_partial_notify.get(key, 0.0)
                if now - last_notice >= OKX_SYNC_NOTIFY_COOLDOWN:
                    self.last_view_partial_notify[key] = now
                    send_telegram(
                        "⚠️ *CHẾ ĐỘ XEM PHÁT HIỆN KHỐI LƯỢNG THAY ĐỔI*\n"
                        f"📍 `{symbol}` - `{side.upper()}`\n"
                        f"🤖 Bot đang nhớ: `{bot_amount}`\n"
                        f"🏦 OKX hiện có: `{okx_amount}`\n"
                        "📌 Bot chỉ hỗ trợ nhận biết khi đóng HẾT, nên chưa thay đổi dữ liệu lệnh ảo."
                    )

    def send_symbol_position_info(self, text):
        query = self.normalize_symbol_query(text)

        if not query:
            return

        matched_positions = [
            p for p in self.positions
            if self.symbol_base_name(p['symbol']) == query
        ]

        if not matched_positions:
            send_telegram(
                f"🔎 Không thấy lệnh nào bot đang quản lý cho `{query}`"
            )
            return

        matched_positions.sort(
            key=lambda p: (
                p.get('root_id', p.get('position_id', 0)),
                0 if not p.get('is_dca_position') else p.get('dca_number', 0),
                p.get('position_id', 0)
            )
        )

        lines = [
            f"📌 *THÔNG TIN LỆNH {query}*"
        ]

        for p in matched_positions:
            current_price = self.update_coin_data(p['symbol'])
            pnl = self.calculate_virtual_pnl(p, current_price) if current_price else 0
            pnl_percent = (pnl / p['trade_amount']) * 100 if p.get('trade_amount', 0) > 0 else 0

            if p.get('is_rescue_position'):
                label = f"RESCUE {p.get('rescue_number', '?')}"
            elif p.get('is_dca_position'):
                label = f"DCA {p.get('dca_number', '?')}"
            else:
                label = "LỆNH GỐC"

            lines.append(
                
                f"\n{label} - `{p['side'].upper()}`"
                f"\n💰 Giá vào: `{p.get('entry_price', 0):,.8f}`"
                f"\n💵 Ký quỹ còn: `${p.get('trade_amount', 0):.4f}`"
                f"\n📦 Khối lượng bot giữ: `{p.get('amount_coin', 0)}`"
                f"\n📈 PnL ước tính: `${pnl:.4f}` (`{pnl_percent:.1f}%`)"
                f"\n🧾 Số fill OKX đã ghi: `{len(p.get('fills', []))}`"
            )

        send_telegram(
            "\n".join(lines)[:3900]
        )

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

        active_dcas = sorted(
            [
                p
                for p in self.positions
                if p.get('root_id') == root_id and p.get('is_dca_position')
            ],
            key=lambda p: (
                int(p.get('dca_number', 0)),
                int(p.get('position_id', 0))
            )
        )

        # Không để hổng cấp DCA: nếu DCA1 đóng trong lúc DCA2 còn tồn tại,
        # DCA2 được đổi vai trò thành DCA1 và cấp tiếp theo vẫn có thể mở lại.
        for number, dca_pos in enumerate(active_dcas, start=1):
            dca_pos['dca_number'] = number
            dca_pos['chain_order'] = number

        active_dca_numbers = [
            p.get('dca_number', 0)
            for p in active_dcas
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
            since = (
                int(c['last_trade_timestamp']) + 1
                if c.get('last_trade_timestamp') is not None else None
            )
            trades = exchange.fetch_trades(symbol, since=since, limit=100)
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
                c['last_trade_timestamp'] = new_trades[-1]['timestamp']

            cutoff = current_time - VOL_WINDOW_SIZE
            while c['vol_trades'] and c['vol_trades'][0][0] < cutoff:
                c['vol_trades'].popleft()

            c['total_buy_30p'] = sum(t[2] for t in c['vol_trades'] if t[1] == 'buy')
            c['total_sell_30p'] = sum(t[2] for t in c['vol_trades'] if t[1] == 'sell')

            if (
                not c['price_history']
                or current_time - c.get('last_ohlcv_fetch_time', 0.0) >= OHLCV_CACHE_SECONDS
            ):
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=60)
                if not ohlcv:
                    return None
                closes = [x[4] for x in ohlcv]
                c['price_history'].clear()
                c['price_history'].extend(closes)
                last_candle = ohlcv[-1]
                c['last_candle_time'] = last_candle[0]
                c['last_open'] = last_candle[1]
                c['last_close'] = last_candle[4]
                c['last_ohlcv_fetch_time'] = current_time

            # Giữ nguyên chiến lược gốc: dùng giá đóng của nến 1h hiện tại.
            current_price = self._safe_float(c['price_history'][-1])
            return current_price


        except Exception as e:
            print(f"Lỗi cập nhật {symbol}: {e}")
            return None

    def get_reversal_5m_candle(self, symbol):
        """Lấy nến 5m đang chạy để xác nhận nhịp đảo chiều sau tín hiệu 1h."""
        now = time.time()
        cached = self.reversal_5m_cache.get(symbol)
        if cached and now - cached['fetch_time'] < REVERSAL_5M_CACHE_SECONDS:
            return cached['candle']
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=2)
            if not ohlcv:
                return None
            row = ohlcv[-1]
            candle = {
                'timestamp': row[0],
                'open': self._safe_float(row[1]),
                'high': self._safe_float(row[2]),
                'low': self._safe_float(row[3]),
                'close': self._safe_float(row[4]),
            }
            if min(
                candle['open'], candle['high'],
                candle['low'], candle['close']
            ) <= 0:
                return None
            self.reversal_5m_cache[symbol] = {
                'fetch_time': now,
                'candle': candle,
            }
            return candle
        except Exception as e:
            print(f"⚠️ Không lấy được nến 5m xác nhận đảo chiều {symbol}: {e}")
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

    def reset_coin_signal(self, coin_state):
        coin_state['pending_side'] = None
        coin_state['waiting_bb'] = False
        coin_state['bb_wait_candle'] = 0
        coin_state['waiting_reversal'] = False
        coin_state['reversal_extreme_price'] = 0
        coin_state['reversal_start_time'] = 0

    def arm_reversal_entry(self, symbol, coin_state, current_price):
        coin_state['waiting_reversal'] = True
        coin_state['reversal_extreme_price'] = current_price
        coin_state['reversal_start_time'] = time.time()

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
        self.guard_unmanaged_okx_positions_on_startup()
        
        while True:

            current_time = time.time()

            # Chế độ XEM đứng trước mọi cơ chế tự động, kể cả bot_paused.
            # Trong nhánh này tuyệt đối không đặt hoặc đóng bất kỳ order nào.
            if self.view_mode:
                self.monitor_view_mode_full_closures()

                if current_time - self.last_status_time >= STATUS_REPORT_INTERVAL:
                    self.send_multi_report()
                    self.last_status_time = current_time

                time.sleep(CHECK_INTERVAL)
                continue

            if not self.is_warmed_up:
                if current_time - self.start_time >= WARMUP_PERIOD:
                    self.is_warmed_up = True
                    send_telegram("✅ *Nạp dữ liệu xong!* Bắt đầu săn tìm cơ hội.")
                else:
                    for sym in SYMBOLS:
                        if self.is_symbol_blacklisted(sym):
                            continue
                        self.update_coin_data(sym)
                        time.sleep(0.01)
                    continue

            # --- TRƯỜNG HỢP 1: ĐI SĂN TÍN HIỆU ---
            # Chỉ slot lệnh gốc mới giới hạn tìm lệnh mới.
            # Lệnh DCA riêng không tính vào slot này.
            if (
                not self.bot_paused
                and not self.search_paused
                and self.count_root_positions() < self.current_max_positions
            ):
                for symbol in SYMBOLS:
                    if self.is_symbol_blacklisted(symbol):
                        continue
                    current_price = self.update_coin_data(symbol)
                    if current_price is None: continue
                    
                    c = self.coins[symbol]
                    # Coin đã có lệnh rồi thì bỏ qua
                    if self.has_open_position(symbol):
                        continue
                    if len(c['price_history']) < 3:
                        continue

                    # price_history dùng nến 1h: đây là giá của 3 mẫu nến gần nhất,
                    # không phải giá 3 phút trước.
                    price_3_candles_ago = c['price_history'][-3]
                    buy_diff = (c['total_buy_30p'] - c['total_sell_30p']) / c['total_sell_30p'] if c['total_sell_30p'] > 0 else 1.0
                    sell_diff = (c['total_sell_30p'] - c['total_buy_30p']) / c['total_buy_30p'] if c['total_buy_30p'] > 0 else 1.0

                    if current_time - c['last_close_time'] >= COOLDOWN_PERIOD:
                        if c['pending_side'] is None:
                            if buy_diff > VOL_DIFF_THRESHOLD and current_price > price_3_candles_ago:
                                c['pending_side'] = 'sell_trigger'
                                c['trigger_price'] = current_price
                                c['trigger_time'] = current_time
                                c['trigger_vol_diff'] = buy_diff
                                c['waiting_reversal'] = False
                                c['reversal_extreme_price'] = current_price
                                c['reversal_start_time'] = 0
                                print(f"🔍 [{symbol}] Chờ SHORT đảo chiều...")
                            elif sell_diff > VOL_DIFF_THRESHOLD and current_price < price_3_candles_ago:
                                c['pending_side'] = 'buy_trigger'
                                c['trigger_price'] = current_price
                                c['trigger_time'] = current_time
                                c['trigger_vol_diff'] = sell_diff
                                c['waiting_reversal'] = False
                                c['reversal_extreme_price'] = current_price
                                c['reversal_start_time'] = 0
                                print(f"🔍 [{symbol}] Chờ LONG đảo chiều...")
                        else:
                            elapsed = current_time - c['trigger_time']
                            price_change = (current_price - c['trigger_price']) / c['trigger_price']
                            
                            if c['pending_side'] == 'sell_trigger':
                                if c.get('waiting_reversal'):
                                    reversal_elapsed = current_time - c.get('reversal_start_time', current_time)
                                    reversal_candle = self.get_reversal_5m_candle(symbol)
                                    if not reversal_candle:
                                        if reversal_elapsed >= REVERSAL_WAIT_TIMEOUT:
                                            self.reset_coin_signal(c)
                                        continue
                                    reversal_price = reversal_candle['close']
                                    c['reversal_extreme_price'] = max(
                                        self._safe_float(c.get('reversal_extreme_price'), reversal_candle['high']),
                                        reversal_candle['high']
                                    )
                                    peak_price = c['reversal_extreme_price']
                                    retrace = (
                                        (peak_price - reversal_price) / peak_price
                                        if peak_price > 0 else 0.0
                                    )
                                    is_reversal_red = (
                                        reversal_candle['close'] < reversal_candle['open']
                                    )
                                    if retrace >= REVERSAL_ENTRY_CALLBACK and is_reversal_red:
                                        send_telegram(
                                            f"✅ `{symbol}` xác nhận đảo chiều SHORT: "
                                            f"nến 5m giảm `{retrace*100:.2f}%` từ đỉnh `{peak_price}`"
                                        )
                                        self.open_position(symbol, 'sell', reversal_price, buy_diff)
                                        self.reset_coin_signal(c)
                                        break

                                    if reversal_elapsed >= REVERSAL_WAIT_TIMEOUT:
                                        print(f"⌛ [{symbol}] SHORT hết thời gian chờ đảo chiều")
                                        self.reset_coin_signal(c)
                                    continue

                                if current_price < c['trigger_price'] * 0.999:
                                    self.reset_coin_signal(c)

                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= SIGNAL_TIMEOUT:
                                        print(f"⌛ [{symbol}] SELL timeout")

                                        c['pending_side'] = None
                                        c['waiting_bb'] = False
                                        c['bb_wait_candle'] = 0
                                        continue
                                    if price_change >= PRICE_SURGE_THRESHOLD and buy_diff > c['trigger_vol_diff']:
                                        upper, middle, lower = self.calculate_bollinger_bands(c['price_history'])

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

                                        self.arm_reversal_entry(symbol, c, current_price)
                                        continue
                                    else:
                                        self.reset_coin_signal(c)
                            
                            elif c['pending_side'] == 'buy_trigger':
                                if c.get('waiting_reversal'):
                                    reversal_elapsed = current_time - c.get('reversal_start_time', current_time)
                                    reversal_candle = self.get_reversal_5m_candle(symbol)
                                    if not reversal_candle:
                                        if reversal_elapsed >= REVERSAL_WAIT_TIMEOUT:
                                            self.reset_coin_signal(c)
                                        continue
                                    reversal_price = reversal_candle['close']
                                    previous_extreme = self._safe_float(
                                        c.get('reversal_extreme_price'), reversal_candle['low']
                                    )
                                    if previous_extreme <= 0:
                                        previous_extreme = reversal_candle['low']
                                    c['reversal_extreme_price'] = min(
                                        previous_extreme,
                                        reversal_candle['low']
                                    )
                                    trough_price = c['reversal_extreme_price']
                                    rebound = (
                                        (reversal_price - trough_price) / trough_price
                                        if trough_price > 0 else 0.0
                                    )
                                    is_reversal_green = (
                                        reversal_candle['close'] > reversal_candle['open']
                                    )
                                    if rebound >= REVERSAL_ENTRY_CALLBACK and is_reversal_green:
                                        send_telegram(
                                            f"✅ `{symbol}` xác nhận đảo chiều LONG: "
                                            f"nến 5m tăng `{rebound*100:.2f}%` từ đáy `{trough_price}`"
                                        )
                                        self.open_position(symbol, 'buy', reversal_price, sell_diff)
                                        self.reset_coin_signal(c)
                                        break

                                    if reversal_elapsed >= REVERSAL_WAIT_TIMEOUT:
                                        print(f"⌛ [{symbol}] LONG hết thời gian chờ đảo chiều")
                                        self.reset_coin_signal(c)
                                    continue

                                if current_price > c['trigger_price'] * 1.001:
                                    self.reset_coin_signal(c)
                                elif elapsed >= CONFIRMATION_TIME:
                                    if elapsed >= SIGNAL_TIMEOUT:
                                        print(f"⌛ [{symbol}] BUY timeout")

                                        c['pending_side'] = None
                                        c['waiting_bb'] = False
                                        c['bb_wait_candle'] = 0
                                        continue
                                    if abs(price_change) >= PRICE_SURGE_THRESHOLD and sell_diff > c['trigger_vol_diff']:
                                        upper, middle, lower = self.calculate_bollinger_bands(c['price_history'])

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

                                        self.arm_reversal_entry(symbol, c, current_price)
                                        continue
                                    else:
                                        self.reset_coin_signal(c)
                    time.sleep(0.01)

            # --- ĐỒNG BỘ VỊ THẾ GỘP OKX + GLOBAL TP ---
            # Lớp OKX chỉ dùng để kiểm tra tổng khối lượng và đóng toàn coin khi tổng đã lời.
            # Giá lệnh gốc/DCA riêng trong self.positions vẫn được giữ nguyên.
            self.sync_okx_positions_and_manage_global_tp()

            # Khi giá hồi về giá trung bình OKX: chốt Rescue lời > 1 USD,
            # ghi sổ lỗ còn treo và gom phần còn lại thành một ROOT mới.
            self.manage_average_price_rescue_reset()

            # --- QUẢN LÝ RESCUE TRƯỚC TP RIÊNG ---
            self.manage_rescue_take_profit()

            # --- QUẢN LÝ DCA / RESCUE / TP RIÊNG ---
            for pos in self.positions[:]:
                symbol = pos['symbol']
                current_price = self.update_coin_data(symbol)
                if not current_price:
                    continue

                # Chỉ lệnh gốc hiện tại điều khiển chuỗi DCA/Rescue.
                if (not pos.get('is_dca_position') and not pos.get('is_rescue_position')):
                    first_price = pos['first_entry_price']
                    if pos['side'] == 'buy':
                        loss_percent = ((first_price - current_price) / first_price) * 100
                    else:
                        loss_percent = ((current_price - first_price) / first_price) * 100

                    chain_key = self.get_position_key(symbol, pos['side'])
                    chain = self.rescue_chains.setdefault(chain_key, {
                        'next_level': 1,
                        'source_position_id': pos['position_id'],
                        'source_slice_index': 0,
                        'next_order': 1,
                        'next_rescue_number': 1,
                        'recovery_debt': self._safe_float(pos.get('recovery_debt')),
                    })

                    # Không còn DCA: mức đi ngược đầu tiên mở RESCUE1.
                    rescue_trigger = chain['next_level'] * (100 / pos['leverage'])
                    if (
                        RESCUE_ENABLED
                        and loss_percent >= rescue_trigger
                        and not chain.get('executing')
                    ):
                        self.execute_rescue(chain_key, current_price)

                # Khi coin đang có Rescue, lệnh gốc/DCA không TP riêng;
                # lợi nhuận dương của chúng được giữ để cộng cho Rescue.
                coin_has_rescue = any(
                    p.get('is_rescue_position') for p in self.positions
                    if p['symbol'] == symbol and p['side'] == pos['side']
                )
                if coin_has_rescue:
                    continue

                # Rescue không TP riêng ở đây; manage_rescue_take_profit xử lý mục tiêu carried_loss + 2 USD.
                if pos.get('is_rescue_position'):
                    continue

                unrealized_pnl = self.calculate_virtual_pnl(pos, current_price)
                exit_fee = (pos['trade_amount'] * pos['leverage']) * FEE_RATE
                net_profit = unrealized_pnl - pos.get('entry_fee', 0.0) - exit_fee
                root_target = max(
                    TP_NET_PROFIT_USD,
                    self._safe_float(pos.get('recovery_debt')) + TP_NET_PROFIT_USD
                )
                if net_profit >= root_target:
                    self.close_position(
                        pos, current_price,
                        f"TP ROOT: bù lỗ treo + {TP_NET_PROFIT_USD:.2f} USD"
                    )
                    continue

            if current_time - self.last_status_time >= STATUS_REPORT_INTERVAL:
                self.send_multi_report()
                self.last_status_time = current_time
            time.sleep(CHECK_INTERVAL)



    def get_position_key(self, symbol, side):
        return (symbol, side)

    def _safe_float(self, value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def get_available_balance(self):
        """Vốn còn có thể cấp cho lệnh mới sau khi trừ ký quỹ đang sử dụng."""
        committed_margin = sum(
            max(0.0, self._safe_float(pos.get('trade_amount')))
            for pos in self.positions
        )
        return max(0.0, self._safe_float(self.balance) - committed_margin)

    def margin_from_fill(self, symbol, amount, price, leverage, fallback=0.0):
        """Quy đổi số contract đã khớp thành ký quỹ thực tế."""
        try:
            market = exchange.market(symbol)
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            leverage = self._safe_float(leverage)
            margin = (
                self._safe_float(amount)
                * self._safe_float(price)
                * contract_size
                / leverage
            ) if leverage > 0 else 0.0
            return margin if margin > 0 else self._safe_float(fallback)
        except Exception:
            return self._safe_float(fallback)

    def refresh_bot_paused_state(self):
        """RESCUE không làm bot dừng săn lệnh ROOT mới."""
        self.bot_paused = False
        return self.bot_paused

    def get_min_order_amount(self, symbol):
        """Khối lượng tối thiểu sàn cho phép đặt lệnh; 0 nếu market không khai báo."""
        try:
            market = exchange.market(symbol)
            limits = market.get('limits') or {}
            amount_limits = limits.get('amount') or {}
            return max(0.0, self._safe_float(amount_limits.get('min')))
        except Exception:
            return 0.0

    def forget_stale_group_when_okx_empty(self, key, group, reason):
        """OKX đã hết vị thế nhưng bot còn dữ liệu ảo: dọn bộ nhớ thay vì khóa TP/DCA."""
        symbol, side = key
        removed = 0
        for pos in list(group):
            if pos in self.positions:
                self.remove_position_from_memory(pos)
                removed += 1
        self._clear_ignored_residual(key)
        self.rescue_chains.pop(key, None)
        if symbol in self.coins:
            self.coins[symbol]['last_close_time'] = time.time()
        send_telegram(
            f"🧹 *ĐÃ DỌN DỮ LIỆU ẢO KHÔNG CÒN TRÊN OKX*\n"
            f"📍 `{symbol}` - `{side.upper()}`\n"
            f"📦 Đã xóa `{removed}` lệnh ảo khỏi bộ nhớ bot.\n"
            f"ℹ️ {reason}\n"
            f"✅ Coin này không còn bị khóa TP/DCA."
        )
        return True

    def _okx_side_from_position(self, position):
        side = str(position.get('side') or '').lower()
        if side in ('long', 'buy'):
            return 'buy'
        if side in ('short', 'sell'):
            return 'sell'

        info = position.get('info') or {}
        pos_side = str(info.get('posSide') or '').lower()
        if pos_side == 'long':
            return 'buy'
        if pos_side == 'short':
            return 'sell'
        return None


    def fetch_okx_position_snapshot_for_symbols(self, symbols):
        """Lấy vị thế OKX cho symbol cụ thể, kể cả khi bot chưa quản lý symbol đó."""
        symbols = sorted({s for s in (symbols or []) if s})
        if not symbols:
            return {}
        try:
            try:
                raw_positions = exchange.fetch_positions(symbols)
            except Exception:
                raw_positions = exchange.fetch_positions()

            snapshot = {}
            for item in raw_positions or []:
                symbol = item.get('symbol')
                side = self._okx_side_from_position(item)
                contracts = abs(self._safe_float(item.get('contracts')))
                if not symbol or side is None or contracts <= 0 or symbol not in symbols:
                    continue
                info = item.get('info') or {}
                snapshot[(symbol, side)] = {
                    'symbol': symbol,
                    'side': side,
                    'contracts': contracts,
                    'entry_price': self._safe_float(
                        item.get('entryPrice'),
                        self._safe_float(info.get('avgPx'))
                    ),
                    'mark_price': self._safe_float(
                        item.get('markPrice'),
                        self._safe_float(info.get('markPx'))
                    ),
                    'unrealized_pnl': self._safe_float(
                        item.get('unrealizedPnl'),
                        self._safe_float(info.get('upl'))
                    ),
                    'leverage': self._safe_float(
                        item.get('leverage'),
                        self._safe_float(info.get('lever'), LEVERAGE)
                    ),
                    'raw': item,
                }
            return snapshot
        except Exception as e:
            print(f"⚠️ Không lấy được vị thế OKX cho TEST: {e}")
            return None

    def fetch_okx_position_snapshot(self):
        """Lấy vị thế gộp thật từ OKX, theo symbol + chiều."""
        symbols = sorted({p['symbol'] for p in self.positions})
        if not symbols:
            return {}

        try:
            try:
                raw_positions = exchange.fetch_positions(symbols)
            except Exception:
                raw_positions = exchange.fetch_positions()

            snapshot = {}
            for item in raw_positions or []:
                symbol = item.get('symbol')
                side = self._okx_side_from_position(item)
                contracts = abs(self._safe_float(item.get('contracts')))

                if not symbol or side is None or contracts <= 0:
                    continue
                if symbol not in symbols:
                    continue

                info = item.get('info') or {}
                entry_price = self._safe_float(
                    item.get('entryPrice'),
                    self._safe_float(info.get('avgPx'))
                )
                mark_price = self._safe_float(
                    item.get('markPrice'),
                    self._safe_float(info.get('markPx'))
                )
                unrealized_pnl = self._safe_float(
                    item.get('unrealizedPnl'),
                    self._safe_float(info.get('upl'))
                )
                notional = abs(self._safe_float(
                    item.get('notional'),
                    self._safe_float(info.get('notionalUsd'))
                ))
                margin = abs(self._safe_float(
                    item.get('initialMargin'),
                    self._safe_float(info.get('margin'))
                ))

                snapshot[(symbol, side)] = {
                    'symbol': symbol,
                    'side': side,
                    'contracts': contracts,
                    'entry_price': entry_price,
                    'mark_price': mark_price,
                    'unrealized_pnl': unrealized_pnl,
                    'notional': notional,
                    'margin': margin,
                    'raw': item,
                }

            return snapshot

        except Exception as e:
            print(f"⚠️ Không lấy được vị thế OKX để đồng bộ: {e}")
            return None

    def _notify_sync_mismatch(self, key, bot_amount, okx_amount):
        """Đã tắt hoàn toàn cảnh báo/khóa coin do lệch dữ liệu bot và OKX."""
        return

    def sync_okx_positions_and_manage_global_tp(self):
        if not OKX_SYNC_ENABLED or not self.positions:
            return

        now = time.time()
        if now - self.last_okx_sync_time < OKX_SYNC_INTERVAL:
            return
        self.last_okx_sync_time = now

        snapshot = self.fetch_okx_position_snapshot()
        if snapshot is None:
            return

        self.okx_position_snapshot = snapshot

        bot_groups = {}
        for pos in self.positions:
            key = self.get_position_key(pos['symbol'], pos['side'])
            bot_groups.setdefault(key, []).append(pos)

        for key, group in bot_groups.items():
            managed_amount = sum(abs(self._safe_float(p.get('amount_coin'))) for p in group)
            ignored_amount = abs(self._safe_float(self.ignored_residuals.get(key)))
            bot_amount = managed_amount + ignored_amount
            okx_pos = snapshot.get(key)
            okx_amount = self._safe_float(okx_pos.get('contracts')) if okx_pos else 0.0

            # Nếu OKX đã hết vị thế thật nhưng bot còn dữ liệu ảo, tự dọn bộ nhớ.
            if okx_amount <= OKX_SYNC_ABS_TOLERANCE and bot_amount > OKX_SYNC_ABS_TOLERANCE:
                self.forget_stale_group_when_okx_empty(
                    key,
                    group,
                    "OKX báo khối lượng bằng 0 nhưng bot vẫn còn ghi nhớ vị thế."
                )
                continue

            # Không còn so sánh để khóa coin, không gửi cảnh báo lệch dữ liệu.
            # Global TP dùng trực tiếp vị thế thật do OKX trả về.
            if GLOBAL_TP_ENABLED and okx_pos:
                self.try_global_tp_for_group(key, group, okx_pos)

    @synchronized_trading
    def try_global_tp_for_group(self, key, group, okx_pos):
        """Nếu vị thế gộp thật đã lời đủ thì đóng lần lượt, lệnh cuối dọn sạch vị thế thật."""
        if any(p.get('is_rescue_position') for p in group):
            return False

        total_margin = sum(max(0.0, self._safe_float(p.get('trade_amount'))) for p in group)
        if total_margin <= 0:
            return False

        gross_pnl = self._safe_float(okx_pos.get('unrealized_pnl'))
        total_entry_fees = sum(max(0.0, self._safe_float(p.get('entry_fee'))) for p in group)
        notional = abs(self._safe_float(okx_pos.get('notional')))
        if notional <= 0:
            market = exchange.market(key[0])
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            mark_price = self._safe_float(okx_pos.get('mark_price'))
            notional = self._safe_float(okx_pos.get('contracts')) * mark_price * contract_size

        estimated_exit_fee = notional * FEE_RATE
        net_pnl = gross_pnl - total_entry_fees - estimated_exit_fee
        net_roe = (net_pnl / total_margin) * 100
        recovery_debt = max(
            [self._safe_float(p.get('recovery_debt')) for p in group] + [0.0]
        )
        required_profit = max(
            GLOBAL_TP_MIN_NET_USD,
            total_margin * GLOBAL_TP_MARGIN_RATIO,
            recovery_debt + TP_NET_PROFIT_USD,
        )
        if net_pnl < required_profit or net_roe < GLOBAL_TP_MIN_ROE_PERCENT:
            return False

        symbol, side = key
        closed_count = 0
        realized_total = 0.0
        ordered_group = sorted(
            [p for p in group if p in self.positions],
            key=lambda p: (p.get('is_dca_position', False), p.get('dca_number', 0), p.get('position_id', 0))
        )
        for p in ordered_group:
            if p not in self.positions:
                continue
            price = self._safe_float(okx_pos.get('mark_price')) or self.update_coin_data(symbol)
            if not price:
                return False
            result = self.close_position(p, price, 'GLOBAL TP toàn coin')
            if p not in self.positions:
                closed_count += 1
            if result:
                realized_total += result

        remaining = [p for p in self.positions if p['symbol'] == symbol and p['side'] == side]
        if remaining:
            send_telegram(
                f"⚠️ *GLOBAL TP CHƯA ĐÓNG HẾT*\n"
                f"📍 `{symbol}` - `{side.upper()}`\n"
                f"📦 Còn `{len(remaining)}` lệnh ảo; bot sẽ tiếp tục thử ở vòng sau."
            )
            return False

        self._clear_ignored_residual(key)
        self.rescue_chains.pop(key, None)
        self.current_max_positions = self.max_positions
        self.active_dca_symbol = None
        self.refresh_bot_paused_state()
        send_telegram(
            f"🎯 *GLOBAL TP TOÀN COIN*\n"
            f"📍 `{symbol}` - `{side.upper()}`\n"
            f"📚 Giá trung bình OKX: `{self._safe_float(okx_pos.get('entry_price'))}`\n"
            f"📦 Đã xử lý `{closed_count}` lệnh bot (gốc + DCA)\n"
            f"💵 Tổng ký quỹ bot: `${total_margin:.4f}`\n"
            f"📈 PnL gộp OKX trước khi đóng: `${gross_pnl:.4f}`\n"
            f"💸 Phí vào + phí đóng ước tính: `${total_entry_fees + estimated_exit_fee:.4f}`\n"
            f"✅ Lời ròng ước tính trước khi đóng: `${net_pnl:.4f}` (`{net_roe:.1f}%` ký quỹ)\n"
            f"💰 Tổng kết quả lệnh ảo ghi nhận: `${realized_total:.4f}`\n"
            f"🔄 Coin đã được reset hoàn toàn về DCA0"
        )
        return True

    def add_profit_to_loss_bank(self, tp_profit):
        if not LOSS_BANK_ENABLED or tp_profit <= 0:
            return 0.0

        contribution = tp_profit * LOSS_CUT_PROFIT_USAGE
        self.loss_bank += contribution

        now = time.time()
        if now - self.last_loss_bank_notify >= LOSS_BANK_NOTIFY_COOLDOWN:
            self.last_loss_bank_notify = now
            send_telegram(
                f"🏦 *QUỸ CẮT LỖ ĐƯỢC CỘNG THÊM*\n"
                f"💰 Lời vừa đóng: `${tp_profit:.4f}`\n"
                f"➕ Trích vào quỹ ({LOSS_CUT_PROFIT_USAGE*100:.0f}%): `${contribution:.4f}`\n"
                f"📊 Số dư quỹ hiện tại: `${self.loss_bank:.4f}`"
            )

        return contribution

    @synchronized_trading
    def try_cut_loser_from_bank(self):
        if not LOSS_BANK_ENABLED or self.loss_bank <= 0:
            return False
        loser, loser_price, biggest_loss, loss_percent = self.find_biggest_loser_for_cut()
        if not loser or not loser_price:
            return False
        key = self.get_position_key(loser['symbol'], loser['side'])
        full_exit_fee = loser['trade_amount'] * loser['leverage'] * FEE_RATE
        full_cost = biggest_loss + full_exit_fee
        if full_cost <= 0:
            return False
        close_ratio = min(self.loss_bank / full_cost, LOSS_BANK_MAX_CUT_RATIO, 0.90)
        if close_ratio < LOSS_BANK_MIN_CUT_RATIO:
            return False
        symbol = loser['symbol']
        requested_amount = self._safe_float(exchange.amount_to_precision(symbol, loser['amount_coin'] * close_ratio))
        if requested_amount <= 0:
            return False
        close_side = 'sell' if loser['side'] == 'buy' else 'buy'
        try:
            clid = self.make_client_order_id('BANKCUT', loser.get('position_id'))
            order = exchange.create_market_order(symbol, close_side, requested_amount, params={
                'tdMode': MARGIN_MODE, 'reduceOnly': True,
                'posSide': 'long' if loser['side'] == 'buy' else 'short',
                'clOrdId': clid,
            })
            fill = self.resolve_order_fill(order, symbol, loser_price, requested_amount)
            fill_amount = min(fill['amount'] or requested_amount, loser['amount_coin'])
            fill_price = fill['price'] or loser_price
            realized_pnl = self.calculate_realized_pnl_from_fill(loser, fill_price, fill_amount)
            exit_fee = fill['fee'] if fill['fee'] > 0 else abs(fill_amount / loser['amount_coin']) * full_exit_fee
            actual_cost = max(0.0, -realized_pnl) + exit_fee
            if actual_cost > self.loss_bank + 0.05:
                send_telegram(f"⚠️ Chi phí cắt thực tế `${actual_cost:.4f}` lớn hơn quỹ `${self.loss_bank:.4f}`")
            ratio = fill_amount / loser['amount_coin']
            old_margin = loser['trade_amount']
            loser['amount_coin'] -= fill_amount
            loser['trade_amount'] *= max(0.0, 1 - ratio)
            loser['entry_fee'] *= max(0.0, 1 - ratio)
            self.add_fill_event(loser, 'BANK_CUT', fill, -(old_margin * ratio), 'Cắt bằng quỹ')
            self.loss_bank = max(0.0, self.loss_bank - actual_cost)
            self.balance += realized_pnl - exit_fee
            send_telegram(
                f"🧯 *CẮT LỆNH BẰNG FILL OKX*\n📍 `{symbol}`\n"
                f"💰 Giá đóng khớp: `{fill_price}`\n📦 Khối lượng khớp: `{fill_amount}`\n"
                f"💸 Lỗ + phí thực tế: `${actual_cost:.4f}`\n🏦 Quỹ còn: `${self.loss_bank:.4f}`"
            )
            self.close_tiny_position_if_needed(loser, fill_price)
            return True
        except Exception as e:
            send_telegram(f"❌ Lỗi cắt lệnh bằng quỹ `{symbol}`:\n`{e}`")
            return False

    def make_client_order_id(self, action, position_id=None):
        """Tạo clOrdId ngắn để truy dấu order trên OKX."""
        action = ''.join(ch for ch in str(action).upper() if ch.isalnum())[:8]
        pid = int(position_id or 0)
        stamp = int(time.time() * 1000) % 10000000000
        return f"B{pid}{action}{stamp}"[:32]

    def resolve_order_fill(self, order, symbol, fallback_price=None, fallback_amount=None):
        """Đọc giá, khối lượng và phí khớp thật của chính order từ OKX."""
        order = order or {}
        order_id = order.get('id')
        fresh = order
        matched_trades = []

        if order_id:
            for _ in range(8):
                try:
                    time.sleep(0.35)
                    fresh = exchange.fetch_order(order_id, symbol) or fresh
                    status = str(fresh.get('status') or '').lower()
                    filled = self._safe_float(fresh.get('filled'))
                    if filled > 0 and status in ('closed', 'filled'):
                        break
                except Exception as e:
                    print(f"⚠️ Chưa đọc được order {order_id}: {e}")

            try:
                trades = exchange.fetch_my_trades(symbol, limit=100)
                matched_trades = [
                    t for t in trades
                    if str(t.get('order')) == str(order_id)
                ]
            except Exception as e:
                print(f"⚠️ Không đọc được fills của order {order_id}: {e}")

        filled_amount = sum(self._safe_float(t.get('amount')) for t in matched_trades)
        if filled_amount <= 0:
            filled_amount = self._safe_float(fresh.get('filled'))
        status = str(fresh.get('status') or order.get('status') or '').lower()
        if filled_amount <= 0:
            raise RuntimeError(
                f"Không xác minh được fill của order {order_id or 'không có id'} "
                f"(status={status or 'unknown'}); không cập nhật vị thế ảo."
            )

        if matched_trades and filled_amount > 0:
            avg_price = sum(
                self._safe_float(t.get('price')) * self._safe_float(t.get('amount'))
                for t in matched_trades
            ) / filled_amount
        else:
            avg_price = self._safe_float(fresh.get('average'))
            if avg_price <= 0:
                avg_price = self._safe_float(fresh.get('price'))
            if avg_price <= 0:
                avg_price = self._safe_float(fallback_price)

        fee = 0.0
        fee_currency = None
        for trade in matched_trades:
            trade_fee = trade.get('fee') or {}
            fee += abs(self._safe_float(trade_fee.get('cost')))
            fee_currency = fee_currency or trade_fee.get('currency')
        if fee <= 0:
            order_fee = fresh.get('fee') or {}
            fee = abs(self._safe_float(order_fee.get('cost')))
            fee_currency = order_fee.get('currency')
        if fee <= 0:
            fees = fresh.get('fees') or []
            fee = sum(abs(self._safe_float(x.get('cost'))) for x in fees)
            fee_currency = next((x.get('currency') for x in fees if x.get('currency')), None)

        return {
            'order_id': order_id,
            'client_order_id': fresh.get('clientOrderId') or order.get('clientOrderId'),
            'price': avg_price,
            'amount': filled_amount,
            'fee': fee,
            'fee_currency': fee_currency,
            'timestamp': fresh.get('timestamp') or int(time.time() * 1000),
            'status': status,
            'trades': matched_trades,
        }

    def add_fill_event(self, pos, action, fill, margin_change=0.0, note=''):
        pos.setdefault('fills', []).append({
            'action': action,
            'order_id': fill.get('order_id'),
            'client_order_id': fill.get('client_order_id'),
            'price': self._safe_float(fill.get('price')),
            'amount': self._safe_float(fill.get('amount')),
            'fee': self._safe_float(fill.get('fee')),
            'fee_currency': fill.get('fee_currency'),
            'margin_change': self._safe_float(margin_change),
            'timestamp': fill.get('timestamp') or int(time.time() * 1000),
            'note': note,
        })

    def weighted_entry_after_add(self, old_price, old_amount, fill_price, fill_amount):
        total = old_amount + fill_amount
        if total <= 0:
            return old_price
        return ((old_price * old_amount) + (fill_price * fill_amount)) / total

    def calculate_realized_pnl_from_fill(self, pos, fill_price, fill_amount):
        market = exchange.market(pos['symbol'])
        contract_size = self._safe_float(market.get('contractSize'), 1.0)
        if pos['side'] == 'buy':
            return (fill_price - pos['entry_price']) * fill_amount * contract_size
        return (pos['entry_price'] - fill_price) * fill_amount * contract_size

    def resolve_order_fill_price(self, order, symbol, fallback_price):
        """Tương thích code cũ: trả về giá khớp thật từ resolve_order_fill."""
        return self.resolve_order_fill(order, symbol, fallback_price).get('price') or float(fallback_price)

    def create_entry_order_with_leverage_fallback(
        self,
        symbol,
        side,
        price,
        trade_amount,
        preferred_leverage
    ):
        """
        Mở lệnh và tự hạ leverage nếu OKX báo vượt giới hạn của coin.
        Trả về: (order, leverage_thực_tế, amount_coin, entry_fee)
        """
        candidates = []

        for lev in [preferred_leverage, LEVERAGE, 20, 10, 5, 3, 2, 1]:
            try:
                lev = int(lev)
            except Exception:
                continue

            if lev > int(preferred_leverage):
                continue

            if lev > 0 and lev not in candidates:
                candidates.append(lev)

        last_error = None

        for leverage_value in candidates:
            try:
                exchange.set_leverage(
                    leverage_value,
                    symbol,
                    params={"mgnMode": MARGIN_MODE}
                )

                market = exchange.market(symbol)

                contract_size = float(
                    market.get("contractSize") or 1
                )

                position_value = (
                    trade_amount *
                    leverage_value
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

                if amount_coin <= 0:
                    raise ValueError(
                        f"Khối lượng sau làm tròn bằng 0 cho {symbol}"
                    )

                entry_fee = (
                    trade_amount *
                    leverage_value
                ) * FEE_RATE

                order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=amount_coin,
                    params={
                        "tdMode": MARGIN_MODE,
                        "posSide": "long" if side == "buy" else "short",
                        "clOrdId": self.make_client_order_id("ENTRY")
                    }
                )

                # Đồng bộ leverage của toàn bộ lệnh ảo cùng coin,
                # vì OKX áp leverage theo vị thế cùng symbol/posSide.
                for existing_pos in self.positions:
                    if (
                        existing_pos.get('symbol') == symbol
                        and existing_pos.get('side') == side
                    ):
                        existing_pos['leverage'] = leverage_value

                if leverage_value != preferred_leverage:
                    send_telegram(
                        f"⚙️ {symbol} không dùng được x{preferred_leverage}, "
                        f"đã tự hạ xuống x{leverage_value}"
                    )

                return (
                    order,
                    leverage_value,
                    amount_coin,
                    entry_fee
                )

            except Exception as e:
                last_error = e
                error_text = str(e)

                is_leverage_limit_error = (
                    "51186" in error_text
                    or "exceeds the platform limit" in error_text
                    or "leverage" in error_text.lower()
                    and "limit" in error_text.lower()
                )

                if is_leverage_limit_error:
                    print(
                        f"⚠️ {symbol} không mở được ở x{leverage_value}, "
                        f"thử leverage thấp hơn..."
                    )
                    continue

                raise

        raise RuntimeError(
            f"Không tìm được leverage phù hợp cho {symbol}: {last_error}"
        )



    def _estimated_net_pnl(self, pos, current_price):
        """PnL ròng ước tính sau phí vào và phí đóng."""
        gross = self.calculate_virtual_pnl(pos, current_price)
        exit_fee = pos.get('trade_amount', 0.0) * pos.get('leverage', LEVERAGE) * FEE_RATE
        return gross - pos.get('entry_fee', 0.0) - exit_fee

    def _get_chain_positions(self, key):
        symbol, side = key
        return sorted(
            [p for p in self.positions if p['symbol'] == symbol and p['side'] == side],
            key=lambda p: (p.get('chain_order', 999999), p.get('position_id', 0))
        )

    def _record_rescue_close_for_reset(self, key, position_id, virtual_credit):
        """Ghi lời ảo và lỗ theo giá gộp OKX để tính ROOT mới sau này."""
        chain = self.rescue_chains.setdefault(key, {})
        details = self.last_close_accounting.pop(position_id, {})
        chain['realized_rescue_credit'] = (
            max(0.0, self._safe_float(chain.get('realized_rescue_credit')))
            + max(0.0, self._safe_float(virtual_credit))
        )
        chain['realized_okx_close_loss'] = (
            max(0.0, self._safe_float(chain.get('realized_okx_close_loss')))
            + max(0.0, -self._safe_float(details.get('okx_net_pnl')))
        )
        current_debt = max(
            0.0,
            self._safe_float(chain.get('recovery_debt'))
            + self._safe_float(chain.get('realized_okx_close_loss'))
            - self._safe_float(chain.get('realized_rescue_credit'))
        )
        chain_positions = self._get_chain_positions(key)
        if chain_positions:
            chain_positions[0]['recovery_debt'] = current_debt

    @synchronized_trading
    def manage_average_price_rescue_reset(self):
        """
        Khi giá chạm/vượt giá trung bình OKX:
        - TP RESCUE có lời ảo ròng > 1 USD.
        - Lỗ còn treo = lỗ cắt ROOT + lỗ đóng theo OKX - lời RESCUE bot.
        - Gom toàn bộ khối lượng thật còn lại thành một ROOT ở giá trung bình OKX.
        """
        groups = {}
        for p in self.positions:
            key = self.get_position_key(p['symbol'], p['side'])
            groups.setdefault(key, []).append(p)

        rescue_groups = {
            key: group for key, group in groups.items()
            if any(p.get('is_rescue_position') for p in group)
        }
        if not rescue_groups:
            return False

        snapshot = self.fetch_okx_position_snapshot_for_symbols(
            sorted({key[0] for key in rescue_groups})
        )
        if snapshot is None:
            return False

        changed = False
        for key, original_group in list(rescue_groups.items()):
            okx_pos = snapshot.get(key)
            if not okx_pos:
                continue
            okx_avg = self._safe_float(okx_pos.get('entry_price'))
            current_price = (
                self._safe_float(okx_pos.get('mark_price'))
                or self.update_coin_data(key[0])
            )
            if okx_avg <= 0 or current_price <= 0:
                continue
            reached_average = (
                current_price >= okx_avg if key[1] == 'buy'
                else current_price <= okx_avg
            )
            if not reached_average:
                continue

            chain = self.rescue_chains.setdefault(key, {})
            closed_labels = []
            for rescue in sorted(
                [p for p in original_group if p.get('is_rescue_position')],
                key=lambda p: (p.get('rescue_number', 0), p.get('position_id', 0))
            ):
                if rescue not in self.positions:
                    continue
                price = self.update_coin_data(rescue['symbol']) or current_price
                virtual_net = self._estimated_net_pnl(rescue, price)
                if virtual_net <= RESCUE_AVG_TP_MIN_NET_USD:
                    continue
                position_id = rescue.get('position_id')
                number = rescue.get('rescue_number', '?')
                result = self.close_position(
                    rescue, price,
                    f"Giá hồi về trung bình OKX: TP RESCUE{number} lời trên "
                    f"{RESCUE_AVG_TP_MIN_NET_USD:.2f} USD"
                )
                if rescue not in self.positions:
                    self._record_rescue_close_for_reset(key, position_id, result)
                    closed_labels.append(f"RESCUE{number}")

            # Đọc lại OKX sau các lệnh đóng để ROOT mới khớp đúng trạng thái thật.
            fresh = self.fetch_okx_position_snapshot_for_symbols([key[0]])
            if fresh is None:
                continue
            fresh_pos = fresh.get(key)
            remaining_group = [
                p for p in self.positions
                if self.get_position_key(p['symbol'], p['side']) == key
            ]
            if not fresh_pos:
                for p in remaining_group:
                    if p in self.positions:
                        self.positions.remove(p)
                self.rescue_chains.pop(key, None)
                self._clear_ignored_residual(key)
                changed = True
                continue

            amount = abs(self._safe_float(fresh_pos.get('contracts')))
            avg = self._safe_float(fresh_pos.get('entry_price'))
            if amount <= OKX_SYNC_ABS_TOLERANCE or avg <= 0:
                continue

            cut_loss = max(0.0, self._safe_float(chain.get('recovery_debt')))
            okx_close_loss = max(
                0.0, self._safe_float(chain.get('realized_okx_close_loss'))
            )
            rescue_credit = max(
                0.0, self._safe_float(chain.get('realized_rescue_credit'))
            )
            remaining_debt = max(0.0, cut_loss + okx_close_loss - rescue_credit)
            leverage = max(
                1.0,
                self._safe_float(fresh_pos.get('leverage'), LEVERAGE)
            )
            margin = self.margin_from_fill(
                key[0], amount, avg, leverage,
                sum(self._safe_float(p.get('trade_amount')) for p in remaining_group)
            )
            remaining_entry_fees = sum(
                max(0.0, self._safe_float(p.get('entry_fee')))
                for p in remaining_group
            )

            for p in remaining_group:
                if p in self.positions:
                    self.positions.remove(p)
            self._clear_ignored_residual(key)

            root_id = self.make_position_id()
            root = {
                'position_id': root_id, 'root_id': root_id,
                'symbol': key[0], 'side': key[1],
                'entry_price': avg, 'first_entry_price': avg,
                'amount_coin': amount, 'trade_amount': margin,
                'original_trade_amount': margin,
                'entry_fee': remaining_entry_fees,
                'leverage': leverage, 'dca_count': 0,
                'waiting_dca': False, 'is_dca_position': False,
                'is_rescue_position': False, 'chain_order': 0,
                'recovery_debt': remaining_debt,
                'tp_trailing_active': False, 'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0, 'rebuild_count': 0, 'fills': []
            }
            self.positions.append(root)
            self.rescue_chains[key] = {
                'next_level': 1,
                'source_position_id': root_id,
                'source_slice_index': 0,
                'next_order': 1,
                'next_rescue_number': 1,
                'recovery_debt': remaining_debt,
            }
            send_telegram(
                f"🔄 *TẠO ROOT MỚI TẠI GIÁ TRUNG BÌNH OKX*\n"
                f"📍 `{key[0]}` - `{key[1].upper()}`\n"
                f"🏦 Giá ROOT mới: `{avg}`\n"
                f"📦 Khối lượng còn lại: `{amount}`\n"
                f"✅ Rescue đã TP > $1: `{', '.join(closed_labels) if closed_labels else 'Không có'}`\n"
                f"✂️ Lỗ cắt nguồn tích lũy: `${cut_loss:.4f}`\n"
                f"📉 Lỗ đóng theo giá gộp OKX: `${okx_close_loss:.4f}`\n"
                f"📈 Lời RESCUE bot ghi nhận: `${rescue_credit:.4f}`\n"
                f"🧾 Lỗ ROOT mới còn gánh: `${remaining_debt:.4f}`\n"
                f"🎯 TP ROOT mới: lỗ gánh + `${TP_NET_PROFIT_USD:.2f}`\n"
                f"🛟 Nếu giá đi ngược, chuỗi bắt đầu lại từ RESCUE1."
            )
            changed = True
        return changed

    def _promote_rescue_chain(self, key):
        """Khi ROOT nguồn hết, đưa RESCUE cũ nhất thành ROOT; không tạo DCA."""
        chain_positions = self._get_chain_positions(key)
        if not chain_positions:
            self.rescue_chains.pop(key, None)
            return

        # Đánh lại vai trò theo vị trí trong hàng đợi.
        for index, p in enumerate(chain_positions):
            p['chain_order'] = index
            if index == 0:
                p['is_dca_position'] = False
                p['is_rescue_position'] = False
                p.pop('dca_number', None)
                p.pop('rescue_number', None)
                p['root_id'] = p['position_id']
                p['dca_count'] = 0
                p['waiting_dca'] = False
            else:
                p['is_dca_position'] = False
                p['is_rescue_position'] = True
                p.pop('dca_number', None)

        new_root = chain_positions[0]
        for p in chain_positions:
            p['root_id'] = new_root['position_id']

        chain = self.rescue_chains.setdefault(key, {})
        chain['source_position_id'] = new_root['position_id']
        chain['source_slice_index'] = 0
        chain['next_order'] = max((p.get('chain_order', 0) for p in chain_positions), default=0) + 1
        new_root['recovery_debt'] = max(
            0.0,
            self._safe_float(chain.get('recovery_debt'))
            + self._safe_float(chain.get('realized_okx_close_loss'))
            - self._safe_float(chain.get('realized_rescue_credit'))
        )

        send_telegram(
            f"🔄 *DỊCH CHUỖI RESCUE*\n"
            f"📍 `{key[0]}` - `{key[1].upper()}`\n"
            f"✅ Lệnh đầu hàng đợi đã thành LỆNH GỐC\n"
            f"✅ Các lệnh phía sau vẫn là RESCUE; chiến lược không còn DCA"
        )

    @synchronized_trading
    def execute_rescue(self, key, current_price):
        """Cắt 1/3 nguồn và mở Rescue bằng 3 lần ký quỹ phần vừa cắt."""
        chain = self.rescue_chains.get(key)
        if not chain or chain.get('executing'):
            return False
        chain['executing'] = True
        try:
            source = next((p for p in self.positions if p.get('position_id') == chain.get('source_position_id')), None)
            if source is None:
                self._promote_rescue_chain(key)
                return False

            symbol = source['symbol']
            old_amount = self._safe_float(source.get('amount_coin'))
            old_margin = self._safe_float(source.get('trade_amount'))
            pending = chain.get('pending_rescue')
            if not pending and (old_amount <= 0 or old_margin <= 0):
                self.remove_position_from_memory(source)
                self._promote_rescue_chain(key)
                return False

            if pending:
                # Lát nguồn đã cắt ở vòng trước nhưng order mở Rescue lỗi.
                # Thử lại đúng Rescue đó, tuyệt đối không cắt thêm nguồn.
                fill_price = self._safe_float(pending.get('fill_price'), current_price)
                cut_margin = self._safe_float(pending.get('cut_margin'))
                carried_loss = self._safe_float(pending.get('carried_loss'))
                ratio = self._safe_float(pending.get('ratio'))
                slice_index = int(pending.get('slice_index', 0))
            else:
                # Cố định mỗi lát theo 1/3 khối lượng lúc bắt đầu cắt nguồn.
                if chain.get('source_slice_amount', 0) <= 0:
                    chain['source_slice_amount'] = old_amount / RESCUE_SOURCE_SLICES
                slice_index = int(chain.get('source_slice_index', 0))
                raw_cut_amount = old_amount if slice_index >= RESCUE_SOURCE_SLICES - 1 else min(old_amount, chain['source_slice_amount'])
                cut_amount = self._safe_float(exchange.amount_to_precision(symbol, raw_cut_amount))
                if cut_amount <= 0:
                    return False

                close_side = 'sell' if source['side'] == 'buy' else 'buy'
                clid = self.make_client_order_id('RSCUT', source.get('position_id'))
                order = exchange.create_market_order(symbol, close_side, cut_amount, params={
                    'tdMode': MARGIN_MODE, 'reduceOnly': True,
                    'posSide': 'long' if source['side'] == 'buy' else 'short',
                    'clOrdId': clid,
                })
                fill = self.resolve_order_fill(order, symbol, current_price, cut_amount)
                fill_amount = min(fill['amount'] or cut_amount, old_amount)
                fill_price = fill['price'] or current_price
                ratio = min(1.0, fill_amount / old_amount)
                realized = self.calculate_realized_pnl_from_fill(source, fill_price, fill_amount)
                allocated_entry_fee = source.get('entry_fee', 0.0) * ratio
                exit_fee = fill['fee'] if fill['fee'] > 0 else old_margin * ratio * source['leverage'] * FEE_RATE
                carried_loss = max(0.0, -(realized - allocated_entry_fee - exit_fee))
                cut_margin = old_margin * ratio

                source['amount_coin'] = max(0.0, old_amount - fill_amount)
                source['trade_amount'] = max(0.0, old_margin - cut_margin)
                source['entry_fee'] = max(0.0, source.get('entry_fee', 0.0) - allocated_entry_fee)
                self.balance += realized - exit_fee

                # Ghi nhận lát cắt ngay sau khi fill đóng đã thành công. Nếu mở
                # Rescue lỗi, vòng sau dùng pending_rescue thay vì cắt thêm.
                chain['source_slice_index'] = slice_index + 1
                chain['pending_rescue'] = {
                    'fill_price': fill_price,
                    'cut_margin': cut_margin,
                    'carried_loss': carried_loss,
                    'ratio': ratio,
                    'slice_index': slice_index,
                }
                # Lỗ đã thành hiện thực ngay khi lệnh cắt nguồn khớp, không chờ
                # order mở RESCUE thành công mới ghi sổ.
                chain['recovery_debt'] = (
                    max(0.0, self._safe_float(chain.get('recovery_debt')))
                    + carried_loss
                )
                source['recovery_debt'] = max(
                    0.0, self._safe_float(chain.get('recovery_debt'))
                )

            rescue_margin = cut_margin * RESCUE_MULTIPLIER
            available_balance = self.get_available_balance()
            if rescue_margin <= 0 or available_balance <= 0:
                send_telegram(f"⚠️ `{symbol}` đã cắt nguồn nhưng không đủ vốn mở Rescue.")
                return False
            rescue_margin = min(available_balance, rescue_margin)
            entry_order, lev, req_amount, est_fee = self.create_entry_order_with_leverage_fallback(
                symbol=symbol, side=source['side'], price=fill_price,
                trade_amount=rescue_margin, preferred_leverage=source.get('leverage', LEVERAGE)
            )
            rfill = self.resolve_order_fill(entry_order, symbol, fill_price, req_amount)
            rprice = rfill['price'] or fill_price
            ramount = rfill['amount'] or req_amount
            rescue_margin = self.margin_from_fill(
                symbol, ramount, rprice, lev, rescue_margin
            )
            rfee = rfill['fee'] if rfill['fee'] > 0 else est_fee
            self.balance -= rfee

            rescue_id = self.make_position_id()
            rescue_number = int(chain.get('next_rescue_number', 1))
            rescue = {
                'position_id': rescue_id, 'root_id': source.get('root_id', source['position_id']),
                'symbol': symbol, 'side': source['side'],
                'entry_price': rprice, 'first_entry_price': rprice,
                'amount_coin': ramount, 'trade_amount': rescue_margin,
                'original_trade_amount': rescue_margin, 'entry_fee': rfee,
                'leverage': lev, 'dca_count': 0, 'waiting_dca': False,
                'is_dca_position': False, 'is_rescue_position': True,
                'rescue_number': rescue_number,
                'carried_loss': carried_loss,
                'rescue_target_net': carried_loss + RESCUE_PROFIT_USD,
                'realized_support_profit': 0.0,
                'chain_order': chain.get('next_order', 1),
                'fills': []
            }
            self.add_fill_event(rescue, 'RESCUE_OPEN', rfill, rescue_margin, f'Mở Rescue {rescue_number}')
            self.positions.append(rescue)
            chain.pop('pending_rescue', None)
            chain['next_order'] = rescue['chain_order'] + 1
            chain['next_level'] = int(chain.get('next_level', 1)) + 1
            chain['next_rescue_number'] = rescue_number + 1

            send_telegram(
                f"🛟 *MỞ RESCUE {rescue_number}*\n"
                f"📍 `{symbol}`\n"
                f"✂️ Đã cắt `{ratio*100:.1f}%` phần nguồn, lỗ ròng gánh: `${carried_loss:.4f}`\n"
                f"💵 Ký quỹ Rescue: `${rescue_margin:.4f}`\n"
                f"🎯 TP Rescue cần: `${carried_loss + RESCUE_PROFIT_USD:.4f}` = lỗ gánh + `${RESCUE_PROFIT_USD:.2f}`"
            )

            # Nguồn đã hết sau 3 lát: xóa và dịch toàn bộ hàng đợi xuống một bậc.
            remaining_precision = self._safe_float(exchange.amount_to_precision(symbol, source.get('amount_coin', 0.0)))
            if chain['source_slice_index'] >= RESCUE_SOURCE_SLICES or remaining_precision <= 0:
                if source in self.positions:
                    self.positions.remove(source)
                chain['source_slice_amount'] = 0.0
                self._promote_rescue_chain(key)
            return True
        except Exception as e:
            send_telegram(f"❌ Lỗi Rescue `{key[0]}`:\n`{e}`")
            return False
        finally:
            chain['executing'] = False

    def manage_rescue_take_profit(self):
        """RESCUE1/2 tự TP; từ RESCUE3 được dùng lệnh lời ở coin khác."""
        if self.active_rescue_closing:
            return False
        rescues = [p for p in self.positions if p.get('is_rescue_position')]
        if not rescues:
            return False
        rescues.sort(key=lambda p: (p.get('chain_order', 999999), p.get('position_id', 0)))
        rescue_keys = {self.get_position_key(p['symbol'], p['side']) for p in rescues}

        for rescue in rescues:
            rprice = self.update_coin_data(rescue['symbol'])
            if not rprice:
                continue
            rescue_net = self._estimated_net_pnl(rescue, rprice)
            target = self._safe_float(rescue.get('rescue_target_net'), self._safe_float(rescue.get('carried_loss')) + RESCUE_PROFIT_USD)
            realized_support = max(0.0, self._safe_float(rescue.get('realized_support_profit')))

            helpers = []
            if int(rescue.get('rescue_number', 0)) >= RESCUE_EXTERNAL_HELP_FROM_NUMBER:
                for p in self.positions:
                    if p is rescue:
                        continue
                    pkey = self.get_position_key(p['symbol'], p['side'])
                    # Không lấy bất kỳ lệnh nào trong chính chuỗi đang được cứu.
                    if pkey == self.get_position_key(rescue['symbol'], rescue['side']):
                        continue
                    price = self.update_coin_data(p['symbol'])
                    if not price:
                        continue
                    net = self._estimated_net_pnl(p, price)
                    if net > 0:
                        # ROOT và cả RESCUE đang lời của coin khác đều được hỗ trợ.
                        helpers.append((net, p, price))
            helpers.sort(key=lambda x: x[0])

            selected = []
            estimated_total = realized_support + rescue_net
            for net, p, price in helpers:
                if estimated_total >= target:
                    break
                selected.append((net, p, price))
                estimated_total += net
            if estimated_total + 1e-9 < target:
                continue

            self.active_rescue_closing = True
            try:
                closed_helpers = []
                added_support = 0.0
                for _, p, price in selected:
                    if p not in self.positions:
                        continue
                    result = self.close_position(p, price, f"Góp lợi nhuận cho Rescue {rescue.get('rescue_number', '?')}")
                    if result > 0:
                        added_support += result
                        closed_helpers.append(p['symbol'])

                if rescue not in self.positions:
                    return True
                rescue['realized_support_profit'] = realized_support + added_support
                rprice = self.update_coin_data(rescue['symbol']) or rprice
                rescue_net = self._estimated_net_pnl(rescue, rprice)
                available_total = rescue['realized_support_profit'] + rescue_net
                if available_total + 1e-9 < target:
                    send_telegram(
                        f"⏳ *RESCUE CHƯA ĐỦ SAU KHI KHỚP THẬT*\n"
                        f"📍 `{rescue['symbol']}`\n"
                        f"🎯 Mục tiêu: `${target:.4f}`\n"
                        f"🏦 Lời hỗ trợ đã chốt và ghi riêng: `${rescue['realized_support_profit']:.4f}`\n"
                        f"📈 PnL Rescue hiện tại: `${rescue_net:.4f}`\n"
                        f"📌 Bot chưa đóng Rescue và sẽ cộng tiếp ở vòng sau."
                    )
                    return False

                rescue_position_id = rescue.get('position_id')
                rescue_key = self.get_position_key(rescue['symbol'], rescue['side'])
                result = self.close_position(rescue, rprice, 'TP Rescue: lỗ gánh + 2 USD lợi nhuận ròng')
                if rescue in self.positions:
                    return False
                actual_total = rescue.get('realized_support_profit', 0.0) + result
                self._record_rescue_close_for_reset(
                    rescue_key, rescue_position_id, actual_total
                )
                send_telegram(
                    f"🎯 *TP GỘP CHO RESCUE*\n"
                    f"📍 Rescue: `{rescue['symbol']}`\n"
                    f"🎯 Mục tiêu: `${target:.4f}`\n"
                    f"💰 Tổng lời ghi nhận: `${actual_total:.4f}`\n"
                    f"🤝 ROOT/RESCUE coin khác hỗ trợ: `{', '.join(closed_helpers) if closed_helpers else 'Không có'}`\n"
                    f"📌 Chỉ RESCUE{RESCUE_EXTERNAL_HELP_FROM_NUMBER}+ mới được nhận hỗ trợ ngoài coin"
                )
                return True
            finally:
                self.active_rescue_closing = False
        return False

    @synchronized_trading
    def open_position(self, symbol, side, price, vol_diff):
        trade_amount = min(self.get_available_balance(), self.default_trade_amount)
        if trade_amount <= 0:
            send_telegram(f"⚠️ Không đủ vốn khả dụng để mở `{symbol}`.")
            return False
        try:
            order, current_leverage, requested_amount, estimated_fee = self.create_entry_order_with_leverage_fallback(
                symbol=symbol, side=side, price=price, trade_amount=trade_amount, preferred_leverage=LEVERAGE
            )
            fill = self.resolve_order_fill(order, symbol, price, requested_amount)
            fill_price = fill['price'] or price
            amount_coin = fill['amount'] or requested_amount
            trade_amount = self.margin_from_fill(
                symbol, amount_coin, fill_price, current_leverage, trade_amount
            )
            entry_fee = fill['fee'] if fill['fee'] > 0 else estimated_fee
            self.balance -= entry_fee
            position_id = self.make_position_id()
            pos = {
                'position_id': position_id, 'root_id': position_id,
                'symbol': symbol, 'side': side,
                'entry_price': fill_price, 'first_entry_price': fill_price,
                'amount_coin': amount_coin, 'trade_amount': trade_amount,
                'original_trade_amount': trade_amount, 'entry_fee': entry_fee,
                'leverage': current_leverage, 'dca_count': 0,
                'waiting_dca': False, 'is_dca_position': False,
                'is_rescue_position': False, 'chain_order': 0,
                'tp_trailing_active': False, 'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0, 'rebuild_count': 0,
                'fills': []
            }
            self.add_fill_event(pos, 'OPEN', fill, trade_amount, 'Mở lệnh gốc')
            self.positions.append(pos)
            print(f"✅ Đã mở lệnh thật: {symbol} x{current_leverage}, amount={amount_coin}, fill={fill_price}")
        except Exception as e:
            self.bot_paused = True
            print(f"❌ Lỗi mở lệnh: {e}")
            send_telegram(
                f"❌ Lỗi mở lệnh {symbol}:\n`{e}`\n"
                "⏸ Đã dừng săn lệnh mới. Hãy kiểm tra vị thế thật trên OKX trước khi tiếp tục."
            )
            return

        emoji = "🔴" if side == "sell" else "🟢"
        send_telegram(
            f"{emoji} *VÀO LỆNH {side.upper()} ({symbol})*\n"
            f"💰 Giá khớp OKX: `{fill_price:,.8f}`\n"
            f"📦 Khối lượng khớp OKX: `{amount_coin}`\n"
            f"⚙️ Đòn bẩy thực tế: `x{current_leverage}`\n"
            f"📊 Vol chênh lệch: `+{vol_diff*100:.1f}%` 🔥\n"
            f"💸 Phí mở lấy từ OKX/ước tính: `${entry_fee:.6f}`\n"
            f"💵 Ký quỹ: `${trade_amount:,.2f}`"
        )
        for s in SYMBOLS:
            self.reset_coin_signal(self.coins[s])

    @synchronized_trading
    def execute_dca(self, pos):
        if pos not in self.positions:
            return False
        symbol = pos['symbol']
        trade_amount = min(self.get_available_balance(), self.default_trade_amount)
        if trade_amount <= 0:
            pos['waiting_dca'] = False
            send_telegram(f"⚠️ Không đủ vốn khả dụng để DCA `{symbol}`.")
            return False
        if pos.get('is_dca_position') or pos.get('is_rescue_position') or pos.get('dca_count', 0) >= MAX_DCA:
            pos['waiting_dca'] = False
            return
        dca_number = pos['dca_count'] + 1
        try:
            current_price = exchange.fetch_ticker(symbol)['last']
            order, actual_leverage, requested_amount, estimated_fee = self.create_entry_order_with_leverage_fallback(
                symbol=symbol, side=pos['side'], price=current_price,
                trade_amount=trade_amount, preferred_leverage=pos.get('leverage', LEVERAGE)
            )
            fill = self.resolve_order_fill(order, symbol, current_price, requested_amount)
            fill_price = fill['price'] or current_price
            amount_coin = fill['amount'] or requested_amount
            trade_amount = self.margin_from_fill(
                symbol, amount_coin, fill_price, actual_leverage, trade_amount
            )
            entry_fee = fill['fee'] if fill['fee'] > 0 else estimated_fee
            self.balance -= entry_fee
            dca_position_id = self.make_position_id()
            root_id = pos.get('root_id', pos.get('position_id'))
            dca_pos = {
                'position_id': dca_position_id, 'root_id': root_id,
                'symbol': symbol, 'side': pos['side'],
                'entry_price': fill_price, 'first_entry_price': fill_price,
                'amount_coin': amount_coin, 'trade_amount': trade_amount,
                'original_trade_amount': trade_amount, 'entry_fee': entry_fee,
                'leverage': actual_leverage, 'dca_count': 0,
                'waiting_dca': False, 'is_dca_position': True,
                'is_rescue_position': False, 'chain_order': dca_number,
                'dca_number': dca_number, 'parent_entry_price': pos['first_entry_price'],
                'tp_trailing_active': False, 'tp_peak_pnl': 0,
                'tp_trailing_stop_pnl': 0, 'rebuild_count': 0, 'fills': []
            }
            self.add_fill_event(dca_pos, 'DCA_OPEN', fill, trade_amount, f'Mở DCA {dca_number}')
            self.positions.append(dca_pos)
            pos['leverage'] = actual_leverage
            pos['dca_count'] = dca_number
            pos['waiting_dca'] = False
            send_telegram(
                f"📉 Đã mở LỆNH DCA RIÊNG lần {dca_number} cho {symbol}\n"
                f"💰 Giá khớp OKX: `{fill_price}`\n"
                f"📦 Khối lượng khớp: `{amount_coin}`\n"
                f"⚙️ Đòn bẩy: `x{actual_leverage}`\n"
                f"💸 Phí OKX/ước tính: `${entry_fee:.6f}`\n"
                f"💵 Ký quỹ: `${trade_amount:,.2f}`"
            )
            if pos['dca_count'] >= MAX_DCA:
                self.refresh_bot_paused_state()
                send_telegram(
                    f"🚨 {symbol} đã mở đủ {MAX_DCA} lệnh DCA riêng\n"
                    "⏸ Bot chỉ dừng săn lệnh gốc mới; TP, Rescue, đồng bộ và quản lý "
                    "các vị thế đang mở vẫn tiếp tục."
                )
        except Exception as e:
            pos['waiting_dca'] = False
            self.bot_paused = True
            print(f"DCA lỗi: {e}")
            send_telegram(
                f"❌ Lỗi mở lệnh DCA riêng {symbol}:\n`{e}`\n"
                "⏸ Đã dừng săn lệnh mới; quản lý vị thế hiện có vẫn chạy."
            )

    @synchronized_trading
    def rebuild_small_loser_position(self, pos):
        if pos not in self.positions:
            return False
        if pos.get('trade_amount', 0) > REBUILD_TRIGGER_TRADE_AMOUNT or pos.get('rebuilding'):
            return False
        if self.get_available_balance() < REBUILD_ADD_AMOUNT:
            return False
        symbol = pos['symbol']
        current_price = self.update_coin_data(symbol)
        if not current_price:
            return False
        current_pnl = self.calculate_virtual_pnl(pos, current_price)
        loss_percent = abs(current_pnl) / pos['trade_amount'] * 100 if pos['trade_amount'] > 0 else 0
        if current_pnl >= 0 or loss_percent < REBUILD_MIN_LOSS_PERCENT:
            return False
        pos['rebuilding'] = True
        try:
            trade_amount = REBUILD_ADD_AMOUNT
            market = exchange.market(symbol)
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            requested_amount = self._safe_float(exchange.amount_to_precision(
                symbol, (trade_amount * pos['leverage']) / (current_price * contract_size)
            ))
            if requested_amount <= 0:
                return False
            clid = self.make_client_order_id('REBUILD', pos.get('position_id'))
            order = exchange.create_order(symbol, 'market', pos['side'], requested_amount, params={
                'tdMode': MARGIN_MODE,
                'posSide': 'long' if pos['side'] == 'buy' else 'short',
                'clOrdId': clid,
            })
            fill = self.resolve_order_fill(order, symbol, current_price, requested_amount)
            fill_amount = fill['amount'] or requested_amount
            fill_price = fill['price'] or current_price
            old_amount = pos['amount_coin']
            pos['entry_price'] = self.weighted_entry_after_add(pos['entry_price'], old_amount, fill_price, fill_amount)
            pos['first_entry_price'] = pos['entry_price']
            pos['amount_coin'] = old_amount + fill_amount
            actual_margin = (fill_amount * fill_price * contract_size) / pos['leverage']
            pos['trade_amount'] += actual_margin
            fee = fill['fee'] if fill['fee'] > 0 else actual_margin * pos['leverage'] * FEE_RATE
            pos['entry_fee'] += fee
            pos['rebuild_count'] = pos.get('rebuild_count', 0) + 1
            self.add_fill_event(pos, 'REBUILD', fill, actual_margin, 'Bơm lại lệnh nhỏ')
            self.balance -= fee
            send_telegram(
                f"🧱 *BƠM LẠI LỆNH NHỎ THEO FILL OKX*\n📍 `{symbol}`\n"
                f"💰 Giá khớp: `{fill_price}`\n📦 Khối lượng khớp: `{fill_amount}`\n"
                f"➕ Ký quỹ thực tế ước theo fill: `${actual_margin:.4f}`\n"
                f"🎯 Giá riêng mới: `{pos['entry_price']}`"
            )
            return True
        except Exception as e:
            send_telegram(f"❌ Lỗi bơm lại lệnh nhỏ {symbol}:\n`{e}`")
            return False
        finally:
            pos['rebuilding'] = False

    def find_biggest_loser_for_cut(self):

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

            trade_amount = p.get('trade_amount', 0)

            loss_percent = (
                abs(pnl) / trade_amount
            ) * 100 if trade_amount > 0 else 0

            if (
                pnl < 0
                and loss_percent >= LOSS_CUT_TRIGGER_PERCENT
                and abs(pnl) > biggest_loss
            ):

                biggest_loser = p
                biggest_loser_price = current_price
                biggest_loss = abs(pnl)
                biggest_loss_percent = loss_percent

        return biggest_loser, biggest_loser_price, biggest_loss, biggest_loss_percent

    def can_cut_loser_with_profit(self, tp_profit):

        # Kiểm tra trước khi TP: nếu đang có lệnh âm đủ điều kiện,
        # lợi nhuận TP hiện tại có đủ để cắt tối thiểu không.
        loss_budget = tp_profit * LOSS_CUT_PROFIT_USAGE

        if loss_budget <= 0:
            return True, None

        biggest_loser, loser_price, biggest_loss, biggest_loss_percent = self.find_biggest_loser_for_cut()

        # Không có lệnh âm đủ ngưỡng thì TP bình thường.
        if not biggest_loser:
            return True, None

        full_exit_fee = (
            biggest_loser['trade_amount'] *
            biggest_loser['leverage']
        ) * FEE_RATE

        full_loss_with_fee = biggest_loss + full_exit_fee

        if full_loss_with_fee <= 0:
            return True, None

        close_ratio = loss_budget / full_loss_with_fee

        info = {
            'symbol': biggest_loser['symbol'],
            'loss_percent': biggest_loss_percent,
            'biggest_loss': biggest_loss,
            'loss_budget': loss_budget,
            'close_ratio': close_ratio,
            'needed_profit': (MIN_PARTIAL_CLOSE_AMOUNT_RATIO * full_loss_with_fee) / LOSS_CUT_PROFIT_USAGE if LOSS_CUT_PROFIT_USAGE > 0 else 0
        }

        if close_ratio < MIN_PARTIAL_CLOSE_AMOUNT_RATIO:
            return False, info

        return True, info


    @synchronized_trading
    def restore_cut_position_to_original(self, pos, current_price, available_profit):
        if not RESTORE_CUT_POSITION_TO_ORIGINAL or not pos or pos not in self.positions:
            return 0
        if not current_price or available_profit <= 0:
            return 0
        target = float(pos.get('original_trade_amount', RESTORE_TARGET_TRADE_AMOUNT))
        missing = target - pos.get('trade_amount', 0)
        restore_amount = min(missing, available_profit, self.get_available_balance())
        if restore_amount < MIN_RESTORE_ADD_AMOUNT:
            return 0
        symbol = pos['symbol']
        try:
            market = exchange.market(symbol)
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            requested_amount = self._safe_float(exchange.amount_to_precision(
                symbol, (restore_amount * pos['leverage']) / (current_price * contract_size)
            ))
            if requested_amount <= 0:
                return 0
            clid = self.make_client_order_id('RESTORE', pos.get('position_id'))
            order = exchange.create_order(symbol, 'market', pos['side'], requested_amount, params={
                'tdMode': MARGIN_MODE,
                'posSide': 'long' if pos['side'] == 'buy' else 'short',
                'clOrdId': clid,
            })
            fill = self.resolve_order_fill(order, symbol, current_price, requested_amount)
            fill_amount = fill['amount'] or requested_amount
            fill_price = fill['price'] or current_price
            old_amount = pos['amount_coin']
            pos['entry_price'] = self.weighted_entry_after_add(pos['entry_price'], old_amount, fill_price, fill_amount)
            pos['first_entry_price'] = pos['entry_price']
            pos['amount_coin'] = old_amount + fill_amount
            actual_margin = (fill_amount * fill_price * contract_size) / pos['leverage']
            pos['trade_amount'] += actual_margin
            fee = fill['fee'] if fill['fee'] > 0 else actual_margin * pos['leverage'] * FEE_RATE
            pos['entry_fee'] += fee
            pos['restore_count'] = pos.get('restore_count', 0) + 1
            self.add_fill_event(pos, 'RESTORE', fill, actual_margin, 'Phục hồi sau cắt')
            self.balance -= fee
            send_telegram(
                f"🧩 *PHỤC HỒI THEO FILL OKX*\n📍 `{symbol}`\n"
                f"💰 Giá khớp: `{fill_price}`\n📦 Khối lượng khớp: `{fill_amount}`\n"
                f"➕ Ký quỹ thực tế: `${actual_margin:.4f}`\n🎯 Giá riêng mới: `{pos['entry_price']}`"
            )
            return actual_margin
        except Exception as e:
            send_telegram(f"❌ Lỗi phục hồi ký quỹ {symbol}:\n`{e}`")
            return 0

    @synchronized_trading
    def reduce_biggest_loser_after_tp(self, tp_profit):

        # Dùng một phần tiền lời vừa TP để cắt bớt lệnh âm nặng,
        # mục tiêu là giảm khối lượng/ký quỹ của lệnh đang âm mà tổng vẫn còn lời.
        loss_budget = tp_profit * LOSS_CUT_PROFIT_USAGE

        if loss_budget <= 0:
            return

        biggest_loser, biggest_loser_price, biggest_loss, biggest_loss_percent = self.find_biggest_loser_for_cut()

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
            order = exchange.create_market_order(
                symbol,
                close_side,
                close_amount,
                params={
                    "tdMode": MARGIN_MODE,
                    "reduceOnly": True,
                    "posSide": "long" if biggest_loser['side'] == "buy" else "short"
                }
            )
            fill = self.resolve_order_fill(
                order, symbol, biggest_loser_price, close_amount
            )
            filled_amount = min(
                self._safe_float(fill.get('amount')),
                self._safe_float(biggest_loser.get('amount_coin'))
            )
            if filled_amount <= 0:
                raise RuntimeError("OKX chưa xác nhận khối lượng cắt lỗ đã khớp.")

        except Exception as e:

            print(f"❌ Lỗi cắt bớt lệnh âm: {e}")

            send_telegram(
                f"❌ Lỗi cắt bớt lệnh âm {symbol}:\n`{e}`"
            )

            return

        close_ratio = min(
            1.0,
            filled_amount / self._safe_float(biggest_loser.get('amount_coin'))
        )
        fill_price = self._safe_float(fill.get('price'), biggest_loser_price)
        realized_pnl = self.calculate_realized_pnl_from_fill(
            biggest_loser, fill_price, filled_amount
        )
        partial_loss = max(0.0, -realized_pnl)
        partial_exit_fee = self._safe_float(fill.get('fee'))
        if partial_exit_fee <= 0:
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
            f"✂️ Lỗ và phí theo fill: `${estimated_net_loss:.4f}`\n"
            f"📦 Đã đóng: `{close_ratio*100:.1f}%` khối lượng lệnh âm\n"
            f"📊 Ký quỹ còn lại: `${biggest_loser['trade_amount']:.4f}`"
        )

        # Nếu phần còn lại quá nhỏ thì dọn luôn, không để rác trên OKX/bot.
        if self.close_tiny_position_if_needed(biggest_loser, biggest_loser_price):
            return

        # Dùng phần lời TP còn lại để bơm lại lệnh vừa bị cắt,
        # tối đa đưa ký quỹ của lệnh đó về 5$.
        remaining_tp_profit = max(
            0,
            tp_profit - estimated_net_loss
        )

        restored_amount = self.restore_cut_position_to_original(
            biggest_loser,
            biggest_loser_price,
            remaining_tp_profit
        )

        if restored_amount:
            remaining_tp_profit -= restored_amount

        if biggest_loser['trade_amount'] <= REBUILD_TRIGGER_TRADE_AMOUNT:

            self.rebuild_small_loser_position(biggest_loser)


    def remove_position_from_memory(self, pos):
        symbol = pos.get('symbol')
        side = pos.get('side')
        root_id = pos.get('root_id')
        was_dca = bool(pos.get('is_dca_position'))
        key = self.get_position_key(symbol, side) if symbol and side else None
        was_source = bool(key and self.rescue_chains.get(key, {}).get('source_position_id') == pos.get('position_id'))

        if pos in self.positions:
            self.positions.remove(pos)

        if was_source and any(p['symbol'] == symbol and p['side'] == side for p in self.positions):
            self._promote_rescue_chain(key)
        elif was_dca and root_id is not None:
            self.refresh_root_dca_count(root_id)

        self.refresh_bot_paused_state()

        if symbol and not any(p['symbol'] == symbol for p in self.positions):
            self.current_max_positions = self.max_positions
            self.active_dca_symbol = None
            if key:
                self.rescue_chains.pop(key, None)

    @synchronized_trading
    def close_tiny_position_if_needed(self, pos, current_price):
        # Dọn các mảnh vị thế còn quá nhỏ sau nhiều lần cắt lỗ một phần.
        # Mục tiêu: tránh kiểu OKX còn hiện khối lượng bé xíu, ký quỹ gần 0,
        # còn bot thì vẫn quản lý và báo cáo như một lệnh bình thường.
        if pos not in self.positions or not current_price:
            return False

        try:
            market = exchange.market(pos['symbol'])
            contract_size = float(market.get("contractSize") or 1)
            position_value = abs(pos.get('amount_coin', 0)) * current_price * contract_size

            if (
                pos.get('trade_amount', 0) > TINY_POSITION_TRADE_AMOUNT
                and position_value > TINY_POSITION_VALUE_USDT
            ):
                return False

            symbol = pos['symbol']
            close_side = 'sell' if pos['side'] == 'buy' else 'buy'
            close_amount = exchange.amount_to_precision(
                symbol,
                pos.get('amount_coin', 0)
            )
            close_amount = float(close_amount)

            if close_amount > 0:
                try:
                    order = exchange.create_market_order(
                        symbol,
                        close_side,
                        close_amount,
                        params={
                            "tdMode": MARGIN_MODE,
                            "reduceOnly": True,
                            "posSide": "long" if pos['side'] == "buy" else "short"
                        }
                    )
                    fill = self.resolve_order_fill(
                        order, symbol, current_price, close_amount
                    )
                    filled_amount = self._safe_float(fill.get('amount'))
                    if filled_amount + OKX_SYNC_ABS_TOLERANCE < close_amount:
                        ratio = max(
                            0.0,
                            min(1.0, filled_amount / close_amount)
                        )
                        pos['amount_coin'] *= (1.0 - ratio)
                        pos['trade_amount'] *= (1.0 - ratio)
                        pos['entry_fee'] *= (1.0 - ratio)
                        send_telegram(
                            f"⚠️ `{symbol}` chỉ khớp một phần lệnh dọn nhỏ; "
                            "bot giữ lại phần chưa đóng để tiếp tục quản lý."
                        )
                        return False
                    send_telegram(
                        f"🧹 *DỌN LỆNH QUÁ NHỎ*\n"
                        f"📍 `{symbol}`\n"
                        f"💵 Ký quỹ ảo còn: `${pos.get('trade_amount', 0):.6f}`\n"
                        f"📦 Giá trị vị thế còn khoảng: `${position_value:.6f}`\n"
                        f"✅ Đã gửi lệnh đóng phần còn lại."
                    )
                except Exception as e:
                    send_telegram(
                        f"⚠️ *CHƯA ĐÓNG ĐƯỢC LỆNH QUÁ NHỎ*\n"
                        f"📍 `{symbol}`\n"
                        f"⚠️ Sàn từ chối lệnh đóng:\n`{e}`\n"
                        "✅ Bot vẫn giữ lệnh trong bộ nhớ và sẽ kiểm tra lại."
                    )
                    snapshot = self.fetch_okx_position_snapshot_for_symbols([symbol])
                    okx_pos = snapshot.get(self.get_position_key(symbol, pos['side'])) if snapshot else None
                    if snapshot is None:
                        remaining = None
                    else:
                        remaining = (
                            abs(self._safe_float(okx_pos.get('contracts')))
                            if okx_pos else 0.0
                        )
                    if remaining is None or remaining > OKX_SYNC_ABS_TOLERANCE:
                        return False

            self.remove_position_from_memory(pos)
            return True

        except Exception as e:
            print(f"Lỗi dọn lệnh quá nhỏ: {e}")
            return False


    @synchronized_trading
    def close_position(self, pos, price, reason):
        if pos not in self.positions:
            return 0.0
        symbol = pos['symbol']
        key = self.get_position_key(symbol, pos['side'])
        close_side = 'sell' if pos['side'] == 'buy' else 'buy'

        same_key_positions = [
            p for p in self.positions
            if p['symbol'] == symbol and p['side'] == pos['side']
        ]
        is_last_managed_position = len(same_key_positions) == 1 and same_key_positions[0] is pos

        # Luôn xem vị thế thật trên OKX trước khi đóng để tránh gửi amount rác.
        snapshot = self.fetch_okx_position_snapshot()
        if snapshot is None:
            send_telegram(
                f"⚠️ Không lấy được snapshot OKX của `{symbol}`.\n"
                "✅ Bot giữ nguyên lệnh trong bộ nhớ và sẽ thử TP lại sau."
            )
            return 0.0
        okx_pos = snapshot.get(key) if snapshot else None
        actual_amount = abs(self._safe_float(okx_pos.get('contracts'))) if okx_pos else 0.0
        okx_average_before_close = self._safe_float(
            okx_pos.get('entry_price') if okx_pos else None
        )
        if okx_average_before_close <= 0:
            okx_average_before_close = self._safe_float(pos.get('entry_price'))

        # OKX đã hết vị thế nhưng bot còn lệnh ảo: quên dữ liệu ảo, không khóa coin.
        if actual_amount <= OKX_SYNC_ABS_TOLERANCE:
            same_group = [
                p for p in self.positions
                if p['symbol'] == symbol and p['side'] == pos['side']
            ]
            self.forget_stale_group_when_okx_empty(
                key,
                same_group,
                "TP kiểm tra lại và thấy OKX không còn vị thế thật."
            )
            return 0.0

        # Lệnh cuối cùng đóng toàn bộ vị thế thật; lệnh khác chỉ đóng phần ảo của nó.
        raw_requested = actual_amount if is_last_managed_position else min(
            abs(self._safe_float(pos.get('amount_coin'))),
            actual_amount
        )
        requested_amount = self._safe_float(
            exchange.amount_to_precision(symbol, raw_requested)
        )
        min_amount = self.get_min_order_amount(symbol)

        # Khối lượng nhỏ hơn mức sàn cho phép: quên slot ảo thay vì thử đặt lệnh rồi lỗi liên tục.
        if requested_amount <= 0 or (min_amount > 0 and requested_amount < min_amount):
            self._remember_ignored_residual(pos, raw_requested)
            self.remove_position_from_memory(pos)
            send_telegram(
                f"🧹 *BỎ LỆNH ẢO QUÁ NHỎ, KHÔNG GỬI ORDER*\n"
                f"📍 `{symbol}` - `{pos['side'].upper()}`\n"
                f"📦 Khối lượng cần đóng: `{raw_requested}`\n"
                f"📏 Mức tối thiểu sàn: `{min_amount}`\n"
                f"✅ Slot ảo đã được giải phóng; phần dư được ghi nhớ để dọn cùng lệnh cuối."
            )
            return 0.0

        try:
            original_virtual_amount = max(
                self._safe_float(pos.get('amount_coin')),
                1e-12
            )
            clid = self.make_client_order_id('TP', pos.get('position_id'))
            order = exchange.create_market_order(symbol, close_side, requested_amount, params={
                'tdMode': MARGIN_MODE,
                'reduceOnly': True,
                'posSide': 'long' if pos['side'] == 'buy' else 'short',
                'clOrdId': clid,
            })
            fill = self.resolve_order_fill(order, symbol, price, requested_amount)
            fill_amount = min(
                self._safe_float(fill.get('amount'), requested_amount),
                requested_amount
            )
            fill_price = self._safe_float(fill.get('price'), price)

            # Phần lời/lỗ trả về chỉ tính trên phần thuộc lệnh ảo này.
            virtual_filled_amount = min(fill_amount, original_virtual_amount)
            raw_pnl = self.calculate_realized_pnl_from_fill(
                pos, fill_price, virtual_filled_amount
            )
            exit_fee = self._safe_float(fill.get('fee'))
            if exit_fee <= 0:
                market = exchange.market(symbol)
                contract_size = self._safe_float(market.get('contractSize'), 1.0)
                exit_fee = fill_amount * fill_price * contract_size * FEE_RATE

            virtual_ratio = min(1.0, virtual_filled_amount / original_virtual_amount)
            allocated_entry_fee = self._safe_float(pos.get('entry_fee')) * virtual_ratio
            real_net_profit = raw_pnl - allocated_entry_fee - exit_fee

            # PnL mà vị thế gộp OKX thực sự ghi nhận cho đúng phần vừa đóng.
            # Giá vào ảo của RESCUE có thể báo lời trong khi giá gộp OKX báo lỗ.
            market = exchange.market(symbol)
            contract_size = self._safe_float(market.get('contractSize'), 1.0)
            if pos['side'] == 'buy':
                okx_raw_pnl = (
                    fill_price - okx_average_before_close
                ) * virtual_filled_amount * contract_size
            else:
                okx_raw_pnl = (
                    okx_average_before_close - fill_price
                ) * virtual_filled_amount * contract_size
            okx_net_pnl = okx_raw_pnl - exit_fee
            self.last_close_accounting[pos.get('position_id')] = {
                'virtual_net': real_net_profit,
                'okx_raw_pnl': okx_raw_pnl,
                'okx_net_pnl': okx_net_pnl,
                'okx_average': okx_average_before_close,
                'fill_price': fill_price,
                'fill_amount': virtual_filled_amount,
            }

            self.add_fill_event(pos, 'CLOSE', fill, -self._safe_float(pos.get('trade_amount')), reason)
            self.balance += raw_pnl - exit_fee
            self.coins[symbol]['last_close_time'] = time.time()

            # Nếu đây là lệnh cuối cùng, kiểm tra và đóng nốt toàn bộ phần thật còn lại.
            extra_closed = 0.0
            leftover_actual = 0.0
            if is_last_managed_position:
                for _ in range(2):
                    snapshot = self.fetch_okx_position_snapshot()
                    if snapshot is None:
                        leftover_actual = max(
                            leftover_actual,
                            max(0.0, requested_amount - fill_amount - extra_closed),
                            self._safe_float(self.ignored_residuals.get(key))
                        )
                        send_telegram(
                            f"⚠️ Không xác minh được phần dư `{symbol}` do lỗi snapshot OKX. "
                            "Bot giữ lại dấu vết residual để kiểm tra vòng sau."
                        )
                        break
                    okx_pos = snapshot.get(key) if snapshot else None
                    remaining_raw = abs(self._safe_float(okx_pos.get('contracts'))) if okx_pos else 0.0
                    remaining_actual = self._safe_float(exchange.amount_to_precision(symbol, remaining_raw))
                    if remaining_actual <= 0:
                        leftover_actual = 0.0
                        break
                    if min_amount > 0 and remaining_actual < min_amount:
                        leftover_actual = remaining_raw
                        break
                    try:
                        extra_order = exchange.create_market_order(
                            symbol,
                            close_side,
                            remaining_actual,
                            params={
                                'tdMode': MARGIN_MODE,
                                'reduceOnly': True,
                                'posSide': 'long' if pos['side'] == 'buy' else 'short',
                                'clOrdId': self.make_client_order_id('TPCLEAN', pos.get('position_id')),
                            }
                        )
                        extra_fill = self.resolve_order_fill(extra_order, symbol, fill_price, remaining_actual)
                        extra_closed += self._safe_float(extra_fill.get('amount'), remaining_actual)
                    except Exception as cleanup_error:
                        leftover_actual = remaining_raw
                        send_telegram(
                            f"⚠️ *KHÔNG DỌN HẾT PHẦN DƯ CUỐI COIN*\n"
                            f"📍 `{symbol}`\n"
                            f"📦 Phần còn lại: `{remaining_raw}`\n"
                            f"⚠️ `{cleanup_error}`"
                        )
                        break

                self.remove_position_from_memory(pos)
                if leftover_actual > 0:
                    self.ignored_residuals[key] = leftover_actual
                else:
                    self._clear_ignored_residual(key)
                self.rescue_chains.pop(key, None)
            else:
                # TP khớp một phần hay đủ đều quên hẳn lệnh ảo để giải phóng slot.
                # Phần chưa khớp vẫn nằm trên OKX và được ghi vào ignored_residuals.
                remaining_virtual = max(
                    0.0, original_virtual_amount - virtual_filled_amount
                )
                if remaining_virtual > 0:
                    self._remember_ignored_residual(pos, remaining_virtual)
                self.remove_position_from_memory(pos)

            send_telegram(
                f"✅ *ĐÓNG LỆNH {symbol} THEO FILL OKX*\n"
                f"📝 {reason}\n"
                f"💰 Giá đóng khớp: `{fill_price}`\n"
                f"📦 Khối lượng khớp lần đầu: `{fill_amount}/{requested_amount}`\n"
                f"🧹 Khối lượng đóng bổ sung cuối coin: `{extra_closed}`\n"
                f"💸 Phí đóng OKX/ước tính: `${exit_fee:.6f}`\n"
                f"💰 Lời/lỗ ròng lệnh ảo: `${real_net_profit:.4f}`\n"
                f"✅ Slot lệnh ảo đã được giải phóng"
            )
            return real_net_profit

        except Exception as e:
            # Một số lần OKX đã nhận reduce-only và đóng sạch vị thế nhưng phản hồi API/CCXT
            # vẫn ném lỗi precision. Luôn xác minh trạng thái thật trước khi kết luận thất bại.
            try:
                time.sleep(0.8)
                verify_snapshot = self.fetch_okx_position_snapshot_for_symbols([symbol])
                if verify_snapshot is None:
                    verify_amount = None
                else:
                    verify_pos = verify_snapshot.get(key)
                    verify_amount = (
                        abs(self._safe_float(verify_pos.get('contracts')))
                        if verify_pos else 0.0
                    )
            except Exception:
                verify_amount = None

            if verify_amount is not None and verify_amount <= OKX_SYNC_ABS_TOLERANCE:
                if pos in self.positions:
                    self.remove_position_from_memory(pos)
                self._clear_ignored_residual(key)
                self.rescue_chains.pop(key, None)
                if symbol in self.coins:
                    self.coins[symbol]['last_close_time'] = time.time()

                send_telegram(
                    f"✅ *OKX ĐÃ ĐÓNG SẠCH VỊ THẾ*\n"
                    f"📍 `{symbol}` - `{pos['side'].upper()}`\n"
                    f"ℹ️ API có trả về cảnh báo: `{e}`\n"
                    f"✅ Nhưng kiểm tra lại OKX cho thấy khối lượng còn `0`.\n"
                    f"🧹 Bot đã xóa lệnh khỏi bộ nhớ và coi TP thành công."
                )
                return 0.0

            send_telegram(
                f"❌ Lỗi đóng lệnh thật {symbol}:\n`{e}`\n"
                f"📦 Khối lượng còn trên OKX: `{verify_amount if verify_amount is not None else 'không kiểm tra được'}`"
            )
            return 0


    def send_multi_report(self):

        if self.positions:

            symbols = [
                pos['symbol']
                for pos in self.positions
            ]

            status_text = ", ".join(symbols)

        else:

            status_text = "Đang săn tín hiệu đảo chiều..."

        if self.view_mode:
            search_status = "👁 CHẾ ĐỘ XEM: không thực hiện bất kỳ giao dịch tự động nào"
        else:
            search_status = "⏸ Đang dừng săn lệnh mới" if self.search_paused else "▶️ Đang săn lệnh mới"

        msg = (
            f"📊 *GIÁM SÁT HỆ THỐNG*\n"
            f"{search_status}\n"
            f"🏦 Vốn ghi sổ: `${self.balance:,.2f}$`\n"
            f"💳 Vốn khả dụng: `${self.get_available_balance():,.2f}$`\n"
            f"📦 Slot lệnh gốc: `{self.count_root_positions()}/{self.max_positions}`\n"
            f"💵 Vốn mỗi lệnh mới: `${self.default_trade_amount:g}`\n"
            f"🏦 Quỹ cắt lỗ: `${self.loss_bank:.4f}`\n"
            f"✅ Khóa coin do lệch OKX: `ĐÃ TẮT`\n"
            f"🚫 Coin trong danh sách đen: `{len(self.blacklist)}`\n"
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
