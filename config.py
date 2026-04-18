# -*- coding: utf-8 -*-

# Cấu hình tài khoản và sàn giao dịch
EXCHANGE_NAME = 'binance'  # Hoặc 'bybit', 'okx', ...
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
IS_DEMO_ACCOUNT = True  # True để dùng tài khoản demo, False cho tài khoản thật

# Cấu hình giao dịch
STARTING_BALANCE = 500.0  # Vốn khởi điểm
TARGET_BALANCE = 1000.0  # Mục tiêu
TIMEFRAME = '1m'  # Khung thời gian 1 phút
LEVERAGE_MIN = 1
LEVERAGE_MAX = 125 # Tự do quyết định đòn bẩy trong khoảng này

# Cấu hình cho mô hình học tăng cường (Reinforcement Learning)
STATE_SIZE = 10  # Kích thước của vector trạng thái (ví dụ: 10 nến gần nhất)
ACTION_SPACE = 3  # Không gian hành động: 0: HOLD, 1: BUY, 2: SELL
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95  # Gamma - Yếu tố chiết khấu cho các phần thưởng trong tương lai
EPSILON_START = 1.0  # Tỷ lệ khám phá (exploration) ban đầu
EPSILON_END = 0.01  # Tỷ lệ khám phá cuối cùng
EPSILON_DECAY_STEPS = 10000  # Số bước để giảm epsilon từ START xuống END

# Cấu hình cho Replay Buffer
BUFFER_SIZE = 100000  # Kích thước bộ nhớ kinh nghiệm
BATCH_SIZE = 64      # Số lượng kinh nghiệm lấy ra để học mỗi lần

# Cấu hình lưu trữ
MODEL_SAVE_PATH = 'saved_models/trading_bot_model.h5'
LOG_FILE_PATH = 'logs/trading_log.txt'
