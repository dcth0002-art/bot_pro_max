# -*- coding: utf-8 -*-

# Cấu hình tài khoản và sàn giao dịch
EXCHANGE_NAME = 'okx' 
IS_DEMO_ACCOUNT = True 

# Cấu hình giao dịch
STARTING_BALANCE = 500.0 
TARGET_BALANCE = 600.0 
TIMEFRAME = '1m' 
LEVERAGE = 3 # Giảm đòn bẩy xuống 3 để an toàn hơn

# Cấu hình cho mô hình AI
STATE_SIZE = 20 # Nhìn lại 20 nến thay vì 10 để thấy xu hướng rõ hơn
ACTION_SPACE = 3 
LEARNING_RATE = 0.0003 # Tốc độ học chậm lại một chút để chắc chắn hơn
DISCOUNT_FACTOR = 0.99 
EPSILON_START = 1.0 
EPSILON_END = 0.05 # Giữ 5% ngẫu nhiên để bot luôn nhạy bén
EPSILON_DECAY_STEPS = 300000 

# Cấu hình cho Replay Buffer - Tăng ký ức lên 200,000
BUFFER_SIZE = 200000 
BATCH_SIZE = 128 

MODEL_SAVE_PATH = 'saved_models/trading_bot_model.weights.h5'
