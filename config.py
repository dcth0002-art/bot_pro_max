# -*- coding: utf-8 -*-

# Cấu hình tài khoản và sàn giao dịch
EXCHANGE_NAME = 'okx' 
IS_DEMO_ACCOUNT = True 

# Cấu hình giao dịch
STARTING_BALANCE = 500.0 
TARGET_BALANCE = 1000.0 
TIMEFRAME = '1m' 

# Cấu hình cho mô hình học tăng cường (Reinforcement Learning)
STATE_SIZE = 10 
ACTION_SPACE = 3 
LEARNING_RATE = 0.0005 # Giảm tốc độ học xuống một chút để ổn định hơn
DISCOUNT_FACTOR = 0.99 # Tăng tầm nhìn xa hơn
EPSILON_START = 1.0 
EPSILON_END = 0.01 
# Tăng lên 200,000 bước để epsilon giảm chậm hơn (khoảng 100 tập)
EPSILON_DECAY_STEPS = 200000 

# Cấu hình cho Replay Buffer
BUFFER_SIZE = 50000 
BATCH_SIZE = 128 # Tăng batch size để học sâu hơn mỗi lần replay

# Cấu hình lưu trữ
MODEL_SAVE_PATH = 'saved_models/trading_bot_model.h5'
