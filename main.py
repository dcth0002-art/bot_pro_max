# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
import os
from environment import TradingEnvironment
from agent import Agent
from config import (MODEL_SAVE_PATH, STATE_SIZE, 
                    TARGET_BALANCE, STARTING_BALANCE, TIMEFRAME)
from firebase_storage import initialize_firebase, upload_model_to_firebase, download_model_from_firebase
from telegram_reporter import initialize_telegram_bot, send_telegram_message
from download_data import download_historical_data # Import hàm tải dữ liệu

FIREBASE_MODEL_NAME = "trading_bot_model.h5" 
DATA_FILE_PATH = "btc_usdt_1m_data.csv"

def fetch_data():
    """Đảm bảo có dữ liệu trước khi chạy."""
    try:
        if not os.path.exists(DATA_FILE_PATH):
            print(f"--- Không tìm thấy {DATA_FILE_PATH}, đang tải mới... ---")
            download_historical_data()
            
        if not os.path.exists(DATA_FILE_PATH):
            return None
            
        df = pd.read_csv(DATA_FILE_PATH)
        if len(df) < 100:
            print("Dữ liệu quá ít, đang tải lại...")
            download_historical_data()
            df = pd.read_csv(DATA_FILE_PATH)
            
        return df.head(2000).copy()
    except Exception as e:
        print(f"Lỗi fetch_data: {e}")
        return None

def train_bot(episodes):
    # Khởi tạo các dịch vụ
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()
    
    # Gửi tin nhắn khởi động
    send_telegram_message("🚀 Bot đang kiểm tra dữ liệu và khởi động...")

    data = fetch_data()
    if data is None:
        send_telegram_message("❌ Lỗi: Không thể chuẩn bị dữ liệu. Bot dừng.")
        return # Thoát ra đây sẽ khiến Railway restart nếu không cẩn thận

    env = TradingEnvironment(data)
    agent = Agent(state_size=STATE_SIZE * 5)

    # Tải trí nhớ từ Firebase
    if firebase_initialized:
        # Xóa file cũ ở local để đảm bảo tải bản mới nhất từ Cloud
        if os.path.exists(MODEL_SAVE_PATH):
            os.remove(MODEL_SAVE_PATH)
            
        if download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH):
             agent.load(MODEL_SAVE_PATH)
             agent.epsilon = 1.0 
             print("--- Đã đồng bộ trí nhớ thành công ---")
        else:
            print("--- Chạy với trí nhớ mới (Trống) ---")

    send_telegram_message("✅ Bot đã sẵn sàng và bắt đầu huấn luyện!")

    for e in range(episodes):
        try:
            state = env.reset()
            state = np.reshape(state, [1, env.state_size])
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action, leverage=5)
                
                if next_state is not None:
                    next_state = np.reshape(next_state, [1, env.state_size])
                    agent.remember(state, action, reward, next_state, done)
                state = next_state

            # Cuối mỗi tập
            agent.replay()
            
            if (e + 1) % 5 == 0:
                agent.update_target_model()

            # Lưu định kỳ
            if (e + 1) % 50 == 0:
                agent.save(MODEL_SAVE_PATH)
                status = "THẮNG" if env.balance >= 600 else ("CHÁY" if env.balance <= 450 else "HẾT NẾN")
                summary = (f"📊 Tập {e+1}: {status}\n- Số dư: ${env.balance:.2f}\n- Epsilon: {agent.epsilon:.4f}")
                send_telegram_message(summary)
                if firebase_initialized:
                    upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                    
        except Exception as ex:
            print(f"Lỗi trong tập {e+1}: {ex}")
            time.sleep(5)

if __name__ == "__main__":
    train_bot(1000)
