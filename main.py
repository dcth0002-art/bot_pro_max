# -*- coding: utf-8 -*-
print("--- [HỆ THỐNG] Đang nạp thư viện... ---")
import numpy as np
import pandas as pd
import time
import os
import sys

# Đảm bảo in log ra ngay lập tức
sys.stdout.reconfigure(line_buffering=True)

try:
    from environment import TradingEnvironment
    from agent import Agent
    from config import (MODEL_SAVE_PATH, STATE_SIZE, 
                        TARGET_BALANCE, STARTING_BALANCE, TIMEFRAME)
    from firebase_storage import initialize_firebase, upload_model_to_firebase, download_model_from_firebase
    from telegram_reporter import initialize_telegram_bot, send_telegram_message
    from download_data import download_historical_data
    print("--- [HỆ THỐNG] Nạp thư viện THÀNH CÔNG ---")
except Exception as e:
    print(f"--- [LỖI] Không thể nạp thư viện: {e} ---")
    sys.exit(1)

FIREBASE_MODEL_NAME = "trading_bot_model.h5" 
DATA_FILE_PATH = "btc_usdt_1m_data.csv"

def fetch_data():
    try:
        if not os.path.exists(DATA_FILE_PATH):
            print("--- [DỮ LIỆU] Thiếu file CSV, đang tải 2000 nến mới... ---")
            download_historical_data()
            
        if not os.path.exists(DATA_FILE_PATH):
            return None
            
        df = pd.read_csv(DATA_FILE_PATH)
        return df.head(2000).copy()
    except Exception as e:
        print(f"--- [LỖI] fetch_data: {e} ---")
        return None

def train_bot(episodes):
    print("--- [KHỞI ĐỘNG] Đang kết nối Firebase & Telegram... ---")
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()
    
    send_telegram_message("🤖 Bot đang kiểm tra dữ liệu và khởi động...")

    data = fetch_data()
    if data is None:
        print("--- [LỖI] Không có dữ liệu để chạy ---")
        send_telegram_message("❌ Lỗi: Không thể chuẩn bị dữ liệu.")
        return

    agent = Agent(state_size=STATE_SIZE * 5)
    env = TradingEnvironment(data)

    if firebase_initialized:
        print("--- [HỆ THỐNG] Đang đồng bộ trí nhớ từ Firebase... ---")
        if download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH):
             agent.load(MODEL_SAVE_PATH)
             agent.epsilon = 1.0 
        else:
            print("--- [HỆ THỐNG] Không có trí nhớ cũ, học từ đầu ---")

    send_telegram_message("✅ Bot đã sẵn sàng! Chúc may mắn.")
    print("--- [HỆ THỐNG] Bắt đầu vòng lặp huấn luyện ---")

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

            # Học tập sau mỗi tập
            agent.replay()
            
            if (e + 1) % 5 == 0:
                agent.update_target_model()

            # Báo cáo mỗi 50 tập
            if (e + 1) % 50 == 0:
                agent.save(MODEL_SAVE_PATH)
                status = "THẮNG" if env.balance >= 600 else ("CHÁY" if env.balance <= 450 else "HẾT DỮ LIỆU")
                summary = (f"📊 Tập {e+1}: {status}\n- Số dư: ${env.balance:.2f}\n- Epsilon: {agent.epsilon:.4f}")
                send_telegram_message(summary)
                if firebase_initialized:
                    upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                    
            if (e + 1) % 10 == 0:
                print(f"Tiến độ: Tập {e+1}/{episodes} - Số dư: ${env.balance:.2f} - Epsilon: {agent.epsilon:.4f}")

        except Exception as ex:
            print(f"--- [LỖI TRONG TẬP {e+1}] {ex} ---")
            time.sleep(5)

if __name__ == "__main__":
    train_bot(1000)
