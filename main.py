# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from environment import TradingEnvironment
from agent import Agent
from config import (MODEL_SAVE_PATH, STATE_SIZE, 
                    TARGET_BALANCE, STARTING_BALANCE, TIMEFRAME)
import os
from firebase_storage import initialize_firebase, upload_model_to_firebase, download_model_from_firebase
from telegram_reporter import initialize_telegram_bot, send_telegram_message

FIREBASE_MODEL_NAME = "trading_bot_model.h5" 
DATA_FILE_PATH = "btc_usdt_1m_data.csv"

def fetch_data():
    try:
        print(f"Đang đọc dữ liệu từ file {DATA_FILE_PATH}...")
        if not os.path.exists(DATA_FILE_PATH):
            return None
            
        df = pd.read_csv(DATA_FILE_PATH)
        # Sử dụng 200 nến để huấn luyện (tăng lên để bot có dữ liệu học)
        df_short = df.head(200).copy()
        
        print("--- Đọc dữ liệu thành công! ---")
        return df_short
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

def train_bot(episodes):
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()

    send_telegram_message("🤖 Bot Trading Pro Max đã khởi động (Chế độ lưu 50 tập/lần)!")

    data = fetch_data()
    if data is None:
        send_telegram_message(f"❌ Lỗi: Không thể đọc file dữ liệu.")
        return

    env = TradingEnvironment(data)
    agent = Agent(state_size=STATE_SIZE * 5)

    if firebase_initialized:
        model_downloaded = download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH)
        if model_downloaded:
             agent.load(MODEL_SAVE_PATH)
             agent.epsilon = agent.epsilon_min 
             msg = "Đã tải trí nhớ từ Firebase."
        else:
            msg = "Không tìm thấy trí nhớ trên Firebase, học từ đầu."
    else:
        msg = "Chạy chế độ Local."
            
    print(msg)
    send_telegram_message(f"▶️ {msg}")

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        
        if env.balance <= 0:
            env.balance = STARTING_BALANCE

        done = False
        total_profit = 0
        step_counter = 0
        
        while not done:
            action = agent.choose_action(state)
            leverage = np.random.randint(1, 11)
            next_state, reward, done = env.step(action, leverage)
            
            if next_state is not None:
                next_state = np.reshape(next_state, [1, env.state_size])
                agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_profit += reward
            step_counter += 1

            if step_counter % 50 == 0:
                print(f"Tập {e+1}: Bước {step_counter}, Số dư: ${env.balance:.2f}")

        # --- KẾT THÚC TẬP ---
        
        # Vẫn cho bot học sau mỗi tập để tích lũy kiến thức liên tục
        agent.replay() 
        
        if (e + 1) % 5 == 0:
            agent.update_target_model()

        # CHỈ LƯU TRÍ NHỚ VÀ GỬI TIN NHẮN TELEGRAM MỖI 50 TẬP
        if (e + 1) % 50 == 0:
            # Lưu xuống ổ đĩa ảo của server
            agent.save(MODEL_SAVE_PATH)
            
            summary_message = (f"📊 Báo cáo định kỳ (Tập {e+1}/{episodes}):\n"
                               f"   - Lợi nhuận tập này: ${total_profit:.4f}\n"
                               f"   - Số dư hiện tại: ${env.balance:.2f}\n"
                               f"   - Epsilon: {agent.epsilon:.4f}")
            send_telegram_message(summary_message)

            # Tải lên Firebase
            if firebase_initialized:
                upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                send_telegram_message(f"💾 Đã sao lưu trí nhớ Tập {e+1} lên Firebase.")
            
            print(f"--- Đã lưu và báo cáo tại tập {e+1} ---")

        if env.balance >= TARGET_BALANCE:
            send_telegram_message("🏆 ĐÃ ĐẠT MỤC TIÊU! Dừng bot.")
            break

if __name__ == "__main__":
    NUM_EPISODES = 1000
    train_bot(NUM_EPISODES)
