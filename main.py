# -*- coding: utf-8 -*-
import sys
import os
import time

# Ép Python in log ngay lập tức
sys.stdout.reconfigure(line_buffering=True)

# Ẩn cảnh báo của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Thêm thư mục hiện tại vào PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("--- [HỆ THỐNG] Đang nạp thư viện... ---")
import numpy as np
import pandas as pd

try:
    from environment import TradingEnvironment
    from agent import Agent
    from config import (MODEL_SAVE_PATH, STATE_SIZE, 
                        TARGET_BALANCE, STARTING_BALANCE)
    from firebase_storage import initialize_firebase, upload_model_to_firebase, download_model_from_firebase
    from telegram_reporter import initialize_telegram_bot, send_telegram_message
    
    import download_data
    
    print("--- [HỆ THỐNG] Nạp thư viện THÀNH CÔNG ---")
except Exception as e:
    print(f"--- [LỖI] Không thể nạp thư viện: {e} ---")
    sys.exit(1)

# Cấu hình đường dẫn lưu trữ
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
MEMORY_SAVE_PATH = MODEL_SAVE_PATH.replace('.weights.h5', '_memory.pkl')

FIREBASE_MODEL_NAME = "trading_bot_model.weights.h5" 
FIREBASE_MEMORY_NAME = "trading_bot_memory.pkl"
DATA_FILE_PATH = "btc_usdt_1m_data.csv"

def fetch_data():
    try:
        if not os.path.exists(DATA_FILE_PATH):
            print("--- [DỮ LIỆU] Không tìm thấy file, bắt đầu tải 2000 nến... ---")
            send_telegram_message("📥 Đang tải dữ liệu thị trường mới...")
            download_data.download_historical_data()
            
        if not os.path.exists(DATA_FILE_PATH):
            return None
            
        df = pd.read_csv(DATA_FILE_PATH)
        return df.head(2000).copy()
    except Exception as e:
        print(f"--- [LỖI] fetch_data: {e} ---")
        return None

def train_bot(episodes):
    print("--- [KHỞI ĐỘNG] Kết nối Firebase & Telegram... ---")
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()
    
    send_telegram_message("🤖 Bot đang khởi động và khôi phục ký ức...")

    data = fetch_data()
    if data is None:
        send_telegram_message("❌ Lỗi: Không thể lấy dữ liệu.")
        return

    agent = Agent(state_size=STATE_SIZE * 5)
    env = TradingEnvironment(data)

    # Đảm bảo Epsilon bắt đầu từ 1.0 nếu là lần đầu tiên
    agent.epsilon = 1.0

    if firebase_initialized:
        print("--- [FIREBASE] Đang khôi phục Model & Ký ức cũ... ---")
        # Tải Trí Não (Model)
        if download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH):
             agent.load(MODEL_SAVE_PATH)
             # Sau khi tải model, chúng ta sẽ cho bot khám phá từ từ
             agent.epsilon = 0.5 

        # Tải Ký Ức (Memory)
        if download_model_from_firebase(FIREBASE_MEMORY_NAME, MEMORY_SAVE_PATH):
             agent.load_memory(MEMORY_SAVE_PATH)

    send_telegram_message(f"✅ Đã khôi phục {len(agent.memory)} ký ức. Bắt đầu phiên huấn luyện!")

    for e in range(episodes):
        try:
            state = env.reset()
            if state is None: break
            
            state = np.reshape(state, [1, env.state_size])
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action, leverage=5)
                if next_state is not None:
                    next_state = np.reshape(next_state, [1, env.state_size])
                    agent.remember(state, action, reward, next_state, done)
                state = next_state

            agent.replay()
            
            if (e + 1) % 5 == 0:
                agent.update_target_model()

            # Báo cáo và đồng bộ mỗi 20 tập
            if (e + 1) % 20 == 0:
                # 1. Lưu trọng số Model
                agent.save(MODEL_SAVE_PATH)
                # 2. Lưu Ký ức thực tế
                agent.save_memory(MEMORY_SAVE_PATH)
                
                status = "THẮNG 🏆" if env.balance >= 600 else ("CHÁY 🔥" if env.balance <= 450 else "HẾT ⌛")
                summary = (f"📊 Tập {e+1}: {status}\n- Số dư: ${env.balance:.2f}\n- Epsilon: {agent.epsilon:.4f}\n- Ký ức: {len(agent.memory)}")
                send_telegram_message(summary)
                
                if firebase_initialized:
                    # Đồng bộ cả 2 file lên Server Firebase
                    upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                    upload_model_to_firebase(MEMORY_SAVE_PATH, FIREBASE_MEMORY_NAME)
                    
            if (e + 1) % 5 == 0:
                print(f"Tiến độ: Tập {e+1}/{episodes} - Số dư: ${env.balance:.2f} - Memory: {len(agent.memory)}")

        except Exception as ex:
            print(f"--- [LỖI] {ex} ---")
            time.sleep(1)

if __name__ == "__main__":
    train_bot(5000)
