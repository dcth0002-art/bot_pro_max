# -*- coding: utf-8 -*-
import sys
import os
import time

# Ép Python in log ngay lập tức lên Railway
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

FIREBASE_MODEL_NAME = "trading_bot_model.h5" 
DATA_FILE_PATH = "btc_usdt_1m_data.csv"

def fetch_data():
    try:
        if not os.path.exists(DATA_FILE_PATH):
            print("--- [DỮ LIỆU] Không tìm thấy file, bắt đầu tải 2000 nến... ---")
            send_telegram_message("📥 Đang tải dữ liệu thị trường mới...")
            download_data.download_historical_data()
            
        if not os.path.exists(DATA_FILE_PATH):
            print("--- [LỖI] Vẫn không tìm thấy file dữ liệu sau khi tải! ---")
            return None
            
        df = pd.read_csv(DATA_FILE_PATH)
        print(f"--- [DỮ LIỆU] Đã nạp {len(df)} dòng dữ liệu ---")
        return df.head(2000).copy()
    except Exception as e:
        print(f"--- [LỖI] fetch_data: {e} ---")
        return None

def train_bot(episodes):
    print("--- [KHỞI ĐỘNG] Kết nối Firebase & Telegram... ---")
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()
    
    send_telegram_message("🤖 Bot đang khởi động và nạp dữ liệu...")

    data = fetch_data()
    if data is None:
        send_telegram_message("❌ Lỗi: Không thể lấy dữ liệu để chạy.")
        print("--- [DỪNG] Kết thúc do thiếu dữ liệu ---")
        return

    print("--- [HỆ THỐNG] Khởi tạo Agent và Môi trường... ---")
    agent = Agent(state_size=STATE_SIZE * 5)
    env = TradingEnvironment(data)

    if firebase_initialized:
        print("--- [FIREBASE] Đang kiểm tra model cũ... ---")
        if download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH):
             agent.load(MODEL_SAVE_PATH)
             agent.epsilon = 0.5 # Bắt đầu khám phá từ 50% nếu đã có model cũ

    send_telegram_message("✅ Bot bắt đầu phiên giao dịch huấn luyện!")
    print("--- [CHẠY] Bắt đầu vòng lặp huấn luyện ---")

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

            if (e + 1) % 20 == 0: # Giảm xuống 20 tập để báo cáo nhanh hơn
                agent.save(MODEL_SAVE_PATH)
                status = "THẮNG" if env.balance >= 600 else ("CHÁY" if env.balance <= 450 else "HẾT")
                summary = (f"📊 Tập {e+1}: {status}\n- Số dư: ${env.balance:.2f}\n- Epsilon: {agent.epsilon:.4f}")
                send_telegram_message(summary)
                if firebase_initialized:
                    upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                    
            if (e + 1) % 5 == 0:
                print(f"Tiến độ: Tập {e+1}/{episodes} - Số dư: ${env.balance:.2f}")

        except Exception as ex:
            print(f"--- [LỖI TRONG VÒNG LẶP] {ex} ---")
            time.sleep(2)

if __name__ == "__main__":
    train_bot(2000)
