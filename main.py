# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import ccxt
from environment import TradingEnvironment
from agent import Agent
from config import (MODEL_SAVE_PATH, STATE_SIZE, 
                    TARGET_BALANCE, STARTING_BALANCE, TIMEFRAME)
import os
from firebase_storage import initialize_firebase, upload_model_to_firebase, download_model_from_firebase
# Import các hàm từ telegram_reporter
from telegram_reporter import initialize_telegram_bot, send_telegram_message

FIREBASE_MODEL_NAME = "trading_bot_model.h5" 

def fetch_data(symbol, timeframe, limit=1000):
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None

def train_bot(episodes):
    # Khởi tạo các dịch vụ
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()

    # Gửi tin nhắn khởi động
    send_telegram_message("🤖 Bot Trading Pro Max đã khởi động trên Railway!")

    # 1. Tải dữ liệu
    symbol = 'BTC/USDT'
    data = fetch_data(symbol, TIMEFRAME)
    if data is None:
        send_telegram_message(" Lỗi nghiêm trọng: Không thể tải dữ liệu giá. Bot sẽ dừng lại.")
        return

    # 2. Khởi tạo Agent
    env = TradingEnvironment(data)
    agent = Agent(state_size=STATE_SIZE * 5)

    # Tải model
    initial_message = ""
    if firebase_initialized:
        model_downloaded = download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH)
        if model_downloaded:
             agent.load(MODEL_SAVE_PATH)
             agent.epsilon = agent.epsilon_min
             initial_message = "Đã tải trí nhớ từ Firebase, bắt đầu học tiếp."
        else:
            initial_message = "Không tìm thấy trí nhớ, bắt đầu học từ đầu."
    else: # Xử lý local
        if os.path.exists(MODEL_SAVE_PATH):
            agent.load(MODEL_SAVE_PATH)
            agent.epsilon = agent.epsilon_min
            initial_message = "Đã tải trí nhớ từ file local, bắt đầu học tiếp."
        else:
            initial_message = "Không tìm thấy trí nhớ, bắt đầu học từ đầu."
            
    print(initial_message)
    send_telegram_message(f" Bắt đầu phiên làm việc mới.\n{initial_message}")

    # 3. Vòng lặp huấn luyện
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

            if step_counter % 100 == 0:
                report_message = (f"📈 Báo cáo nhanh - Tập {e+1}:\n"
                                  f"   - Đã xử lý {step_counter} bước.\n"
                                  f"   - Số dư tạm thời: ${env.balance:.2f}")
                print(report_message)
                send_telegram_message(report_message)

            if done:
                summary_message = (f"✅ Hoàn thành Tập {e+1}/{episodes}:\n"
                                   f"   - Lợi nhuận: ${total_profit:.4f}\n"
                                   f"   - Số dư cuối: ${env.balance:.2f}\n"
                                   f"   - Epsilon: {agent.epsilon:.4f}")
                print(summary_message)
                send_telegram_message(summary_message)
                
                if env.balance >= TARGET_BALANCE:
                    final_message = "🏆 ĐÃ ĐẠT MỤC TIÊU! Bot sẽ dừng lại."
                    print(final_message)
                    send_telegram_message(final_message)
                    agent.save(MODEL_SAVE_PATH)
                    if firebase_initialized:
                         upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)
                    return

            agent.replay()
            
        if e % 10 == 0:
            agent.update_target_model()

        # Lưu model và tải lên Firebase
        agent.save(MODEL_SAVE_PATH)
        save_message = f"💾 Đã lưu trí nhớ sau Tập {e+1}."
        print(save_message)
        send_telegram_message(save_message)
        if firebase_initialized:
            upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)


if __name__ == "__main__":
    NUM_EPISODES = 1000
    train_bot(NUM_EPISODES)
