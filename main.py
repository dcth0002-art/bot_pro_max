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
        if not os.path.exists(DATA_FILE_PATH):
            return None
        df = pd.read_csv(DATA_FILE_PATH)
        df_short = df.head(2000).copy() 
        return df_short
    except Exception as e:
        print(f"Lỗi đọc dữ liệu: {e}")
        return None

def train_bot(episodes):
    firebase_initialized = initialize_firebase()
    initialize_telegram_bot()
    send_telegram_message("🚀 Khởi động Bot (100% Khám phá - Mục tiêu: 600$, Cháy: 450$)")

    data = fetch_data()
    if data is None: return

    env = TradingEnvironment(data)
    agent = Agent(state_size=STATE_SIZE * 5)

    if firebase_initialized:
        if download_model_from_firebase(FIREBASE_MODEL_NAME, MODEL_SAVE_PATH):
             agent.load(MODEL_SAVE_PATH)
             # Đặt Epsilon = 1.0 để bot tò mò tối đa ngay từ đầu
             agent.epsilon = 1.0 
             print("--- Đã tải trí nhớ và đặt Epsilon = 1.0 (Khám phá tối đa) ---")

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        
        done = False
        total_profit = 0
        step_counter = 0
        
        while not done:
            action = agent.choose_action(state)
            leverage = 5 
            next_state, reward, done = env.step(action, leverage)
            
            if next_state is not None:
                next_state = np.reshape(next_state, [1, env.state_size])
                agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_profit += reward
            step_counter += 1

        # --- Hết 1 Tập ---
        agent.replay() 
        
        status = "BÌNH THƯỜNG"
        if env.balance >= 600: status = "THẮNG (600$)"
        elif env.balance <= 450: status = "CHÁY (450$)"
        
        if (e + 1) % 10 == 0:
            print(f"Tập {e+1}: Số dư ${env.balance:.2f}, Epsilon: {agent.epsilon:.4f}, {status}")

        if (e + 1) % 5 == 0:
            agent.update_target_model()

        # Lưu và báo cáo mỗi 50 tập
        if (e + 1) % 50 == 0:
            agent.save(MODEL_SAVE_PATH)
            summary = (f"📊 Báo cáo (Tập {e+1}):\n"
                       f"- Số dư: ${env.balance:.2f}\n"
                       f"- Trạng thái: {status}\n"
                       f"- Epsilon: {agent.epsilon:.4f}")
            send_telegram_message(summary)

            if firebase_initialized:
                upload_model_to_firebase(MODEL_SAVE_PATH, FIREBASE_MODEL_NAME)

if __name__ == "__main__":
    train_bot(1000)
