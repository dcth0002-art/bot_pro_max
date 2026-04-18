# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from config import STARTING_BALANCE, STATE_SIZE

class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.state_size = STATE_SIZE * 5
        self.current_step = STATE_SIZE
        self.initial_balance = STARTING_BALANCE
        self.balance = STARTING_BALANCE
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0
        self.done = False
        self.fee = 0.0004 

    def _get_state(self):
        if self.current_step >= len(self.data):
            return None
        start = self.current_step - STATE_SIZE + 1
        end = self.current_step + 1
        state_data = self.data.iloc[start:end]
        close_price = state_data['close'].values[-1]
        if close_price == 0: close_price = 1
        state = (state_data[['open', 'high', 'low', 'close', 'volume']].values / close_price).flatten()
        return state

    def reset(self):
        self.current_step = STATE_SIZE
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def step(self, action, leverage=1):
        if self.done:
            return None, 0, True

        reward = 0
        current_price = self.data['close'].iloc[self.current_step]
        prev_price = self.data['close'].iloc[self.current_step - 1]

        # 1. Tính toán biến động số dư theo giá thị trường
        if self.position == 'long':
            price_change_pct = (current_price - prev_price) / prev_price
            self.balance += self.balance * price_change_pct * leverage
            reward = price_change_pct * leverage
        elif self.position == 'short':
            price_change_pct = (prev_price - current_price) / prev_price
            self.balance += self.balance * price_change_pct * leverage
            reward = price_change_pct * leverage

        # 2. Xử lý hành động giao dịch
        if action == 1 and self.position != 'long': # Mở Long
            self.balance -= self.balance * self.fee 
            self.position = 'long'
            self.entry_price = current_price
            reward -= self.fee
        elif action == 2 and self.position != 'short': # Mở Short
            self.balance -= self.balance * self.fee
            self.position = 'short'
            self.entry_price = current_price
            reward -= self.fee
        elif action == 0 and self.position is not None: # Thoát lệnh
            self.balance -= self.balance * self.fee
            self.position = None
            reward -= self.fee

        # 3. KIỂM TRA ĐIỀU KIỆN MỤC TIÊU VÀ RỦI RO (YÊU CẦU MỚI)
        
        # Thắng: Đạt 600$ (Lãi 20%)
        if self.balance >= 600:
            reward += 10 # Thưởng lớn để bot ghi nhớ hành vi này
            self.done = True
            print(f"--- Tập này THẮNG: Đạt ${self.balance:.2f} ---")

        # Thua: Tụt 10% vốn (Còn 450$)
        elif self.balance <= (self.initial_balance * 0.9):
            reward -= 10 # Phạt nặng
            self.done = True
            print(f"--- Tập này THUA: Số dư còn ${self.balance:.2f} ---")

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done
