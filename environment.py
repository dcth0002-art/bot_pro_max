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
        self.position = None 
        self.entry_price = 0
        self.done = False
        self.fee = 0.0002 # Phí giao dịch 0.02%

    def _get_state(self):
        if self.current_step >= len(self.data):
            return None
        start = self.current_step - STATE_SIZE + 1
        end = self.current_step + 1
        
        cols = ['open', 'high', 'low', 'close', 'volume']
        state_data = self.data.iloc[start:end][cols].copy()
        
        # Chuẩn hóa giá dựa trên giá đóng cửa cuối cùng
        last_close = state_data['close'].values[-1]
        price_cols = ['open', 'high', 'low', 'close']
        state_data[price_cols] = state_data[price_cols] / last_close
        
        # Chuẩn hóa Volume
        max_vol = state_data['volume'].max()
        if max_vol > 0:
            state_data['volume'] = state_data['volume'] / max_vol
        
        return state_data.values.flatten()

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

        # 1. Tính lợi nhuận thực tế (P&L) dựa trên giá thay đổi giữa 2 bước
        if self.position == 'long':
            pnl_pct = (current_price - prev_price) / prev_price
            self.balance += self.balance * pnl_pct * leverage
            reward = pnl_pct * 10 # Reward nhỏ theo biến động giá
        elif self.position == 'short':
            pnl_pct = (prev_price - current_price) / prev_price
            self.balance += self.balance * pnl_pct * leverage
            reward = pnl_pct * 10

        # 2. Xử lý hành động giao dịch
        if action == 1 and self.position != 'long': # Mở Long
            self.balance -= self.balance * self.fee 
            self.position = 'long'
            reward -= 0.1 # Phạt nhẹ để tránh trade quá nhiều (overtrading)
        elif action == 2 and self.position != 'short': # Mở Short
            self.balance -= self.balance * self.fee
            self.position = 'short'
            reward -= 0.1
        elif action == 0 and self.position is not None: # Đóng lệnh
            self.balance -= self.balance * self.fee
            self.position = None
            reward -= 0.05

        # 3. KIỂM TRA KẾT THÚC VÀ THƯỞNG/PHẠT NẶNG
        # THẮNG: Đạt 600$ (+20% vốn)
        if self.balance >= 600:
            reward += 100 # THƯỞNG LỚN
            self.done = True
        
        # THUA: Cháy/Lỗ xuống 450$ (-10% vốn)
        elif self.balance <= 450:
            reward -= 200 # PHẠT RẤT NẶNG
            self.done = True

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done
