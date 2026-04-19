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
        self.fee = 0.0002 # Giảm phí xuống một chút (0.02%) để bot dễ thở hơn lúc đầu

    def _get_state(self):
        if self.current_step >= len(self.data):
            return None
        start = self.current_step - STATE_SIZE + 1
        end = self.current_step + 1
        state_data = self.data.iloc[start:end].copy()
        
        # CHUẨN HÓA RIÊNG BIỆT:
        # Giá chia cho giá đóng cửa cuối cùng
        last_close = state_data['close'].values[-1]
        price_cols = ['open', 'high', 'low', 'close']
        state_data[price_cols] = state_data[price_cols] / last_close
        
        # Volume chia cho max volume trong đoạn đó để về khoảng [0, 1]
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

        # 1. Tính lợi nhuận thực tế (P&L)
        if self.position == 'long':
            pnl_pct = (current_price - prev_price) / prev_price
            self.balance += self.balance * pnl_pct * leverage
            reward = pnl_pct * leverage
        elif self.position == 'short':
            pnl_pct = (prev_price - current_price) / prev_price
            self.balance += self.balance * pnl_pct * leverage
            reward = pnl_pct * leverage

        # 2. Xử lý hành động (Cộng thêm reward nhỏ để khuyến khích giữ lệnh đúng)
        if action == 1 and self.position != 'long': # Buy
            self.balance -= self.balance * self.fee 
            self.position = 'long'
            self.entry_price = current_price
            reward -= self.fee 
        elif action == 2 and self.position != 'short': # Sell
            self.balance -= self.balance * self.fee
            self.position = 'short'
            self.entry_price = current_price
            reward -= self.fee
        elif action == 0 and self.position is not None: # Close
            self.balance -= self.balance * self.fee
            self.position = None
            reward -= self.fee

        # 3. KIỂM TRA ĐIỀU KIỆN KẾT THÚC (Mục tiêu lãi 20%, lỗ 10%)
        # Thắng: 600$
        if self.balance >= 600:
            reward += 20 # Tăng thưởng lên gấp đôi
            self.done = True
        
        # Cháy: 450$ (Lỗ 10%)
        elif self.balance <= (self.initial_balance * 0.9):
            reward -= 20 # Phạt nặng hơn để bot sợ
            self.done = True

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_state = self._get_state()
        return next_state, reward, self.done
