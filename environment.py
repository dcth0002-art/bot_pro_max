# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from config import STARTING_BALANCE, STATE_SIZE

class TradingEnvironment:
    def __init__(self, data):
        """
        Khởi tạo môi trường giao dịch.
        :param data: DataFrame của pandas chứa dữ liệu giá (OHLCV).
        """
        self.data = data
        self.state_size = STATE_SIZE * 5
        self.current_step = STATE_SIZE
        self.initial_balance = STARTING_BALANCE
        self.balance = STARTING_BALANCE
        self.position = None  # None, 'long', 'short'
        self.entry_price = 0
        self.done = False

    def _get_state(self):
        """
        Lấy ra trạng thái hiện tại của thị trường.
        Ví dụ: lấy dữ liệu của STATE_SIZE cây nến gần nhất.
        """
        if self.current_step >= len(self.data):
            return None
        start = self.current_step - STATE_SIZE + 1
        end = self.current_step + 1
        state_data = self.data.iloc[start:end]
        # Chuẩn hóa dữ liệu để mô hình học tốt hơn (ví dụ: chia cho giá đóng cửa cuối cùng)
        close_price = state_data['close'].values[-1]
        state = (state_data[['open', 'high', 'low', 'close', 'volume']].values / close_price).flatten()
        return state

    def reset(self):
        """
        Reset lại môi trường về trạng thái ban đầu.
        """
        self.current_step = STATE_SIZE
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.done = False
        return self._get_state()

    def step(self, action, leverage=1):
        """
        Thực hiện một hành động và chuyển sang trạng thái tiếp theo.
        :param action: 0 (Hold), 1 (Buy), 2 (Sell)
        :param leverage: Đòn bẩy
        :return: (next_state, reward, done)
        """
        if self.done:
            return None, 0, True

        reward = 0
        current_price = self.data['close'].iloc[self.current_step]

        # Xử lý hành động
        if action == 1: # Buy / Long
            if self.position is None:
                self.position = 'long'
                self.entry_price = current_price
            elif self.position == 'short': # Đóng vị thế short và mở long
                profit = (self.entry_price - current_price) / self.entry_price
                reward = profit * leverage
                self.balance += self.balance * reward
                self.position = 'long'
                self.entry_price = current_price

        elif action == 2: # Sell / Short
            if self.position is None:
                self.position = 'short'
                self.entry_price = current_price
            elif self.position == 'long': # Đóng vị thế long và mở short
                profit = (current_price - self.entry_price) / self.entry_price
                reward = profit * leverage
                self.balance += self.balance * reward
                self.position = 'short'
                self.entry_price = current_price
        
        # Nếu đang có vị thế, tính toán reward tạm thời
        if self.position == 'long':
            unrealized_profit = (current_price - self.entry_price) / self.entry_price
            reward = unrealized_profit * leverage
        elif self.position == 'short':
            unrealized_profit = (self.entry_price - current_price) / self.entry_price
            reward = unrealized_profit * leverage

        # Cập nhật bước thời gian
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Kiểm tra điều kiện cháy tài khoản
        if self.balance <= 0:
            self.done = True
            reward = -100 # Phạt thật nặng khi cháy tài khoản
            self.balance = 0

        next_state = self._get_state()
        if next_state is None:
            self.done = True

        return next_state, reward, self.done
