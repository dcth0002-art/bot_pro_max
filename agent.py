# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import deque
from model import create_model
from config import (STATE_SIZE, ACTION_SPACE, LEARNING_RATE, DISCOUNT_FACTOR,
                    EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
                    BUFFER_SIZE, BATCH_SIZE)

class Agent:
    def __init__(self, state_size=STATE_SIZE, action_space=ACTION_SPACE):
        self.state_size = state_size
        self.action_space = action_space
        
        # Hyperparameters
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.gamma = DISCOUNT_FACTOR    # discount rate
        self.epsilon = EPSILON_START  # exploration rate
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
        self.learning_rate = LEARNING_RATE
        
        # Model
        self.model = create_model(state_size, action_space, self.learning_rate)
        self.target_model = create_model(state_size, action_space, self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """Sao chép trọng số từ model chính sang target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Lưu lại kinh nghiệm vào bộ nhớ."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """
        Quyết định hành động: khám phá (ngẫu nhiên) hoặc khai thác (dựa trên mô hình).
        """
        if np.random.rand() <= self.epsilon:
            # Chọn hành động ngẫu nhiên (khám phá)
            return random.randrange(self.action_space)
        
        # Dự đoán Q-value từ trạng thái hiện tại (khai thác)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size=BATCH_SIZE):
        """
        Học lại từ những kinh nghiệm đã lưu trong bộ nhớ.
        """
        if len(self.memory) < batch_size:
            return # Chưa đủ kinh nghiệm để học
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Công thức Bellman Equation để tính giá trị mục tiêu (target Q-value)
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state, verbose=0)[0]))
            
            # Lấy Q-value hiện tại từ model chính
            target_f = self.model.predict(state, verbose=0)
            # Cập nhật Q-value cho hành động đã thực hiện
            target_f[0][action] = target
            
            # Huấn luyện model với cặp (state, target_f)
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        # Giảm epsilon sau mỗi lần học để giảm tỷ lệ khám phá
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
