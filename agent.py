# -*- coding: utf-8 -*-
import numpy as np
import random
import os
from collections import deque
from model import create_model
from config import (STATE_SIZE, ACTION_SPACE, LEARNING_RATE, DISCOUNT_FACTOR,
                    EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
                    BUFFER_SIZE, BATCH_SIZE)

class Agent:
    def __init__(self, state_size=STATE_SIZE, action_space=ACTION_SPACE):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS
        self.learning_rate = LEARNING_RATE
        
        # Khởi tạo model từ cấu trúc trong model.py
        self.model = create_model(state_size, action_space, self.learning_rate)
        self.target_model = create_model(state_size, action_space, self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=BATCH_SIZE):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(batch_size):
            states[i] = minibatch[i][0]
            actions.append(minibatch[i][1])
            rewards.append(minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(minibatch[i][4])

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(states, targets, epochs=1, verbose=0)

    def load(self, name):
        """Chỉ tải trọng số để tránh lỗi version Keras."""
        if os.path.exists(name):
            try:
                self.model.load_weights(name)
                self.update_target_model()
                print(f"--- Đã tải trọng số model từ {name} ---")
            except Exception as e:
                print(f"Lỗi khi tải trọng số: {e}")

    def save(self, name):
        """Chỉ lưu trọng số cho an toàn."""
        self.model.save_weights(name)
