# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def create_model(state_size, action_space, learning_rate):
    """
    Tạo và biên dịch mô hình mạng neuron (Deep Q-Network).
    
    :param state_size: Kích thước của input (số lượng features của trạng thái).
    :param action_space: Kích thước của output (số lượng hành động có thể).
    :param learning_rate: Tốc độ học của mô hình.
    :return: Mô hình Keras đã được biên dịch.
    """
    model = Sequential()
    
    # Input layer và các hidden layers
    # Sử dụng các lớp Dense (fully connected) là phổ biến cho dạng bài toán này.
    model.add(Dense(128, input_dim=state_size, activation='relu'))
    model.add(BatchNormalization()) # Giúp ổn định quá trình huấn luyện
    model.add(Dropout(0.2)) # Giảm overfitting
    
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    
    # Output layer
    # Số neuron bằng với số hành động, activation là 'linear' cho bài toán regression (dự đoán Q-value)
    model.add(Dense(action_space, activation='linear'))
    
    # Biên dịch mô hình
    # Sử dụng Mean Squared Error làm hàm mất mát để so sánh Q-value dự đoán và Q-value mục tiêu.
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    
    return model
