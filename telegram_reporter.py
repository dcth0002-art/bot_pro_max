# -*- coding: utf-8 -*-
import telegram
import os
import http.client
import json

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

is_initialized = False

def initialize_telegram_bot():
    """
    Kiểm tra xem các biến môi trường đã được cấu hình chưa.
    """
    global is_initialized
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        is_initialized = True
        print("--- Cấu hình Telegram đã sẵn sàng! ---")
        return True
    else:
        print("--- Không tìm thấy thông tin Telegram, sẽ không gửi báo cáo. ---")
        is_initialized = False
        return False

def send_telegram_message(message):
    """
    Gửi tin nhắn bằng cách gọi trực tiếp API của Telegram (cách này ổn định hơn).
    """
    if not is_initialized:
        return

    # Cấu trúc payload cho API
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'  # Cho phép định dạng chữ
    }
    
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # Tạo kết nối HTTPS
        conn = http.client.HTTPSConnection("api.telegram.org")
        
        # Tạo đường dẫn API
        api_path = f"/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        # Gửi yêu cầu POST
        conn.request("POST", api_path, json.dumps(payload), headers)
        
        # Đọc phản hồi (quan trọng để đóng kết nối)
        response = conn.getresponse()
        response.read()
        
        conn.close()
    except Exception as e:
        # In ra lỗi nếu có sự cố
        print(f"Lỗi khi gửi tin nhắn Telegram qua API: {e}")
