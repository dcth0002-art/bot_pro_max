# -*- coding: utf-8 -*-
import telegram
import os
import asyncio

# Lấy thông tin từ biến môi trường
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

bot = None
is_initialized = False

def initialize_telegram_bot():
    """
    Khởi tạo bot Telegram.
    """
    global bot, is_initialized
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
            is_initialized = True
            print("--- Kết nối Telegram Bot thành công! ---")
            return True
        except Exception as e:
            print(f"Lỗi kết nối Telegram: {e}")
            is_initialized = False
            return False
    else:
        print("--- Không tìm thấy thông tin Telegram, sẽ không gửi báo cáo. ---")
        is_initialized = False
        return False

async def send_telegram_message_async(message):
    """
    Hàm bất đồng bộ để gửi tin nhắn.
    """
    if is_initialized:
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Lỗi gửi tin nhắn Telegram: {e}")

def send_telegram_message(message):
    """
    Hàm đồng bộ để gọi và chạy hàm bất đồng bộ.
    """
    try:
        # Chạy vòng lặp sự kiện cho đến khi hàm async hoàn thành
        asyncio.run(send_telegram_message_async(message))
    except RuntimeError as e:
        # Xử lý lỗi nếu một vòng lặp sự kiện đã chạy
        if "cannot run loop while another loop is running" in str(e):
            loop = asyncio.get_event_loop()
            loop.create_task(send_telegram_message_async(message))
        else:
            raise e
