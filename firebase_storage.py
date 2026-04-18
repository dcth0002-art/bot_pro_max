# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, storage
import os

def initialize_firebase():
    """
    Khởi tạo kết nối tới Firebase bằng service account key.
    """
    # Railway sẽ đọc biến môi trường GOOGLE_CREDENTIALS_JSON
    # và tạo file tạm thời cho chúng ta.
    if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        try:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {
                'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
            })
            print("--- Kết nối Firebase thành công! ---")
            return True
        except Exception as e:
            print(f"Lỗi kết nối Firebase: {e}")
            return False
    else:
        print("--- Không tìm thấy thông tin kết nối Firebase, bỏ qua... ---")
        return False


def upload_model_to_firebase(local_file_path, destination_blob_name):
    """
    Tải file model từ local lên Firebase Storage.
    :param local_file_path: Đường dẫn file local (e.g., 'saved_models/trading_bot_model.h5')
    :param destination_blob_name: Tên file trên Firebase (e.g., 'trading_bot_model.h5')
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"--- Model đã được tải lên Firebase: {destination_blob_name} ---")
    except Exception as e:
        print(f"Lỗi khi tải model lên Firebase: {e}")


def download_model_from_firebase(source_blob_name, destination_file_path):
    """
    Tải file model từ Firebase Storage về local.
    :param source_blob_name: Tên file trên Firebase.
    :param destination_file_path: Đường dẫn lưu file local.
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(source_blob_name)
        if blob.exists():
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            blob.download_to_filename(destination_file_path)
            print(f"--- Model đã được tải về từ Firebase: {destination_file_path} ---")
            return True
        else:
            print(f"--- Không tìm thấy model trên Firebase. ---")
            return False
    except Exception as e:
        print(f"Lỗi khi tải model từ Firebase: {e}")
        return False
