# -*- coding: utf-8 -*-
import firebase_admin
from firebase_admin import credentials, storage
import os
import json

def initialize_firebase():
    """
    Khởi tạo kết nối tới Firebase bằng cách đọc nội dung JSON từ biến môi trường.
    """
    creds_json_str = os.environ.get('GOOGLE_CREDENTIALS_JSON')
    bucket_name = os.environ.get('FIREBASE_STORAGE_BUCKET')

    if creds_json_str and bucket_name:
        try:
            # Chuyển đổi chuỗi JSON thành một dictionary
            creds_json = json.loads(creds_json_str)
            
            # Sử dụng dictionary này để khởi tạo kết nối
            cred = credentials.Certificate(creds_json)
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
            print("--- Kết nối Firebase thành công! ---")
            return True
        except Exception as e:
            print(f"Lỗi kết nối Firebase: {e}")
            return False
    else:
        print("--- Không tìm thấy GOOGLE_CREDENTIALS_JSON hoặc FIREBASE_STORAGE_BUCKET, bỏ qua Firebase... ---")
        return False


def upload_model_to_firebase(local_file_path, destination_blob_name):
    """
    Tải file model từ local lên Firebase Storage.
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
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(source_blob_name)
        if blob.exists():
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            blob.download_to_filename(destination_file_path)
            print(f"--- Model đã được tải về từ Firebase: {destination_file_path} ---")
            return True
        else:
            # Đây không phải lỗi, chỉ là không có file để tải về
            return False
    except Exception as e:
        print(f"Lỗi khi tải model từ Firebase: {e}")
        return False
