# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import time
import os

# --- CẤU HÌNH ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m'
TOTAL_LIMIT = 2000  # Tổng số nến mục tiêu
OUTPUT_FILE = 'btc_usdt_1m_data.csv'

def download_historical_data():
    """
    Tải 2000 cây nến bằng cách tính toán mốc thời gian bắt đầu lùi về quá khứ.
    """
    print("--- Đang kết nối tới sàn OKX (ccxt) ---")
    exchange = ccxt.okx()
    
    # Tính toán mốc thời gian: Bây giờ trừ đi (2000 phút * 60 giây * 1000 ms)
    # Thêm một chút trừ hao (2100 phút) để đảm bảo lấy đủ 2000 nến
    duration_ms = (TOTAL_LIMIT + 100) * 60 * 1000 
    since = exchange.milliseconds() - duration_ms
    
    all_ohlcv = []
    
    print(f"Bắt đầu tải dữ liệu từ: {pd.to_datetime(since, unit='ms')}")

    while len(all_ohlcv) < TOTAL_LIMIT:
        try:
            # OKX giới hạn 100 nến/lần fetch
            limit = 100
            ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                print("Không còn dữ liệu để tải.")
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Mốc since tiếp theo là sau nến cuối cùng vừa tải được
            since = ohlcv[-1][0] + 1 
            
            print(f"Đã tải: {len(all_ohlcv)}/{TOTAL_LIMIT} nến...")
            
            # Tránh bị rate limit
            time.sleep(exchange.rateLimit / 1000) 

        except Exception as e:
            print(f"Lỗi khi tải: {e}")
            break

    if len(all_ohlcv) > 0:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Xóa trùng và lấy đúng 2000 nến mới nhất
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        df = df.tail(TOTAL_LIMIT)
        
        df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*50)
        print(f"✅ THÀNH CÔNG! Đã lưu {len(df)} nến vào {OUTPUT_FILE}")
        print(f"Bắt đầu: {pd.to_datetime(df['timestamp'].iloc[0], unit='ms')}")
        print(f"Kết thúc: {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
        print("="*50)
    else:
        print("❌ Lỗi: Không tải được dữ liệu.")

if __name__ == "__main__":
    download_historical_data()
