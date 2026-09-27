import os
import sys
from data.downloader import download_all_timeframes

def download_august():
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'august_raw')
    os.makedirs(data_dir, exist_ok=True)
    
    timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
    symbol = "BTCUSDT"
    start_date = "2026-08-01"
    end_date = "2026-08-28"
    
    print(f"Downloading August data from {start_date} to {end_date}...")
    try:
        download_all_timeframes(symbol, start_date, end_date, timeframes, data_dir)
        print("August data downloaded successfully!")
    except Exception as e:
        print(f"Download error: {e}")

if __name__ == "__main__":
    download_august()
