import os
import sys

project_root = r"C:\Users\chint\BTCUSDT_Quant_System"
sys.path.append(project_root)

from data.downloader import download_all_timeframes

def run_downloads():
    assets = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
    start_date = "2017-01-01"
    end_date = "2026-07-20"  # Current date
    
    timeframes = ["15m", "30m", "1h", "4h"]
    
    for asset in assets:
        print(f"\nDownloading data for {asset}...")
        download_all_timeframes(
            symbol=asset,
            start_date=start_date,
            end_date=end_date,
            timeframes=timeframes
        )

if __name__ == '__main__':
    run_downloads()
