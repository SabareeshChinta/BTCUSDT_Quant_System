import os
import pandas as pd
from config.settings import SystemConfig
from data.downloader import download_all_timeframes
from data.parquet_manager import load_data
from backtesting.ablation_runner import AblationRunner

def test_last_month():
    config = SystemConfig.default()
    
    timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
    symbol = "BTCUSDT"
    start_date = "2026-07-01"
    end_date = "2026-07-31"
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'last_month_raw')
    os.makedirs(data_dir, exist_ok=True)
    
    all_exist = all(os.path.exists(os.path.join(data_dir, f"{symbol}_{tf}.parquet")) for tf in timeframes)
    if not all_exist:
        print(f"Downloading data from {start_date} to {end_date}...")
        try:
            download_all_timeframes(symbol, start_date, end_date, timeframes, data_dir)
        except Exception as e:
            print(f"Download warning (will use existing files if available): {e}")
    else:
        print(f"Using cached data from {data_dir} for {start_date} to {end_date}.")
    
    runner = AblationRunner(config)
    
    for tf in timeframes:
        file_path = os.path.join(data_dir, f"{symbol}_{tf}.parquet")
        try:
            df = load_data(file_path)
            results = runner.run_all(df, tf)
            
            print(f"\n--- Last Month Backtest Results for {tf} ---")
            for name, res in results.items():
                print(f"\n{name} Configuration ({tf}):")
                print(res.summary())
        except Exception as e:
            print(f"Error testing timeframe {tf}: {e}")

if __name__ == "__main__":
    test_last_month()
