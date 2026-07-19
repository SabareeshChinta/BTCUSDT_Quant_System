import os
import pandas as pd
from config.settings import SystemConfig
from data.downloader import download_all_timeframes
from data.parquet_manager import load_data
from backtesting.ablation_runner import AblationRunner

def test_last_week():
    config = SystemConfig.default()
    
    # We will test on 15m or 1h timeframe
    timeframes = ["15m"] 
    symbol = "BTCUSDT"
    start_date = "2026-06-15"
    end_date = "2026-07-15"
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'last_week_raw')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading data from {start_date} to {end_date}...")
    download_all_timeframes(symbol, start_date, end_date, timeframes, data_dir)
    
    file_path = os.path.join(data_dir, f"{symbol}_15m.parquet")
    df = load_data(file_path)
    
    # Disable volatility filter for testing
    config.strategy.atr_percentile_threshold = 0.0
    
    runner = AblationRunner(config)
    results = runner.run_all(df, "15m")
    
    print("\n--- Last Month Backtest Results (Volatility Filter DISABLED) ---")
    for name, res in results.items():
        print(f"\n{name} Configuration:")
        print(res.summary())

if __name__ == "__main__":
    test_last_week()
