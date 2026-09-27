import os
import pandas as pd
import numpy as np

def audit_dataset():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data', 'raw')
    timeframes = ['5m', '15m', '30m', '1h', '4h', '1d']
    
    print("=== DATASET INTEGRITY & SANITY AUDIT ===")
    for tf in timeframes:
        filepath = os.path.join(data_dir, f"BTCUSDT_{tf}.parquet")
        if not os.path.exists(filepath):
            filepath = os.path.join(root_dir, 'data', 'last_1_year_raw', f"BTCUSDT_{tf}.parquet")
            
        if not os.path.exists(filepath):
            print(f"[{tf}] Missing file: {filepath}")
            continue
            
        df = pd.read_parquet(filepath)
        if 'Open Time' not in df.columns and df.index.name == 'Open Time':
            df = df.reset_index()
            
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        
        # Check duplicates
        dupes = df['Open Time'].duplicated().sum()
        
        # Check sort
        is_sorted = df['Open Time'].is_monotonic_increasing
        
        # Check range
        start = df['Open Time'].min()
        end = df['Open Time'].max()
        
        # Check missing candles (expected time step)
        freq_map = {'5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h', '1d': '1D'}
        if tf in freq_map:
            full_range = pd.date_range(start=start, end=end, freq=freq_map[tf])
            missing_count = len(full_range) - len(df)
        else:
            missing_count = 0
            
        # Check volume and price validity
        null_counts = df[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().sum().to_dict()
        zero_vol = (df['Volume'] <= 0).sum()
        
        print(f"\nTimeframe: {tf}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date Range: {start} -> {end}")
        print(f"  Monotonic Increasing: {is_sorted}")
        print(f"  Duplicates: {dupes}")
        print(f"  Missing timestamps: {missing_count}")
        print(f"  Null values: {null_counts}")
        print(f"  Zero/Negative Volume bars: {zero_vol}")

if __name__ == "__main__":
    audit_dataset()
