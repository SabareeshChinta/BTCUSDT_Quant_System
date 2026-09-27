import os
from data.downloader import download_all_timeframes

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    download_all_timeframes("BTCUSDT", "2019-01-01", "2026-06-22", timeframes=["5m"], output_dir=data_dir)

if __name__ == "__main__":
    main()
