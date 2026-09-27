import os
import pandas as pd
from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.ablation_runner import AblationRunner

def test_all():
    config = SystemConfig.default()
    timeframes = config.timeframes
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    print(f"Testing Timeframes: {timeframes}")
    
    runner = AblationRunner(config)
    
    for tf in timeframes:
        file_path = get_data_path("BTCUSDT", tf, data_dir)
        try:
            df = load_data(file_path)
            # Split data
            df_bt, df_fw, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
            
            print(f"\n{'='*60}")
            print(f"[{tf}] Backtest Phase (until {config.backtest.backtest_end})")
            print(f"{'='*60}")
            
            if not df_bt.empty:
                results_bt = runner.run_all(df_bt, tf)
                for name, res in results_bt.items():
                    print(f"\n{name} Configuration ({tf} - Backtest):")
                    print(res.summary())
            else:
                print("No data for backtest.")
                
            print(f"\n{'='*60}")
            print(f"[{tf}] Forward Phase ({config.backtest.backtest_end} to {config.backtest.forward_end})")
            print(f"{'='*60}")
            
            if not df_fw.empty:
                results_fw = runner.run_all(df_fw, tf)
                for name, res in results_fw.items():
                    print(f"\n{name} Configuration ({tf} - Forward):")
                    print(res.summary())
            else:
                print("No data for forward test.")
                
        except Exception as e:
            print(f"Error testing timeframe {tf}: {e}")

if __name__ == "__main__":
    test_all()
