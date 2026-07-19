import os
import pandas as pd
from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.backtest_engine import BacktestEngine

def evaluate_filters():
    config = SystemConfig.default()
    timeframe = "1h"
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    file_path = get_data_path("BTCUSDT", timeframe, data_dir)
    
    try:
        df = load_data(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    df_dev, df_forward, df_paper = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    print("Evaluating S2 (Risk Managed 1% Strategy) with Additional Filters...\n")
    
    configs = [
        ("Base S2 (No New Filters)", {}),
        ("S2 + ATR Filter (ATR > 20 SMA)", {"use_atr_filter": True}),
        ("S2 + Trend Filter (Price vs 200 EMA)", {"use_trend_filter": True}),
        ("S2 + Minimum R:R Filter (Reward:Risk >= 1.5)", {"use_min_rr_filter": True}),
        ("S2 + ALL 3 Filters Combined", {"use_atr_filter": True, "use_trend_filter": True, "use_min_rr_filter": True})
    ]
    
    results_table = []
    
    for label, kwargs in configs:
        print(f"Running {label}...")
        engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
        res = engine.run(df_dev, timeframe=timeframe, config_name="S2", **kwargs)
        m = res.metrics
        
        results_table.append({
            "Configuration": label,
            "Total Return": f"{m.get('Return (%)', 0):.2f}%",
            "Sharpe Ratio": f"{m.get('Sharpe Ratio', 0):.2f}",
            "Max Drawdown": f"{m.get('Max Drawdown (%)', 0):.2f}%",
            "Win Rate": f"{m.get('Win Rate', 0):.1%}",
            "Total Trades": m.get('Total Trades', len(res.trades_df))
        })
        
    df_res = pd.DataFrame(results_table)
    print("\n" + "="*85)
    print(" "*25 + "FILTER EVALUATION RESULTS SUMMARY")
    print("="*85)
    print(df_res.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    evaluate_filters()
