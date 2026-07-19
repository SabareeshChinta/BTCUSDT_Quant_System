import os
import pandas as pd
from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from main import load_data, split_data

def inspect_trades():
    timeframe = "1h"
    config = SystemConfig()
    
    # Temporarily disable hard drawdown halt so we get the exact trade stream
    config.risk.hard_drawdown_halt_pct = 1.0
    
    df = load_data(f"data/raw/BTCUSDT_{timeframe}.parquet")
    df_dev, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    print("Running Backtest (Base S1 - Full Capital Allocation)...")
    s1_engine = BacktestEngine(config, use_risk_management=False, use_ml_filter=False)
    s1_res = s1_engine.run(df_dev, timeframe=timeframe, config_name="S1")
    
    print("\n--- FIRST 5 TRADES (S1 BASE STRATEGY) ---")
    s1_trades = s1_res.trades_df
    if not s1_trades.empty:
        for idx, row in s1_trades.head(5).iterrows():
            dir_str = "LONG" if row['direction'] == 1 else "SHORT"
            notional = row['size'] * row['entry_price']
            print(f"Trade {idx+1}: Direction={dir_str}, Entry={row['entry_price']:.2f}, Exit={row['exit_price']:.2f}, Size={row['size']:.4f} BTC, Notional=${notional:.2f}, PnL=${row['net_pnl']:.2f}")
    else:
        print("No trades generated in S1.")
        
    print("\n--- SUMMARY METRICS (S1 BASE STRATEGY) ---")
    m1 = s1_res.metrics
    print(f"Total Return: {m1.get('Return (%)', 0):.2f}%")
    print(f"Sharpe Ratio: {m1.get('Sharpe Ratio', 0):.2f}")
    print(f"Max Drawdown: {m1.get('Max Drawdown (%)', 0):.2f}%")
    print(f"Win Rate: {m1.get('Win Rate', 0):.1%}")
    print(f"Total Trades: {m1.get('Total Trades', len(s1_trades))}")
        
    print("\nRunning Backtest (S2 - Risk Management 1% Sizing)...")
    s2_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
    s2_res = s2_engine.run(df_dev, timeframe=timeframe, config_name="S2")
    
    print("\n--- FIRST 5 TRADES (S2 RISK MANAGED STRATEGY) ---")
    s2_trades = s2_res.trades_df
    if not s2_trades.empty:
        for idx, row in s2_trades.head(5).iterrows():
            dir_str = "LONG" if row['direction'] == 1 else "SHORT"
            notional = row['size'] * row['entry_price']
            print(f"Trade {idx+1}: Direction={dir_str}, Entry={row['entry_price']:.2f}, Exit={row['exit_price']:.2f}, Size={row['size']:.4f} BTC, Notional=${notional:.2f}, PnL=${row['net_pnl']:.2f}")
    else:
        print("No trades generated in S2.")

    print("\n--- SUMMARY METRICS (S2 RISK MANAGED STRATEGY) ---")
    m2 = s2_res.metrics
    print(f"Total Return: {m2.get('Return (%)', 0):.2f}%")
    print(f"Sharpe Ratio: {m2.get('Sharpe Ratio', 0):.2f}")
    print(f"Max Drawdown: {m2.get('Max Drawdown (%)', 0):.2f}%")
    print(f"Win Rate: {m2.get('Win Rate', 0):.1%}")
    print(f"Total Trades: {m2.get('Total Trades', len(s2_trades))}")

if __name__ == "__main__":
    inspect_trades()
