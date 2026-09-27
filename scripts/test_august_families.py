import os
import sys
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.fast_backtest_engine import FastBacktestEngine
import scripts.strategy_library as sl

def test_august_families():
    data_dir = os.path.join(root_dir, 'data', 'august_raw')
    timeframes = ['1h', '30m', '15m', '5m', '4h']
    
    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005,
        slippage_pct=0.0005
    )
    
    results = []

    for tf in timeframes:
        f = os.path.join(data_dir, f"BTCUSDT_{tf}.parquet")
        if not os.path.exists(f): continue
        df = pd.read_parquet(f)
        if 'Open Time' not in df.columns: df = df.reset_index()
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        df = df.sort_values('Open Time').reset_index(drop=True)
        atr = sl.compute_atr(df).values

        # 1. Supertrend RSI
        sig = sl.strat_supertrend_rsi(df, 10, 2.5, 14, 75.0, 25.0)
        _, m = engine.run_backtest(df, sig, atr, 2.0, 4.0)
        results.append({'Name': 'Supertrend+RSI (10, 2.5)', 'TF': tf, 'Metrics': m})

        # 2. Volatility Squeeze Breakout
        sig = sl.strat_volatility_squeeze_breakout(df, 20, 2.0, 20, 1.2, 12)
        _, m = engine.run_backtest(df, sig, atr, 1.5, 3.0)
        results.append({'Name': 'Volatility Squeeze (1.2, 12)', 'TF': tf, 'Metrics': m})

        # 3. MACD Trend
        sig = sl.strat_macd_trend(df, 8, 17, 9, 50)
        _, m = engine.run_backtest(df, sig, atr, 2.5, 4.5)
        results.append({'Name': 'MACD + Trend EMA (8, 17, 9)', 'TF': tf, 'Metrics': m})

        # 4. HTF Trend + Pullback
        sig = sl.strat_htf_trend_pullback(df, 20, 80, 14, 40.0, 60.0)
        _, m = engine.run_backtest(df, sig, atr, 2.5, 4.5)
        results.append({'Name': 'HTF Trend + RSI Pullback', 'TF': tf, 'Metrics': m})

        # 5. Bollinger Reversion
        sig = sl.strat_bollinger_rsi_reversion(df, 20, 2.0, 14, 70.0, 30.0)
        _, m = engine.run_backtest(df, sig, atr, 2.0, 3.5)
        results.append({'Name': 'Bollinger + RSI Reversion', 'TF': tf, 'Metrics': m})

    print("="*90)
    print("                     ALL STRATEGY FAMILIES TEST FOR AUGUST 2026                        ")
    print("="*90)
    
    rows = []
    for r in results:
        m = r['Metrics']
        rows.append({
            'Strategy': r['Name'],
            'TF': r['TF'],
            'Trades': m['total_trades'],
            'Win Rate': f"{m['win_rate']:.1f}%",
            'PF': f"{m['profit_factor']:.2f}",
            'Net PnL': f"${m['net_pnl']:+,.2f}",
            'Max DD': f"{m['max_drawdown_pct']:.2f}%",
            'Sharpe': f"{m['sharpe_ratio']:.2f}"
        })
        
    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    test_august_families()
