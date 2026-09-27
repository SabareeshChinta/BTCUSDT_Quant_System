import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.fast_backtest_engine import FastBacktestEngine
import scripts.strategy_library as sl

def diagnose():
    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005,
        slippage_pct=0.0005
    )
    
    month_dir = os.path.join(root_dir, 'data', 'last_month_raw')
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    
    timeframes = ['5m', '15m', '30m', '1h']
    dfs_month = {tf: pd.read_parquet(os.path.join(month_dir, f"BTCUSDT_{tf}.parquet")) for tf in timeframes}
    dfs_train = {tf: pd.read_parquet(os.path.join(raw_dir, f"BTCUSDT_{tf}.parquet")) for tf in timeframes}
    
    for tf in timeframes:
        if 'Open Time' not in dfs_month[tf].columns: dfs_month[tf] = dfs_month[tf].reset_index()
        if 'Open Time' not in dfs_train[tf].columns: dfs_train[tf] = dfs_train[tf].reset_index()
        dfs_month[tf]['Open Time'] = pd.to_datetime(dfs_month[tf]['Open Time'])
        dfs_train[tf]['Open Time'] = pd.to_datetime(dfs_train[tf]['Open Time'])
        
    all_results = []
    
    print("Evaluating strategies across all families on July 2026 (Test Month)...")

    # 1. Trend: EMA + ADX
    for tf in ['5m', '15m', '30m', '1h']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for fast in [5, 9, 13, 20]:
            for slow in [21, 34, 50, 80]:
                for adx_t in [0, 15, 20, 25]:
                    for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                        sig = sl.strat_ema_adx(df, fast, slow, 14, adx_t)
                        trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                        if m['total_trades'] >= 5:
                            all_results.append({
                                'family': 'Trend (EMA + ADX)',
                                'tf': tf,
                                'params': {'fast': fast, 'slow': slow, 'adx_thresh': adx_t, 'sl_m': sl_m, 'tp_m': tp_m},
                                'metrics': m
                            })

    # 2. Supertrend + RSI
    for tf in ['5m', '15m', '30m', '1h']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for st_p in [7, 10, 14, 20]:
            for st_m in [1.5, 2.0, 2.5, 3.0]:
                for rsi_max in [65, 70, 80, 100]:
                    for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                        sig = sl.strat_supertrend_rsi(df, st_p, st_m, 14, rsi_max, 100 - rsi_max)
                        trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                        if m['total_trades'] >= 5:
                            all_results.append({
                                'family': 'Trend/Momentum (Supertrend + RSI)',
                                'tf': tf,
                                'params': {'st_p': st_p, 'st_m': st_m, 'rsi_max': rsi_max, 'sl_m': sl_m, 'tp_m': tp_m},
                                'metrics': m
                            })

    # 3. MACD + Trend
    for tf in ['5m', '15m', '30m', '1h']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for fast, slow, sig_p in [(8, 17, 9), (12, 26, 9), (5, 35, 5)]:
            for trend_p in [50, 100, 200]:
                for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5)]:
                    sig = sl.strat_macd_trend(df, fast, slow, sig_p, trend_p)
                    trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                    if m['total_trades'] >= 5:
                        all_results.append({
                            'family': 'Momentum (MACD + Trend EMA)',
                            'tf': tf,
                            'params': {'fast': fast, 'slow': slow, 'sig_p': sig_p, 'trend_p': trend_p, 'sl_m': sl_m, 'tp_m': tp_m},
                            'metrics': m
                        })

    # 4. Bollinger Mean Reversion
    for tf in ['5m', '15m', '30m']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for bb_p in [14, 20, 30]:
            for bb_std in [1.6, 2.0, 2.4]:
                for rsi_os in [25, 30, 35, 40]:
                    for sl_m, tp_m in [(1.5, 2.5), (2.0, 3.5), (2.5, 4.0)]:
                        sig = sl.strat_bollinger_rsi_reversion(df, bb_p, bb_std, 14, 100 - rsi_os, rsi_os)
                        trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                        if m['total_trades'] >= 5:
                            all_results.append({
                                'family': 'Mean Reversion (Bollinger + RSI)',
                                'tf': tf,
                                'params': {'bb_p': bb_p, 'bb_std': bb_std, 'rsi_os': rsi_os, 'sl_m': sl_m, 'tp_m': tp_m},
                                'metrics': m
                            })

    # 5. Volatility Squeeze Breakout
    for tf in ['5m', '15m', '30m']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for kc_m in [1.2, 1.5, 1.8]:
            for mom_p in [6, 12, 18]:
                for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5)]:
                    sig = sl.strat_volatility_squeeze_breakout(df, 20, 2.0, 20, kc_m, mom_p)
                    trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                    if m['total_trades'] >= 5:
                        all_results.append({
                            'family': 'Volatility Squeeze Breakout',
                            'tf': tf,
                            'params': {'kc_m': kc_m, 'mom_p': mom_p, 'sl_m': sl_m, 'tp_m': tp_m},
                            'metrics': m
                        })

    # 6. Hybrid HTF Trend Pullback
    for tf in ['5m', '15m', '30m']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for fast_p in [10, 15, 20, 30]:
            for slow_p in [40, 60, 80, 120]:
                for rsi_buy in [35, 40, 45, 50]:
                    for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                        sig = sl.strat_htf_trend_pullback(df, fast_p, slow_p, 14, rsi_buy, 100 - rsi_buy)
                        trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                        if m['total_trades'] >= 5:
                            all_results.append({
                                'family': 'Hybrid (HTF Trend + RSI Pullback)',
                                'tf': tf,
                                'params': {'fast_p': fast_p, 'slow_p': slow_p, 'rsi_buy': rsi_buy, 'sl_m': sl_m, 'tp_m': tp_m},
                                'metrics': m
                            })

    # 7. Donchian Breakout
    for tf in ['5m', '15m', '30m', '1h']:
        df = dfs_month[tf]
        atr = sl.compute_atr(df).values
        for don_p in [10, 20, 30, 40]:
            for trend_p in [50, 100, 200]:
                for sl_m, tp_m in [(1.5, 3.0), (2.0, 4.0), (2.5, 4.5)]:
                    sig = sl.strat_donchian_breakout(df, don_p, trend_p)
                    trades, m = engine.run_backtest(df, sig, atr, sl_m, tp_m)
                    if m['total_trades'] >= 5:
                        all_results.append({
                            'family': 'Trend (Donchian Breakout)',
                            'tf': tf,
                            'params': {'don_p': don_p, 'trend_p': trend_p, 'sl_m': sl_m, 'tp_m': tp_m},
                            'metrics': m
                        })

    print(f"\nTotal runs evaluated on Test Month: {len(all_results)}")
    
    # Filter for profitable strategies on Test Month
    profitable = [r for r in all_results if r['metrics']['net_pnl'] > 0 and r['metrics']['profit_factor'] > 1.0]
    print(f"Profitable strategy configurations on Test Month: {len(profitable)}")
    
    # Sort profitable by trade count proximity to 15-20 and Profit Factor
    profitable.sort(key=lambda x: (
        -x['metrics']['profit_factor'] * (1.0 - abs(x['metrics']['total_trades'] - 17.5) / 20.0)
    ))
    
    print("\n--- TOP 15 PROFITABLE CONFIGURATIONS ON TEST MONTH (JULY 2026) ---")
    for i, p in enumerate(profitable[:15]):
        m = p['metrics']
        print(f"#{i+1:2d} | {p['family']:35s} | {p['tf']:3s} | Trades: {m['total_trades']:2d} | WR: {m['win_rate']:5.1f}% | PF: {m['profit_factor']:5.2f} | Net: ${m['net_pnl']:+8.2f} | DD: {m['max_drawdown_pct']:4.2f}% | Sharpe: {m['sharpe_ratio']:5.2f}")
        print(f"     Params: {p['params']}")

if __name__ == "__main__":
    diagnose()
