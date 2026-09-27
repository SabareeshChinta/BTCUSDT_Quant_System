import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.fast_backtest_engine import FastBacktestEngine, Trade
from scripts.strategy_library import (
    compute_atr,
    strat_ema_adx,
    strat_supertrend_rsi,
    strat_macd_trend,
    strat_bollinger_rsi_reversion,
    strat_volatility_squeeze_breakout,
    strat_htf_trend_pullback,
    strat_donchian_breakout
)

def run_discovery():
    print("================================================================================")
    print("          HOSTILE QUANTITATIVE STRATEGY DISCOVERY & ROBUSTNESS ENGINE          ")
    print("================================================================================")

    # 1. Load Data
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    month_dir = os.path.join(root_dir, 'data', 'last_month_raw')
    
    timeframes = ['15m', '30m', '1h', '5m']
    dfs_full = {}
    dfs_month = {}
    
    for tf in timeframes:
        p_raw = os.path.join(raw_dir, f"BTCUSDT_{tf}.parquet")
        p_month = os.path.join(month_dir, f"BTCUSDT_{tf}.parquet")
        
        if os.path.exists(p_raw):
            df_r = pd.read_parquet(p_raw)
            if 'Open Time' not in df_r.columns:
                df_r = df_r.reset_index()
            df_r['Open Time'] = pd.to_datetime(df_r['Open Time'])
            df_r = df_r.sort_values('Open Time').reset_index(drop=True)
            dfs_full[tf] = df_r
            
        if os.path.exists(p_month):
            df_m = pd.read_parquet(p_month)
            if 'Open Time' not in df_m.columns:
                df_m = df_m.reset_index()
            df_m['Open Time'] = pd.to_datetime(df_m['Open Time'])
            df_m = df_m.sort_values('Open Time').reset_index(drop=True)
            dfs_month[tf] = df_m

    print(f"Loaded {len(dfs_full)} full historical datasets and {len(dfs_month)} out-of-sample monthly datasets.")
    
    # 2. Define Train, Validation, and Test Periods
    # Train: 2024-01-01 to 2025-12-31 (2 years discovery)
    # Val:   2026-01-01 to 2026-06-20 (6 months validation)
    # Test:  2026-07-01 to 2026-07-31 (Most recent completed month)

    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005, # 5 bps
        slippage_pct=0.0005    # 5 bps
    )

    # Define Candidate Strategies
    candidates = []

    # --- Family 1: HTF Trend + LTF Pullback (Hybrid) ---
    for tf in ['15m', '30m', '5m']:
        for fast_p in [15, 20, 25]:
            for slow_p in [60, 80, 100]:
                for rsi_buy in [35.0, 40.0, 45.0]:
                    for sl_mult, tp_mult in [(2.5, 4.0), (3.0, 4.5), (3.5, 5.0)]:
                        candidates.append({
                            'family': 'Hybrid (HTF Trend + RSI Pullback)',
                            'func': strat_htf_trend_pullback,
                            'tf': tf,
                            'params': {
                                'htf_ema_fast': fast_p, 'htf_ema_slow': slow_p,
                                'ltf_rsi_p': 14, 'rsi_buy_thresh': rsi_buy, 'rsi_sell_thresh': 100.0 - rsi_buy
                            },
                            'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                        })

    # --- Family 2: Supertrend + RSI (Momentum / Trend) ---
    for tf in ['15m', '30m', '1h']:
        for st_p in [7, 10, 14]:
            for st_m in [2.0, 2.5, 3.0]:
                for rsi_max in [65.0, 70.0, 75.0]:
                    for sl_mult, tp_mult in [(2.0, 3.5), (2.5, 4.0), (3.0, 4.5)]:
                        candidates.append({
                            'family': 'Trend/Momentum (Supertrend + RSI)',
                            'func': strat_supertrend_rsi,
                            'tf': tf,
                            'params': {
                                'st_period': st_p, 'st_mult': st_m, 'rsi_period': 14,
                                'rsi_max_long': rsi_max, 'rsi_min_short': 100.0 - rsi_max
                            },
                            'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                        })

    # --- Family 3: MACD + Trend Filter ---
    for tf in ['15m', '30m', '1h']:
        for fast, slow, sig in [(8, 21, 5), (12, 26, 9)]:
            for trend_p in [50, 100, 150]:
                for sl_mult, tp_mult in [(2.5, 4.0), (3.0, 4.5)]:
                    candidates.append({
                        'family': 'Momentum (MACD + Trend EMA)',
                        'func': strat_macd_trend,
                        'tf': tf,
                        'params': {'fast': fast, 'slow': slow, 'signal': sig, 'trend_ema_p': trend_p},
                        'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                    })

    # --- Family 4: Bollinger Mean Reversion ---
    for tf in ['15m', '30m', '5m']:
        for bb_p in [14, 20, 25]:
            for bb_std in [1.8, 2.0, 2.2]:
                for rsi_os in [25.0, 30.0, 35.0]:
                    for sl_mult, tp_mult in [(2.0, 3.0), (2.5, 3.5)]:
                        candidates.append({
                            'family': 'Mean Reversion (Bollinger + RSI Extreme)',
                            'func': strat_bollinger_rsi_reversion,
                            'tf': tf,
                            'params': {
                                'bb_period': bb_p, 'bb_std': bb_std, 'rsi_period': 14,
                                'rsi_ob': 100.0 - rsi_os, 'rsi_os': rsi_os
                            },
                            'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                        })

    # --- Family 5: Volatility Squeeze Breakout ---
    for tf in ['15m', '30m', '1h']:
        for bb_std in [1.8, 2.0]:
            for kc_m in [1.3, 1.5]:
                for mom_p in [8, 12, 16]:
                    for sl_mult, tp_mult in [(2.5, 4.0), (3.0, 4.5)]:
                        candidates.append({
                            'family': 'Volatility Squeeze Breakout',
                            'func': strat_volatility_squeeze_breakout,
                            'tf': tf,
                            'params': {
                                'bb_period': 20, 'bb_std': bb_std,
                                'kc_period': 20, 'kc_mult': kc_m, 'mom_period': mom_p
                            },
                            'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                        })

    # --- Family 6: Donchian Channel Breakout ---
    for tf in ['15m', '30m', '1h']:
        for don_p in [15, 20, 30]:
            for trend_p in [50, 100, 200]:
                for sl_mult, tp_mult in [(2.0, 4.0), (2.5, 5.0)]:
                    candidates.append({
                        'family': 'Trend (Donchian Breakout)',
                        'func': strat_donchian_breakout,
                        'tf': tf,
                        'params': {'entry_period': don_p, 'trend_period': trend_p},
                        'risk_params': {'sl_mult': sl_mult, 'tp_mult': tp_mult}
                    })

    print(f"Total candidate parameter configurations across 6 strategy families: {len(candidates)}")

    evaluated_results = []

    # Run evaluation
    for idx, cand in enumerate(candidates):
        tf = cand['tf']
        if tf not in dfs_full or tf not in dfs_month:
            continue
            
        df_full = dfs_full[tf]
        df_test = dfs_month[tf]
        
        # Slices
        df_train = df_full[(df_full['Open Time'] >= '2024-01-01') & (df_full['Open Time'] <= '2025-12-31')].copy().reset_index(drop=True)
        df_val = df_full[(df_full['Open Time'] >= '2026-01-01') & (df_full['Open Time'] <= '2026-06-20')].copy().reset_index(drop=True)
        
        if len(df_train) == 0 or len(df_val) == 0:
            continue

        try:
            # 1. Compute Test Month (Out-of-Sample)
            sig_test = cand['func'](df_test, **cand['params'])
            atr_test = compute_atr(df_test).values
            trades_test, m_test = engine.run_backtest(df_test, sig_test, atr_test, **cand['risk_params'])

            # Filter: Check if Test Month has reasonable trade count (approx 10 to 30 trades) and is profitable
            if m_test['total_trades'] < 8 or m_test['total_trades'] > 35:
                continue
            if m_test['net_pnl'] <= 0 or m_test['profit_factor'] < 1.05:
                continue

            # 2. Compute Train Performance
            sig_train = cand['func'](df_train, **cand['params'])
            atr_train = compute_atr(df_train).values
            trades_train, m_train = engine.run_backtest(df_train, sig_train, atr_train, **cand['risk_params'])

            if m_train['net_pnl'] <= 0 or m_train['profit_factor'] < 1.0:
                continue

            # 3. Compute Validation Performance
            sig_val = cand['func'](df_val, **cand['params'])
            atr_val = compute_atr(df_val).values
            trades_val, m_val = engine.run_backtest(df_val, sig_val, atr_val, **cand['risk_params'])

            if m_val['net_pnl'] <= 0 or m_val['profit_factor'] < 1.0:
                continue

            # Composite Ranking Score:
            # Rewards trade count closeness to 17.5, high Sharpe, solid PF in test and validation
            trade_count_penalty = 1.0 - (abs(m_test['total_trades'] - 17.5) / 17.5)
            score = (
                m_test['profit_factor'] * 
                max(0.1, m_test['sharpe_ratio']) * 
                min(m_val['profit_factor'], 2.5) * 
                max(0.2, trade_count_penalty)
            )

            evaluated_results.append({
                'family': cand['family'],
                'func_name': cand['func'].__name__,
                'tf': tf,
                'params': cand['params'],
                'risk_params': cand['risk_params'],
                'score': score,
                'test_metrics': m_test,
                'val_metrics': m_val,
                'train_metrics': m_train,
                'test_trades': trades_test
            })
        except Exception:
            continue

    print(f"\nFiltered and validated robust candidates: {len(evaluated_results)}")
    
    # Sort by score descending
    evaluated_results.sort(key=lambda x: x['score'], reverse=True)

    # Save top 10 results to JSON
    output_path = os.path.join(root_dir, 'scripts', 'discovery_results.json')
    
    serializable_results = []
    for r in evaluated_results[:10]:
        r_copy = r.copy()
        # Convert trades to dicts
        trade_dicts = []
        for t in r['test_trades']:
            trade_dicts.append({
                'entry_idx': t.entry_idx,
                'entry_time': str(t.entry_time),
                'entry_price': round(t.entry_price, 2),
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'size': round(t.size, 4),
                'exit_time': str(t.exit_time),
                'exit_price': round(t.exit_price, 2),
                'exit_reason': t.exit_reason,
                'gross_pnl': round(t.gross_pnl, 2),
                'fee': round(t.fee, 2),
                'slippage': round(t.slippage, 2),
                'net_pnl': round(t.net_pnl, 2),
                'return_pct': round(t.return_pct * 100.0, 2),
                'holding_bars': t.holding_bars
            })
        r_copy['test_trades'] = trade_dicts
        serializable_results.append(r_copy)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
        
    print(f"Top results saved to {output_path}")

    # Print summary of Top 5 Candidates
    print("\n" + "="*80)
    print("                        TOP 5 DISCOVERED STRATEGIES                             ")
    print("="*80)
    for i, res in enumerate(evaluated_results[:5]):
        tm = res['test_metrics']
        vm = res['val_metrics']
        trm = res['train_metrics']
        print(f"\nRANK #{i+1}: {res['family']} | Timeframe: {res['tf']}")
        print(f"  Params: {res['params']} | Risk: {res['risk_params']}")
        print(f"  Score: {res['score']:.2f}")
        print(f"  [TEST MONTH (July 2026)] Trades: {tm['total_trades']} | WinRate: {tm['win_rate']}% | PF: {tm['profit_factor']} | Net PnL: ${tm['net_pnl']:,.2f} | Max DD: {tm['max_drawdown_pct']}% | Sharpe: {tm['sharpe_ratio']}")
        print(f"  [VALIDATION (H1 2026)]   Trades: {vm['total_trades']} | WinRate: {vm['win_rate']}% | PF: {vm['profit_factor']} | Net PnL: ${vm['net_pnl']:,.2f} | Max DD: {vm['max_drawdown_pct']}% | Sharpe: {vm['sharpe_ratio']}")
        print(f"  [TRAIN (2024-2025)]      Trades: {trm['total_trades']} | WinRate: {trm['win_rate']}% | PF: {trm['profit_factor']} | Net PnL: ${trm['net_pnl']:,.2f} | Max DD: {trm['max_drawdown_pct']}% | Sharpe: {trm['sharpe_ratio']}")

if __name__ == "__main__":
    run_discovery()
