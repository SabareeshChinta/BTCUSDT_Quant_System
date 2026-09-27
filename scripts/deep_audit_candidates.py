import os
import sys
import json
import numpy as np
import pandas as pd

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from scripts.fast_backtest_engine import FastBacktestEngine, Trade
import scripts.strategy_library as sl

def run_deep_audit():
    print("================================================================================")
    print("                 DEEP VALIDATION, ROBUSTNESS & HOSTILE AUDIT                    ")
    print("================================================================================")
    
    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005,
        slippage_pct=0.0005
    )
    
    raw_dir = os.path.join(root_dir, 'data', 'raw')
    month_dir = os.path.join(root_dir, 'data', 'last_month_raw')
    
    timeframes = ['30m', '1h']
    dfs_month = {}
    dfs_full = {}
    
    for tf in timeframes:
        p_m = os.path.join(month_dir, f"BTCUSDT_{tf}.parquet")
        p_f = os.path.join(raw_dir, f"BTCUSDT_{tf}.parquet")
        
        df_m = pd.read_parquet(p_m)
        if 'Open Time' not in df_m.columns: df_m = df_m.reset_index()
        df_m['Open Time'] = pd.to_datetime(df_m['Open Time'])
        dfs_month[tf] = df_m.sort_values('Open Time').reset_index(drop=True)
        
        df_f = pd.read_parquet(p_f)
        if 'Open Time' not in df_f.columns: df_f = df_f.reset_index()
        df_f['Open Time'] = pd.to_datetime(df_f['Open Time'])
        dfs_full[tf] = df_f.sort_values('Open Time').reset_index(drop=True)

    # Define Top 5 Candidate Strategies
    candidates = [
        {
            'rank': 1,
            'name': 'Supertrend-RSI Momentum Filter (ST-RSI)',
            'family': 'Trend & Momentum Hybrid',
            'tf': '1h',
            'func': sl.strat_supertrend_rsi,
            'params': {'st_period': 10, 'st_mult': 2.5, 'rsi_period': 14, 'rsi_max_long': 75.0, 'rsi_min_short': 25.0},
            'risk_params': {'sl_mult': 2.0, 'tp_mult': 4.0},
            'description': 'Supertrend (10, 2.5) regime transitions filtered by RSI boundary (25-75). Trades exit via 2.0x ATR Stop Loss or 4.0x ATR Take Profit (1:2 R:R) or opposite flip.'
        },
        {
            'rank': 2,
            'name': 'Dual EMA Cross with ADX Trend Strength (EMA-ADX)',
            'family': 'Trend Following',
            'tf': '1h',
            'func': sl.strat_ema_adx,
            'params': {'fast_p': 20, 'slow_p': 50, 'adx_p': 14, 'adx_thresh': 15.0},
            'risk_params': {'sl_mult': 2.5, 'tp_mult': 4.5},
            'description': '20 EMA / 50 EMA golden/death cross confirmed by ADX(14) >= 15. Risk managed by 2.5x ATR SL and 4.5x ATR TP.'
        },
        {
            'rank': 3,
            'name': 'MACD Momentum with Macro Trend Filter (MACD-Trend)',
            'family': 'Momentum',
            'tf': '1h',
            'func': sl.strat_macd_trend,
            'params': {'fast': 8, 'slow': 17, 'signal': 9, 'trend_ema_p': 50},
            'risk_params': {'sl_mult': 2.5, 'tp_mult': 4.5},
            'description': 'Fast MACD (8, 17, 9) signal line crossover in the direction of the 50 EMA trend.'
        },
        {
            'rank': 4,
            'name': 'Volatility Squeeze Breakout (Bollinger / Keltner)',
            'family': 'Volatility Breakout',
            'tf': '30m',
            'func': sl.strat_volatility_squeeze_breakout,
            'params': {'bb_period': 20, 'bb_std': 2.0, 'kc_period': 20, 'kc_mult': 1.2, 'mom_period': 12},
            'risk_params': {'sl_mult': 1.5, 'tp_mult': 3.0},
            'description': 'Bollinger Band compression inside Keltner Channel, firing long/short based on 12-bar momentum on squeeze release.'
        },
        {
            'rank': 5,
            'name': 'Donchian Channel Breakout with Trend Baseline',
            'family': 'Breakout Trend',
            'tf': '30m',
            'func': sl.strat_donchian_breakout,
            'params': {'entry_period': 20, 'trend_period': 100},
            'risk_params': {'sl_mult': 2.0, 'tp_mult': 4.0},
            'description': '20-period price extreme breakout aligned with the 100-period EMA baseline.'
        }
    ]

    full_audit_results = []

    for cand in candidates:
        tf = cand['tf']
        df_month = dfs_month[tf]
        df_full = dfs_full[tf]
        
        # 1. Test Month Performance (July 2026)
        sig_test = cand['func'](df_month, **cand['params'])
        atr_test = sl.compute_atr(df_month).values
        trades_test, m_test = engine.run_backtest(df_month, sig_test, atr_test, **cand['risk_params'])

        # 2. Historical Multi-Period Validation:
        # Period 1: Train 2024-01-01 to 2024-12-31
        # Period 2: Train 2025-01-01 to 2025-12-31
        # Period 3: Validation 2026-01-01 to 2026-06-20
        df_train_2024 = df_full[(df_full['Open Time'] >= '2024-01-01') & (df_full['Open Time'] <= '2024-12-31')].copy().reset_index(drop=True)
        df_train_2025 = df_full[(df_full['Open Time'] >= '2025-01-01') & (df_full['Open Time'] <= '2025-12-31')].copy().reset_index(drop=True)
        df_val_2026 = df_full[(df_full['Open Time'] >= '2026-01-01') & (df_full['Open Time'] <= '2026-06-20')].copy().reset_index(drop=True)

        sig_2024 = cand['func'](df_train_2024, **cand['params'])
        atr_2024 = sl.compute_atr(df_train_2024).values
        _, m_2024 = engine.run_backtest(df_train_2024, sig_2024, atr_2024, **cand['risk_params'])

        sig_2025 = cand['func'](df_train_2025, **cand['params'])
        atr_2025 = sl.compute_atr(df_train_2025).values
        _, m_2025 = engine.run_backtest(df_train_2025, sig_2025, atr_2025, **cand['risk_params'])

        sig_val = cand['func'](df_val_2026, **cand['params'])
        atr_val = sl.compute_atr(df_val_2026).values
        _, m_val = engine.run_backtest(df_val_2026, sig_val, atr_val, **cand['risk_params'])

        # 3. Monte Carlo Trade Order Reshuffling (1,000 iterations)
        mc_net_pnls = []
        mc_max_dds = []
        trade_pnls = [t.net_pnl for t in trades_test]
        
        np.random.seed(42)
        if len(trade_pnls) > 0:
            for _ in range(1000):
                shuffled = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
                eq = np.cumsum(shuffled) + 100_000.0
                peak = np.maximum.accumulate(eq)
                dd = (peak - eq) / peak
                mc_net_pnls.append(sum(shuffled))
                mc_max_dds.append(np.max(dd) * 100.0)
                
            mc_profitable_pct = round((np.array(mc_net_pnls) > 0).mean() * 100.0, 1)
            mc_worst_dd = round(np.percentile(mc_max_dds, 95), 2)
            mc_var_95 = round(np.percentile(mc_net_pnls, 5), 2)
        else:
            mc_profitable_pct = 0.0
            mc_worst_dd = 0.0
            mc_var_95 = 0.0

        # 4. Single-Trade Dominance Test (Hostile Audit)
        sorted_trades = sorted(trades_test, key=lambda t: t.net_pnl, reverse=True)
        if len(sorted_trades) > 1:
            without_top1 = [t.net_pnl for t in sorted_trades[1:]]
            without_top2 = [t.net_pnl for t in sorted_trades[2:]]
            
            top1_wins = sum(p for p in without_top1 if p > 0)
            top1_loss = abs(sum(p for p in without_top1 if p < 0))
            pf_without_top1 = round(top1_wins / (top1_loss + 1e-6), 2)
            net_without_top1 = round(sum(without_top1), 2)
            
            top2_wins = sum(p for p in without_top2 if p > 0)
            top2_loss = abs(sum(p for p in without_top2 if p < 0))
            pf_without_top2 = round(top2_wins / (top2_loss + 1e-6), 2)
            net_without_top2 = round(sum(without_top2), 2)
        else:
            pf_without_top1, net_without_top1 = 0.0, 0.0
            pf_without_top2, net_without_top2 = 0.0, 0.0

        # 5. Parameter Neighborhood Sensitivity Matrix (Test Month)
        sensitivity_results = []
        if cand['rank'] == 1:
            # Vary ST Period (8, 10, 12) and Mult (2.2, 2.5, 2.8)
            for p_var in [8, 10, 12]:
                for m_var in [2.2, 2.5, 2.8]:
                    p_dict = cand['params'].copy()
                    p_dict['st_period'] = p_var
                    p_dict['st_mult'] = m_var
                    sig_var = cand['func'](df_month, **p_dict)
                    _, m_var_res = engine.run_backtest(df_month, sig_var, atr_test, **cand['risk_params'])
                    sensitivity_results.append({
                        'params': f"ST_P={p_var}, Mult={m_var}",
                        'trades': m_var_res['total_trades'],
                        'win_rate': m_var_res['win_rate'],
                        'pf': m_var_res['profit_factor'],
                        'net_pnl': m_var_res['net_pnl']
                    })

        full_audit_results.append({
            'rank': cand['rank'],
            'name': cand['name'],
            'family': cand['family'],
            'tf': cand['tf'],
            'description': cand['description'],
            'params': cand['params'],
            'risk_params': cand['risk_params'],
            'test_month_metrics': m_test,
            'train_2024_metrics': m_2024,
            'train_2025_metrics': m_2025,
            'val_2026_metrics': m_val,
            'monte_carlo': {
                'profitable_simulations_pct': mc_profitable_pct,
                'worst_case_dd_95th': mc_worst_dd,
                'pnl_var_95th': mc_var_95
            },
            'single_trade_dominance': {
                'pf_without_top1': pf_without_top1,
                'net_without_top1': net_without_top1,
                'pf_without_top2': pf_without_top2,
                'net_without_top2': net_without_top2,
                'top1_pnl': round(sorted_trades[0].net_pnl, 2) if sorted_trades else 0.0
            },
            'sensitivity': sensitivity_results,
            'trades': [
                {
                    'entry_time': str(t.entry_time)[:19],
                    'entry_price': round(t.entry_price, 2),
                    'exit_time': str(t.exit_time)[:19],
                    'exit_price': round(t.exit_price, 2),
                    'direction': 'LONG' if t.direction == 1 else 'SHORT',
                    'size': round(t.size, 4),
                    'gross_pnl': round(t.gross_pnl, 2),
                    'fee': round(t.fee, 2),
                    'net_pnl': round(t.net_pnl, 2),
                    'return_pct': round(t.return_pct * 100.0, 2),
                    'exit_reason': t.exit_reason
                }
                for t in trades_test
            ]
        })

    with open(os.path.join(root_dir, 'scripts', 'deep_audit_summary.json'), 'w') as f:
        json.dump(full_audit_results, f, indent=4)
        
    print("Deep Audit Complete. Saved to scripts/deep_audit_summary.json")

if __name__ == "__main__":
    run_deep_audit()
