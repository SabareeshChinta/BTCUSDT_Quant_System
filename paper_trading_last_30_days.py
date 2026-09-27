import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config.settings import SystemConfig
from strategy.theses import SupertrendRSI
from scripts.fast_backtest_engine import FastBacktestEngine
import scripts.strategy_library as sl

def run_paper_trading_30_days():
    data_dir = os.path.join(root_dir, 'data', 'last_30_days_raw')
    timeframes = ["1h", "30m", "15m", "5m", "4h", "1d"]
    symbol = "BTCUSDT"
    
    print("="*90)
    print("         PAPER TRADING RESULTS: EXACT LAST 30 DAYS (AUG 28 - SEP 27, 2026)          ")
    print("="*90)
    print(f"Date Boundary: 2026-08-28 00:00:00  -->  2026-09-27 23:59:59 (UTC)")
    print(f"Capital: $100,000 | Risk/Trade: 1.0% | Taker Fee: 0.05% | Slippage: 0.05% per side")
    print("="*90)

    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005,
        slippage_pct=0.0005
    )

    all_strategy_results = []

    for tf in timeframes:
        file_path = os.path.join(data_dir, f"{symbol}_{tf}.parquet")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_parquet(file_path)
        if 'Open Time' not in df.columns:
            df = df.reset_index()
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        df = df.sort_values('Open Time').reset_index(drop=True)
        atr = sl.compute_atr(df, period=14).values

        # Strategy 1: HTF Trend + RSI Pullback (30m / 1h optimal setup)
        sig_htf = sl.strat_htf_trend_pullback(df, htf_ema_fast=20, htf_ema_slow=150, ltf_rsi_p=14, rsi_buy_thresh=45.0, rsi_sell_thresh=55.0)
        trades_htf, m_htf = engine.run_backtest(df, sig_htf, atr, sl_mult=2.5, tp_mult=4.5)
        all_strategy_results.append({
            'Strategy': 'HTF Trend + RSI Pullback (20/150)',
            'TF': tf,
            'Metrics': m_htf,
            'Trades': trades_htf
        })

        # Strategy 2: Supertrend-RSI Momentum Filter
        sig_st = sl.strat_supertrend_rsi(df, st_period=10, st_mult=2.5, rsi_period=14, rsi_max_long=75.0, rsi_min_short=25.0)
        trades_st, m_st = engine.run_backtest(df, sig_st, atr, sl_mult=2.0, tp_mult=4.0)
        all_strategy_results.append({
            'Strategy': 'Supertrend + RSI Filter (10, 2.5)',
            'TF': tf,
            'Metrics': m_st,
            'Trades': trades_st
        })

        # Strategy 3: MACD Trend Filtered Momentum
        sig_macd = sl.strat_macd_trend(df, fast=8, slow=17, signal=9, trend_ema_p=50)
        trades_macd, m_macd = engine.run_backtest(df, sig_macd, atr, sl_mult=2.5, tp_mult=4.5)
        all_strategy_results.append({
            'Strategy': 'MACD + Trend EMA (8, 17, 9)',
            'TF': tf,
            'Metrics': m_macd,
            'Trades': trades_macd
        })

    # Print Comparative Table across all strategies & timeframes
    rows = []
    for res in all_strategy_results:
        m = res['Metrics']
        rows.append({
            'Strategy': res['Strategy'],
            'TF': res['TF'],
            'Trades': m['total_trades'],
            'Win Rate': f"{m['win_rate']:.1f}%",
            'Profit Factor': f"{m['profit_factor']:.2f}",
            'Gross PnL': f"${m['gross_pnl']:+,.2f}",
            'Fees': f"${m['fees']:,.2f}",
            'Net PnL': f"${m['net_pnl']:+,.2f}",
            'Max DD': f"{m['max_drawdown_pct']:.2f}%",
            'Sharpe': f"{m['sharpe_ratio']:.2f}",
            'Avg Trade': f"${m['avg_trade_pnl']:+,.2f}"
        })

    df_summary = pd.DataFrame(rows)
    print("\n--- COMPARATIVE PERFORMANCE SUMMARY (LAST 30 DAYS: AUG 28 - SEP 27, 2026) ---")
    print(df_summary.to_string(index=False))

    # Print trade log for top performing candidate in last 30 days
    best = max(all_strategy_results, key=lambda x: x['Metrics']['net_pnl'])
    print("\n" + "="*90)
    print(f"       TOP PERFORMING PAPER TRADING STRATEGY (LAST 30 DAYS): {best['Strategy']} ({best['TF']})")
    print("="*90)
    bm = best['Metrics']
    print(f"Total Trades: {bm['total_trades']} | Win Rate: {bm['win_rate']:.1f}% | Profit Factor: {bm['profit_factor']:.2f} | Net PnL: ${bm['net_pnl']:+,.2f} | Max DD: {bm['max_drawdown_pct']:.2f}% | Sharpe: {bm['sharpe_ratio']:.2f}")
    
    header = f"{'#':<3} | {'Entry Time':<19} | {'Entry Px':<9} | {'Exit Time':<19} | {'Exit Px':<9} | {'Dir':<5} | {'Size':<6} | {'Gross PnL':<10} | {'Fee':<7} | {'Net PnL':<10} | {'Exit Reason'}"
    print("\n" + header)
    print("-" * len(header))
    for i, t in enumerate(best['Trades']):
        print(f"{i+1:<3} | {str(t.entry_time)[:19]:<19} | ${t.entry_price:<8.2f} | {str(t.exit_time)[:19]:<19} | ${t.exit_price:<8.2f} | {'LONG' if t.direction==1 else 'SHORT':<5} | {t.size:<6.3f} | ${t.gross_pnl:<9.2f} | ${t.fee:<6.2f} | ${t.net_pnl:<9.2f} | {t.exit_reason}")

if __name__ == "__main__":
    run_paper_trading_30_days()
