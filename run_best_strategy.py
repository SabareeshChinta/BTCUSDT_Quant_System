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
from scripts.strategy_library import compute_atr

def run_best_strategy():
    config = SystemConfig.default()
    
    print("="*80)
    print("           EXECUTING BEST STRATEGY: SUPERTREND-RSI MOMENTUM FILTER              ")
    print("="*80)
    print(f"Strategy Type: {config.strategy.strategy_type}")
    print(f"Supertrend Period: {config.strategy.st_period}, Multiplier: {config.strategy.st_mult}")
    print(f"RSI Filter (Period: {config.strategy.rsi_period}): Long <= {config.strategy.rsi_max_long}, Short >= {config.strategy.rsi_min_short}")
    print(f"Risk Management: Stop Loss = {config.risk.stop_loss_atr_mult}x ATR, Take Profit = {config.risk.take_profit_atr_mult}x ATR (1:2 R:R)")
    print(f"Trading Friction: Commission = {config.risk.commission_pct*100:.2f}%, Slippage = {config.risk.slippage_pct*100:.2f}% per side")
    print("="*80)

    month_file = os.path.join(root_dir, 'data', 'last_month_raw', 'BTCUSDT_1h.parquet')
    if not os.path.exists(month_file):
        raise FileNotFoundError(f"Missing data file: {month_file}")
        
    df = pd.read_parquet(month_file)
    if 'Open Time' not in df.columns:
        df = df.reset_index()
    df['Open Time'] = pd.to_datetime(df['Open Time'])
    df = df.sort_values('Open Time').reset_index(drop=True)

    engine = FastBacktestEngine(
        initial_capital=config.backtest.initial_capital,
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        commission_pct=config.risk.commission_pct,
        slippage_pct=config.risk.slippage_pct
    )

    thesis = SupertrendRSI(
        st_period=config.strategy.st_period,
        st_mult=config.strategy.st_mult,
        rsi_period=config.strategy.rsi_period,
        rsi_max_long=config.strategy.rsi_max_long,
        rsi_min_short=config.strategy.rsi_min_short
    )

    signals_df = thesis.generate_signals(df)
    
    # Map signals to integer array: 1 = BUY, -1 = SELL, 0 = None
    signals_arr = np.where(
        signals_df['signal_changed'] & (signals_df['raw_signal'] == 'BUY'), 1,
        np.where(signals_df['signal_changed'] & (signals_df['raw_signal'] == 'SELL'), -1, 0)
    )

    atr = compute_atr(df, period=14).values

    trades, metrics = engine.run_backtest(
        df=df,
        signals=signals_arr,
        atr=atr,
        sl_mult=config.risk.stop_loss_atr_mult,
        tp_mult=config.risk.take_profit_atr_mult
    )

    print("\n--- PERFORMANCE SUMMARY (JULY 2026 OUT-OF-SAMPLE) ---")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    print("\n--- COMPLETED TRADE LOG ---")
    header = f"{'#':<3} | {'Entry Time':<19} | {'Entry Px':<9} | {'Exit Time':<19} | {'Exit Px':<9} | {'Dir':<5} | {'Size':<6} | {'Gross PnL':<10} | {'Fee':<7} | {'Net PnL':<10} | {'Exit Reason'}"
    print(header)
    print("-" * len(header))
    
    for i, t in enumerate(trades):
        print(f"{i+1:<3} | {str(t.entry_time)[:19]:<19} | ${t.entry_price:<8.2f} | {str(t.exit_time)[:19]:<19} | ${t.exit_price:<8.2f} | {'LONG' if t.direction==1 else 'SHORT':<5} | {t.size:<6.3f} | ${t.gross_pnl:<9.2f} | ${t.fee:<6.2f} | ${t.net_pnl:<9.2f} | {t.exit_reason}")

if __name__ == "__main__":
    run_best_strategy()
