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

def test_august():
    config = SystemConfig.default()
    
    timeframes = ["1h", "30m", "15m", "5m", "4h", "1d"]
    symbol = "BTCUSDT"
    data_dir = os.path.join(root_dir, 'data', 'august_raw')
    
    print("="*80)
    print("           AUGUST 2026 BACKTEST: SUPERTREND-RSI MOMENTUM FILTER                 ")
    print("="*80)
    print(f"Date Range: 2026-08-01 to 2026-08-28")
    print(f"Strategy: Supertrend(Period={config.strategy.st_period}, Mult={config.strategy.st_mult}) + RSI(14) Filter [25.0, 75.0]")
    print(f"Risk Management: SL={config.risk.stop_loss_atr_mult}x ATR, TP={config.risk.take_profit_atr_mult}x ATR (1:2 R:R)")
    print(f"Friction: 0.05% Taker Fee + 0.05% Slippage per side (10 bps round-trip)")
    print("="*80)

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

    summary_rows = []

    for tf in timeframes:
        file_path = os.path.join(data_dir, f"{symbol}_{tf}.parquet")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_parquet(file_path)
        if 'Open Time' not in df.columns:
            df = df.reset_index()
        df['Open Time'] = pd.to_datetime(df['Open Time'])
        df = df.sort_values('Open Time').reset_index(drop=True)

        signals_df = thesis.generate_signals(df)
        signals_arr = np.where(
            signals_df['signal_changed'] & (signals_df['raw_signal'] == 'BUY'), 1,
            np.where(signals_df['signal_changed'] & (signals_df['raw_signal'] == 'SELL'), -1, 0)
        )

        atr = compute_atr(df, period=14).values

        trades, m = engine.run_backtest(
            df=df,
            signals=signals_arr,
            atr=atr,
            sl_mult=config.risk.stop_loss_atr_mult,
            tp_mult=config.risk.take_profit_atr_mult
        )

        summary_rows.append({
            'Timeframe': tf,
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

        if tf == "1h":
            print("\n--- 1H PRIMARY TIMEFRAME TRADE LOG (AUGUST 2026) ---")
            header = f"{'#':<3} | {'Entry Time':<19} | {'Entry Px':<9} | {'Exit Time':<19} | {'Exit Px':<9} | {'Dir':<5} | {'Size':<6} | {'Gross PnL':<10} | {'Fee':<7} | {'Net PnL':<10} | {'Exit Reason'}"
            print(header)
            print("-" * len(header))
            for i, t in enumerate(trades):
                print(f"{i+1:<3} | {str(t.entry_time)[:19]:<19} | ${t.entry_price:<8.2f} | {str(t.exit_time)[:19]:<19} | ${t.exit_price:<8.2f} | {'LONG' if t.direction==1 else 'SHORT':<5} | {t.size:<6.3f} | ${t.gross_pnl:<9.2f} | ${t.fee:<6.2f} | ${t.net_pnl:<9.2f} | {t.exit_reason}")

    print("\n" + "="*80)
    print("                  AUGUST 2026 SUMMARY ACROSS ALL TIMEFRAMES                     ")
    print("="*80)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    test_august()
