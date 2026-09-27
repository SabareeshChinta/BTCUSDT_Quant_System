import os
import sys
import pandas as pd
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if os.path.dirname(root_dir) not in sys.path:
    sys.path.insert(0, os.path.dirname(root_dir))

from scripts.fast_backtest_engine import FastBacktestEngine, Trade
import scripts.strategy_library as sl

def search_august():
    data_dir_aug = os.path.join(os.path.dirname(root_dir), 'data', 'august_raw')
    data_dir_jul = os.path.join(os.path.dirname(root_dir), 'data', 'last_month_raw')
    
    engine = FastBacktestEngine(
        initial_capital=100_000.0,
        risk_per_trade_pct=0.01,
        commission_pct=0.0005,
        slippage_pct=0.0005
    )
    
    timeframes = ['1h', '30m', '15m']
    dfs_aug = {tf: pd.read_parquet(os.path.join(data_dir_aug, f"BTCUSDT_{tf}.parquet")) for tf in timeframes}
    dfs_jul = {tf: pd.read_parquet(os.path.join(data_dir_jul, f"BTCUSDT_{tf}.parquet")) for tf in timeframes}
    
    for tf in timeframes:
        for d in [dfs_aug[tf], dfs_jul[tf]]:
            if 'Open Time' not in d.columns: d = d.reset_index()
            d['Open Time'] = pd.to_datetime(d['Open Time'])
            d.sort_values('Open Time', inplace=True)
            d.reset_index(drop=True, inplace=True)
            
    candidates = []

    # 1. HTF Trend + RSI Pullback (Adaptive Thresholds)
    for tf in ['1h', '30m', '15m']:
        df_aug = dfs_aug[tf]
        df_jul = dfs_jul[tf]
        atr_aug = sl.compute_atr(df_aug).values
        atr_jul = sl.compute_atr(df_jul).values
        
        for fast_p in [10, 20, 30]:
            for slow_p in [50, 80, 100, 150]:
                for rsi_buy in [35.0, 40.0, 45.0, 50.0]:
                    for sl_m, tp_m in [(2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                        sig_aug = sl.strat_htf_trend_pullback(df_aug, fast_p, slow_p, 14, rsi_buy, 100.0 - rsi_buy)
                        trades_aug, m_aug = engine.run_backtest(df_aug, sig_aug, atr_aug, sl_m, tp_m)
                        
                        if m_aug['net_pnl'] > 0 and m_aug['total_trades'] >= 5:
                            # Evaluate on July too
                            sig_jul = sl.strat_htf_trend_pullback(df_jul, fast_p, slow_p, 14, rsi_buy, 100.0 - rsi_buy)
                            trades_jul, m_jul = engine.run_backtest(df_jul, sig_jul, atr_jul, sl_m, tp_m)
                            
                            candidates.append({
                                'name': 'HTF Trend + RSI Pullback',
                                'tf': tf,
                                'params': {'fast_p': fast_p, 'slow_p': slow_p, 'rsi_buy': rsi_buy, 'sl_m': sl_m, 'tp_m': tp_m},
                                'aug_metrics': m_aug,
                                'jul_metrics': m_jul,
                                'aug_trades': trades_aug,
                                'combined_pnl': m_aug['net_pnl'] + m_jul['net_pnl']
                            })

    # 2. ADX Trend Filtered MACD
    for tf in ['1h', '30m', '15m']:
        df_aug = dfs_aug[tf]
        df_jul = dfs_jul[tf]
        atr_aug = sl.compute_atr(df_aug).values
        atr_jul = sl.compute_atr(df_jul).values
        
        for fast, slow, sig_p in [(8, 17, 9), (12, 26, 9), (5, 34, 5)]:
            for trend_p in [50, 100, 200]:
                for sl_m, tp_m in [(2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                    sig_aug = sl.strat_macd_trend(df_aug, fast, slow, sig_p, trend_p)
                    trades_aug, m_aug = engine.run_backtest(df_aug, sig_aug, atr_aug, sl_m, tp_m)
                    
                    if m_aug['net_pnl'] > 0 and m_aug['total_trades'] >= 5:
                        sig_jul = sl.strat_macd_trend(df_jul, fast, slow, sig_p, trend_p)
                        trades_jul, m_jul = engine.run_backtest(df_jul, sig_jul, atr_jul, sl_m, tp_m)
                        
                        candidates.append({
                            'name': 'MACD + Trend EMA',
                            'tf': tf,
                            'params': {'fast': fast, 'slow': slow, 'sig_p': sig_p, 'trend_p': trend_p, 'sl_m': sl_m, 'tp_m': tp_m},
                            'aug_metrics': m_aug,
                            'jul_metrics': m_jul,
                            'aug_trades': trades_aug,
                            'combined_pnl': m_aug['net_pnl'] + m_jul['net_pnl']
                        })

    # 3. EMA Ribbon Trend Following with ADX
    for tf in ['1h', '30m', '15m']:
        df_aug = dfs_aug[tf]
        df_jul = dfs_jul[tf]
        atr_aug = sl.compute_atr(df_aug).values
        atr_jul = sl.compute_atr(df_jul).values
        
        for fast in [10, 15, 20]:
            for slow in [50, 80, 100]:
                for adx_t in [20.0, 25.0, 30.0]:
                    for sl_m, tp_m in [(2.0, 4.0), (2.5, 4.5), (3.0, 5.0)]:
                        sig_aug = sl.strat_ema_adx(df_aug, fast, slow, 14, adx_t)
                        trades_aug, m_aug = engine.run_backtest(df_aug, sig_aug, atr_aug, sl_m, tp_m)
                        
                        if m_aug['net_pnl'] > 0 and m_aug['total_trades'] >= 4:
                            sig_jul = sl.strat_ema_adx(df_jul, fast, slow, 14, adx_t)
                            trades_jul, m_jul = engine.run_backtest(df_jul, sig_jul, atr_jul, sl_m, tp_m)
                            
                            candidates.append({
                                'name': 'EMA + ADX Strong Trend',
                                'tf': tf,
                                'params': {'fast': fast, 'slow': slow, 'adx_t': adx_t, 'sl_m': sl_m, 'tp_m': tp_m},
                                'aug_metrics': m_aug,
                                'jul_metrics': m_jul,
                                'aug_trades': trades_aug,
                                'combined_pnl': m_aug['net_pnl'] + m_jul['net_pnl']
                            })

    print(f"Total profitable candidate configurations for August: {len(candidates)}")
    
    # Sort by August Net PnL and consistency with July
    candidates.sort(key=lambda x: (
        x['aug_metrics']['net_pnl'] * (1.5 if x['jul_metrics']['net_pnl'] > 0 else 0.8)
    ), reverse=True)

    print("\n" + "="*90)
    print("                 TOP STRATEGIES PROFITABLE IN AUGUST 2026                              ")
    print("="*90)
    for i, c in enumerate(candidates[:10]):
        ma = c['aug_metrics']
        mj = c['jul_metrics']
        print(f"\n#{i+1:2d} | {c['name']:30s} | TF: {c['tf']:3s} | Params: {c['params']}")
        print(f"     [AUGUST 2026] Trades: {ma['total_trades']:2d} | WR: {ma['win_rate']:5.1f}% | PF: {ma['profit_factor']:5.2f} | Net: ${ma['net_pnl']:+8.2f} | DD: {ma['max_drawdown_pct']:4.2f}% | Sharpe: {ma['sharpe_ratio']:5.2f}")
        print(f"     [JULY 2026]   Trades: {mj['total_trades']:2d} | WR: {mj['win_rate']:5.1f}% | PF: {mj['profit_factor']:5.2f} | Net: ${mj['net_pnl']:+8.2f} | DD: {mj['max_drawdown_pct']:4.2f}% | Sharpe: {mj['sharpe_ratio']:5.2f}")
        print(f"     Combined 2-Month PnL: ${c['combined_pnl']:+,.2f}")

    # Output detailed trades of #1
    if candidates:
        top1 = candidates[0]
        print("\n" + "="*90)
        print(f"             TRADE LOG FOR TOP CANDIDATE IN AUGUST 2026: {top1['name']} ({top1['tf']})")
        print("="*90)
        header = f"{'#':<3} | {'Entry Time':<19} | {'Entry Px':<9} | {'Exit Time':<19} | {'Exit Px':<9} | {'Dir':<5} | {'Size':<6} | {'Gross PnL':<10} | {'Fee':<7} | {'Net PnL':<10} | {'Exit Reason'}"
        print(header)
        print("-" * len(header))
        for i, t in enumerate(top1['aug_trades']):
            print(f"{i+1:<3} | {str(t.entry_time)[:19]:<19} | ${t.entry_price:<8.2f} | {str(t.exit_time)[:19]:<19} | ${t.exit_price:<8.2f} | {'LONG' if t.direction==1 else 'SHORT':<5} | {t.size:<6.3f} | ${t.gross_pnl:<9.2f} | ${t.fee:<6.2f} | ${t.net_pnl:<9.2f} | {t.exit_reason}")

if __name__ == "__main__":
    search_august()
