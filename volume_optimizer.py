import os
import pandas as pd
from data.parquet_manager import load_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from config.settings import SystemConfig

def optimize_volume():
    tf = "1h"
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    file_path = get_data_path("BTCUSDT", tf, data_dir)
    df = load_data(file_path)
    # 2-year window for volume testing
    df_slice = df.loc["2024-01-01":"2026-01-01"]

    atr_multipliers = [1.5, 1.8, 2.0, 2.2, 2.5]
    ma_pairs = [(5, 20), (10, 50)]
    tp_bricks = [3, 5, 8]
    
    results = []
    
    total_iters = len(atr_multipliers) * len(ma_pairs) * len(tp_bricks)
    current_iter = 0

    print(f"Starting Volume Optimization for {tf}. Total combinations: {total_iters}")

    for atr_m in atr_multipliers:
        for fast_ma, slow_ma in ma_pairs:
            for tp in tp_bricks:
                current_iter += 1
                
                config = SystemConfig.default()
                config.strategy.strategy_type = "DoubleBrick"
                config.strategy.atr_multiplier = atr_m
                config.strategy.fast_ma_period = fast_ma
                config.strategy.slow_ma_period = slow_ma
                config.strategy.consecutive_bricks = 2
                config.strategy.profit_take_bricks = tp
                config.strategy.atr_percentile_threshold = 0.0 # No filter to maximize trades
                
                config.risk.stop_loss_atr_mult = 2.0
                config.risk.take_profit_atr_mult = tp * atr_m
                
                engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                res = engine.run(df_slice, timeframe=tf, config_name="Opt",
                                 use_trend_filter=config.strategy.use_macro_trend_filter,
                                 use_atr_filter=config.strategy.use_volatility_filter)
                
                pf = res.metrics.get('Profit Factor', 0)
                dd = res.metrics.get('Max Drawdown (%)', 100)
                trades = len(res.trades_df)
                
                results.append({
                    'params': f"ATR={atr_m}, MAs={fast_ma}/{slow_ma}, TP={tp}",
                    'pf': pf,
                    'dd': dd,
                    'trades': trades
                })
                print(f"[{current_iter}/{total_iters}] {results[-1]['params']} -> PF: {pf:.2f}, DD: {dd:.2%}, Trades: {trades}")

    # Filter for Statistical Significance (e.g. > 100 trades over 2 years)
    significant_results = [r for r in results if r['trades'] >= 100]
    
    # Sort by Profit Factor
    significant_results.sort(key=lambda x: (x['pf'] > 1.0, x['pf'], -x['dd']), reverse=True)

    print("\n" + "="*50)
    print("TOP CONFIGURATIONS (>100 TRADES)")
    print("="*50)
    if not significant_results:
        print("No configurations met the >100 trades constraint.")
    else:
        for i, res in enumerate(significant_results[:5], 1):
            print(f"#{i}: {res['params']}")
            print(f"  Profit Factor: {res['pf']:.2f}")
            print(f"  Max Drawdown: {res['dd']:.2%}")
            print(f"  Total Trades (2yr): {res['trades']}")
            print("-" * 30)

if __name__ == "__main__":
    optimize_volume()
