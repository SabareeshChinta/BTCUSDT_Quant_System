import os
import pandas as pd
from data.parquet_manager import load_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from config.settings import SystemConfig

def optimize_multi_tf():
    timeframes = ["5m", "15m", "30m"]
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    
    dfs = {}
    for tf in timeframes:
        file_path = get_data_path("BTCUSDT", tf, data_dir)
        df = load_data(file_path)
        # Filter to a 2-year window for more trades (2024-01-01 to 2026-01-01)
        df_slice = df.loc["2024-01-01":"2026-01-01"]
        dfs[tf] = df_slice

    atr_multipliers = [1.5, 2.0, 3.0]
    ma_pairs = [(10, 30), (10, 50)]
    tp_bricks = [3, 5]
    rsi_params = [(65.0, 35.0), (70.0, 30.0)]
    
    results = []

    total_iters = len(atr_multipliers) * len(ma_pairs) * len(tp_bricks) * len(rsi_params)
    current_iter = 0

    print(f"Starting Grid Search. Total combinations: {total_iters}")

    for atr_m in atr_multipliers:
        for fast_ma, slow_ma in ma_pairs:
            for tp in tp_bricks:
                for rsi_b, rsi_s in rsi_params:
                    current_iter += 1
                    
                    config = SystemConfig.default()
                    config.strategy.atr_multiplier = atr_m
                    config.strategy.fast_ma_period = fast_ma
                    config.strategy.slow_ma_period = slow_ma
                    config.strategy.atr_percentile_threshold = 0.0 # Disable to rely on RSI
                    config.strategy.profit_take_bricks = tp
                    config.strategy.use_rsi_filter = True
                    config.strategy.rsi_buy_max = rsi_b
                    config.strategy.rsi_sell_min = rsi_s
                    
                    config.risk.stop_loss_atr_mult = 2.0
                    config.risk.take_profit_atr_mult = tp * atr_m
                    
                    tf_metrics = {}
                    for tf in timeframes:
                        engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                        res = engine.run(dfs[tf], timeframe=tf, config_name="Opt",
                                         use_trend_filter=config.strategy.use_macro_trend_filter,
                                         use_atr_filter=config.strategy.use_volatility_filter)
                        
                        pf = res.metrics.get('Profit Factor', 0)
                        dd = res.metrics.get('Max Drawdown (%)', 100)
                        trades = len(res.trades_df)
                        
                        tf_metrics[tf] = {'pf': pf, 'dd': dd, 'trades': trades}
                    
                    avg_pf = sum(m['pf'] for m in tf_metrics.values()) / len(timeframes)
                    min_pf = min(m['pf'] for m in tf_metrics.values())
                    avg_trades = sum(m['trades'] for m in tf_metrics.values()) / len(timeframes)
                    max_dd = max(m['dd'] for m in tf_metrics.values())
                    
                    # Store everything, we'll sort later
                    results.append({
                        'params': f"ATR={atr_m}, MAs={fast_ma}/{slow_ma}, RSI={rsi_b}/{rsi_s}, TP_Bricks={tp}",
                        'avg_pf': avg_pf,
                        'min_pf': min_pf,
                        'max_dd': max_dd,
                        'avg_trades': avg_trades,
                        'tf_details': tf_metrics
                    })
                    print(f"[{current_iter}/{total_iters}] {results[-1]['params']} -> Min PF: {min_pf:.2f}, Max DD: {max_dd:.2%}")

    # Sort results by min_pf across all timeframes (best worst-case scenario), then penalize extreme drawdowns
    # We want max min_pf, but we also want max_dd to be reasonable
    results.sort(key=lambda x: (x['min_pf'] > 1.0, -x['max_dd'], x['avg_trades']), reverse=True)

    print("\n" + "="*50)
    print("TOP 3 CONFIGURATIONS (SCALE-INVARIANT)")
    print("="*50)
    for i, res in enumerate(results[:3], 1):
        print(f"#{i}: {res['params']}")
        print(f"  Overall Min PF: {res['min_pf']:.2f}")
        print(f"  Overall Avg PF: {res['avg_pf']:.2f}")
        print(f"  Overall Avg Trades (1yr): {res['avg_trades']:.1f}")
        print("  Breakdown:")
        for tf, metrics in res['tf_details'].items():
            print(f"    - {tf}: PF={metrics['pf']:.2f}, DD={metrics['dd']:.2%}, Trades={metrics['trades']}")
        print("-" * 30)

if __name__ == "__main__":
    optimize_multi_tf()
