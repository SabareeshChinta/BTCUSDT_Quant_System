import os
import sys
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.backtest_engine import BacktestEngine

def evaluate_params(args):
    atr_period, atr_mult, fast_ma, slow_ma = args
    timeframes = ["15m", "30m", "1h", "4h"]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    
    results = {}
    total_pnl = 0
    all_profitable = True
    
    for tf in timeframes:
        file_path = get_data_path("BTCUSDT", tf, data_dir)
        try:
            df = load_data(file_path)
            # Use only backtest period for speed
            config = SystemConfig.default()
            df_bt, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
            
            # Apply our double brick settings
            config.strategy.strategy_type = "DoubleBrick"
            config.strategy.consecutive_bricks = 2
            config.strategy.atr_period = atr_period
            config.strategy.atr_multiplier = atr_mult
            config.strategy.fast_ma_period = fast_ma
            config.strategy.slow_ma_period = slow_ma
            config.strategy.use_macro_trend_filter = False # disable time-based trend filter
            
            # Run
            engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
            res = engine.run(df_bt, timeframe=tf, config_name="Opt")
            
            pnl = res.metrics.get('Net PnL', 0)
            if pnl <= 0:
                all_profitable = False
            total_pnl += pnl
            results[tf] = pnl
        except Exception as e:
            return None
            
    if all_profitable:
        return (args, total_pnl, results)
    return None

def main():
    atr_periods = [10, 14, 20]
    atr_mults = [1.5, 2.0, 3.0]
    fast_mas = [10, 20, 50]
    slow_mas = [50, 100, 200]
    
    param_grid = []
    for p in itertools.product(atr_periods, atr_mults, fast_mas, slow_mas):
        if p[2] < p[3]: # fast_ma < slow_ma
            param_grid.append(p)
            
    print(f"Total combinations to test: {len(param_grid)}")
    
    best_args = None
    best_pnl = -1
    
    with ProcessPoolExecutor() as executor:
        for result in executor.map(evaluate_params, param_grid):
            if result is not None:
                args, total_pnl, tf_results = result
                print(f"Found profitable combination! {args} -> Total PnL: {total_pnl:.2f}, Breakdown: {tf_results}")
                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    best_args = args
                    
    print(f"\nBest parameters: {best_args} with total PnL {best_pnl}")

if __name__ == '__main__':
    main()
