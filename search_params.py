import pandas as pd
import os
import json
from data.parquet_manager import load_data, get_data_path, split_data
from backtesting.backtest_engine import BacktestEngine
from config.settings import SystemConfig

def grid_search():
    timeframe = "15m"
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    file_path = get_data_path("BTCUSDT", timeframe, data_dir)
    df = load_data(file_path)
    
    config = SystemConfig.default()
    df_backtest, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    best_pf = 0
    best_params = {}
    
    # Grid of parameters to test
    atr_multipliers = [1.5, 2.0, 2.5]
    fast_mas = [10, 20]
    slow_mas = [50, 100]
    percentiles = [0.3, 0.5, 0.6]
    
    results = []
    
    for atr_m in atr_multipliers:
        for fast_ma in fast_mas:
            for slow_ma in slow_mas:
                for pct in percentiles:
                    print(f"Testing ATR={atr_m}, FastMA={fast_ma}, SlowMA={slow_ma}, Pct={pct}...")
                    
                    config = SystemConfig.default()
                    config.strategy.atr_multiplier = atr_m
                    config.strategy.fast_ma_period = fast_ma
                    config.strategy.slow_ma_period = slow_ma
                    config.strategy.atr_percentile_threshold = pct
                    
                    engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                    res = engine.run(df_backtest, timeframe=timeframe, config_name="GridSearch",
                                     use_trend_filter=config.strategy.use_macro_trend_filter,
                                     use_atr_filter=config.strategy.use_volatility_filter)
                    
                    metrics = res.metrics
                    pf = metrics.get('Profit Factor', 0)
                    trades = len(res.trades_df)
                    dd = metrics.get('Max Drawdown (%)', 100)
                    
                    print(f"  -> Trades: {trades}, Profit Factor: {pf:.2f}, Max Drawdown: {dd:.2%}")
                    
                    results.append({
                        'atr_multiplier': atr_m,
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma,
                        'percentile': pct,
                        'trades': trades,
                        'profit_factor': pf,
                        'max_drawdown': dd
                    })
                    
                    if pf > best_pf and trades > 30 and dd < 0.20:
                        best_pf = pf
                        best_params = results[-1]

    print("\nBest Parameters found:")
    print(json.dumps(best_params, indent=4))
    
    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    grid_search()
