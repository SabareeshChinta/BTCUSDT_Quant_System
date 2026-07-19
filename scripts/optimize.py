import os
import sys
import pandas as pd
import itertools

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from data.parquet_manager import load_data, get_data_path

def run_optimization():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    file_path = get_data_path("BTCUSDT", "1h", data_dir)
    try:
        df = load_data(file_path)
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    # Remove the date cutoff to run on the full 2019-2025 dataset
    # if 'Open Time' in df.columns:
    #     df = df[df['Open Time'] < '2021-01-01']
    # else:
    #     df = df[df.index < '2021-01-01']
        
    print(f"Running optimization on {len(df)} candles...")

    # Parameters to test (expanded for EMA tuning)
    atr_multipliers = [1.5, 2.0, 3.0]
    fast_mas = [10, 20]
    slow_mas = [50, 100]
    sl_mults = [1.5, 2.0]
    tp_mults = [3.0, 4.5]

    best_pnl = -float('inf')
    best_params = None
    results = []

    for atr_m, fast_ma, slow_ma, sl_m, tp_m in itertools.product(atr_multipliers, fast_mas, slow_mas, sl_mults, tp_mults):
        config = SystemConfig.default()
        config.strategy.atr_multiplier = atr_m
        config.strategy.fast_ma_period = fast_ma
        config.strategy.slow_ma_period = slow_ma
        config.risk.stop_loss_atr_mult = sl_m
        config.risk.take_profit_atr_mult = tp_m
        
        # Disable halts for optimization so we can see full performance
        config.risk.hard_drawdown_halt_pct = 1.0 
        
        engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
        res = engine.run(df, timeframe='1h', config_name="Opt")
        
        pnl = res.metrics.get('Net PnL', 0)
        win_rate = res.metrics.get('Win Rate', 0)
        drawdown = res.metrics.get('Max Drawdown (%)', 0)
        trades = len(res.trades_df)
        
        print(f"Params (ATR:{atr_m}, Fast:{fast_ma}, Slow:{slow_ma}, SL:{sl_m}, TP:{tp_m}) -> PnL: {pnl:.2f}, Win Rate: {win_rate:.2%}, DD: {drawdown:.2%}, Trades: {trades}")
        
        results.append({
            'atr_mult': atr_m,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'sl_mult': sl_m,
            'tp_mult': tp_m,
            'pnl': pnl,
            'win_rate': win_rate,
            'drawdown': drawdown,
            'trades': trades
        })

        if pnl > best_pnl and trades > 10: # ensure enough trades
            best_pnl = pnl
            best_params = (atr_m, fast_ma, slow_ma, sl_m, tp_m)

    print("\n=== Optimization Complete ===")
    if best_params:
        print(f"Best Params: ATR Mult={best_params[0]}, Fast MA={best_params[1]}, Slow MA={best_params[2]}, SL Mult={best_params[3]}, TP Mult={best_params[4]}")
        print(f"Best PnL: {best_pnl:.2f}")
    else:
        print("No profitable parameters found.")

if __name__ == "__main__":
    run_optimization()
