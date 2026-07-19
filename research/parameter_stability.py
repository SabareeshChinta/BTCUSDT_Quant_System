import pandas as pd
from typing import List, Dict
import copy

from backtesting.backtest_engine import BacktestEngine
from config.settings import SystemConfig

class ParameterStabilityTester:
    """Runs a grid search to test for parameter region stability."""
    
    def __init__(self, df: pd.DataFrame, timeframe: str):
        self.df = df
        self.timeframe = timeframe
        
    def test_grid(self, base_config: SystemConfig) -> List[Dict]:
        results = []
        
        atr_periods = [10, 14, 20]
        fast_emas = [10, 20, 30]
        
        for atr in atr_periods:
            for ema in fast_emas:
                config = copy.deepcopy(base_config)
                config.strategy.atr_period = atr
                config.strategy.fast_ema_period = ema
                
                engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                res = engine.run(self.df, self.timeframe, f"Grid_{atr}_{ema}")
                
                results.append({
                    'atr_period': atr,
                    'fast_ema': ema,
                    'trades': len(res.trades_df),
                    'win_rate': res.metrics.get('Win Rate', 0),
                    'net_pnl': res.metrics.get('Net PnL', 0)
                })
                
        return results
