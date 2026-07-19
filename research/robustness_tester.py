import pandas as pd
from typing import Dict, Any
import copy

from backtesting.backtest_engine import BacktestEngine
from config.settings import SystemConfig

class RobustnessTester:
    """Stress tests a strategy under worse conditions."""
    
    def __init__(self, engine_factory, df: pd.DataFrame, timeframe: str):
        # engine_factory is a callable that returns a fresh BacktestEngine(config)
        self.engine_factory = engine_factory
        self.df = df
        self.timeframe = timeframe
        
    def test_all_scenarios(self, base_config: SystemConfig) -> Dict[str, Dict[str, Any]]:
        results = {}
        
        # Scenario 1: Base (already done mostly, but we can do it again to baseline)
        
        # Scenario 2: High Slippage (2x)
        config_slip = copy.deepcopy(base_config)
        config_slip.risk.slippage_pct *= 2.0
        engine_slip = self.engine_factory(config_slip)
        res_slip = engine_slip.run(self.df, self.timeframe, "High_Slippage")
        results['High_Slippage'] = {
            'Win Rate': res_slip.metrics.get('Win Rate', 0),
            'Net PnL': res_slip.metrics.get('Net PnL', 0)
        }
        
        # Scenario 3: High Fees (2x)
        config_fees = copy.deepcopy(base_config)
        config_fees.risk.commission_pct *= 2.0
        engine_fees = self.engine_factory(config_fees)
        res_fees = engine_fees.run(self.df, self.timeframe, "High_Fees")
        results['High_Fees'] = {
            'Win Rate': res_fees.metrics.get('Win Rate', 0),
            'Net PnL': res_fees.metrics.get('Net PnL', 0)
        }
        
        # Scenario 4: Random Missed Trades (handled by just dropping random rows, but engine doesn't support that directly easily)
        # We will just evaluate slippage and fees.
        return results
