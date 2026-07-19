import pandas as pd
from typing import Dict, Any

from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine, BacktestResult

class ForwardTestEngine:
    """Forward testing on unseen data."""
    
    def __init__(self, config: SystemConfig):
        self.config = config

    def run(self, backtest_data: pd.DataFrame, forward_data: pd.DataFrame, timeframe: str = '1h', ml_predictor=None) -> Dict[str, BacktestResult]:
        """
        Run all 4 ablation configs on forward_data.
        CRITICAL: ML model must be the one trained on backtest_data ONLY.
        No retraining on forward data.
        """
        results = {}
        
        # S1: Baseline (No RM, No ML)
        config_s1 = SystemConfig(**self.config.__dict__)
        config_s1.use_risk_management = False
        config_s1.use_ml_filter = False
        engine_s1 = BacktestEngine(config_s1)
        results['S1_Baseline'] = engine_s1.run(forward_data, timeframe=timeframe)
        
        # S2: Baseline + RM
        config_s2 = SystemConfig(**self.config.__dict__)
        config_s2.use_risk_management = True
        config_s2.use_ml_filter = False
        engine_s2 = BacktestEngine(config_s2)
        results['S2_RM'] = engine_s2.run(forward_data, timeframe=timeframe)
        
        # S3: Baseline + RM + ML
        config_s3 = SystemConfig(**self.config.__dict__)
        config_s3.use_risk_management = True
        config_s3.use_ml_filter = True
        engine_s3 = BacktestEngine(config_s3)
        results['S3_RM_ML'] = engine_s3.run(forward_data, timeframe=timeframe, ml_predictor=ml_predictor)
        
        # S4: Baseline + RM + ML + Kelly
        config_s4 = SystemConfig(**self.config.__dict__)
        config_s4.use_risk_management = True
        config_s4.use_ml_filter = True
        config_s4.use_kelly_sizing = True
        engine_s4 = BacktestEngine(config_s4)
        results['S4_Full'] = engine_s4.run(forward_data, timeframe=timeframe, ml_predictor=ml_predictor)
        
        return results
