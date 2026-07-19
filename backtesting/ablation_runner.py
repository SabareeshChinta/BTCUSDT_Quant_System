"""
Ablation Runner for testing combinations of Risk Management and ML layers.
Runs S1, S2, S3, S4 sequentially and compares them.
"""

import os
import json
import pandas as pd
from typing import Dict

from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine, BacktestResult

class AblationRunner:
    def __init__(self, config: SystemConfig):
        self.config = config

    def run_all(self, df: pd.DataFrame, timeframe: str = '1h', ml_predictor=None) -> Dict[str, BacktestResult]:
        results = {}
        
        for variant in self.config.ablation.variants:
            print(f"Running Ablation {variant.name} (RM: {variant.use_risk_management}, ML: {variant.use_ml_filter})")
            
            engine = BacktestEngine(
                config=self.config,
                use_risk_management=variant.use_risk_management,
                use_ml_filter=variant.use_ml_filter,
                ml_predictor=ml_predictor if variant.use_ml_filter else None
            )
            
            res = engine.run(df, timeframe=timeframe, config_name=variant.name)
            results[variant.name] = res
            
        return results

    def compare_results(self, results: Dict[str, BacktestResult]) -> pd.DataFrame:
        data = []
        for name, res in results.items():
            m = res.metrics
            data.append({
                'Config': name,
                'Total Trades': len(res.trades_df),
                'Win Rate': m.get('Win Rate', 0.0),
                'Profit Factor': m.get('Profit Factor', 0.0),
                'Max Drawdown': m.get('Max Drawdown (%)', 0.0),
                'Sharpe': m.get('Sharpe Ratio', 0.0),
                'Sortino': m.get('Sortino Ratio', 0.0),
                'Expectancy': m.get('Expectancy', 0.0),
                'Net PnL': m.get('Net PnL', 0.0),
                'Avg Holding Period': m.get('Avg Holding Period (h)', 0.0)
            })
            
        return pd.DataFrame(data)

    def save_results(self, results: Dict[str, BacktestResult], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison
        comp_df = self.compare_results(results)
        comp_df.to_csv(os.path.join(output_dir, 'ablation_comparison.csv'), index=False)
        
        for name, res in results.items():
            config_dir = os.path.join(output_dir, name)
            os.makedirs(config_dir, exist_ok=True)
            
            # Trades
            res.trades_df.to_csv(os.path.join(config_dir, 'trades.csv'), index=False)
            
            # Metrics
            # Convert np and pd types to native python for JSON serialization
            metrics_clean = {}
            for k, v in res.metrics.items():
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    continue # Skip dataframes in json
                if pd.isna(v) or v == float('inf') or v == float('-inf'):
                    metrics_clean[k] = None
                else:
                    metrics_clean[k] = float(v) if isinstance(v, float) else v
            
            with open(os.path.join(config_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_clean, f, indent=4)
