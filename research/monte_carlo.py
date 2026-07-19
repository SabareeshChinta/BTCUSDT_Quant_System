import numpy as np
import pandas as pd
from typing import Dict, Tuple

class MonteCarloSimulator:
    """Monte Carlo simulation for trading results and SQN calculation."""
    
    def __init__(self, trades_df: pd.DataFrame, num_simulations: int = 1000):
        self.trades = trades_df
        self.n_sims = num_simulations
        
    def calculate_sqn(self) -> float:
        if self.trades.empty or len(self.trades) < 2:
            return 0.0
            
        if 'r_multiple' in self.trades.columns and not self.trades['r_multiple'].isna().all():
            r = self.trades['r_multiple']
        else:
            # Fallback to PnL
            r = self.trades['net_pnl']
            
        mean = r.mean()
        std = r.std()
        
        if std == 0:
            return 0.0
            
        return np.sqrt(len(self.trades)) * (mean / std)
        
    def run_simulations(self) -> Dict[str, float]:
        if self.trades.empty:
            return {}
            
        pnl = self.trades['net_pnl'].values
        
        final_profits = []
        max_drawdowns = []
        
        for _ in range(self.n_sims):
            # Resample with replacement
            sim_pnl = np.random.choice(pnl, size=len(pnl), replace=True)
            
            # Cumulative equity
            equity_curve = np.cumsum(sim_pnl)
            
            final_profits.append(equity_curve[-1])
            
            # Max DD
            rolling_max = np.maximum.accumulate(equity_curve)
            drawdowns = rolling_max - equity_curve
            max_drawdowns.append(np.max(drawdowns))
            
        return {
            'mc_profit_5th': float(np.percentile(final_profits, 5)),
            'mc_profit_50th': float(np.percentile(final_profits, 50)),
            'mc_profit_95th': float(np.percentile(final_profits, 95)),
            'mc_dd_50th': float(np.percentile(max_drawdowns, 50)),
            'mc_dd_95th': float(np.percentile(max_drawdowns, 95))
        }
