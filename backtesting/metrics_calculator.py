import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

class MetricsCalculator:
    """Comprehensive performance metrics calculator for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        
    def calculate_all_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates all performance metrics for a given DataFrame of trades."""
        if trades_df.empty:
            return {
                'Total Trades': 0,
                'Win Rate': 0.0,
                'Profit Factor': 0.0,
                'Max Drawdown (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Expectancy': 0.0,
                'Net PnL': 0.0,
                'Return (%)': 0.0,
                'Avg Holding Period (h)': 0.0
            }
            
        equity_curve = self.build_equity_curve(trades_df, self.initial_capital)
        total_pnl = trades_df['net_pnl'].sum() if 'net_pnl' in trades_df.columns else 0.0
        return_pct = (total_pnl / self.initial_capital) * 100.0
        
        # Determine time covered for annualized metrics
        if 'exit_time' in trades_df.columns and len(trades_df) > 0:
            start_date = pd.to_datetime(trades_df['entry_time'].iloc[0])
            end_date = pd.to_datetime(trades_df['exit_time'].iloc[-1])
            days = max((end_date - start_date).days, 1)
            years = days / 365.25
        else:
            years = 1.0
            
        annual_return = return_pct / years if years > 0 else return_pct
        
        max_dd_abs, max_dd_pct = self.max_drawdown(equity_curve)
        win_rate = self.win_rate(trades_df)
        profit_factor = self.profit_factor(trades_df)
        
        # Calculate returns series for Sharpe/Sortino
        # Using trade-by-trade returns as percentage of capital at that moment
        returns = equity_curve.pct_change().dropna()
        
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        expectancy = self.expectancy(trades_df)
        calmar = self.calmar_ratio(annual_return, max_dd_pct)
        recovery = self.recovery_factor(total_pnl, max_dd_abs)
        max_wins, max_losses = self.consecutive_wins_losses(trades_df)
        avg_holding = self.avg_holding_period(trades_df)
        
        return {
            'Total Trades': self.total_trades(trades_df),
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Max Drawdown ($)': max_dd_abs,
            'Max Drawdown (%)': max_dd_pct,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Expectancy': expectancy,
            'Calmar Ratio': calmar,
            'Recovery Factor': recovery,
            'Max Consecutive Wins': max_wins,
            'Max Consecutive Losses': max_losses,
            'Avg Holding Period (h)': avg_holding,
            'Net PnL': total_pnl,
            'Return (%)': return_pct,
            'Annual Return (%)': annual_return
        }

    def profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor (Gross Profit / Gross Loss)."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0.0
            
        gross_profit = trades[trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(trades[trades['net_pnl'] < 0]['net_pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return float(gross_profit / gross_loss)

    def max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, float]:
        """Calculate maximum absolute and percentage drawdown."""
        if equity_curve.empty:
            return 0.0, 0.0
            
        rolling_max = equity_curve.cummax()
        drawdown_abs = rolling_max - equity_curve
        
        # Guard against zero rolling max to avoid division by zero
        drawdown_pct = np.where(rolling_max > 0, (drawdown_abs / rolling_max) * 100, 0.0)
        
        max_dd_abs = float(drawdown_abs.max())
        max_dd_pct = float(np.max(drawdown_pct))
        
        return max_dd_abs, max_dd_pct

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe Ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())
        return float(sharpe)

    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate annualized Sortino Ratio (downside risk only)."""
        if returns.empty:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0
            
        sortino = np.sqrt(periods_per_year) * (excess_returns.mean() / downside_returns.std())
        return float(sortino)

    def expectancy(self, trades: pd.DataFrame) -> float:
        """Calculate expectancy per trade."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0.0
            
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        loss_rate = 1.0 - win_rate
        
        avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0.0
        avg_loss = abs(losing_trades['net_pnl'].mean()) if not losing_trades.empty else 0.0
        
        return float((win_rate * avg_win) - (loss_rate * avg_loss))

    def build_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
        """Build cumulative equity curve from sequential trade PnLs."""
        if trades_df.empty or 'net_pnl' not in trades_df.columns:
            return pd.Series([initial_capital])
            
        pnl_series = trades_df['net_pnl'].copy()
        
        # Use exit time as index for the equity curve if available
        if 'exit_time' in trades_df.columns:
            pnl_series.index = pd.to_datetime(trades_df['exit_time'])
            
        equity_curve = initial_capital + pnl_series.cumsum()
        
        # Add initial capital point
        if 'entry_time' in trades_df.columns and len(trades_df) > 0:
            start_time = pd.to_datetime(trades_df['entry_time'].iloc[0]) - pd.Timedelta(minutes=1)
            initial_point = pd.Series([initial_capital], index=[start_time])
            equity_curve = pd.concat([initial_point, equity_curve])
            
        return equity_curve

    def calmar_ratio(self, annual_return: float, max_dd_pct: float) -> float:
        """Calculate Calmar Ratio."""
        if max_dd_pct == 0:
            return float('inf') if annual_return > 0 else 0.0
        return float(annual_return / max_dd_pct)

    def recovery_factor(self, total_profit: float, max_dd_abs: float) -> float:
        """Calculate Recovery Factor."""
        if max_dd_abs == 0:
            return float('inf') if total_profit > 0 else 0.0
        return float(total_profit / max_dd_abs)

    def consecutive_wins_losses(self, trades: pd.DataFrame) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0, 0
            
        is_win = trades['net_pnl'] > 0
        
        # Group by consecutive identical values
        blocks = (is_win != is_win.shift()).cumsum()
        
        wins = []
        losses = []
        for _, group in is_win.groupby(blocks):
            if group.iloc[0]:
                wins.append(len(group))
            else:
                losses.append(len(group))
        
        max_wins = max(wins) if wins else 0
        max_losses = max(losses) if losses else 0
        
        return max_wins, max_losses

    def monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns from equity curve."""
        if equity_curve.empty or len(equity_curve) < 2:
            return pd.DataFrame()
            
        # Ensure index is datetime
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            return pd.DataFrame()
            
        # Resample to monthly end and calculate pct change
        monthly_equity = equity_curve.resample('M').last().dropna()
        if len(monthly_equity) < 2:
            return pd.DataFrame()
            
        monthly_ret = monthly_equity.pct_change().dropna() * 100
        
        # Format into a pivot table (Year x Month)
        df = pd.DataFrame({'Return': monthly_ret})
        df['Year'] = df.index.year
        df['Month'] = df.index.strftime('%b')
        
        pivot = df.pivot(index='Year', columns='Month', values='Return')
        
        # Reorder columns to standard month order
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        available_months = [m for m in months if m in pivot.columns]
        
        return pivot[available_months]

    def win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate as a float between 0 and 1."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0.0
            
        winning_trades = len(trades[trades['net_pnl'] > 0])
        return float(winning_trades / len(trades))

    def avg_win_loss_ratio(self, trades: pd.DataFrame) -> float:
        """Calculate ratio of average win to average loss."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return 0.0
            
        winning_trades = trades[trades['net_pnl'] > 0]
        losing_trades = trades[trades['net_pnl'] <= 0]
        
        avg_win = winning_trades['net_pnl'].mean() if not winning_trades.empty else 0.0
        avg_loss = abs(losing_trades['net_pnl'].mean()) if not losing_trades.empty else 0.0
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
            
        return float(avg_win / avg_loss)

    def total_trades(self, trades: pd.DataFrame) -> int:
        """Return total number of trades."""
        return len(trades)

    def avg_holding_period(self, trades: pd.DataFrame) -> float:
        """Calculate average holding period in hours."""
        if trades.empty:
            return 0.0
            
        if 'holding_duration_hours' in trades.columns:
            return float(trades['holding_duration_hours'].mean())
            
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            durations = pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])
            return float(durations.dt.total_seconds().mean() / 3600.0)
            
        return 0.0

    def trade_stability(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate various trade stability metrics."""
        if trades.empty or 'net_pnl' not in trades.columns:
            return {}
            
        win_rate = self.win_rate(trades)
        pf = self.profit_factor(trades)
        awl = self.avg_win_loss_ratio(trades)
        
        stability_score = win_rate * pf * awl
        
        # PnL standard deviation
        pnl_std = float(trades['net_pnl'].std()) if len(trades) > 1 else 0.0
        
        return {
            'Stability Score': stability_score,
            'PnL Std Dev': pnl_std
        }
