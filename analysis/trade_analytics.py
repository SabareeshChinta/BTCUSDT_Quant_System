import pandas as pd
import numpy as np

class TradeAnalytics:
    """
    Advanced trade analysis including MAE, MFE, holding durations,
    exit reasons, and streaks.
    """
    def __init__(self, trades_df: pd.DataFrame):
        self.trades_df = trades_df.copy()

    def mae_analysis(self) -> dict:
        """Maximum Adverse Excursion analysis."""
        if self.trades_df.empty or 'mae' not in self.trades_df.columns:
            return {}
        
        mae = self.trades_df['mae'].dropna()
        if mae.empty:
            return {}

        winners = self.trades_df[self.trades_df['pnl'] > 0]['mae'].dropna()
        losers = self.trades_df[self.trades_df['pnl'] <= 0]['mae'].dropna()

        correlation = self.trades_df['mae'].corr(self.trades_df['pnl']) if len(self.trades_df) > 1 else np.nan

        return {
            'avg_mae': mae.mean(),
            'median_mae': mae.median(),
            'max_mae': mae.max(),
            'mae_by_outcome': {
                'winners_avg_mae': winners.mean() if not winners.empty else 0.0,
                'losers_avg_mae': losers.mean() if not losers.empty else 0.0,
            },
            'correlation_mae_pnl': correlation
        }

    def mfe_analysis(self) -> dict:
        """Maximum Favorable Excursion analysis."""
        if self.trades_df.empty or 'mfe' not in self.trades_df.columns:
            return {}
            
        mfe = self.trades_df['mfe'].dropna()
        if mfe.empty:
            return {}

        profit_left = (self.trades_df['mfe'] - self.trades_df['pnl']).clip(lower=0)
        
        mfe_by_exit = {}
        if 'exit_reason' in self.trades_df.columns:
            mfe_by_exit = self.trades_df.groupby('exit_reason')['mfe'].mean().to_dict()

        return {
            'avg_mfe': mfe.mean(),
            'median_mfe': mfe.median(),
            'max_mfe': mfe.max(),
            'avg_profit_left_on_table': profit_left.mean(),
            'mfe_by_exit_reason': mfe_by_exit
        }

    def holding_duration_analysis(self) -> dict:
        """Analysis of trade holding durations."""
        if self.trades_df.empty or 'duration' not in self.trades_df.columns:
            return {}
            
        durations = pd.to_timedelta(self.trades_df['duration']).dropna()
        if durations.empty:
            return {}
            
        winners = self.trades_df[self.trades_df['pnl'] > 0]['duration'].dropna()
        losers = self.trades_df[self.trades_df['pnl'] <= 0]['duration'].dropna()
        
        duration_by_exit = {}
        if 'exit_reason' in self.trades_df.columns:
            grouped = self.trades_df.groupby('exit_reason')['duration']
            duration_by_exit = grouped.apply(
                lambda x: pd.to_timedelta(x).mean().total_seconds() if not x.empty else 0.0
            ).to_dict()

        return {
            'avg_duration_seconds': durations.mean().total_seconds(),
            'median_duration_seconds': durations.median().total_seconds(),
            'min_duration_seconds': durations.min().total_seconds(),
            'max_duration_seconds': durations.max().total_seconds(),
            'duration_by_outcome_seconds': {
                'winners_avg': pd.to_timedelta(winners).mean().total_seconds() if not winners.empty else 0.0,
                'losers_avg': pd.to_timedelta(losers).mean().total_seconds() if not losers.empty else 0.0,
            },
            'duration_by_exit_reason_seconds': duration_by_exit
        }

    def exit_reason_breakdown(self) -> pd.DataFrame:
        """Breakdown of trades by exit reason."""
        if self.trades_df.empty or 'exit_reason' not in self.trades_df.columns:
            return pd.DataFrame()
            
        breakdown = []
        total_trades = len(self.trades_df)
        
        for reason, group in self.trades_df.groupby('exit_reason'):
            count = len(group)
            win_rate = (group['pnl'] > 0).mean()
            avg_pnl = group['pnl'].mean()
            
            breakdown.append({
                'exit_reason': reason,
                'count': count,
                'percentage': (count / total_trades) * 100,
                'avg_pnl': avg_pnl,
                'win_rate': win_rate * 100
            })
            
        return pd.DataFrame(breakdown).set_index('exit_reason')

    def streak_analysis(self) -> dict:
        """Analysis of winning and losing streaks."""
        if self.trades_df.empty or 'pnl' not in self.trades_df.columns:
            return {}
            
        wins = self.trades_df['pnl'] > 0
        
        streaks = (wins != wins.shift()).cumsum()
        streak_lengths = wins.groupby(streaks).size()
        streak_types = wins.groupby(streaks).first()
        
        win_streaks = streak_lengths[streak_types == True]
        loss_streaks = streak_lengths[streak_types == False]
        
        current_streak_type = wins.iloc[-1] if not wins.empty else None
        current_streak_length = streak_lengths.iloc[-1] if not streak_lengths.empty else 0

        return {
            'max_consecutive_wins': int(win_streaks.max()) if not win_streaks.empty else 0,
            'max_consecutive_losses': int(loss_streaks.max()) if not loss_streaks.empty else 0,
            'avg_win_streak': float(win_streaks.mean()) if not win_streaks.empty else 0.0,
            'avg_loss_streak': float(loss_streaks.mean()) if not loss_streaks.empty else 0.0,
            'current_streak': {
                'type': 'win' if current_streak_type else 'loss',
                'length': int(current_streak_length)
            }
        }

    def generate_full_report(self) -> dict:
        """Combine all analyses into a comprehensive report."""
        return {
            'mae_analysis': self.mae_analysis(),
            'mfe_analysis': self.mfe_analysis(),
            'holding_duration': self.holding_duration_analysis(),
            'exit_breakdown': self.exit_reason_breakdown().to_dict(orient='index'),
            'streaks': self.streak_analysis()
        }
