import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

class StrategyVisualizer:
    """Chart generation using matplotlib."""
    
    def __init__(self, style: str = 'dark_background'):
        plt.style.use(style)
        self.colors = {
            'equity': '#00ff00',
            'drawdown': '#ff0000',
            'win': '#00ff00',
            'loss': '#ff0000',
            's1': '#aaaaaa',
            's2': '#5555ff',
            's3': '#ff55ff',
            's4': '#00ff00'
        }

    def plot_equity_curve(self, equity_curve: pd.Series, title: str = '', save_path: str = None) -> None:
        """Plot the equity curve over time."""
        if equity_curve is None or equity_curve.empty:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(equity_curve.index, equity_curve.values, color=self.colors['equity'], linewidth=1.5)
        
        ax.set_title(f'Equity Curve: {title}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Equity', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_drawdown(self, equity_curve: pd.Series, title: str = '', save_path: str = None) -> None:
        """Plot the drawdown curve over time."""
        if equity_curve is None or equity_curve.empty:
            return
            
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak * 100
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0, color=self.colors['drawdown'], alpha=0.5)
        ax.plot(drawdown.index, drawdown.values, color=self.colors['drawdown'], linewidth=1)
        
        ax.set_title(f'Drawdown (%): {title}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Drawdown %', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_monthly_returns_heatmap(self, equity_curve: pd.Series, save_path: str = None) -> None:
        """Plot a heatmap of monthly returns."""
        if equity_curve is None or equity_curve.empty:
            return
            
        daily_returns = equity_curve.resample('D').last().pct_change().dropna()
        if daily_returns.empty:
            return
            
        monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        df = pd.DataFrame({'return': monthly_returns})
        df['year'] = df.index.year
        df['month'] = df.index.month
        pivot = df.pivot(index='year', columns='month', values='return') * 100 
        
        for i in range(1, 13):
            if i not in pivot.columns:
                pivot[i] = np.nan
        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig, ax = plt.subplots(figsize=(12, len(pivot) * 0.8 + 2))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=ax, 
                    cbar_kws={'label': 'Return %'})
        
        ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Year', fontsize=12)
        ax.set_xlabel('Month', fontsize=12)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_trade_distribution(self, trades_df: pd.DataFrame, save_path: str = None) -> None:
        """Plot a histogram of PnL per trade."""
        if trades_df is None or trades_df.empty or 'pnl' not in trades_df.columns:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        pnl = trades_df['pnl']
        
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        
        ax.hist(wins, bins=30, color=self.colors['win'], alpha=0.7, label='Wins')
        ax.hist(losses, bins=30, color=self.colors['loss'], alpha=0.7, label='Losses')
        
        ax.axvline(0, color='white', linestyle='--', linewidth=1)
        ax.set_title('Trade PnL Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('PnL', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_mae_mfe_scatter(self, trades_df: pd.DataFrame, save_path: str = None) -> None:
        """Scatter plot of MAE vs MFE, colored by outcome."""
        if trades_df is None or trades_df.empty or 'mae' not in trades_df.columns or 'mfe' not in trades_df.columns:
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        ax.scatter(wins['mae'], wins['mfe'], color=self.colors['win'], alpha=0.6, label='Wins', s=50)
        ax.scatter(losses['mae'], losses['mfe'], color=self.colors['loss'], alpha=0.6, label='Losses', s=50)
        
        max_val = max(trades_df['mae'].abs().max(), trades_df['mfe'].max())
        if not np.isnan(max_val):
            ax.plot([0, max_val], [0, max_val], color='white', linestyle='--', alpha=0.5)
            
        ax.set_title('MAE vs MFE', fontsize=14, fontweight='bold')
        ax.set_xlabel('Maximum Adverse Excursion (MAE)', fontsize=12)
        ax.set_ylabel('Maximum Favorable Excursion (MFE)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_ablation_comparison(self, results: dict, save_path: str = None) -> None:
        """Bar charts comparing S1/S2/S3/S4 across key metrics."""
        if not results:
            return
            
        metrics_to_plot = {
            'total_return_pct': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown_pct': 'Max Drawdown (%)',
            'win_rate': 'Win Rate'
        }
        
        configs = list(results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, (metric_key, metric_name) in enumerate(metrics_to_plot.items()):
            ax = axes[i]
            values = []
            for config in configs:
                # We assume results[config] has a metrics object
                m = getattr(results[config], 'metrics', None)
                val = getattr(m, metric_key, 0) if m else 0
                values.append(val)
                
            colors = [self.colors.get(c.lower().split('_')[0], '#888888') for c in configs]
            bars = ax.bar(configs, values, color=colors, alpha=0.8)
            
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom')
                            
        plt.suptitle('Ablation Study Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_renko_bricks(self, bricks_df: pd.DataFrame, price_df: pd.DataFrame = None, max_bricks: int = 100, save_path: str = None) -> None:
        """Visual Renko chart with colored bricks."""
        if bricks_df is None or bricks_df.empty:
            return
            
        df = bricks_df.tail(max_bricks).copy()
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        brick_size = df['brick_size'].iloc[0] if 'brick_size' in df.columns else 0
        if brick_size == 0 and len(df) > 1:
            brick_size = abs(df['close'].iloc[1] - df['close'].iloc[0])
            
        for i, row in enumerate(df.itertuples()):
            color = self.colors['win'] if row.trend == 1 else self.colors['loss']
            rect = plt.Rectangle((i - 0.4, min(row.open, row.close)), 0.8, abs(row.close - row.open), 
                                 facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            
        ax.set_xlim(-1, len(df))
        ax.set_ylim(df[['open', 'close']].min().min() - brick_size, df[['open', 'close']].max().max() + brick_size)
        
        ax.set_title('Renko Chart', fontsize=14, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12)
        ax.set_xticks(np.arange(0, len(df), max(1, len(df)//10)))
        if 'timestamp' in df.columns:
            labels = df['timestamp'].iloc[ax.get_xticks()].dt.strftime('%Y-%m-%d')
            ax.set_xticklabels(labels, rotation=45)
            
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_cumulative_returns_comparison(self, results: dict, save_path: str = None) -> None:
        """Overlay equity curves for all ablation configs."""
        if not results:
            return
            
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for config_name, result in results.items():
            if not hasattr(result, 'equity_curve') or result.equity_curve is None or result.equity_curve.empty:
                continue
                
            color = self.colors.get(config_name.lower().split('_')[0], None)
            
            cum_ret = result.equity_curve / result.equity_curve.iloc[0] - 1
            ax.plot(cum_ret.index, cum_ret.values * 100, label=config_name, color=color, linewidth=1.5)
            
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def generate_all_plots(self, backtest_result, output_dir: str) -> list[str]:
        """Generate all relevant plots for a backtest result, save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        config_name = getattr(backtest_result, 'config_name', 'strategy')
        
        if hasattr(backtest_result, 'equity_curve') and backtest_result.equity_curve is not None:
            eq_path = os.path.join(output_dir, f'{config_name}_equity.png')
            self.plot_equity_curve(backtest_result.equity_curve, title=config_name, save_path=eq_path)
            saved_files.append(eq_path)
            
            dd_path = os.path.join(output_dir, f'{config_name}_drawdown.png')
            self.plot_drawdown(backtest_result.equity_curve, title=config_name, save_path=dd_path)
            saved_files.append(dd_path)
            
            hm_path = os.path.join(output_dir, f'{config_name}_monthly_returns.png')
            self.plot_monthly_returns_heatmap(backtest_result.equity_curve, save_path=hm_path)
            saved_files.append(hm_path)
        
        if hasattr(backtest_result, 'trades') and backtest_result.trades is not None and not backtest_result.trades.empty:
            trades_df = backtest_result.trades
            
            td_path = os.path.join(output_dir, f'{config_name}_trade_dist.png')
            self.plot_trade_distribution(trades_df, save_path=td_path)
            saved_files.append(td_path)
            
            if 'mae' in trades_df.columns and 'mfe' in trades_df.columns:
                mae_mfe_path = os.path.join(output_dir, f'{config_name}_mae_mfe.png')
                self.plot_mae_mfe_scatter(trades_df, save_path=mae_mfe_path)
                saved_files.append(mae_mfe_path)
                
        return saved_files
