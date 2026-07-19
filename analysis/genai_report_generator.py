from typing import Dict, Any

class GenAIReportGenerator:
    """Generate structured markdown reports for GenAI analysis."""

    def __init__(self):
        pass

    def generate_backtest_report(self, backtest_result: Any, trades_analytics: dict, config_name: str) -> str:
        """Returns a structured markdown report for a backtest run."""
        metrics = getattr(backtest_result, 'metrics', None)
        
        report = f"# Backtest Report: {config_name}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n"
        report += f"- **Total Return**: {getattr(metrics, 'total_return_pct', 0):.2f}%\n"
        report += f"- **Sharpe Ratio**: {getattr(metrics, 'sharpe_ratio', 0):.2f}\n"
        report += f"- **Max Drawdown**: {getattr(metrics, 'max_drawdown_pct', 0):.2f}%\n"
        report += f"- **Win Rate**: {getattr(metrics, 'win_rate', 0) * 100:.2f}%\n"
        report += f"- **Total Trades**: {getattr(metrics, 'total_trades', 0)}\n\n"
        
        # Performance Metrics Table
        report += "## Performance Metrics\n"
        report += "| Metric | Value |\n|---|---|\n"
        report += f"| Profit Factor | {getattr(metrics, 'profit_factor', 0):.2f} |\n"
        report += f"| Sortino Ratio | {getattr(metrics, 'sortino_ratio', 0):.2f} |\n"
        report += f"| Calmar Ratio | {getattr(metrics, 'calmar_ratio', 0):.2f} |\n"
        report += f"| Average Win | {getattr(metrics, 'avg_win', 0):.2f} |\n"
        report += f"| Average Loss | {getattr(metrics, 'avg_loss', 0):.2f} |\n\n"
        
        # Trade Analysis
        report += "## Trade Analysis\n"
        if 'streaks' in trades_analytics and trades_analytics['streaks']:
            streaks = trades_analytics['streaks']
            report += f"- **Max Consecutive Wins**: {streaks.get('max_consecutive_wins', 0)}\n"
            report += f"- **Max Consecutive Losses**: {streaks.get('max_consecutive_losses', 0)}\n"
            report += f"- **Avg Win Streak**: {streaks.get('avg_win_streak', 0):.2f}\n"
            report += f"- **Avg Loss Streak**: {streaks.get('avg_loss_streak', 0):.2f}\n\n"
            
        # MAE/MFE Analysis
        report += "## MAE / MFE Analysis\n"
        if 'mae_analysis' in trades_analytics and trades_analytics['mae_analysis']:
            mae = trades_analytics['mae_analysis']
            report += f"- **Average MAE**: {mae.get('avg_mae', 0):.4f}\n"
            report += f"- **Max MAE**: {mae.get('max_mae', 0):.4f}\n"
        if 'mfe_analysis' in trades_analytics and trades_analytics['mfe_analysis']:
            mfe = trades_analytics['mfe_analysis']
            report += f"- **Average MFE**: {mfe.get('avg_mfe', 0):.4f}\n"
            report += f"- **Max MFE**: {mfe.get('max_mfe', 0):.4f}\n"
            report += f"- **Avg Profit Left on Table**: {mfe.get('avg_profit_left_on_table', 0):.4f}\n\n"
            
        # Exit Reason Breakdown
        report += "## Exit Reason Breakdown\n"
        if 'exit_breakdown' in trades_analytics and trades_analytics['exit_breakdown']:
            report += "| Exit Reason | Count | Percentage | Win Rate | Avg PnL |\n|---|---|---|---|---|\n"
            for reason, data in trades_analytics['exit_breakdown'].items():
                report += f"| {reason} | {data.get('count', 0)} | {data.get('percentage', 0):.2f}% | {data.get('win_rate', 0):.2f}% | {data.get('avg_pnl', 0):.2f} |\n"
            report += "\n"
            
        # Prompt for GenAI
        report += "## AI Analysis Prompt\n"
        report += "> Based on the above data, analyze:\n"
        report += "> 1. How effective is the current stop-loss and take-profit positioning given the MAE/MFE stats?\n"
        report += "> 2. What does the exit reason breakdown suggest about the strategy's risk management?\n"
        report += "> 3. Are the streaks indicative of over-trading or strategy fragility during certain regimes?\n"
        report += "> 4. Provide specific recommendations to improve the Profit Factor and Sharpe Ratio.\n\n"
        
        return report

    def generate_comparison_report(self, all_results: Dict[str, Any]) -> str:
        """Compare S1/S2/S3/S4 results."""
        report = "# Ablation Study Comparison Report\n\n"
        
        report += "## Metrics Comparison\n"
        report += "| Configuration | Total Return | Sharpe | Max DD | Win Rate | Profit Factor |\n"
        report += "|---|---|---|---|---|---|\n"
        
        for name, result in all_results.items():
            metrics = getattr(result, 'metrics', None)
            if metrics:
                report += f"| **{name}** | {getattr(metrics, 'total_return_pct', 0):.2f}% | {getattr(metrics, 'sharpe_ratio', 0):.2f} | {getattr(metrics, 'max_drawdown_pct', 0):.2f}% | {getattr(metrics, 'win_rate', 0)*100:.2f}% | {getattr(metrics, 'profit_factor', 0):.2f} |\n"
            
        report += "\n## Impact Analysis\n"
        report += "- **S1 to S2**: Impact of adding Risk Management.\n"
        report += "- **S2 to S3**: Impact of adding Machine Learning predictions.\n"
        report += "- **S3 to S4**: Impact of Kelly Sizing / Dynamic Capital Allocation.\n\n"
        
        report += "## AI Analysis Prompt\n"
        report += "> Based on the comparison above, analyze:\n"
        report += "> 1. Did the addition of ML models significantly improve risk-adjusted returns compared to pure technical rules?\n"
        report += "> 2. How did Kelly sizing affect the Maximum Drawdown?\n"
        report += "> 3. Which configuration offers the best trade-off between return and risk?\n\n"
        
        return report

    def generate_forward_test_report(self, backtest_result: Any, forward_result: Any) -> str:
        """Compare backtest vs forward test."""
        bm = getattr(backtest_result, 'metrics', None)
        fm = getattr(forward_result, 'metrics', None)
        
        report = "# Forward Testing & Validation Report\n\n"
        
        report += "## Performance Degradation Analysis\n"
        report += "| Metric | Backtest | Forward Test | Change | Degradation %\n"
        report += "|---|---|---|---|---|\n"
        
        metrics_to_compare = {
            'total_return_pct': 'Total Return (%)',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown_pct': 'Max DD (%)',
            'win_rate': 'Win Rate',
            'profit_factor': 'Profit Factor'
        }
        
        if bm and fm:
            for key, name in metrics_to_compare.items():
                b_val = getattr(bm, key, 0)
                f_val = getattr(fm, key, 0)
                diff = f_val - b_val
                deg = 0
                if b_val != 0:
                    if key == 'max_drawdown_pct':
                        deg = ((f_val - b_val) / abs(b_val)) * 100
                    else:
                        deg = ((b_val - f_val) / abs(b_val)) * 100
                        
                report += f"| {name} | {b_val:.2f} | {f_val:.2f} | {diff:.2f} | {deg:.2f}% |\n"
            
        report += "\n## Stability Assessment\n"
        report += "*(See ValidationReport for programmatic overfitting detection)*\n\n"
        
        report += "## AI Analysis Prompt\n"
        report += "> Based on the backtest vs forward test comparison, analyze:\n"
        report += "> 1. Is there evidence of curve-fitting/overfitting in the backtest?\n"
        report += "> 2. Which metrics degraded the most, and why might that be?\n"
        report += "> 3. Can this strategy be considered robust enough for live deployment?\n\n"
        
        return report

    def save_report(self, report: str, filepath: str) -> None:
        """Save markdown report to file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
