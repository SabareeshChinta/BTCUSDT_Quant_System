import pandas as pd
from typing import Dict, Any, Tuple

from backtesting.backtest_engine import BacktestResult

class ValidationReport:
    """Validation comparison between backtest and forward test."""
    
    def __init__(self):
        pass

    def compare_phases(self, backtest_results: Dict[str, BacktestResult], forward_results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """
        Side-by-side comparison of metrics from both phases.
        Calculate degradation percentage for each metric.
        """
        comparisons = []
        
        metrics_list = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate', 'profit_factor']
        
        for config_name in backtest_results.keys():
            if config_name not in forward_results:
                continue
                
            b_res = backtest_results[config_name]
            f_res = forward_results[config_name]
            
            b_metrics = getattr(b_res, 'metrics', None)
            f_metrics = getattr(f_res, 'metrics', None)
            
            if not b_metrics or not f_metrics:
                continue
            
            for m in metrics_list:
                b_val = getattr(b_metrics, m, 0.0)
                f_val = getattr(f_metrics, m, 0.0)
                
                deg = 0.0
                flag = False
                
                if b_val != 0:
                    if m == 'max_drawdown_pct':
                        deg = ((f_val - b_val) / abs(b_val)) * 100
                        flag = deg > 30.0
                    else:
                        deg = ((b_val - f_val) / abs(b_val)) * 100
                        flag = deg > 30.0
                        
                comparisons.append({
                    'Configuration': config_name,
                    'Metric': m,
                    'Backtest': b_val,
                    'Forward': f_val,
                    'Degradation %': deg,
                    'Flagged (>30%)': flag
                })
                
        return pd.DataFrame(comparisons)

    def detect_overfitting(self, backtest_metrics: Any, forward_metrics: Any) -> Dict[str, Any]:
        """
        Detect overfitting indicators between a backtest and forward test metric set.
        """
        degraded_metrics = []
        stable_metrics = []
        score = 0.0
        max_score = 4.0
        
        if not backtest_metrics or not forward_metrics:
            return {
                'overfitting_score': 0.0,
                'degraded_metrics': [],
                'stable_metrics': [],
                'assessment': "No Data"
            }
        
        # Profit Factor drop > 50%
        b_pf = getattr(backtest_metrics, 'profit_factor', 0)
        f_pf = getattr(forward_metrics, 'profit_factor', 0)
        if b_pf > 0:
            pf_drop = (b_pf - f_pf) / b_pf
            if pf_drop > 0.5:
                degraded_metrics.append('profit_factor')
                score += 1
            else:
                stable_metrics.append('profit_factor')
                
        # Sharpe drop > 50%
        b_sh = getattr(backtest_metrics, 'sharpe_ratio', 0)
        f_sh = getattr(forward_metrics, 'sharpe_ratio', 0)
        if b_sh > 0:
            sh_drop = (b_sh - f_sh) / b_sh
            if sh_drop > 0.5:
                degraded_metrics.append('sharpe_ratio')
                score += 1
            else:
                stable_metrics.append('sharpe_ratio')
                
        # Win rate drop > 20%
        b_wr = getattr(backtest_metrics, 'win_rate', 0)
        f_wr = getattr(forward_metrics, 'win_rate', 0)
        if b_wr > 0:
            wr_drop = (b_wr - f_wr) / b_wr
            if wr_drop > 0.2:
                degraded_metrics.append('win_rate')
                score += 1
            else:
                stable_metrics.append('win_rate')
                
        # Max DD increase > 50%
        b_dd = getattr(backtest_metrics, 'max_drawdown_pct', 0)
        f_dd = getattr(forward_metrics, 'max_drawdown_pct', 0)
        if b_dd > 0:
            dd_inc = (f_dd - b_dd) / b_dd
            if dd_inc > 0.5:
                degraded_metrics.append('max_drawdown_pct')
                score += 1
            else:
                stable_metrics.append('max_drawdown_pct')
                
        overfitting_score = score / max_score
        
        assessment = "Robust"
        if overfitting_score >= 0.75:
            assessment = "Highly Overfitted"
        elif overfitting_score >= 0.5:
            assessment = "Moderately Overfitted"
        elif overfitting_score > 0:
            assessment = "Slight Degradation"
            
        return {
            'overfitting_score': overfitting_score,
            'degraded_metrics': degraded_metrics,
            'stable_metrics': stable_metrics,
            'assessment': assessment
        }

    def generate_validation_summary(self, backtest_results: Dict[str, BacktestResult], forward_results: Dict[str, BacktestResult]) -> str:
        """Markdown summary with pass/fail assessment."""
        report = "# Forward Testing Validation Summary\n\n"
        
        df = self.compare_phases(backtest_results, forward_results)
        
        for config_name in backtest_results.keys():
            if config_name not in forward_results:
                continue
                
            report += f"## Configuration: {config_name}\n"
            
            b_metrics = getattr(backtest_results[config_name], 'metrics', None)
            f_metrics = getattr(forward_results[config_name], 'metrics', None)
            
            overfit_data = self.detect_overfitting(b_metrics, f_metrics)
            
            report += f"- **Overfitting Score**: {overfit_data['overfitting_score']:.2f} (0=Robust, 1=Overfitted)\n"
            report += f"- **Assessment**: **{overfit_data['assessment']}**\n"
            
            report += "- **Degraded Metrics**: " + (", ".join(overfit_data['degraded_metrics']) if overfit_data['degraded_metrics'] else "None") + "\n"
            report += "- **Stable Metrics**: " + (", ".join(overfit_data['stable_metrics']) if overfit_data['stable_metrics'] else "None") + "\n\n"
            
            config_df = df[df['Configuration'] == config_name]
            if not config_df.empty:
                report += "### Metric Degradation\n"
                report += "| Metric | Backtest | Forward | Degradation % | Status |\n"
                report += "|---|---|---|---|---|\n"
                for _, row in config_df.iterrows():
                    status = "⚠️ FAIL" if row['Flagged (>30%)'] else "✅ PASS"
                    report += f"| {row['Metric']} | {row['Backtest']:.4f} | {row['Forward']:.4f} | {row['Degradation %']:.2f}% | {status} |\n"
                
            report += "\n"
            
        return report
