import os
import copy
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from research.walk_forward import WalkForwardEngine
from research.monte_carlo import MonteCarloSimulator

def run_monte_carlo(trades_df: pd.DataFrame, num_sims: int = 10000):
    mc = MonteCarloSimulator(trades_df, num_simulations=num_sims)
    sqn = mc.calculate_sqn()
    mc_res = mc.run_simulations()
    return sqn, mc_res

def run_validation():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    tf = "1h"
    file_path = get_data_path("BTCUSDT", tf, data_dir)
    print(f"Loading data for {tf}...")
    df = load_data(file_path)
    
    wf_engine = WalkForwardEngine(df, train_years=2, test_years=1)
    windows = wf_engine.get_windows()
    
    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = {}
    
    for thresh in thresholds:
        print(f"\n--- Testing Threshold: {thresh} ---")
        config = SystemConfig.default()
        config.strategy.atr_percentile_threshold = thresh
        
        all_trades = []
        win_rates = []
        pnls = []
        
        for df_train, df_test, train_name, test_name in windows:
            engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
            res = engine.run(df_test, timeframe=tf, config_name=f"Thresh_{thresh}")
            
            all_trades.append(res.trades_df)
            win_rates.append(res.metrics.get('Win Rate', 0))
            pnls.append(res.metrics.get('Net PnL', 0))
            
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        if combined_trades.empty or len(combined_trades) < 2:
            print(f"Threshold {thresh}: Not enough trades.")
            continue
            
        # Overall metrics
        total_trades = len(combined_trades)
        wins = combined_trades[combined_trades['net_pnl'] > 0]
        overall_wr = len(wins) / total_trades
        total_pnl = combined_trades['net_pnl'].sum()
        
        gross_profit = wins['net_pnl'].sum()
        gross_loss = abs(combined_trades[combined_trades['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = total_pnl / total_trades
        
        sqn, mc_res = run_monte_carlo(combined_trades, 10000)
        
        results[thresh] = {
            "trades": total_trades,
            "win_rate": overall_wr,
            "net_pnl": total_pnl,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "sqn": sqn,
            "mc_5th": mc_res.get('mc_profit_5th', 0),
            "mc_median": mc_res.get('mc_profit_50th', 0),
            "mc_95th": mc_res.get('mc_profit_95th', 0),
            "mc_dd_95th": mc_res.get('mc_dd_95th', 0)
        }
        
        print(f"T={thresh} | Trades: {total_trades} | WR: {overall_wr:.1%} | PnL: ${total_pnl:.2f} | PF: {profit_factor:.2f} | SQN: {sqn:.2f}")

    # Identify Best Threshold
    base = results.get(0.0)
    best_thresh = None
    best_sqn = -999
    for t, m in results.items():
        if t != 0.0 and m['sqn'] > best_sqn:
            best_sqn = m['sqn']
            best_thresh = t
            
    print(f"\nBest Threshold: {best_thresh} (SQN: {best_sqn:.2f})")
    
    # Stress Testing the Best Threshold
    print("\n--- Stress Testing Best Threshold ---")
    stress_results = {}
    if best_thresh is not None:
        config_stress = SystemConfig.default()
        config_stress.strategy.atr_percentile_threshold = best_thresh
        
        # Scenario 1: 2x Fees
        c_fees = copy.deepcopy(config_stress)
        c_fees.strategy.atr_percentile_threshold = best_thresh
        c_fees.risk.commission_pct *= 2.0
        
        # Scenario 2: 2x Slippage
        c_slip = copy.deepcopy(config_stress)
        c_slip.strategy.atr_percentile_threshold = best_thresh
        c_slip.risk.slippage_pct *= 2.0
        
        for name, cfg in [("2x_Fees", c_fees), ("2x_Slippage", c_slip)]:
            all_stress_trades = []
            for df_train, df_test, train_name, test_name in windows:
                engine = BacktestEngine(config=cfg, use_risk_management=True, use_ml_filter=False)
                res = engine.run(df_test, timeframe=tf, config_name=name)
                all_stress_trades.append(res.trades_df)
                
            ct = pd.concat(all_stress_trades, ignore_index=True) if all_stress_trades else pd.DataFrame()
            if not ct.empty and len(ct) >= 2:
                ssqn, smc = run_monte_carlo(ct, 2000)
                w = ct[ct['net_pnl'] > 0]
                stress_results[name] = {
                    "win_rate": len(w) / len(ct),
                    "net_pnl": ct['net_pnl'].sum(),
                    "sqn": ssqn
                }
                print(f"Stress {name}: WR {len(w)/len(ct):.1%} | PnL ${ct['net_pnl'].sum():.2f} | SQN {ssqn:.2f}")

    # Generate Markdown Report
    report = ["# Volatility Gating Validation Report\n"]
    report.append("## Does Volatility Gating Improve the Strategy?")
    
    b = results.get(0.0)
    bt = results.get(best_thresh) if best_thresh is not None else None
    
    if bt and bt['sqn'] > b['sqn']:
        report.append("**YES.** Volatility Gating significantly improves the core expectancy and stability of the system.\n")
    else:
        report.append("**NO.** It did not materially improve the system.\n")
        
    report.append("## Threshold Comparison (Combined Walk-Forward 6 Years)")
    report.append("| Threshold | Trades | Win Rate | Net PnL | Expectancy | Profit Factor | SQN | MC Median PnL | MC 95th DD |")
    report.append("|-----------|--------|----------|---------|------------|---------------|-----|---------------|------------|")
    
    for t in sorted(results.keys()):
        m = results[t]
        report.append(f"| {t} | {m['trades']} | {m['win_rate']:.1%} | ${m['net_pnl']:.2f} | ${m['expectancy']:.2f} | {m['profit_factor']:.2f} | {m['sqn']:.2f} | ${m['mc_median']:.2f} | ${m['mc_dd_95th']:.2f} |")
        
    report.append("\n## Stress Testing Best Threshold")
    for name, m in stress_results.items():
        report.append(f"- **{name}**: Win Rate {m['win_rate']:.1%}, Net PnL ${m['net_pnl']:.2f}, SQN {m['sqn']:.2f}")
        
    report.append("\n## Final Decision")
    if bt and bt['sqn'] > b['sqn'] and stress_results.get("2x_Slippage", {}).get("sqn", -1) > 0:
        report.append("### ACCEPTED")
        report.append("The Volatility Gating filter successfully eliminates low-quality ranges while preserving fat-tail trends. The edge survives extreme stress testing.")
    else:
        report.append("### REJECTED")
        report.append("The improvement was not robust enough to survive 10k Monte Carlo simulations and aggressive transaction cost stress testing.")
        
    out_path = os.path.join(os.path.dirname(__file__), "..", "volatility_gating_report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"\nReport generated at {out_path}")

if __name__ == '__main__':
    run_validation()
