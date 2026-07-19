import os
import json
import pandas as pd
from scipy import stats

def analyze_statistics():
    data_path = os.path.join(os.path.dirname(__file__), "market_regimes.csv")
    if not os.path.exists(data_path):
        print("Run regime_analyzer.py first.")
        return
        
    df = pd.read_csv(data_path)
    if df.empty:
        return
        
    winners = df[df['net_pnl'] > 0]
    losers = df[df['net_pnl'] <= 0]
    
    # 1. Fat Tails
    pnl_90th = df['net_pnl'].quantile(0.9)
    pnl_10th = df['net_pnl'].quantile(0.1)
    
    fat_winners = df[df['net_pnl'] >= pnl_90th]
    fat_losers = df[df['net_pnl'] <= pnl_10th]
    
    # 2. Extract significant feature differences
    features = ['rsi', 'rsi_slope', 'adx', 'adx_slope', 'atr_percentile', 'ema_distance', 'ema_slope', 'time_of_day']
    
    findings = []
    
    for feat in features:
        # T-test for Winners vs Losers
        t_stat, p_val = stats.ttest_ind(winners[feat], losers[feat], equal_var=False)
        mean_w = winners[feat].mean()
        mean_l = losers[feat].mean()
        
        if p_val < 0.05:
            findings.append({
                "feature": feat,
                "comparison": "Winners vs Losers",
                "winner_mean": mean_w,
                "loser_mean": mean_l,
                "p_value": p_val,
                "significance": "High" if p_val < 0.01 else "Medium"
            })
            
        # Fat winners vs normal
        t_stat_fw, p_val_fw = stats.ttest_ind(fat_winners[feat], df[feat], equal_var=False)
        if p_val_fw < 0.05:
            findings.append({
                "feature": feat,
                "comparison": "Fat Winners vs Rest",
                "fat_winner_mean": fat_winners[feat].mean(),
                "population_mean": df[feat].mean(),
                "p_value": p_val_fw
            })
            
    # Compile JSON
    hypotheses = {
        "metadata": {
            "total_trades": len(df),
            "win_rate": len(winners) / len(df)
        },
        "statistical_findings": findings,
        "fat_tail_characteristics": {
            "avg_adx": fat_winners['adx'].mean(),
            "avg_rsi": fat_winners['rsi'].mean(),
            "avg_atr_pct": fat_winners['atr_percentile'].mean()
        },
        "worst_loss_characteristics": {
            "avg_adx": fat_losers['adx'].mean(),
            "avg_rsi": fat_losers['rsi'].mean(),
            "avg_atr_pct": fat_losers['atr_percentile'].mean()
        }
    }
    
    out_path = os.path.join(os.path.dirname(__file__), "hypothesis_candidates.json")
    with open(out_path, "w") as f:
        json.dump(hypotheses, f, indent=4)
        
    print(f"Statistical analysis complete! Saved to {out_path}")

if __name__ == '__main__':
    analyze_statistics()
