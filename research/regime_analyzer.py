import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def rule_based_regime(row):
    if pd.isna(row['adx']): return "Unknown"
    if row['adx'] > 25:
        if row['atr_percentile'] > 0.7:
            return "Trending_Volatile"
        else:
            return "Trending_Calm"
    else:
        if row['atr_percentile'] > 0.7:
            return "Ranging_Volatile"
        else:
            return "Ranging_Calm"

def analyze_regimes():
    data_path = os.path.join(os.path.dirname(__file__), "trade_context_dataset.csv")
    if not os.path.exists(data_path):
        print("Run feature_extractor.py first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    # Features for clustering
    cluster_features = ['rsi', 'adx', 'atr_percentile', 'ema_distance']
    X = df[cluster_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means (4 regimes)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['kmeans_regime'] = kmeans.fit_predict(X_scaled)
    
    # GMM (4 regimes)
    gmm = GaussianMixture(n_components=4, random_state=42)
    df['gmm_regime'] = gmm.fit_predict(X_scaled)
    
    # Rule-Based
    df['rule_regime'] = df.apply(rule_based_regime, axis=1)
    
    # Save output
    out_path = os.path.join(os.path.dirname(__file__), "market_regimes.csv")
    df.to_csv(out_path, index=False)
    print(f"Regime analysis complete! Saved to {out_path}")
    
    # Print high-level stats for Rule-Based
    print("\nRule-Based Regime Stats:")
    for regime, group in df.groupby('rule_regime'):
        wins = group[group['net_pnl'] > 0]
        win_rate = len(wins) / len(group) if len(group) > 0 else 0
        print(f"{regime}: {len(group)} trades, Win Rate: {win_rate:.1%}, Net PnL: ${group['net_pnl'].sum():.2f}")

if __name__ == '__main__':
    analyze_regimes()
