import os
import pandas as pd
import numpy as np
from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from strategy.theses import DonchianBreakout, ATRBurst, MultiTimeframeMomentum, BollingerMeanReversion, ADXRegime
from main import load_data, split_data

def run_theses_validation():
    timeframe = "1h"
    config = SystemConfig()
    
    # Use standard risk settings, not overfitted ones
    config.risk.trailing_stop_atr_mult = 1.0
    config.risk.stop_loss_atr_mult = 1.5
    config.risk.risk_per_trade_pct = 0.01
    
    df = load_data(f"data/raw/BTCUSDT_{timeframe}.parquet")
    df_dev, df_holdout, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    if 'Open Time' not in df_dev.columns:
        df_dev = df_dev.reset_index(names='Open Time')
    if 'Open Time' not in df_holdout.columns:
        df_holdout = df_holdout.reset_index(names='Open Time')
    
    theses = {
        "1_Donchian": DonchianBreakout(),
        "2_ATRBurst": ATRBurst(),
        "3_Momentum": MultiTimeframeMomentum(),
        "4_MeanRev": BollingerMeanReversion(),
        "5_Regime": ADXRegime()
    }
    
    results = []
    
    for name, thesis in theses.items():
        print(f"\nEvaluating: {name}")
        
        # --- DEVELOPMENT SET ---
        signals_df_dev = thesis.generate_signals(df_dev)
        sig_dev = signals_df_dev[signals_df_dev['signal_changed'] == True].copy()
        # Filter out NEUTRAL to avoid trying to trade it
        sig_dev = sig_dev[sig_dev['raw_signal'] != 'NEUTRAL']
        
        precomp_dev = (sig_dev, len(sig_dev), df_dev)
        
        engine_dev = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
        res_dev = engine_dev.run(df_dev, timeframe=timeframe, config_name=f"{name}_Dev",
                                 precomputed_signals=precomp_dev, precomputed_features=None)
        
        dev_sharpe = res_dev.metrics.get('Sharpe Ratio', 0)
        dev_pf = res_dev.metrics.get('Profit Factor', 0)
        
        # --- HOLDOUT SET ---
        signals_df_holdout = thesis.generate_signals(df_holdout)
        sig_holdout = signals_df_holdout[signals_df_holdout['signal_changed'] == True].copy()
        sig_holdout = sig_holdout[sig_holdout['raw_signal'] != 'NEUTRAL']
        
        precomp_holdout = (sig_holdout, len(sig_holdout), df_holdout)
        
        engine_holdout = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
        res_holdout = engine_holdout.run(df_holdout, timeframe=timeframe, config_name=f"{name}_Holdout",
                                         precomputed_signals=precomp_holdout, precomputed_features=None)
        
        hold_sharpe = res_holdout.metrics.get('Sharpe Ratio', 0)
        hold_pf = res_holdout.metrics.get('Profit Factor', 0)
        
        # --- DEGRADATION SCORE ---
        # Handle division by zero or negative dev
        if dev_sharpe <= 0:
            sharpe_deg = 0.0 # If dev is already bad, degradation score is 0
        else:
            sharpe_deg = hold_sharpe / dev_sharpe
            
        if dev_pf <= 0:
            pf_deg = 0.0
        else:
            pf_deg = hold_pf / dev_pf
            
        deg_score = (sharpe_deg + pf_deg) / 2.0
        
        results.append({
            "Strategy": name,
            "Dev Sharpe": dev_sharpe,
            "Dev PF": dev_pf,
            "Holdout Sharpe": hold_sharpe,
            "Holdout PF": hold_pf,
            "Degradation Score": deg_score
        })
        
    print("\n--- THESIS EVALUATION RESULTS ---")
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Degradation Score", ascending=False)
    
    print("| Strategy | Dev Sharpe | Dev PF | Holdout Sharpe | Holdout PF | Degradation Score |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, row in results_df.iterrows():
        print(f"| **{row['Strategy']}** | {row['Dev Sharpe']:.2f} | {row['Dev PF']:.2f} | {row['Holdout Sharpe']:.2f} | {row['Holdout PF']:.2f} | {row['Degradation Score']:.2f} |")

if __name__ == "__main__":
    run_theses_validation()
