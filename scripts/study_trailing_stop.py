import os
import pandas as pd
import numpy as np
from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from ml.feature_engineering import FeatureEngineer
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor
from main import load_data, split_data

def run_study():
    timeframe = "1h"
    config = SystemConfig()
    
    config.risk.hard_drawdown_halt_pct = 1.0
    
    df = load_data(f"data/raw/BTCUSDT_{timeframe}.parquet")
    df_backtest, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    print(f"Running S1 backtest on {timeframe} to generate labels...")
    engine = BacktestEngine(config, use_risk_management=False, use_ml_filter=False)
    s1_res = engine.run(df_backtest, timeframe=timeframe, config_name="S1")
    trades_df = s1_res.trades_df
    
    print("Computing features...")
    fe = FeatureEngineer()
    db = DatasetBuilder(feature_engineer=fe)
    X, y = db.build_from_trades(trades_df, df_backtest)
    
    # Train a single final model for the study
    print("Training final model...")
    trainer = ModelTrainer(random_state=42)
    model = trainer.train(X, y)
        
    print("\nPrecomputing Renko signals and ML features for fast batch testing...")
    from strategy.signal_generator import SignalGenerator
    from indicators.renko_engine import RenkoEngine
    
    renko = RenkoEngine(atr_period=config.strategy.atr_period, atr_multiplier=config.strategy.atr_multiplier)
    df_raw = df_backtest.copy()
    if 'Open Time' not in df_raw.columns:
        df_raw = df_raw.reset_index(names='Open Time')
    if df_raw['Open Time'].dt.tz is not None:
        df_raw['Open Time'] = df_raw['Open Time'].dt.tz_localize(None)
        
    bricks = renko.build_bricks(df_raw)
    bricks['consecutive_count'] = renko.get_consecutive_count(bricks)
    sg = SignalGenerator(consecutive_bricks=config.strategy.consecutive_bricks)
    signals_df = sg.generate_signals(bricks)
    signals = signals_df[signals_df['signal_changed'] == True]
    precomputed_signals = (signals, len(signals), df_raw)
    
    if 'Open Time' not in df_backtest.columns:
        df_feat_input = df_backtest.reset_index(names='Open Time')
    else:
        df_feat_input = df_backtest.copy()
        
    if df_feat_input['Open Time'].dt.tz is not None:
        df_feat_input['Open Time'] = df_feat_input['Open Time'].dt.tz_localize(None)
        
    df_ml_features = fe.compute_features(df_feat_input)
    df_ml_features.set_index('Open Time', inplace=True)

    print("\n--- TRAILING STOP SENSITIVITY STUDY ---")
    config.risk.hard_drawdown_halt_pct = 0.25 # Reset back to normal
    best_threshold = 0.55
    
    trailing_stops = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    predictor = MLPredictor(model=model, feature_engineer=fe, confidence_threshold=best_threshold)
    
    print("| Trailing Stop | Trades | Profit Factor | Sharpe Ratio | Max Drawdown | Expectancy | Avg Win | Avg Loss |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for ts in trailing_stops:
        config.risk.trailing_stop_atr_mult = ts
        
        s4_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
        res = s4_engine.run(df_backtest, timeframe=timeframe, config_name=f"TS_{ts}", 
                            precomputed_signals=precomputed_signals, 
                            precomputed_features=df_ml_features)
        
        m = res.metrics
        trades = m.get('Total Trades', 0)
        pf = m.get('Profit Factor', 0)
        sharpe = m.get('Sharpe Ratio', 0)
        max_dd = m.get('Max Drawdown (%)', 0)
        exp = m.get('Expectancy', 0)
        avg_win = m.get('Avg Win', 0)
        avg_loss = m.get('Avg Loss', 0)
        
        print(f"| **{ts}** | {trades} | {pf:.2f} | {sharpe:.2f} | {max_dd:.2f}% | ${exp:.2f} | ${avg_win:.2f} | ${avg_loss:.2f} |")

if __name__ == "__main__":
    run_study()
