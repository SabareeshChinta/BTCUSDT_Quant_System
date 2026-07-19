import os
import json
import pandas as pd
import numpy as np
import joblib
from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from ml.feature_engineering import FeatureEngineer
from ml.predictor import MLPredictor
from data.parquet_manager import load_data, split_data, get_data_path

def run_forward_test(timeframe="1h"):
    print("=== FORWARD VALIDATION (2023-2025) ===")
    config = SystemConfig.default()
    
    # 1. Load the forward testing data block
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    file_path = get_data_path("BTCUSDT", timeframe, data_dir)
    df = load_data(file_path)
    
    # split_data returns df_backtest, df_forward, df_paper
    _, df_forward, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    
    if 'Open Time' not in df_forward.columns:
        df_forward = df_forward.reset_index(names='Open Time')
        
    print(f"Forward Data Size: {len(df_forward)} candles.")
    
    # 2. Load the trained ML Predictor
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgboost_final.pkl")
    if not os.path.exists(model_path):
        print("Error: Trained model not found. Run train_ml.py first.")
        return
        
    final_model = joblib.load(model_path)
    fe = FeatureEngineer()
    
    # We use the same 80th percentile threshold we discovered during training: 0.8952
    optimal_threshold = 0.8952
    predictor = MLPredictor(model=final_model, feature_engineer=fe, confidence_threshold=optimal_threshold)
    
    # 3. Run S2 (Base Strategy with Risk Management)
    print("\nRunning S2 (Base Strategy) on Forward Data...")
    s2_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
    s2_res = s2_engine.run(df_forward, timeframe=timeframe, config_name="S2", 
                           use_atr_filter=False, use_trend_filter=False, use_min_rr_filter=False)
                           
    # 4. Run S3 (ML Mentor Filter Only)
    print("Running S3 (ML Mentor Filter) on Forward Data...")
    s3_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
    s3_res = s3_engine.run(df_forward, timeframe=timeframe, config_name="S3", 
                           use_atr_filter=False, use_trend_filter=False, use_min_rr_filter=False)
                           
    # 5. Run S4 (ML + Legacy Triple-Gate Filters)
    print("Running S4 (ML + Legacy Triple-Gate) on Forward Data...")
    s4_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
    s4_res = s4_engine.run(df_forward, timeframe=timeframe, config_name="S4", 
                           use_atr_filter=True, use_trend_filter=True, use_min_rr_filter=True)
                           
    print("\n--- FORWARD TEST PERFORMANCE (UNSEEN 2023-2025 DATA) ---")
    for name, res in [("S2 (Base RM)", s2_res), ("S3 (ML Mentor)", s3_res), ("S4 (ML + Legacy)", s4_res)]:
        m = res.metrics
        print(f"{name:<20} | Trades: {m.get('Total Trades', 0):<4} | "
              f"Ret: {m.get('Return (%)', 0):>6.2f}% | MaxDD: {m.get('Max Drawdown (%)', 0):>5.2f}% | "
              f"PF: {m.get('Profit Factor', 0):>4.2f} | Sharpe: {m.get('Sharpe Ratio', 0):>5.2f}")
              
if __name__ == "__main__":
    run_forward_test("1h")
