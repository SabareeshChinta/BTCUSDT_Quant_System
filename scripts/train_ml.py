import os
import json
import pandas as pd
import numpy as np
from config.settings import SystemConfig
from backtesting.backtest_engine import BacktestEngine
from ml.feature_engineering import FeatureEngineer
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor
from data.parquet_manager import load_data, split_data, get_data_path

def train_and_evaluate(timeframe="1h"):
    config = SystemConfig.default()
    
    # Disable hard halt for data generation to get all trades
    config.risk.hard_drawdown_halt_pct = 1.0
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    file_path = get_data_path("BTCUSDT", timeframe, data_dir)
    df = load_data(file_path)
    df_backtest, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    if 'Open Time' not in df_backtest.columns:
        df_backtest = df_backtest.reset_index(names='Open Time')
    
    # 1. Run S2 Strategy (Risk Management ON, no ML) to get dataset trades
    print(f"Running S2 backtest on {timeframe} to generate trade log for ML training...")
    engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
    # We don't use the triple filters here so we can give the ML model a rich dataset of 100+ trades to learn from
    s2_res = engine.run(df_backtest, timeframe=timeframe, config_name="S2", use_atr_filter=False, use_trend_filter=False, use_min_rr_filter=False)
    trades_df = s2_res.trades_df
    print(f"Total S2 trades generated for training: {len(trades_df)}")
    
    if len(trades_df) == 0:
        print("No trades generated. Aborting ML training.")
        return
        
    # 2 & 3. Build Features and Dataset
    print("Building leakage-free Mentor Context ML dataset...")
    fe = FeatureEngineer()
    db = DatasetBuilder(feature_engineer=fe)
    
    from indicators.atr_engine import compute_atr
    atr_series = compute_atr(df_backtest, period=config.strategy.atr_period).values
    X, y = db.build_from_trades(trades_df, df_backtest, atr_series=atr_series)
    feature_names = fe.get_feature_names()
    
    print(f"Dataset built: {len(X)} samples.")
    if len(X) == 0:
        return
    print(f"Class balance (Positives): {y.mean():.2%}")
    
    # 4. Walk-Forward Fold Training
    print("\n--- Walk-Forward Validation Fold Metrics ---")
    splits = db.build_walk_forward_splits(X, y, n_splits=5, min_train_size=max(10, len(X)//5))
    
    fold_metrics = []
    trainer = ModelTrainer(random_state=42)
    
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        if len(X_train) == 0 or len(X_test) == 0:
            continue
        model = trainer.train(X_train, y_train)
        metrics = trainer.evaluate(model, X_test, y_test)
        fold_metrics.append(metrics)
        print(f"Fold {i+1} | Train: {len(X_train)} Test: {len(X_test)} | "
              f"Acc: {metrics['accuracy']:.2f} Precision: {metrics['precision']:.2f} "
              f"ROC_AUC: {metrics['auc_roc']:.2f}")
              
    # 5. Train Final Model on all data
    print("\nTraining final model on all data...")
    final_model = trainer.train(X, y)
    
    probs = final_model.predict_proba(X)[:, 1]
    optimal_threshold = float(np.percentile(probs, 80))
    print(f"Dynamic ML threshold (80th percentile): {optimal_threshold:.4f}")
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"), exist_ok=True)
    import joblib
    joblib.dump(final_model, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgboost_final.pkl"))
    
    # 6. Run Ablation: Compare S2 (no filters) vs S3 (ML only) vs S4 (ML + All Filters)
    config.risk.hard_drawdown_halt_pct = 0.25 # Restore 25% halt limit
    predictor = MLPredictor(model=final_model, feature_engineer=fe, confidence_threshold=optimal_threshold)
    
    print("\nRunning S3 (ML Mentor Filter Only)...")
    s3_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
    s3_res = s3_engine.run(df_backtest, timeframe=timeframe, config_name="S3", use_atr_filter=False, use_trend_filter=False, use_min_rr_filter=False)
    
    print("Running S4 (ML Mentor Filter + Triple-Gate Legacy Filters)...")
    s4_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
    s4_res = s4_engine.run(df_backtest, timeframe=timeframe, config_name="S4", use_atr_filter=True, use_trend_filter=True, use_min_rr_filter=True)
    
    print("\n--- ML UPGRADE PERFORMANCE COMPARISON ---")
    for name, res in [("S2 (Base RM)", s2_res), ("S3 (ML Mentor)", s3_res), ("S4 (ML + Legacy)", s4_res)]:
        m = res.metrics
        print(f"{name:<20} | Trades: {m.get('Total Trades', 0):<4} | "
              f"Ret: {m.get('Return (%)', 0):>6.2f}% | MaxDD: {m.get('Max Drawdown (%)', 0):>5.2f}% | "
              f"PF: {m.get('Profit Factor', 0):>4.2f} | Sharpe: {m.get('Sharpe Ratio', 0):>5.2f}")
              
    # Save Report Data
    report = {
        "train_samples": len(X),
        "class_balance_pos": float(y.mean()),
        "feature_importance": dict(zip(feature_names, final_model.feature_importances_.tolist())),
    }
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ml_report.json"), "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    train_and_evaluate("1h")
