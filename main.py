"""
Main CLI entry point for the BTCUSDT Quant System.
"""

import argparse
import sys
import os
import pandas as pd

from config.settings import SystemConfig
from data.downloader import download_all_timeframes
from data.parquet_manager import load_data, split_data, get_data_path

def main():
    parser = argparse.ArgumentParser(description="BTCUSDT Quant Trading System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command
    parser_down = subparsers.add_parser("download", help="Download Binance data")
    parser_down.add_argument("--verify", action="store_true", help="Verify data integrity")

    # Backtest command
    parser_bt = subparsers.add_parser("backtest", help="Run backtest")
    parser_bt.add_argument("--config", type=str, default="S1", choices=["S1", "S2", "S3", "S4"], help="Ablation config to run")
    parser_bt.add_argument("--ablation", action="store_true", help="Run all ablation configs")
    parser_bt.add_argument("--timeframe", type=str, default="1h", help="Timeframe to test")

    # Forward test command
    parser_ft = subparsers.add_parser("forward-test", help="Run forward test")
    parser_ft.add_argument("--ablation", action="store_true", help="Run all configs on forward data")
    parser_ft.add_argument("--timeframe", type=str, default="1h")

    # Train ML command
    parser_ml = subparsers.add_parser("train-ml", help="Train ML models")
    parser_ml.add_argument("--validate", action="store_true", help="Run walk-forward validation")
    parser_ml.add_argument("--timeframe", type=str, default="1h", help="Timeframe to train on")

    # Paper trade command
    parser_pt = subparsers.add_parser("paper-trade", help="Start paper trading simulation")
    parser_pt.add_argument("--timeframe", type=str, default="15m", help="Timeframe for live feed (default: 15m)")
    
    # Report command
    parser_rep = subparsers.add_parser("report", help="Generate reports")
    parser_rep.add_argument("--phase", choices=["backtest", "forward", "paper"], required=True)

    args = parser.parse_args()
    config = SystemConfig.default()

    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    if args.command == "download":
        print(f"Downloading data for {config.timeframes}...")
        download_all_timeframes("BTCUSDT", "2019-01-01", "2026-06-22", config.timeframes, data_dir)
        if args.verify:
            print("Verification logic here...")

    elif args.command == "backtest":
        print(f"Running backtest for timeframe {args.timeframe}...")
        file_path = get_data_path("BTCUSDT", args.timeframe, data_dir)
        
        try:
            df = load_data(file_path)
        except Exception as e:
            print(f"Could not load data: {e}. Try running 'python main.py download' first.")
            return

        df_backtest, df_forward, df_paper = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
        
        from backtesting.ablation_runner import AblationRunner
        runner = AblationRunner(config)
        
        if args.ablation:
            results = runner.run_all(df_backtest, args.timeframe)
            runner.save_results(results, os.path.join(os.path.dirname(__file__), 'reports', 'backtest', args.timeframe))
            print("Ablation testing complete. Reports saved.")
        else:
            print(f"Running single config: {args.config}")
            # ... run single ...

    elif args.command == "train-ml":
        print("Training ML models...")
        from ml.dataset_builder import DatasetBuilder
        from ml.model_trainer import ModelTrainer
        from backtesting.backtest_engine import BacktestEngine
        
        file_path = get_data_path("BTCUSDT", args.timeframe, data_dir)
        try:
            df = load_data(file_path)
        except Exception as e:
            print(f"Could not load data: {e}. Run 'python main.py download' first.")
            return
            
        df_backtest, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
        
        print("1. Running base strategy backtest to generate labels...")
        engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
        res = engine.run(df_backtest, timeframe=args.timeframe, config_name="S2", 
                         use_trend_filter=config.strategy.use_macro_trend_filter, 
                         use_atr_filter=config.strategy.use_volatility_filter)
        
        if res.trades_df.empty:
            print("No trades generated. Cannot train ML model.")
            return
            
        print("2. Building features and labels from trades...")
        # Get ATR series used by strategy
        from indicators.atr_engine import compute_atr
        atr_series = compute_atr(df_backtest, period=config.strategy.atr_period)
        
        builder = DatasetBuilder()
        X, y = builder.build_from_trades(res.trades_df, df_backtest, atr_series)
        print(f"Extracted {len(X)} labeled examples. Positive class ratio: {y.mean():.2%}")
        
        trainer = ModelTrainer(model_type=config.ml.model_type)
        
        if args.validate:
            print("3. Running Walk-Forward Validation...")
            val_results = trainer.cross_validate_walk_forward(X, y, n_splits=5)
            print("\nValidation Metrics (Averaged):")
            for k, v in val_results['aggregate_metrics'].items():
                print(f"  {k}: {v:.4f}")
                
        print("4. Training final model on all data...")
        model = trainer.train(X, y)
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'xgboost_model.joblib')
        
        trainer.save_model(model, model_path)
        print(f"Model saved to {model_path}")

    elif args.command == "forward-test":
        print("Running forward test on unseen data...")
        from backtesting.backtest_engine import BacktestEngine
        from ml.predictor import MLPredictor
        
        file_path = get_data_path("BTCUSDT", args.timeframe, data_dir)
        try:
            df = load_data(file_path)
        except Exception as e:
            print(f"Could not load data: {e}")
            return
            
        _, df_forward, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
        
        if df_forward.empty:
            print("Forward dataset is empty.")
            return
            
        print(f"Loaded {len(df_forward)} candles for forward test.")
        
        predictor = MLPredictor(confidence_threshold=config.ml.confidence_threshold)
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.joblib')
        try:
            predictor.load_model(model_path)
            print("Successfully loaded ML model for prediction.")
        except Exception as e:
            print(f"Failed to load ML model: {e}. Run 'python main.py train-ml' first.")
            return
            
        engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
        res = engine.run(df_forward, timeframe=args.timeframe, config_name="S4_Forward",
                         use_trend_filter=config.strategy.use_macro_trend_filter, 
                         use_atr_filter=config.strategy.use_volatility_filter)
        
        print("\n--- Forward Test Results (S4 Configuration) ---")
        print(res.summary())

    elif args.command == "paper-trade":
        print("Starting paper trade engine & webhook server...")
        from paper_trading.webhook_server import start_server
        import threading
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        
        from deployment.live_feed import run_live_feed
        run_live_feed("BTCUSDT", args.timeframe, config)

    elif args.command == "report":
        print(f"Generating reports for {args.phase} phase...")

if __name__ == "__main__":
    main()
