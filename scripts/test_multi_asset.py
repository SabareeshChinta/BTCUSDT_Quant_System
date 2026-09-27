import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = r"C:\Users\chint\BTCUSDT_Quant_System"
sys.path.append(project_root)

from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor
from indicators.atr_engine import compute_atr

def run_multi_asset_test():
    assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    timeframes = ["15m", "30m", "1h", "4h"]
    data_dir = os.path.join(project_root, 'data', 'raw')
    
    print(f"{'Asset':<10} | {'TF':<4} | {'Phase':<12} | {'Trades':<8} | {'Win Rate':<10} | {'Max DD (%)':<10} | {'Net PnL':<10}")
    print("-" * 88)
    
    total_portfolio_pnl = 0.0
    total_portfolio_trades = 0
    
    for tf in timeframes:
        for asset in assets:
            config = SystemConfig.default()
            
            try:
                file_path = get_data_path(asset, tf, data_dir)
                if not os.path.exists(file_path):
                    print(f"{asset:<10} | {tf:<4} | {'ERROR':<12} | Data file not found.")
                    continue
                    
                df = load_data(file_path)
                df_bt, df_ft, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
                
                # 1. Base Backtest
                engine_base_bt = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                res_bt_base = engine_base_bt.run(df_bt, timeframe=tf, config_name=f"Base_BT_{asset}")
                
                if res_bt_base.trades_df.empty:
                    print(f"{asset:<10} | {tf:<4} | {'Base BT':<12} | 0 trades.")
                    continue
                    
                # 2. Train ML Model
                atr_series = compute_atr(df_bt, period=config.strategy.atr_period)
                builder = DatasetBuilder()
                X, y = builder.build_from_trades(res_bt_base.trades_df, df_bt, atr_series)
                
                if len(X) < 10:
                    print(f"{asset:<10} | {tf:<4} | {'ML Forward':<12} | Not enough training data for ML ({len(X)} samples).")
                    continue
                    
                trainer = ModelTrainer(model_type=config.ml.model_type)
                model = trainer.train(X, y)
                
                model_path = os.path.join(os.path.dirname(__file__), f'xgboost_multi_{asset}_{tf}.joblib')
                trainer.save_model(model, model_path)
                
                predictor = MLPredictor(confidence_threshold=config.ml.confidence_threshold)
                predictor.load_model(model_path)
                
                # 3. ML Forward Test (Out of Sample)
                if not df_ft.empty:
                    engine_ml_ft = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
                    res_ft_ml = engine_ml_ft.run(df_ft, timeframe=tf, config_name=f"ML_FT_{asset}")
                    
                    ft_trades = len(res_ft_ml.trades_df)
                    ft_wr = res_ft_ml.metrics.get('Win Rate', 0)
                    ft_dd = res_ft_ml.metrics.get('Max Drawdown (%)', 0)
                    ft_pnl = res_ft_ml.metrics.get('Net PnL', 0)
                    
                    print(f"{asset:<10} | {tf:<4} | {'ML Forward':<12} | {ft_trades:<8} | {ft_wr:<10.1%} | {ft_dd:<9.2f}% | ${ft_pnl:<9.2f}")
                    
                    if tf == "15m":
                        total_portfolio_pnl += ft_pnl
                        total_portfolio_trades += ft_trades
                        
            except Exception as e:
                print(f"Error on {asset} {tf}: {e}")
                
        print("-" * 88)
        
    print(f"\n--- PORTFOLIO SUMMARY (15m Timeframe) ---")
    print(f"Total Combined Forward Test Trades: {total_portfolio_trades}")
    print(f"Total Combined Forward Test Net PnL: ${total_portfolio_pnl:.2f}")

if __name__ == '__main__':
    run_multi_asset_test()
