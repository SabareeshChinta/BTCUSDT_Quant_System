import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor
from indicators.atr_engine import compute_atr

def run_all():
    timeframes = ["15m", "30m", "1h", "4h"]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    config = SystemConfig.default()
    
    print(f"{'Timeframe':<10} | {'Strategy':<12} | {'Phase':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Max DD (%)':<10} | {'Net PnL':<10}")
    print("-" * 90)
    
    for tf in timeframes:
        file_path = get_data_path("BTCUSDT", tf, data_dir)
        try:
            df = load_data(file_path)
            df_bt, df_ft, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
            
            # --- Base Strategy ---
            # 1. Base Backtest
            engine_base_bt = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
            res_bt_base = engine_base_bt.run(df_bt, timeframe=tf, config_name="Base_BT")
            
            if res_bt_base.trades_df.empty:
                print(f"No trades generated for {tf}. Skipping.")
                continue
                
            bt_trades = len(res_bt_base.trades_df)
            bt_wr = res_bt_base.metrics.get('Win Rate', 0)
            bt_dd = res_bt_base.metrics.get('Max Drawdown (%)', 0)
            bt_pnl = res_bt_base.metrics.get('Net PnL', 0)
            print(f"{tf:<10} | {'Base':<12} | {'Backtest':<10} | {bt_trades:<8} | {bt_wr:<10.1%} | {bt_dd:<9.2f}% | ${bt_pnl:<9.2f}")
            
            # 2. Base Forward
            if not df_ft.empty:
                engine_base_ft = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
                res_ft_base = engine_base_ft.run(df_ft, timeframe=tf, config_name="Base_FT")
                ft_trades = len(res_ft_base.trades_df)
                ft_wr = res_ft_base.metrics.get('Win Rate', 0)
                ft_dd = res_ft_base.metrics.get('Max Drawdown (%)', 0)
                ft_pnl = res_ft_base.metrics.get('Net PnL', 0)
                print(f"{tf:<10} | {'Base':<12} | {'Forward':<10} | {ft_trades:<8} | {ft_wr:<10.1%} | {ft_dd:<9.2f}% | ${ft_pnl:<9.2f}")

            # --- ML Strategy ---
            # 3. Train ML Model
            atr_series = compute_atr(df_bt, period=config.strategy.atr_period)
            builder = DatasetBuilder()
            X, y = builder.build_from_trades(res_bt_base.trades_df, df_bt, atr_series)
            
            if len(X) < 10:
                print(f"Not enough training data for {tf}. Skipping ML.")
                print("-" * 90)
                continue
                
            trainer = ModelTrainer(model_type=config.ml.model_type)
            model = trainer.train(X, y)
            
            model_path = os.path.join(os.path.dirname(__file__), f'xgboost_{tf}.joblib')
            trainer.save_model(model, model_path)
            
            predictor = MLPredictor(confidence_threshold=config.ml.confidence_threshold)
            predictor.load_model(model_path)
            
            # 4. ML Backtest
            engine_ml_bt = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
            res_bt_ml = engine_ml_bt.run(df_bt, timeframe=tf, config_name="ML_Backtest")
            
            bt_trades = len(res_bt_ml.trades_df)
            bt_wr = res_bt_ml.metrics.get('Win Rate', 0)
            bt_dd = res_bt_ml.metrics.get('Max Drawdown (%)', 0)
            bt_pnl = res_bt_ml.metrics.get('Net PnL', 0)
            print(f"{tf:<10} | {'ML':<12} | {'Backtest':<10} | {bt_trades:<8} | {bt_wr:<10.1%} | {bt_dd:<9.2f}% | ${bt_pnl:<9.2f}")
            
            # 5. ML Forward
            if not df_ft.empty:
                engine_ml_ft = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
                res_ft_ml = engine_ml_ft.run(df_ft, timeframe=tf, config_name="ML_Forward")
                ft_trades = len(res_ft_ml.trades_df)
                ft_wr = res_ft_ml.metrics.get('Win Rate', 0)
                ft_dd = res_ft_ml.metrics.get('Max Drawdown (%)', 0)
                ft_pnl = res_ft_ml.metrics.get('Net PnL', 0)
                print(f"{tf:<10} | {'ML':<12} | {'Forward':<10} | {ft_trades:<8} | {ft_wr:<10.1%} | {ft_dd:<9.2f}% | ${ft_pnl:<9.2f}")
            
            print("-" * 90)
            
        except Exception as e:
            print(f"Error on {tf}: {e}")

if __name__ == '__main__':
    run_all()
