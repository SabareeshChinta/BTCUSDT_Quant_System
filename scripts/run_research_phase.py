import os
import sys
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from ml.dataset_builder import DatasetBuilder
from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor
from indicators.atr_engine import compute_atr

from research.walk_forward import WalkForwardEngine
from research.monte_carlo import MonteCarloSimulator
from research.robustness_tester import RobustnessTester
from research.parameter_stability import ParameterStabilityTester

def run_research():
    timeframes = ["15m", "1h"]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    config = SystemConfig.default()
    
    report = {
        "WalkForward": {},
        "Robustness": {},
        "ParameterStability": {}
    }
    
    for tf in timeframes:
        file_path = get_data_path("BTCUSDT", tf, data_dir)
        try:
            df = load_data(file_path)
        except:
            continue
            
        print(f"--- Starting Research for {tf} ---")
        
        # 1. Walk Forward
        wf_engine = WalkForwardEngine(df, train_years=2, test_years=1)
        windows = wf_engine.get_windows()
        
        tf_results = []
        for df_train, df_test, train_name, test_name in windows:
            print(f"  Window Train: {train_name} | Test: {test_name}")
            
            # Base Train (just to get labels for ML)
            engine_base = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
            res_train_base = engine_base.run(df_train, timeframe=tf, config_name="Base_Train")
            
            # Base Test
            engine_base_test = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
            res_test_base = engine_base_test.run(df_test, timeframe=tf, config_name="Base_Test")
            
            # ML Train
            atr_series = compute_atr(df_train, period=config.strategy.atr_period)
            builder = DatasetBuilder()
            X, y = builder.build_from_trades(res_train_base.trades_df, df_train, atr_series)
            
            ml_metrics = None
            if len(X) >= 10:
                trainer = ModelTrainer(model_type=config.ml.model_type)
                model = trainer.train(X, y)
                model_path = os.path.join(os.path.dirname(__file__), f'xgboost_research.joblib')
                trainer.save_model(model, model_path)
                
                predictor = MLPredictor(confidence_threshold=config.ml.confidence_threshold)
                predictor.load_model(model_path)
                
                # ML Test
                engine_ml_test = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
                res_test_ml = engine_ml_test.run(df_test, timeframe=tf, config_name="ML_Test")
                
                # SQN & Monte Carlo for ML Test
                mc_ml = MonteCarloSimulator(res_test_ml.trades_df)
                ml_sqn = mc_ml.calculate_sqn()
                ml_mc = mc_ml.run_simulations()
                
                ml_metrics = {
                    "Trades": len(res_test_ml.trades_df),
                    "WinRate": res_test_ml.metrics.get('Win Rate', 0),
                    "NetPnL": res_test_ml.metrics.get('Net PnL', 0),
                    "MaxDD": res_test_ml.metrics.get('Max Drawdown (%)', 0),
                    "SQN": ml_sqn,
                    "MC_Profit_5th": ml_mc.get('mc_profit_5th', 0),
                    "MC_DD_95th": ml_mc.get('mc_dd_95th', 0)
                }
            
            # SQN & Monte Carlo for Base Test
            mc_base = MonteCarloSimulator(res_test_base.trades_df)
            base_sqn = mc_base.calculate_sqn()
            base_mc = mc_base.run_simulations()
            
            base_metrics = {
                "Trades": len(res_test_base.trades_df),
                "WinRate": res_test_base.metrics.get('Win Rate', 0),
                "NetPnL": res_test_base.metrics.get('Net PnL', 0),
                "MaxDD": res_test_base.metrics.get('Max Drawdown (%)', 0),
                "SQN": base_sqn,
                "MC_Profit_5th": base_mc.get('mc_profit_5th', 0),
                "MC_DD_95th": base_mc.get('mc_dd_95th', 0)
            }
            
            tf_results.append({
                "Train": train_name,
                "Test": test_name,
                "Base": base_metrics,
                "ML": ml_metrics
            })
            
        report["WalkForward"][tf] = tf_results
        
        # 2. Robustness (on last test window for simplicity)
        if windows:
            _, df_last_test, _, _ = windows[-1]
            def factory(c): return BacktestEngine(config=c, use_risk_management=True, use_ml_filter=False)
            rt = RobustnessTester(factory, df_last_test, tf)
            rob_res = rt.test_all_scenarios(config)
            report["Robustness"][tf] = rob_res
            
        # 3. Parameter Stability (only on 15m to save time)
        if tf == "15m" and windows:
            _, df_last_test, _, _ = windows[-1]
            pst = ParameterStabilityTester(df_last_test, tf)
            ps_res = pst.test_grid(config)
            report["ParameterStability"][tf] = ps_res

    # Save report
    with open("research_report_raw.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Research Phase Complete! Output saved to research_report_raw.json")

if __name__ == '__main__':
    run_research()
