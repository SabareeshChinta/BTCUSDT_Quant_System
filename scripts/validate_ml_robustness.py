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

def run_validation():
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
    feature_names = fe.get_feature_names()
    
    print("Training models across 10 random seeds...")
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
    models = []
    for seed in seeds:
        trainer = ModelTrainer(random_state=seed)
        model = trainer.train(X, y)
        models.append(model)
        
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    
    print("\n--- CONFIDENCE THRESHOLD ROBUSTNESS ---")
    config.risk.hard_drawdown_halt_pct = 0.25 # Reset back to normal
    
    table_data = []
    
    # Precompute signals and features to speed up 53 backtest iterations!
    from strategy.signal_generator import SignalGenerator
    from indicators.renko_engine import RenkoEngine
    import time
    
    print("Precomputing Renko signals and ML features for fast batch testing...")
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
    
    for thresh in thresholds:
        metrics_list = []
        for model in models:
            predictor = MLPredictor(model=model, feature_engineer=fe, confidence_threshold=thresh)
            s4_engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
            res = s4_engine.run(df_backtest, timeframe=timeframe, config_name="S4", 
                                precomputed_signals=precomputed_signals, 
                                precomputed_features=df_ml_features)
            m = res.metrics
            metrics_list.append({
                'trades': m.get('Total Trades', 0),
                'pf': m.get('Profit Factor', 0),
                'sharpe': m.get('Sharpe Ratio', 0),
                'max_dd': m.get('Max Drawdown (%)', 0),
                'exp': m.get('Expectancy', 0)
            })
            
        avg_trades = np.mean([m['trades'] for m in metrics_list])
        avg_pf = np.mean([m['pf'] for m in metrics_list])
        avg_sharpe = np.mean([m['sharpe'] for m in metrics_list])
        avg_dd = np.mean([m['max_dd'] for m in metrics_list])
        avg_exp = np.mean([m['exp'] for m in metrics_list])
        
        table_data.append((thresh, avg_trades, avg_pf, avg_sharpe, avg_dd, avg_exp))
        print(f"Thresh {thresh:.2f} | Trades: {avg_trades:.1f} | PF: {avg_pf:.2f} | Sharpe: {avg_sharpe:.2f} | MaxDD: {avg_dd:.2f}% | Exp: ${avg_exp:.2f}")

    print("\n--- FEATURE ABLATION ---")
    # Groups
    groups = {
        'No Momentum': [f for f in feature_names if f not in ['returns_1', 'returns_5', 'returns_10', 'returns_20', 'rsi_14', 'rsi_28', 'macd_line', 'macd_signal', 'macd_histogram']],
        'No Volatility': [f for f in feature_names if f not in ['bb_width', 'bb_position', 'volatility_20', 'range_pct', 'atr_ratio']],
        'No Trend': [f for f in feature_names if f not in ['adx_14', 'price_vs_sma20', 'price_vs_sma50', 'higher_high', 'lower_low']]
    }
    
    for group_name, feats in groups.items():
        X_sub = X[feats]
        trainer = ModelTrainer(random_state=42)
        model = trainer.train(X_sub, y)
        
        class AblationPredictor(MLPredictor):
            def __init__(self, model, fe, threshold, features_to_keep):
                super().__init__(model, fe, threshold)
                self.features_to_keep = features_to_keep
            def predict(self, features):
                if self.model is None or not features: return 0.5, False
                X_p = pd.DataFrame([{f: features.get(f, 0.0) for f in self.features_to_keep}])
                try: probs = self.model.predict_proba(X_p); prob_success = float(probs[0, 1])
                except: prob_success = 0.5
                return prob_success, prob_success >= self.confidence_threshold
                
        predictor = AblationPredictor(model, fe, 0.60, feats) # Using 0.60 as baseline
        engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=True, ml_predictor=predictor)
        res = engine.run(df_backtest, timeframe=timeframe, config_name="Ablation", precomputed_signals=precomputed_signals, precomputed_features=df_ml_features)
        m = res.metrics
        print(f"{group_name:<15} | Trades: {m.get('Total Trades', 0):<4} | PF: {m.get('Profit Factor', 0):.2f} | Sharpe: {m.get('Sharpe Ratio', 0):.2f} | MaxDD: {m.get('Max Drawdown (%)', 0):.2f}% | Exp: ${m.get('Expectancy', 0):.2f}")

if __name__ == "__main__":
    run_validation()
