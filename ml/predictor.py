"""
Predictor for ML Pipeline.
Real-time prediction interface used during backtesting and paper trading.
"""

import pandas as pd
import joblib

class MLPredictor:
    def __init__(self, model=None, feature_engineer=None, confidence_threshold: float = 0.6):
        self.model = model
        self.confidence_threshold = confidence_threshold
        
        if feature_engineer is None:
            from ml.feature_engineering import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = feature_engineer

    def load_model(self, filepath: str) -> None:
        self.model = joblib.load(filepath)

    def predict(self, features: dict) -> tuple[float, bool]:
        """
        Returns (probability_of_success, passes_threshold).
        If model is None, safe default is (0.5, False).
        """
        if self.model is None or not features:
            return 0.5, False
            
        feature_names = self.feature_engineer.get_feature_names()
        X = pd.DataFrame([{f: features.get(f, 0.0) for f in feature_names}])
        
        # XGBoost predict_proba
        try:
            probs = self.model.predict_proba(X)
            # Probability of class 1
            prob_success = float(probs[0, 1])
        except Exception:
            prob_success = 0.5
            
        passes = prob_success >= self.confidence_threshold
        return prob_success, passes

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Batch prediction. Returns df with probability and pass/fail.
        """
        out = pd.DataFrame(index=X.index)
        if self.model is None or X.empty:
            out['probability'] = 0.5
            out['passes_threshold'] = False
            return out
            
        feature_names = self.feature_engineer.get_feature_names()
        X_filtered = X[[f for f in feature_names if f in X.columns]].copy()
        
        # Fill missing with 0 for batch prediction
        for f in feature_names:
            if f not in X_filtered.columns:
                X_filtered[f] = 0.0
                
        probs = self.model.predict_proba(X_filtered)
        prob_success = probs[:, 1]
        
        out['probability'] = prob_success
        out['passes_threshold'] = prob_success >= self.confidence_threshold
        
        return out
