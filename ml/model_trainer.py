"""
Model Trainer for ML Pipeline.
Trains XGBoost models on built datasets.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

class ModelTrainer:
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> object:
        # Handle class imbalance
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

        if X_val is not None and y_val is not None:
            model = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1.0,
                reg_lambda=1.0,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=self.random_state,
                eval_metric='logloss',
                early_stopping_rounds=5
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=1.0,
                reg_lambda=1.0,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train, verbose=False)

        return model

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        
        cm = confusion_matrix(y_test, preds).tolist()
        cr = classification_report(y_test, preds, zero_division=0)

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'classification_report': cr
        }

    def cross_validate_walk_forward(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        from ml.dataset_builder import DatasetBuilder
        builder = DatasetBuilder()
        splits = builder.build_walk_forward_splits(X, y, n_splits=n_splits)

        fold_metrics = []
        for train_idx, test_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

            # Use last 20% of train for early stopping
            val_size = int(len(X_tr) * 0.2)
            if val_size > 0:
                X_t, y_t = X_tr.iloc[:-val_size], y_tr.iloc[:-val_size]
                X_v, y_v = X_tr.iloc[-val_size:], y_tr.iloc[-val_size:]
                model = self.train(X_t, y_t, X_v, y_v)
            else:
                model = self.train(X_tr, y_tr)

            metrics = self.evaluate(model, X_te, y_te)
            fold_metrics.append(metrics)

        # Aggregate metrics
        agg = {k: np.mean([f[k] for f in fold_metrics if isinstance(f[k], (int, float))]) for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))}
        
        return {
            'fold_metrics': fold_metrics,
            'aggregate_metrics': agg
        }

    def get_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        importances = model.feature_importances_
        df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save_model(self, model, filepath: str) -> None:
        joblib.dump(model, filepath)

    def load_model(self, filepath: str) -> object:
        return joblib.load(filepath)
