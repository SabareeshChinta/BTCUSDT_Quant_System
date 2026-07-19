"""
Dataset Builder for ML Pipeline.
Builds labeled datasets from backtest trade outcomes.
"""

import pandas as pd
import numpy as np

class DatasetBuilder:
    def __init__(self, feature_engineer=None):
        if feature_engineer is None:
            from ml.feature_engineering import FeatureEngineer
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = feature_engineer

    def build_from_trades(self, trades_df: pd.DataFrame, price_df: pd.DataFrame, atr_series: pd.Series = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build features X and labels y from trades_df.
        Label: 1 if net_pnl > 0, 0 otherwise.
        """
        if trades_df.empty:
            return pd.DataFrame(), pd.Series(dtype=int)

        if hasattr(price_df.index, 'tz') and price_df.index.tz is not None:
            price_df = price_df.copy()
            price_df.index = price_df.index.tz_localize(None)

        # Precompute all features for price_df
        df_feat = self.feature_engineer.compute_features(price_df, atr_series)
        feature_names = self.feature_engineer.get_feature_names()

        # We will match each trade's entry_time to the latest feature row <= entry_time
        if 'Open Time' in df_feat.columns:
            df_feat = df_feat.set_index('Open Time')
        df_feat = df_feat.sort_index()
        
        X_list = []
        y_list = []

        for _, trade in trades_df.iterrows():
            if 'sig_time' not in trade:
                continue
                
            sig_time = trade['sig_time']
            if pd.isna(sig_time):
                continue
            
            # Check if features are already in the trade record (from context)
            features = {}
            has_all_features = True
            for f in feature_names:
                if f in trade and pd.notna(trade[f]):
                    features[f] = trade[f]
                else:
                    has_all_features = False
                    break
            
            if not has_all_features:
                if hasattr(sig_time, 'tzinfo') and sig_time.tzinfo is not None:
                    sig_time = sig_time.tz_localize(None)
                    
                # Compute on the fly if missing
                idx = df_feat.index[df_feat.index <= sig_time]
                if len(idx) == 0:
                    continue
                
                # Note: To fully compute mentor features on the fly, we need target_distance/stop_distance.
                # If they aren't in the trade record, we must estimate or skip.
                # Ideally, the backtest engine should always provide them now.
                # If missing, we fall back to 0 (suboptimal, but prevents crash).
                features = self.feature_engineer.compute_trade_context(
                    df_features=df_feat.reset_index(),
                    signal_time=sig_time,
                    direction=trade.get('direction', 1),
                    entry_price=trade.get('entry_price', 0),
                    sl_distance=trade.get('stop_distance', 0),
                    target_distance=trade.get('stop_distance', 0) * 1.5, # Estimate
                    brick_size=0
                )
            
            label = 1 if trade['net_pnl'] > 0 else 0
            
            X_list.append(features)
            y_list.append(label)

        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)

        # Drop NaNs
        valid_idx = X.dropna().index
        X = X.loc[valid_idx].reset_index(drop=True)
        y = y.loc[valid_idx].reset_index(drop=True)

        return X, y

    def build_walk_forward_splits(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, min_train_size: int = 50) -> list[tuple]:
        """
        Time-series aware walk-forward splits. No shuffling.
        Returns list of (train_idx, test_idx).
        """
        splits = []
        n_samples = len(X)
        if n_samples < min_train_size * 2:
            return splits # Not enough data
            
        test_size = (n_samples - min_train_size) // n_splits
        
        for i in range(n_splits):
            train_end = min_train_size + i * test_size
            test_end = train_end + test_size if i < n_splits - 1 else n_samples
            
            train_idx = list(range(0, train_end))
            test_idx = list(range(train_end, test_end))
            splits.append((train_idx, test_idx))
            
        return splits

    def build_sentiment_dataset(self, price_df: pd.DataFrame, sentiment_snapshots: list) -> pd.DataFrame:
        """
        Builds a dataset specifically for sentiment reaction modeling.
        Generates targets: target_1h, target_4h, target_24h based on future price movement.
        """
        if price_df.empty or not sentiment_snapshots:
            return pd.DataFrame()
            
        df = price_df.copy()
        if 'Open Time' in df.columns:
            df = df.set_index('Open Time')
        df = df.sort_index()
        
        dataset = []
        for snap in sentiment_snapshots:
            ts = snap.timestamp
            if ts not in df.index:
                # Find nearest previous price
                idx = df.index[df.index <= ts]
                if len(idx) == 0:
                    continue
                ts_price = idx[-1]
            else:
                ts_price = ts
                
            current_price = df.loc[ts_price, 'Close']
            
            # Find future prices for targets (1h, 4h, 24h)
            target_1h_ts = ts + pd.Timedelta(hours=1)
            target_4h_ts = ts + pd.Timedelta(hours=4)
            target_24h_ts = ts + pd.Timedelta(hours=24)
            
            def get_future_move(future_ts):
                future_idx = df.index[df.index >= future_ts]
                if len(future_idx) > 0:
                    fut_price = df.loc[future_idx[0], 'Close']
                    if fut_price > current_price * 1.001: return 'UP'
                    if fut_price < current_price * 0.999: return 'DOWN'
                return 'NO_MOVE'
                
            dataset.append({
                'timestamp': ts,
                'sentiment_score': snap.sentiment_score,
                'confidence': snap.confidence,
                'probability_up': snap.probability_up,
                'probability_down': snap.probability_down,
                'bullish_ratio': snap.bullish_ratio,
                'bearish_ratio': snap.bearish_ratio,
                'open_price': df.loc[ts_price, 'Open'],
                'high_price': df.loc[ts_price, 'High'],
                'low_price': df.loc[ts_price, 'Low'],
                'close_price': current_price,
                'volume': df.loc[ts_price, 'Volume'] if 'Volume' in df.columns else 0,
                'target_1h': get_future_move(target_1h_ts),
                'target_4h': get_future_move(target_4h_ts),
                'target_24h': get_future_move(target_24h_ts)
            })
            
        return pd.DataFrame(dataset)
