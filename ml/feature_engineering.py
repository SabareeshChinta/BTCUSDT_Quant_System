"""
Feature Engineering for ML Pipeline.
Computes precise trade context features based on the mentor's institutional framework.
"""

import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame, atr_series: pd.Series = None) -> pd.DataFrame:
        """Computes foundational indicators for context generation."""
        df_feat = df.copy()
        close = df_feat['Close']
        high = df_feat['High']
        low = df_feat['Low']

        # 1. 200 EMA (Trend)
        df_feat['ema_200'] = close.ewm(span=200, adjust=False).mean()
        
        # 2. RSI 14 (Momentum)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = -delta.clip(upper=0).ewm(com=13, min_periods=14).mean()
        rs = gain / loss.replace(0, 1e-8)
        df_feat['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 3. ATR 14 (Volatility)
        if atr_series is not None:
            df_feat['atr_14'] = atr_series
        else:
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_feat['atr_14'] = tr.ewm(alpha=1/14, min_periods=14).mean()
            
        # 4. ATR 20 SMA (Volatility Baseline)
        df_feat['atr_sma_20'] = df_feat['atr_14'].rolling(20).mean()

        # 5. ADX (Market Regime proxy)
        up_move = high.diff()
        down_move = -low.diff()
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        atr14 = df_feat['atr_14']
        pos_di = 100 * pd.Series(pos_dm, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / atr14
        neg_di = 100 * pd.Series(neg_dm, index=df.index).ewm(alpha=1/14, min_periods=14).mean() / atr14
        dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-8).abs()
        df_feat['adx_14'] = dx.ewm(alpha=1/14, min_periods=14).mean()
        
        # 6. Fast & Slow MA for Slope (Assume standard 10/20 on close as proxy)
        df_feat['sma_10'] = close.rolling(10).mean()
        df_feat['sma_20'] = close.rolling(20).mean()

        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df_feat

    def get_feature_names(self) -> list[str]:
        return [
            'direction', 'brick_size', 'ma_slope', 'market_regime', 
            'trend_strength', 'momentum', 'volatility', 'risk_pressure', 
            'reward_pressure', 'signal_quality', 'sl_points', 'tp_points', 'rr_ratio',
            'sentiment_score', 'confidence', 'bullish_ratio', 'bearish_ratio',
            'probability_up', 'probability_down'
        ]

    def compute_trade_context(self, df_features: pd.DataFrame, signal_time: pd.Timestamp, 
                              direction: int, entry_price: float, sl_distance: float, target_distance: float, 
                              brick_size: float, sentiment_snapshot: dict = None) -> dict:
        """
        Calculates the exact mentor features at the moment of trade entry.
        """
        if 'Open Time' in df_features.columns:
            mask = df_features['Open Time'] <= signal_time
        else:
            mask = df_features.index <= signal_time
            
        if not mask.any():
            return {f: 0.0 for f in self.get_feature_names()}
            
        row = df_features[mask].iloc[-1]
        prev_row = df_features[mask].iloc[-2] if sum(mask) > 1 else row
        
        # Compute Mentor Features
        # 1. direction, brick_size
        f_dir = float(direction)
        f_brick = brick_size
        
        # 2. ma_slope: Rate of change of the 20 SMA
        f_ma_slope = (row['sma_20'] - prev_row['sma_20']) / prev_row['sma_20'] * 100.0 if prev_row['sma_20'] != 0 else 0.0
        
        # 3. market_regime: 1 if ADX > 25 (Trending), 0 if ADX <= 25 (Ranging)
        f_regime = 1.0 if row['adx_14'] > 25 else 0.0
        
        # 4. trend_strength: Distance from 200 EMA
        f_trend = (entry_price - row['ema_200']) / row['ema_200'] * 100.0 if row['ema_200'] != 0 else 0.0
        
        # 5. momentum: RSI
        f_mom = row['rsi_14']
        
        # 6. volatility: Current ATR ratio relative to baseline
        f_vol = row['atr_14'] / row['atr_sma_20'] if row['atr_sma_20'] > 0 else 1.0
        
        # 7. risk_pressure & reward_pressure: Normalized to ATR
        f_risk_pres = sl_distance / row['atr_14'] if row['atr_14'] > 0 else 0.0
        f_rew_pres = target_distance / row['atr_14'] if row['atr_14'] > 0 else 0.0
        
        # 8. signal_quality: Composite score (Trend alignment * Momentum)
        # If long, we want positive trend & RSI > 50. If short, negative trend & RSI < 50.
        rsi_centered = row['rsi_14'] - 50.0
        f_quality = (f_trend * rsi_centered) * direction
        
        # 9. sl, tp, rr
        f_sl = sl_distance
        f_tp = target_distance
        f_rr = target_distance / sl_distance if sl_distance > 0 else 0.0
        
        features = {
            'direction': f_dir,
            'brick_size': f_brick,
            'ma_slope': f_ma_slope,
            'market_regime': f_regime,
            'trend_strength': f_trend,
            'momentum': f_mom,
            'volatility': f_vol,
            'risk_pressure': f_risk_pres,
            'reward_pressure': f_rew_pres,
            'signal_quality': f_quality,
            'sl_points': f_sl,
            'tp_points': f_tp,
            'rr_ratio': f_rr,
            'sentiment_score': sentiment_snapshot.get('sentiment_score', 0.0) if sentiment_snapshot else 0.0,
            'confidence': sentiment_snapshot.get('confidence', 0.0) if sentiment_snapshot else 0.0,
            'bullish_ratio': sentiment_snapshot.get('bullish_ratio', 0.0) if sentiment_snapshot else 0.0,
            'bearish_ratio': sentiment_snapshot.get('bearish_ratio', 0.0) if sentiment_snapshot else 0.0,
            'probability_up': sentiment_snapshot.get('probability_up', 0.0) if sentiment_snapshot else 0.0,
            'probability_down': sentiment_snapshot.get('probability_down', 0.0) if sentiment_snapshot else 0.0
        }
        
        return features
