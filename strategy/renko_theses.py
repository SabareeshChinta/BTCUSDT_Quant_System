import pandas as pd
import numpy as np

class BaseRenkoThesis:
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

def _format_renko_signals(bricks: pd.DataFrame, signal_series: pd.Series) -> pd.DataFrame:
    """Format signals into structure expected by BacktestEngine"""
    signals_df = pd.DataFrame(index=bricks.index)
    signals_df['brick_close_time'] = bricks['brick_close_time']
    
    # Map 1 to BUY, -1 to SELL, 0 to NEUTRAL
    signal_strs = np.where(signal_series == 1, 'BUY', np.where(signal_series == -1, 'SELL', 'NEUTRAL'))
    signals_df['raw_signal'] = signal_strs
    
    # Trigger when signal changes
    shifted = signal_series.shift(1).fillna(0)
    signals_df['signal_changed'] = (signal_series != 0) & (signal_series != shifted)
    
    return signals_df

class RenkoDonchianBreakout(BaseRenkoThesis):
    """
    1. Renko Donchian Breakout
    Buy when a new brick forms at a price higher than the highest brick of the last 20 bricks.
    Sell when lower than lowest.
    """
    def __init__(self, period=20):
        self.period = period
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        # shifted by 1 to not include the current brick in the lookback
        highs = bricks['brick_close'].rolling(self.period).max().shift(1)
        lows = bricks['brick_close'].rolling(self.period).min().shift(1)
        
        signal = pd.Series(0, index=bricks.index)
        signal[bricks['brick_close'] > highs] = 1
        signal[bricks['brick_close'] < lows] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)

class RenkoMomentumContinuation(BaseRenkoThesis):
    """
    2. Renko Momentum Continuation
    Buy after 4 consecutive green bricks. Sell after 4 consecutive red bricks.
    """
    def __init__(self, consec_bricks=4):
        self.consec_bricks = consec_bricks
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        # Create consecutive count
        dir_changed = bricks['direction'] != bricks['direction'].shift(1)
        group = dir_changed.cumsum()
        consec_count = bricks.groupby(group)['direction'].cumcount() + 1
        
        signal = pd.Series(0, index=bricks.index)
        signal[(bricks['direction'] == 1) & (consec_count >= self.consec_bricks)] = 1
        signal[(bricks['direction'] == -1) & (consec_count >= self.consec_bricks)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)

class RenkoVelocityExpansion(BaseRenkoThesis):
    """
    3. Renko Velocity (Time-Based) Expansion
    Buy if a green brick forms in < 25% of the avg time of the last 10 bricks.
    Sell if a red brick forms in < 25%.
    """
    def __init__(self, avg_period=10, velocity_threshold=0.25):
        self.avg_period = avg_period
        self.velocity_threshold = velocity_threshold
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        brick_times = pd.to_datetime(bricks['brick_close_time'])
        # Time taken to form the brick in seconds
        time_delta = brick_times.diff().dt.total_seconds()
        
        # Avg time of the PREVIOUS 10 bricks
        avg_time = time_delta.rolling(self.avg_period).mean().shift(1)
        
        # Velocity burst: current time is much smaller than average
        burst = time_delta < (avg_time * self.velocity_threshold)
        
        signal = pd.Series(0, index=bricks.index)
        signal[burst & (bricks['direction'] == 1)] = 1
        signal[burst & (bricks['direction'] == -1)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)

class RenkoTrendStrengthAlignment(BaseRenkoThesis):
    """
    4. Renko Trend Strength Alignment
    Standard 2-brick reversal, but strictly filtered by a 50-brick SMA.
    """
    def __init__(self, sma_period=50):
        self.sma_period = sma_period
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        sma = bricks['brick_close'].rolling(self.sma_period).mean()
        
        dir_changed = bricks['direction'] != bricks['direction'].shift(1)
        group = dir_changed.cumsum()
        consec_count = bricks.groupby(group)['direction'].cumcount() + 1
        
        reversal = consec_count >= 2
        
        signal = pd.Series(0, index=bricks.index)
        signal[reversal & (bricks['direction'] == 1) & (bricks['brick_close'] > sma)] = 1
        signal[reversal & (bricks['direction'] == -1) & (bricks['brick_close'] < sma)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)

class RenkoRegimeDetection(BaseRenkoThesis):
    """
    5. Renko Regime (Brick Ratio) Detection
    Ratio of green/red bricks over last 20.
    >70% green or <30% green -> Trending -> Trade Continuations (consec > 3)
    30% - 70% -> Ranging -> Trade Reversals (consec = 2)
    """
    def __init__(self, lookback=20, trend_threshold=0.70):
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        dir_changed = bricks['direction'] != bricks['direction'].shift(1)
        group = dir_changed.cumsum()
        consec_count = bricks.groupby(group)['direction'].cumcount() + 1
        
        # 1 if green, 0 if red
        is_green = (bricks['direction'] == 1).astype(int)
        green_ratio = is_green.rolling(self.lookback).mean()
        
        is_trending_up = green_ratio >= self.trend_threshold
        is_trending_down = green_ratio <= (1.0 - self.trend_threshold)
        is_ranging = (~is_trending_up) & (~is_trending_down)
        
        signal = pd.Series(0, index=bricks.index)
        
        # Trend Continuations
        signal[is_trending_up & (bricks['direction'] == 1) & (consec_count >= 3)] = 1
        signal[is_trending_down & (bricks['direction'] == -1) & (consec_count >= 3)] = -1
        
        # Range Reversals
        signal[is_ranging & (bricks['direction'] == 1) & (consec_count == 2)] = 1
        signal[is_ranging & (bricks['direction'] == -1) & (consec_count == 2)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)

class RenkoATRConsecutiveThesis(BaseRenkoThesis):
    """
    ATR-Based Renko No-Consecutive Strategy
    Buy: 2 consecutive bullish bricks AND fast_ma > slow_ma
    Sell: 2 consecutive bearish bricks AND fast_ma < slow_ma
    """
    def __init__(self, fast_period=20, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, bricks: pd.DataFrame) -> pd.DataFrame:
        dir_changed = bricks['direction'] != bricks['direction'].shift(1)
        group = dir_changed.cumsum()
        consec_count = bricks.groupby(group)['direction'].cumcount() + 1
        
        # Calculate Moving Averages on brick_close
        fast_ma = bricks['brick_close'].rolling(self.fast_period).mean()
        slow_ma = bricks['brick_close'].rolling(self.slow_period).mean()
        
        signal = pd.Series(0, index=bricks.index)
        
        # 2-brick validation + MA filter
        is_buy_setup = (bricks['direction'] == 1) & (consec_count >= 2) & (fast_ma > slow_ma)
        is_sell_setup = (bricks['direction'] == -1) & (consec_count >= 2) & (fast_ma < slow_ma)
        
        signal[is_buy_setup] = 1
        signal[is_sell_setup] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_renko_signals(bricks, signal)
