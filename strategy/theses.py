import pandas as pd
import numpy as np

class BaseThesis:
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

def _format_signals(df: pd.DataFrame, signal_series: pd.Series) -> pd.DataFrame:
    """Helper to format the signals into the structure expected by BacktestEngine"""
    signals_df = pd.DataFrame(index=df.index)
    
    time_source = df['Open Time'] if 'Open Time' in df.columns else pd.Series(df.index, index=df.index)
    
    if hasattr(time_source.dt, 'tz') and time_source.dt.tz is not None:
        signals_df['brick_close_time'] = time_source.dt.tz_localize(None)
    else:
        signals_df['brick_close_time'] = time_source
    
    # Map 1 to BUY, -1 to SELL, 0 to NEUTRAL
    signal_strs = np.where(signal_series == 1, 'BUY', np.where(signal_series == -1, 'SELL', 'NEUTRAL'))
    signals_df['raw_signal'] = signal_strs
    
    # We only want to trigger when the signal changes from neutral/opposite to a new active direction
    # Wait, for continuous systems, we might just want to fire whenever there's a 1 or -1.
    # To mimic Renko, we fire when signal != shifted signal
    shifted = signal_series.shift(1).fillna(0)
    signals_df['signal_changed'] = (signal_series != 0) & (signal_series != shifted)
    
    return signals_df


class DonchianBreakout(BaseThesis):
    """
    1. Trend Following: Donchian Breakout
    Buy when price closes above 20-period highest high.
    Sell when price closes below 20-period lowest low.
    """
    def __init__(self, period=20):
        self.period = period
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        highs = df['High'].rolling(window=self.period).max().shift(1)
        lows = df['Low'].rolling(window=self.period).min().shift(1)
        
        signal = pd.Series(0, index=df.index)
        signal[df['Close'] > highs] = 1
        signal[df['Close'] < lows] = -1
        
        # Forward fill the signal so we maintain the state
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        
        return _format_signals(df, signal)

class ATRBurst(BaseThesis):
    """
    2. Volatility Expansion: ATR Burst
    Buy when short-term ATR > 1.5x long-term ATR and close > 20 EMA.
    Sell when short-term ATR > 1.5x long-term ATR and close < 20 EMA.
    """
    def __init__(self, short_period=5, long_period=20, threshold=1.5):
        self.short_period = short_period
        self.long_period = long_period
        self.threshold = threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        def calc_atr(period):
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift(1)).abs()
            tr3 = (df['Low'] - df['Close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()
            
        short_atr = calc_atr(self.short_period)
        long_atr = calc_atr(self.long_period)
        ema = df['Close'].ewm(span=20, adjust=False).mean()
        
        burst = short_atr > (long_atr * self.threshold)
        
        signal = pd.Series(0, index=df.index)
        signal[burst & (df['Close'] > ema)] = 1
        signal[burst & (df['Close'] < ema)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_signals(df, signal)

class MultiTimeframeMomentum(BaseThesis):
    """
    3. Momentum: Multi-Timeframe Alignment
    Buy when 5-period, 15-period, and 50-period ROC are all positive.
    Sell when all are negative.
    """
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        roc5 = df['Close'].pct_change(5)
        roc15 = df['Close'].pct_change(15)
        roc50 = df['Close'].pct_change(50)
        
        signal = pd.Series(0, index=df.index)
        signal[(roc5 > 0) & (roc15 > 0) & (roc50 > 0)] = 1
        signal[(roc5 < 0) & (roc15 < 0) & (roc50 < 0)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_signals(df, signal)

class BollingerMeanReversion(BaseThesis):
    """
    4. Mean Reversion: Bollinger Band Extremes
    Buy when Close < Lower Band (20, 2.5).
    Sell when Close > Upper Band (20, 2.5).
    """
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper = sma + (2.5 * std)
        lower = sma - (2.5 * std)
        
        signal = pd.Series(0, index=df.index)
        # We fire precisely on the close crossing the extreme.
        # But we hold the position until the opposite signal or risk takes us out.
        signal[df['Close'] < lower] = 1
        signal[df['Close'] > upper] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_signals(df, signal)

class ADXRegime(BaseThesis):
    """
    5. Regime Detection: ADX Trend vs. Range Filter
    Trending (ADX > 25): Buy if fast MA > slow MA. Sell if fast MA < slow MA.
    Ranging (ADX < 20): Mean reversion on RSI extremes (Buy < 30, Sell > 70).
    Neutral (20-25): No new positions.
    """
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate +DI, -DI, ADX (14 period)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up = df['High'] - df['High'].shift(1)
        down = df['Low'].shift(1) - df['Low']
        
        pos_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        neg_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        
        tr_smooth = tr.rolling(14).sum()
        pos_dm_smooth = pos_dm.rolling(14).sum()
        neg_dm_smooth = neg_dm.rolling(14).sum()
        
        plus_di = 100 * (pos_dm_smooth / tr_smooth)
        minus_di = 100 * (neg_dm_smooth / tr_smooth)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(14).mean()
        
        # MAs for trend
        fast_ma = df['Close'].rolling(20).mean()
        slow_ma = df['Close'].rolling(50).mean()
        
        # RSI for range
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signal = pd.Series(0, index=df.index)
        
        # Trending Regime Signals
        trend_mask = adx > 25
        signal[trend_mask & (fast_ma > slow_ma)] = 1
        signal[trend_mask & (fast_ma < slow_ma)] = -1
        
        # Ranging Regime Signals
        range_mask = adx < 20
        signal[range_mask & (rsi < 30)] = 1
        signal[range_mask & (rsi > 70)] = -1
        
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_signals(df, signal)

class SupertrendRSI(BaseThesis):
    """
    Trend & Momentum Hybrid: Supertrend (10, 2.5) with RSI (14) Boundary Filter.
    - Long: Supertrend flips to Bullish while RSI <= 75.0 (not overbought).
    - Short: Supertrend flips to Bearish while RSI >= 25.0 (not oversold).
    """
    def __init__(
        self,
        st_period: int = 10,
        st_mult: float = 2.5,
        rsi_period: int = 14,
        rsi_max_long: float = 75.0,
        rsi_min_short: float = 25.0
    ):
        self.st_period = st_period
        self.st_mult = st_mult
        self.rsi_period = rsi_period
        self.rsi_max_long = rsi_max_long
        self.rsi_min_short = rsi_min_short

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Compute ATR
        high = df['High']
        low = df['Low']
        close_prev = df['Close'].shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=self.st_period, adjust=False).mean()
        
        # 2. Compute Supertrend
        hl2 = (high + low) / 2.0
        upper_band = hl2 + (self.st_mult * atr)
        lower_band = hl2 - (self.st_mult * atr)
        
        close = df['Close'].values
        ub = upper_band.values
        lb = lower_band.values
        n = len(df)
        
        st_dir = np.ones(n)
        for i in range(1, n):
            if close[i] > ub[i-1]:
                st_dir[i] = 1
            elif close[i] < lb[i-1]:
                st_dir[i] = -1
            else:
                st_dir[i] = st_dir[i-1]
                if st_dir[i] == 1 and lb[i] < lb[i-1]:
                    lb[i] = lb[i-1]
                if st_dir[i] == -1 and ub[i] > ub[i-1]:
                    ub[i] = ub[i-1]
                    
        # 3. Compute RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).values
        
        # 4. Generate directional signals
        signal = pd.Series(0, index=df.index)
        for i in range(1, n):
            if st_dir[i-1] == -1 and st_dir[i] == 1 and rsi[i] <= self.rsi_max_long:
                signal.iloc[i] = 1
            elif st_dir[i-1] == 1 and st_dir[i] == -1 and rsi[i] >= self.rsi_min_short:
                signal.iloc[i] = -1
                
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return _format_signals(df, signal)

