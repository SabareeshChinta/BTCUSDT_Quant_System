import numpy as np
import pandas as pd
from typing import Tuple

# ==============================================================================
# CAUSAL INDICATOR COMPUTATION (STRICTLY NO FUTURE LEAKAGE)
# ==============================================================================

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def compute_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = compute_ema(df['Close'], ema_period)
    atr = compute_atr(df, atr_period)
    upper = middle + (atr * atr_mult)
    lower = middle - (atr * atr_mult)
    return upper, middle, lower

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (tr_smooth + 1e-10))
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    atr = compute_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1 for bull, -1 for bear
    
    close = df['Close'].values
    ub = upper_band.values
    lb = lower_band.values
    n = len(df)
    
    st_arr = np.zeros(n)
    dir_arr = np.ones(n)
    
    for i in range(1, n):
        if close[i] > ub[i-1]:
            dir_arr[i] = 1
        elif close[i] < lb[i-1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i-1]
            if dir_arr[i] == 1 and lb[i] < lb[i-1]:
                lb[i] = lb[i-1]
            if dir_arr[i] == -1 and ub[i] > ub[i-1]:
                ub[i] = ub[i-1]
                
        st_arr[i] = lb[i] if dir_arr[i] == 1 else ub[i]
        
    return pd.Series(st_arr, index=df.index), pd.Series(dir_arr, index=df.index)

# ==============================================================================
# STRATEGY SIGNAL GENERATORS (PRODUCING 1 = BUY, -1 = SELL, 0 = NEUTRAL)
# ==============================================================================

def strat_ema_adx(df: pd.DataFrame, fast_p: int = 10, slow_p: int = 40, adx_p: int = 14, adx_thresh: float = 20.0) -> np.ndarray:
    """Trend: Dual EMA Cross confirmed by ADX trend strength."""
    fast_ema = compute_ema(df['Close'], fast_p).values
    slow_ema = compute_ema(df['Close'], slow_p).values
    adx = compute_adx(df, adx_p).values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        # Bullish cross with strong ADX
        if fast_ema[i-1] <= slow_ema[i-1] and fast_ema[i] > slow_ema[i] and adx[i] >= adx_thresh:
            signals[i] = 1
        # Bearish cross with strong ADX
        elif fast_ema[i-1] >= slow_ema[i-1] and fast_ema[i] < slow_ema[i] and adx[i] >= adx_thresh:
            signals[i] = -1
            
    return signals

def strat_supertrend_rsi(df: pd.DataFrame, st_period: int = 10, st_mult: float = 2.5, rsi_period: int = 14, rsi_max_long: float = 70.0, rsi_min_short: float = 30.0) -> np.ndarray:
    """Trend/Momentum: Supertrend Flip with RSI not overbought/oversold."""
    _, st_dir = compute_supertrend(df, st_period, st_mult)
    rsi = compute_rsi(df['Close'], rsi_period).values
    st_dir_arr = st_dir.values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        if st_dir_arr[i-1] == -1 and st_dir_arr[i] == 1 and rsi[i] <= rsi_max_long:
            signals[i] = 1
        elif st_dir_arr[i-1] == 1 and st_dir_arr[i] == -1 and rsi[i] >= rsi_min_short:
            signals[i] = -1
            
    return signals

def strat_macd_trend(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, trend_ema_p: int = 100) -> np.ndarray:
    """Momentum: MACD cross in the direction of macro EMA."""
    macd, sig, _ = compute_macd(df['Close'], fast, slow, signal)
    trend_ema = compute_ema(df['Close'], trend_ema_p).values
    macd_val = macd.values
    sig_val = sig.values
    close_val = df['Close'].values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        # MACD bullish cross above trend EMA
        if macd_val[i-1] <= sig_val[i-1] and macd_val[i] > sig_val[i] and close_val[i] > trend_ema[i]:
            signals[i] = 1
        # MACD bearish cross below trend EMA
        elif macd_val[i-1] >= sig_val[i-1] and macd_val[i] < sig_val[i] and close_val[i] < trend_ema[i]:
            signals[i] = -1
            
    return signals

def strat_bollinger_rsi_reversion(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14, rsi_ob: float = 70.0, rsi_os: float = 30.0) -> np.ndarray:
    """Mean Reversion: Piercing Bollinger Band + RSI Extreme."""
    upper, _, lower = compute_bollinger_bands(df['Close'], bb_period, bb_std)
    rsi = compute_rsi(df['Close'], rsi_period).values
    close = df['Close'].values
    low = df['Low'].values
    high = df['High'].values
    up_val = upper.values
    low_val = lower.values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        # Oversold bounce
        if low[i-1] <= low_val[i-1] and close[i] > low_val[i] and rsi[i-1] <= rsi_os:
            signals[i] = 1
        # Overbought rejection
        elif high[i-1] >= up_val[i-1] and close[i] < up_val[i] and rsi[i-1] >= rsi_ob:
            signals[i] = -1
            
    return signals

def strat_volatility_squeeze_breakout(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0, kc_period: int = 20, kc_mult: float = 1.5, mom_period: int = 12) -> np.ndarray:
    """Volatility: Squeeze (BB inside KC) breakout with momentum direction."""
    bb_upper, _, bb_lower = compute_bollinger_bands(df['Close'], bb_period, bb_std)
    kc_upper, _, kc_lower = compute_keltner_channels(df, kc_period, 10, kc_mult)
    mom = df['Close'].diff(mom_period).values
    
    bb_u = bb_upper.values
    bb_l = bb_lower.values
    kc_u = kc_upper.values
    kc_l = kc_lower.values
    
    # Squeeze is on when BB is inside KC
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        # Squeeze just fired (was on yesterday, now off)
        if squeeze_on[i-1] and not squeeze_on[i]:
            if mom[i] > 0:
                signals[i] = 1
            elif mom[i] < 0:
                signals[i] = -1
                
    return signals

def strat_htf_trend_pullback(df: pd.DataFrame, htf_ema_fast: int = 20, htf_ema_slow: int = 80, ltf_rsi_p: int = 14, rsi_buy_thresh: float = 40.0, rsi_sell_thresh: float = 60.0) -> np.ndarray:
    """Hybrid: Macro Trend Alignment + Micro Pullback Entry."""
    ema_fast = compute_ema(df['Close'], htf_ema_fast).values
    ema_slow = compute_ema(df['Close'], htf_ema_slow).values
    rsi = compute_rsi(df['Close'], ltf_rsi_p).values
    close = df['Close'].values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        # Macro Bullish Trend + Micro Pullback to Value
        if ema_fast[i] > ema_slow[i] and close[i] > ema_slow[i]:
            # RSI oversold turning up
            if rsi[i-1] < rsi_buy_thresh and rsi[i] >= rsi_buy_thresh:
                signals[i] = 1
        # Macro Bearish Trend + Micro Rally into Resistance
        elif ema_fast[i] < ema_slow[i] and close[i] < ema_slow[i]:
            # RSI overbought turning down
            if rsi[i-1] > rsi_sell_thresh and rsi[i] <= rsi_sell_thresh:
                signals[i] = -1
                
    return signals

def strat_donchian_breakout(df: pd.DataFrame, entry_period: int = 20, trend_period: int = 100) -> np.ndarray:
    """Trend Breakout: Price piercing 20-bar Donchian High/Low in direction of 100 EMA."""
    donchian_high = df['High'].rolling(entry_period).max().shift(1).values
    donchian_low = df['Low'].rolling(entry_period).min().shift(1).values
    trend_ema = compute_ema(df['Close'], trend_period).values
    close = df['Close'].values
    
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        if close[i] > donchian_high[i] and close[i] > trend_ema[i]:
            signals[i] = 1
        elif close[i] < donchian_low[i] and close[i] < trend_ema[i]:
            signals[i] = -1
            
    return signals
