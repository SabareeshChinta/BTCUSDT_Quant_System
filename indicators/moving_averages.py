"""
Moving Average utilities — SMA, EMA, trend-state detection.

All functions are pure (no side effects) and return Series aligned to the
input index.

Usage:
    sma = compute_sma(df['Close'], period=20)
    ema = compute_ema(df['Close'], period=50)
    ma  = compute_ma(df['Close'], period=20, ma_type='EMA')

    trend = get_trend_state(fast_ma=sma_20, slow_ma=sma_50)
    # trend: 1 = bullish, -1 = bearish, 0 = equal/NaN
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ======================================================================
# Core MA functions
# ======================================================================

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Parameters
    ----------
    series : pd.Series
        Input price series.
    period : int
        Look-back window (must be >= 1).

    Returns
    -------
    pd.Series
        SMA values.  The first ``period - 1`` values will be NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = series.rolling(window=period, min_periods=period).mean()
    result.name = f"SMA_{period}"
    return result


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average.

    Uses *span* = ``period`` (standard interpretation) with ``adjust=False``
    so the first valid value is seeded from the raw observation and the EMA
    recursion runs forward from there.

    Parameters
    ----------
    series : pd.Series
        Input price series.
    period : int
        EMA span (must be >= 1).

    Returns
    -------
    pd.Series
        EMA values.  The first ``period - 1`` values will be NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    result = series.ewm(span=period, adjust=False, min_periods=period).mean()
    result.name = f"EMA_{period}"
    return result


# ======================================================================
# Dispatcher
# ======================================================================

_MA_DISPATCH = {
    "SMA": compute_sma,
    "EMA": compute_ema,
}


def compute_ma(
    series: pd.Series,
    period: int,
    ma_type: str = "SMA",
) -> pd.Series:
    """Compute a moving average of the requested type.

    Parameters
    ----------
    series : pd.Series
        Input price series.
    period : int
        Look-back / span.
    ma_type : str
        ``'SMA'`` or ``'EMA'`` (case-insensitive).

    Returns
    -------
    pd.Series
        Moving-average values.
    """
    key = ma_type.upper()
    func = _MA_DISPATCH.get(key)
    if func is None:
        raise ValueError(
            f"Unsupported ma_type '{ma_type}'. Choose from {set(_MA_DISPATCH)}"
        )
    return func(series, period)


# ======================================================================
# Trend state
# ======================================================================

def get_trend_state(
    fast_ma: pd.Series,
    slow_ma: pd.Series,
) -> pd.Series:
    """Determine trend direction from two moving averages.

    Parameters
    ----------
    fast_ma : pd.Series
        Faster (shorter-period) moving average.
    slow_ma : pd.Series
        Slower (longer-period) moving average.

    Returns
    -------
    pd.Series
        Integer series:
        -  ``1``  — bullish (fast > slow)
        - ``-1``  — bearish (fast < slow)
        -  ``0``  — equal or either value is NaN
    """
    diff = fast_ma - slow_ma
    # Where either MA is NaN the diff is NaN → np.sign returns NaN → fillna(0)
    state = np.sign(diff).fillna(0).astype(int)
    result = pd.Series(state, index=fast_ma.index, name="trend_state")
    return result
