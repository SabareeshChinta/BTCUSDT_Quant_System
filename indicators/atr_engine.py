"""
ATR Engine — Configurable Average True Range computation.

Supports three smoothing methods:
  - 'sma'    : Simple Moving Average of True Range
  - 'ema'    : Exponential Moving Average of True Range
  - 'wilder' : Wilder's smoothing (alpha = 1/period), the classical ATR method

Usage:
    engine = ATREngine(period=14, smoothing='wilder')
    atr_series = engine.compute(df)

    # Or use the convenience function:
    atr_series = compute_atr(df, period=14, smoothing='sma')
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ATREngine:
    """Configurable Average True Range calculator.

    Parameters
    ----------
    period : int
        Look-back window for the ATR smoothing (default 14).
    smoothing : str
        Smoothing method — one of 'sma', 'ema', 'wilder' (default 'sma').
    """

    _VALID_SMOOTHING = {"sma", "ema", "wilder"}

    def __init__(self, period: int = 14, smoothing: str = "sma") -> None:
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        smoothing = smoothing.lower()
        if smoothing not in self._VALID_SMOOTHING:
            raise ValueError(
                f"smoothing must be one of {self._VALID_SMOOTHING}, got '{smoothing}'"
            )
        self.period: int = period
        self.smoothing: str = smoothing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR from an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns 'High', 'Low', 'Close'.

        Returns
        -------
        pd.Series
            ATR values aligned to the input index.  The first ``period``
            values will be NaN (insufficient history).
        """
        self._validate_columns(df)

        tr = self._true_range(df)
        atr = self._smooth(tr)
        atr.name = "ATR"
        return atr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        """Ensure required OHLC columns exist."""
        required = {"High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"DataFrame is missing required columns: {missing}")

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        """Compute True Range (vectorised, no mutation of *df*).

        TR = max(
            High - Low,
            |High - prev_Close|,
            |Low  - prev_Close|,
        )
        """
        high: pd.Series = df["High"]
        low: pd.Series = df["Low"]
        prev_close: pd.Series = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr.name = "TR"
        return tr

    def _smooth(self, tr: pd.Series) -> pd.Series:
        """Apply the configured smoothing to the True Range series."""
        if self.smoothing == "sma":
            return tr.rolling(window=self.period, min_periods=self.period).mean()

        if self.smoothing == "ema":
            return tr.ewm(span=self.period, adjust=False, min_periods=self.period).mean()

        # Wilder's smoothing:  ATR_t = ATR_{t-1} * (n-1)/n + TR_t / n
        # Equivalent to EMA with alpha = 1/n.
        return tr.ewm(alpha=1.0 / self.period, adjust=False, min_periods=self.period).mean()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ATREngine(period={self.period}, smoothing='{self.smoothing}')"


# ======================================================================
# Convenience function
# ======================================================================

def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    smoothing: str = "sma",
) -> pd.Series:
    """Compute ATR in a single call (thin wrapper around :class:`ATREngine`).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with 'High', 'Low', 'Close' columns.
    period : int
        ATR look-back period (default 14).
    smoothing : str
        Smoothing method — 'sma', 'ema', or 'wilder' (default 'sma').

    Returns
    -------
    pd.Series
        ATR values aligned to *df*'s index.
    """
    return ATREngine(period=period, smoothing=smoothing).compute(df)
