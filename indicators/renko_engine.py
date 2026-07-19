"""
Renko Engine — ATR-based dynamic brick construction.

This is the core component of the BTCUSDT Quant Trading System.  It builds
Renko bricks whose size adapts to market volatility via the ATR indicator.

Key guarantees:
  - **No look-ahead bias**: ATR is sampled at or before the candle that
    completes a brick — never from the future.
  - **Only completed bricks**: The currently-forming brick is never included
    in the output.
  - **Multi-brick candles**: If a single candle's price movement spans
    multiple brick sizes, all intermediate bricks are emitted.

Usage:
    engine = RenkoEngine(atr_period=14, atr_multiplier=1.0, atr_smoothing='sma')
    bricks = engine.build_bricks(df)        # df is an OHLCV DataFrame
    bricks['consec'] = engine.get_consecutive_count(bricks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from indicators.atr_engine import ATREngine


# ======================================================================
# Data container for a single completed Renko brick
# ======================================================================

@dataclass
class _Brick:
    """Internal representation of a completed Renko brick."""

    brick_open_time: pd.Timestamp
    brick_close_time: pd.Timestamp
    brick_open: float
    brick_close: float
    brick_high: float
    brick_low: float
    brick_size: float
    direction: int           # +1 bullish, -1 bearish
    brick_number: int


# ======================================================================
# RenkoEngine
# ======================================================================

class RenkoEngine:
    """Build ATR-based dynamic Renko bricks from OHLCV data.

    Parameters
    ----------
    atr_period : int
        Period for the underlying ATR computation (default 14).
    atr_multiplier : float
        Multiplier applied to the ATR to determine brick size (default 1.0).
    atr_smoothing : str
        ATR smoothing method — 'sma', 'ema', or 'wilder' (default 'sma').
    """

    # Columns expected in the input DataFrame
    _REQUIRED_COLS = {"Open Time", "Open", "High", "Low", "Close"}

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,
        atr_smoothing: str = "sma",
    ) -> None:
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")

        self.atr_period: int = atr_period
        self.atr_multiplier: float = atr_multiplier
        self.atr_smoothing: str = atr_smoothing
        self._atr_engine = ATREngine(period=atr_period, smoothing=atr_smoothing)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_bricks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct Renko bricks from an OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``Open Time``, ``Open``, ``High``,
            ``Low``, ``Close``.  Rows must be sorted chronologically.

        Returns
        -------
        pd.DataFrame
            One row per *completed* brick with columns:
            ``brick_open_time``, ``brick_close_time``, ``brick_open``,
            ``brick_close``, ``brick_high``, ``brick_low``,
            ``brick_size``, ``direction``, ``brick_number``.
        """
        self._validate(df)

        # --- Compute ATR (vectorised, using only past data by construction) --
        atr_series: pd.Series = self._atr_engine.compute(df)

        # --- Prepare numpy arrays for the tight loop ---
        timestamps = df["Open Time"].values          # datetime64 array
        closes     = df["Close"].values.astype(float)
        highs      = df["High"].values.astype(float)
        lows       = df["Low"].values.astype(float)
        atr_vals   = atr_series.values.astype(float)
        n          = len(df)

        # --- Find the first row with a valid (non-NaN) ATR ---
        first_valid = self._first_valid_index(atr_vals)
        if first_valid is None or first_valid >= n - 1:
            # Not enough data to even start building bricks
            return self._empty_result()

        # --- Initialise brick-building state ---
        brick_open       = closes[first_valid]
        brick_start_time = timestamps[first_valid]
        current_atr      = atr_vals[first_valid]
        brick_size       = current_atr * self.atr_multiplier

        # Track extremes during the formation period of the *current* brick
        forming_high = highs[first_valid]
        forming_low  = lows[first_valid]

        bricks: List[_Brick] = []
        brick_counter = 0

        # --- Main loop: iterate from the candle AFTER initialisation ---
        for i in range(first_valid + 1, n):
            close_i = closes[i]
            high_i  = highs[i]
            low_i   = lows[i]

            # Update formation-period extremes
            forming_high = max(forming_high, high_i)
            forming_low  = min(forming_low, low_i)

            # Use ATR at THIS candle (not future). Fall back to last
            # known ATR if current value is NaN (should not happen after
            # first_valid, but defensive).
            atr_i = atr_vals[i]
            if not np.isnan(atr_i):
                current_atr = atr_i
                brick_size  = current_atr * self.atr_multiplier

            # Guard against degenerate ATR (would cause infinite loop)
            if brick_size <= 0:
                continue

            # Check for completed bricks (possibly multiple in one candle)
            while True:
                bullish_target = brick_open + brick_size
                bearish_target = brick_open - brick_size

                if close_i >= bullish_target:
                    # ---- BULLISH brick completed ----
                    brick_counter += 1
                    bricks.append(
                        _Brick(
                            brick_open_time=pd.Timestamp(brick_start_time),
                            brick_close_time=pd.Timestamp(timestamps[i]),
                            brick_open=brick_open,
                            brick_close=bullish_target,
                            brick_high=forming_high,
                            brick_low=forming_low,
                            brick_size=brick_size,
                            direction=1,
                            brick_number=brick_counter,
                        )
                    )
                    # Reset for next brick
                    brick_open       = bullish_target
                    brick_start_time = timestamps[i]
                    forming_high     = high_i
                    forming_low      = low_i

                    # Re-sample ATR at completion candle (already current_atr)
                    brick_size = current_atr * self.atr_multiplier

                elif close_i <= bearish_target:
                    # ---- BEARISH brick completed ----
                    brick_counter += 1
                    bricks.append(
                        _Brick(
                            brick_open_time=pd.Timestamp(brick_start_time),
                            brick_close_time=pd.Timestamp(timestamps[i]),
                            brick_open=brick_open,
                            brick_close=bearish_target,
                            brick_high=forming_high,
                            brick_low=forming_low,
                            brick_size=brick_size,
                            direction=-1,
                            brick_number=brick_counter,
                        )
                    )
                    # Reset for next brick
                    brick_open       = bearish_target
                    brick_start_time = timestamps[i]
                    forming_high     = high_i
                    forming_low      = low_i

                    brick_size = current_atr * self.atr_multiplier

                else:
                    # Price has not moved enough — stay in forming state
                    break

        return self._bricks_to_dataframe(bricks)

    # ------------------------------------------------------------------

    @staticmethod
    def get_consecutive_count(bricks_df: pd.DataFrame) -> pd.Series:
        """Count consecutive same-direction bricks (including current).

        Example
        -------
        directions = [1, -1, 1, 1, 1, -1, -1]
        result     = [1,  1, 1, 2, 3,  1,  2]

        Parameters
        ----------
        bricks_df : pd.DataFrame
            Must contain a ``direction`` column.

        Returns
        -------
        pd.Series
            Integer series of consecutive counts, aligned to the input index.
        """
        if bricks_df.empty or "direction" not in bricks_df.columns:
            return pd.Series(dtype=int, name="consecutive_count")

        directions = bricks_df["direction"].values
        counts = np.empty(len(directions), dtype=int)
        counts[0] = 1
        for i in range(1, len(directions)):
            if directions[i] == directions[i - 1]:
                counts[i] = counts[i - 1] + 1
            else:
                counts[i] = 1

        return pd.Series(
            counts, index=bricks_df.index, name="consecutive_count"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """Verify the DataFrame has the columns we need and is non-empty."""
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            raise KeyError(f"DataFrame is missing required columns: {missing}")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

    @staticmethod
    def _first_valid_index(arr: np.ndarray) -> int | None:
        """Return the index of the first non-NaN value, or None."""
        mask = ~np.isnan(arr)
        indices = np.where(mask)[0]
        return int(indices[0]) if len(indices) > 0 else None

    @staticmethod
    def _bricks_to_dataframe(bricks: List[_Brick]) -> pd.DataFrame:
        """Convert a list of ``_Brick`` dataclasses to a DataFrame."""
        if not bricks:
            return RenkoEngine._empty_result()

        return pd.DataFrame(
            [
                {
                    "brick_open_time": b.brick_open_time,
                    "brick_close_time": b.brick_close_time,
                    "brick_open": b.brick_open,
                    "brick_close": b.brick_close,
                    "brick_high": b.brick_high,
                    "brick_low": b.brick_low,
                    "brick_size": b.brick_size,
                    "direction": b.direction,
                    "brick_number": b.brick_number,
                }
                for b in bricks
            ]
        )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        return pd.DataFrame(
            columns=[
                "brick_open_time",
                "brick_close_time",
                "brick_open",
                "brick_close",
                "brick_high",
                "brick_low",
                "brick_size",
                "direction",
                "brick_number",
            ]
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"RenkoEngine(atr_period={self.atr_period}, "
            f"atr_multiplier={self.atr_multiplier}, "
            f"atr_smoothing='{self.atr_smoothing}')"
        )
