"""
Trade Validator — Multi-layer signal validation pipeline.

Validates raw signals against configurable filters (moving-average trend,
ML classifier) before they are forwarded to position management.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Optional ML predictor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class MLPredictor(Protocol):
    """Minimal interface any ML model must satisfy to be used as a filter."""

    def predict_proba(self, features: dict) -> float:
        """Return the probability [0, 1] that the signal is profitable."""
        ...


class TradeValidator:
    """Multi-layer validation gate for raw trading signals.

    Parameters
    ----------
    use_ma_filter : bool
        Enable / disable the moving-average trend filter.
    use_ml_filter : bool
        Enable / disable the machine-learning confidence filter.
    fast_ma_period : int
        Look-back window for the fast moving average.
    slow_ma_period : int
        Look-back window for the slow moving average.
    ma_type : str
        Type of moving average — ``'SMA'`` or ``'EMA'``.
    ml_predictor : MLPredictor | None
        An object implementing ``predict_proba(features) -> float``.
    ml_confidence_threshold : float
        Minimum probability required from the ML model to accept a signal.
    """

    VALID_MA_TYPES = {"SMA", "EMA"}

    def __init__(
        self,
        use_ma_filter: bool = True,
        use_ml_filter: bool = False,
        fast_ma_period: int = 20,
        slow_ma_period: int = 50,
        ma_type: str = "SMA",
        ml_predictor: Any = None,
        ml_confidence_threshold: float = 0.6,
    ) -> None:
        if fast_ma_period >= slow_ma_period:
            raise ValueError(
                f"fast_ma_period ({fast_ma_period}) must be < "
                f"slow_ma_period ({slow_ma_period})"
            )
        ma_type_upper = ma_type.upper()
        if ma_type_upper not in self.VALID_MA_TYPES:
            raise ValueError(
                f"ma_type must be one of {self.VALID_MA_TYPES}, got '{ma_type}'"
            )

        self.use_ma_filter = use_ma_filter
        self.use_ml_filter = use_ml_filter
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.ma_type = ma_type_upper
        self.ml_predictor = ml_predictor
        self.ml_confidence_threshold = ml_confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: str,
        signal_time: pd.Timestamp,
        price_data: pd.DataFrame,
        features: dict | None = None,
    ) -> tuple[bool, dict]:
        """Run a raw signal through every enabled validation layer.

        Parameters
        ----------
        signal : str
            ``'BUY'`` or ``'SELL'``.
        signal_time : pd.Timestamp
            Timestamp associated with the signal (used to look up the
            relevant moving-average values).
        price_data : pd.DataFrame
            OHLCV bar data with a ``'Close'`` column.  Must be indexed by
            timestamp **or** contain an ``'open_time'`` / ``'timestamp'``
            column that can be matched against *signal_time*.
        features : dict | None
            Feature dictionary forwarded to the ML predictor (if enabled).

        Returns
        -------
        tuple[bool, dict]
            ``(is_valid, validation_details)`` where *validation_details*
            contains per-layer results.

        Notes
        -----
        All moving averages are computed from data available **up to and
        including** *signal_time* — no look-ahead bias.
        """
        if signal not in ("BUY", "SELL"):
            raise ValueError(f"signal must be 'BUY' or 'SELL', got '{signal}'")

        details: dict[str, Any] = {
            "ma_passed": True,
            "ml_passed": True,
            "ml_probability": None,
            "all_passed": True,
        }

        # --- Layer 1: Moving-Average Trend Filter --------------------------
        if self.use_ma_filter:
            ma_passed = self._check_ma_filter(signal, signal_time, price_data)
            details["ma_passed"] = ma_passed
        else:
            ma_passed = True

        # --- Layer 2: ML Confidence Filter ---------------------------------
        ml_passed = True
        if self.use_ml_filter and self.ml_predictor is not None:
            ml_prob = self._check_ml_filter(features or {})
            details["ml_probability"] = ml_prob
            ml_passed = ml_prob >= self.ml_confidence_threshold
            details["ml_passed"] = ml_passed

        all_passed = ma_passed and ml_passed
        details["all_passed"] = all_passed

        return all_passed, details

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_ma_filter(
        self,
        signal: str,
        signal_time: pd.Timestamp,
        price_data: pd.DataFrame,
    ) -> bool:
        """Return True if the MA trend agrees with the signal direction.

        Only data up to *signal_time* is used (no look-ahead).
        """
        closes = self._get_closes_up_to(price_data, signal_time)

        # Need at least slow_ma_period data points to compute the slow MA.
        if len(closes) < self.slow_ma_period:
            # Insufficient data — conservatively reject.
            return False

        fast_ma = self._compute_ma(closes, self.fast_ma_period)
        slow_ma = self._compute_ma(closes, self.slow_ma_period)

        if np.isnan(fast_ma) or np.isnan(slow_ma):
            return False

        if signal == "BUY":
            return fast_ma > slow_ma
        else:  # SELL
            return fast_ma < slow_ma

    def _check_ml_filter(self, features: dict) -> float:
        """Query the ML predictor and return a probability in [0, 1]."""
        try:
            prob = float(self.ml_predictor.predict_proba(features))
            return max(0.0, min(1.0, prob))  # clamp
        except Exception:
            # If the predictor fails, conservatively return 0.
            return 0.0

    # ------------------------------------------------------------------
    # MA computation utilities
    # ------------------------------------------------------------------

    def _compute_ma(self, closes: pd.Series, period: int) -> float:
        """Return the latest moving-average value."""
        if len(closes) < period:
            return np.nan
        if self.ma_type == "EMA":
            return closes.ewm(span=period, adjust=False).mean().iloc[-1]
        # Default: SMA
        return closes.rolling(window=period).mean().iloc[-1]

    @staticmethod
    def _get_closes_up_to(
        price_data: pd.DataFrame,
        signal_time: pd.Timestamp,
    ) -> pd.Series:
        """Extract Close prices up to and including *signal_time*.

        Handles both DatetimeIndex and column-based timestamps.
        """
        if isinstance(price_data.index, pd.DatetimeIndex):
            mask = price_data.index <= signal_time
            return price_data.loc[mask, "Close"]

        # Try common timestamp column names.
        for col in ("open_time", "timestamp", "date", "time"):
            if col in price_data.columns:
                mask = price_data[col] <= signal_time
                return price_data.loc[mask, "Close"]

        # Fallback — use the entire dataset (caller is responsible for
        # ensuring no future data leaks).
        return price_data["Close"]
