"""
Data validation, cleaning, and basic feature engineering.

Every function operates on a *copy* of the input DataFrame to avoid
mutating caller data.  No look-ahead bias is introduced — all
computations are strictly causal.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Run a battery of sanity checks on an OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume``.

    Returns
    -------
    tuple[bool, list[str]]
        ``(is_valid, issues)`` where *is_valid* is ``True`` when no
        issues are found and *issues* is a (possibly empty) list of
        human-readable problem descriptions.
    """
    issues: List[str] = []
    required = ["Open", "High", "Low", "Close", "Volume"]

    # --- presence check ---------------------------------------------------
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
        return False, issues

    if df.empty:
        issues.append("DataFrame is empty.")
        return False, issues

    # --- null check -------------------------------------------------------
    for col in required:
        n_null = df[col].isna().sum()
        if n_null > 0:
            issues.append(f"{col} has {n_null:,} null values.")

    # --- negative prices / volume -----------------------------------------
    for col in ["Open", "High", "Low", "Close"]:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            issues.append(f"{col} has {n_neg:,} negative values.")

    n_neg_vol = (df["Volume"] < 0).sum()
    if n_neg_vol > 0:
        issues.append(f"Volume has {n_neg_vol:,} negative values.")

    # --- OHLC relationship checks -----------------------------------------
    high_lt_low = (df["High"] < df["Low"]).sum()
    if high_lt_low > 0:
        issues.append(f"High < Low in {high_lt_low:,} rows.")

    high_lt_open = (df["High"] < df["Open"]).sum()
    if high_lt_open > 0:
        issues.append(f"High < Open in {high_lt_open:,} rows.")

    high_lt_close = (df["High"] < df["Close"]).sum()
    if high_lt_close > 0:
        issues.append(f"High < Close in {high_lt_close:,} rows.")

    low_gt_open = (df["Low"] > df["Open"]).sum()
    if low_gt_open > 0:
        issues.append(f"Low > Open in {low_gt_open:,} rows.")

    low_gt_close = (df["Low"] > df["Close"]).sum()
    if low_gt_close > 0:
        issues.append(f"Low > Close in {low_gt_close:,} rows.")

    # --- monotonicity of index --------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        if not df.index.is_monotonic_increasing:
            issues.append("Index is not monotonically increasing.")

    is_valid = len(issues) == 0
    return is_valid, issues


# ---------------------------------------------------------------------------
# Time-gap analysis
# ---------------------------------------------------------------------------

def check_time_gaps(
    df: pd.DataFrame,
    expected_interval_minutes: int,
) -> pd.DataFrame:
    """Identify gaps in a time-indexed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex``.
    expected_interval_minutes : int
        The expected spacing between consecutive rows in minutes.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``gap_start``, ``gap_end``, and
        ``gap_minutes`` for every detected gap larger than the expected
        interval.  Empty if no gaps are found.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    if len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes"])

    expected_td = pd.Timedelta(minutes=expected_interval_minutes)
    deltas = df.index.to_series().diff()

    # A gap is any interval *strictly* larger than the expected one
    mask = deltas > expected_td
    gap_indices = df.index[mask]

    if gap_indices.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "gap_minutes"])

    # Position of each gap in the original index
    positions = np.where(mask)[0]
    gap_starts = df.index[positions - 1]
    gap_ends = df.index[positions]
    gap_minutes = (gap_ends - gap_starts).total_seconds() / 60.0

    return pd.DataFrame(
        {
            "gap_start": gap_starts,
            "gap_end": gap_ends,
            "gap_minutes": gap_minutes,
        }
    )


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    max_ffill: int = 3,
) -> pd.DataFrame:
    """Clean an OHLCV DataFrame.

    Steps
    -----
    1. Sort by index.
    2. Drop exact-duplicate index entries.
    3. Forward-fill small gaps (up to *max_ffill* consecutive NaNs).
    4. Drop any rows still containing NaN in critical OHLCV columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data.
    max_ffill : int
        Maximum number of consecutive NaN rows to forward-fill.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df*.
    """
    out = df.copy()

    # Ensure chronological order
    out = out.sort_index()

    # Remove duplicated timestamps
    out = out[~out.index.duplicated(keep="first")]

    # Forward-fill small gaps (limit prevents filling over large outages)
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    present = [c for c in ohlcv_cols if c in out.columns]
    out[present] = out[present].ffill(limit=max_ffill)

    # Drop rows where critical columns are still NaN
    out = out.dropna(subset=present)

    return out


# ---------------------------------------------------------------------------
# Returns computation (strictly causal — no look-ahead)
# ---------------------------------------------------------------------------

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append log-return and percentage-return columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new columns: ``log_return`` and
        ``pct_return``.
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain a 'Close' column.")

    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["pct_return"] = out["Close"].pct_change()
    return out
