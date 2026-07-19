"""
Efficient Parquet I/O and dataset splitting utilities.

Provides helpers for loading, saving, filtering, and splitting OHLCV
Parquet files into backtest / forward-test / paper-trade segments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_data(
    filepath: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load a Parquet file with optional date filtering.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to a ``.parquet`` file.
    start_date : str | None
        Inclusive lower bound (ISO-8601 or any ``pd.Timestamp``-parseable
        string).  ``None`` means no lower bound.
    end_date : str | None
        Inclusive upper bound.  ``None`` means no upper bound.

    Returns
    -------
    pd.DataFrame
        The (optionally filtered) DataFrame with a ``DatetimeIndex``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_parquet(filepath, engine="pyarrow")

    # Ensure the index is datetime-typed for slicing
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert the index
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()

    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df.loc[df.index >= start_ts]

    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df.loc[df.index <= end_ts]

    return df


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to Parquet, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    filepath : str
        Destination path (should end in ``.parquet``).
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, engine="pyarrow")


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    backtest_end: str,
    forward_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into three non-overlapping segments.

    The splits are:
    * **backtest** — everything up to and including *backtest_end*
    * **forward**  — everything after *backtest_end* up to and including
      *forward_end*
    * **paper**    — everything after *forward_end*

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DatetimeIndex``.
    backtest_end : str
        End of the backtest window (inclusive).
    forward_end : str
        End of the forward-test window (inclusive).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(backtest_df, forward_df, paper_df)``
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")

    df = df.sort_index()

    bt_end_ts = pd.Timestamp(backtest_end, tz="UTC")
    fw_end_ts = pd.Timestamp(forward_end, tz="UTC")

    backtest_df = df.loc[df.index <= bt_end_ts]
    forward_df = df.loc[(df.index > bt_end_ts) & (df.index <= fw_end_ts)]
    paper_df = df.loc[df.index > fw_end_ts]

    return backtest_df, forward_df, paper_df


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------

def get_data_path(
    symbol: str,
    timeframe: str,
    data_dir: str = os.path.join("data", "raw"),
) -> str:
    """Build a canonical Parquet file path for the given symbol/timeframe.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``'BTCUSDT'``.
    timeframe : str
        Candle interval, e.g. ``'1h'``.
    data_dir : str
        Directory that holds the Parquet files.

    Returns
    -------
    str
        Full file path, e.g. ``data/raw/BTCUSDT_1h.parquet``.
    """
    filename = f"{symbol}_{timeframe}.parquet"
    return os.path.join(data_dir, filename)
