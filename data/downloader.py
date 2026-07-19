"""
Multi-timeframe Binance data downloader.

Downloads historical kline (candlestick) data from Binance's public API
and persists it as Parquet files for downstream processing.

No API key is required — only publicly available market data is used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from binance.client import Client

from config.timeframes import get_binance_interval

# ---------------------------------------------------------------------------
# Column schema for Binance klines
# ---------------------------------------------------------------------------

KLINE_COLUMNS: List[str] = [
    "Open Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close Time",
    "Quote Asset Volume",
    "Number of Trades",
    "Taker Buy Base",
    "Taker Buy Quote",
    "Ignore",
]

# Columns that should be cast to float64
_NUMERIC_COLUMNS: List[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Quote Asset Volume",
    "Taker Buy Base",
    "Taker Buy Quote",
    "Ignore",
]


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------

def download_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download kline data from Binance for a single timeframe.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``'BTCUSDT'``.
    interval : str
        Human-readable timeframe string, e.g. ``'1h'``, ``'4h'``.
        Internally mapped to a ``binance.client.Client`` constant.
    start_date : str
        Start date in ``'YYYY-MM-DD'`` or any ``pd.Timestamp``-parseable
        format.
    end_date : str
        End date (inclusive) in the same format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns defined by :data:`KLINE_COLUMNS`, an
        ``Open Time`` datetime index, and all numeric columns properly
        typed.
    """
    client = Client()  # no API key needed for public data

    binance_const = get_binance_interval(interval)
    binance_interval = getattr(Client, binance_const)

    print(f"  ⏳ Downloading {symbol} {interval} from {start_date} to {end_date} …")

    raw_klines = client.get_historical_klines(
        symbol=symbol,
        interval=binance_interval,
        start_str=start_date,
        end_str=end_date,
    )

    if not raw_klines:
        print(f"  ⚠  No data returned for {symbol} {interval}.")
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(raw_klines, columns=KLINE_COLUMNS)

    # --- type conversions ------------------------------------------------
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Number of Trades"] = pd.to_numeric(
        df["Number of Trades"], errors="coerce"
    ).astype("Int64")

    # Timestamps are returned as millisecond epoch integers
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms", utc=True)
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit="ms", utc=True)

    # Use Open Time as the index (standard for OHLCV data)
    df = df.set_index("Open Time").sort_index()

    # Drop exact-duplicate timestamps (Binance occasionally returns them)
    df = df[~df.index.duplicated(keep="first")]

    n_rows = len(df)
    print(f"  ✅ Downloaded {n_rows:,} rows for {symbol} {interval}.")
    return df


# ---------------------------------------------------------------------------
# Batch download across timeframes
# ---------------------------------------------------------------------------

def download_all_timeframes(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframes: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Download kline data for multiple timeframes and save to Parquet.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``'BTCUSDT'``.
    start_date, end_date : str
        Date range boundaries.
    timeframes : list[str] | None
        Timeframes to download.  Defaults to
        :data:`config.settings.TIMEFRAMES`.
    output_dir : str | None
        Root directory for saved Parquet files.  Defaults to
        ``data/raw/`` under the current working directory.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of timeframe label → downloaded DataFrame.
    """
    if timeframes is None:
        from config.settings import TIMEFRAMES
        timeframes = TIMEFRAMES

    if output_dir is None:
        output_dir = os.path.join("data", "raw")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    print(f"{'=' * 60}")
    print(f"Downloading {symbol} — {len(timeframes)} timeframes")
    print(f"Range: {start_date} → {end_date}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 60}")

    for i, tf in enumerate(timeframes, 1):
        print(f"\n[{i}/{len(timeframes)}] {tf}")
        df = download_klines(symbol, tf, start_date, end_date)

        if not df.empty:
            filename = f"{symbol}_{tf}.parquet"
            filepath = os.path.join(output_dir, filename)
            df.to_parquet(filepath, engine="pyarrow")
            print(f"  💾 Saved → {filepath}")

        results[tf] = df

    print(f"\n{'=' * 60}")
    print("Download complete.")
    print(f"{'=' * 60}")

    return results
