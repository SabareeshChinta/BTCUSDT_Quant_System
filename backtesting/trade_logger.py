"""
Trade Logger — Structured trade record management.

Stores individual trade records with full attribution (PnL, MAE/MFE,
costs, exit reason) and provides DataFrame / CSV / Parquet persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


class TradeLogger:
    """Collects and persists structured trade records.

    Each trade record is a ``dict`` with a fixed schema (see
    :attr:`REQUIRED_FIELDS`).  Records are appended one at a time via
    :meth:`log_trade` and can be exported as a DataFrame or saved to disk.
    """

    REQUIRED_FIELDS: tuple[str, ...] = (
        "trade_id",
        "entry_time",
        "exit_time",
        "direction",
        "entry_price",
        "exit_price",
        "size",
        "gross_pnl",
        "net_pnl",
        "pnl_pct",
        "exit_reason",
        "holding_duration_hours",
        "entry_atr",
        "mae",
        "mfe",
        "commission_paid",
        "slippage_cost",
        "sig_time",
        "dollar_risk",
        "stop_distance",
        "r_multiple",
    )

    def __init__(self) -> None:
        self._trades: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade(self, trade_record: dict) -> None:
        """Append a single trade record.

        Parameters
        ----------
        trade_record : dict
            Must contain at least the keys listed in
            :attr:`REQUIRED_FIELDS`.  Extra keys are preserved but not
            enforced.

        Raises
        ------
        ValueError
            If any required field is missing.
        """
        missing = set(self.REQUIRED_FIELDS) - set(trade_record.keys())
        if missing:
            raise ValueError(
                f"Trade record is missing required fields: {missing}"
            )
        self._trades.append(trade_record.copy())

    def get_trades_df(self) -> pd.DataFrame:
        """Return all logged trades as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per trade.  If no trades have been logged yet an
            empty DataFrame with the correct columns is returned.
        """
        if not self._trades:
            return pd.DataFrame(columns=list(self.REQUIRED_FIELDS))
        return pd.DataFrame(self._trades)

    def save_trades(self, filepath: str) -> None:
        """Save all trades to both CSV and Parquet.

        Parameters
        ----------
        filepath : str
            Base path **without extension**.  Two files are created:
            ``<filepath>.csv`` and ``<filepath>.parquet``.
        """
        df = self.get_trades_df()
        base = Path(filepath)

        # Ensure parent directory exists
        base.parent.mkdir(parents=True, exist_ok=True)

        csv_path = base.with_suffix(".csv")
        parquet_path = base.with_suffix(".parquet")

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

    @staticmethod
    def load_trades(filepath: str) -> pd.DataFrame:
        """Load trades from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file (with or without ``.csv`` extension).

        Returns
        -------
        pd.DataFrame
            Loaded trade records.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(filepath)
        if path.suffix == "":
            path = path.with_suffix(".csv")
        if not path.exists():
            raise FileNotFoundError(f"Trade file not found: {path}")

        df = pd.read_csv(path)

        # Restore datetime types
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._trades)

    def __repr__(self) -> str:
        return f"TradeLogger(num_trades={len(self._trades)})"
