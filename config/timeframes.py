"""
Timeframe utility functions.

Provides helpers to convert human-readable timeframe strings (e.g. ``'1h'``,
``'4h'``) to minutes and to the constant names used by ``python-binance``.
"""

from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------------------------
# Timeframe → minutes mapping
# ---------------------------------------------------------------------------

_TF_TO_MINUTES: Dict[str, int] = {
    "1m":  1,
    "3m":  3,
    "5m":  5,
    "15m": 15,
    "30m": 30,
    "1h":  60,
    "2h":  120,
    "4h":  240,
    "6h":  360,
    "8h":  480,
    "12h": 720,
    "1d":  1440,
    "3d":  4320,
    "1w":  10080,
    "1M":  43200,  # ~30 days
}


def timeframe_to_minutes(tf: str) -> int:
    """Convert a timeframe string to its duration in minutes.

    Parameters
    ----------
    tf : str
        Timeframe label, e.g. ``'15m'``, ``'4h'``, ``'1d'``.

    Returns
    -------
    int
        Number of minutes in one period of the given timeframe.

    Raises
    ------
    ValueError
        If *tf* is not a recognised timeframe string.
    """
    try:
        return _TF_TO_MINUTES[tf]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe: {tf!r}. "
            f"Valid options: {list(_TF_TO_MINUTES.keys())}"
        ) from None


# ---------------------------------------------------------------------------
# Timeframe → Binance Client constant mapping
# ---------------------------------------------------------------------------

_TF_TO_BINANCE: Dict[str, str] = {
    "1m":  "KLINE_INTERVAL_1MINUTE",
    "3m":  "KLINE_INTERVAL_3MINUTE",
    "5m":  "KLINE_INTERVAL_5MINUTE",
    "15m": "KLINE_INTERVAL_15MINUTE",
    "30m": "KLINE_INTERVAL_30MINUTE",
    "1h":  "KLINE_INTERVAL_1HOUR",
    "2h":  "KLINE_INTERVAL_2HOUR",
    "4h":  "KLINE_INTERVAL_4HOUR",
    "6h":  "KLINE_INTERVAL_6HOUR",
    "8h":  "KLINE_INTERVAL_8HOUR",
    "12h": "KLINE_INTERVAL_12HOUR",
    "1d":  "KLINE_INTERVAL_1DAY",
    "3d":  "KLINE_INTERVAL_3DAY",
    "1w":  "KLINE_INTERVAL_1WEEK",
    "1M":  "KLINE_INTERVAL_1MONTH",
}


def get_binance_interval(tf: str) -> str:
    """Return the ``binance.client.Client`` constant name for *tf*.

    The returned string is the **attribute name** on ``Client``, e.g.
    ``'KLINE_INTERVAL_1HOUR'``.  At runtime you can resolve it with::

        from binance.client import Client
        interval = getattr(Client, get_binance_interval('1h'))

    Parameters
    ----------
    tf : str
        Timeframe label, e.g. ``'1h'``, ``'4h'``.

    Returns
    -------
    str
        Name of the corresponding ``Client`` class attribute.

    Raises
    ------
    ValueError
        If *tf* is not a recognised timeframe string.
    """
    try:
        return _TF_TO_BINANCE[tf]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe: {tf!r}. "
            f"Valid options: {list(_TF_TO_BINANCE.keys())}"
        ) from None
