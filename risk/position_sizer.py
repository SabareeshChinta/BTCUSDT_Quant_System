"""
Position Sizer — Capital-based position sizing.

Determines the number of units to trade for a given amount of capital
and current price.
"""

from __future__ import annotations


class PositionSizer:
    """Simple position sizer.

    Parameters
    ----------
    method : str
        Sizing method.  Currently supports ``'fixed_fraction'`` and ``'risk_based'``.
    fraction : float
        Fraction of capital to deploy per trade (e.g. ``1.0`` = 100 %) for fixed_fraction, 
        or risk fraction per trade (e.g. ``0.01`` = 1%) for risk_based.
    """

    SUPPORTED_METHODS = {"fixed_fraction", "risk_based"}

    def __init__(
        self,
        method: str = "risk_based",
        fraction: float = 0.01,
    ) -> None:
        method_lower = method.lower()
        if method_lower not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Choose from {self.SUPPORTED_METHODS}."
            )
        if not 0 < fraction <= 1.0:
            raise ValueError(
                f"fraction must be in (0, 1.0], got {fraction}"
            )

        self.method = method_lower
        self.fraction = fraction

    def calculate_size(self, capital: float, price: float, stop_distance: float = None, risk_multiplier: float = 1.0) -> float:
        """Calculate the number of units to trade.

        Parameters
        ----------
        capital : float
            Available capital in quote currency.
        price : float
            Current asset price.
        stop_distance : float
            Distance from entry price to stop loss (required for risk_based).
        risk_multiplier : float
            Multiplier to adjust risk (e.g., from EventDetectionEngine).

        Returns
        -------
        float
            Number of units (may be fractional for crypto assets).

        Raises
        ------
        ValueError
            If *capital* or *price* is non-positive.
        """
        if capital <= 0:
            raise ValueError(f"capital must be > 0, got {capital}")
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")

        if self.method == "fixed_fraction":
            return ((capital * self.fraction) / price) * risk_multiplier
            
        if self.method == "risk_based":
            if not stop_distance or stop_distance <= 0:
                raise ValueError("stop_distance must be > 0 for risk_based sizing")
            dollar_risk = capital * self.fraction * risk_multiplier
            raw_size = dollar_risk / stop_distance
            max_size = capital / price
            return min(raw_size, max_size)

        # Shouldn't reach here, but guard anyway.
        raise NotImplementedError(f"Method '{self.method}' not implemented.")
