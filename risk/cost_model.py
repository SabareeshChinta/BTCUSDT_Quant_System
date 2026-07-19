"""
Cost Model — Realistic transaction-cost modelling.

Accounts for slippage (entry and exit) and per-side commissions to produce
net PnL figures.
"""

from __future__ import annotations


class CostModel:
    """Models slippage and commission for realistic backtest accounting.

    Parameters
    ----------
    commission_pct : float
        Commission as a fraction of notional value per side (e.g. ``0.001``
        = 10 bps).
    slippage_pct : float
        Slippage as a fraction of price per side (e.g. ``0.0005`` = 5 bps).
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ) -> None:
        if commission_pct < 0:
            raise ValueError(
                f"commission_pct must be >= 0, got {commission_pct}"
            )
        if slippage_pct < 0:
            raise ValueError(
                f"slippage_pct must be >= 0, got {slippage_pct}"
            )
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------

    def apply_entry_slippage(self, price: float, direction: int) -> float:
        """Adjust the entry price for adverse slippage.

        * **LONG** entry: filled *higher* than the quoted price.
        * **SHORT** entry: filled *lower* than the quoted price.

        Parameters
        ----------
        price : float
            Quoted / theoretical entry price.
        direction : int
            ``1`` for long, ``-1`` for short.

        Returns
        -------
        float
            Adjusted entry price.
        """
        return price * (1 + direction * self.slippage_pct)

    def apply_exit_slippage(self, price: float, direction: int) -> float:
        """Adjust the exit price for adverse slippage.

        * **LONG** exit (selling): filled *lower* than the quoted price.
        * **SHORT** exit (buying back): filled *higher* than the quoted price.

        Parameters
        ----------
        price : float
            Quoted / theoretical exit price.
        direction : int
            ``1`` for long, ``-1`` for short.

        Returns
        -------
        float
            Adjusted exit price.
        """
        return price * (1 - direction * self.slippage_pct)

    # ------------------------------------------------------------------
    # Commission
    # ------------------------------------------------------------------

    def calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission for a single side (entry *or* exit).

        Parameters
        ----------
        price : float
            Fill price.
        size : float
            Number of units traded.

        Returns
        -------
        float
            Commission in quote currency.
        """
        return abs(price * size) * self.commission_pct

    # ------------------------------------------------------------------
    # Aggregated cost helpers
    # ------------------------------------------------------------------

    def calculate_total_costs(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
    ) -> float:
        """Calculate total round-trip costs (both-side commission + slippage).

        This is a simplified estimate that returns commission costs only
        (slippage is already embedded in the adjusted prices).

        Parameters
        ----------
        entry_price : float
            Entry fill price (already slippage-adjusted if applicable).
        exit_price : float
            Exit fill price (already slippage-adjusted if applicable).
        size : float
            Number of units traded.

        Returns
        -------
        float
            Total round-trip commission cost in quote currency.
        """
        entry_commission = self.calculate_commission(entry_price, size)
        exit_commission = self.calculate_commission(exit_price, size)
        return entry_commission + exit_commission

    def calculate_net_pnl(
        self,
        gross_pnl: float,
        entry_price: float,
        exit_price: float,
        size: float,
    ) -> float:
        """Compute net PnL after subtracting round-trip costs.

        Parameters
        ----------
        gross_pnl : float
            PnL before costs (may already include slippage adjustment in
            the prices used to calculate it).
        entry_price : float
            Entry fill price.
        exit_price : float
            Exit fill price.
        size : float
            Number of units traded.

        Returns
        -------
        float
            Net PnL in quote currency.
        """
        total_costs = self.calculate_total_costs(entry_price, exit_price, size)
        return gross_pnl - total_costs
