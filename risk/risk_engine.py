"""
Risk Engine — Stop-Loss / Take-Profit / Trailing-Stop management.

Computes risk levels from ATR-based multipliers and checks whether the
current bar triggers an exit.  Uses conservative fill assumptions to avoid
optimistic backtest results.
"""

from __future__ import annotations

import pandas as pd

from strategy.position_manager import Position


class RiskEngine:
    """ATR-based risk management with SL, TP, and trailing stop.

    Parameters
    ----------
    sl_atr_mult : float
        Stop-loss distance as a multiple of ATR.
    tp_atr_mult : float
        Take-profit distance as a multiple of ATR.
    tsl_atr_mult : float
        Trailing-stop distance as a multiple of ATR.
    enabled : bool
        Master switch — when ``False``, ``check_exit`` always returns
        ``(False, '', 0.0)``.
    """

    def __init__(
        self,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.0,
        tsl_atr_mult: float = 1.0,
        enabled: bool = True,
    ) -> None:
        if sl_atr_mult <= 0:
            raise ValueError(f"sl_atr_mult must be > 0, got {sl_atr_mult}")
        if tp_atr_mult <= 0:
            raise ValueError(f"tp_atr_mult must be > 0, got {tp_atr_mult}")
        if tsl_atr_mult <= 0:
            raise ValueError(f"tsl_atr_mult must be > 0, got {tsl_atr_mult}")

        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.tsl_atr_mult = tsl_atr_mult
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_levels(
        self,
        entry_price: float,
        direction: int,
        atr: float,
    ) -> dict:
        """Compute SL, TP, and trailing-stop distance from ATR.

        Parameters
        ----------
        entry_price : float
            Position entry price.
        direction : int
            ``1`` for long, ``-1`` for short.
        atr : float
            ATR value at entry.

        Returns
        -------
        dict
            Keys: ``stop_loss``, ``take_profit``, ``trailing_stop_distance``,
            ``trailing_stop`` (initial trailing-stop level, same as SL side).
        """
        if direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {direction}")
        if atr <= 0:
            raise ValueError(f"atr must be > 0, got {atr}")

        sl_distance = atr * self.sl_atr_mult
        tp_distance = atr * self.tp_atr_mult
        tsl_distance = atr * self.tsl_atr_mult

        if direction == 1:  # LONG
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
            trailing_stop = entry_price - tsl_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            trailing_stop = entry_price + tsl_distance

        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop_distance": tsl_distance,
            "trailing_stop": trailing_stop,
        }

    def check_exit(
        self,
        position: Position,
        current_open: float,
        current_high: float,
        current_low: float,
        current_close: float,
        current_time: pd.Timestamp,
        levels: dict,
    ) -> tuple[bool, str, float]:
        """Check whether any risk level has been breached.

        Parameters
        ----------
        position : Position
            The open position.
        current_open : float
            Open of the current bar.
        current_high : float
            High of the current bar.
        current_low : float
            Low of the current bar.
        current_close : float
            Close of the current bar.
        current_time : pd.Timestamp
            Timestamp of the current bar.
        levels : dict
            Output of :meth:`calculate_levels` (possibly with an updated
            ``trailing_stop``).

        Returns
        -------
        tuple[bool, str, float]
            ``(should_exit, exit_reason, exit_price)``.  If no exit is
            triggered, returns ``(False, '', 0.0)``.

        Notes
        -----
        **Conservative fill assumptions** — when an exit is triggered the
        fill price is set to the *level* price, not to the bar extreme.
        This avoids crediting the backtest with unrealistically favourable
        fills.

        Check order: Stop-Loss → Take-Profit → Trailing-Stop.
        """
        if not self.enabled:
            return False, "", 0.0

        sl = levels["stop_loss"]
        tp = levels["take_profit"]
        tsl = levels.get("trailing_stop", sl)  # fallback to SL if missing

        direction = position.direction

        # --- LONG exits ----------------------------------------------------
        if direction == 1:
            if current_low <= sl:
                exit_price = min(current_open, sl) if current_open <= sl else sl
                return True, "stop_loss", exit_price
            if current_high >= tp:
                exit_price = max(current_open, tp) if current_open >= tp else tp
                return True, "take_profit", exit_price
            if current_low <= tsl:
                exit_price = min(current_open, tsl) if current_open <= tsl else tsl
                return True, "trailing_stop", exit_price

        # --- SHORT exits ---------------------------------------------------
        else:
            if current_high >= sl:
                exit_price = max(current_open, sl) if current_open >= sl else sl
                return True, "stop_loss", exit_price
            if current_low <= tp:
                exit_price = min(current_open, tp) if current_open <= tp else tp
                return True, "take_profit", exit_price
            if current_high >= tsl:
                exit_price = max(current_open, tsl) if current_open >= tsl else tsl
                return True, "trailing_stop", exit_price

        return False, "", 0.0

    def update_trailing_stop(
        self,
        direction: int,
        current_high: float,
        current_low: float,
        current_trailing_stop: float,
        trailing_distance: float,
    ) -> float:
        """Move the trailing stop in the favourable direction only.

        Parameters
        ----------
        direction : int
            ``1`` for long, ``-1`` for short.
        current_high : float
            High of the current bar.
        current_low : float
            Low of the current bar.
        current_trailing_stop : float
            Current trailing-stop level.
        trailing_distance : float
            Fixed distance from the bar extreme to the stop.

        Returns
        -------
        float
            Updated trailing-stop level (never moves backwards).
        """
        if trailing_distance > 10000:
            return current_trailing_stop
            
        if direction == 1:
            candidate = current_high - trailing_distance
            return max(candidate, current_trailing_stop)
        else:
            candidate = current_low + trailing_distance
            return min(candidate, current_trailing_stop)
