"""
Position Manager — Stateful position tracking for the trading system.

Tracks the current position (FLAT / LONG / SHORT), handles open / close /
flip operations, and emits structured trade-record dicts for downstream
analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pandas as pd


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class PositionState(Enum):
    """Possible position states."""

    FLAT = auto()
    LONG = auto()
    SHORT = auto()


@dataclass
class Position:
    """Snapshot of an open position.

    Attributes
    ----------
    state : PositionState
        Current state (LONG or SHORT — a FLAT position is represented by
        ``None`` rather than a ``Position`` instance).
    entry_price : float
        Fill price at entry.
    entry_time : pd.Timestamp
        Timestamp of the entry bar / brick.
    direction : int
        ``1`` for long, ``-1`` for short.
    size : float
        Number of units held (always positive).
    entry_atr : float
        ATR value at entry time, used for risk-level calculations.
    """

    state: PositionState
    entry_price: float
    entry_time: pd.Timestamp
    direction: int  # 1 = long, -1 = short
    size: float
    entry_atr: float
    sig_time: pd.Timestamp = None
    dollar_risk: float = 0.0
    stop_distance: float = 0.0
    context: dict = None


# ---------------------------------------------------------------------------
# Position Manager
# ---------------------------------------------------------------------------

class PositionManager:
    """Manages position lifecycle: open → hold → close / flip.

    The manager ensures that only one position can be active at a time
    and generates trade records on close.
    """

    def __init__(self) -> None:
        self._position: Position | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_position(self) -> Position | None:
        """The currently open position, or ``None`` if flat."""
        return self._position

    @property
    def is_flat(self) -> bool:
        """``True`` if no position is open."""
        return self._position is None

    @property
    def is_long(self) -> bool:
        """``True`` if currently holding a long position."""
        return (
            self._position is not None
            and self._position.state == PositionState.LONG
        )

    @property
    def is_short(self) -> bool:
        """``True`` if currently holding a short position."""
        return (
            self._position is not None
            and self._position.state == PositionState.SHORT
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def open_position(
        self,
        direction: int,
        price: float,
        time: pd.Timestamp,
        size: float,
        atr: float,
        sig_time: pd.Timestamp = None,
        dollar_risk: float = 0.0,
        stop_distance: float = 0.0,
        context: dict = None,
    ) -> Position:
        """Open a new position.  Must be FLAT beforehand.

        Parameters
        ----------
        direction : int
            ``1`` for long, ``-1`` for short.
        price : float
            Entry fill price.
        time : pd.Timestamp
            Entry timestamp.
        size : float
            Number of units (positive).
        atr : float
            ATR value at entry for risk calculations.
        sig_time: pd.Timestamp
            Timestamp of the signal generation.

        Returns
        -------
        Position
            The newly created position.

        Raises
        ------
        RuntimeError
            If a position is already open.
        ValueError
            If *direction* is not ``1`` or ``-1``, or *size* / *atr* are
            non-positive.
        """
        if not self.is_flat:
            raise RuntimeError(
                "Cannot open a position while another is active. "
                "Close or flip the current position first."
            )
        self._validate_direction(direction)
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if atr <= 0:
            raise ValueError(f"atr must be > 0, got {atr}")

        state = PositionState.LONG if direction == 1 else PositionState.SHORT
        self._position = Position(
            state=state,
            entry_price=price,
            entry_time=time,
            direction=direction,
            size=size,
            entry_atr=atr,
            sig_time=sig_time,
            dollar_risk=dollar_risk,
            stop_distance=stop_distance,
            context=context,
        )
        return self._position

    def close_position(
        self,
        price: float,
        time: pd.Timestamp,
        reason: str,
    ) -> dict:
        """Close the current position and return a trade record.

        Parameters
        ----------
        price : float
            Exit fill price.
        time : pd.Timestamp
            Exit timestamp.
        reason : str
            Human-readable exit reason (e.g. ``'SL'``, ``'TP'``,
            ``'signal_flip'``).

        Returns
        -------
        dict
            Trade record with keys: ``entry_time``, ``exit_time``,
            ``direction``, ``entry_price``, ``exit_price``, ``size``,
            ``gross_pnl``, ``holding_duration_hours``, ``exit_reason``.

        Raises
        ------
        RuntimeError
            If no position is currently open.
        """
        if self.is_flat:
            raise RuntimeError("No position to close.")

        pos = self._position
        gross_pnl = (price - pos.entry_price) * pos.direction * pos.size

        holding_td = time - pos.entry_time
        holding_hours = holding_td.total_seconds() / 3600.0

        # Calculate R-multiple
        r_multiple = 0.0
        if pos.dollar_risk > 0:
            r_multiple = gross_pnl / pos.dollar_risk

        record = {
            "entry_time": pos.entry_time,
            "exit_time": time,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": pos.size,
            "gross_pnl": gross_pnl,
            "holding_duration_hours": holding_hours,
            "exit_reason": reason,
            "entry_atr": pos.entry_atr,
            "sig_time": pos.sig_time,
            "dollar_risk": pos.dollar_risk,
            "stop_distance": pos.stop_distance,
            "r_multiple": r_multiple,
        }
        
        if pos.context:
            record.update(pos.context)

        self._position = None
        return record

    def scale_out_position(
        self,
        fraction: float,
        price: float,
        time: pd.Timestamp,
        reason: str,
    ) -> dict:
        """Close a fraction of the current position and return a trade record.

        Parameters
        ----------
        fraction : float
            Fraction of the position to close (e.g., 0.5 for 50%).
        price : float
            Exit fill price.
        time : pd.Timestamp
            Exit timestamp.
        reason : str
            Human-readable exit reason.

        Returns
        -------
        dict
            Trade record for the closed fraction.
        """
        if self.is_flat:
            raise RuntimeError("No position to scale out.")
        if not (0 < fraction < 1):
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

        pos = self._position
        exit_size = pos.size * fraction
        gross_pnl = (price - pos.entry_price) * pos.direction * exit_size

        holding_td = time - pos.entry_time
        holding_hours = holding_td.total_seconds() / 3600.0

        r_multiple = 0.0
        # For partial exit, dollar risk is effectively fraction of original
        partial_risk = pos.dollar_risk * fraction
        if partial_risk > 0:
            r_multiple = gross_pnl / partial_risk

        record = {
            "entry_time": pos.entry_time,
            "exit_time": time,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": price,
            "size": exit_size,
            "gross_pnl": gross_pnl,
            "holding_duration_hours": holding_hours,
            "exit_reason": reason,
            "entry_atr": pos.entry_atr,
            "sig_time": pos.sig_time,
            "dollar_risk": partial_risk,
            "stop_distance": pos.stop_distance,
            "r_multiple": r_multiple,
        }

        if pos.context:
            record.update(pos.context)

        # Update remaining position size and dollar risk
        pos.size -= exit_size
        pos.dollar_risk -= partial_risk

        return record

    def flip_position(
        self,
        new_direction: int,
        price: float,
        time: pd.Timestamp,
        size: float,
        atr: float,
        sig_time: pd.Timestamp = None,
        dollar_risk: float = 0.0,
        stop_distance: float = 0.0,
        context: dict = None,
    ) -> tuple[dict, Position]:
        """Close the current position and immediately open one in the
        opposite direction.

        Parameters
        ----------
        new_direction : int
            Direction of the **new** position (``1`` or ``-1``).
        price : float
            Fill price used for both exit and entry.
        time : pd.Timestamp
            Timestamp of the flip.
        size : float
            Size of the new position.
        atr : float
            ATR at the time of the flip.
        sig_time: pd.Timestamp
            Timestamp of the signal generation.

        Returns
        -------
        tuple[dict, Position]
            ``(closed_trade_record, new_position)``.

        Raises
        ------
        RuntimeError
            If no position is currently open.
        ValueError
            If *new_direction* equals the current direction.
        """
        if self.is_flat:
            raise RuntimeError("No position to flip — currently flat.")
        self._validate_direction(new_direction)

        if self._position.direction == new_direction:
            raise ValueError(
                "Cannot flip to the same direction. Use close + open instead."
            )

        closed_record = self.close_position(price, time, reason="signal_flip")
        new_pos = self.open_position(
            new_direction, price, time, size, atr, 
            sig_time=sig_time, dollar_risk=dollar_risk, stop_distance=stop_distance, context=context
        )
        return closed_record, new_pos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_direction(direction: int) -> None:
        if direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got {direction}")
