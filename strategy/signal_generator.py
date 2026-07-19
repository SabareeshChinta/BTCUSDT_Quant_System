"""
Signal Generator — ATR Renko Two-Brick Confirmation Strategy.

Generates BUY/SELL signals from Renko brick sequences using a configurable
consecutive-brick confirmation threshold.  A signal fires exactly ONCE when
the threshold is first met and will not re-fire until an opposite signal
has been generated.
"""

from __future__ import annotations

import pandas as pd


class SignalGenerator:
    """Produces directional signals from a stream of Renko bricks.

    Parameters
    ----------
    consecutive_bricks : int
        Number of same-direction bricks required before a signal is
        confirmed.  Default is 2 (the classic two-brick reversal rule).
    """

    def __init__(self, consecutive_bricks: int = 2) -> None:
        if consecutive_bricks < 1:
            raise ValueError("consecutive_bricks must be >= 1")
        self.consecutive_bricks = consecutive_bricks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self, renko_bricks: pd.DataFrame) -> pd.DataFrame:
        """Augment a Renko-bricks DataFrame with signal columns.

        Parameters
        ----------
        renko_bricks : pd.DataFrame
            Must contain at least a ``direction`` column with values in
            {1, -1}.  Typically the output of ``RenkoEngine.build_bricks``.

        Returns
        -------
        pd.DataFrame
            A *copy* of the input with three new columns:

            * ``consecutive_count`` – running count of same-direction bricks.
            * ``raw_signal`` – ``'BUY'``, ``'SELL'``, or ``None``.
            * ``signal_changed`` – ``True`` only on the brick where a *new*
              signal first fires.

        Notes
        -----
        No look-ahead bias: every row is computed using only current and
        prior rows.
        """
        if renko_bricks.empty:
            return self._empty_result(renko_bricks)

        if "direction" not in renko_bricks.columns:
            raise ValueError("renko_bricks must contain a 'direction' column")

        df = renko_bricks.copy()
        directions = df["direction"].values

        n = len(directions)
        consecutive_counts = self._compute_consecutive_counts(directions, n)
        raw_signals, signal_changed = self._compute_signals(
            directions, consecutive_counts, n
        )

        df["consecutive_count"] = consecutive_counts
        df["raw_signal"] = raw_signals
        df["signal_changed"] = signal_changed

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_consecutive_counts(
        directions, n: int
    ) -> list[int]:
        """Count consecutive same-direction bricks (no look-ahead)."""
        counts: list[int] = [0] * n
        if n == 0:
            return counts

        counts[0] = 1
        for i in range(1, n):
            if directions[i] == directions[i - 1]:
                counts[i] = counts[i - 1] + 1
            else:
                counts[i] = 1
        return counts

    def _compute_signals(
        self,
        directions,
        consecutive_counts: list[int],
        n: int,
    ) -> tuple[list[str | None], list[bool]]:
        """Determine raw_signal and signal_changed for each brick.

        A signal fires exactly once when ``consecutive_count`` first reaches
        the ``consecutive_bricks`` threshold.  It will not fire again for the
        same direction until the opposite signal has been generated.
        """
        raw_signals: list[str | None] = [None] * n
        signal_changed: list[bool] = [False] * n

        # Track the last signal that was emitted so we can suppress
        # duplicate signals of the same type.
        last_emitted_signal: str | None = None

        threshold = self.consecutive_bricks

        for i in range(n):
            if consecutive_counts[i] == threshold:
                # Threshold just met on this brick.
                signal = "BUY" if directions[i] == 1 else "SELL"

                if signal != last_emitted_signal:
                    raw_signals[i] = signal
                    signal_changed[i] = True
                    last_emitted_signal = signal
                # If signal == last_emitted_signal the threshold was met
                # again (e.g. after a series of bricks that didn't trigger
                # the opposite signal).  We suppress it.

        return raw_signals, signal_changed

    @staticmethod
    def _empty_result(df: pd.DataFrame) -> pd.DataFrame:
        """Return an empty DataFrame with the expected output columns."""
        out = df.copy()
        out["consecutive_count"] = pd.Series(dtype="int64")
        out["raw_signal"] = pd.Series(dtype="object")
        out["signal_changed"] = pd.Series(dtype="bool")
        return out
