"""
Central configuration for the BTCUSDT Quant Trading System.

All tunable parameters are organized into dataclasses with sensible defaults.
Use ``SystemConfig.default()`` to obtain a fully-initialized configuration
object that bundles strategy, risk, ML, backtest, and ablation settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Supported timeframes (ascending granularity)
# ---------------------------------------------------------------------------

TIMEFRAMES: List[str] = ["15m", "30m", "1h", "4h", "1d", "1w"]


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Parameters that govern the Supertrend + RSI Momentum Strategy."""

    strategy_type: str = "SupertrendRSI"
    st_period: int = 10
    st_mult: float = 2.5
    rsi_period: int = 14
    rsi_max_long: float = 75.0
    rsi_min_short: float = 25.0
    
    # Legacy Renko parameters (if running Renko variant)
    atr_period: int = 20
    atr_multiplier: float = 3.0
    fast_ma_period: int = 10
    slow_ma_period: int = 50
    ma_type: str = "EMA"
    consecutive_bricks: int = 2
    use_macro_trend_filter: bool = True
    use_volatility_filter: bool = True
    use_reference_strategy: bool = False
    profit_take_bricks: int = 5
    atr_percentile_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.st_period < 1:
            raise ValueError("st_period must be >= 1")
        if self.st_mult <= 0:
            raise ValueError("st_mult must be > 0")


# ---------------------------------------------------------------------------
# Risk management configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """Parameters for position sizing, stops, and transaction costs."""

    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    trailing_stop_atr_mult: float = 999.0
    commission_pct: float = 0.0005      # 0.05 %  (5 bps taker futures)
    slippage_pct: float = 0.0005        # 0.05 %  (5 bps)
    
    # Account Protection
    risk_per_trade_pct: float = 0.01   # 1% risk per trade
    soft_drawdown_warning_pct: float = 0.15 # 15% drawdown from peak
    hard_drawdown_halt_pct: float = 0.25 # 25% drawdown from peak

    def __post_init__(self) -> None:
        if self.stop_loss_atr_mult <= 0:
            raise ValueError("stop_loss_atr_mult must be > 0")
        if self.take_profit_atr_mult <= 0:
            raise ValueError("take_profit_atr_mult must be > 0")


# ---------------------------------------------------------------------------
# Machine-learning overlay configuration
# ---------------------------------------------------------------------------

@dataclass
class MLConfig:
    """Parameters for the ML confidence filter / signal booster."""

    model_type: str = "xgboost"
    feature_window: int = 10
    confidence_threshold: float = 0.50
    train_test_split: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.confidence_threshold < 1:
            raise ValueError("confidence_threshold must be in (0, 1)")
        if not 0 < self.train_test_split < 1:
            raise ValueError("train_test_split must be in (0, 1)")


# ---------------------------------------------------------------------------
# Backtest / walk-forward / paper-trade date ranges
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Date boundaries for the three evaluation phases."""

    initial_capital: float = 100_000.0
    backtest_start: str = "2019-01-01"
    backtest_end: str = "2022-12-31"
    forward_start: str = "2023-01-01"
    forward_end: str = "2025-12-31"
    paper_start: str = "2026-01-01"


# ---------------------------------------------------------------------------
# Ablation study configurations
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """A single ablation-study configuration."""

    name: str
    use_risk_management: bool
    use_ml_filter: bool


@dataclass
class AblationConfig:
    """Pre-defined ablation variants for systematic strategy evaluation.

    S1 – Base strategy only (no RM, no ML)
    S2 – Strategy + Risk Management
    S3 – Strategy + ML filter
    S4 – Full system (RM + ML)
    """

    variants: List[AblationVariant] = field(default_factory=lambda: [
        AblationVariant(name="S1", use_risk_management=False, use_ml_filter=False),
        AblationVariant(name="S2", use_risk_management=True,  use_ml_filter=False),
        AblationVariant(name="S3", use_risk_management=False, use_ml_filter=True),
        AblationVariant(name="S4", use_risk_management=True,  use_ml_filter=True),
    ])

    def get_variant(self, name: str) -> AblationVariant:
        """Return the variant matching *name* (e.g. ``'S2'``)."""
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"Unknown ablation variant: {name!r}")


# ---------------------------------------------------------------------------
# Top-level system configuration
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Aggregated configuration that bundles every sub-config."""

    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    timeframes: List[str] = field(default_factory=lambda: list(TIMEFRAMES))

    @classmethod
    def default(cls) -> "SystemConfig":
        """Factory that returns a ``SystemConfig`` with all default values."""
        return cls()
