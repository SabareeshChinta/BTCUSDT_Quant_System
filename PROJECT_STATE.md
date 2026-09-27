# BTCUSDT Quant System — Complete Project State & Memory Blueprint

**Last Updated:** September 27, 2026  
**Project Workspace:** `c:\Users\chint\BTCUSDT_Quant_System`  
**Asset & Exchange:** `BTC/USDT` Perpetual Futures (Binance)

---

## 1. System Overview & Core Parameters

All core system configurations are stored in `config/settings.py`.

* **Active Production Strategy:** `SupertrendRSI` (Supertrend Momentum + RSI Boundary Filter)
* **Default Timeframe:** `1h` (with `30m` supported)
* **Strategy Parameters (`StrategyConfig` in `config/settings.py`):**
  * `st_period = 10` (Supertrend ATR Period)
  * `st_mult = 2.5` (Supertrend Band Multiplier)
  * `rsi_period = 14` (RSI Lookback)
  * `rsi_max_long = 75.0` (Max RSI allowed for Long entries)
  * `rsi_min_short = 25.0` (Min RSI allowed for Short entries)
* **Risk Parameters (`RiskConfig` in `config/settings.py`):**
  * `stop_loss_atr_mult = 2.0` (Dynamic ATR Stop Loss)
  * `take_profit_atr_mult = 4.0` (Dynamic ATR Take Profit — 1:2 Risk-to-Reward Ratio)
  * `risk_per_trade_pct = 0.01` (1.0% Account Risk per trade)
  * `commission_pct = 0.0005` (5 bps Taker Fee)
  * `slippage_pct = 0.0005` (5 bps Slippage per fill)

---

## 2. Key Codebase Files & Architecture

| File Path | Description |
| :--- | :--- |
| `config/settings.py` | Central dataclass configurations (`SystemConfig`, `StrategyConfig`, `RiskConfig`, `MLConfig`, `BacktestConfig`). |
| `strategy/theses.py` | Contains `SupertrendRSI`, `HTFTrendPullback`, `DonchianBreakout`, `BollingerMeanReversion`, `ADXRegime`, `ATRBurst`. |
| `scripts/fast_backtest_engine.py` | High-performance zero-lookahead backtest execution engine with slippage, taker fees, and position management. |
| `scripts/strategy_library.py` | Strict causal indicator library (`compute_supertrend`, `compute_rsi`, `compute_atr`, `compute_ema`, etc.). |
| `run_best_strategy.py` | Execution runner for the active strategy on historical/recent datasets. |
| `paper_trading_last_30_days.py` | Paper trading test script for the exact last 30 days (Aug 28 – Sep 27, 2026). |
| `test_august.py` | August 2026 backtesting script across all timeframes. |
| `test_last_month.py` | July 2026 backtesting script across all timeframes. |

---

## 3. Latest Benchmark Results

### A. Last 30 Days (Aug 28 – Sep 27, 2026) — Paper Trading Benchmark
* **Strategy:** `Supertrend + RSI Filter (10, 2.5)` on `1h`
* **Trades:** 23 Completed Trades
* **Win Rate:** 39.1%
* **Profit Factor:** 1.09
* **Net Realized P&L:** **+$1,274.77** (after deducting $0.05% fee + 0.05% slippage per side)
* **Max Drawdown:** 4.70%
* **Sharpe Ratio:** 1.47

### B. July 2026 Out-of-Sample Benchmark
* **Strategy:** `Supertrend + RSI Filter (10, 2.5)` on `1h`
* **Trades:** 18 Completed Trades
* **Win Rate:** 44.4%
* **Profit Factor:** 1.47
* **Net Realized P&L:** **+$4,340.88** (after deducting fees & slippage)
* **Max Drawdown:** 2.53%
* **Sharpe Ratio:** 2.70

### C. August 2026 Out-of-Sample Benchmark
* **Strategy:** `HTF Trend + RSI Pullback (20/150)` on `30m`
* **Trades:** 20 Completed Trades
* **Win Rate:** 60.0%
* **Profit Factor:** 1.90
* **Net Realized P&L:** **+$8,226.82** (after deducting fees & slippage)
* **Max Drawdown:** 2.46%
* **Sharpe Ratio:** 2.49

---

## 4. Instructions for Any New AI Assistant / Session

If you start a new chat or switch accounts, instruct the AI:

> "Read `PROJECT_STATE.md` in the workspace root to load all project context, strategy configurations, codebase architecture, and latest backtest benchmarks."

The AI will inspect this file and immediately pick up right where we left off.
