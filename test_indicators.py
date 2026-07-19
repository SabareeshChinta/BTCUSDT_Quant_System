"""Smoke-test for indicator engines."""

import pandas as pd
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators.atr_engine import ATREngine, compute_atr
from indicators.renko_engine import RenkoEngine
from indicators.moving_averages import compute_sma, compute_ema, compute_ma, get_trend_state

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btcusdt_1h.parquet")

def main() -> None:
    print("=" * 60)
    print("INDICATOR ENGINE SMOKE TEST")
    print("=" * 60)

    df = pd.read_parquet(DATA_FILE)
    print(f"\nLoaded {len(df):,} candles  ({df['Open Time'].iloc[0]} → {df['Open Time'].iloc[-1]})")

    # ---- ATR Engine ----
    print("\n--- ATR Engine ---")
    for method in ("sma", "ema", "wilder"):
        atr = compute_atr(df, period=14, smoothing=method)
        last_val = atr.dropna().iloc[-1]
        print(f"  ATR-14 ({method:>6s}): {last_val:>10.2f}  (non-NaN count: {atr.notna().sum():,})")

    # ---- Moving Averages ----
    print("\n--- Moving Averages ---")
    sma_20 = compute_sma(df["Close"], 20)
    ema_50 = compute_ema(df["Close"], 50)
    sma_50 = compute_ma(df["Close"], 50, "SMA")
    trend = get_trend_state(sma_20, sma_50)
    print(f"  SMA-20 last: {sma_20.dropna().iloc[-1]:.2f}")
    print(f"  EMA-50 last: {ema_50.dropna().iloc[-1]:.2f}")
    print(f"  SMA-50 last: {sma_50.dropna().iloc[-1]:.2f}")
    print(f"  Trend (SMA20 vs SMA50) last: {trend.iloc[-1]}")

    # ---- Renko Engine ----
    print("\n--- Renko Engine ---")
    renko = RenkoEngine(atr_period=14, atr_multiplier=1.0, atr_smoothing="sma")
    bricks = renko.build_bricks(df)
    print(f"  Total bricks:    {len(bricks):,}")
    if not bricks.empty:
        bullish = (bricks["direction"] == 1).sum()
        bearish = (bricks["direction"] == -1).sum()
        print(f"  Bullish bricks:  {bullish:,}")
        print(f"  Bearish bricks:  {bearish:,}")
        print(f"  Avg brick size:  {bricks['brick_size'].mean():.2f}")
        print(f"  First brick:     {bricks.iloc[0]['brick_open_time']}")
        print(f"  Last brick:      {bricks.iloc[-1]['brick_close_time']}")

        consec = renko.get_consecutive_count(bricks)
        print(f"  Max consecutive: {consec.max()}")
        print(f"\n  Last 5 bricks:")
        print(bricks.tail().to_string(index=False))

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
