import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from indicators.atr_engine import compute_atr

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up = high - high.shift()
    down = low.shift() - low
    
    pos_dm = np.where((up > down) & (up > 0), up, 0)
    neg_dm = np.where((down > up) & (down > 0), down, 0)
    
    tr_smooth = tr.rolling(period).sum()
    pos_dm_smooth = pd.Series(pos_dm, index=df.index).rolling(period).sum()
    neg_dm_smooth = pd.Series(neg_dm, index=df.index).rolling(period).sum()
    
    pos_di = 100 * (pos_dm_smooth / tr_smooth)
    neg_di = 100 * (neg_dm_smooth / tr_smooth)
    
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di)
    adx = dx.rolling(period).mean()
    return adx

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def run_extraction():
    config = SystemConfig.default()
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    tf = "1h" # We'll do 1h since it was the robust one, or 15m. Let's do 1h for clearer structure.
    file_path = get_data_path("BTCUSDT", tf, data_dir)
    
    print("Loading data...")
    df = load_data(file_path)
    
    print("Computing contextual indicators...")
    # Compute indicators BEFORE the trade execution (shift by 1 to ensure zero look-ahead)
    rsi = compute_rsi(df['Close'], 14).shift(1)
    adx = compute_adx(df, 14).shift(1)
    atr = compute_atr(df, 14).shift(1)
    
    rsi_slope = rsi.diff(3)
    adx_slope = adx.diff(3)
    
    # 90-day rolling ATR Percentile (roughly 2160 hours)
    atr_percentile = atr.rolling(2160).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).shift(1)
    
    fast_ema = compute_ema(df['Close'], config.strategy.fast_ma_period).shift(1)
    slow_ema = compute_ema(df['Close'], config.strategy.slow_ma_period).shift(1)
    ema_distance = (fast_ema - slow_ema) / slow_ema * 100
    ema_slope = fast_ema.diff(3)
    
    # Run the base backtest over the full period
    print("Running Base Engine to generate trades...")
    engine = BacktestEngine(config=config, use_risk_management=True, use_ml_filter=False)
    res = engine.run(df, timeframe=tf, config_name="Base_Extraction")
    
    trades = res.trades_df.copy()
    if trades.empty:
        print("No trades generated.")
        return
        
    print("Annotating trades with contextual features...")
    # Trades have 'entry_time'. We match it with the features.
    
    features_list = []
    for idx, row in trades.iterrows():
        entry_time = pd.to_datetime(row['entry_time'])
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize('UTC')
            
        exit_time = pd.to_datetime(row['exit_time'])
        if pd.notna(exit_time) and exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize('UTC')
        # Find the exact candle in df
        if entry_time in df.index:
            t_idx = entry_time
        else:
            # Fallback to closest previous
            t_idx = df.index[df.index <= entry_time][-1]
            
        feat = {
            "trade_id": idx,
            "direction": row['direction'],
            "entry_time": entry_time,
            "exit_time": exit_time,
            "net_pnl": row['net_pnl'],
            "trade_duration_hrs": (exit_time - entry_time).total_seconds() / 3600,
            "MAE": row['mae'],
            "MFE": row['mfe'],
            "rsi": rsi.loc[t_idx],
            "rsi_slope": rsi_slope.loc[t_idx],
            "adx": adx.loc[t_idx],
            "adx_slope": adx_slope.loc[t_idx],
            "atr": atr.loc[t_idx],
            "atr_percentile": atr_percentile.loc[t_idx],
            "ema_distance": ema_distance.loc[t_idx],
            "ema_slope": ema_slope.loc[t_idx],
            "time_of_day": entry_time.hour,
            "day_of_week": entry_time.dayofweek
        }
        features_list.append(feat)
        
    annotated_df = pd.DataFrame(features_list)
    out_path = os.path.join(os.path.dirname(__file__), "trade_context_dataset.csv")
    annotated_df.to_csv(out_path, index=False)
    print(f"Extraction complete! Saved to {out_path}")

if __name__ == '__main__':
    run_extraction()
