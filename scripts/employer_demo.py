import os
import sys
import time
import threading
import datetime
import requests
import pandas as pd
import numpy as np

# Add parent directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SystemConfig
from data.parquet_manager import load_data, split_data, get_data_path
from backtesting.backtest_engine import BacktestEngine
from paper_trading.webhook_server import start_server

def print_executive_banner():
    banner = """
===================================================================================
             BTCUSDT QUANTITATIVE TRADING SYSTEM - EXECUTIVE DEMO
===================================================================================
                  QUANTITATIVE TRADING INTERNSHIP FINAL DEMO
         Renko Market Representation | Triple-Gate Institutional Filtering
===================================================================================
"""
    print(banner)

def run_live_demo_stream():
    print_executive_banner()
    print("[System] Launching FastAPI Interactive Dashboard Server on port 8000...")
    
    # Start webhook server in background thread
    server_thread = threading.Thread(target=start_server, kwargs={"host": "127.0.0.1", "port": 8000}, daemon=True)
    server_thread.start()
    
    # Give server a moment to start
    time.sleep(2)
    
    print("\n" + "="*83)
    print(" DASHBOARD READY: Open http://localhost:8000 in your browser to view the demo!")
    print("="*83 + "\n")
    
    print("[System] Loading historical BTCUSDT data feed for high-fidelity simulation...")
    config = SystemConfig.default()
    timeframe = "1h"
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw')
    file_path = get_data_path("BTCUSDT", timeframe, data_dir)
    
    try:
        df = load_data(file_path)
    except Exception as e:
        print(f"[Error] Could not load parquet data: {e}")
        return
        
    df_dev, _, _ = split_data(df, config.backtest.backtest_end, config.backtest.forward_end)
    if 'Open Time' not in df_dev.columns:
        df_dev = df_dev.reset_index(names='Open Time')
    
    # Run the backtest with all 3 filters to get the exact trade timeline
    print("[System] Pre-computing strategy signals with ATR + Trend + Min R:R Filters...")
    engine = BacktestEngine(config, use_risk_management=True, use_ml_filter=False)
    res = engine.run(df_dev, timeframe=timeframe, config_name="S2_Filters", use_atr_filter=True, use_trend_filter=True, use_min_rr_filter=True)
    trades_df = res.trades_df
    
    # Pre-compute indicators for the stream
    from indicators.atr_engine import compute_atr
    from indicators.moving_averages import compute_ema
    atr_series = compute_atr(df_dev, period=14).values
    ema_200 = compute_ema(df_dev['Close'], 200).values
    
    print("[System] Starting Mode-Switching Webhook Stream to Dashboard...")
    webhook_url = "http://127.0.0.1:8000/webhook"
    status_url = "http://127.0.0.1:8000/status"
    binance_url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    
    capital = config.backtest.initial_capital
    trade_idx = 0
    
    # Send initial status log
    requests.post(webhook_url, json={
        "action": "LOG",
        "price": float(df_dev['Close'].iloc[0]),
        "capital": capital,
        "log_message": "🚀 Live simulation stream initialized. Ready for mode switching...",
        "timestamp": str(df_dev['Open Time'].iloc[0])
    })
    
    if trades_df.empty:
        print("[System] No trades generated in filtered backtest.")
        return

    # Main dual-mode streaming loop
    try:
        while True:
            # Check active mode from server
            try:
                status_resp = requests.get(status_url, timeout=2)
                if status_resp.status_code == 200:
                    current_mode = status_resp.json().get("mode", "historical")
                else:
                    current_mode = "historical"
            except Exception:
                current_mode = "historical"

            if current_mode == "live":
                # Fetch real-time live price from Binance REST API
                try:
                    resp = requests.get(binance_url, timeout=5)
                    if resp.status_code == 200:
                        live_price = float(resp.json()['price'])
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        latest_atr = float(atr_series[-1])
                        latest_ema = float(ema_200[-1])
                        latest_brick = latest_atr * config.strategy.atr_multiplier
                        
                        requests.post(webhook_url, json={
                            "action": "LOG",
                            "price": live_price,
                            "capital": capital,
                            "log_message": f"🔴 LIVE BINANCE FEED: BTC/USDT @ ${live_price:,.2f} | ATR: {latest_atr:.2f} | 200 EMA: ${latest_ema:,.2f}",
                            "timestamp": ts,
                            "curr_atr": latest_atr,
                            "curr_ema": latest_ema,
                            "brick_size": latest_brick
                        })
                except Exception:
                    pass # Suppress temporary network hiccups
                time.sleep(2)
            else:
                # Perform historical simulation step
                if trade_idx >= len(trades_df):
                    trade_idx = 0 # Loop historical demo so it's always ready for presentation
                    
                trade = trades_df.iloc[trade_idx]
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                
                # Find index in df_dev matching entry_time
                mask = df_dev['Open Time'] <= entry_time
                if mask.any():
                    bar_idx = df_dev[mask].index[-1]
                else:
                    bar_idx = 0
                    
                curr_atr = float(atr_series[bar_idx])
                curr_ema = float(ema_200[bar_idx])
                brick_size = curr_atr * config.strategy.atr_multiplier
                
                # Send pre-trade market ticks
                for offset in [2, 1]:
                    if bar_idx >= offset:
                        t_idx = bar_idx - offset
                        p = float(df_dev['Close'].iloc[t_idx])
                        ts = str(df_dev['Open Time'].iloc[t_idx])
                        requests.post(webhook_url, json={
                            "action": "LOG",
                            "price": p,
                            "capital": capital,
                            "log_message": f"📊 Market Update: BTC/USDT @ ${p:,.2f} | ATR: {curr_atr:.2f} | 200 EMA: ${curr_ema:,.2f}",
                            "timestamp": ts,
                            "curr_atr": curr_atr,
                            "curr_ema": curr_ema,
                            "brick_size": brick_size
                        })
                        time.sleep(1.0)
                        
                # Send OPEN trade
                direction_str = "LONG" if trade['direction'] == 1 else "SHORT"
                print(f"[{entry_time}] OPEN {direction_str} @ ${trade['entry_price']:,.2f} | Size: {trade['size']:.4f} BTC")
                requests.post(webhook_url, json={
                    "action": "OPEN",
                    "direction": direction_str,
                    "price": float(trade['entry_price']),
                    "capital": capital,
                    "timestamp": str(entry_time),
                    "curr_atr": curr_atr,
                    "curr_ema": curr_ema,
                    "brick_size": brick_size
                })
                time.sleep(1.5)
                
                # Send mid-trade tick
                requests.post(webhook_url, json={
                    "action": "LOG",
                    "price": float((trade['entry_price'] + trade['exit_price']) / 2.0),
                    "capital": capital,
                    "log_message": f"⏳ Position Active: Tracking trailing stop and profit targets...",
                    "timestamp": str(entry_time + (exit_time - entry_time)/2)
                })
                time.sleep(1.2)
                
                # Send CLOSE trade
                capital += trade['net_pnl']
                print(f"[{exit_time}] CLOSE position @ ${trade['exit_price']:,.2f} | PnL: ${trade['net_pnl']:,.2f} | Capital: ${capital:,.2f}")
                requests.post(webhook_url, json={
                    "action": "CLOSE",
                    "price": float(trade['exit_price']),
                    "reason": trade.get('exit_reason', 'risk_management'),
                    "pnl": float(trade['net_pnl']),
                    "capital": capital,
                    "timestamp": str(exit_time),
                    "curr_atr": curr_atr,
                    "curr_ema": curr_ema,
                    "brick_size": brick_size
                })
                
                trade_idx += 1
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("[System] Shutting down demo server.")

if __name__ == "__main__":
    run_live_demo_stream()
