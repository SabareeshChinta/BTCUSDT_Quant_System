"""
Live Data Feed for Deployment Simulation.
Uses Binance WebSockets to stream real-time price updates.
"""

import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from binance import AsyncClient, BinanceSocketManager

from config.settings import SystemConfig
from paper_trading.paper_trade_engine import PaperTradeEngine

class LiveFeedManager:
    def __init__(self, symbol: str, timeframe: str, paper_engine: PaperTradeEngine):
        self.symbol = symbol
        self.timeframe = timeframe
        self.paper_engine = paper_engine
        
    async def start_stream(self):
        client = await AsyncClient.create()
        bm = BinanceSocketManager(client)
        
        # We listen to klines (candlesticks)
        ts = bm.kline_socket(self.symbol, interval=self.timeframe)
        print(f"Starting Live Feed for {self.symbol} on {self.timeframe} interval...")
        
        async with ts as tscm:
            while True:
                try:
                    res = await tscm.recv()
                    if res and 'k' in res:
                        kline = res['k']
                        is_closed = kline['x']
                        close_price = float(kline['c'])
                        timestamp = pd.to_datetime(kline['T'], unit='ms')
                        
                        # Process tick for trailing stops/TP/SL
                        self.paper_engine.process_tick(close_price, timestamp)
                        
                        if self.paper_engine.webhook_callback:
                            self.paper_engine.webhook_callback({
                                "action": "TICK",
                                "price": close_price,
                                "capital": self.paper_engine.capital,
                                "timestamp": timestamp.strftime("%H:%M:%S")
                            })
                            
                        if is_closed:
                            print(f"[{timestamp}] Candle Closed: {close_price}")
                            # In a real system, here we would:
                            # 1. Update historical dataframe
                            # 2. Re-run Renko engine / MA / ML
                            # 3. Check for new signals
                            # 4. paper_engine.execute_signal(...) if new signal found
                            
                except Exception as e:
                    print(f"WebSocket Error: {e}")
                    await asyncio.sleep(5)
        
        await client.close_connection()

import requests

def webhook_post(payload: dict):
    try:
        requests.post("http://127.0.0.1:8000/webhook", json=payload, timeout=1)
    except:
        pass

def run_live_feed(symbol: str, timeframe: str, config: SystemConfig):
    engine = PaperTradeEngine(config, webhook_callback=webhook_post)
    feed = LiveFeedManager(symbol, timeframe, engine)
    
    asyncio.run(feed.start_stream())

if __name__ == "__main__":
    from config.settings import SystemConfig
    run_live_feed("BTCUSDT", "1m", SystemConfig())
