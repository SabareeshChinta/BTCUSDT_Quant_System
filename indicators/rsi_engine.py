import pandas as pd
import numpy as np

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Computes the Relative Strength Index (RSI) for a given price series.
    """
    delta = prices.diff()
    
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calculate exponential moving average (EMA) of gains and losses
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle division by zero (if avg_loss is 0, RSI is 100)
    rsi = rsi.fillna(100).where(avg_loss != 0, 100)
    
    return rsi
