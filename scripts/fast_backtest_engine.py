import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import itertools

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data.parquet_manager import load_data

@dataclass
class Trade:
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1 for Long, -1 for Short
    size: float
    dollar_risk: float
    stop_loss: float
    take_profit: float
    exit_idx: int = -1
    exit_time: pd.Timestamp = None
    exit_price: float = 0.0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    holding_bars: int = 0

@dataclass
class StrategyResult:
    name: str
    family: str
    timeframe: str
    params: dict
    trades: List[Trade]
    metrics: dict

class FastBacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade_pct: float = 0.01,
        commission_pct: float = 0.0005,  # 5 bps
        slippage_pct: float = 0.0005,    # 5 bps
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,  # 1 for Buy, -1 for Sell, 0 for None (generated on bar t Close)
        atr: np.ndarray,
        sl_mult: float = 2.5,
        tp_mult: float = 4.0,
        min_sl_pct: float = 0.005,
        max_holding_bars: int = 100
    ) -> Tuple[List[Trade], dict]:
        """
        Executes trades sequentially with zero lookahead bias:
        - Signal generated on candle t close.
        - Trade entered on candle t+1 open + slippage.
        - Stops and targets evaluated on candle t+1 to exit.
        """
        n = len(df)
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        times = df['Open Time'].values if 'Open Time' in df.columns else df.index.values

        capital = self.initial_capital
        trades: List[Trade] = []
        in_position = False
        curr_trade: Trade = None

        for i in range(1, n):
            time_now = pd.Timestamp(times[i])
            open_now = opens[i]
            high_now = highs[i]
            low_now = lows[i]
            close_now = closes[i]

            # 1. Manage open position on candle i
            if in_position:
                curr_trade.holding_bars += 1
                direction = curr_trade.direction
                sl = curr_trade.stop_loss
                tp = curr_trade.take_profit
                
                exited = False
                exit_price = 0.0
                exit_reason = ""

                # Check Long exit
                if direction == 1:
                    # Check SL
                    if low_now <= sl:
                        exited = True
                        exit_reason = "stop_loss"
                        exit_price = min(open_now, sl)  # conservative fill
                    # Check TP
                    elif high_now >= tp:
                        exited = True
                        exit_reason = "take_profit"
                        exit_price = max(open_now, tp)
                    # Check opposite signal
                    elif signals[i-1] == -1:
                        exited = True
                        exit_reason = "signal_reversal"
                        exit_price = open_now
                    # Check max holding period
                    elif curr_trade.holding_bars >= max_holding_bars:
                        exited = True
                        exit_reason = "time_exit"
                        exit_price = open_now

                # Check Short exit
                else:
                    if high_now >= sl:
                        exited = True
                        exit_reason = "stop_loss"
                        exit_price = max(open_now, sl)
                    elif low_now <= tp:
                        exited = True
                        exit_reason = "take_profit"
                        exit_price = min(open_now, tp)
                    elif signals[i-1] == 1:
                        exited = True
                        exit_reason = "signal_reversal"
                        exit_price = open_now
                    elif curr_trade.holding_bars >= max_holding_bars:
                        exited = True
                        exit_reason = "time_exit"
                        exit_price = open_now

                if exited:
                    # Apply exit slippage
                    fill_exit = exit_price * (1.0 - self.slippage_pct if direction == 1 else 1.0 + self.slippage_pct)
                    curr_trade.exit_idx = i
                    curr_trade.exit_time = time_now
                    curr_trade.exit_price = fill_exit
                    curr_trade.exit_reason = exit_reason

                    # Calculate PnL
                    gross_diff = (fill_exit - curr_trade.entry_price) * direction
                    gross_pnl = gross_diff * curr_trade.size
                    
                    entry_notional = curr_trade.entry_price * curr_trade.size
                    exit_notional = fill_exit * curr_trade.size
                    fee = (entry_notional + exit_notional) * self.commission_pct
                    slip_cost = (abs(curr_trade.entry_price - open_now) + abs(fill_exit - exit_price)) * curr_trade.size
                    
                    net_pnl = gross_pnl - fee
                    curr_trade.gross_pnl = gross_pnl
                    curr_trade.fee = fee
                    curr_trade.slippage = slip_cost
                    curr_trade.net_pnl = net_pnl
                    curr_trade.return_pct = net_pnl / (entry_notional if entry_notional > 0 else 1.0)

                    capital += net_pnl
                    trades.append(curr_trade)
                    in_position = False
                    curr_trade = None

            # 2. Check entry from previous candle signal (signal at i-1 close, executed at candle i open)
            if not in_position and i > 0 and signals[i-1] != 0:
                sig = signals[i-1]
                entry_fill = open_now * (1.0 + self.slippage_pct if sig == 1 else 1.0 - self.slippage_pct)
                curr_atr = atr[i-1] if not np.isnan(atr[i-1]) and atr[i-1] > 0 else open_now * 0.01
                
                # ATR-based Stop Distance
                stop_dist = max(curr_atr * sl_mult, entry_fill * min_sl_pct)
                dollar_risk = capital * self.risk_per_trade_pct
                position_size = dollar_risk / stop_dist
                
                # Cap position size to max 3x leverage for safety
                max_size = (capital * 3.0) / entry_fill
                position_size = min(position_size, max_size)

                if sig == 1:
                    sl_price = entry_fill - stop_dist
                    tp_price = entry_fill + (curr_atr * tp_mult)
                else:
                    sl_price = entry_fill + stop_dist
                    tp_price = entry_fill - (curr_atr * tp_mult)

                curr_trade = Trade(
                    entry_idx=i,
                    entry_time=time_now,
                    entry_price=entry_fill,
                    direction=sig,
                    size=position_size,
                    dollar_risk=dollar_risk,
                    stop_loss=sl_price,
                    take_profit=tp_price
                )
                in_position = True

        # Close open position at end of backtest
        if in_position and curr_trade is not None:
            last_close = closes[-1]
            last_time = pd.Timestamp(times[-1])
            direction = curr_trade.direction
            fill_exit = last_close * (1.0 - self.slippage_pct if direction == 1 else 1.0 + self.slippage_pct)
            
            gross_pnl = (fill_exit - curr_trade.entry_price) * direction * curr_trade.size
            entry_notional = curr_trade.entry_price * curr_trade.size
            exit_notional = fill_exit * curr_trade.size
            fee = (entry_notional + exit_notional) * self.commission_pct
            net_pnl = gross_pnl - fee
            
            curr_trade.exit_idx = n - 1
            curr_trade.exit_time = last_time
            curr_trade.exit_price = fill_exit
            curr_trade.exit_reason = "end_of_period"
            curr_trade.gross_pnl = gross_pnl
            curr_trade.fee = fee
            curr_trade.net_pnl = net_pnl
            trades.append(curr_trade)

        metrics = self._calculate_metrics(trades)
        return trades, metrics

    def _calculate_metrics(self, trades: List[Trade]) -> dict:
        if not trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'gross_pnl': 0.0,
                'fees': 0.0, 'slippage': 0.0, 'net_pnl': 0.0,
                'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
                'avg_trade_pnl': 0.0, 'avg_trade_pct': 0.0, 'profit_expectancy': 0.0
            }

        pnls = [t.net_pnl for t in trades]
        gross_pnls = [t.gross_pnl for t in trades]
        fees = sum(t.fee for t in trades)
        slippages = sum(t.slippage for t in trades)
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_gain = sum(wins)
        total_loss = abs(sum(losses))
        
        if total_loss == 0:
            profit_factor = 999.0 if total_gain > 0 else 0.0
        else:
            profit_factor = total_gain / total_loss

        # Equity Curve and Max Drawdown
        equity_curve = [self.initial_capital]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)
            
        eq = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq)
        drawdowns = (peaks - eq) / peaks
        max_drawdown_pct = np.max(drawdowns) * 100.0 if len(drawdowns) > 0 else 0.0

        # Sharpe & Sortino
        returns = np.array([t.return_pct for t in trades])
        if len(returns) > 1 and np.std(returns) > 0:
            # Annualized based on ~20 trades/month = ~240 trades/year
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(240)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 1 else np.std(returns)
            sortino = (np.mean(returns) / (downside_std if downside_std > 0 else 1.0)) * np.sqrt(240)
        else:
            sharpe = 0.0
            sortino = 0.0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate * 100.0, 2),
            'profit_factor': round(profit_factor, 2),
            'gross_pnl': round(sum(gross_pnls), 2),
            'fees': round(fees, 2),
            'slippage': round(slippages, 2),
            'net_pnl': round(sum(pnls), 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'avg_trade_pnl': round(np.mean(pnls) if pnls else 0.0, 2),
            'avg_trade_pct': round(np.mean(returns) * 100.0 if len(returns) > 0 else 0.0, 3),
        }
