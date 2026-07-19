"""
Paper Trading Engine.
Simulates a live trading environment using historical or real-time data feeds.
"""

import pandas as pd
from typing import Callable, Optional

from config.settings import SystemConfig
from strategy.position_manager import PositionManager
from risk.risk_engine import RiskEngine
from risk.cost_model import CostModel
from risk.position_sizer import PositionSizer
from ml.predictor import MLPredictor

class PaperTradeEngine:
    def __init__(self, config: SystemConfig, ml_predictor: Optional[MLPredictor] = None, webhook_callback: Optional[Callable] = None):
        self.config = config
        self.ml_predictor = ml_predictor
        self.webhook_callback = webhook_callback
        
        self.pos_manager = PositionManager()
        self.risk_engine = RiskEngine(
            sl_atr_mult=config.risk.stop_loss_atr_mult,
            tp_atr_mult=config.risk.take_profit_atr_mult,
            tsl_atr_mult=config.risk.trailing_stop_atr_mult,
            enabled=True
        )
        self.cost_model = CostModel(
            commission_pct=config.risk.commission_pct,
            slippage_pct=config.risk.slippage_pct
        )
        self.sizer = PositionSizer()
        
        self.capital = config.backtest.initial_capital
        self.active_risk_levels = None
        
    def process_tick(self, current_price: float, timestamp: pd.Timestamp):
        """Processes a single price tick (or candle close)."""
        pos = self.pos_manager.current_position
        if pos is not None and self.active_risk_levels:
            should_exit, reason, exit_price = self.risk_engine.check_exit(
                pos, current_price, current_price, current_price, timestamp, self.active_risk_levels
            )
            
            if should_exit:
                exit_price_with_slip = self.cost_model.apply_exit_slippage(exit_price, pos.direction)
                trade_rec = self.pos_manager.close_position(exit_price_with_slip, timestamp, reason)
                self.capital += self._calculate_net_pnl(trade_rec)
                self.active_risk_levels = None
                
                self._trigger_webhook({
                    "action": "CLOSE",
                    "direction": "LONG" if pos.direction == 1 else "SHORT",
                    "price": exit_price_with_slip,
                    "reason": reason,
                    "pnl": trade_rec['net_pnl'],
                    "capital": self.capital
                })
            else:
                self.active_risk_levels['trailing_stop'] = self.risk_engine.update_trailing_stop(
                    pos.direction, current_price, current_price,
                    self.active_risk_levels.get('trailing_stop', self.active_risk_levels['stop_loss']),
                    self.active_risk_levels['trailing_stop_distance']
                )

    def execute_signal(self, direction: int, price: float, timestamp: pd.Timestamp, curr_atr: float, ml_features: dict = None):
        """Executes a validated strategy signal."""
        # ML Validation
        if self.ml_predictor and ml_features:
            _, passed = self.ml_predictor.predict(ml_features)
            if not passed:
                print(f"[{timestamp}] Signal ignored: ML confidence too low.")
                return

        exec_price = self.cost_model.apply_entry_slippage(price, direction)
        size = self.sizer.calculate_size(self.capital, exec_price)
        
        pos = self.pos_manager.current_position
        if pos is not None:
            if pos.direction != direction:
                trade_rec, new_pos = self.pos_manager.flip_position(direction, exec_price, timestamp, size, curr_atr)
                self.capital += self._calculate_net_pnl(trade_rec)
                self.active_risk_levels = self.risk_engine.calculate_levels(exec_price, direction, curr_atr)
                
                self._trigger_webhook({
                    "action": "FLIP",
                    "closed_direction": "LONG" if pos.direction == 1 else "SHORT",
                    "new_direction": "LONG" if direction == 1 else "SHORT",
                    "price": exec_price,
                    "pnl": trade_rec['net_pnl'],
                    "capital": self.capital
                })
        else:
            self.pos_manager.open_position(direction, exec_price, timestamp, size, curr_atr)
            self.active_risk_levels = self.risk_engine.calculate_levels(exec_price, direction, curr_atr)
            
            self._trigger_webhook({
                "action": "OPEN",
                "direction": "LONG" if direction == 1 else "SHORT",
                "price": exec_price,
                "capital": self.capital
            })

    def _calculate_net_pnl(self, trade_rec: dict) -> float:
        comm_entry = self.cost_model.calculate_commission(trade_rec['entry_price'], trade_rec['size'])
        comm_exit = self.cost_model.calculate_commission(trade_rec['exit_price'], trade_rec['size'])
        
        if trade_rec['direction'] == 1:
            gross = (trade_rec['exit_price'] - trade_rec['entry_price']) * trade_rec['size']
        else:
            gross = (trade_rec['entry_price'] - trade_rec['exit_price']) * trade_rec['size']
            
        trade_rec['gross_pnl'] = gross
        trade_rec['net_pnl'] = gross - (comm_entry + comm_exit)
        return trade_rec['net_pnl']

    def _trigger_webhook(self, payload: dict):
        print(f"PAPER TRADE EXECUTION: {payload}")
        if self.webhook_callback:
            self.webhook_callback(payload)
