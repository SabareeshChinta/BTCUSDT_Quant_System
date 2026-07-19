"""
Event-driven backtesting engine.
Anti-look-ahead bias guarantees:
- Signal generates when brick completes.
- Trade execution happens on the NEXT candle's open after signal time.
- Fills use level prices, not candle extremes.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple

from config.settings import SystemConfig
from indicators.renko_engine import RenkoEngine
from indicators.moving_averages import compute_sma, compute_ema, get_trend_state
from strategy.signal_generator import SignalGenerator
from strategy.renko_theses import RenkoATRConsecutiveThesis
from strategy.trade_validator import TradeValidator
from strategy.position_manager import PositionManager, PositionState
from risk.risk_engine import RiskEngine
from risk.cost_model import CostModel
from risk.position_sizer import PositionSizer
from backtesting.trade_logger import TradeLogger
from backtesting.metrics_calculator import MetricsCalculator

@dataclass
class BacktestResult:
    trades_df: pd.DataFrame
    metrics: dict
    equity_curve: pd.Series
    config_name: str
    timeframe: str
    date_range: tuple
    renko_bricks: pd.DataFrame
    total_signals: int
    filtered_signals: int

    def summary(self) -> str:
        s = f"--- Backtest Summary ({self.config_name}) ---\n"
        s += f"Timeframe: {self.timeframe} | Dates: {self.date_range[0]} to {self.date_range[1]}\n"
        s += f"Trades: {len(self.trades_df)} | Signals: {self.filtered_signals}/{self.total_signals}\n"
        m = self.metrics
        s += f"Profit Factor: {m.get('Profit Factor', 0):.2f}\n"
        s += f"Win Rate: {m.get('Win Rate', 0):.1%}\n"
        s += f"Expectancy: {m.get('Expectancy', 0):.2f}\n"
        s += f"Max Drawdown: {m.get('Max Drawdown (%)', 0):.2%}\n"
        s += f"Sharpe Ratio: {m.get('Sharpe Ratio', 0):.2f}\n"
        return s

class BacktestEngine:
    def __init__(self, config: SystemConfig, use_risk_management: bool = True, use_ml_filter: bool = False, ml_predictor = None):
        self.config = config
        self.use_risk_management = use_risk_management
        self.use_ml_filter = use_ml_filter
        self.ml_predictor = ml_predictor

        self.renko = RenkoEngine(
            atr_period=config.strategy.atr_period,
            atr_multiplier=config.strategy.atr_multiplier,
            atr_smoothing='sma'
        )
        self.signal_gen = SignalGenerator(consecutive_bricks=config.strategy.consecutive_bricks)
        self.validator = TradeValidator(
            use_ma_filter=True,
            use_ml_filter=use_ml_filter,
            fast_ma_period=config.strategy.fast_ma_period,
            slow_ma_period=config.strategy.slow_ma_period,
            ma_type=config.strategy.ma_type,
            ml_predictor=ml_predictor,
            ml_confidence_threshold=config.ml.confidence_threshold
        )
        self.pos_manager = PositionManager()
        self.risk_engine = RiskEngine(
            sl_atr_mult=config.risk.stop_loss_atr_mult,
            tp_atr_mult=config.risk.take_profit_atr_mult,
            tsl_atr_mult=config.risk.trailing_stop_atr_mult,
            enabled=use_risk_management
        )
        self.cost_model = CostModel(
            commission_pct=config.risk.commission_pct,
            slippage_pct=config.risk.slippage_pct
        )
        self.sizer = PositionSizer(
            method="risk_based" if use_risk_management else "fixed_fraction", 
            fraction=config.risk.risk_per_trade_pct if use_risk_management else 1.0
        )
        self.logger = TradeLogger()
        self.metrics_calc = MetricsCalculator(initial_capital=config.backtest.initial_capital)

    def run(self, df: pd.DataFrame, timeframe: str = '1h', config_name: str = "Base", precomputed_signals=None, precomputed_features=None, use_atr_filter: bool = False, use_trend_filter: bool = False, use_min_rr_filter: bool = False) -> BacktestResult:
        if df.empty:
            raise ValueError("Empty dataframe provided to BacktestEngine")
            
        if 'Open Time' not in df.columns:
            df = df.reset_index(names='Open Time')
            
        if df['Open Time'].dt.tz is not None:
            df['Open Time'] = df['Open Time'].dt.tz_localize(None)

        # 1. Build Renko & Generate Signals
        signals_df = None
        if precomputed_signals is not None:
            signals, total_signals, df = precomputed_signals
        else:
            bricks = self.renko.build_bricks(df)
            if bricks.empty:
                return self._empty_result(df, config_name, timeframe)
            bricks['consecutive_count'] = self.renko.get_consecutive_count(bricks)
            
            if getattr(self.config.strategy, 'strategy_type', '') == "DoubleBrick":
                thesis = RenkoATRConsecutiveThesis(
                    fast_period=self.config.strategy.fast_ma_period,
                    slow_period=self.config.strategy.slow_ma_period
                )
                signals_df = thesis.generate_signals(bricks)
                signals_df['brick_size'] = bricks['brick_size']
                signals_df['brick_close'] = bricks['brick_close']
            else:
                signals_df = self.signal_gen.generate_signals(bricks)
                
            signals = signals_df[signals_df['signal_changed'] == True]
            total_signals = len(signals)
            
        if len(signals) == 0:
            return self._empty_result(df, config_name, timeframe)

        filtered_signals = 0

        # 3. Compute MAs
        if self.config.strategy.ma_type.upper() == 'EMA':
            fast_ma = compute_ema(df['Close'], self.config.strategy.fast_ma_period)
            slow_ma = compute_ema(df['Close'], self.config.strategy.slow_ma_period)
        else:
            fast_ma = compute_sma(df['Close'], self.config.strategy.fast_ma_period)
            slow_ma = compute_sma(df['Close'], self.config.strategy.slow_ma_period)
            
        trend_state = get_trend_state(fast_ma, slow_ma)
        df_ma = pd.DataFrame({'fast_ma': fast_ma, 'slow_ma': slow_ma, 'trend_state': trend_state}, index=df.index)

        # Precompute ML features if needed
        df_ml_features = precomputed_features
        if df_ml_features is None and self.use_ml_filter and self.ml_predictor:
            # Reconstruct Open Time if it's the index
            if 'Open Time' not in df.columns:
                df_feat_input = df.reset_index()
            else:
                df_feat_input = df.copy()
            df_ml_features = self.ml_predictor.feature_engineer.compute_features(df_feat_input)
            df_ml_features.set_index('Open Time', inplace=True)

        # Prep for iteration
        df = df.sort_values('Open Time').reset_index(drop=True)
        times = df['Open Time'].values
        opens = df['Open'].values
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        # We need ATR for position sizing and risk limits
        from indicators.atr_engine import compute_atr
        atr_series = compute_atr(df, period=self.config.strategy.atr_period).values
        
        # Precompute arrays for new filters
        ema_200 = compute_ema(df['Close'], 200).values
        atr_sma_20 = pd.Series(atr_series).rolling(20).mean().values
        
        # 90-day rolling ATR Percentile (roughly 2160 hours for 1h, scale appropriately if needed, but we'll use fixed 2160 bars)
        atr_pct_series = pd.Series(atr_series).rolling(2160, min_periods=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1]).values

        capital = self.config.backtest.initial_capital
        high_water_mark = capital
        halt_trading = False
        
        current_candle_idx = 0
        n_candles = len(df)
        
        active_risk_levels = None
        mae = 0.0
        mfe = 0.0
        
        signal_idx = 0
        n_signals = len(signals)

        while current_candle_idx < n_candles:
            if halt_trading:
                break
                
            time_now = pd.Timestamp(times[current_candle_idx])
            high_now = highs[current_candle_idx]
            low_now = lows[current_candle_idx]
            close_now = closes[current_candle_idx]
            open_now = opens[current_candle_idx]
            
            # --- Manage existing position ---
            pos = self.pos_manager.current_position
            if pos is not None:
                # Update MAE/MFE
                if pos.direction == 1:
                    unrealized_low = low_now - pos.entry_price
                    unrealized_high = high_now - pos.entry_price
                else:
                    unrealized_low = pos.entry_price - high_now
                    unrealized_high = pos.entry_price - low_now
                    
                mae = min(mae, unrealized_low)
                mfe = max(mfe, unrealized_high)

                # Check account equity protections
                unrealized_pnl = pos.size * (close_now - pos.entry_price) * pos.direction
                current_equity = capital + unrealized_pnl
                
                # Bankruptcy prevention
                if current_equity <= 0:
                    print(f"[{time_now}] BANKRUPTCY: Equity hit zero. Halting.")
                    trade_rec = self.pos_manager.close_position(close_now, time_now, 'bankruptcy')
                    self._log_and_update_capital(trade_rec, mae, mfe)
                    capital += trade_rec['net_pnl']
                    halt_trading = True
                    break
                    
                high_water_mark = max(high_water_mark, current_equity)
                drawdown_pct = (high_water_mark - current_equity) / high_water_mark
                
                if drawdown_pct >= self.config.risk.hard_drawdown_halt_pct:
                    print(f"[{time_now}] HARD HALT: Drawdown reached {drawdown_pct:.2%}. Halting trading.")
                    trade_rec = self.pos_manager.close_position(close_now, time_now, 'max_drawdown_halt')
                    self._log_and_update_capital(trade_rec, mae, mfe)
                    capital += trade_rec['net_pnl']
                    halt_trading = True
                    break
                elif drawdown_pct >= self.config.risk.soft_drawdown_warning_pct:
                    # Soft warning: print once per day or similar, but for now we'll just continue
                    pass

                # Reference Strategy Exits
                ref_exit_triggered = False
                if self.config.strategy.use_reference_strategy and pos.direction == 1:
                    # 1. Price drops below EMA21
                    curr_fast = fast_ma.iloc[current_candle_idx]
                    if close_now < curr_fast:
                        exit_price_with_slip = self.cost_model.apply_exit_slippage(close_now, pos.direction)
                        trade_rec = self.pos_manager.close_position(exit_price_with_slip, time_now, 'price_below_ema')
                        self._log_and_update_capital(trade_rec, mae, mfe)
                        capital += trade_rec['net_pnl']
                        active_risk_levels = None
                        mae = 0.0
                        mfe = 0.0
                        ref_exit_triggered = True
                    
                    # 2. Scale out on consecutive bricks (partial exit)
                    elif not pos.context.get("partial_exit_done", False):
                        brick_size = pos.context.get("brick_size", 0.0)
                        if brick_size > 0:
                            # 5 consecutive green bricks means price moved entry + 4 * brick_size
                            target_price = pos.entry_price + (self.config.strategy.profit_take_bricks - 1) * brick_size
                            if high_now >= target_price:
                                scale_out_price = target_price # assume filled at target
                                trade_rec = self.pos_manager.scale_out_position(0.5, scale_out_price, time_now, 'partial_profit')
                                self._log_and_update_capital(trade_rec, mae, mfe)
                                capital += trade_rec['net_pnl']
                                pos.context["partial_exit_done"] = True

                # Check risk exits
                if not ref_exit_triggered and self.use_risk_management and active_risk_levels:
                    should_exit, reason, exit_price = self.risk_engine.check_exit(
                        pos, open_now, high_now, low_now, close_now, time_now, active_risk_levels
                    )
                    
                    if should_exit:
                        exit_price_with_slip = self.cost_model.apply_exit_slippage(exit_price, pos.direction)
                        trade_rec = self.pos_manager.close_position(exit_price_with_slip, time_now, reason)
                        self._log_and_update_capital(trade_rec, mae, mfe)
                        capital += trade_rec['net_pnl']
                        active_risk_levels = None
                        mae = 0.0
                        mfe = 0.0
                    else:
                        # Update trailing stop
                        active_risk_levels['trailing_stop'] = self.risk_engine.update_trailing_stop(
                            pos.direction, high_now, low_now, 
                            active_risk_levels.get('trailing_stop', active_risk_levels['stop_loss']), 
                            active_risk_levels['trailing_stop_distance']
                        )

            # --- Check for new signals ---
            while signal_idx < n_signals:
                sig_row = signals.iloc[signal_idx]
                sig_time = sig_row['brick_close_time']
                
                # We execute on the NEXT candle after the signal completes
                if sig_time < time_now:
                    # Valid to process
                    raw_sig = sig_row['raw_signal']
                    direction = 1 if raw_sig == 'BUY' else -1
                    
                    # Validate
                    # Find MA state AT or BEFORE signal_time
                    ma_idx_mask = df['Open Time'] <= sig_time
                    if not ma_idx_mask.any():
                        signal_idx += 1
                        continue
                        
                    ma_idx = df[ma_idx_mask].index[-1]
                    sig_trend = df_ma['trend_state'].iloc[ma_idx]
                    
                    # Manual MA validation
                    ma_passed = False
                    if getattr(self.config.strategy, 'strategy_type', '') == "DoubleBrick":
                        ma_passed = True # Brick-based MA already validated by thesis
                    elif self.config.strategy.use_reference_strategy:
                        if direction == 1:
                            curr_fast = fast_ma.iloc[ma_idx]
                            curr_slow = slow_ma.iloc[ma_idx]
                            prev_fast = fast_ma.iloc[ma_idx-1] if ma_idx > 0 else curr_fast
                            prev_slow = slow_ma.iloc[ma_idx-1] if ma_idx > 0 else curr_slow
                            
                            above_both = sig_row['brick_close'] > curr_fast and sig_row['brick_close'] > curr_slow
                            golden_cross = prev_fast <= prev_slow and curr_fast > curr_slow
                            if above_both and golden_cross:
                                ma_passed = True
                    else:
                        if (direction == 1 and sig_trend == 1) or (direction == -1 and sig_trend == -1):
                            ma_passed = True
                            
                    # Compute execution variables first (needed for Mentor Features)
                    exec_price = self.cost_model.apply_entry_slippage(open_now, direction)
                    prev_idx = max(0, current_candle_idx - 1)
                    curr_atr = atr_series[prev_idx]
                    if pd.isna(curr_atr) or curr_atr <= 0:
                        curr_atr = 1.0
                    stop_distance = curr_atr * self.config.risk.stop_loss_atr_mult
                        
                    ml_passed = True
                    trade_context = {"brick_size": sig_row['brick_size'], "partial_exit_done": False}
                    if df_ml_features is not None:
                        # Compute Mentor Context
                        if self.ml_predictor:
                            fe = self.ml_predictor.feature_engineer
                        else:
                            from ml.feature_engineering import FeatureEngineer
                            fe = FeatureEngineer()
                        
                        trade_context.update(fe.compute_trade_context(
                            df_features=df_ml_features,
                            signal_time=sig_time,
                            direction=direction,
                            entry_price=exec_price,
                            sl_distance=stop_distance,
                            target_distance=sig_row['brick_size'],
                            brick_size=sig_row['brick_size']
                        ))
                    
                    if self.use_ml_filter and self.ml_predictor:
                        _, ml_passed = self.ml_predictor.predict(trade_context)
                        
                    if ma_passed and ml_passed:
                        
                        passed_new_filters = True
                        if use_atr_filter:
                            if curr_atr < atr_sma_20[prev_idx]:
                                passed_new_filters = False
                                
                        if use_trend_filter:
                            curr_ema = ema_200[prev_idx]
                            if direction == 1 and open_now < curr_ema:
                                passed_new_filters = False
                            elif direction == -1 and open_now > curr_ema:
                                passed_new_filters = False
                                
                        if use_min_rr_filter:
                            brick_close = sig_row['brick_close']
                            brick_size = sig_row['brick_size']
                            if direction == 1:
                                target = brick_close + brick_size
                                reward = target - exec_price
                            else:
                                target = brick_close - brick_size
                                reward = exec_price - target
                                
                            if reward <= 0 or (reward / stop_distance) < 1.5:
                                passed_new_filters = False
                                
                        if getattr(self.config.strategy, 'atr_percentile_threshold', 0.0) > 0.0:
                            curr_atr_pct = atr_pct_series[prev_idx]
                            if pd.notna(curr_atr_pct) and curr_atr_pct < self.config.strategy.atr_percentile_threshold:
                                passed_new_filters = False
                                
                        if passed_new_filters:
                            filtered_signals += 1
                            dollar_risk = capital * self.config.risk.risk_per_trade_pct
                            
                            # Size position based on risk
                            size = self.sizer.calculate_size(capital, exec_price, stop_distance)
                            
                            pos = self.pos_manager.current_position
                            if pos is not None:
                                if pos.direction != direction:
                                    # Reverse
                                    trade_rec, new_pos = self.pos_manager.flip_position(
                                        direction, exec_price, time_now, size, curr_atr, 
                                        sig_time=sig_time, dollar_risk=dollar_risk, stop_distance=stop_distance,
                                        context=trade_context
                                    )
                                    self._log_and_update_capital(trade_rec, mae, mfe)
                                    capital += trade_rec['net_pnl']
                                    mae = 0.0
                                    mfe = 0.0
                                    if self.use_risk_management:
                                        active_risk_levels = self.risk_engine.calculate_levels(exec_price, direction, curr_atr)
                            else:
                                # Open new
                                self.pos_manager.open_position(
                                    direction, exec_price, time_now, size, curr_atr, 
                                    sig_time=sig_time, dollar_risk=dollar_risk, stop_distance=stop_distance,
                                    context=trade_context
                                )
                                mae = 0.0
                                mfe = 0.0
                                if self.use_risk_management:
                                    active_risk_levels = self.risk_engine.calculate_levels(exec_price, direction, curr_atr)

                    signal_idx += 1
                else:
                    break
                    
            current_candle_idx += 1

        # End of data - close open positions
        pos = self.pos_manager.current_position
        if pos is not None:
            last_close = closes[-1]
            exit_price = self.cost_model.apply_exit_slippage(last_close, pos.direction)
            last_time = pd.Timestamp(times[-1])
            trade_rec = self.pos_manager.close_position(exit_price, last_time, 'end_of_data')
            self._log_and_update_capital(trade_rec, mae, mfe)

        trades_df = self.logger.get_trades_df()
        metrics = self.metrics_calc.calculate_all_metrics(trades_df)
        equity_curve = self.metrics_calc.build_equity_curve(trades_df, self.config.backtest.initial_capital)

        date_range = (str(df['Open Time'].min()), str(df['Open Time'].max()))

        return BacktestResult(
            trades_df=trades_df,
            metrics=metrics,
            equity_curve=equity_curve,
            config_name=config_name,
            timeframe=timeframe,
            date_range=date_range,
            renko_bricks=signals_df,
            total_signals=total_signals,
            filtered_signals=filtered_signals
        )

    def _log_and_update_capital(self, trade_rec: dict, mae: float, mfe: float):
        trade_rec['mae'] = mae
        trade_rec['mfe'] = mfe
        
        comm_entry = self.cost_model.calculate_commission(trade_rec['entry_price'], trade_rec['size'])
        comm_exit = self.cost_model.calculate_commission(trade_rec['exit_price'], trade_rec['size'])
        trade_rec['commission_paid'] = comm_entry + comm_exit
        
        # Calculate true net pnl
        if trade_rec['direction'] == 1:
            gross = (trade_rec['exit_price'] - trade_rec['entry_price']) * trade_rec['size']
        else:
            gross = (trade_rec['entry_price'] - trade_rec['exit_price']) * trade_rec['size']
            
        trade_rec['gross_pnl'] = gross
        trade_rec['net_pnl'] = gross - trade_rec['commission_paid']
        
        trade_rec['pnl_pct'] = trade_rec['net_pnl'] / (trade_rec['entry_price'] * trade_rec['size'])
        trade_rec['trade_id'] = getattr(self, '_trade_counter', 0)
        self._trade_counter = trade_rec['trade_id'] + 1
        trade_rec['slippage_cost'] = 0.0 # Approximate, already baked into entry/exit price
        
        if trade_rec.get('dollar_risk', 0) > 0:
            trade_rec['r_multiple'] = trade_rec['net_pnl'] / trade_rec['dollar_risk']
        else:
            trade_rec['r_multiple'] = 0.0
        
        self.logger.log_trade(trade_rec)

    def _empty_result(self, df: pd.DataFrame, config_name: str, timeframe: str) -> BacktestResult:
        date_range = (str(df['Open Time'].min()), str(df['Open Time'].max())) if not df.empty else ("None", "None")
        return BacktestResult(
            trades_df=pd.DataFrame(),
            metrics={},
            equity_curve=pd.Series(),
            config_name=config_name,
            timeframe=timeframe,
            date_range=date_range,
            renko_bricks=pd.DataFrame(),
            total_signals=0,
            filtered_signals=0
        )
