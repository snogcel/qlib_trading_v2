"""
Hummingbot-compatible backtesting framework for quantile predictions
Uses Hummingbot's signal format and trading logic
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

class HummingbotQuantileBacktester:
    """
    Backtester that simulates Hummingbot AI livestream trading with your quantile predictions
    """
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 trading_pair: str = "BTCUSDT",
                 long_threshold: float = 0.6,
                 short_threshold: float = 0.6,
                 max_position_pct: float = 0.5,
                 fee_rate: float = 0.001,
                 # Position management parameters
                 neutral_close_threshold: float = 0.7,  # Close if neutral prob > this
                 min_confidence_hold: float = 1.0,      # Min confidence to hold position
                 opposing_signal_threshold: float = 0.4, # Close if opposing signal > this
                 # Position sizing method
                 sizing_method: str = "enhanced"):  # 'simple', 'kelly', 'volatility', 'sharpe', 'risk_parity', 'enhanced'
        
        self.initial_balance = initial_balance
        self.trading_pair = trading_pair
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate
        
        # Position management parameters
        self.neutral_close_threshold = neutral_close_threshold
        self.min_confidence_hold = min_confidence_hold
        self.opposing_signal_threshold = opposing_signal_threshold
        
        # Position sizing method
        self.sizing_method = sizing_method
        
        # Trading state
        self.balance = initial_balance
        self.position = 0.0  # Current position in base currency
        self.position_entry_price = 0.0  # Track average entry price for PnL calculation
        self.trades = []
        self.holds = []  # Track hold states
        self.portfolio_history = []
        
    def quantiles_to_probabilities(self, q10: float, q50: float, q90: float,
                                  spread_thresh: float = None) -> Tuple[float, float, float]:
        """
        Convert quantile predictions to [short_prob, neutral_prob, long_prob]
        Based on validated prob_up_piecewise logic with improved spread handling
        
        Args:
            q10, q50, q90: Quantile predictions
            spread_thresh: 90th percentile of historical spreads (validated static threshold)
        """
        # Calculate probability of upside movement (exact match with training script)
        if q90 <= 0:
            prob_up = 0.0
        elif q10 >= 0:
            prob_up = 1.0
        elif q10 < 0 <= q50:
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            prob_up = 1 - cdf0
        else:
            cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
            prob_up = 1 - cdf0
        
        prob_down = 1 - prob_up
        
        # Use spread for neutral probability weighting
        spread = q90 - q10
        
        # Improved spread normalization based on validation results
        if spread_thresh is not None and spread_thresh > 0:
            # Use validated static 90th percentile threshold
            spread_normalized = min(spread / spread_thresh, 1.0)
        else:
            # Dynamic fallback: use recent spread statistics if no threshold provided
            # This is more robust than a fixed 0.02 threshold
            spread_normalized = min(spread / max(spread * 2, 0.01), 1.0)
        
        # Neutral weight: higher spread = more uncertainty = higher neutral probability
        neutral_weight = spread_normalized * 0.3  # max 30% neutral
        
        # Redistribute probabilities (ensure they sum to 1.0)
        prob_up_adj = prob_up * (1 - neutral_weight)
        prob_down_adj = prob_down * (1 - neutral_weight)
        prob_neutral = neutral_weight
        
        # Validation: ensure probabilities sum to 1.0 (within floating point precision)
        total_prob = prob_up_adj + prob_down_adj + prob_neutral
        if abs(total_prob - 1.0) > 1e-10:
            # Normalize if there's a significant deviation
            prob_up_adj /= total_prob
            prob_down_adj /= total_prob
            prob_neutral /= total_prob
        
        return prob_down_adj, prob_neutral, prob_up_adj
    
    def calculate_target_pct(self, q10: float, q50: float, q90: float, abs_q50: float, tier_confidence: float,
                           spread_thresh: float = None, signal_thresh: float = None, 
                           sizing_method: str = "enhanced", target_vol: float = 0.009) -> float:
        """
        Calculate target position percentage using advanced sizing methods
        
        Args:
            sizing_method: 'simple', 'kelly', 'volatility', 'sharpe', 'risk_parity', 'enhanced'
        """
                
        if sizing_method == "simple":
            return self._simple_sizing(q10, q50, q90, abs_q50, tier_confidence, spread_thresh, signal_thresh)
        elif sizing_method == "kelly":
            return self._kelly_sizing(q10, q50, q90, tier_confidence, spread_thresh)
        elif sizing_method == "volatility":
            return self._volatility_sizing(q10, q50, q90, tier_confidence, spread_thresh, target_vol)
        elif sizing_method == "sharpe":
            return self._sharpe_sizing(q10, q50, q90, tier_confidence)
        elif sizing_method == "risk_parity":
            return self._risk_parity_sizing(q10, q50, q90, tier_confidence)
        else:  # enhanced (default)
            return self._enhanced_sizing(q10, q50, q90, abs_q50, tier_confidence, spread_thresh, signal_thresh)
    
    def _simple_sizing(self, q10: float, q50: float, q90: float, abs_q50: float, tier_confidence: float,
                      spread_thresh: float = None, signal_thresh: float = None) -> float:
        """Your current implementation"""
        base_target = abs(q50) * 10
        
        # confidence_multiplier = ((abs_q50 - signal_thresh) / signal_thresh) + 1 if signal_thresh else 1.0
        
        spread = q90 - q10
        if spread_thresh is not None and spread_thresh > 0:
            risk_adjustment = max(1.0, spread / spread_thresh)
        else:
            risk_adjustment = max(1.0, spread / 0.01)
        
        # target_pct = base_target * confidence_multiplier / risk_adjustment
        target_pct = base_target * 1 / risk_adjustment
        return float(np.clip(target_pct, 0.01, self.max_position_pct))
    
    def _kelly_sizing(self, q10: float, q50: float, q90: float, tier_confidence: float,
                     spread_thresh: float = None, signal_thresh: float = None) -> float:
        """Validated Kelly Criterion sizing based on proven predictive features"""
        
        # Calculate spread (validated as predictive of future volatility)
        spread = q90 - q10
        abs_q50 = abs(q50)
        
        # Use validated Kelly approach
        prob_up = self.prob_up_piecewise(q10, q50, q90)
        
        if prob_up > 0.5:  # Long position
            expected_win = q90
            expected_loss = abs(q10) if q10 < 0 else 0.001
            win_prob = prob_up
        else:  # Short position
            expected_win = abs(q10) if q10 < 0 else 0.001
            expected_loss = q90
            win_prob = 1 - prob_up
        
        if expected_loss <= 0:
            return 0.001
        
        # Kelly formula
        payoff_ratio = expected_win / expected_loss
        kelly_pct = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
        
        # Base conservative Kelly with confidence
        confidence_adj = tier_confidence / 10.0
        base_kelly = kelly_pct * 0.25 * confidence_adj
        
        # VALIDATED ADJUSTMENTS based on spread analysis results
        
        # 1. Signal threshold adjustment (T-stat 7.10, highly significant)
        signal_quality_multiplier = 1.0
        if signal_thresh and abs_q50 > signal_thresh:
            signal_quality_multiplier = 1.3  # Boost for validated signals (Sharpe 0.077 vs -0.002)
        else:
            signal_quality_multiplier = 0.8   # Reduce for unvalidated signals
        
        # 2. Spread threshold adjustment (proven better risk-adjusted returns)
        spread_risk_multiplier = 1.0
        if spread_thresh and spread < spread_thresh:
            spread_risk_multiplier = 1.2  # Tight spread: +0.025% return vs -0.12%
        else:
            spread_risk_multiplier = 0.8   # Wide spread: higher volatility
        
        # 3. Combined quality bonus (both conditions met)
        combined_bonus = 1.0
        if (signal_thresh and abs_q50 > signal_thresh and 
            spread_thresh and spread < spread_thresh):
            combined_bonus = 1.15  # Best combination
        
        # Final Kelly calculation
        final_kelly = base_kelly * signal_quality_multiplier * spread_risk_multiplier * combined_bonus
        
        return max(0.001, min(final_kelly, self.max_position_pct))
    
    def _volatility_sizing(self, q10: float, q50: float, q90: float, tier_confidence: float,
                          spread_thresh: float = None, target_vol: float = 0.009) -> float:
        """Volatility-adjusted sizing for consistent risk"""
        signal_strength = abs(q50)
        base_position = signal_strength * 8
        
        # Estimate position volatility from quantile spread
        position_vol = (q90 - q10) / 3.29  # Rough std dev estimate
        
        if position_vol > 0:
            vol_adjustment = target_vol / position_vol
        else:
            vol_adjustment = 1.0
        
        # Confidence adjustment (quadratic scaling)
        confidence_adj = 1 + ((tier_confidence / 3.0) ** 1.5)
        
        # Spread penalty
        spread = q90 - q10
        if spread_thresh and spread_thresh > 0:
            spread_penalty = 1.0 / (1.0 + (spread / spread_thresh) * 2)
        else:
            spread_penalty = 1.0 / (1.0 + spread * 50)
        
        # print(f"position_vol: {position_vol}, target_vol: {target_vol}, confidence_adj: {confidence_adj}")

        position_pct = base_position * vol_adjustment * confidence_adj # * spread_penalty 
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def _sharpe_sizing(self, q10: float, q50: float, q90: float, tier_confidence: float) -> float:
        """Sharpe ratio optimized sizing"""
        expected_return = q50
        expected_vol = (q90 - q10) / 3.29
        
        if expected_vol <= 0:
            return 0.001
        
        expected_sharpe = abs(expected_return) / expected_vol
        base_position = expected_sharpe * 0.05  # Scale factor
        
        # Confidence boost
        confidence_boost = 1 + ((tier_confidence / 3.0) ** 1.5)
        
        position_pct = base_position * confidence_boost
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def _risk_parity_sizing(self, q10: float, q50: float, q90: float, tier_confidence: float,
                           portfolio_vol: float = 0.15) -> float:
        """Risk parity approach"""
        position_vol = (q90 - q10) / 3.29
        
        if position_vol <= 0:
            return 0.001
        
        # Target 2% risk contribution
        target_risk = portfolio_vol * 0.02
        risk_based_size = target_risk / position_vol
        
        # Signal and confidence adjustments
        signal_adj = abs(q50) * 15
        confidence_adj = tier_confidence / 10.0
        
        position_pct = risk_based_size * signal_adj * confidence_adj
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def _enhanced_sizing(self, q10: float, q50: float, q90: float, abs_q50: float, tier_confidence: float,
                        spread_thresh: float = None, signal_thresh: float = None, target_vol: float = 0.009) -> float:
        """Enhanced ensemble approach using validated features"""
        
        # Get individual method results (now using validated Kelly)
        kelly = self._kelly_sizing(q10, q50, q90, tier_confidence, spread_thresh, signal_thresh)
        volatility = self._volatility_sizing(q10, q50, q90, tier_confidence, spread_thresh, target_vol)
        sharpe = self._sharpe_sizing(q10, q50, q90, tier_confidence)
        risk_parity = self._risk_parity_sizing(q10, q50, q90, tier_confidence)
        
        # Reweight based on validation results - Kelly performed well
        weights = {
            'kelly': 0.4,        # Increase Kelly weight (validated approach)
            'volatility': 0.3,   # Keep volatility (also performed well)
            'sharpe': 0.15,      # Reduce Sharpe
            'risk_parity': 0.15  # Reduce risk parity
        }
        
        # Weighted ensemble
        ensemble_size = (
            kelly * weights['kelly'] +
            volatility * weights['volatility'] +
            sharpe * weights['sharpe'] +
            risk_parity * weights['risk_parity']
        )
        
        # VALIDATED ENHANCEMENTS
        
        # 1. Signal quality boost (proven significant with T-stat 7.10)
        signal_quality_multiplier = 1.0
        if signal_thresh and abs_q50 and signal_thresh > 0:
            if abs_q50 > signal_thresh:
                # Above threshold: Sharpe 0.077 vs -0.002 below
                signal_quality_multiplier = 1.4
            else:
                signal_quality_multiplier = 0.7
        
        # 2. Spread-based risk adjustment (validated predictive power)
        spread = q90 - q10
        spread_multiplier = 1.0
        if spread_thresh and spread_thresh > 0:
            if spread < spread_thresh:
                # Tight spread: better risk-adjusted returns
                spread_multiplier = 1.3
            else:
                # Wide spread: reduce size due to higher expected volatility
                spread_multiplier = 0.7
        
        # 3. Confidence scaling (keep existing)
        confidence_scale = (tier_confidence / 10.0) ** 1.1
        
        # 4. Combined validation bonus
        validation_bonus = 1.0
        if (signal_thresh and abs_q50 > signal_thresh and 
            spread_thresh and spread < spread_thresh):
            validation_bonus = 1.2  # Both validated conditions met
        
        # Final calculation
        final_size = (ensemble_size * 
                     signal_quality_multiplier * 
                     spread_multiplier * 
                     confidence_scale * 
                     validation_bonus)
        
        return max(0.001, min(final_size, self.max_position_pct))
    
    def prob_up_piecewise(self, q10: float, q50: float, q90: float) -> float:
        """
        Calculate probability of upside movement using your PPO logic
        Exact implementation from ppo_sweep_optuna_tuned.py
        """
        if q90 <= 0:
            return 0.0
        if q10 >= 0:
            return 1.0
        # 0 lies between q10 and q50
        if q10 < 0 <= q50:
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            return 1 - cdf0
        # 0 lies between q50 and q90
        cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
        return 1 - cdf0
    
    def generate_hummingbot_signal(self, row: pd.Series) -> Dict:
        """
        Generate Hummingbot-compatible signal from quantile predictions
        """
        
        # print(row)
        
        q10, q50, q90 = row['q10'], row['q50'], row['q90']
        side = row.get('side', None)
        # Map signal_tier to numeric confidence 
        # Handle both numeric (0,1,2,3) and letter (A,B,C,D) formats
        signal_tier = row.get('signal_tier', 3)
        if isinstance(signal_tier, str):
            tier_confidence_map = {'A': 10.0, 'B': 7.0, 'C': 5.0, 'D': 3.0}
            tier_confidence = tier_confidence_map.get(signal_tier, 3.0)
        else:
            # Numeric format: 0=best, 3=worst (reverse of letter system)
            numeric_confidence_map = {0: 10.0, 1: 7.0, 2: 5.0, 3: 3.0}
            tier_confidence = numeric_confidence_map.get(signal_tier, 3.0)

        realized_vol_6 = row.get('$realized_vol_6', 0.009)        
        abs_q50 = abs(q50)

        signal_thresh = row.get('signal_thresh_adaptive', row.get('signal_thresh', None))
        spread_thresh = row.get('spread_thresh', None)  # Get adaptive threshold
        
        # Convert to probabilities using adaptive threshold
        short_prob, neutral_prob, long_prob = self.quantiles_to_probabilities(
            q10, q50, q90, spread_thresh
        )
        
        # Calculate target percentage using selected sizing method
        target_pct = self.calculate_target_pct(
            q10, q50, q90, abs_q50 or abs(q50), tier_confidence, 
            spread_thresh, signal_thresh, self.sizing_method, realized_vol_6
        )
        
        # Determine signal direction
        # if long_prob > self.long_threshold:
        #     signal_direction = "LONG"
        #     signal_strength = long_prob
        # elif short_prob > self.short_threshold:
        #     signal_direction = "SHORT"
        #     signal_strength = short_prob
        # else:
        #     signal_direction = "NEUTRAL"
        #     signal_strength = neutral_prob


        # Calculate prob_up using your exact PPO logic
        prob_up = self.prob_up_piecewise(q10, q50, q90)
        
        # Determine signal direction using your PPO logic
        if side is not None:
            # Use provided side column if available (from your PPO preprocessing)
            if side == 1:
                signal_direction = "LONG"
                signal_strength = long_prob
            elif side == 0:
                signal_direction = "SHORT"
                signal_strength = short_prob
            else:  # side == -1 (HOLD)
                signal_direction = "HOLD"  # Use HOLD instead of NEUTRAL to match your logic
                signal_strength = neutral_prob
        else:
            # Recreate your PPO logic if side column not available
            signal_thresh = row.get('signal_thresh_adaptive', row.get('signal_thresh', 0.01))  # Default threshold            

            # Your exact masks from training script
            buy_mask = (abs_q50 > signal_thresh) & (prob_up > 0.5)
            sell_mask = (abs_q50 > signal_thresh) & (prob_up < 0.5)
            
            if buy_mask:
                signal_direction = "LONG"
                signal_strength = prob_up
                side = 1
            elif sell_mask:
                signal_direction = "SHORT"
                signal_strength = 1 - prob_up
                side = 0
            else:
                signal_direction = "HOLD"  # Default to HOLD (matches your -1)
                signal_strength = neutral_prob
                side = -1
        
        return {
            "timestamp": row.name,
            "probabilities": [short_prob, neutral_prob, long_prob],
            "target_pct": target_pct,
            "signal_direction": signal_direction,
            "signal_strength": signal_strength,
            "confidence": tier_confidence,
            "side": side,  # Your PPO side encoding (1=BUY, 0=SELL, -1=HOLD)
            "prob_up": prob_up,  # Your calculated prob_up
            "signal_thresh": signal_thresh,
            "raw_quantiles": {"q10": q10, "q50": q50, "q90": q90},
            "is_position_closure": False  # Regular signals are not position closures
        }
    
    def execute_trade(self, signal: Dict, current_price: float, timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Execute trade based on Hummingbot signal logic
        Returns trade record or None, but also tracks hold states separately
        """
        direction = signal["signal_direction"]
        target_pct = signal["target_pct"]
        
        if direction == "NEUTRAL" or direction == "HOLD":
            # NEUTRAL means "no strong directional bias" - don't automatically close positions
            if abs(self.position) > 0.001:
                # We have a position - should we close it?
                should_close, close_reason = self._should_close_on_neutral(signal, current_price)
                
                if should_close:
                    close_details = {
                        "close_reason": close_reason,
                        "probabilities": signal.get("probabilities", [0, 0, 0]),
                        "confidence": signal.get("confidence", 0),
                        "side": signal.get("side", -1)
                    }
                    return self._close_position(current_price, timestamp, "NEUTRAL_CLOSE", close_details)

                else:
                    # Hold existing position
                    hold_reason = "HOLD_POSITION" if direction == "HOLD" else "NEUTRAL_HOLD_POSITION"
                    self._record_hold_state(current_price, timestamp, hold_reason, signal)
                    return None
            else:
                # No position - just stay out of market
                hold_reason = "HOLD_NO_POSITION" if direction == "HOLD" else "NEUTRAL_NO_POSITION"
                self._record_hold_state(current_price, timestamp, hold_reason, signal)
                return None
        
        # Simple directional trading: LONG = BUY, SHORT = SELL
        portfolio_value = self.balance + self.position * current_price
        position_value = portfolio_value * target_pct
        position_size = position_value / current_price
        
        if direction == "LONG":
            # LONG signal always means BUY (positive position change)
            if self.position < 0:
                # Close short position first, then go long
                position_change = position_size - self.position  # This will be positive (BUY)
            else:
                # Add to long position or create new long
                position_change = position_size  # Always positive (BUY)
        else:  # SHORT
            # SHORT signal always means SELL (negative position change)
            if self.position > 0:
                # Close long position first, then go short
                position_change = -position_size - self.position  # This will be negative (SELL)
            else:
                # Add to short position or create new short
                position_change = -position_size  # Always negative (SELL)
        
        # Only trade if change is significant
        if abs(position_change) < 0.001:
            # Record hold state (signal present but no significant change needed)
            hold_reason = f"{direction}_HOLD" if abs(self.position) > 0.001 else f"{direction}_NO_POSITION"
            self._record_hold_state(current_price, timestamp, hold_reason, signal)
            return None
        
        # Execute the trade
        return self._execute_position_change(position_change, current_price, timestamp, signal)
    
    def _execute_position_change(self, position_change: float, price: float, 
                                timestamp: pd.Timestamp, signal: Dict) -> Dict:
        """
        Execute position change with fees and slippage
        """
        position_before = self.position
        
        # Calculate costs
        notional_value = abs(position_change * price)
        fee_cost = notional_value * self.fee_rate
        
        # Simple slippage model (0.05% for market orders)
        slippage_cost = notional_value * 0.0005
        
        # Calculate trade PnL before executing the trade
        trade_pnl = 0.0
        
        # If we're reducing/closing a position, calculate PnL on the closed portion
        if (position_before > 0 and position_change < 0) or (position_before < 0 and position_change > 0):
            # We're closing or reducing a position
            closed_quantity = min(abs(position_before), abs(position_change))
            if self.position_entry_price > 0:
                if position_before > 0:  # Closing long position
                    trade_pnl = closed_quantity * (price - self.position_entry_price)
                else:  # Closing short position
                    trade_pnl = closed_quantity * (self.position_entry_price - price)
        
        # Update balance and position
        if position_change > 0:  # Buying
            cost = notional_value + fee_cost + slippage_cost
            if cost <= self.balance:
                self.balance -= cost
                self.position += position_change
                side = "BUY"
            else:
                return None  # Insufficient balance
        else:  # Selling
            proceeds = notional_value - fee_cost - slippage_cost
            self.balance += proceeds
            self.position += position_change  # position_change is negative
            side = "SELL"
        
        # Update position entry price (weighted average for additions, reset for full closes)
        if abs(self.position) < 0.001:  # Position fully closed
            self.position_entry_price = 0.0
        elif (position_before >= 0 and position_change > 0) or (position_before <= 0 and position_change < 0):
            # Adding to position - calculate weighted average entry price
            if abs(position_before) < 0.001:  # Opening new position
                self.position_entry_price = price
            else:  # Adding to existing position
                total_cost_before = abs(position_before) * self.position_entry_price
                additional_cost = abs(position_change) * price
                total_quantity = abs(position_before) + abs(position_change)
                self.position_entry_price = (total_cost_before + additional_cost) / total_quantity
        
        # Subtract fees from PnL
        trade_pnl -= (fee_cost + slippage_cost)
        
        # Record trade with enhanced information including PnL
        trade = {
            "timestamp": timestamp,
            "side": side,
            "quantity": abs(position_change),
            "price": price,
            "fee_cost": fee_cost,
            "slippage_cost": slippage_cost,
            "position_before": position_before,
            "position_after": self.position,
            "balance_after": self.balance,
            "pnl": trade_pnl,  # Add PnL to trade record
            "entry_price": self.position_entry_price,
            "signal": signal,
            "trade_type": "POSITION_CLOSURE" if signal.get("is_position_closure", False) else "DIRECTIONAL",
            "original_signal": signal.get("signal_direction", "UNKNOWN")
        }
        
        self.trades.append(trade)
        return trade
    
    def _close_position(self, price: float, timestamp: pd.Timestamp, reason: str, 
                       close_details: Dict = None) -> Optional[Dict]:
        """
        Close current position with detailed reasoning
        """
        if abs(self.position) < 0.001:
            return None
        
        position_change = -self.position
        signal = {
            "signal_direction": reason, 
            "confidence": 0,
            "is_position_closure": True,  # Flag to indicate this is closing a position
            "close_details": close_details or {}  # Additional details about why we closed
        }
        
        return self._execute_position_change(position_change, price, timestamp, signal)
    
    def _record_hold_state(self, price: float, timestamp: pd.Timestamp, hold_reason: str, signal: Dict):
        """
        Record hold state (no trade executed but signal processed)
        """
        hold_record = {
            "timestamp": timestamp,
            "hold_reason": hold_reason,
            "position": self.position,
            "price": price,
            "portfolio_value": self.balance + self.position * price,
            "signal": signal,
            "signal_direction": signal.get("signal_direction", "UNKNOWN"),
            "signal_strength": signal.get("signal_strength", 0),
            "probabilities": signal.get("probabilities", [0, 0, 0])
        }
        
        self.holds.append(hold_record)
    
    def _should_close_on_neutral(self, signal: Dict, current_price: float) -> Tuple[bool, str]:
        """
        Determine if we should close position on neutral signal
        More sophisticated logic than automatic closure
        """
        # Get signal characteristics
        probabilities = signal.get("probabilities", [0.33, 0.34, 0.33])  # [short, neutral, long]
        short_prob, neutral_prob, long_prob = probabilities

        abs_q50 = signal.get("abs_q50", 0.008)
        signal_threshold = signal.get("signal_thresh", 0.013)

        confidence = signal.get("confidence", 5.0)
        
        # Current position direction
        is_long = self.position > 0
        is_short = self.position < 0
        
        # Reasons to close position on neutral:
        
        # 1. Very high neutral probability (high uncertainty)
        # if neutral_prob > self.neutral_close_threshold:
        #     return True, f"high_neutral_prob_{neutral_prob:.3f}"
        
        # 2. Low confidence in predictions
        if confidence < self.min_confidence_hold:
            return True, f"low_confidence_{confidence:.1f}"
        
        # 3. Opposing signal is getting stronger
        # if is_long and short_prob > self.opposing_signal_threshold:
        #     return True, f"opposing_short_signal_{short_prob:.3f}"
        # elif is_short and long_prob > self.opposing_signal_threshold:
        #     return True, f"opposing_long_signal_{long_prob:.3f}"
        
        # 4. Very low confidence in current direction
        #min_directional_prob = (1 - self.neutral_close_threshold) / 2
        #if is_long and long_prob < min_directional_prob:
        #    return True, f"weak_long_prob_{long_prob:.3f}"
        #elif is_short and short_prob < min_directional_prob:
        #    return True, f"weak_short_prob_{short_prob:.3f}"
        
        # 5. PPO-specific: Check if we have strong opposing signal from PPO side
        ppo_side = signal.get('side', None)
        if ppo_side is not None:
            if is_long and ppo_side == 0:  # Long position but PPO says SELL
                return True, f"ppo_opposing_sell_signal"
            elif is_short and ppo_side == 1:  # Short position but PPO says BUY
                return True, f"ppo_opposing_buy_signal"
        
        # 6. Check if we've been in this position too long (optional risk management)
        # This would require tracking position entry time - for now, skip
        
        # Default: Hold the position (don't close on weak HOLD/NEUTRAL signals)
        return False, "hold_position"
    
    def run_backtest(self, df: pd.DataFrame, price_data: pd.DataFrame = None, 
                    price_col: str = 'close', return_col: str = 'truth') -> pd.DataFrame:
        """
        Run Hummingbot-style backtest on prediction data with real price data
        
        Args:
            df: DataFrame with predictions (q10, q50, q90, etc.)
            price_data: DataFrame with actual price data (datetime index, 'close' column)
            price_col: Column name for price in price_data
            return_col: Column name for returns in df (for validation)
        """
        print(f"Starting Hummingbot backtest with {len(df)} observations...")
        print(f"Trading pair: {self.trading_pair}")
        print(f"Long threshold: {self.long_threshold}, Short threshold: {self.short_threshold}")
        
        # Ensure datetime index for predictions
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df = df.set_index('datetime')
            else:
                print("Warning: No datetime index found, using range index")
        
        # Handle price data
        if price_data is not None:
            print(f"Using provided price data with {len(price_data)} price points")
            
            # Ensure datetime index for price data
            if not isinstance(price_data.index, pd.DatetimeIndex):
                if 'datetime' in price_data.columns:
                    price_data = price_data.set_index('datetime')
                else:
                    print("Warning: Price data has no datetime index")
            
            # Align price data with prediction data
            aligned_data = df.join(price_data[[price_col]], how='inner', rsuffix='_price')
            
            if len(aligned_data) == 0:
                print("ERROR: No matching timestamps between predictions and price data")
                return pd.DataFrame()
            
            print(f"Aligned data: {len(aligned_data)} observations")
            print(f"Date range: {aligned_data.index.min()} to {aligned_data.index.max()}")
            
            # Use aligned data
            df = aligned_data
            use_real_prices = True
            
        else:
            print("No price data provided, simulating from returns...")
            use_real_prices = False
        
        # Reset trading state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_entry_price = 0.0
        self.trades = []
        self.holds = []
        self.portfolio_history = []
        
        # Initialize price tracking
        if use_real_prices:
            print(f"Using real prices from column: {price_col}")
        else:
            cumulative_price = 100.0  # Starting price for simulation
            print("Simulating prices from returns")
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Get current price
            if use_real_prices:
                current_price = row[price_col]
                if pd.isna(current_price):
                    print(f"Warning: Missing price at {timestamp}, skipping...")
                    continue
            else:
                # Simulate price from returns (fallback method)
                if i > 0:
                    price_return = row[return_col]
                    if not pd.isna(price_return):
                        cumulative_price *= (1 + price_return)
                current_price = cumulative_price
            
            # Generate Hummingbot signal
            signal = self.generate_hummingbot_signal(row)
            
            # Execute trade if signal warrants it
            trade = self.execute_trade(signal, current_price, timestamp)
            
            # Calculate portfolio value
            portfolio_value = self.balance + self.position * current_price
            
            # Calculate P&L from previous period
            pnl = 0.0
            if i > 0 and len(self.portfolio_history) > 0:
                prev_portfolio = self.portfolio_history[-1]
                prev_price = prev_portfolio['price']
                if use_real_prices and not pd.isna(prev_price):
                    # Calculate actual P&L from price change
                    price_change = (current_price - prev_price) / prev_price                    
                    pnl = self.position * prev_price * price_change                    

                else:
                    # Use return-based P&L (fallback)
                    if return_col in row and not pd.isna(row[return_col]):
                        pnl = self.position * prev_price * row[return_col]
            
            # Determine action taken
            action_taken = "TRADE" if trade is not None else "HOLD"
            hold_reason = None
            
            # Get hold reason if no trade was executed
            if trade is None and len(self.holds) > 0:
                # Check if this timestamp matches the last hold record
                last_hold = self.holds[-1]
                if last_hold["timestamp"] == timestamp:
                    hold_reason = last_hold["hold_reason"]
            
            # Record portfolio state
            self.portfolio_history.append({
                "timestamp": timestamp,
                "price": current_price,
                "balance": self.balance,
                "position": self.position,
                "portfolio_value": portfolio_value,
                "pnl": pnl,
                "signal": signal,
                "action_taken": action_taken,
                "hold_reason": hold_reason,
                "trade_executed": trade is not None,
                "price_source": "real" if use_real_prices else "simulated"
            })
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(df)} observations, Price: ${current_price:.2f}, Portfolio: ${portfolio_value:,.2f}")
        
        # Close final position
        if abs(self.position) > 0.001:
            final_trade = self._close_position(current_price, timestamp, "BACKTEST_END")
        
        final_portfolio_value = self.balance + self.position * current_price
        print(f"Backtest completed. Final portfolio value: ${final_portfolio_value:,.2f}")
        
        return self._create_results_dataframe()
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """
        Create comprehensive results DataFrame
        """
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        if len(portfolio_df) > 0:
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df['cumulative_returns'] = (portfolio_df['portfolio_value'] / self.initial_balance) - 1
            
            # Calculate drawdown
            portfolio_df['peak'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['peak']) / portfolio_df['peak']
        
        return portfolio_df
    
    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if len(results_df) == 0:
            return {}
        
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_balance) - 1
        
        returns = results_df['returns'].dropna()
        
        # Annualized metrics (assuming hourly data)
        periods_per_year = 365 * 24  # Trading hours per year
        
        # Calculate hold statistics
        total_periods = len(results_df)
        hold_periods = len(self.holds)
        trade_periods = len(self.trades)
                
        # Analyze hold reasons
        hold_reasons = {}
        if self.holds:
            for hold in self.holds:
                reason = hold['hold_reason']
                hold_reasons[reason] = hold_reasons.get(reason, 0) + 1
        
        metrics = {
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (periods_per_year / len(results_df)) - 1,
            "volatility": returns.std() * np.sqrt(periods_per_year),
            "sharpe_ratio": (returns.mean() * periods_per_year) / (returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0,
            "max_drawdown": results_df['drawdown'].min(),
            "total_trades": len(self.trades),
            "total_holds": len(self.holds),
            "total_periods": total_periods,
            "trade_frequency": trade_periods / total_periods if total_periods > 0 else 0,
            "hold_frequency": hold_periods / total_periods if total_periods > 0 else 0,
            "hold_reasons": hold_reasons,
            "win_rate": len([t for t in self.trades if t.get('pnl', 0) > 0]) / len(self.trades) if self.trades else 0,
            "final_balance": self.balance,
            "final_position": self.position,
            "final_portfolio_value": final_value
        }
        
        return metrics
    
    def generate_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate comprehensive backtest report
        """
        metrics = self.calculate_metrics(results_df)
        
        report = f"""
=== HUMMINGBOT QUANTILE BACKTEST REPORT ===

CONFIGURATION:
- Trading Pair: {self.trading_pair}
- Initial Balance: ${self.initial_balance:,.2f}
- Long Threshold: {self.long_threshold:.2f}
- Short Threshold: {self.short_threshold:.2f}
- Max Position: {self.max_position_pct:.1%}
- Fee Rate: {self.fee_rate:.3%}

POSITION MANAGEMENT:
- Neutral Close Threshold: {self.neutral_close_threshold:.2f}
- Min Confidence to Hold: {self.min_confidence_hold:.1f}
- Opposing Signal Threshold: {self.opposing_signal_threshold:.2f}

PERFORMANCE SUMMARY:
- Total Return: {metrics.get('total_return', 0):.2%}
- Annualized Return: {metrics.get('annualized_return', 0):.2%}
- Volatility: {metrics.get('volatility', 0):.2%}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}

TRADING STATISTICS:
- Total Trades: {metrics.get('total_trades', 0):,}
- Total Holds: {metrics.get('total_holds', 0):,}
- Trade Frequency: {metrics.get('trade_frequency', 0):.2%}
- Hold Frequency: {metrics.get('hold_frequency', 0):.2%}
- Win Rate: {metrics.get('win_rate', 0):.2%}

HOLD ANALYSIS:
{self._format_hold_reasons(metrics.get('hold_reasons', {}))}

FINAL STATE:
- Final Balance: ${metrics.get('final_balance', 0):,.2f}
- Final Position: {metrics.get('final_position', 0):.4f}
- Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}

OBSERVATIONS: {len(results_df):,}
"""
        
        return report
    
    def _format_hold_reasons(self, hold_reasons: Dict) -> str:
        """Format hold reasons for the report"""
        if not hold_reasons:
            return "- No hold periods recorded"
        
        formatted = []
        for reason, count in sorted(hold_reasons.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"- {reason}: {count:,} periods")
        
        return "\n".join(formatted)
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str = "./hummingbot_backtest_results"):
        """
        Save backtest results in Hummingbot-compatible format
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save portfolio history with proper timestamps and action analysis
        portfolio_save = results_df.copy()
        if 'timestamp' in portfolio_save.columns:
            portfolio_save['timestamp_formatted'] = pd.to_datetime(portfolio_save['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add date/time analysis columns
            timestamp_series = pd.to_datetime(portfolio_save['timestamp'])
            portfolio_save['date'] = timestamp_series.dt.date
            portfolio_save['hour'] = timestamp_series.dt.hour
            portfolio_save['day_of_week'] = timestamp_series.dt.day_name()
        
        portfolio_save.to_csv(output_path / "portfolio_history.csv", index=False)
        
        # Save detailed hold analysis
        if self.holds:
            holds_df = pd.DataFrame(self.holds)
            holds_df['timestamp_formatted'] = pd.to_datetime(holds_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            holds_df.to_csv(output_path / "holds_analysis.csv", index=False)
        
        # Save trades with enhanced information
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df['timestamp_formatted'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add explanatory columns
            trades_df['trade_explanation'] = trades_df.apply(
                lambda row: f"{row['trade_type']}: {row['original_signal']}" if 'trade_type' in row else row['side'], 
                axis=1
            )
            
            # Reorder columns for clarity
            column_order = [
                'timestamp_formatted', 'side', 'trade_type', 'original_signal', 'trade_explanation',
                'quantity', 'price', 'position_before', 'position_after',
                'fee_cost', 'slippage_cost', 'balance_after'
            ]
            
            # Only include columns that exist
            available_columns = [col for col in column_order if col in trades_df.columns]
            remaining_columns = [col for col in trades_df.columns if col not in available_columns]
            final_columns = available_columns + remaining_columns
            
            trades_df = trades_df[final_columns]
            trades_df.to_csv(output_path / "trades.csv", index=False)
        
        # Create pivot-ready signal analysis CSV
        self._create_signal_analysis_csv(results_df, output_path)
        
        # Save metrics
        metrics = self.calculate_metrics(results_df)
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save report
        report = self.generate_report(results_df)
        with open(output_path / "report.txt", 'w') as f:
            f.write(report)
        
        print(f"\nHummingbot backtest results saved to {output_path}")
        print(f"  - portfolio_history.csv (portfolio evolution with hold/trade actions)")
        print(f"  - trades.csv (individual trades)")
        print(f"  - holds_analysis.csv (detailed hold state analysis)")
        print(f"  - signal_analysis_pivot.csv (pivot-ready signal vs trade analysis)")
        print(f"  - metrics.json (performance metrics)")
        print(f"  - report.txt (summary report)")
    
    def _create_signal_analysis_csv(self, results_df: pd.DataFrame, output_path: Path):
        """
        Create comprehensive signal analysis CSV in pivot-ready format
        """
        signal_analysis = []
        
        for _, row in results_df.iterrows():
            signal = row.get('signal', {})
            
            # Extract signal characteristics
            timestamp = row['timestamp']
            price = row['price']
            action_taken = row['action_taken']
            trade_executed = row['trade_executed']
            hold_reason = row.get('hold_reason', 'N/A')
            
            # Signal details
            signal_direction = signal.get('signal_direction', 'UNKNOWN')
            signal_strength = signal.get('signal_strength', 0)
            confidence = signal.get('confidence', 0)
            target_pct = signal.get('target_pct', 0)
            side = signal.get('side', -1)
            prob_up = signal.get('prob_up', 0.5)
            signal_thresh = signal.get('signal_thresh', 0)
            
            # Probabilities
            probabilities = signal.get('probabilities', [0, 0, 0])
            short_prob, neutral_prob, long_prob = probabilities
            
            # Raw quantiles
            raw_quantiles = signal.get('raw_quantiles', {})
            q10 = raw_quantiles.get('q10', 0)
            q50 = raw_quantiles.get('q50', 0)
            q90 = raw_quantiles.get('q90', 0)
            spread = q90 - q10
            abs_q50 = abs(q50)
            
            # Portfolio state
            position = row['position']
            portfolio_value = row['portfolio_value']
            pnl = row.get('pnl', 0)
            
            # Time analysis
            dt = pd.to_datetime(timestamp)
            date = dt.date()
            hour = dt.hour
            day_of_week = dt.day_name()
            month = dt.month
            quarter = dt.quarter
            
            # Signal classification
            signal_above_thresh = abs_q50 > signal_thresh if signal_thresh > 0 else False
            prob_directional = prob_up if prob_up > 0.5 else (1 - prob_up)
            prob_confidence_level = "High" if prob_directional > 0.7 else "Medium" if prob_directional > 0.6 else "Low"
            
            # Trade outcome classification
            trade_outcome = "NO_TRADE"
            if trade_executed:
                trade_outcome = "EXECUTED"
            elif action_taken == "HOLD":
                if hold_reason and "POSITION" in hold_reason:
                    trade_outcome = "HOLD_POSITION"
                else:
                    trade_outcome = "HOLD_NO_POSITION"
            
            # Signal strength buckets
            strength_bucket = "Very_Low" if signal_strength < 0.3 else \
                            "Low" if signal_strength < 0.5 else \
                            "Medium" if signal_strength < 0.7 else \
                            "High" if signal_strength < 0.9 else "Very_High"
            
            # Confidence buckets
            confidence_bucket = "Very_Low" if confidence < 3 else \
                              "Low" if confidence < 5 else \
                              "Medium" if confidence < 7 else \
                              "High" if confidence < 9 else "Very_High"
            
            # Target position buckets
            target_bucket = "Very_Small" if target_pct < 0.05 else \
                          "Small" if target_pct < 0.15 else \
                          "Medium" if target_pct < 0.25 else \
                          "Large" if target_pct < 0.35 else "Very_Large"
            
            # Spread buckets (for analysis)
            spread_bucket = "Tight" if spread < 0.01 else \
                          "Normal" if spread < 0.02 else \
                          "Wide" if spread < 0.04 else "Very_Wide"
            
            signal_analysis.append({
                # Timestamp and identifiers
                'timestamp': timestamp,
                'timestamp_formatted': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'date': date,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'quarter': quarter,
                
                # Market data
                'price': price,
                'portfolio_value': portfolio_value,
                'position': position,
                'pnl': pnl,
                
                # Raw signal data
                'q10': q10,
                'q50': q50,
                'q90': q90,
                'abs_q50': abs_q50,
                'spread': spread,
                'signal_thresh': signal_thresh,
                'prob_up': prob_up,
                
                # Signal characteristics
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'target_pct': target_pct,
                'side': side,
                
                # Probabilities
                'short_prob': short_prob,
                'neutral_prob': neutral_prob,
                'long_prob': long_prob,
                'prob_directional': prob_directional,
                
                # Action and outcome
                'action_taken': action_taken,
                'trade_executed': trade_executed,
                'trade_outcome': trade_outcome,
                'hold_reason': hold_reason,
                
                # Classifications for pivot analysis
                'signal_above_thresh': signal_above_thresh,
                'prob_confidence_level': prob_confidence_level,
                'strength_bucket': strength_bucket,
                'confidence_bucket': confidence_bucket,
                'target_bucket': target_bucket,
                'spread_bucket': spread_bucket,
                
                # Derived metrics
                'signal_quality_score': (signal_strength * confidence * prob_directional) / 100,
                'risk_reward_ratio': abs_q50 / max(spread, 0.001),
                'position_utilization': abs(position) / self.max_position_pct if self.max_position_pct > 0 else 0,
                
                # Boolean flags for easy filtering
                'is_long_signal': signal_direction == 'LONG',
                'is_short_signal': signal_direction == 'SHORT',
                'is_hold_signal': signal_direction == 'HOLD',
                'has_position': abs(position) > 0.001,
                'high_confidence': confidence >= 7,
                'strong_signal': signal_strength >= 0.7,
                'above_threshold': signal_above_thresh,
                'weekend': day_of_week in ['Saturday', 'Sunday'],
                'market_hours': True,  # Crypto markets are 24/7
            })
        
        # Convert to DataFrame and save
        signal_df = pd.DataFrame(signal_analysis)
        signal_df.to_csv(output_path / "signal_analysis_pivot.csv", index=False)
        
        # Create summary statistics for quick analysis
        summary_stats = self._create_signal_summary_stats(signal_df)
        with open(output_path / "signal_summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"  - signal_summary_stats.json (quick signal analysis statistics)")
    
    def _create_signal_summary_stats(self, signal_df: pd.DataFrame) -> Dict:
        """Create summary statistics for signal analysis"""
        
        total_signals = len(signal_df)
        
        stats = {
            "total_observations": total_signals,
            "signal_direction_distribution": signal_df['signal_direction'].value_counts().to_dict(),
            "trade_outcome_distribution": signal_df['trade_outcome'].value_counts().to_dict(),
            "execution_rate_by_direction": {},
            "execution_rate_by_confidence": {},
            "execution_rate_by_strength": {},
            "average_metrics_by_outcome": {},
            "threshold_analysis": {},
            "time_analysis": {}
        }
        
        # Execution rates by signal direction
        for direction in signal_df['signal_direction'].unique():
            direction_data = signal_df[signal_df['signal_direction'] == direction]
            execution_rate = direction_data['trade_executed'].mean()
            stats["execution_rate_by_direction"][direction] = {
                "execution_rate": execution_rate,
                "total_signals": len(direction_data),
                "executed_trades": direction_data['trade_executed'].sum()
            }
        
        # Execution rates by confidence bucket
        for confidence in signal_df['confidence_bucket'].unique():
            conf_data = signal_df[signal_df['confidence_bucket'] == confidence]
            execution_rate = conf_data['trade_executed'].mean()
            stats["execution_rate_by_confidence"][confidence] = {
                "execution_rate": execution_rate,
                "total_signals": len(conf_data),
                "avg_target_pct": conf_data['target_pct'].mean()
            }
        
        # Execution rates by strength bucket
        for strength in signal_df['strength_bucket'].unique():
            strength_data = signal_df[signal_df['strength_bucket'] == strength]
            execution_rate = strength_data['trade_executed'].mean()
            stats["execution_rate_by_strength"][strength] = {
                "execution_rate": execution_rate,
                "total_signals": len(strength_data),
                "avg_confidence": strength_data['confidence'].mean()
            }
        
        # Average metrics by trade outcome
        for outcome in signal_df['trade_outcome'].unique():
            outcome_data = signal_df[signal_df['trade_outcome'] == outcome]
            stats["average_metrics_by_outcome"][outcome] = {
                "count": len(outcome_data),
                "avg_signal_strength": outcome_data['signal_strength'].mean(),
                "avg_confidence": outcome_data['confidence'].mean(),
                "avg_target_pct": outcome_data['target_pct'].mean(),
                "avg_prob_directional": outcome_data['prob_directional'].mean(),
                "avg_signal_quality_score": outcome_data['signal_quality_score'].mean()
            }
        
        # Threshold analysis
        above_thresh = signal_df[signal_df['above_threshold'] == True]
        below_thresh = signal_df[signal_df['above_threshold'] == False]
        
        stats["threshold_analysis"] = {
            "above_threshold": {
                "count": len(above_thresh),
                "execution_rate": above_thresh['trade_executed'].mean() if len(above_thresh) > 0 else 0,
                "avg_signal_strength": above_thresh['signal_strength'].mean() if len(above_thresh) > 0 else 0
            },
            "below_threshold": {
                "count": len(below_thresh),
                "execution_rate": below_thresh['trade_executed'].mean() if len(below_thresh) > 0 else 0,
                "avg_signal_strength": below_thresh['signal_strength'].mean() if len(below_thresh) > 0 else 0
            }
        }
        
        # Time-based analysis
        stats["time_analysis"] = {
            "execution_rate_by_hour": signal_df.groupby('hour')['trade_executed'].mean().to_dict(),
            "execution_rate_by_day": signal_df.groupby('day_of_week')['trade_executed'].mean().to_dict(),
            "weekend_vs_weekday": {
                "weekend_execution_rate": signal_df[signal_df['weekend'] == True]['trade_executed'].mean(),
                "weekday_execution_rate": signal_df[signal_df['weekend'] == False]['trade_executed'].mean()
            }
        }
        
        return stats

# Example usage
def load_price_data(price_file: str = None) -> pd.DataFrame:
    """
    Load 60-minute price data for backtesting
    
    Args:
        price_file: Path to CSV with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']
    
    Returns:
        DataFrame with datetime index and OHLCV data
    """
    if price_file is None:
        print("No price file specified, will simulate from returns")
        return None
    
    try:
        print(f"Loading price data from {price_file}...")
        price_data = pd.read_csv(price_file)
        
        # Convert datetime column
        if 'datetime' in price_data.columns:
            price_data['datetime'] = pd.to_datetime(price_data['datetime'])
            price_data = price_data.set_index('datetime')
        elif 'timestamp' in price_data.columns:
            price_data['datetime'] = pd.to_datetime(price_data['timestamp'])
            price_data = price_data.set_index('datetime')
        
        print(f"Loaded {len(price_data)} price points")
        print(f"Price data range: {price_data.index.min()} to {price_data.index.max()}")
        print(f"Available columns: {list(price_data.columns)}")
        
        return price_data
        
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None

def run_hummingbot_backtest(price_file: str = None):
    """
    Example of running Hummingbot-style backtest with real price data
    
    Args:
        price_file: Path to CSV with 60min price data
    """
    # Load prediction data
    df = pd.read_csv("df_all_macro_analysis.csv")
    
    # Set up datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Load price data
    price_data = load_price_data(price_file)
    
    # Filter for recent data (optional)
    df = df.tail(10000)  # Last 10k observations
    
    # Configure backtester
    backtester = HummingbotQuantileBacktester(
        initial_balance=100000.0,
        trading_pair="BTCUSDT",
        long_threshold=0.6,
        short_threshold=0.6,
        max_position_pct=0.3,
        fee_rate=0.001
    )
    
    # Run backtest with real price data
    results = backtester.run_backtest(df, price_data=price_data, price_col='close', return_col='truth')
    
    # Generate and print report
    print(backtester.generate_report(results))
    
    # Save results
    backtester.save_results(results)
    
    return backtester, results

if __name__ == "__main__":
    backtester, results = run_hummingbot_backtest()