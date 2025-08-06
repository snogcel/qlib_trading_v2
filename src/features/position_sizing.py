"""
Advanced position sizing strategies for quantile-based trading
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats

class AdvancedPositionSizer:
    """
    Collection of sophisticated position sizing methods
    """
    
    def __init__(self, max_position_pct: float = 0.5, lookback_window: int = 252):
        self.max_position_pct = max_position_pct
        self.lookback_window = lookback_window
        self.historical_returns = []
        self.historical_predictions = []
    
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

    def kelly_criterion_sizing(self, q10: float, q50: float, q90: float, 
                              tier_confidence: float, historical_data: pd.DataFrame = None) -> float:
        """
        Kelly Criterion: Optimal position size based on win probability and payoff ratio -- a great example of why validation is good
        
        Kelly % = (bp - q) / b
        where:
        - b = odds (payoff ratio)
        - p = probability of winning
        - q = probability of losing (1-p)
        """
        # Estimate win probability from quantiles
        prob_up = self._prob_up_piecewise(q10, q50, q90)
        prob_down = 1 - prob_up
        
        # Estimate expected payoff ratios
        if prob_up > 0.5:  # Long position
            expected_win = abs(q90)  # Upside potential
            expected_loss = abs(q10)  # Downside risk
            win_prob = prob_up
        else:  # Short position
            expected_win = abs(q10)  # Downside capture
            expected_loss = abs(q90)  # Upside risk
            win_prob = prob_down
        
        if expected_loss == 0:
            return 0.01  # Avoid division by zero
        
        # Kelly formula
        payoff_ratio = expected_win / expected_loss
        kelly_pct = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
        
        # Apply confidence adjustment
        confidence_adj = tier_confidence / 10.0
        kelly_pct *= confidence_adj
        
        # Conservative Kelly (use fraction of full Kelly)
        conservative_kelly = kelly_pct * 0.25  # Use 25% of full Kelly
        
        return max(0.001, min(conservative_kelly, self.max_position_pct))
    
    def volatility_adjusted_sizing(self, q10: float, q50: float, q90: float,
                                  tier_confidence: float, current_volatility: float,
                                  target_volatility: float = 0.15) -> float:
        """
        Volatility-adjusted position sizing to maintain consistent risk
        """
        # Base position from signal strength
        signal_strength = abs(q50)
        base_position = signal_strength * 5  # Scale factor
        
        # Volatility adjustment
        if current_volatility > 0:
            vol_adjustment = target_volatility / current_volatility
        else:
            vol_adjustment = 1.0
        
        # Confidence adjustment
        confidence_adj = (tier_confidence / 10.0) ** 2  # Quadratic scaling
        
        # Spread penalty (wider spread = lower position)
        spread = q90 - q10
        spread_penalty = 1.0 / (1.0 + spread * 50)  # Penalty for wide spreads
        
        position_pct = base_position * vol_adjustment * confidence_adj * spread_penalty
        
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def sharpe_optimized_sizing(self, q10: float, q50: float, q90: float,
                               tier_confidence: float, historical_sharpe: float = None) -> float:
        """
        Position sizing optimized for Sharpe ratio
        """
        # Expected return
        expected_return = q50
        
        # Expected volatility (rough estimate from quantile spread)
        expected_vol = (q90 - q10) / 3.29  # Approximate std dev from quantiles
        
        if expected_vol <= 0:
            return 0.001
        
        # Expected Sharpe ratio
        expected_sharpe = expected_return / expected_vol
        
        # Position size proportional to expected Sharpe
        base_position = abs(expected_sharpe) * 0.1  # Scale factor
        
        # Confidence boost
        confidence_boost = 1 + (tier_confidence - 5) / 10  # 0.6x to 1.5x
        
        # Historical Sharpe adjustment
        if historical_sharpe is not None and historical_sharpe > 0:
            sharpe_adj = min(expected_sharpe / historical_sharpe, 2.0)  # Cap at 2x
        else:
            sharpe_adj = 1.0
        
        position_pct = base_position * confidence_boost * sharpe_adj
        
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def risk_parity_sizing(self, q10: float, q50: float, q90: float,
                          tier_confidence: float, portfolio_volatility: float = 0.15) -> float:
        """
        Risk parity approach - size position based on risk contribution
        """
        # Estimate position volatility
        position_vol = (q90 - q10) / 3.29  # Rough std dev estimate
        
        if position_vol <= 0:
            return 0.001
        
        # Target risk contribution (e.g., 2% of portfolio volatility)
        target_risk_contrib = portfolio_volatility * 0.02
        
        # Position size to achieve target risk
        risk_based_size = target_risk_contrib / position_vol
        
        # Signal strength adjustment
        signal_adj = abs(q50) * 10  # Scale by signal strength
        
        # Confidence adjustment
        confidence_adj = tier_confidence / 10.0
        
        position_pct = risk_based_size * signal_adj * confidence_adj
        
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def adaptive_momentum_sizing(self, q10: float, q50: float, q90: float,
                                tier_confidence: float, recent_performance: float = 0) -> float:
        """
        Adaptive sizing based on recent performance
        """
        # Base position from quantiles
        base_position = abs(q50) * 8
        
        # Confidence scaling
        confidence_scale = (tier_confidence / 10.0) ** 1.5
        
        # Performance-based adjustment
        if recent_performance > 0:
            # Increase size after wins (momentum)
            perf_adj = 1 + min(recent_performance * 2, 0.5)  # Max 50% increase
        else:
            # Decrease size after losses (risk management)
            perf_adj = 1 + max(recent_performance * 1, -0.3)  # Max 30% decrease
        
        # Spread adjustment
        spread = q90 - q10
        spread_adj = 1.0 / (1.0 + spread * 30)
        
        position_pct = base_position * confidence_scale * perf_adj * spread_adj
        
        return max(0.001, min(position_pct, self.max_position_pct))
    
    def ensemble_sizing(self, q10: float, q50: float, q90: float, tier_confidence: float,
                       current_volatility: float = 0.2, recent_performance: float = 0,
                       historical_sharpe: float = 1.0) -> Dict[str, float]:
        """
        Ensemble of multiple sizing methods with weights
        """
        methods = {
            'kelly': self.kelly_criterion_sizing(q10, q50, q90, tier_confidence),
            'volatility': self.volatility_adjusted_sizing(q10, q50, q90, tier_confidence, current_volatility),
            'sharpe': self.sharpe_optimized_sizing(q10, q50, q90, tier_confidence, historical_sharpe),
            'risk_parity': self.risk_parity_sizing(q10, q50, q90, tier_confidence),
            'momentum': self.adaptive_momentum_sizing(q10, q50, q90, tier_confidence, recent_performance)
        }
        
        # Weights for ensemble (can be optimized)
        weights = {
            'kelly': 0.3,
            'volatility': 0.25,
            'sharpe': 0.2,
            'risk_parity': 0.15,
            'momentum': 0.1
        }
        
        # Weighted average
        ensemble_size = sum(methods[method] * weights[method] for method in methods)
        
        methods['ensemble'] = max(0.001, min(ensemble_size, self.max_position_pct))
        
        return methods
    
    def _prob_up_piecewise(self, q10: float, q50: float, q90: float) -> float:
        """Helper function for probability calculation"""
        if q90 <= 0:
            return 0.0
        if q10 >= 0:
            return 1.0
        if q10 < 0 <= q50:
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            return 1 - cdf0
        cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
        return 1 - cdf0

def test_position_sizing_methods():
    """
    Test different position sizing methods
    """
    print("=== POSITION SIZING METHOD COMPARISON ===")
    
    sizer = AdvancedPositionSizer(max_position_pct=0.5)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Strong Bull Signal',
            'q10': 0.005, 'q50': 0.015, 'q90': 0.025,
            'tier_confidence': 8.5, 'volatility': 0.15
        },
        {
            'name': 'Weak Bull Signal',
            'q10': -0.005, 'q50': 0.008, 'q90': 0.020,
            'tier_confidence': 4.2, 'volatility': 0.25
        },
        {
            'name': 'Strong Bear Signal',
            'q10': -0.030, 'q50': -0.018, 'q90': -0.005,
            'tier_confidence': 7.8, 'volatility': 0.18
        },
        {
            'name': 'High Uncertainty',
            'q10': -0.025, 'q50': 0.002, 'q90': 0.030,
            'tier_confidence': 3.1, 'volatility': 0.35
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Quantiles: Q10={scenario['q10']:.3f}, Q50={scenario['q50']:.3f}, Q90={scenario['q90']:.3f}")
        print(f"  Confidence: {scenario['tier_confidence']:.1f}, Volatility: {scenario['volatility']:.2f}")
        
        methods = sizer.ensemble_sizing(
            scenario['q10'], scenario['q50'], scenario['q90'],
            scenario['tier_confidence'], scenario['volatility']
        )
        
        print("  Position Sizes:")
        for method, size in methods.items():
            print(f"    {method:12}: {size:.3f} ({size*100:.1f}%)")

if __name__ == "__main__":
    test_position_sizing_methods()