#!/usr/bin/env python3
"""
Regime-aware Kelly methods using binary flag interactions
Building on validated Kelly success with sophisticated regime detection
"""

import numpy as np
import pandas as pd

def regime_aware_kelly_sizing(q10, q50, q90, regime_flags, base_kelly_fraction=0.25):
    """
    Kelly sizing with sophisticated regime awareness using binary flags
    
    Args:
        q10, q50, q90: Quantile predictions
        regime_flags: Dict of binary regime flags
        base_kelly_fraction: Conservative Kelly fraction
    """
    
    # Base Kelly calculation (validated approach)
    prob_up = calculate_prob_up_piecewise(q10, q50, q90)
    
    if prob_up > 0.5:
        expected_win = q90
        expected_loss = abs(q10) if q10 < 0 else 0.001
        win_prob = prob_up
    else:
        expected_win = abs(q10) if q10 < 0 else 0.001
        expected_loss = q90
        win_prob = 1 - prob_up
    
    if expected_loss <= 0:
        return 0.001
    
    payoff_ratio = expected_win / expected_loss
    kelly_fraction = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
    base_kelly = kelly_fraction * base_kelly_fraction
    
    # === REGIME-BASED MULTIPLIERS ===
    
    regime_multiplier = 1.0
    
    # 1. CRISIS/OPPORTUNITY DETECTION
    if regime_flags.get('crisis_signal', 0):
        # Bear market bottom - strong contrarian opportunity
        if prob_up > 0.6:
            regime_multiplier *= 2.0  # Double down on contrarian signals
        else:
            regime_multiplier *= 0.5  # Reduce bearish bets at bottom
    
    elif regime_flags.get('euphoria_warning', 0):
        # Altcoin bubble warning - be very cautious
        if prob_up > 0.6:
            regime_multiplier *= 0.3  # Fade bullish signals in euphoria
        else:
            regime_multiplier *= 1.5  # Boost bearish signals
    
    elif regime_flags.get('flight_to_quality', 0):
        # Money flowing to BTC - boost BTC-positive signals
        regime_multiplier *= 1.3
    
    # 2. VOLATILITY REGIME ADJUSTMENTS
    if regime_flags.get('vol_extreme_high', 0):
        if regime_flags.get('strong_signal_extreme_vol', 0):
            # Strong signal in extreme volatility - rare and valuable
            regime_multiplier *= 1.8
        else:
            # Weak signal in extreme volatility - reduce exposure
            regime_multiplier *= 0.6
    
    elif regime_flags.get('vol_extreme_low', 0):
        # Low volatility - slightly reduce all positions
        regime_multiplier *= 0.9
    
    # 3. SENTIMENT REGIME ADJUSTMENTS
    if regime_flags.get('contrarian_fear_bullish', 0):
        # Bullish signal during extreme fear - classic contrarian
        regime_multiplier *= 1.8
    
    elif regime_flags.get('contrarian_greed_bearish', 0):
        # Bearish signal during extreme greed - classic contrarian
        regime_multiplier *= 1.8
    
    elif regime_flags.get('calm_fear', 0):
        # Quiet despair - often precedes major moves
        if prob_up > 0.5:
            regime_multiplier *= 1.4  # Boost bullish signals
    
    elif regime_flags.get('calm_greed', 0):
        # Complacency - dangerous, reduce exposure
        regime_multiplier *= 0.7
    
    # 4. DOMINANCE CYCLE ADJUSTMENTS
    if regime_flags.get('bear_bottom_signal', 0):
        # Classic bear market bottom pattern
        if prob_up > 0.5:
            regime_multiplier *= 2.2  # Strong contrarian opportunity
    
    elif regime_flags.get('alt_euphoria', 0):
        # Altcoin euphoria - very dangerous
        if prob_up > 0.5:
            regime_multiplier *= 0.2  # Heavily fade bullish signals
        else:
            regime_multiplier *= 1.6  # Boost bearish signals
    
    # 5. REGIME TRANSITION BONUSES
    if regime_flags.get('crisis_to_recovery', 0):
        # Transitioning out of crisis - strong opportunity
        regime_multiplier *= 1.5
    
    elif regime_flags.get('bubble_to_crash', 0):
        # Bubble bursting - boost bearish signals
        if prob_up < 0.4:
            regime_multiplier *= 1.8
    
    # 6. SIGNAL QUALITY ADJUSTMENTS
    if regime_flags.get('tight_spread_high_vol', 0):
        # Rare combination - tight spread in high volatility
        regime_multiplier *= 1.6
    
    # Apply regime multiplier with safety bounds
    final_kelly = base_kelly * regime_multiplier
    
    return max(0.001, min(final_kelly, 0.6))  # Cap at 60% for extreme opportunities

def tiered_regime_kelly(q10, q50, q90, regime_flags, confidence_tier=3.0):
    """
    Simplified tiered approach using regime classifications
    """
    
    # Determine regime tier based on flags
    if regime_flags.get('crisis_signal', 0) or regime_flags.get('bear_bottom_signal', 0):
        regime_tier = "CRISIS_OPPORTUNITY"
        base_size = 0.25  # Large positions for crisis opportunities
        
    elif regime_flags.get('euphoria_warning', 0) or regime_flags.get('alt_euphoria', 0):
        regime_tier = "EUPHORIA_WARNING"
        base_size = 0.05  # Very small positions in euphoria
        
    elif (regime_flags.get('contrarian_fear_bullish', 0) or 
          regime_flags.get('contrarian_greed_bearish', 0)):
        regime_tier = "CONTRARIAN_SIGNAL"
        base_size = 0.18  # Good-sized contrarian positions
        
    elif regime_flags.get('vol_extreme_high', 0):
        regime_tier = "HIGH_VOLATILITY"
        base_size = 0.08  # Smaller positions in high vol
        
    elif regime_flags.get('vol_extreme_low', 0):
        regime_tier = "LOW_VOLATILITY"
        base_size = 0.12  # Moderate positions in low vol
        
    else:
        regime_tier = "NORMAL_MARKET"
        base_size = 0.10  # Standard positions
    
    # Confidence adjustment
    confidence_multiplier = confidence_tier / 10.0
    
    # Signal direction adjustment
    prob_up = calculate_prob_up_piecewise(q10, q50, q90)
    
    # Direction-regime interaction
    direction_multiplier = 1.0
    if regime_tier == "CRISIS_OPPORTUNITY" and prob_up > 0.6:
        direction_multiplier = 1.5  # Boost bullish signals in crisis
    elif regime_tier == "EUPHORIA_WARNING" and prob_up < 0.4:
        direction_multiplier = 1.3  # Boost bearish signals in euphoria
    
    final_size = base_size * confidence_multiplier * direction_multiplier
    
    return {
        'position_size': max(0.001, min(final_size, 0.4)),
        'regime_tier': regime_tier,
        'base_size': base_size,
        'confidence_multiplier': confidence_multiplier,
        'direction_multiplier': direction_multiplier
    }

def adaptive_regime_kelly(q10, q50, q90, regime_flags, historical_performance=None):
    """
    Kelly sizing that adapts based on regime performance history
    """
    
    base_kelly = regime_aware_kelly_sizing(q10, q50, q90, regime_flags)
    
    if historical_performance is None:
        return base_kelly
    
    # Get current regime
    current_regime = get_current_regime(regime_flags)
    
    # Adjust based on historical performance in this regime
    regime_performance = historical_performance.get(current_regime, {})
    
    performance_multiplier = 1.0
    
    # Recent Sharpe ratio in this regime
    regime_sharpe = regime_performance.get('sharpe_ratio', 1.0)
    if regime_sharpe > 1.5:
        performance_multiplier *= 1.2  # Boost if performing well in this regime
    elif regime_sharpe < 0.5:
        performance_multiplier *= 0.7  # Reduce if performing poorly
    
    # Win rate in this regime
    regime_win_rate = regime_performance.get('win_rate', 0.5)
    if regime_win_rate > 0.6:
        performance_multiplier *= 1.1
    elif regime_win_rate < 0.4:
        performance_multiplier *= 0.8
    
    return base_kelly * performance_multiplier

def get_current_regime(regime_flags):
    """
    Determine current market regime from flags
    """
    if regime_flags.get('crisis_signal', 0):
        return 'crisis'
    elif regime_flags.get('euphoria_warning', 0):
        return 'euphoria'
    elif regime_flags.get('vol_extreme_high', 0):
        return 'high_volatility'
    elif regime_flags.get('vol_extreme_low', 0):
        return 'low_volatility'
    elif regime_flags.get('fg_extreme_fear', 0):
        return 'extreme_fear'
    elif regime_flags.get('fg_extreme_greed', 0):
        return 'extreme_greed'
    else:
        return 'normal'

def calculate_prob_up_piecewise(q10, q50, q90):
    """Your existing probability calculation"""
    if q90 <= 0:
        return 0.0
    if q10 >= 0:
        return 1.0
    if q10 < 0 <= q50:
        cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
        return 1 - cdf0
    cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
    return 1 - cdf0

def create_regime_kelly_backtester_integration():
    """
    Integration code for your hummingbot backtester
    """
    
    def _regime_aware_kelly_sizing(self, q10, q50, q90, abs_q50, tier_confidence,
                                  spread_thresh=None, signal_thresh=None, 
                                  target_vol=0.009):
        """
        Enhanced Kelly method using regime flags for your backtester
        """
        
        # Extract regime flags from current row (you'd need to modify backtester)
        regime_flags = {
            'vol_extreme_high': getattr(self, '_current_vol_extreme_high', 0),
            'vol_extreme_low': getattr(self, '_current_vol_extreme_low', 0),
            'fg_extreme_fear': getattr(self, '_current_fg_extreme_fear', 0),
            'fg_extreme_greed': getattr(self, '_current_fg_extreme_greed', 0),
            'btc_dom_high': getattr(self, '_current_btc_dom_high', 0),
            'btc_dom_low': getattr(self, '_current_btc_dom_low', 0),
            'crisis_signal': getattr(self, '_current_crisis_signal', 0),
            'euphoria_warning': getattr(self, '_current_euphoria_warning', 0),
            'contrarian_fear_bullish': getattr(self, '_current_contrarian_fear_bullish', 0),
            'contrarian_greed_bearish': getattr(self, '_current_contrarian_greed_bearish', 0),
        }
        
        return regime_aware_kelly_sizing(q10, q50, q90, regime_flags)
    
    return _regime_aware_kelly_sizing

def test_regime_kelly_methods():
    """Test regime-aware Kelly methods"""
    
    test_cases = [
        # Crisis opportunity
        {
            'q10': -0.02, 'q50': 0.01, 'q90': 0.03,
            'regime_flags': {
                'crisis_signal': 1, 'vol_extreme_high': 1, 
                'fg_extreme_fear': 1, 'btc_dom_high': 1
            },
            'description': 'Crisis opportunity (bear bottom)'
        },
        
        # Euphoria warning
        {
            'q10': -0.01, 'q50': 0.005, 'q90': 0.02,
            'regime_flags': {
                'euphoria_warning': 1, 'vol_high': 1,
                'fg_extreme_greed': 1, 'btc_dom_low': 1
            },
            'description': 'Euphoria warning (bubble territory)'
        },
        
        # Contrarian signal
        {
            'q10': -0.015, 'q50': 0.008, 'q90': 0.025,
            'regime_flags': {
                'contrarian_fear_bullish': 1, 'fg_extreme_fear': 1
            },
            'description': 'Contrarian bullish during fear'
        },
        
        # Normal market
        {
            'q10': -0.008, 'q50': 0.003, 'q90': 0.012,
            'regime_flags': {},
            'description': 'Normal market conditions'
        }
    ]
    
    print("Testing Regime-Aware Kelly Methods")
    print("=" * 60)
    
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}: {case['description']}")
        print(f"  Quantiles: q10={case['q10']:.3f}, q50={case['q50']:.3f}, q90={case['q90']:.3f}")
        print(f"  Active flags: {[k for k, v in case['regime_flags'].items() if v]}")
        
        # Test different methods
        regime_kelly = regime_aware_kelly_sizing(
            case['q10'], case['q50'], case['q90'], case['regime_flags']
        )
        
        tiered_result = tiered_regime_kelly(
            case['q10'], case['q50'], case['q90'], case['regime_flags']
        )
        
        print(f"  Regime Kelly: {regime_kelly:.4f}")
        print(f"  Tiered Kelly: {tiered_result['position_size']:.4f} ({tiered_result['regime_tier']})")

if __name__ == "__main__":
    test_regime_kelly_methods()