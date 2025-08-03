#!/usr/bin/env python3
"""
Kelly implementation using VALIDATED features from spread analysis
Now we know spread and thresholds have real predictive power!
"""

import numpy as np
import pandas as pd

def validated_kelly_sizing(q10, q50, q90, spread, signal_thresh, spread_thresh, 
                          fg_index=0.5, vol_scaled=0.3, confidence_tier=3.0, 
                          base_fraction=0.25):
    """
    Kelly sizing using validated predictive features
    
    Based on validation results:
    - Spread predicts future volatility (0.61-0.66 correlation)
    - Signal thresholds meaningfully differentiate performance
    - Spread thresholds identify better risk-adjusted opportunities
    """
    
    # Calculate base Kelly from quantiles
    prob_up = calculate_prob_up_piecewise(q10, q50, q90)
    
    if prob_up > 0.5:  # Long position
        expected_win = q90
        expected_loss = abs(q10)
        win_prob = prob_up
    else:  # Short position
        expected_win = abs(q10)
        expected_loss = q90
        win_prob = 1 - prob_up
    
    if expected_loss <= 0:
        expected_loss = 0.001
    
    # Kelly formula
    payoff_ratio = expected_win / expected_loss
    kelly_fraction = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
    
    # Base conservative Kelly
    abs_q50 = abs(q50)
    confidence_multiplier = confidence_tier / 10.0
    base_kelly = kelly_fraction * base_fraction * confidence_multiplier
    
    # VALIDATED ADJUSTMENTS
    
    # 1. Signal threshold adjustment (proven significant)
    signal_quality_multiplier = 1.0
    if abs_q50 > signal_thresh:
        # Above threshold: Sharpe 0.077 vs -0.002 below
        signal_quality_multiplier = 1.3  # Boost for validated signals
    else:
        signal_quality_multiplier = 0.7   # Reduce for unvalidated signals
    
    # 2. Spread-based risk adjustment (proven predictive)
    spread_risk_multiplier = 1.0
    if spread < spread_thresh:
        # Tight spread: better risk-adjusted returns (0.025% vs -0.12%)
        spread_risk_multiplier = 1.2  # Boost for better risk profile
    else:
        # Wide spread: higher volatility, worse returns
        spread_risk_multiplier = 0.8  # Reduce for higher risk
    
    # 3. Spread decile-based volatility adjustment
    # From validation: spread strongly predicts future volatility
    spread_decile = get_spread_decile(spread)
    
    if spread_decile >= 8:  # Top 20% spread (high future volatility)
        vol_adjustment = 0.7  # Reduce size for expected high volatility
    elif spread_decile >= 6:  # High spread
        vol_adjustment = 0.85
    elif spread_decile <= 2:  # Low spread (low future volatility)
        vol_adjustment = 1.1  # Slightly increase for stable conditions
    else:
        vol_adjustment = 1.0
    
    # 4. Fear & Greed regime adjustment (your validated insight)
    fg_multiplier = 1.0
    if fg_index < 0.2:  # Extreme fear
        if prob_up > 0.5:  # Contrarian bullish
            fg_multiplier = 1.3
    elif fg_index > 0.8:  # Extreme greed
        if prob_up < 0.5:  # Contrarian bearish
            fg_multiplier = 1.3
        elif prob_up > 0.5:  # Bullish during greed
            fg_multiplier = 0.7
    
    # 5. Combined signal quality (both thresholds met)
    if abs_q50 > signal_thresh and spread < spread_thresh:
        # Best combination: strong signal + tight spread
        combined_quality_bonus = 1.2
    else:
        combined_quality_bonus = 1.0
    
    # Final Kelly calculation
    final_kelly = (base_kelly * 
                   signal_quality_multiplier * 
                   spread_risk_multiplier * 
                   vol_adjustment * 
                   fg_multiplier * 
                   combined_quality_bonus)
    
    return max(0.001, min(final_kelly, 0.5))

def get_spread_decile(spread):
    """
    Convert spread to decile based on your data distribution
    You'd need to calculate these thresholds from your actual data
    """
    # Placeholder - replace with actual decile thresholds from your data
    spread_thresholds = [0.005, 0.008, 0.011, 0.014, 0.017, 0.021, 0.026, 0.032, 0.041, 0.055]
    
    for i, threshold in enumerate(spread_thresholds):
        if spread <= threshold:
            return i
    return 9

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

def tiered_kelly_sizing(q10, q50, q90, spread, signal_thresh, spread_thresh,
                       fg_index=0.5, confidence_tier=3.0):
    """
    Simplified tiered approach based on validation results
    """
    abs_q50 = abs(q50)
    
    # Tier classification based on validated thresholds
    if abs_q50 > signal_thresh and spread < spread_thresh:
        tier = "A"  # Best: strong signal + tight spread
        base_size = 0.15
    elif abs_q50 > signal_thresh:
        tier = "B"  # Good: strong signal only
        base_size = 0.10
    elif spread < spread_thresh:
        tier = "C"  # Okay: tight spread only
        base_size = 0.06
    else:
        tier = "D"  # Weak: neither condition met
        base_size = 0.03
    
    # Confidence adjustment
    confidence_multiplier = confidence_tier / 10.0
    
    # Fear & Greed adjustment
    fg_multiplier = 1.0
    prob_up = calculate_prob_up_piecewise(q10, q50, q90)
    
    if fg_index < 0.2 and prob_up > 0.5:  # Contrarian bullish
        fg_multiplier = 1.3
    elif fg_index > 0.8 and prob_up < 0.5:  # Contrarian bearish
        fg_multiplier = 1.3
    elif fg_index > 0.8 and prob_up > 0.5:  # Bullish during greed
        fg_multiplier = 0.7
    
    final_size = base_size * confidence_multiplier * fg_multiplier
    
    return {
        'position_size': max(0.001, min(final_size, 0.3)),
        'tier': tier,
        'base_size': base_size,
        'confidence_multiplier': confidence_multiplier,
        'fg_multiplier': fg_multiplier
    }

def integrate_validated_kelly_with_backtester():
    """
    Integration code for your hummingbot backtester
    """
    
    def _validated_kelly_sizing(self, q10, q50, q90, abs_q50, tier_confidence,
                               spread_thresh=None, signal_thresh=None, 
                               target_vol=0.009):
        """
        Replace your existing Kelly method with validated version
        """
        # Get current row data (you'd need to modify backtester to pass this)
        spread = getattr(self, '_current_spread', (q90 - q10))
        fg_index = getattr(self, '_current_fg_index', 0.5)
        vol_scaled = getattr(self, '_current_vol_scaled', 0.3)
        
        # Use validated Kelly sizing
        return validated_kelly_sizing(
            q10, q50, q90, spread, 
            signal_thresh or 0.01, spread_thresh or 0.02,
            fg_index, vol_scaled, tier_confidence
        )
    
    return _validated_kelly_sizing

def test_validated_kelly():
    """Test the validated Kelly methods"""
    
    test_cases = [
        # Strong signal + tight spread (Tier A)
        {'q10': -0.01, 'q50': 0.008, 'q90': 0.02, 'spread': 0.01, 
         'signal_thresh': 0.005, 'spread_thresh': 0.015, 'fg_index': 0.5},
        
        # Strong signal + wide spread (Tier B)  
        {'q10': -0.02, 'q50': 0.008, 'q90': 0.03, 'spread': 0.025,
         'signal_thresh': 0.005, 'spread_thresh': 0.015, 'fg_index': 0.1},
        
        # Weak signal + tight spread (Tier C)
        {'q10': -0.005, 'q50': 0.002, 'q90': 0.008, 'spread': 0.008,
         'signal_thresh': 0.005, 'spread_thresh': 0.015, 'fg_index': 0.9},
        
        # Weak signal + wide spread (Tier D)
        {'q10': -0.008, 'q50': 0.001, 'q90': 0.012, 'spread': 0.020,
         'signal_thresh': 0.005, 'spread_thresh': 0.015, 'fg_index': 0.5},
    ]
    
    print("Testing Validated Kelly Methods")
    print("=" * 50)
    
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}:")
        print(f"  q50: {case['q50']:.3f}, spread: {case['spread']:.3f}")
        print(f"  signal_thresh: {case['signal_thresh']:.3f}, spread_thresh: {case['spread_thresh']:.3f}")
        print(f"  fg_index: {case['fg_index']:.1f}")
        
        validated_size = validated_kelly_sizing(**case)
        tiered_result = tiered_kelly_sizing(**case)
        
        print(f"  Validated Kelly: {validated_size:.4f}")
        print(f"  Tiered approach: {tiered_result['position_size']:.4f} (Tier {tiered_result['tier']})")

if __name__ == "__main__":
    test_validated_kelly()