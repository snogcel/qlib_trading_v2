#!/usr/bin/env python3
"""
Analysis of regime features usage: Standalone vs Multiplicative (Mask) approach
"""

import pandas as pd
import numpy as np

def analyze_regime_feature_approaches():
    """Analyze standalone vs multiplicative approaches for regime features"""
    
    print("=" * 100)
    print("REGIME FEATURES: STANDALONE vs MULTIPLICATIVE (MASK) ANALYSIS")
    print("=" * 100)
    
    print("\n🎯 THE FUNDAMENTAL QUESTION:")
    print("   Should regime features be:")
    print("   A) Standalone binary features (0/1)")
    print("   B) Multiplicative masks applied to other variables")
    
    print("\n📊 ANALYSIS OF TOP PERFORMING FEATURES:")
    
    top_features = [
        {
            "name": "btc_dom_high_streak",
            "lift": 3.01,
            "frequency": "51482.0%",
            "description": "Extended BTC dominance periods",
            "current_type": "Standalone (streak count)",
            "mask_potential": "HIGH - Could amplify signals during BTC dominance"
        },
        {
            "name": "btc_flight", 
            "lift": 2.69,
            "frequency": "0.6%",
            "description": "Flight to BTC safety during volatility",
            "current_type": "Standalone binary (0/1)",
            "mask_potential": "VERY HIGH - Perfect for amplifying signals during flight-to-safety"
        },
        {
            "name": "crisis_signal",
            "lift": 2.19, 
            "frequency": "0.3%",
            "description": "Bear market bottom signal",
            "current_type": "Standalone binary (0/1)",
            "mask_potential": "EXTREME - Should dramatically amplify contrarian signals"
        },
        {
            "name": "greed_vol_spike",
            "lift": 2.01,
            "frequency": "1.9%", 
            "description": "FOMO buying during volatility",
            "current_type": "Standalone binary (0/1)",
            "mask_potential": "HIGH - Could dampen/reverse signals during FOMO periods"
        },
        {
            "name": "fear_vol_spike",
            "lift": 1.98,
            "frequency": "3.1%",
            "description": "Panic selling during volatility", 
            "current_type": "Standalone binary (0/1)",
            "mask_potential": "HIGH - Could amplify contrarian signals during panic"
        }
    ]
    
    for feature in top_features:
        print(f"\n🔍 {feature['name']} (Lift: {feature['lift']}x):")
        print(f"   Current: {feature['current_type']}")
        print(f"   Mask Potential: {feature['mask_potential']}")
        print(f"   Description: {feature['description']}")
    
    print("\n⚖️  STANDALONE vs MULTIPLICATIVE COMPARISON:")
    
    comparison_aspects = [
        {
            "aspect": "Interpretability",
            "standalone": "HIGH - Clear binary signal (crisis: yes/no)",
            "multiplicative": "MEDIUM - Modifies existing signals",
            "winner": "STANDALONE"
        },
        {
            "aspect": "Signal Strength", 
            "standalone": "MEDIUM - Fixed 0/1 contribution",
            "multiplicative": "HIGH - Can amplify strong signals dramatically",
            "winner": "MULTIPLICATIVE"
        },
        {
            "aspect": "Model Complexity",
            "standalone": "LOW - Simple binary features",
            "multiplicative": "MEDIUM - Creates interaction terms",
            "winner": "STANDALONE"
        },
        {
            "aspect": "Overfitting Risk",
            "standalone": "LOW - Simple binary flags",
            "multiplicative": "HIGHER - Complex interactions",
            "winner": "STANDALONE"
        },
        {
            "aspect": "Contextual Adaptation",
            "standalone": "LOW - Same weight regardless of signal strength",
            "multiplicative": "HIGH - Adapts based on underlying signal",
            "winner": "MULTIPLICATIVE"
        },
        {
            "aspect": "Feature Engineering",
            "standalone": "SIMPLE - Direct usage",
            "multiplicative": "COMPLEX - Need to choose what to multiply",
            "winner": "STANDALONE"
        }
    ]
    
    print("\n   ┌─────────────────────┬─────────────────────────────────┬─────────────────────────────────┬─────────────┐")
    print("   │ Aspect              │ Standalone Approach             │ Multiplicative Approach        │ Winner      │")
    print("   ├─────────────────────┼─────────────────────────────────┼─────────────────────────────────┼─────────────┤")
    
    for comp in comparison_aspects:
        print(f"   │ {comp['aspect']:<19} │ {comp['standalone']:<31} │ {comp['multiplicative']:<31} │ {comp['winner']:<11} │")
    
    print("   └─────────────────────┴─────────────────────────────────┴─────────────────────────────────┴─────────────┘")
    
    print("\n💡 SPECIFIC USE CASE ANALYSIS:")
    
    use_cases = [
        {
            "feature": "crisis_signal",
            "standalone_usage": "if crisis_signal == 1: increase_position_size()",
            "multiplicative_usage": "position_size = base_size * (1 + crisis_signal * 2.0)",
            "recommendation": "MULTIPLICATIVE - Crisis should amplify contrarian signals",
            "reason": "During crisis, strong contrarian signals should get much larger positions"
        },
        {
            "feature": "btc_flight", 
            "standalone_usage": "if btc_flight == 1: trade_btc_only()",
            "multiplicative_usage": "btc_signal_strength = base_signal * (1 + btc_flight * 1.5)",
            "recommendation": "MULTIPLICATIVE - Flight should amplify BTC signals",
            "reason": "During flight-to-safety, BTC signals become more reliable"
        },
        {
            "feature": "greed_vol_spike",
            "standalone_usage": "if greed_vol_spike == 1: reduce_risk()",
            "multiplicative_usage": "signal_strength = base_signal * (1 - greed_vol_spike * 0.5)",
            "recommendation": "MULTIPLICATIVE - FOMO should dampen signals",
            "reason": "During FOMO, signals become less reliable, should reduce confidence"
        },
        {
            "feature": "fear_vol_spike",
            "standalone_usage": "if fear_vol_spike == 1: look_for_contrarian()",
            "multiplicative_usage": "contrarian_boost = base_signal * (1 + fear_vol_spike * 1.0)",
            "recommendation": "MULTIPLICATIVE - Panic should amplify contrarian signals",
            "reason": "During panic, contrarian signals become more valuable"
        }
    ]
    
    for case in use_cases:
        print(f"\n📈 {case['feature'].upper()}:")
        print(f"   Standalone: {case['standalone_usage']}")
        print(f"   Multiplicative: {case['multiplicative_usage']}")
        print(f"   Recommendation: {case['recommendation']}")
        print(f"   Reason: {case['reason']}")
    
    print("\n🎯 HYBRID APPROACH RECOMMENDATION:")
    
    hybrid_strategy = """
    BEST APPROACH: Create BOTH standalone AND multiplicative versions
    
    1. STANDALONE FEATURES (for model training):
       • Keep current binary flags for ML models
       • Use for feature importance analysis
       • Simple interpretability for debugging
    
    2. MULTIPLICATIVE FEATURES (for signal enhancement):
       • crisis_multiplier = 1 + crisis_signal * 2.0
       • btc_flight_multiplier = 1 + btc_flight * 1.5  
       • fear_panic_multiplier = 1 + fear_vol_spike * 1.0
       • greed_damper = 1 - greed_vol_spike * 0.3
    
    3. ENHANCED SIGNALS:
       • enhanced_q50 = abs_q50 * crisis_multiplier * btc_flight_multiplier
       • regime_adjusted_signal = base_signal * fear_panic_multiplier * greed_damper
       • contextual_position_size = base_size * regime_multipliers
    """
    
    print(hybrid_strategy)
    
    print("\n🔧 IMPLEMENTATION EXAMPLES:")
    
    implementation_code = '''
def create_regime_multipliers(df):
    """Create multiplicative regime features"""
    
    multipliers = {}
    
    # Crisis amplification (2-3x during crisis)
    multipliers['crisis_amplifier'] = 1 + df['crisis_signal'] * 2.0
    
    # BTC flight amplification (1.5x during flight-to-safety)
    multipliers['btc_flight_amplifier'] = 1 + df['btc_flight'] * 1.5
    
    # Fear panic amplification (contrarian boost)
    multipliers['fear_contrarian_boost'] = 1 + df['fear_vol_spike'] * 1.0
    
    # Greed dampening (reduce confidence during FOMO)
    multipliers['greed_damper'] = 1 - df['greed_vol_spike'] * 0.3
    
    # Combined regime adjustment
    multipliers['regime_combined'] = (
        multipliers['crisis_amplifier'] * 
        multipliers['btc_flight_amplifier'] * 
        multipliers['fear_contrarian_boost'] * 
        multipliers['greed_damper']
    )
    
    return multipliers

def apply_regime_adjustments(df):
    """Apply regime adjustments to signals"""
    
    # Get multipliers
    multipliers = create_regime_multipliers(df)
    
    # Enhanced signals
    df['regime_adjusted_q50'] = df['abs_q50'] * multipliers['regime_combined']
    df['regime_adjusted_spread'] = df['spread'] * multipliers['crisis_amplifier']
    df['regime_position_size'] = df['kelly_position_size'] * multipliers['regime_combined']
    
    return df
    '''
    
    print(implementation_code)
    
    print("\n📊 EXPECTED BENEFITS:")
    
    benefits = [
        "Crisis periods: 2-3x signal amplification during bear market bottoms",
        "BTC flight: 1.5x amplification of BTC-related signals during safety periods", 
        "Fear spikes: Contrarian signal boost during panic selling",
        "Greed spikes: Signal dampening during FOMO to reduce false signals",
        "Combined effect: Contextual signal strength based on market regime",
        "Maintains interpretability while adding sophisticated regime awareness"
    ]
    
    for benefit in benefits:
        print(f"   • {benefit}")
    
    print("\n🚨 RISKS TO CONSIDER:")
    
    risks = [
        "Overfitting: Complex interactions may not generalize",
        "Multiplier calibration: Need to tune amplification factors",
        "Regime misclassification: Wrong regime = wrong signal adjustment",
        "Model complexity: Harder to debug and understand",
        "Backtest overfitting: May work in-sample but fail out-of-sample"
    ]
    
    for risk in risks:
        print(f"   ⚠️  {risk}")
    
    print("\n🎯 FINAL RECOMMENDATION:")
    
    final_rec = """
    IMPLEMENT HYBRID APPROACH:
    
    1. KEEP standalone features for ML model training
    2. CREATE multiplicative versions for signal enhancement  
    3. TEST both approaches in backtesting
    4. USE multiplicative for position sizing and risk management
    5. MONITOR for overfitting and regime misclassification
    
    START WITH: crisis_signal and btc_flight multiplicative versions
    (highest lift ratios = lowest risk, highest reward)
    """
    
    print(final_rec)
    
    print("\n" + "=" * 100)
    print("CONCLUSION: Multiplicative approach likely superior for regime features")
    print("But implement hybrid approach to get benefits of both methods")
    print("=" * 100)

if __name__ == "__main__":
    analyze_regime_feature_approaches()