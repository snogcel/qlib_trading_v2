#!/usr/bin/env python3
"""
Corrected Analysis of Volatility Features with Historical Context
"""

import pandas as pd
import numpy as np

def create_corrected_analysis():
    """Create corrected analysis with historical context"""
    
    print("=" * 100)
    print("VOLATILITY FEATURES: CORRECTED ANALYSIS WITH HISTORICAL CONTEXT")
    print("=" * 100)
    
    print("\n HISTORICAL IMPLEMENTATIONS (OLD):")
    
    print("\n1. vol_risk (formerly vol_scaled):")
    print("   DEFINITION: Quantile-normalized realized volatility")
    print("   FORMULA: ((vol - q_low.shift(1)) / (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)")
    print("   WHERE:")
    print("     • vol = $realized_vol_3 (3-period realized volatility)")
    print("     • q_low = 30-period rolling 1st percentile")
    print("     • q_high = 30-period rolling 99th percentile")
    print("   RANGE: 0.0 to 1.0 (bounded)")
    print("   PURPOSE: Risk management and position sizing")
    
    print("\n2. vol_raw_momentum (formerly vol_momentum):")
    print("   DEFINITION: 3-period percentage change of vol_signal")
    print("   FORMULA: df['vol_signal'].pct_change(periods=3)")
    print("   WHERE: vol_signal = $realized_vol_3")
    print("   RANGE: Unbounded percentage changes")
    print("   PURPOSE: Volatility trend detection")
    
    print("\n3. vol_raw_decile:")
    print("   DEFINITION: Decile ranking of vol_signal")
    print("   FORMULA: vol_signal.apply(get_vol_raw_decile)")
    print("   WHERE: vol_signal = $realized_vol_3")
    print("   RANGE: 0 to 9 (discrete deciles)")
    print("   PURPOSE: Volatility regime classification")
    
    print("\n🆕 NEW IMPLEMENTATIONS:")
    
    print("\n1. vol_risk (NEW):")
    print("   DEFINITION: Volatility squared (variance)")
    print("   FORMULA: Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)")
    print("   RANGE: 0 to ~0.016 (unbounded but small)")
    print("   PURPOSE: Variance-based risk measure")
    
    print("\n2. vol_raw_momentum (NEW):")
    print("   DEFINITION: 1-period difference of 6-period realized volatility")
    print("   FORMULA: Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)")
    print("   RANGE: ~±0.07 (small bounded changes)")
    print("   PURPOSE: Short-term volatility momentum")
    
    print("\n3. vol_raw_decile (NEW):")
    print("   DEFINITION: 180-period rolling rank of 6-period volatility, scaled to 0-10")
    print("   FORMULA: Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10")
    print("   RANGE: 0 to ~0.056 (continuous, not discrete)")
    print("   PURPOSE: Long-term volatility percentile ranking")
    
    print("\n  KEY DIFFERENCES:")
    
    print("\n1. CONCEPTUAL CHANGES:")
    print("   • vol_risk: From quantile-normalized (0-1) to variance (0-0.016)")
    print("   • vol_raw_momentum: From 3-period % change to 1-period absolute difference")
    print("   • vol_raw_decile: From discrete deciles (0-9) to continuous percentiles (0-0.056)")
    
    print("\n2. SCALE CHANGES:")
    print("   • vol_risk: 1,405x scale reduction (0-1 → 0-0.016)")
    print("   • vol_raw_momentum: 2,215x scale reduction (±500 → ±0.07)")
    print("   • vol_raw_decile: 175x scale reduction (0-9 → 0-0.056)")
    
    print("\n3. MATHEMATICAL CHANGES:")
    print("   • Base volatility: $realized_vol_3 → Std(Log($close/Ref($close,1)), 6)")
    print("   • Time horizon: 3-period → 6-period for base, 30-period → 180-period for ranking")
    print("   • Normalization: Quantile-based → Rank-based")
    
    print("\nIMPACT ON USAGE:")
    
    print("\n1. POSITION SIZING (vol_risk):")
    print("   OLD: vol_risk ∈ [0, 1] - perfect for direct position scaling")
    print("   NEW: vol_risk ∈ [0, 0.016] - needs rescaling for position sizing")
    print("   IMPACT: Position sizes will be ~1,400x smaller unless rescaled!")
    
    print("\n2. REGIME DETECTION (vol_risk quantiles):")
    print("   OLD: Quantiles of [0, 1] range - e.g., 0.8 quantile ≈ 0.8")
    print("   NEW: Quantiles of [0, 0.016] range - e.g., 0.8 quantile ≈ 0.013")
    print("   IMPACT: Hardcoded thresholds (0.8, 0.6, 0.3) will be completely wrong!")
    
    print("\n3. VOLATILITY MOMENTUM:")
    print("   OLD: Large percentage changes (±500) - high sensitivity")
    print("   NEW: Small absolute changes (±0.07) - low sensitivity")
    print("   IMPACT: Momentum signals will be much weaker")
    
    print("\n4. DECILE CLASSIFICATION:")
    print("   OLD: Discrete buckets 0-9 - easy thresholding")
    print("   NEW: Continuous 0-0.056 - needs new thresholds")
    print("   IMPACT: Regime flags (>=8, >=6, <=2, <=1) won't work!")
    
    print("\n CRITICAL FIXES NEEDED:")
    
    print("\n1. IMMEDIATE (BREAKING CHANGES):")
    print("   • Position sizing: Rescale vol_risk or use different logic")
    print("   • Regime detection: Recalculate quantile thresholds")
    print("   • Decile thresholds: Update hardcoded values (>=8, >=6, <=2, <=1)")
    
    print("\n2. CODE LOCATIONS TO FIX:")
    
    fixes = [
        {
            'file': 'ppo_sweep_optuna_tuned_v2.py',
            'line': 309,
            'old': "df['vol_risk'] > df['vol_risk'].quantile(0.8)",
            'issue': "Quantile 0.8 of new vol_risk ≈ 0.013, not meaningful threshold",
            'fix': "Recalculate thresholds or use different logic"
        },
        {
            'file': 'ppo_sweep_optuna_tuned_v2.py', 
            'line': 460,
            'old': 'vol_risk = row.get("vol_risk", 0.3)',
            'issue': "Default 0.3 is 18x larger than max possible new vol_risk",
            'fix': "Update default to ~0.01 or rescale vol_risk"
        },
        {
            'file': 'fix_feature_compatibility.py',
            'line': 33,
            'old': "df['vol_extreme_high'] = (df['vol_raw_decile'] >= 8).astype(int)",
            'issue': "New vol_raw_decile max is 0.056, never >= 8",
            'fix': "Use percentile thresholds like >= 0.045"
        }
    ]
    
    for fix in fixes:
        print(f"\n   📝 {fix['file']} (line {fix['line']}):")
        print(f"      OLD: {fix['old']}")
        print(f"      ISSUE: {fix['issue']}")
        print(f"      FIX: {fix['fix']}")
    
    print("\nRECOMMENDED SOLUTION:")
    
    print("\n1. HYBRID APPROACH:")
    print("   • Keep new implementations for their theoretical benefits")
    print("   • Add scaling/normalization to match old behavior where needed")
    print("   • Create compatibility functions for critical usage")
    
    print("\n2. SPECIFIC RECOMMENDATIONS:")
    print("   • vol_risk: Create vol_risk_scaled = vol_risk / vol_risk.quantile(0.99)")
    print("   • vol_raw_decile: Create discrete buckets from continuous values")
    print("   • vol_raw_momentum: Consider normalizing by volatility level")
    
    print("\n3. TESTING PRIORITY:")
    print("   • Position sizing logic (CRITICAL - affects money)")
    print("   • Regime detection (HIGH - affects strategy)")
    print("   • Feature importance (MEDIUM - affects model)")
    
    print("\nDATA VALIDATION:")
    
    # Load and validate the data
    try:
        df = pd.read_csv('Research/v2_model/feature_QA_1.csv')
        df.columns = ['vol_raw_momentum_old', 'vol_raw_momentum_new', 
                      'vol_risk_old', 'vol_risk_new', 
                      'vol_raw_decile_old', 'vol_raw_decile_new']
        
        print(f"\n   ✓ Data loaded: {df.shape[0]:,} rows")
        print(f"   ✓ vol_risk_old range: [{df['vol_risk_old'].min():.3f}, {df['vol_risk_old'].max():.3f}]")
        print(f"   ✓ vol_risk_new range: [{df['vol_risk_new'].min():.6f}, {df['vol_risk_new'].max():.6f}]")
        print(f"   ✓ Scale ratio confirmed: {df['vol_risk_old'].std() / df['vol_risk_new'].std():.0f}x")
        
        # Check if old vol_risk was indeed bounded 0-1
        old_in_range = ((df['vol_risk_old'] >= 0) & (df['vol_risk_old'] <= 1)).all()
        print(f"   ✓ Old vol_risk bounded [0,1]: {old_in_range}")
        
    except Exception as e:
        print(f"   ✗ Data validation failed: {e}")

if __name__ == "__main__":
    create_corrected_analysis()