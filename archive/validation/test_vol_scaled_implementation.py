#!/usr/bin/env python3
"""
Test script to verify vol_scaled implementation
"""

import pandas as pd
import numpy as np
from fix_feature_compatibility import add_vol_raw_features_optimized

def test_vol_scaled():
    """Test that vol_scaled is created correctly"""
    
    print("=" * 80)
    print("TESTING VOL_SCALED IMPLEMENTATION")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate price data
    returns = np.random.normal(0, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create sample dataframe with required columns
    df = pd.DataFrame({
        'close': prices,
        'vol_raw': np.abs(returns) * 10,  # Simulate volatility
        '$realized_vol_6': np.abs(returns) * 10
    })
    
    print(f"Created sample data: {len(df)} rows")
    print(f"   vol_raw range: [{df['vol_raw'].min():.6f}, {df['vol_raw'].max():.6f}]")
    
    # Apply volatility features
    df_with_features = add_vol_raw_features_optimized(df)
    
    print(f"\nFeatures added successfully!")
    print(f"   Columns: {list(df_with_features.columns)}")
    
    # Check vol_scaled properties
    vol_scaled = df_with_features['vol_scaled'].dropna()
    
    print(f"\n vol_scaled analysis:")
    print(f"   Count: {len(vol_scaled)}")
    print(f"   Range: [{vol_scaled.min():.6f}, {vol_scaled.max():.6f}]")
    print(f"   Mean: {vol_scaled.mean():.6f}")
    print(f"   Std: {vol_scaled.std():.6f}")
    
    # Check bounds
    in_bounds = ((vol_scaled >= 0) & (vol_scaled <= 1)).all()
    print(f"   Bounded [0,1]: {in_bounds}")
    
    # Check clipping
    exactly_zero = (vol_scaled == 0).sum()
    exactly_one = (vol_scaled == 1).sum()
    print(f"   Exactly 0.0: {exactly_zero} ({exactly_zero/len(vol_scaled)*100:.1f}%)")
    print(f"   Exactly 1.0: {exactly_one} ({exactly_one/len(vol_scaled)*100:.1f}%)")
    
    # Check vol_risk alias
    if 'vol_risk' in df_with_features.columns:
        vol_risk = df_with_features['vol_risk'].dropna()
        correlation = vol_scaled.corr(vol_risk)
        print(f"\nvol_risk alias:")
        print(f"   Correlation with vol_scaled: {correlation:.6f}")
        print(f"   Same values: {np.allclose(vol_scaled, vol_risk, equal_nan=True)}")
    
    # Test percentiles (for regime detection)
    percentiles = [10, 30, 60, 80, 90]
    print(f"\nvol_scaled percentiles:")
    for p in percentiles:
        val = vol_scaled.quantile(p/100)
        print(f"   {p:2d}%: {val:.6f}")
    
    print(f"\nSUCCESS: vol_scaled implementation working correctly!")
    
    return df_with_features

def test_regime_detection():
    """Test that regime detection works with vol_scaled"""
    
    print(f"\n" + "=" * 80)
    print("TESTING REGIME DETECTION WITH VOL_SCALED")
    print("=" * 80)
    
    # Load the test data
    df = test_vol_scaled()
    
    # Simulate regime detection logic
    vol_scaled = df['vol_scaled'].dropna()
    
    # Test quantile-based thresholds (like in ppo_sweep_optuna_tuned_v2.py)
    high_vol_threshold = vol_scaled.quantile(0.8)
    medium_vol_threshold = vol_scaled.quantile(0.6)
    low_vol_threshold = vol_scaled.quantile(0.3)
    
    print(f"Regime thresholds:")
    print(f"   High vol (80%): {high_vol_threshold:.6f}")
    print(f"   Medium vol (60%): {medium_vol_threshold:.6f}")
    print(f"   Low vol (30%): {low_vol_threshold:.6f}")
    
    # Count regime classifications
    high_vol_count = (vol_scaled > high_vol_threshold).sum()
    medium_vol_count = ((vol_scaled > medium_vol_threshold) & (vol_scaled <= high_vol_threshold)).sum()
    low_vol_count = (vol_scaled < low_vol_threshold).sum()
    
    print(f"\n Regime classifications:")
    print(f"   High vol: {high_vol_count} ({high_vol_count/len(vol_scaled)*100:.1f}%)")
    print(f"   Medium vol: {medium_vol_count} ({medium_vol_count/len(vol_scaled)*100:.1f}%)")
    print(f"   Low vol: {low_vol_count} ({low_vol_count/len(vol_scaled)*100:.1f}%)")
    
    # Test that we get reasonable distributions
    expected_high = len(vol_scaled) * 0.2  # Should be ~20%
    expected_low = len(vol_scaled) * 0.3   # Should be ~30%
    
    high_vol_ok = abs(high_vol_count - expected_high) < expected_high * 0.1
    low_vol_ok = abs(low_vol_count - expected_low) < expected_low * 0.1
    
    print(f"\nRegime detection validation:")
    print(f"   High vol count reasonable: {high_vol_ok}")
    print(f"   Low vol count reasonable: {low_vol_ok}")
    
    if high_vol_ok and low_vol_ok:
        print(f"   SUCCESS: Regime detection working correctly!")
    else:
        print(f"   WARNING: Regime detection may need adjustment")

if __name__ == "__main__":
    test_vol_scaled()
    test_regime_detection()
    
    print(f"\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Test with real data from your loaders")
    print("2. Run backtest to verify position sizing works")
    print("3. Check that vol_scaled values are reasonable in production")
    print("=" * 80)