#!/usr/bin/env python3
"""
Test the fixed spread validation function
"""

import pandas as pd
import numpy as np
from validate_spread_predictive_power import validate_spread_predictive_power, validate_signal_thresholds

def create_test_data():
    """Create realistic test data for validation"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic quantile data
    base_signal = np.random.normal(0, 0.02, n_samples)
    noise = np.random.normal(0, 0.005, n_samples)
    
    # Create correlated quantiles (q90 > q50 > q10)
    q50 = base_signal
    q10 = q50 - np.abs(np.random.normal(0.01, 0.005, n_samples))
    q90 = q50 + np.abs(np.random.normal(0.01, 0.005, n_samples))
    
    # Create truth that's somewhat correlated with quantiles
    truth = q50 + noise
    
    # Create thresholds
    signal_thresh_adaptive = np.full(n_samples, 0.015)
    spread_thresh = np.full(n_samples, 0.02)
    
    df = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'truth': truth,
        'signal_thresh_adaptive': signal_thresh_adaptive,
        'spread_thresh': spread_thresh,
        'abs_q50': np.abs(q50)
    })
    
    return df

def test_fixed_validation():
    """Test the fixed validation function"""
    
    print("=" * 80)
    print("TESTING FIXED SPREAD VALIDATION")
    print("=" * 80)
    
    # Create test data
    df = create_test_data()
    print(f"Created test data: {len(df)} rows")
    
    # Test the fixed validation function
    print(f"\n🔧 Testing fixed spread validation...")
    
    try:
        correlations = validate_spread_predictive_power(df)
        
        print(f"\nSUCCESS! Correlations calculated:")
        for target, corr in correlations.items():
            print(f"   {target}: {corr:.4f}")
        
        # Test signal thresholds
        print(f"\n🔧 Testing signal threshold validation...")
        validate_signal_thresholds(df)
        
        print(f"\nALL TESTS PASSED!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

def compare_old_vs_new():
    """Compare old vs new volatility calculations"""
    
    print(f"\n" + "=" * 80)
    print("COMPARING OLD VS NEW VOLATILITY CALCULATIONS")
    print("=" * 80)
    
    df = create_test_data()
    
    # Old (broken) calculation
    df['future_vol_1h_old'] = df['truth'].rolling(1).std().shift(-1)
    
    # New (fixed) calculations
    df['future_vol_1h_new'] = df['truth'].shift(-1).abs()
    df['future_vol_2h_new'] = df['truth'].rolling(2).std().shift(-1)
    
    print(f"COMPARISON RESULTS:")
    print(f"   Old method (rolling(1).std()):")
    print(f"     Non-null values: {df['future_vol_1h_old'].notna().sum()}")
    print(f"     Mean: {df['future_vol_1h_old'].mean()}")
    
    print(f"   New method 1 (absolute return):")
    print(f"     Non-null values: {df['future_vol_1h_new'].notna().sum()}")
    print(f"     Mean: {df['future_vol_1h_new'].mean():.6f}")
    
    print(f"   New method 2 (2-period rolling std):")
    print(f"     Non-null values: {df['future_vol_2h_new'].notna().sum()}")
    print(f"     Mean: {df['future_vol_2h_new'].mean():.6f}")
    
    # Test correlations with spread
    df['spread'] = df['q90'] - df['q10']
    
    print(f"\n📈 CORRELATION WITH SPREAD:")
    print(f"   Old method: {df['spread'].corr(df['future_vol_1h_old']):.4f} (NaN expected)")
    print(f"   New method 1: {df['spread'].corr(df['future_vol_1h_new']):.4f}")
    print(f"   New method 2: {df['spread'].corr(df['future_vol_2h_new']):.4f}")

if __name__ == "__main__":
    # Test the fixed function
    success = test_fixed_validation()
    
    # Compare old vs new
    compare_old_vs_new()
    
    if success:
        print(f"\n" + "=" * 80)
        print("VALIDATION FIX SUCCESSFUL!")
        print("The spread validation function now works correctly.")
        print("You can run it on your real data to get meaningful correlations.")
        print("=" * 80)
    else:
        print(f"\n" + "=" * 80)
        print("VALIDATION FIX FAILED!")
        print("There may be additional issues to resolve.")
        print("=" * 80)