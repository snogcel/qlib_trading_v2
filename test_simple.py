#!/usr/bin/env python3
"""
Simple test to validate basic functionality without heavy dependencies
"""

import pytest
import pandas as pd
import numpy as np

def test_basic_functionality():
    """Test basic pandas and numpy functionality"""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'q10': np.random.normal(-0.01, 0.02, n_samples),
        'q50': np.random.normal(0, 0.02, n_samples),
        'q90': np.random.normal(0.01, 0.02, n_samples),
        'vol_raw': np.random.uniform(0.001, 0.02, n_samples),
        'vol_risk': np.random.uniform(0.0001, 0.001, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Basic validation tests
    assert len(df) == n_samples
    assert 'q50' in df.columns
    assert df['vol_risk'].min() >= 0
    assert df['vol_risk'].max() < 0.01
    
    print("✓ Basic functionality test passed!")

def test_transaction_cost_logic():
    """Test transaction cost logic without dependencies"""
    transaction_cost = 0.0005  # 5 bps
    
    # Sample expected values
    expected_values = np.array([0.0003, 0.0008, 0.0001, 0.0012])
    
    # Economic significance filter
    economically_significant = expected_values > transaction_cost
    
    expected_result = np.array([False, True, False, True])
    
    assert np.array_equal(economically_significant, expected_result)
    print("✓ Transaction cost logic test passed!")

def test_variance_regime_thresholds():
    """Test variance regime threshold logic"""
    np.random.seed(42)
    vol_risk = np.random.uniform(0.0001, 0.001, 1000)
    
    # Calculate percentiles
    vol_risk_30th = np.percentile(vol_risk, 30)
    vol_risk_70th = np.percentile(vol_risk, 70)
    vol_risk_90th = np.percentile(vol_risk, 90)
    
    # Create regime classifications
    low_regime = vol_risk <= vol_risk_30th
    high_regime = (vol_risk > vol_risk_70th) & (vol_risk <= vol_risk_90th)
    extreme_regime = vol_risk > vol_risk_90th
    
    # Validate percentages are approximately correct
    assert abs(low_regime.sum() / len(vol_risk) - 0.30) < 0.05
    assert abs(extreme_regime.sum() / len(vol_risk) - 0.10) < 0.05
    
    print("✓ Variance regime thresholds test passed!")

if __name__ == "__main__":
    test_basic_functionality()
    test_transaction_cost_logic()
    test_variance_regime_thresholds()
    print("\n🎉 All simplified tests passed!")