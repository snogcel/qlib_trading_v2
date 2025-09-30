#!/usr/bin/env python3
"""
Phase 1 Demo: Statistical Validation for Key Features
Demonstrates the statistical validation approach for core features
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from scipy import stats

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training_pipeline import prob_up_piecewise, get_vol_raw_decile

class TestPhase1Demo:
    """Demo of Phase 1 statistical validation approach"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n = 500
        
        # Create realistic quantile data
        returns = np.random.normal(0, 0.02, n)
        q50_values = returns + np.random.normal(0, 0.005, n)
        spread = np.random.uniform(0.01, 0.04, n)
        
        data = pd.DataFrame({
            'q10': q50_values - spread/2,
            'q50': q50_values,
            'q90': q50_values + spread/2,
            'returns': returns,
            'vol_raw': np.random.uniform(0.001, 0.02, n)
        })
        
        return data
    
    def test_q50_statistical_significance(self, sample_data):
        """Test Q50 has statistically significant correlation with returns"""
        # Time-series aware test (no look-ahead bias)
        correlation = sample_data['q50'].corr(sample_data['returns'].shift(-1))
        
        # Statistical significance test
        n = len(sample_data) - 1  # Account for shift
        t_stat = correlation * np.sqrt((n-2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
        
        print(f"Q50 Statistical Validation:")
        print(f"   Correlation: {correlation:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Sample size: {n}")
        
        assert abs(correlation) > 0.01, f"Q50 should have meaningful correlation: {correlation:.4f}"
        assert p_value < 0.1, f"Correlation should be significant: p={p_value:.4f}"
    
    def test_probability_calculation_logic(self, sample_data):
        """Test probability calculation follows economic logic"""
        # Test various scenarios
        test_cases = [
            {'q10': -0.02, 'q50': 0.01, 'q90': 0.03},  # Positive q50
            {'q10': -0.03, 'q50': -0.01, 'q90': 0.02}, # Negative q50
            {'q10': 0.01, 'q50': 0.02, 'q90': 0.03},   # All positive
            {'q10': -0.03, 'q50': -0.02, 'q90': -0.01} # All negative
        ]
        
        results = []
        for case in test_cases:
            row = pd.Series(case)
            prob_up = prob_up_piecewise(row)
            results.append(prob_up)
            
            # Economic logic: probability should be in [0, 1]
            assert 0 <= prob_up <= 1, f"Probability should be in [0,1]: {prob_up}"
        
        print(f"Probability Calculation Validation:")
        print(f"   Test cases: {len(test_cases)}")
        print(f"   Results: {[f'{r:.3f}' for r in results]}")
        print(f"   All in [0,1]: {all(0 <= r <= 1 for r in results)}")
        
        # Positive q50 should generally have higher prob_up
        assert results[0] > 0.5, "Positive q50 should have prob_up > 0.5"
        assert results[1] < 0.5, "Negative q50 should have prob_up < 0.5"
    
    def test_vol_raw_decile_stability(self, sample_data):
        """Test vol_raw decile calculation is stable and reasonable"""
        vol_values = sample_data['vol_raw'].values
        deciles = [get_vol_raw_decile(vol) for vol in vol_values]
        
        # Should use multiple deciles
        unique_deciles = len(set(deciles))
        decile_distribution = pd.Series(deciles).value_counts().sort_index()
        
        print(f"Vol_Raw Decile Validation:")
        print(f"   Unique deciles: {unique_deciles}")
        print(f"   Distribution: {dict(decile_distribution)}")
        print(f"   Range: [{min(deciles)}, {max(deciles)}]")
        
        assert unique_deciles >= 5, f"Should use multiple deciles: {unique_deciles}"
        assert all(0 <= d <= 9 for d in deciles), "All deciles should be in [0, 9]"
        
        # Higher volatility should generally get higher deciles
        high_vol_mask = sample_data['vol_raw'] > sample_data['vol_raw'].quantile(0.8)
        low_vol_mask = sample_data['vol_raw'] < sample_data['vol_raw'].quantile(0.2)
        
        high_vol_deciles = [deciles[i] for i in range(len(deciles)) if high_vol_mask.iloc[i]]
        low_vol_deciles = [deciles[i] for i in range(len(deciles)) if low_vol_mask.iloc[i]]
        
        if len(high_vol_deciles) > 0 and len(low_vol_deciles) > 0:
            avg_high = np.mean(high_vol_deciles)
            avg_low = np.mean(low_vol_deciles)
            assert avg_high > avg_low, f"High vol should have higher deciles: {avg_high:.1f} vs {avg_low:.1f}"
    
    def test_feature_stability_over_time(self, sample_data):
        """Test feature stability using rolling windows"""
        # Split data into time windows
        window_size = 100
        correlations = []
        
        for i in range(window_size, len(sample_data)-1, 50):  # Every 50 periods
            window_data = sample_data.iloc[i-window_size:i]
            corr = window_data['q50'].corr(window_data['returns'].shift(-1))
            if not np.isnan(corr):
                correlations.append(corr)
        
        if len(correlations) >= 3:
            stability_metric = np.std(correlations) / (np.abs(np.mean(correlations)) + 1e-6)
            
            print(f"Feature Stability Validation:")
            print(f"   Windows tested: {len(correlations)}")
            print(f"   Correlations: {[f'{c:.3f}' for c in correlations]}")
            print(f"   Stability metric: {stability_metric:.2f}")
            
            assert stability_metric < 10.0, f"Feature should be reasonably stable: {stability_metric:.2f}"
        else:
            print(" Not enough data for stability testing")
    
    def test_economic_logic_validation(self, sample_data):
        """Test that features follow economic logic"""
        # Test 1: Q50 should be centered around actual returns
        q50_mean = sample_data['q50'].mean()
        returns_mean = sample_data['returns'].mean()
        
        # Should be reasonably close (within 2 standard deviations)
        returns_std = sample_data['returns'].std()
        difference = abs(q50_mean - returns_mean)
        
        print(f"Economic Logic Validation:")
        print(f"   Q50 mean: {q50_mean:.4f}")
        print(f"   Returns mean: {returns_mean:.4f}")
        print(f"   Difference: {difference:.4f}")
        print(f"   Returns std: {returns_std:.4f}")
        
        assert difference < 2 * returns_std, f"Q50 should be close to returns mean: diff={difference:.4f}"
        
        # Test 2: Spread (q90 - q10) should be positive
        sample_data['spread'] = sample_data['q90'] - sample_data['q10']
        assert (sample_data['spread'] > 0).all(), "Spread should always be positive"
        
        # Test 3: Q50 should be between Q10 and Q90
        assert (sample_data['q10'] <= sample_data['q50']).all(), "Q10 should be <= Q50"
        assert (sample_data['q50'] <= sample_data['q90']).all(), "Q50 should be <= Q90"
        
        print(f"   Spread always positive: {(sample_data['spread'] > 0).all()}")
        print(f"   Q10 <= Q50 <= Q90: {((sample_data['q10'] <= sample_data['q50']) & (sample_data['q50'] <= sample_data['q90'])).all()}")


if __name__ == "__main__":
    # Run the demo tests
    pytest.main([__file__, "-v", "-s"])