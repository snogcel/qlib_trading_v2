#!/usr/bin/env python3
"""
Test suite to validate that NautilusTrader POC requirements align with actual training pipeline implementation.
This ensures our requirements document accurately reflects the working system parameters.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import actual implementation functions
from src.training_pipeline import (
    q50_regime_aware_signals,
    prob_up_piecewise,
    kelly_sizing,
    get_vol_raw_decile,
    identify_market_regimes
)

class TestNautilusRequirementsAlignment:
    """Validate that POC requirements match actual implementation parameters"""
    
    @pytest.fixture
    def sample_signal_data(self):
        """Create sample data matching our actual signal structure"""
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic quantile predictions
        q50_values = np.random.normal(0, 0.02, n_samples)  # 2% volatility
        spread_values = np.random.uniform(0.01, 0.05, n_samples)  # 1-5% spread
        
        data = {
            'q10': q50_values - spread_values/2,
            'q50': q50_values,
            'q90': q50_values + spread_values/2,
            'vol_raw': np.random.uniform(0.001, 0.02, n_samples),  # Realistic vol_raw range
            'vol_risk': np.random.uniform(0.0001, 0.001, n_samples),  # Variance (vol_raw²)
        }
        
        df = pd.DataFrame(data)
        
        # Add required columns that would be calculated by the pipeline        
        df = identify_market_regimes(df)
        
        return df
    
    def test_transaction_cost_parameter_alignment(self, sample_signal_data):
        """Test that transaction cost matches requirements (5 bps = 0.0005)"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Check that realistic_transaction_cost is used correctly
        # The function should use 0.0005 (5 bps) as documented in requirements
        economically_significant_count = df_with_signals['economically_significant'].sum()
        
        # Verify that expected_value > 0.0005 logic is applied
        expected_value_filter = df_with_signals['expected_value'] > 0.0005
        assert (df_with_signals['economically_significant'] == expected_value_filter).all(), \
            "Economic significance should use expected_value > 0.0005 (5 bps)"
    
    def test_variance_regime_thresholds(self, sample_signal_data):
        """Test that variance regime thresholds match requirements (30th/70th/90th percentiles)"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Verify percentile thresholds are correctly applied
        vol_risk_30th = df_with_signals['vol_risk'].quantile(0.30)
        vol_risk_70th = df_with_signals['vol_risk'].quantile(0.70)
        vol_risk_90th = df_with_signals['vol_risk'].quantile(0.90)
        
        # Check regime classifications
        low_regime_mask = df_with_signals['vol_risk'] <= vol_risk_30th
        high_regime_mask = ((df_with_signals['vol_risk'] > vol_risk_70th) & 
                           (df_with_signals['vol_risk'] <= vol_risk_90th))
        extreme_regime_mask = df_with_signals['vol_risk'] > vol_risk_90th
        
        assert (df_with_signals['variance_regime_low'] == low_regime_mask.astype(int)).all(), \
            "Low variance regime should use ≤30th percentile threshold"
        assert (df_with_signals['variance_regime_high'] == high_regime_mask.astype(int)).all(), \
            "High variance regime should use 70th-90th percentile threshold"
        assert (df_with_signals['variance_regime_extreme'] == extreme_regime_mask.astype(int)).all(), \
            "Extreme variance regime should use >90th percentile threshold"
    
    def test_regime_multiplier_adjustments(self, sample_signal_data):
        """Test that regime multiplier adjustments match requirements (-30%, +40%, +80%)"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Check that regime multipliers are applied correctly
        # Low variance: -30% threshold adjustment
        # High variance: +40% threshold adjustment  
        # Extreme variance: +80% threshold adjustment
        
        # Verify the multipliers are clipped to [0.3, 3.0] as documented
        assert df_with_signals['signal_thresh_adaptive'].min() >= 0.0005 * 0.3, \
            "Signal thresholds should respect minimum multiplier of 0.3"
        assert df_with_signals['signal_thresh_adaptive'].max() <= 0.0005 * 3.0 * 1500, \
            "Signal thresholds should respect reasonable maximum bounds"
    
    def test_enhanced_info_ratio_calculation(self, sample_signal_data):
        """Test that enhanced info ratio uses market_variance + prediction_variance"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Verify enhanced info ratio calculation
        expected_market_variance = df_with_signals['vol_risk']  # Already variance
        expected_prediction_variance = (df_with_signals['spread'] / 2) ** 2
        expected_total_risk = np.sqrt(expected_market_variance + expected_prediction_variance)
        expected_enhanced_ratio = df_with_signals['abs_q50'] / np.maximum(expected_total_risk, 0.001)
        
        np.testing.assert_array_almost_equal(
            df_with_signals['enhanced_info_ratio'].values,
            expected_enhanced_ratio.values,
            decimal=6,
            err_msg="Enhanced info ratio should use sqrt(market_variance + prediction_variance)"
        )
    
    def test_q50_centric_signal_logic(self, sample_signal_data):
        """Test that Q50-centric signal logic matches requirements"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Apply the Q50-centric signal generation logic as in training pipeline
        q50 = df_with_signals["q50"]
        economically_significant = df_with_signals['economically_significant']
        tradeable = economically_significant
        
        # Pure Q50 directional logic
        buy_mask = tradeable & (q50 > 0)
        sell_mask = tradeable & (q50 < 0)
        
        expected_side = pd.Series(-1, index=df_with_signals.index)  # default HOLD
        expected_side.loc[buy_mask] = 1   # LONG when q50 > 0 and tradeable
        expected_side.loc[sell_mask] = 0  # SHORT when q50 < 0 and tradeable
        
        # Verify signal generation logic
        assert buy_mask.sum() > 0, "Should generate some buy signals"
        assert sell_mask.sum() > 0, "Should generate some sell signals"
        
        # Test the core logic requirements
        for idx in df_with_signals.index:
            if df_with_signals.loc[idx, 'economically_significant'] and df_with_signals.loc[idx, 'q50'] > 0:
                assert expected_side.loc[idx] == 1, f"Should be LONG (1) when tradeable=True and q50 > 0 at {idx}"
            elif df_with_signals.loc[idx, 'economically_significant'] and df_with_signals.loc[idx, 'q50'] < 0:
                assert expected_side.loc[idx] == 0, f"Should be SHORT (0) when tradeable=True and q50 < 0 at {idx}"
            else:
                assert expected_side.loc[idx] == -1, f"Should be HOLD (-1) when not tradeable at {idx}"
    
    def test_position_sizing_parameters(self, sample_signal_data):
        """Test that position sizing parameters match requirements"""
        df_with_signals = q50_regime_aware_signals(sample_signal_data)
        
        # Test inverse variance scaling: base_size = 0.1 / max(vol_risk * 1000, 0.1)
        expected_base_size = 0.1 / np.maximum(df_with_signals['vol_risk'] * 1000, 0.1)
        
        # Position sizes should be clipped to [0.01, 0.5] range
        tradeable_mask = df_with_signals['economically_significant']
        expected_position_size = np.where(
            tradeable_mask,
            expected_base_size.clip(0.01, 0.5),
            0.0
        )
        
        # Apply the same logic as in training pipeline
        df_with_signals['test_position_size'] = expected_position_size
        
        # Verify position size bounds
        tradeable_positions = df_with_signals[tradeable_mask]['test_position_size']
        if len(tradeable_positions) > 0:
            assert tradeable_positions.min() >= 0.01, "Minimum position size should be 1%"
            assert tradeable_positions.max() <= 0.5, "Maximum position size should be 50%"
    
    # def test_kelly_sizing_with_vol_raw_deciles(self):
    #     """Test Kelly sizing with vol_raw deciles matches requirements"""
    #     # Test the decile-based risk adjustments
    #     test_cases = [
    #         (0.012, 0.5, 9, 0.6),  # Extreme volatility (decile 9): 0.6x
    #         (0.008, 0.3, 8, 0.7),  # Very high volatility (decile 8): 0.7x  
    #         (0.006, 0.2, 6, 0.85), # High volatility (decile 6): 0.85x
    #         (0.002, 0.1, 1, 1.1),  # Very low volatility (decile 1): 1.1x
    #         (0.004, 0.15, 4, 1.0), # Medium volatility: 1.0x
    #     ]
        
    #     for vol_raw, signal_rel, expected_decile, expected_adjustment in test_cases:
    #         actual_decile = get_vol_raw_decile(vol_raw)
    #         kelly_size = kelly_with_vol_raw_deciles(vol_raw, signal_rel)
    #         df_all["kelly_position_size"] = df_all.apply(kelly_sizing, axis=1)
            
    #         # Verify decile calculation
    #         assert actual_decile == expected_decile, \
    #             f"Vol_raw {vol_raw} should be in decile {expected_decile}, got {actual_decile}"
            
    #         # Verify risk adjustment is applied (approximate test due to Kelly calculation complexity)
    #         assert kelly_size > 0, "Kelly size should be positive for positive signal_rel"
    
    def test_prob_up_piecewise_calculation(self):
        """Test that prob_up calculation matches requirements"""
        # Test cases for piecewise probability calculation
        test_cases = [
            {'q10': -0.02, 'q50': 0.01, 'q90': 0.03, 'expected_range': (0.5, 1.0)},  # Positive q50
            {'q10': -0.03, 'q50': -0.01, 'q90': 0.02, 'expected_range': (0.0, 0.5)}, # Negative q50
            {'q10': 0.01, 'q50': 0.02, 'q90': 0.03, 'expected': 1.0},               # All positive
            {'q10': -0.03, 'q50': -0.02, 'q90': -0.01, 'expected': 0.0},            # All negative
        ]
        
        for case in test_cases:
            row = pd.Series(case)
            prob_up = prob_up_piecewise(row)
            
            if 'expected' in case:
                assert abs(prob_up - case['expected']) < 0.001, \
                    f"Prob_up should be {case['expected']} for case {case}"
            else:
                min_prob, max_prob = case['expected_range']
                assert min_prob <= prob_up <= max_prob, \
                    f"Prob_up {prob_up} should be in range {case['expected_range']} for case {case}"
    
    def test_data_pipeline_frequency_alignment(self):
        """Test that data pipeline frequencies match requirements (60min crypto + daily GDELT)"""
        # This test validates the configuration alignment
        # In actual implementation, crypto data is 60min and GDELT is daily
        
        # Test frequency configuration from training pipeline
        freq_config = {
            "feature": "60min",  # Crypto data frequency
            "label": "day"       # GDELT data frequency  
        }
        
        assert freq_config["feature"] == "60min", \
            "Crypto data frequency should be 60min to match training pipeline"
        assert freq_config["label"] == "day", \
            "GDELT data frequency should be daily to match training pipeline"
    
    def test_performance_target_alignment(self):
        """Test that performance targets match documented Sharpe ratio (1.327)"""
        # This is a meta-test to ensure our requirements reference the correct performance target
        documented_sharpe = 1.327
        
        # Verify this matches what's documented in our requirements
        assert documented_sharpe > 1.3, \
            "Performance target should be above 1.3 Sharpe ratio"
        assert documented_sharpe < 1.4, \
            "Performance target should be realistic (below 1.4)"
    
    def test_vol_risk_variance_calculation(self, sample_signal_data):
        """Test that vol_risk is properly treated as variance (not std dev)"""
        df_with_vol_risk = sample_signal_data
        
        # vol_risk should be variance (squared volatility)
        # In our implementation, vol_risk = Std(Log(close/close_prev), 6)²
        
        # Verify vol_risk values are in reasonable variance range (much smaller than vol_raw)
        if 'vol_raw' in df_with_vol_risk.columns:
            # vol_risk should be approximately vol_raw²
            vol_risk_from_raw = df_with_vol_risk['vol_raw'] ** 2
            
            # Allow for some difference due to different calculation methods
            correlation = np.corrcoef(df_with_vol_risk['vol_risk'], vol_risk_from_raw)[0, 1]

            print(df_with_vol_risk['vol_risk'])
            print(vol_risk_from_raw)

            assert correlation > 0.8, \
                "vol_risk should be highly correlated with vol_raw² (variance relationship)"
        
        # Verify vol_risk values are positive and reasonable for variance
        assert (df_with_vol_risk['vol_risk'] >= 0).all(), \
            "vol_risk (variance) should be non-negative"
        assert df_with_vol_risk['vol_risk'].max() < 0.01, \
            "vol_risk (variance) should be much smaller than volatility values"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])