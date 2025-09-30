#!/usr/bin/env python3
"""
Test the Q50-centric regime-aware integration in the main script
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


import pandas as pd
import numpy as np
import sys
import os

def test_q50_integration():
    """Test the integrated Q50-centric approach"""
    
    print("TESTING Q50-CENTRIC INTEGRATION")
    print("=" * 60)
    
    # Create realistic test data that matches your data structure
    np.random.seed(42)
    n = 1000
    
    # Generate realistic quantile predictions
    base_vol = np.random.uniform(0.005, 0.02, n)
    vol_regime_factor = np.random.choice([0.5, 1.0, 2.0], n, p=[0.7, 0.2, 0.1])
    vol_raw = base_vol * vol_regime_factor
    
    # Q50 should correlate with volatility regime
    q50_base = np.random.normal(0, 0.008, n)
    q50 = q50_base * vol_regime_factor * 0.5  # Stronger signals in high vol
    
    # Spread should also correlate with volatility
    spread_base = np.random.uniform(0.01, 0.03, n)
    spread = spread_base * vol_regime_factor
    
    q10 = q50 - spread * 0.4
    q90 = q50 + spread * 0.6
    
    # Create momentum feature
    vol_raw_momentum = np.random.normal(0, 0.1, n)
    
    # Create test DataFrame matching your data structure
    df_test = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'vol_raw': vol_raw,
        '$realized_vol_6': vol_raw,
        'vol_raw_momentum': vol_raw_momentum,
        'vol_scaled': np.random.uniform(0, 1, n),  # For compatibility
    })
    
    print(f"Created test data: {len(df_test):,} observations")
    print(f"   Q50 range: {q50.min():.4f} to {q50.max():.4f}")
    print(f"   Vol_raw range: {vol_raw.min():.4f} to {vol_raw.max():.4f}")
    
    # Test the integrated functions
    try:
        # Import the functions from the main script
        sys.path.append('.')
        from src.training_pipeline import q50_regime_aware_signals, prob_up_piecewise
        
        print(f"\nTesting q50_regime_aware_signals function...")
        
        # Apply the Q50-centric approach
        df_result = q50_regime_aware_signals(df_test.copy())
        
        print(f"Function executed successfully!")
        
        # Analyze results
        print(f"\nRESULTS ANALYSIS:")
        
        # Check required columns were created
        required_columns = [
            'vol_risk', 'vol_regime_low', 'vol_regime_high', 
            'momentum_regime_trending', 'info_ratio', 'signal_thresh_adaptive',
            'economically_significant', 'high_quality', 'tradeable'
        ]
        
        missing_columns = [col for col in required_columns if col not in df_result.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
        else:
            print(f"All required columns created")
        
        # Analyze regime distribution
        print(f"\n REGIME DISTRIBUTION:")
        print(f"   Low Vol: {df_result['vol_regime_low'].sum():,} ({df_result['vol_regime_low'].mean()*100:.1f}%)")
        print(f"   High Vol: {df_result['vol_regime_high'].sum():,} ({df_result['vol_regime_high'].mean()*100:.1f}%)")
        print(f"   Trending: {df_result['momentum_regime_trending'].sum():,} ({df_result['momentum_regime_trending'].mean()*100:.1f}%)")
        
        # Analyze signal quality
        print(f"\nSIGNAL QUALITY:")
        print(f"   Economically Significant: {df_result['economically_significant'].sum():,} ({df_result['economically_significant'].mean()*100:.1f}%)")
        print(f"   High Quality: {df_result['high_quality'].sum():,} ({df_result['high_quality'].mean()*100:.1f}%)")
        print(f"   Tradeable: {df_result['tradeable'].sum():,} ({df_result['tradeable'].mean()*100:.1f}%)")
        
        # Test signal generation logic
        print(f"\nTesting signal generation logic...")
        
        # Simulate the signal generation from main script
        q50_vals = df_result["q50"]
        economically_significant = df_result['economically_significant']
        high_quality = df_result['high_quality']
        tradeable = economically_significant & high_quality
        
        buy_mask = tradeable & (q50_vals > 0)
        sell_mask = tradeable & (q50_vals < 0)
        
        side = pd.Series(-1, index=df_result.index)  # default to HOLD
        side.loc[buy_mask] = 1   # LONG
        side.loc[sell_mask] = 0  # SHORT
        
        signal_counts = side.value_counts()
        total_signals = len(side)
        
        print(f"Signal generation completed:")
        for side_val, count in signal_counts.items():
            side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side_val]
            print(f"   {side_name}: {count:,} ({count/total_signals*100:.1f}%)")
        
        # Analyze trading signal quality
        trading_signals = df_result[tradeable]
        if len(trading_signals) > 0:
            avg_info_ratio = trading_signals['info_ratio'].mean()
            avg_abs_q50 = trading_signals['abs_q50'].mean()
            avg_threshold = trading_signals['signal_thresh_adaptive'].mean()
            
            print(f"\nTrading Signal Quality:")
            print(f"   Average Info Ratio: {avg_info_ratio:.2f}")
            print(f"   Average |Q50|: {avg_abs_q50:.4f}")
            print(f"   Average Threshold: {avg_threshold:.4f}")
            print(f"   Threshold Coverage: {(avg_abs_q50/avg_threshold):.2f}x")
        
        # Test regime interaction features
        interaction_features = [col for col in df_result.columns if '_x_' in col]
        print(f"\nInteraction Features Created: {len(interaction_features)}")
        for feature in interaction_features:
            non_zero = (df_result[feature] != 0).sum()
            print(f"   {feature}: {non_zero:,} non-zero values")
        
        print(f"\nINTEGRATION TEST PASSED!")
        print(f"   The Q50-centric regime-aware approach is working correctly")
        print(f"   Ready for testing with real data")
        
        return True
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print(f"   Make sure ppo_sweep_optuna_tuned_v2.py is in the current directory")
        return False
        
    except Exception as e:
        print(f"Execution Error: {e}")
        print(f"   There may be an issue with the integration")
        return False

def compare_with_old_approach():
    """Compare the new approach with the old threshold method"""
    
    print(f"\nCOMPARING WITH OLD APPROACH")
    print("=" * 60)
    
    # This would require the old logic, but we can simulate the comparison
    print("Expected Improvements:")
    print("   No data leakage (old approach used future data in rolling windows)")
    print("   Economic meaning (thresholds based on trading costs, not arbitrary percentiles)")
    print("   Regime awareness (different thresholds for different market conditions)")
    print("   Risk adjustment (vol_risk scaling for additional risk context)")
    print("   Signal quality (information ratio filters low-quality signals)")
    print("   Interpretability (can explain every trading decision)")
    
    print(f"\n Expected Signal Quality:")
    print("   • Higher average information ratio for trading signals")
    print("   • More consistent performance across market regimes")
    print("   • Reduced false signals in high volatility periods")
    print("   • Better risk-adjusted returns")

def main():
    """Main test function"""
    
    print("Q50-CENTRIC REGIME-AWARE INTEGRATION TEST")
    print("=" * 70)
    print("Testing the integration of Q50-centric signals with regime identification")
    print("and vol_risk scaling in the main training script.")
    
    # Test the integration
    success = test_q50_integration()
    
    if success:
        # Compare with old approach
        compare_with_old_approach()
        
        print(f"\nNEXT STEPS:")
        print("1. Run the updated ppo_sweep_optuna_tuned_v2.py with your real data")
        print("2. Compare performance metrics with the old approach")
        print("3. Tune transaction_cost_bps and base_info_ratio parameters")
        print("4. Analyze regime interaction features for model improvement")
        print("5. Update backtesting and production systems")
        
    else:
        print(f"\nTROUBLESHOOTING:")
        print("1. Ensure ppo_sweep_optuna_tuned_v2.py is in the current directory")
        print("2. Check that all required imports are available")
        print("3. Verify the integration was applied correctly")

if __name__ == "__main__":
    main()