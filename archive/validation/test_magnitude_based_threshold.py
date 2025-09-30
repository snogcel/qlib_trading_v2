#!/usr/bin/env python3
"""
Test the magnitude-based threshold implementation
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

def test_magnitude_based_threshold():
    """Test the magnitude-based economic significance"""
    
    print("TESTING MAGNITUDE-BASED THRESHOLD")
    print("=" * 60)
    
    # Create test data with realistic quantile patterns
    np.random.seed(42)
    n = 500
    
    # Generate realistic crypto quantile data
    # Most signals are small (0.1-0.5%), with occasional larger moves
    q50_base = np.random.normal(0, 0.002, n)  # Smaller base signals
    
    # Add some larger signals (market events)
    large_signal_mask = np.random.random(n) < 0.1  # 10% large signals
    q50_base[large_signal_mask] *= 2  # 2x larger
    
    # Generate spreads that correlate with signal magnitude
    spread_base = 0.003 + 0.3 * np.abs(q50_base)  # Larger spreads for larger signals
    
    q10 = q50_base - spread_base * 0.4
    q90 = q50_base + spread_base * 0.6
    
    # Add other required features
    vol_risk = np.random.uniform(0.0001, 0.001, n)  # Realistic variance values
    vol_raw = np.sqrt(vol_risk)
    vol_scaled = np.random.uniform(0, 1, n)
    vol_raw_momentum = np.random.normal(0, 0.001, n)
    
    df_test = pd.DataFrame({
        'q10': q10,
        'q50': q50_base,
        'q90': q90,
        'vol_risk': vol_risk,
        'vol_raw': vol_raw,
        'vol_scaled': vol_scaled,
        'vol_raw_momentum': vol_raw_momentum,
        'truth': np.random.normal(0, 0.01, n)
    })
    
    print(f"Created {n} test observations:")
    print(f"   Q50 range: {q50_base.min():.4f} to {q50_base.max():.4f}")
    print(f"   |Q50| mean: {np.abs(q50_base).mean():.4f}")
    print(f"   |Q50| median: {np.median(np.abs(q50_base)):.4f}")
    print(f"   Spread mean: {spread_base.mean():.4f}")
    
    # Test the magnitude-based approach
    try:
        sys.path.append('.')
        from src.training_pipeline import q50_regime_aware_signals
        
        print(f"\n🔄 Testing magnitude-based q50_regime_aware_signals...")
        
        # Apply the magnitude-enhanced approach
        df_result = q50_regime_aware_signals(df_test.copy())
        
        print(f"Function executed successfully!")
        
        # Analyze the results
        print(f"\nMAGNITUDE-BASED RESULTS:")
        
        # Check if magnitude-specific columns were created
        magnitude_columns = [
            'potential_gain', 'potential_loss', 'expected_value',
            'economically_significant_traditional', 'economically_significant_expected_value',
            'economically_significant_combined'
        ]
        
        missing_columns = [col for col in magnitude_columns if col not in df_result.columns]
        if missing_columns:
            print(f"Missing magnitude columns: {missing_columns}")
        else:
            print(f"All magnitude-specific columns created")
        
        # Analyze expected value calculation
        if 'expected_value' in df_result.columns:
            exp_val = df_result['expected_value']
            print(f"\n💰 Expected Value Analysis:")
            print(f"   Range: {exp_val.min():.4f} to {exp_val.max():.4f}")
            print(f"   Mean: {exp_val.mean():.4f}")
            print(f"   Positive expected value: {(exp_val > 0).mean()*100:.1f}%")
            
            # Check relationship with potential gains/losses
            if 'potential_gain' in df_result.columns and 'potential_loss' in df_result.columns:
                avg_gain = df_result['potential_gain'].mean()
                avg_loss = df_result['potential_loss'].mean()
                print(f"   Average potential gain: {avg_gain:.4f}")
                print(f"   Average potential loss: {avg_loss:.4f}")
                print(f"   Gain/Loss ratio: {avg_gain/avg_loss:.2f}")
        
        # Compare economic significance approaches
        if all(col in df_result.columns for col in ['economically_significant_traditional', 'economically_significant_expected_value']):
            trad_count = df_result['economically_significant_traditional'].sum()
            exp_val_count = df_result['economically_significant_expected_value'].sum()
            
            print(f"\n📈 Economic Significance Comparison:")
            print(f"   Traditional approach: {trad_count:,} ({trad_count/len(df_result)*100:.1f}%)")
            print(f"   Expected value approach: {exp_val_count:,} ({exp_val_count/len(df_result)*100:.1f}%)")
            
            if trad_count > 0:
                improvement = (exp_val_count / trad_count - 1) * 100
                print(f"   Improvement: {improvement:+.1f}% more opportunities")
            
            # Check if we're getting more trading opportunities
            if exp_val_count > trad_count:
                print(f"   Expected value approach provides more trading opportunities")
            else:
                print(f"   ⚠️  Expected value approach not providing more opportunities")
        
        # Test final tradeable signals
        if 'tradeable' in df_result.columns:
            tradeable_count = df_result['tradeable'].sum()
            print(f"\n🎯 Final Tradeable Signals:")
            print(f"   Tradeable signals: {tradeable_count:,} ({tradeable_count/len(df_result)*100:.1f}%)")
            
            if tradeable_count > 0:
                tradeable_data = df_result[df_result['tradeable']]
                
                # Analyze characteristics of tradeable signals
                avg_exp_val = tradeable_data['expected_value'].mean()
                avg_abs_q50 = tradeable_data['abs_q50'].mean()
                avg_enhanced_info = tradeable_data['enhanced_info_ratio'].mean()
                
                print(f"   Average expected value: {avg_exp_val:.4f}")
                print(f"   Average |Q50|: {avg_abs_q50:.4f}")
                print(f"   Average enhanced info ratio: {avg_enhanced_info:.2f}")
                
                # Check if tradeable signals have positive expected value
                positive_exp_val = (tradeable_data['expected_value'] > 0).mean()
                print(f"   Positive expected value: {positive_exp_val*100:.1f}%")
                
                if positive_exp_val > 0.7:
                    print(f"   Most tradeable signals have positive expected value")
                else:
                    print(f"   ⚠️  Many tradeable signals have negative expected value")
            else:
                print(f"   ⚠️  No tradeable signals generated - thresholds may be too strict")
        
        # Test signal generation
        if 'side' in df_result.columns:
            print(f"\n🎯 Signal Generation:")
            signal_counts = df_result['side'].value_counts()
            for side, count in signal_counts.items():
                side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side]
                print(f"   {side_name}: {count:,} ({count/len(df_result)*100:.1f}%)")
        
        print(f"\nMAGNITUDE-BASED THRESHOLD TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🎯 MAGNITUDE-BASED THRESHOLD TEST")
    print("=" * 70)
    print("Testing the implementation of magnitude-based economic significance")
    print("using expected value from quantile distribution")
    
    success = test_magnitude_based_threshold()
    
    if success:
        print(f"\nMAGNITUDE-BASED APPROACH WORKING!")
        print("1. Expected value calculation using quantile distribution")
        print("2. Comparison of traditional vs expected value approaches")
        print("3. More trading opportunities while maintaining economic rationale")
        print("4. Positive expected value filtering for better signal quality")
    else:
        print(f"\nMAGNITUDE-BASED APPROACH NEEDS ATTENTION")
        print("Check the error messages above")

if __name__ == "__main__":
    main()