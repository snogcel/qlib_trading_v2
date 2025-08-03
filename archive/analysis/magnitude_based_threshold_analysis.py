#!/usr/bin/env python3
"""
Analyze magnitude-based thresholds using quantile predictions for better economic significance
"""
import pandas as pd
import numpy as np

def analyze_quantile_magnitudes(df):
    """
    Analyze the actual magnitudes in quantile predictions to set realistic thresholds
    """
    
    print("🔍 QUANTILE MAGNITUDE ANALYSIS")
    print("=" * 60)
    
    if not all(col in df.columns for col in ['q10', 'q50', 'q90']):
        print("❌ Missing quantile columns")
        return df
    
    # Basic statistics
    print("📊 Quantile Statistics:")
    for col in ['q10', 'q50', 'q90']:
        values = df[col].dropna()
        print(f"   {col}: mean={values.mean():.4f}, std={values.std():.4f}, "
              f"range=[{values.min():.4f}, {values.max():.4f}]")
    
    # Magnitude analysis
    df['abs_q50'] = df['q50'].abs()
    df['spread'] = df['q90'] - df['q10']
    
    print(f"\n📈 Magnitude Analysis:")
    print(f"   |Q50| mean: {df['abs_q50'].mean():.4f}")
    print(f"   |Q50| median: {df['abs_q50'].median():.4f}")
    print(f"   |Q50| 90th percentile: {df['abs_q50'].quantile(0.9):.4f}")
    print(f"   Spread mean: {df['spread'].mean():.4f}")
    
    # Potential gain/loss estimation
    df['potential_gain'] = np.where(df['q50'] > 0, df['q90'], np.abs(df['q10']))
    df['potential_loss'] = np.where(df['q50'] > 0, np.abs(df['q10']), df['q90'])
    
    print(f"\n💰 Potential Gain/Loss Analysis:")
    print(f"   Average potential gain: {df['potential_gain'].mean():.4f}")
    print(f"   Average potential loss: {df['potential_loss'].mean():.4f}")
    print(f"   Gain/Loss ratio: {df['potential_gain'].mean() / df['potential_loss'].mean():.2f}")
    
    return df

def calculate_expected_value_threshold(df, transaction_cost_bps=20):
    """
    Calculate threshold based on expected value considering full quantile distribution
    """
    
    print(f"\n🎯 EXPECTED VALUE THRESHOLD CALCULATION")
    print("=" * 60)
    
    transaction_cost = transaction_cost_bps / 10000
    
    # Method 1: Expected value using quantile distribution
    # E[return] = q50 (median estimate)
    # Risk = spread/2 (rough std dev estimate)
    df['expected_return'] = df['q50']
    df['estimated_risk'] = df['spread'] / 2
    
    # Method 2: Probability-weighted expected value
    # Use prob_up to weight potential gains vs losses
    if 'prob_up' in df.columns:
        prob_up = df['prob_up']
    else:
        # Calculate prob_up if not available
        def prob_up_calc(row):
            q10, q50, q90 = row['q10'], row['q50'], row['q90']
            if q90 <= 0:
                return 0.0
            if q10 >= 0:
                return 1.0
            if q10 < 0 <= q50:
                cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
                return 1 - cdf0
            cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
            return 1 - cdf0
        
        prob_up = df.apply(prob_up_calc, axis=1)
    
    # Expected value = prob_up * potential_gain - (1-prob_up) * potential_loss
    df['expected_value'] = (prob_up * df['potential_gain'] - 
                           (1 - prob_up) * df['potential_loss'])
    
    print(f"📊 Expected Value Analysis:")
    print(f"   Mean expected value: {df['expected_value'].mean():.4f}")
    print(f"   Median expected value: {df['expected_value'].median():.4f}")
    print(f"   Positive expected value: {(df['expected_value'] > 0).mean()*100:.1f}%")
    
    # Method 3: Risk-adjusted threshold
    # Threshold should be: transaction_cost + risk_premium
    # Risk premium = some multiple of estimated risk
    risk_premium_multiplier = 1.0  # 1x the estimated risk
    df['risk_adjusted_threshold'] = transaction_cost + risk_premium_multiplier * df['estimated_risk']
    
    print(f"\n🎯 Threshold Recommendations:")
    print(f"   Fixed transaction cost: {transaction_cost:.4f} ({transaction_cost_bps} bps)")
    print(f"   Average risk-adjusted threshold: {df['risk_adjusted_threshold'].mean():.4f}")
    print(f"   Median risk-adjusted threshold: {df['risk_adjusted_threshold'].median():.4f}")
    
    # Compare with current |q50| values
    current_threshold = 0.002  # Current 20 bps
    signals_above_current = (df['abs_q50'] > current_threshold).sum()
    signals_above_risk_adj = (df['abs_q50'] > df['risk_adjusted_threshold']).sum()
    
    print(f"\n📈 Signal Frequency Comparison:")
    print(f"   Current threshold (0.002): {signals_above_current:,} signals ({signals_above_current/len(df)*100:.1f}%)")
    print(f"   Risk-adjusted threshold: {signals_above_risk_adj:,} signals ({signals_above_risk_adj/len(df)*100:.1f}%)")
    
    return df

def magnitude_based_economic_significance(df, method='adaptive_risk'):
    """
    Create magnitude-based economic significance using quantile information
    """
    
    print(f"\n🎯 MAGNITUDE-BASED ECONOMIC SIGNIFICANCE")
    print("=" * 60)
    
    # Ensure we have required columns
    if 'expected_value' not in df.columns:
        df = calculate_expected_value_threshold(df)
    
    if method == 'adaptive_risk':
        # Method 1: Adaptive risk-based threshold
        # Threshold = transaction_cost + risk_premium based on individual prediction uncertainty
        base_cost = 0.0005  # 5 bps base cost (more realistic for crypto)
        risk_multiplier = 0.5  # 50% of estimated risk as premium
        
        df['magnitude_threshold'] = base_cost + risk_multiplier * df['estimated_risk']
        
    elif method == 'expected_value':
        # Method 2: Expected value must exceed transaction costs
        base_cost = 0.0005
        df['magnitude_threshold'] = base_cost
        df['economically_significant_magnitude'] = df['expected_value'] > df['magnitude_threshold']
        
    elif method == 'percentile_based':
        # Method 3: Use percentiles of actual |q50| distribution
        threshold_percentile = 0.3  # 30th percentile (more trades than 90th)
        fixed_threshold = df['abs_q50'].quantile(threshold_percentile)
        df['magnitude_threshold'] = fixed_threshold
        
    elif method == 'kelly_based':
        # Method 4: Kelly criterion - only trade when Kelly > minimum
        # Kelly = (prob_win * avg_win - prob_lose * avg_lose) / avg_lose
        prob_up = df.get('prob_up', 0.5)
        kelly_fraction = ((prob_up * df['potential_gain'] - 
                          (1 - prob_up) * df['potential_loss']) / 
                         np.maximum(df['potential_loss'], 0.001))
        
        min_kelly = 0.01  # Minimum 1% Kelly to trade
        df['magnitude_threshold'] = np.where(kelly_fraction > min_kelly, 0.0001, 0.1)  # Very low if Kelly good, high if not
    
    # Apply the magnitude-based threshold
    df['economically_significant_magnitude'] = df['abs_q50'] > df['magnitude_threshold']
    
    # Compare with current approach
    current_threshold = 0.002
    df['economically_significant_current'] = df['abs_q50'] > current_threshold
    
    # Results
    magnitude_signals = df['economically_significant_magnitude'].sum()
    current_signals = df['economically_significant_current'].sum()
    
    print(f"📊 Results for method '{method}':")
    print(f"   Average threshold: {df['magnitude_threshold'].mean():.4f}")
    print(f"   Magnitude-based signals: {magnitude_signals:,} ({magnitude_signals/len(df)*100:.1f}%)")
    print(f"   Current approach signals: {current_signals:,} ({current_signals/len(df)*100:.1f}%)")
    print(f"   Improvement: {((magnitude_signals/current_signals - 1)*100):+.1f}% more signals" if current_signals > 0 else "   Improvement: N/A")
    
    return df

def test_magnitude_approaches():
    """Test different magnitude-based approaches"""
    
    print("🧪 TESTING MAGNITUDE-BASED APPROACHES")
    print("=" * 70)
    
    # Create realistic test data
    np.random.seed(42)
    n = 2000
    
    # Generate realistic quantile data based on crypto market characteristics
    # Most signals are small, with occasional larger moves
    q50_base = np.random.normal(0, 0.003, n)  # Smaller base signals
    
    # Add some larger signals (market events)
    large_signal_mask = np.random.random(n) < 0.05  # 5% large signals
    q50_base[large_signal_mask] *= 3  # 3x larger
    
    # Generate spreads that correlate with signal magnitude
    spread_base = 0.005 + 0.5 * np.abs(q50_base)  # Larger spreads for larger signals
    
    q10 = q50_base - spread_base * 0.4
    q90 = q50_base + spread_base * 0.6
    
    df_test = pd.DataFrame({
        'q10': q10,
        'q50': q50_base,
        'q90': q90
    })
    
    print(f"📊 Created {n:,} realistic test observations")
    
    # Analyze magnitudes
    df_test = analyze_quantile_magnitudes(df_test)
    
    # Test different methods
    methods = ['adaptive_risk', 'expected_value', 'percentile_based', 'kelly_based']
    
    results = {}
    for method in methods:
        print(f"\n" + "="*50)
        df_method = magnitude_based_economic_significance(df_test.copy(), method=method)
        
        signal_count = df_method['economically_significant_magnitude'].sum()
        avg_threshold = df_method['magnitude_threshold'].mean()
        
        results[method] = {
            'signal_count': signal_count,
            'signal_rate': signal_count / len(df_method),
            'avg_threshold': avg_threshold
        }
    
    # Summary comparison
    print(f"\n📊 METHOD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} | {'Signals':<8} | {'Rate':<8} | {'Avg Threshold':<12}")
    print("-" * 70)
    
    for method, result in results.items():
        print(f"{method:<20} | {result['signal_count']:<8,} | {result['signal_rate']:<8.1%} | {result['avg_threshold']:<12.4f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. 'adaptive_risk': Adjusts threshold based on prediction uncertainty")
    print(f"2. 'expected_value': Uses probability-weighted expected returns")
    print(f"3. 'percentile_based': Simple percentile of actual |q50| distribution")
    print(f"4. 'kelly_based': Uses Kelly criterion for economic significance")
    
    return results

def main():
    """Main analysis function"""
    
    print("🎯 MAGNITUDE-BASED THRESHOLD ANALYSIS")
    print("=" * 80)
    print("Analyzing quantile magnitudes to create realistic economic thresholds")
    
    # Test the approaches
    results = test_magnitude_approaches()
    
    print(f"\n🚀 IMPLEMENTATION READY!")
    print("Choose the method that provides the right balance of:")
    print("• Signal frequency (enough trading opportunities)")
    print("• Economic realism (thresholds based on actual potential)")
    print("• Risk management (considers prediction uncertainty)")

if __name__ == "__main__":
    main()