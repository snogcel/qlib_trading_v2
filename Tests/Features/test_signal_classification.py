
# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np

def even_simpler_classificatiozzn(row):
    """
    Simplified classification focusing on what actually works
    """
    # Core validated components
    abs_q50 = row.get("abs_q50", 0)
    signal_thresh = row.get("signal_thresh_adaptive", 0.01)
    prob_up = row.get("prob_up", 0.5)
    vol_decile = row.get("vol_raw_decile", 5)
    
    # Base signal strength (this is your strongest predictor)
    if abs_q50 >= signal_thresh:
        base_tier = 3.0
    elif abs_q50 >= signal_thresh * 0.8:
        base_tier = 2.0
    elif abs_q50 >= signal_thresh * 0.6:
        base_tier = 1.0
    else:
        base_tier = 0.0
    
    # Only apply volatility adjustment to strong signals (tier 2+)
    if base_tier >= 2.0:
        if vol_decile >= 8:  # Extreme high vol - boost strong signals
            vol_adjustment = 1.2
        elif vol_decile <= 1:  # Very low vol - reduce all signals
            vol_adjustment = 0.8
        else:
            vol_adjustment = 1.0
    else:
        vol_adjustment = 1.0  # Don't adjust weak signals
    
    # Minor directional confidence (keep it simple)
    prob_confidence = abs(prob_up - 0.5) * 2
    confidence_adjustment = 0.95 + (prob_confidence * 0.1)  # 0.95 to 1.05
    
    final_tier = base_tier * vol_adjustment * confidence_adjustment
    
    # Round to clean tiers (0, 1, 2, 3) - no fractional tiers
    return round(final_tier)


def even_simpler_classification(row):
    """
    Ultra-simple version - just use what works
    """
    
    print(row)
    raise SystemExit()
    
    abs_q50 = row.get("abs_q50", 0)
    signal_thresh = row.get("signal_thresh_adaptive", 0.01)
    
    # Just use the validated threshold approach
    if abs_q50 >= signal_thresh:
        return 3  # Strong signal
    elif abs_q50 >= signal_thresh * 0.8:
        return 2  # Medium signal
    elif abs_q50 >= signal_thresh * 0.6:
        return 1  # Weak signal
    else:
        return 0  # No signal

    # TODO -- check alignment of q50 and signal_thresh_adaptive


def final_signal_classification(row):
    """
    Final simplified approach based on validation
    """
    abs_q50 = row.get("abs_q50", 0)
    signal_thresh = row.get("signal_thresh_adaptive", 0.01)
    prob_up = row.get("prob_up", 0.5)
    
    # Base tier (this is what actually works)
    if abs_q50 >= signal_thresh:
        tier = 3
    elif abs_q50 >= signal_thresh * 0.8:
        tier = 2
    elif abs_q50 >= signal_thresh * 0.6:
        tier = 1
    else:
        tier = 0
    
    # Only boost tier 3 signals with high directional confidence
    if tier == 3 and abs(prob_up - 0.5) > 0.2:  # High confidence
        tier = 4  # Premium tier
    
    return tier


def validate_signal_classification_function(df):
    """
    Validate the signal classification function performance
    """
    print("=== SIGNAL CLASSIFICATION VALIDATION ===")
    
    # Apply your classification function
    df['signal_tier_new'] = df.apply(even_simpler_classification, axis=1)

    #df['signal_tier_new'] = df['classification_result'].apply(lambda x: x['signal_tier'])
    
    #df['vol_regime'] = df['classification_result'].apply(lambda x: x['vol_regime'])
    #df['vol_multiplier'] = df['classification_result'].apply(lambda x: x['vol_multiplier'])
    
    future_returns = df['truth'].shift(-1)
    
    # Test performance by signal tier
    print("Performance by Signal Tier:")
    print(f"{'Tier':<8} {'Avg Return':<12} {'Hit Rate':<10} {'Count':<8} {'Sharpe':<8}")
    print("-" * 50)
    
    tier_performance = {}
    for tier in sorted(df['signal_tier_new'].unique()):
        if pd.isna(tier):
            continue
            
        tier_mask = df['signal_tier_new'] == tier
        tier_returns = future_returns[tier_mask]
        
        if len(tier_returns) > 10:
            avg_return = tier_returns.mean()
            hit_rate = (tier_returns > 0).mean()
            count = len(tier_returns)
            sharpe = avg_return / tier_returns.std() if tier_returns.std() > 0 else 0
            
            tier_performance[tier] = {
                'avg_return': avg_return,
                'hit_rate': hit_rate,
                'count': count,
                'sharpe': sharpe
            }
            
            print(f"{tier:<8.1f} {avg_return:<12.6f} {hit_rate:<10.2%} {count:<8} {sharpe:<8.4f}")
    
    # Test performance by volatility regime
    # print(f"\nPerformance by Volatility Regime:")
    # print(f"{'Regime':<15} {'Avg Return':<12} {'Hit Rate':<10} {'Count':<8}")
    # print("-" * 50)
    
    # for regime in df['vol_regime'].unique():
    #     if pd.isna(regime):
    #         continue
            
    #     regime_mask = df['vol_regime'] == regime
    #     regime_returns = future_returns[regime_mask]
        
    #     if len(regime_returns) > 10:
    #         avg_return = regime_returns.mean()
    #         hit_rate = (regime_returns > 0).mean()
    #         count = len(regime_returns)
            
    #         print(f"{regime:<15} {avg_return:<12.6f} {hit_rate:<10.2%} {count:<8}")
    
    # Test if higher tiers actually perform better
    print(f"\nTier Monotonicity Test:")
    tier_returns = []
    for tier in sorted([t for t in tier_performance.keys() if not pd.isna(t)]):
        tier_returns.append(tier_performance[tier]['avg_return'])
    
    if len(tier_returns) > 1:
        correlation = np.corrcoef(range(len(tier_returns)), tier_returns)[0,1]
        print(f"Correlation between tier and performance: {correlation:.4f}")
        if correlation > 0.5:
            print("Higher tiers generally perform better")
        elif correlation > 0.2:
            print("⚠️  Weak positive relationship between tiers and performance")
        else:
            print("No clear relationship between tiers and performance")
    
    return tier_performance

def test_signal_classification_edge_cases(df):
    """
    Test edge cases and potential issues
    """
    print(f"\n=== EDGE CASE ANALYSIS ===")
    
    # Check for extreme signal tiers
    df['signal_tier_new'] = df.apply(even_simpler_classification, axis=1)
    #df['signal_tier_new'] = df['classification_result'].apply(lambda x: x['signal_tier'])
    
    print(f"Signal Tier Statistics:")
    print(f"  Min: {df['signal_tier_new'].min():.2f}")
    print(f"  Max: {df['signal_tier_new'].max():.2f}")
    print(f"  Mean: {df['signal_tier_new'].mean():.2f}")
    print(f"  Std: {df['signal_tier_new'].std():.2f}")
    
    # Check for missing vol_decile issues
    missing_vol_decile = (df.get('vol_raw_decile', pd.Series([-1] * len(df))) == -1).sum()
    print(f"  Rows with missing vol_decile: {missing_vol_decile}")
    
    # Check tier distribution
    print(f"\nTier Distribution:")
    tier_counts = df['signal_tier_new'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df) * 100
        print(f"  Tier {tier:.1f}: {count:,} ({pct:.1f}%)")
    
    # Check for very high tiers (potential issues)
    high_tiers = df['signal_tier_new'] > 4.0
    if high_tiers.sum() > 0:
        print(f"  ⚠️  {high_tiers.sum()} signals with tier > 4.0 (might be too high)")

def compare_with_simple_classification(df):
    """
    Compare your complex classification with a simple version
    """
    print(f"\n=== COMPLEXITY vs SIMPLICITY COMPARISON ===")
    
    # Your complex classification
    df['signal_tier_complex'] = df.apply(even_simpler_classification, axis=1)
    #df['signal_tier_complex'] = df['classification_result'].apply(lambda x: x['signal_tier'])
    
    # Simple classification (just abs_q50 vs threshold)
    def simple_classification(row):
        abs_q50 = row.get("abs_q50", 0)
        signal_thresh = row.get("signal_thresh_adaptive", 0.01)
        
        if abs_q50 >= signal_thresh:
            return 3.0
        elif abs_q50 >= signal_thresh * 0.8:
            return 2.0
        elif abs_q50 >= signal_thresh * 0.6:
            return 1.0
        else:
            return 0.0
    
    df['signal_tier_simple'] = df.apply(simple_classification, axis=1)
    
    future_returns = df['truth'].shift(-1)
    
    # Compare performance
    complex_signals = df['signal_tier_complex'] >= 2.0
    simple_signals = df['signal_tier_simple'] >= 2.0
    
    complex_performance = {
        'avg_return': future_returns[complex_signals].mean(),
        'hit_rate': (future_returns[complex_signals] > 0).mean(),
        'count': complex_signals.sum(),
        'sharpe': future_returns[complex_signals].mean() / future_returns[complex_signals].std() if complex_signals.sum() > 10 else 0
    }
    
    simple_performance = {
        'avg_return': future_returns[simple_signals].mean(),
        'hit_rate': (future_returns[simple_signals] > 0).mean(),
        'count': simple_signals.sum(),
        'sharpe': future_returns[simple_signals].mean() / future_returns[simple_signals].std() if simple_signals.sum() > 10 else 0
    }
    
    print(f"Complex Classification Performance:")
    print(f"  Avg Return: {complex_performance['avg_return']:.6f}")
    print(f"  Hit Rate: {complex_performance['hit_rate']:.2%}")
    print(f"  Count: {complex_performance['count']:,}")
    print(f"  Sharpe: {complex_performance['sharpe']:.4f}")
    
    print(f"\nSimple Classification Performance:")
    print(f"  Avg Return: {simple_performance['avg_return']:.6f}")
    print(f"  Hit Rate: {simple_performance['hit_rate']:.2%}")
    print(f"  Count: {simple_performance['count']:,}")
    print(f"  Sharpe: {simple_performance['sharpe']:.4f}")
    
    # Calculate improvement
    return_improvement = complex_performance['avg_return'] - simple_performance['avg_return']
    sharpe_improvement = complex_performance['sharpe'] - simple_performance['sharpe']
    
    print(f"\nComplex vs Simple Improvements:")
    print(f"  Return: {return_improvement:+.6f}")
    print(f"  Sharpe: {sharpe_improvement:+.4f}")
    
    if sharpe_improvement > 0.01:
        print("Complex classification shows meaningful improvement")
    elif sharpe_improvement > 0:
        print("⚠️  Marginal improvement - consider if complexity is worth it")
    else:
        print("Simple classification performs as well or better")
    
    return complex_performance, simple_performance

# Main validation function
def run_signal_classification_validation(df):
    """
    Complete validation of signal classification function
    """
    print("🔬 SIGNAL CLASSIFICATION FUNCTION VALIDATION")
    print("=" * 55)
    
    # 1. Basic performance validation
    tier_performance = validate_signal_classification_function(df)
    
    # 2. Edge case testing
    test_signal_classification_edge_cases(df)
    
    # 3. Complexity comparison
    complex_perf, simple_perf = compare_with_simple_classification(df)
    
    return tier_performance, complex_perf, simple_perf

# Usage:
# tier_perf, complex_perf, simple_perf = run_signal_classification_validation(df)

if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")
    tier_perf, complex_perf, simple_perf = run_signal_classification_validation(df)
