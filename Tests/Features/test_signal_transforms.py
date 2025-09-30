
# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd

def validate_signal_transformations(df):
    """
    Test all signal transformation features for predictive value
    """
    import scipy.stats as stats
    from sklearn.metrics import mutual_info_score
    import numpy as np
    import pandas as pd
    
    # Features to test
    signal_features = [
        'signal_rel',           # Original
        'signal_rel_clipped',   # Clipped version
        'signal_tanh',          # Tanh transformation
        'signal_sigmoid',       # Sigmoid transformation  
        'signal_score',         # Weighted combination
    ]
    
    spread_features = [
        'spread',               # Original (we know this is weak from previous tests)
        'spread_rel',           # Relative to threshold
        'spread_rel_clipped',   # Clipped version
        'spread_tanh',          # Tanh transformation
        'spread_sigmoid',       # Sigmoid transformation
        'spread_score',         # Weighted combination
        'spread_tier',          # Quantile-based tier
    ]
    
    composite_features = [
        'tier_confidence',      # Combined signal + spread metric
    ]
    
    all_features = signal_features + spread_features + composite_features
    
    # Target variables
    future_returns = df['truth'].shift(-1)
    future_volatility = df['truth'].rolling(5).std().shift(-1)
    
    results = {}
    
    print("=== SIGNAL TRANSFORMATION VALIDATION ===")
    print(f"{'Feature':<20} {'Return_Corr':<12} {'Vol_Corr':<10} {'Mutual_Info':<12} {'Monotonic':<10} {'Range':<15}")
    print("-" * 85)
    
    for feature in all_features:
        if feature not in df.columns:
            print(f"{feature:<20} {'MISSING':<12}")
            continue
            
        feature_values = df[feature].dropna()
        if len(feature_values) == 0:
            continue
            
        # Basic correlations
        return_corr = feature_values.corr(future_returns)
        vol_corr = feature_values.corr(future_volatility)
        
        # Mutual information
        try:
            feature_bins = pd.qcut(feature_values, 5, labels=False, duplicates='drop')
            return_bins = pd.qcut(future_returns.dropna(), 5, labels=False, duplicates='drop')
            
            aligned_data = pd.DataFrame({
                'feature_bins': feature_bins,
                'return_bins': return_bins
            }).dropna()
            
            if len(aligned_data) > 10:
                mutual_info = mutual_info_score(aligned_data['feature_bins'], aligned_data['return_bins'])
            else:
                mutual_info = 0
        except:
            mutual_info = 0
        
        # Test for monotonic relationship (decile analysis)
        try:
            feature_deciles = pd.qcut(feature_values, 10, labels=False, duplicates='drop')
            decile_returns = pd.DataFrame({
                'decile': feature_deciles,
                'return': future_returns
            }).groupby('decile')['return'].mean()
            
            # Check if generally monotonic (allowing for some noise)
            monotonic_score = np.corrcoef(decile_returns.index, decile_returns.values)[0,1]
            is_monotonic = abs(monotonic_score) > 0.7
        except:
            is_monotonic = False
            monotonic_score = 0
        
        # Feature range
        feature_range = f"{feature_values.min():.3f} to {feature_values.max():.3f}"
        
        results[feature] = {
            'return_corr': return_corr,
            'vol_corr': vol_corr,
            'mutual_info': mutual_info,
            'monotonic': is_monotonic,
            'monotonic_score': monotonic_score,
            'range': feature_range
        }
        
        print(f"{feature:<20} {return_corr:<12.4f} {vol_corr:<10.4f} {mutual_info:<12.4f} {str(is_monotonic):<10} {feature_range:<15}")
    
    return results

def compare_transformation_effectiveness(df):
    """
    Compare different transformation approaches for the same base feature
    """
    print("\n=== TRANSFORMATION EFFECTIVENESS COMPARISON ===")
    
    # Signal transformations comparison
    signal_base = 'signal_rel'
    signal_transforms = ['signal_rel_clipped', 'signal_tanh', 'signal_sigmoid', 'signal_score']
    
    if signal_base in df.columns:
        print(f"\nSignal transformations (base: {signal_base}):")
        base_corr = df[signal_base].corr(df['truth'].shift(-1))
        print(f"  {signal_base:<20} (baseline): {base_corr:.4f}")
        
        for transform in signal_transforms:
            if transform in df.columns:
                transform_corr = df[transform].corr(df['truth'].shift(-1))
                improvement = transform_corr - base_corr
                print(f"  {transform:<20}: {transform_corr:.4f} (Δ{improvement:+.4f})")
    
    # Spread transformations comparison
    spread_base = 'spread'
    spread_transforms = ['spread_rel', 'spread_rel_clipped', 'spread_tanh', 'spread_sigmoid', 'spread_score']
    
    if spread_base in df.columns:
        print(f"\nSpread transformations (base: {spread_base}):")
        base_corr = df[spread_base].corr(df['truth'].shift(-1))
        print(f"  {spread_base:<20} (baseline): {base_corr:.4f}")
        
        for transform in spread_transforms:
            if transform in df.columns:
                transform_corr = df[transform].corr(df['truth'].shift(-1))
                improvement = transform_corr - base_corr
                print(f"  {transform:<20}: {transform_corr:.4f} (Δ{improvement:+.4f})")

def validate_composite_features(df):
    """
    Test the composite features (signal_score, spread_score, tier_confidence)
    """
    print("\n=== COMPOSITE FEATURE VALIDATION ===")
    
    future_returns = df['truth'].shift(-1)
    
    # Test signal_score components
    if all(col in df.columns for col in ['signal_rel', 'signal_tanh', 'signal_sigmoid', 'signal_score']):
        print("\nSignal Score Component Analysis:")
        components = ['signal_rel', 'signal_tanh', 'signal_sigmoid']
        component_corrs = {}
        
        for comp in components:
            corr = df[comp].corr(future_returns)
            component_corrs[comp] = corr
            print(f"  {comp}: {corr:.4f}")
        
        composite_corr = df['signal_score'].corr(future_returns)
        print(f"  signal_score (composite): {composite_corr:.4f}")
        
        # Check if composite is better than best individual component
        best_individual = max(component_corrs.values(), key=abs)
        improvement = abs(composite_corr) - abs(best_individual)
        print(f"  Improvement over best component: {improvement:+.4f}")
    
    # Test spread_score components
    if all(col in df.columns for col in ['spread_rel', 'spread_sigmoid', 'spread_score']):
        print("\nSpread Score Component Analysis:")
        spread_rel_corr = df['spread_rel'].corr(future_returns)
        spread_sigmoid_corr = df['spread_sigmoid'].corr(future_returns)
        spread_score_corr = df['spread_score'].corr(future_returns)
        
        print(f"  spread_rel: {spread_rel_corr:.4f}")
        print(f"  spread_sigmoid: {spread_sigmoid_corr:.4f}")
        print(f"  spread_score (composite): {spread_score_corr:.4f}")
    
    # Test tier_confidence
    if 'tier_confidence' in df.columns:
        print("\nTier Confidence Analysis:")
        tier_conf_corr = df['tier_confidence'].corr(future_returns)
        print(f"  tier_confidence: {tier_conf_corr:.4f}")
        
        # Compare to individual components if available
        if 'signal_tier' in df.columns:
            signal_tier_corr = df['signal_tier'].corr(future_returns)
            print(f"  signal_tier (component): {signal_tier_corr:.4f}")
        
        if 'spread_tier' in df.columns:
            spread_tier_corr = df['spread_tier'].corr(future_returns)
            print(f"  spread_tier (component): {spread_tier_corr:.4f}")

def test_feature_redundancy(df):
    """
    Test for redundancy between similar features
    """
    print("\n=== FEATURE REDUNDANCY ANALYSIS ===")
    
    # Signal feature correlations
    signal_features = ['signal_rel', 'signal_rel_clipped', 'signal_tanh', 'signal_sigmoid', 'signal_score']
    signal_features = [f for f in signal_features if f in df.columns]
    
    if len(signal_features) > 1:
        print("\nSignal Feature Intercorrelations:")
        signal_corr_matrix = df[signal_features].corr()
        
        for i, feat1 in enumerate(signal_features):
            for j, feat2 in enumerate(signal_features):
                if i < j:  # Only show upper triangle
                    corr = signal_corr_matrix.loc[feat1, feat2]
                    redundancy = "HIGH" if abs(corr) > 0.95 else "MEDIUM" if abs(corr) > 0.8 else "LOW"
                    print(f"  {feat1} vs {feat2}: {corr:.4f} ({redundancy})")
    
    # Spread feature correlations
    spread_features = ['spread', 'spread_rel', 'spread_tanh', 'spread_sigmoid', 'spread_score']
    spread_features = [f for f in spread_features if f in df.columns]
    
    if len(spread_features) > 1:
        print("\nSpread Feature Intercorrelations:")
        spread_corr_matrix = df[spread_features].corr()
        
        for i, feat1 in enumerate(spread_features):
            for j, feat2 in enumerate(spread_features):
                if i < j:
                    corr = spread_corr_matrix.loc[feat1, feat2]
                    redundancy = "HIGH" if abs(corr) > 0.95 else "MEDIUM" if abs(corr) > 0.8 else "LOW"
                    print(f"  {feat1} vs {feat2}: {corr:.4f} ({redundancy})")

def recommend_feature_selection(results):
    """
    Provide recommendations based on validation results
    """
    print("\n=== FEATURE SELECTION RECOMMENDATIONS ===")
    
    # Sort features by absolute return correlation
    sorted_features = sorted(results.items(), key=lambda x: abs(x[1]['return_corr']), reverse=True)
    
    print("\nTop performing features (by return correlation):")
    for i, (feature, metrics) in enumerate(sorted_features[:10]):
        print(f"  {i+1:2d}. {feature:<20}: {metrics['return_corr']:+.4f}")
    
    print("\nRecommendations:")
    
    # Signal features
    signal_features = {k: v for k, v in results.items() if k.startswith('signal_')}
    if signal_features:
        best_signal = max(signal_features.items(), key=lambda x: abs(x[1]['return_corr']))
        print(f"  Best signal feature: {best_signal[0]} (corr: {best_signal[1]['return_corr']:+.4f})")
    
    # Spread features  
    spread_features = {k: v for k, v in results.items() if k.startswith('spread_')}
    if spread_features:
        best_spread = max(spread_features.items(), key=lambda x: abs(x[1]['return_corr']))
        print(f"  ⚠️  Best spread feature: {best_spread[0]} (corr: {best_spread[1]['return_corr']:+.4f})")
        if abs(best_spread[1]['return_corr']) < 0.01:
            print(f"     (Still very weak - consider removing spread features entirely)")
    
    # Composite features
    if 'tier_confidence' in results:
        tier_corr = results['tier_confidence']['return_corr']
        print(f"  Tier confidence: {tier_corr:+.4f}")
        if abs(tier_corr) > 0.02:
            print(f"     Keep - shows meaningful correlation")
        else:
            print(f"     Consider simplifying - weak correlation")
    
    # Features to potentially remove
    weak_features = [k for k, v in results.items() if abs(v['return_corr']) < 0.005]
    if weak_features:
        print(f"\n  🗑️  Consider removing (very weak correlation):")
        for feat in weak_features:
            print(f"     - {feat}: {results[feat]['return_corr']:+.4f}")

# Main validation function
def run_complete_signal_transform_validation(df):
    """
    Run all validation tests for signal transformations
    """
    print("🔬 COMPREHENSIVE SIGNAL TRANSFORMATION VALIDATION")
    print("=" * 60)
    
    # 1. Basic validation
    results = validate_signal_transformations(df)
    
    # 2. Transformation comparison
    compare_transformation_effectiveness(df)
    
    # 3. Composite feature validation
    validate_composite_features(df)
    
    # 4. Redundancy analysis
    test_feature_redundancy(df)
    
    # 5. Recommendations
    recommend_feature_selection(results)
    
    return results

# Usage:
# results = run_complete_signal_transform_validation(df)


if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")
    results = run_complete_signal_transform_validation(df)
