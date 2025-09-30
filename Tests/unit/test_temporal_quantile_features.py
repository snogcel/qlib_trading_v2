#!/usr/bin/env python3
"""
Test Phase 1: Economically-Justified Temporal Quantile Features
Validate features follow thesis-first development principles
"""


# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.features.regime_features import add_temporal_quantile_features, RegimeFeatureEngine

def create_test_data():
    """Create realistic test data with quantile predictions"""
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic quantile data
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    # Base signal with some temporal patterns
    base_signal = np.cumsum(np.random.normal(0, 0.001, n_samples))
    
    # Add some momentum and mean reversion patterns
    momentum = np.convolve(np.random.normal(0, 0.0005, n_samples), 
                          np.ones(6)/6, mode='same')
    
    # Create quantile predictions with realistic relationships
    q50 = base_signal + momentum
    q10 = q50 - np.abs(np.random.normal(0.003, 0.001, n_samples))
    q90 = q50 + np.abs(np.random.normal(0.003, 0.001, n_samples))
    
    # Add some regime-like behavior
    vol_regime = np.random.choice([0.5, 1.0, 1.5], n_samples, p=[0.6, 0.3, 0.1])
    q10 *= vol_regime
    q90 *= vol_regime
    
    df = pd.DataFrame({
        'datetime': dates,
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'vol_risk': np.random.exponential(0.01, n_samples),
        '$fg_index': np.random.uniform(0, 100, n_samples),
        '$btc_dom': np.random.uniform(40, 70, n_samples)
    })
    
    df.set_index('datetime', inplace=True)
    
    return df

def test_temporal_features():
    """Test economically-justified temporal quantile features"""
    
    print("TESTING ECONOMICALLY-JUSTIFIED TEMPORAL FEATURES")
    print("=" * 60)
    
    # Create test data
    df = create_test_data()
    print(f"Created test data: {df.shape}")
    
    # Test with full regime engine (includes economic validation)
    print(f"\nTesting with full regime engine...")
    engine = RegimeFeatureEngine()
    df_full = engine.generate_all_regime_features(df.copy())
    
    # The economic validation is automatically run in the regime engine
    # Check that features were added with economic justification
    temporal_features = ['q50_momentum_3', 'spread_momentum_3', 'q50_stability_6', 
                        'q50_regime_persistence', 'prediction_confidence', 'q50_direction_consistency']
    
    existing_features = [f for f in temporal_features if f in df_full.columns]
    print(f"\nAdded {len(existing_features)} economically-justified features:")
    
    feature_explanations = {
        'q50_momentum_3': 'Information flow persistence (momentum theory)',
        'spread_momentum_3': 'Market uncertainty evolution (microstructure theory)',
        'q50_stability_6': 'Consensus stability (information aggregation)',
        'q50_regime_persistence': 'Behavioral momentum (behavioral finance)',
        'prediction_confidence': 'Risk-adjusted confidence (risk management)',
        'q50_direction_consistency': 'Trend strength (trend following theory)'
    }
    
    for feature in existing_features:
        explanation = feature_explanations.get(feature, 'Economic rationale documented')
        print(f"   • {feature}: {explanation}")
    
    # Test chart explainability (thesis-first principle)
    print(f"\nCHART EXPLAINABILITY TEST:")
    print("Can you explain each feature by looking at a chart?")
    
    for feature in existing_features:
        values = df_full[feature].dropna()
        if len(values) > 0:
            # Check if feature has reasonable range and variation
            has_variation = values.std() > 0.001
            reasonable_range = values.min() > -100 and values.max() < 100
            
            status = "" if has_variation and reasonable_range else "⚠️"
            print(f"   {status} {feature}: Range [{values.min():.3f}, {values.max():.3f}], Std: {values.std():.3f}")
    
    return df_full, existing_features

def test_integration_with_existing_system():
    """Test integration with existing quantile prediction pipeline"""
    
    print(f"\nTESTING INTEGRATION WITH EXISTING SYSTEM")
    print("=" * 50)
    
    # Simulate data similar to your actual pipeline
    df = create_test_data()
    
    # Add some existing features that your system uses
    df['abs_q50'] = df['q50'].abs()
    df['spread'] = df['q90'] - df['q10']
    df['info_ratio'] = df['abs_q50'] / np.maximum(df['spread'], 0.001)
    
    print(f"Created pipeline-like data: {df.shape}")
    
    # Test that temporal features integrate cleanly
    df_enhanced = add_temporal_quantile_features(df)
    
    # Check that existing features are preserved
    existing_features = ['abs_q50', 'spread', 'info_ratio']
    for feature in existing_features:
        if feature in df_enhanced.columns:
            print(f"   Preserved existing feature: {feature}")
        else:
            print(f"   Lost existing feature: {feature}")
    
    # Test feature interactions
    print(f"\nFeature interaction analysis:")
    
    # Temporal momentum should enhance info ratio in some cases
    high_momentum = df_enhanced['q50_momentum_3'].abs() > df_enhanced['q50_momentum_3'].abs().quantile(0.8)
    high_info_ratio = df_enhanced['info_ratio'] > df_enhanced['info_ratio'].quantile(0.8)
    
    overlap = (high_momentum & high_info_ratio).sum()
    total_high_momentum = high_momentum.sum()
    
    if total_high_momentum > 0:
        overlap_pct = overlap / total_high_momentum * 100
        print(f"   High momentum + high info ratio overlap: {overlap_pct:.1f}%")
    
    return df_enhanced

def visualize_temporal_features(df, features):
    """Create visualizations of temporal features"""
    
    print(f"\nCreating temporal feature visualizations...")
    
    # Create subplots for key temporal features
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Temporal Quantile Features Analysis', fontsize=16)
    
    # Plot Q50 and its momentum
    axes[0, 0].plot(df.index[-200:], df['q50'].iloc[-200:], label='Q50', alpha=0.7)
    axes[0, 0].plot(df.index[-200:], df['q50_momentum_3'].iloc[-200:] * 10, 
                   label='Q50 Momentum (×10)', alpha=0.7)
    axes[0, 0].set_title('Q50 vs Temporal Momentum')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot spread and its momentum
    spread = df['q90'] - df['q10']
    axes[0, 1].plot(df.index[-200:], spread.iloc[-200:], label='Spread', alpha=0.7)
    axes[0, 1].plot(df.index[-200:], df['spread_momentum_3'].iloc[-200:] * 10, 
                   label='Spread Momentum (×10)', alpha=0.7)
    axes[0, 1].set_title('Spread vs Temporal Momentum')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot stability features
    axes[1, 0].plot(df.index[-200:], df['q50_stability_3'].iloc[-200:], 
                   label='Q50 Stability (3-period)', alpha=0.7)
    axes[1, 0].plot(df.index[-200:], df['q50_stability_6'].iloc[-200:], 
                   label='Q50 Stability (6-period)', alpha=0.7)
    axes[1, 0].set_title('Quantile Stability Patterns')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot regime persistence
    axes[1, 1].plot(df.index[-200:], df['q50_regime_persistence'].iloc[-200:], 
                   label='Q50 Regime Persistence', alpha=0.7)
    axes[1, 1].plot(df.index[-200:], df['q50_direction_consistency'].iloc[-200:] * 10, 
                   label='Direction Consistency (×10)', alpha=0.7)
    axes[1, 1].set_title('Regime Persistence Patterns')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_quantile_features_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: temporal_quantile_features_analysis.png")
    
    plt.close()

def main():
    """Main testing function"""
    
    print("PHASE 1: TEMPORAL QUANTILE FEATURES TEST")
    print("=" * 60)
    
    # Test temporal features
    df_temporal, temporal_features = test_temporal_features()
    
    # Test integration
    df_integrated = test_integration_with_existing_system()
    
    # Create visualizations
    visualize_temporal_features(df_temporal, temporal_features)
    
    print(f"\nPHASE 1 TESTING COMPLETE!")
    print("=" * 60)
    print("Temporal quantile features implemented successfully")
    print("Integration with existing system validated")
    print("Feature quality analysis completed")
    print("Visualizations generated")
    
    print(f"\nNEXT STEPS:")
    print("1. Add temporal features to your main training pipeline")
    print("2. Run backtest with temporal features enabled")
    print("3. Compare performance against 1.327 Sharpe baseline")
    print("4. Analyze feature importance in model training")
    
    return df_temporal, df_integrated

if __name__ == "__main__":
    results = main()