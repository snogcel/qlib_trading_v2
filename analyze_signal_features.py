#!/usr/bin/env python3
"""
Analyze signal_strength and signal_tier features to understand their implementation,
usage patterns, and performance impact.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_signal_strength():
    """Analyze signal_strength feature implementation and usage"""
    print("🔍 SIGNAL_STRENGTH ANALYSIS")
    print("=" * 50)
    
    # From code analysis, signal_strength appears to be:
    print("📋 IMPLEMENTATION PATTERNS:")
    print("1. Basic: signal_strength = abs(q50)")
    print("2. Threshold-based: signal_strength = abs_q50 / signal_thresh")
    print("3. Enhanced: signal_strength = abs_q50 * enhanced_info_ratio / threshold")
    print("4. Volatility-adjusted: Used in position sizing calculations")
    
    print("\n📊 USAGE CONTEXTS:")
    print("- Position sizing: base_position = signal_strength * scale_factor")
    print("- Signal quality assessment: Combined with confidence metrics")
    print("- Threshold decisions: Compared against minimum strength requirements")
    print("- Risk management: Integrated with volatility adjustments")
    
    print("\n🎯 BUCKETING LOGIC:")
    print("- Very_Low: < 0.3")
    print("- Low: 0.3-0.5")
    print("- Medium: 0.5-0.7") 
    print("- High: 0.7-0.9")
    print("- Very_High: >= 0.9")
    
    return {
        'type': 'Magnitude-based confidence metric',
        'primary_calculation': 'abs(q50)',
        'enhanced_calculation': 'abs_q50 * enhanced_info_ratio / threshold',
        'usage': ['position_sizing', 'signal_quality', 'risk_management'],
        'status': 'NEEDS_STANDARDIZATION'
    }

def analyze_signal_tier():
    """Analyze signal_tier feature implementation and usage"""
    print("\n🔍 SIGNAL_TIER ANALYSIS")
    print("=" * 50)
    
    print("📋 IMPLEMENTATION PATTERNS:")
    print("1. Classification-based: Numeric tiers (0-3) or letter grades (A-D)")
    print("2. ML-based: RandomForest model predicting optimal tiers")
    print("3. Rule-based: Complex logic combining multiple factors")
    print("4. Confidence mapping: Tiers converted to confidence scores")
    
    print("\n📊 TIER SYSTEMS:")
    print("Numeric System (0=best, 3=worst):")
    print("- 0: 10.0 confidence")
    print("- 1: 7.0 confidence") 
    print("- 2: 5.0 confidence")
    print("- 3: 3.0 confidence")
    
    print("\nLetter System:")
    print("- A: 10.0 confidence")
    print("- B: 7.0 confidence")
    print("- C: 5.0 confidence") 
    print("- D: 3.0 confidence")
    
    print("\n🎯 USAGE CONTEXTS:")
    print("- Signal filtering: Higher tiers get priority")
    print("- Position sizing: Tier confidence affects size")
    print("- Risk management: Lower tiers get reduced exposure")
    print("- Performance tracking: Tier-based analytics")
    
    return {
        'type': 'Signal quality classification system',
        'formats': ['numeric_0_3', 'letter_A_D'],
        'confidence_mapping': {0: 10.0, 1: 7.0, 2: 5.0, 3: 3.0},
        'usage': ['signal_filtering', 'position_sizing', 'risk_management'],
        'status': 'MULTIPLE_IMPLEMENTATIONS'
    }

def load_and_analyze_data():
    """Load actual data to analyze these features if available"""
    print("\n🔍 DATA ANALYSIS")
    print("=" * 50)
    
    # Look for recent backtest results or signal analysis files
    data_files = [
        'signal_analysis_pivot.csv',
        'validated_backtest_results/signal_analysis.csv',
        'data3/macro_features.pkl'
    ]
    
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"📁 Found data file: {file_path}")
            try:
                if file_path.endswith('.pkl'):
                    df = pd.read_pickle(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                print(f"   Shape: {df.shape}")
                
                # Check for signal_strength
                if 'signal_strength' in df.columns:
                    strength_stats = df['signal_strength'].describe()
                    print(f"   signal_strength stats:")
                    print(f"     Mean: {strength_stats['mean']:.4f}")
                    print(f"     Std: {strength_stats['std']:.4f}")
                    print(f"     Range: [{strength_stats['min']:.4f}, {strength_stats['max']:.4f}]")
                
                # Check for signal_tier
                if 'signal_tier' in df.columns:
                    tier_counts = df['signal_tier'].value_counts().sort_index()
                    print(f"   signal_tier distribution:")
                    for tier, count in tier_counts.head(10).items():
                        pct = count / len(df) * 100
                        print(f"     {tier}: {count} ({pct:.1f}%)")
                
                return df
                
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
    
    print("❌ No data files found for analysis")
    return None

def generate_recommendations(signal_strength_info, signal_tier_info, df=None):
    """Generate recommendations for feature standardization"""
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    
    print("📋 SIGNAL_STRENGTH STANDARDIZATION:")
    print("1. ✅ Standardize on: signal_strength = abs(q50) * info_ratio_multiplier")
    print("2. ✅ Use consistent bucketing: [0.3, 0.5, 0.7, 0.9] thresholds")
    print("3. ✅ Integrate with position sizing: base_position = signal_strength * scale")
    print("4. ⚠️  Document variance vs standard deviation usage")
    
    print("\n📋 SIGNAL_TIER STANDARDIZATION:")
    print("1. ✅ Choose single format: Recommend numeric 0-3 system")
    print("2. ✅ Standardize confidence mapping: {0: 10.0, 1: 7.0, 2: 5.0, 3: 3.0}")
    print("3. ✅ Implement ML-based tier prediction as primary method")
    print("4. ⚠️  Deprecate rule-based classification methods")
    
    print("\n📋 INTEGRATION IMPROVEMENTS:")
    print("1. 🔄 Create unified signal quality score: strength * tier_confidence")
    print("2. 🔄 Implement feature validation tests")
    print("3. 🔄 Add performance tracking by feature values")
    print("4. 🔄 Document feature lifecycle and dependencies")
    
    if df is not None:
        print("\n📊 DATA-DRIVEN INSIGHTS:")
        
        # Analyze correlation with returns if available
        return_cols = [col for col in df.columns if 'return' in col.lower()]
        if return_cols and 'signal_strength' in df.columns:
            corr = df['signal_strength'].corr(df[return_cols[0]])
            print(f"- signal_strength correlation with returns: {corr:.4f}")
        
        if return_cols and 'signal_tier' in df.columns:
            # Analyze tier performance
            tier_performance = df.groupby('signal_tier')[return_cols[0]].agg(['mean', 'std', 'count'])
            print("- Tier performance analysis:")
            for tier, stats in tier_performance.iterrows():
                print(f"  Tier {tier}: Return={stats['mean']:.4f}, Std={stats['std']:.4f}, Count={stats['count']}")

def main():
    """Main analysis function"""
    print("🚀 SIGNAL FEATURE ANALYSIS")
    print("=" * 60)
    
    # Analyze implementations
    signal_strength_info = analyze_signal_strength()
    signal_tier_info = analyze_signal_tier()
    
    # Load and analyze data
    df = load_and_analyze_data()
    
    # Generate recommendations
    generate_recommendations(signal_strength_info, signal_tier_info, df)
    
    print("\n✅ Analysis complete! Check recommendations above.")
    
    return {
        'signal_strength': signal_strength_info,
        'signal_tier': signal_tier_info,
        'data_available': df is not None
    }

if __name__ == "__main__":
    results = main()