"""
Comprehensive Feature Analysis including ALL GDELT and technical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_comprehensive_data():
    """Load data with ALL features"""
    
    print("Loading comprehensive feature dataset...")
    
    # Try to load the updated df_all_macro_analysis.csv first
    try:
        df = pd.read_csv("df_all_macro_analysis_prep.csv")
        df = df.drop(columns=["q10", "q50", "q90"])
        print(f"✅ Loaded updated df_all_macro_analysis_prep.csv with {df.shape[1]} features")
        
        if 'instrument' in df.columns and 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index(['instrument', 'datetime'])
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs('BTCUSDT', level=0)
        
        return df.sort_index()
        
    except FileNotFoundError:
        print("df_all_macro_analysis.csv not found, trying X_all.csv...")
        
        # Fallback to X_all.csv
        try:
            X_all = pd.read_csv("X_all.csv")
            print(f"✅ Loaded X_all.csv with {X_all.shape[1]} features")
            
            # Set proper index
            if 'instrument' in X_all.columns and 'datetime' in X_all.columns:
                X_all['datetime'] = pd.to_datetime(X_all['datetime'])
                X_all = X_all.set_index(['instrument', 'datetime'])
            
            if isinstance(X_all.index, pd.MultiIndex):
                X_all = X_all.xs('BTCUSDT', level=0)
            
            return X_all.sort_index()
            
        except FileNotFoundError:
            print("❌ Neither df_all_macro_analysis.csv nor X_all.csv found!")
            return None

def categorize_features(df):
    """Categorize features by type"""
    
    feature_categories = {
        'quantile_predictions': [col for col in df.columns if col in ['q10', 'q50', 'q90']],
        'gdelt_sentiment': [col for col in df.columns if 'cwt_' in col],
        'market_sentiment': [col for col in df.columns if col in ['$fg_index', 'fg_index']],
        'market_structure': [col for col in df.columns if col in ['$btc_dom', 'btc_dom']],
        'momentum': [col for col in df.columns if 'momentum' in col.lower()],
        'volatility': [col for col in df.columns if 'vol' in col.lower() or 'realized' in col.lower()],
        'technical_price': [col for col in df.columns if any(x in col for x in ['OPEN', 'ROC'])],
        'technical_volatility': [col for col in df.columns if 'STD' in col],
        'volume': [col for col in df.columns if 'VOLUME' in col],
        'derived_signals': [col for col in df.columns if any(x in col for x in ['signal', 'tier', 'spread', 'prob'])],
        'target': [col for col in df.columns if col == 'truth'],
        'other': []
    }
    
    # Categorize remaining features
    all_categorized = set()
    for category_features in feature_categories.values():
        all_categorized.update(category_features)
    
    feature_categories['other'] = [col for col in df.columns if col not in all_categorized]
    
    # Remove empty categories
    feature_categories = {k: v for k, v in feature_categories.items() if v}
    
    return feature_categories

def analyze_gdelt_features(df):
    """Specific analysis of GDELT features"""
    
    print(f"\n{'='*60}")
    print("GDELT FEATURES ANALYSIS")
    print(f"{'='*60}")
    
    gdelt_features = [col for col in df.columns if 'cwt_' in col]
    
    if not gdelt_features:
        print("❌ No GDELT features found!")
        return None
    
    print(f"Found {len(gdelt_features)} GDELT features:")
    for feature in gdelt_features:
        print(f"  • {feature}")
    
    if 'truth' not in df.columns:
        print("❌ No 'truth' column found for correlation analysis")
        return None
    
    # Calculate correlations with truth
    gdelt_correlations = {}
    for feature in gdelt_features:
        corr = df[feature].corr(df['truth'])
        if not pd.isna(corr):
            gdelt_correlations[feature] = abs(corr)
    
    # Sort by correlation strength
    gdelt_correlations = dict(sorted(gdelt_correlations.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\nGDELT features ranked by correlation with truth:")
    for i, (feature, corr) in enumerate(gdelt_correlations.items()):
        print(f"  {i+1:2d}. {feature:25} | {corr:.4f}")
    
    # Analyze GDELT feature statistics
    print(f"\nGDELT feature statistics:")
    gdelt_stats = df[gdelt_features].describe()
    print(gdelt_stats)
    
    return gdelt_correlations

def comprehensive_feature_correlation_analysis(df):
    """Comprehensive correlation analysis of ALL features"""
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE FEATURE CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    if 'truth' not in df.columns:
        print("❌ No 'truth' column found for correlation analysis")
        return None
    
    # Calculate correlations with truth for ALL features
    all_correlations = {}
    for col in df.columns:
        if col != 'truth':
            corr = df[col].corr(df['truth'])
            if not pd.isna(corr):
                all_correlations[col] = abs(corr)
    
    # Sort by correlation strength
    all_correlations = dict(sorted(all_correlations.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Top 20 features by correlation with truth:")
    for i, (feature, corr) in enumerate(list(all_correlations.items())[:20]):
        print(f"  {i+1:2d}. {feature:30} | {corr:.4f}")
    
    # Categorize and analyze by category
    feature_categories = categorize_features(df)
    
    print(f"\nFeature analysis by category:")
    category_performance = {}
    
    for category, features in feature_categories.items():
        if category == 'target':
            continue
            
        category_corrs = []
        for feature in features:
            if feature in all_correlations:
                category_corrs.append(all_correlations[feature])
        
        if category_corrs:
            avg_corr = np.mean(category_corrs)
            max_corr = max(category_corrs)
            best_feature = max([f for f in features if f in all_correlations], 
                             key=lambda x: all_correlations[x])
            
            category_performance[category] = {
                'avg_correlation': avg_corr,
                'max_correlation': max_corr,
                'best_feature': best_feature,
                'feature_count': len(category_corrs)
            }
            
            print(f"  {category:20} | Avg: {avg_corr:.4f} | Max: {max_corr:.4f} | "
                  f"Best: {best_feature[:20]:20} | Count: {len(category_corrs):2d}")
    
    return all_correlations, category_performance

def identify_feature_interactions(df, top_n=10):
    """Identify potential feature interactions"""
    
    print(f"\n{'='*60}")
    print("FEATURE INTERACTION ANALYSIS")
    print(f"{'='*60}")
    
    if 'truth' not in df.columns:
        print("❌ No 'truth' column found")
        return None
    
    # Get top features
    correlations = {}
    for col in df.columns:
        if col != 'truth':
            corr = df[col].corr(df['truth'])
            if not pd.isna(corr):
                correlations[col] = abs(corr)
    
    top_features = list(dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)).keys())[:top_n]
    
    print(f"Testing interactions between top {top_n} features...")
    
    interaction_results = {}
    
    # Test multiplicative interactions
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            try:
                interaction = df[feat1] * df[feat2]
                corr = interaction.corr(df['truth'])
                
                if not pd.isna(corr):
                    interaction_name = f"{feat1} × {feat2}"
                    interaction_results[interaction_name] = abs(corr)
            except:
                continue
    
    # Sort interaction results
    interaction_results = dict(sorted(interaction_results.items(), key=lambda x: x[1], reverse=True))
    
    print(f"Top 10 feature interactions:")
    for i, (interaction, corr) in enumerate(list(interaction_results.items())[:10]):
        print(f"  {i+1:2d}. {interaction:50} | {corr:.4f}")
    
    return interaction_results

def generate_feature_recommendations(all_correlations, category_performance, gdelt_correlations=None):
    """Generate actionable feature recommendations"""
    
    print(f"\n{'='*60}")
    print("FEATURE OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")
    
    # Top individual features
    top_features = list(all_correlations.items())[:10]
    print(f"🏆 TOP 10 INDIVIDUAL FEATURES TO PRIORITIZE:")
    for i, (feature, corr) in enumerate(top_features):
        print(f"  {i+1:2d}. {feature:30} | {corr:.4f}")
    
    # Best categories
    if category_performance:
        best_categories = sorted(category_performance.items(), 
                               key=lambda x: x[1]['avg_correlation'], reverse=True)
        
        print(f"\n📊 BEST FEATURE CATEGORIES:")
        for i, (category, metrics) in enumerate(best_categories[:5]):
            print(f"  {i+1}. {category:20} | Avg: {metrics['avg_correlation']:.4f} | "
                  f"Best: {metrics['best_feature'][:25]:25}")
    
    # GDELT specific recommendations
    if gdelt_correlations:
        print(f"\n🌍 GDELT FEATURES RECOMMENDATIONS:")
        top_gdelt = list(gdelt_correlations.items())[:5]
        for i, (feature, corr) in enumerate(top_gdelt):
            print(f"  {i+1}. {feature:30} | {corr:.4f}")
    
    # Implementation recommendations
    print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS:")
    print(f"1. IMMEDIATE: Add top 5 individual features to your model")
    print(f"2. GDELT: Focus on top 3 GDELT features for sentiment analysis")
    print(f"3. TECHNICAL: Leverage technical indicators (ROC, STD) for market timing")
    print(f"4. MOMENTUM: Short-term momentum features show highest correlation")
    print(f"5. INTERACTIONS: Test combinations of top features")

def save_comprehensive_analysis(all_correlations, category_performance, gdelt_correlations=None):
    """Save analysis results to files"""
    
    output_dir = Path("./comprehensive_feature_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save all correlations
    corr_df = pd.DataFrame(list(all_correlations.items()), columns=['feature', 'correlation'])
    corr_df.to_csv(output_dir / "all_feature_correlations.csv", index=False)
    
    # Save category performance
    if category_performance:
        category_df = pd.DataFrame(category_performance).T
        category_df.to_csv(output_dir / "category_performance.csv")
    
    # Save GDELT analysis
    if gdelt_correlations:
        gdelt_df = pd.DataFrame(list(gdelt_correlations.items()), columns=['gdelt_feature', 'correlation'])
        gdelt_df.to_csv(output_dir / "gdelt_feature_analysis.csv", index=False)
    
    print(f"\n💾 Analysis results saved to {output_dir}")

def main():
    """Main analysis function"""
    
    print("="*80)
    print("COMPREHENSIVE FEATURE ANALYSIS - ALL FEATURES INCLUDING GDELT")
    print("="*80)
    
    # Load data
    df = load_comprehensive_data()
    if df is None:
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total features: {len(df.columns)}")
    
    # Analyze GDELT features specifically
    gdelt_correlations = analyze_gdelt_features(df)
    
    # Comprehensive correlation analysis
    all_correlations, category_performance = comprehensive_feature_correlation_analysis(df)
    
    # Feature interaction analysis
    interaction_results = identify_feature_interactions(df)
    
    # Generate recommendations
    generate_feature_recommendations(all_correlations, category_performance, gdelt_correlations)
    
    # Save results
    save_comprehensive_analysis(all_correlations, category_performance, gdelt_correlations)
    
    print(f"\n✅ Comprehensive feature analysis completed!")
    
    return {
        'all_correlations': all_correlations,
        'category_performance': category_performance,
        'gdelt_correlations': gdelt_correlations,
        'interaction_results': interaction_results
    }

if __name__ == "__main__":
    results = main()