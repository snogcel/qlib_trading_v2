#!/usr/bin/env python3
"""
Analysis of volatility features: vol_raw_momentum, vol_risk, and vol_raw_decile
Comparing old vs new implementations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_and_analyze_volatility_features():
    """Load and analyze the volatility feature comparison data"""
    
    # Load the CSV data
    df = pd.read_csv('Research/v2_model/feature_QA_1.csv')
    
    print("=== VOLATILITY FEATURES ANALYSIS ===\n")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean column names
    df.columns = ['vol_raw_momentum_old', 'vol_raw_momentum_new', 
                  'vol_risk_old', 'vol_risk_new', 
                  'vol_raw_decile_old', 'vol_raw_decile_new']
    
    print("\n=== DATA OVERVIEW ===")
    print(df.describe())
    
    # Check for missing values
    print("\n=== MISSING VALUES ===")
    missing_counts = df.isnull().sum()
    print(missing_counts)
    
    # Calculate correlations between old and new implementations
    print("\n=== CORRELATIONS (Old vs New) ===")
    
    # vol_raw_momentum correlation
    valid_momentum = df.dropna(subset=['vol_raw_momentum_old', 'vol_raw_momentum_new'])
    if len(valid_momentum) > 0:
        momentum_corr = valid_momentum['vol_raw_momentum_old'].corr(valid_momentum['vol_raw_momentum_new'])
        print(f"vol_raw_momentum correlation: {momentum_corr:.4f}")
    
    # vol_risk correlation  
    valid_risk = df.dropna(subset=['vol_risk_old', 'vol_risk_new'])
    if len(valid_risk) > 0:
        risk_corr = valid_risk['vol_risk_old'].corr(valid_risk['vol_risk_new'])
        print(f"vol_risk correlation: {risk_corr:.4f}")
    
    # vol_raw_decile correlation
    valid_decile = df.dropna(subset=['vol_raw_decile_old', 'vol_raw_decile_new'])
    if len(valid_decile) > 0:
        decile_corr = valid_decile['vol_raw_decile_old'].corr(valid_decile['vol_raw_decile_new'])
        print(f"vol_raw_decile correlation: {decile_corr:.4f}")
    
    # Analyze the differences
    print("\n=== FEATURE DIFFERENCES ANALYSIS ===")
    
    # vol_raw_momentum differences
    if len(valid_momentum) > 0:
        momentum_diff = valid_momentum['vol_raw_momentum_new'] - valid_momentum['vol_raw_momentum_old']
        print(f"\nvol_raw_momentum differences:")
        print(f"  Mean difference: {momentum_diff.mean():.6f}")
        print(f"  Std difference: {momentum_diff.std():.6f}")
        print(f"  Min difference: {momentum_diff.min():.6f}")
        print(f"  Max difference: {momentum_diff.max():.6f}")
        print(f"  Median difference: {momentum_diff.median():.6f}")
    
    # vol_risk differences
    if len(valid_risk) > 0:
        risk_diff = valid_risk['vol_risk_new'] - valid_risk['vol_risk_old']
        print(f"\nvol_risk differences:")
        print(f"  Mean difference: {risk_diff.mean():.6f}")
        print(f"  Std difference: {risk_diff.std():.6f}")
        print(f"  Min difference: {risk_diff.min():.6f}")
        print(f"  Max difference: {risk_diff.max():.6f}")
        print(f"  Median difference: {risk_diff.median():.6f}")
    
    # vol_raw_decile differences
    if len(valid_decile) > 0:
        decile_diff = valid_decile['vol_raw_decile_new'] - valid_decile['vol_raw_decile_old']
        print(f"\nvol_raw_decile differences:")
        print(f"  Mean difference: {decile_diff.mean():.6f}")
        print(f"  Std difference: {decile_diff.std():.6f}")
        print(f"  Min difference: {decile_diff.min():.6f}")
        print(f"  Max difference: {decile_diff.max():.6f}")
        print(f"  Median difference: {decile_diff.median():.6f}")
    
    # Statistical significance tests
    print("\n=== STATISTICAL TESTS ===")
    
    if len(valid_momentum) > 0:
        t_stat, p_val = stats.ttest_rel(valid_momentum['vol_raw_momentum_old'], 
                                       valid_momentum['vol_raw_momentum_new'])
        print(f"vol_raw_momentum paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    
    if len(valid_risk) > 0:
        t_stat, p_val = stats.ttest_rel(valid_risk['vol_risk_old'], 
                                       valid_risk['vol_risk_new'])
        print(f"vol_risk paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    
    if len(valid_decile) > 0:
        t_stat, p_val = stats.ttest_rel(valid_decile['vol_raw_decile_old'], 
                                       valid_decile['vol_raw_decile_new'])
        print(f"vol_raw_decile paired t-test: t={t_stat:.4f}, p={p_val:.6f}")
    
    return df, valid_momentum, valid_risk, valid_decile

def create_comparison_plots(df, valid_momentum, valid_risk, valid_decile):
    """Create comparison plots for the volatility features"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Volatility Features: Old vs New Implementation Comparison', fontsize=16)
    
    # vol_raw_momentum scatter plot
    if len(valid_momentum) > 0:
        axes[0, 0].scatter(valid_momentum['vol_raw_momentum_old'], 
                          valid_momentum['vol_raw_momentum_new'], 
                          alpha=0.6, s=20)
        axes[0, 0].plot([valid_momentum['vol_raw_momentum_old'].min(), 
                        valid_momentum['vol_raw_momentum_old'].max()],
                       [valid_momentum['vol_raw_momentum_old'].min(), 
                        valid_momentum['vol_raw_momentum_old'].max()], 
                       'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Old vol_raw_momentum')
        axes[0, 0].set_ylabel('New vol_raw_momentum')
        axes[0, 0].set_title('vol_raw_momentum: Old vs New')
        axes[0, 0].grid(True, alpha=0.3)
    
    # vol_risk scatter plot
    if len(valid_risk) > 0:
        axes[0, 1].scatter(valid_risk['vol_risk_old'], 
                          valid_risk['vol_risk_new'], 
                          alpha=0.6, s=20)
        axes[0, 1].plot([valid_risk['vol_risk_old'].min(), 
                        valid_risk['vol_risk_old'].max()],
                       [valid_risk['vol_risk_old'].min(), 
                        valid_risk['vol_risk_old'].max()], 
                       'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Old vol_risk')
        axes[0, 1].set_ylabel('New vol_risk')
        axes[0, 1].set_title('vol_risk: Old vs New')
        axes[0, 1].grid(True, alpha=0.3)
    
    # vol_raw_decile scatter plot
    if len(valid_decile) > 0:
        axes[0, 2].scatter(valid_decile['vol_raw_decile_old'], 
                          valid_decile['vol_raw_decile_new'], 
                          alpha=0.6, s=20)
        axes[0, 2].plot([valid_decile['vol_raw_decile_old'].min(), 
                        valid_decile['vol_raw_decile_old'].max()],
                       [valid_decile['vol_raw_decile_old'].min(), 
                        valid_decile['vol_raw_decile_old'].max()], 
                       'r--', alpha=0.8)
        axes[0, 2].set_xlabel('Old vol_raw_decile')
        axes[0, 2].set_ylabel('New vol_raw_decile')
        axes[0, 2].set_title('vol_raw_decile: Old vs New')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Difference histograms
    if len(valid_momentum) > 0:
        momentum_diff = valid_momentum['vol_raw_momentum_new'] - valid_momentum['vol_raw_momentum_old']
        axes[1, 0].hist(momentum_diff, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(momentum_diff.mean(), color='red', linestyle='--', 
                          label=f'Mean: {momentum_diff.mean():.4f}')
        axes[1, 0].set_xlabel('Difference (New - Old)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('vol_raw_momentum Differences')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if len(valid_risk) > 0:
        risk_diff = valid_risk['vol_risk_new'] - valid_risk['vol_risk_old']
        axes[1, 1].hist(risk_diff, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(risk_diff.mean(), color='red', linestyle='--', 
                          label=f'Mean: {risk_diff.mean():.6f}')
        axes[1, 1].set_xlabel('Difference (New - Old)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('vol_risk Differences')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    if len(valid_decile) > 0:
        decile_diff = valid_decile['vol_raw_decile_new'] - valid_decile['vol_raw_decile_old']
        axes[1, 2].hist(decile_diff, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(decile_diff.mean(), color='red', linestyle='--', 
                          label=f'Mean: {decile_diff.mean():.4f}')
        axes[1, 2].set_xlabel('Difference (New - Old)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('vol_raw_decile Differences')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('volatility_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_implementation_changes():
    """Analyze the specific implementation changes"""
    
    print("\n=== IMPLEMENTATION CHANGES ANALYSIS ===")
    
    print("\n1. vol_raw_momentum:")
    print("   OLD: df['vol_raw'].pct_change(periods=3)")
    print("   NEW: Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)")
    print("   CHANGE: From 3-period percentage change of vol_raw to 1-period difference of 6-period volatility")
    
    print("\n2. vol_risk:")
    print("   OLD: Not clearly defined in the data")
    print("   NEW: Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6) (vol^2)")
    print("   CHANGE: Now explicitly defined as volatility squared")
    
    print("\n3. vol_raw_decile:")
    print("   OLD: Likely based on vol_raw percentile ranking")
    print("   NEW: Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10")
    print("   CHANGE: Now explicitly using 180-period rolling rank of 6-period volatility")
    
    print("\n=== KEY INSIGHTS ===")
    print("• vol_raw_momentum changed from momentum of vol_raw to momentum of realized volatility")
    print("• This makes it more theoretically sound as it tracks changes in actual volatility")
    print("• The new implementation uses log returns which is more standard in finance")
    print("• All features now consistently use the same base volatility measure")

if __name__ == "__main__":
    # Load and analyze the data
    df, valid_momentum, valid_risk, valid_decile = load_and_analyze_volatility_features()
    
    # Create comparison plots
    create_comparison_plots(df, valid_momentum, valid_risk, valid_decile)
    
    # Analyze implementation changes
    analyze_implementation_changes()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. The new vol_raw_momentum is more theoretically sound")
    print("2. Consider the correlation between old and new implementations")
    print("3. Test both versions in backtesting to see performance impact")
    print("4. The new implementation provides more consistent volatility measures")