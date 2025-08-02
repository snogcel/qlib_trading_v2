#!/usr/bin/env python3
"""
Validate the predictive power of quantile spread (q90 - q10)
Before using spread in position sizing, prove it has predictive value
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def validate_spread_predictive_power(df):
    """
    Test if quantile spread predicts future volatility/returns
    """
    print("Validating Quantile Spread Predictive Power")
    print("=" * 50)
    
    # Calculate spread
    df['spread'] = df['q90'] - df['q10']
    
    # Calculate future realized volatility (what spread should predict)
    # FIXED: rolling(1).std() always returns NaN - use absolute return as proxy
    df['future_vol_1h'] = df['truth'].shift(-1).abs()  # Next period absolute return
    df['future_vol_2h'] = df['truth'].rolling(2).std().shift(-1)  # 2-period rolling std
    df['future_vol_6h'] = df['truth'].rolling(6).std().shift(-6)
    df['future_vol_24h'] = df['truth'].rolling(24).std().shift(-24)
    
    # Calculate future absolute returns
    df['future_abs_return_1h'] = df['truth'].shift(-1).abs()
    df['future_abs_return_6h'] = df['truth'].shift(-6).abs()
    
    # Test correlations
    correlations = {}
    
    for target in ['future_vol_1h', 'future_vol_2h', 'future_vol_6h', 'future_vol_24h', 
                   'future_abs_return_1h', 'future_abs_return_6h']:
        if target in df.columns:
            corr = df['spread'].corr(df[target])
            correlations[target] = corr
            print(f"Spread vs {target}: {corr:.4f}")
    
    # Test if spread predicts volatility better than individual quantiles
    print(f"\nComparison with individual quantiles (1h absolute return):")
    print(f"abs(q10) vs future_vol_1h: {df['q10'].abs().corr(df['future_vol_1h']):.4f}")
    print(f"abs(q50) vs future_vol_1h: {df['q50'].abs().corr(df['future_vol_1h']):.4f}")
    print(f"abs(q90) vs future_vol_1h: {df['q90'].abs().corr(df['future_vol_1h']):.4f}")
    print(f"spread vs future_vol_1h: {df['spread'].corr(df['future_vol_1h']):.4f}")
    
    # Also test with 2h rolling volatility for comparison
    if 'future_vol_2h' in df.columns:
        print(f"\nComparison with individual quantiles (2h rolling volatility):")
        print(f"abs(q10) vs future_vol_2h: {df['q10'].abs().corr(df['future_vol_2h']):.4f}")
        print(f"abs(q50) vs future_vol_2h: {df['q50'].abs().corr(df['future_vol_2h']):.4f}")
        print(f"abs(q90) vs future_vol_2h: {df['q90'].abs().corr(df['future_vol_2h']):.4f}")
        print(f"spread vs future_vol_2h: {df['spread'].corr(df['future_vol_2h']):.4f}")
    
    # Test spread deciles vs future performance
    df['spread_decile'] = pd.qcut(df['spread'], 10, labels=False)
    decile_analysis = df.groupby('spread_decile').agg({
        'future_vol_1h': 'mean',
        'future_abs_return_1h': 'mean',
        'truth': ['mean', 'std']
    }).round(4)
    
    print(f"\nSpread Decile Analysis:")
    print(decile_analysis)
    
    # Statistical significance test
    high_spread = df[df['spread_decile'] >= 8]['future_vol_1h'].dropna()
    low_spread = df[df['spread_decile'] <= 1]['future_vol_1h'].dropna()
    
    if len(high_spread) > 0 and len(low_spread) > 0:
        t_stat, p_value = stats.ttest_ind(high_spread, low_spread)
        print(f"\nT-test (high vs low spread deciles):")
        print(f"High spread mean vol: {high_spread.mean():.6f}")
        print(f"Low spread mean vol: {low_spread.mean():.6f}")
        print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}")
        print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    return correlations

def validate_signal_thresholds(df):
    """
    Validate if your signal_thresh and spread_thresh are meaningful
    """
    print(f"\n" + "=" * 50)
    print("Validating Signal Thresholds")
    print("=" * 50)
    
    # Check if signals above threshold perform better
    if 'signal_thresh_adaptive' in df.columns:
        df['above_signal_thresh'] = df['abs_q50'] > df['signal_thresh_adaptive']
        
        above_thresh = df[df['above_signal_thresh']]['truth'].shift(-1).dropna()
        below_thresh = df[~df['above_signal_thresh']]['truth'].shift(-1).dropna()
        
        print(f"Above signal threshold:")
        print(f"  Count: {len(above_thresh)}")
        print(f"  Mean return: {above_thresh.mean():.6f}")
        print(f"  Std return: {above_thresh.std():.6f}")
        print(f"  Sharpe: {above_thresh.mean()/above_thresh.std():.4f}")
        
        print(f"Below signal threshold:")
        print(f"  Count: {len(below_thresh)}")
        print(f"  Mean return: {below_thresh.mean():.6f}")
        print(f"  Std return: {below_thresh.std():.6f}")
        print(f"  Sharpe: {below_thresh.mean()/below_thresh.std():.4f}")
        
        if len(above_thresh) > 0 and len(below_thresh) > 0:
            t_stat, p_value = stats.ttest_ind(above_thresh.abs(), below_thresh.abs())
            print(f"T-test (above vs below threshold absolute returns):")
            print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.6f}")
    
    # Check spread threshold
    if 'spread_thresh' in df.columns:
        df['below_spread_thresh'] = df['spread'] < df['spread_thresh']
        
        tight_spread = df[df['below_spread_thresh']]['truth'].shift(-1).dropna()
        wide_spread = df[~df['below_spread_thresh']]['truth'].shift(-1).dropna()
        
        print(f"\nTight spread (below threshold):")
        print(f"  Count: {len(tight_spread)}")
        print(f"  Mean return: {tight_spread.mean():.6f}")
        print(f"  Std return: {tight_spread.std():.6f}")
        
        print(f"Wide spread (above threshold):")
        print(f"  Count: {len(wide_spread)}")
        print(f"  Mean return: {wide_spread.mean():.6f}")
        print(f"  Std return: {wide_spread.std():.6f}")

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("df_all_macro_analysis.csv")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Add abs_q50 if missing
    if 'abs_q50' not in df.columns:
        df['abs_q50'] = df['q50'].abs()
    
    # Run validations
    correlations = validate_spread_predictive_power(df)
    validate_signal_thresholds(df)
    
    print(f"\n" + "=" * 50)
    print("CONCLUSIONS:")
    print("=" * 50)
    print("1. If spread correlations with future volatility are < 0.3, consider removing spread-based features")
    print("2. If signal thresholds don't show significant performance differences, they may be arbitrary")
    print("3. Focus on features with proven predictive power for Kelly calculations")