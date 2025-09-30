#!/usr/bin/env python3
"""
Detailed summary of volatility features analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_detailed_summary():
    """Create a detailed summary of the volatility features analysis"""
    
    # Load the data
    df = pd.read_csv('Research/v2_model/feature_QA_1.csv')
    df.columns = ['vol_raw_momentum_old', 'vol_raw_momentum_new', 
                  'vol_risk_old', 'vol_risk_new', 
                  'vol_raw_decile_old', 'vol_raw_decile_new']
    
    print("=" * 80)
    print("VOLATILITY FEATURES ANALYSIS SUMMARY")
    print("=" * 80)
    
    print("\n🔍 IMPLEMENTATION CHANGES:")
    print("\n1. vol_raw_momentum:")
    print("   OLD: df['vol_raw'].pct_change(periods=3)")
    print("        → 3-period percentage change of some vol_raw measure")
    print("   NEW: Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)")
    print("        → 1-period difference of 6-period realized volatility")
    
    print("\n2. vol_risk:")
    print("   OLD: Unknown/inconsistent implementation")
    print("   NEW: Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)")
    print("        → Volatility squared (variance)")
    
    print("\n3. vol_raw_decile:")
    print("   OLD: Some form of decile ranking")
    print("   NEW: Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10")
    print("        → 180-period rolling rank of 6-period volatility, scaled to 0-10")
    
    print("\nSTATISTICAL COMPARISON:")
    
    # Calculate valid data
    valid_momentum = df.dropna(subset=['vol_raw_momentum_old', 'vol_raw_momentum_new'])
    valid_risk = df.dropna(subset=['vol_risk_old', 'vol_risk_new'])
    valid_decile = df.dropna(subset=['vol_raw_decile_old', 'vol_raw_decile_new'])
    
    print(f"\n• vol_raw_momentum:")
    print(f"  - Correlation: {valid_momentum['vol_raw_momentum_old'].corr(valid_momentum['vol_raw_momentum_new']):.4f} (LOW)")
    print(f"  - Old range: [{valid_momentum['vol_raw_momentum_old'].min():.2f}, {valid_momentum['vol_raw_momentum_old'].max():.2f}]")
    print(f"  - New range: [{valid_momentum['vol_raw_momentum_new'].min():.6f}, {valid_momentum['vol_raw_momentum_new'].max():.6f}]")
    print(f"  - Scale difference: ~{abs(valid_momentum['vol_raw_momentum_old'].std() / valid_momentum['vol_raw_momentum_new'].std()):.0f}x")
    
    print(f"\n• vol_risk:")
    print(f"  - Correlation: {valid_risk['vol_risk_old'].corr(valid_risk['vol_risk_new']):.4f} (LOW)")
    print(f"  - Old range: [{valid_risk['vol_risk_old'].min():.3f}, {valid_risk['vol_risk_old'].max():.3f}]")
    print(f"  - New range: [{valid_risk['vol_risk_new'].min():.6f}, {valid_risk['vol_risk_new'].max():.6f}]")
    print(f"  - Scale difference: ~{abs(valid_risk['vol_risk_old'].std() / valid_risk['vol_risk_new'].std()):.0f}x")
    
    print(f"\n• vol_raw_decile:")
    print(f"  - Correlation: {valid_decile['vol_raw_decile_old'].corr(valid_decile['vol_raw_decile_new']):.4f} (MODERATE)")
    print(f"  - Old range: [{valid_decile['vol_raw_decile_old'].min():.0f}, {valid_decile['vol_raw_decile_old'].max():.0f}]")
    print(f"  - New range: [{valid_decile['vol_raw_decile_new'].min():.6f}, {valid_decile['vol_raw_decile_new'].max():.6f}]")
    print(f"  - Scale difference: ~{abs(valid_decile['vol_raw_decile_old'].std() / valid_decile['vol_raw_decile_new'].std()):.0f}x")
    
    print("\n⚠️  CRITICAL FINDINGS:")
    print("\n1. SCALE MISMATCH:")
    print("   • All features have dramatically different scales between old and new")
    print("   • vol_raw_momentum: old values ~±500, new values ~±0.07")
    print("   • vol_risk: old values 0-1, new values 0-0.016")
    print("   • vol_raw_decile: old values 0-9, new values 0-0.056")
    
    print("\n2. LOW CORRELATIONS:")
    print("   • vol_raw_momentum: 0.14 correlation (essentially different features)")
    print("   • vol_risk: 0.13 correlation (essentially different features)")
    print("   • vol_raw_decile: 0.56 correlation (moderate relationship)")
    
    print("\n3. THEORETICAL IMPROVEMENTS:")
    print("   • New implementations use proper log returns")
    print("   • Consistent 6-period volatility base")
    print("   • More standard financial volatility measures")
    
    print("\nRECOMMENDATIONS:")
    print("\n1. IMMEDIATE ACTIONS:")
    print("   • Update any models/strategies using these features")
    print("   • Retrain models with new feature scales")
    print("   • Update feature normalization/scaling procedures")
    
    print("\n2. TESTING PRIORITIES:")
    print("   • Backtest performance with new vs old features")
    print("   • Check if feature importance rankings change")
    print("   • Validate that new features provide better signal")
    
    print("\n3. FEATURE ENGINEERING:")
    print("   • Consider creating hybrid features if old ones had value")
    print("   • Test different volatility windows (not just 6)")
    print("   • Add volatility regime detection features")
    
    print("\n4. MODEL UPDATES:")
    print("   • Retrain all models using these features")
    print("   • Update hyperparameters for new feature scales")
    print("   • Consider ensemble approaches using both old and new")
    
    # Create a simple comparison table
    print("\n📋 FEATURE COMPARISON TABLE:")
    print("-" * 80)
    print(f"{'Feature':<20} {'Old Mean':<12} {'New Mean':<12} {'Old Std':<12} {'New Std':<12} {'Correlation':<12}")
    print("-" * 80)
    print(f"{'vol_raw_momentum':<20} {valid_momentum['vol_raw_momentum_old'].mean():<12.3f} {valid_momentum['vol_raw_momentum_new'].mean():<12.6f} {valid_momentum['vol_raw_momentum_old'].std():<12.3f} {valid_momentum['vol_raw_momentum_new'].std():<12.6f} {valid_momentum['vol_raw_momentum_old'].corr(valid_momentum['vol_raw_momentum_new']):<12.4f}")
    print(f"{'vol_risk':<20} {valid_risk['vol_risk_old'].mean():<12.3f} {valid_risk['vol_risk_new'].mean():<12.6f} {valid_risk['vol_risk_old'].std():<12.3f} {valid_risk['vol_risk_new'].std():<12.6f} {valid_risk['vol_risk_old'].corr(valid_risk['vol_risk_new']):<12.4f}")
    print(f"{'vol_raw_decile':<20} {valid_decile['vol_raw_decile_old'].mean():<12.3f} {valid_decile['vol_raw_decile_new'].mean():<12.6f} {valid_decile['vol_raw_decile_old'].std():<12.3f} {valid_decile['vol_raw_decile_new'].std():<12.6f} {valid_decile['vol_raw_decile_old'].corr(valid_decile['vol_raw_decile_new']):<12.4f}")
    print("-" * 80)

if __name__ == "__main__":
    create_detailed_summary()