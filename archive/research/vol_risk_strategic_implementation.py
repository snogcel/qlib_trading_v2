#!/usr/bin/env python3
"""
Strategic implementation using vol_risk as variance for enhanced position sizing and risk measurement
vol_risk = Std(Log(close/close_prev), 6)² = VARIANCE (not standard deviation)
"""
import pandas as pd
import numpy as np

def analyze_vol_risk_as_variance(df):
    """
    Analyze vol_risk as variance measure and its strategic applications
    """
    
    print("🔍 VOL_RISK AS VARIANCE ANALYSIS")
    print("=" * 60)
    
    if 'vol_risk' not in df.columns:
        print("vol_risk not found in data")
        return df
    
    # Basic statistics
    vol_risk_stats = df['vol_risk'].describe()
    print(f"Vol_Risk (Variance) Statistics:")
    print(f"   Mean: {vol_risk_stats['mean']:.6f}")
    print(f"   Std:  {vol_risk_stats['std']:.6f}")
    print(f"   Min:  {vol_risk_stats['min']:.6f}")
    print(f"   Max:  {vol_risk_stats['max']:.6f}")
    
    # Convert to standard deviation for intuition
    df['vol_risk_sqrt'] = np.sqrt(df['vol_risk'])  # This gives us back the std dev
    
    print(f"\n📈 Converted to Std Dev (√vol_risk):")
    vol_std_stats = df['vol_risk_sqrt'].describe()
    print(f"   Mean: {vol_std_stats['mean']:.4f} ({vol_std_stats['mean']*100:.2f}%)")
    print(f"   Std:  {vol_std_stats['std']:.4f}")
    print(f"   Min:  {vol_std_stats['min']:.4f}")
    print(f"   Max:  {vol_std_stats['max']:.4f}")
    
    # Variance vs std dev relationship
    print(f"\n🔬 Variance vs Std Dev Relationship:")
    print(f"   Variance amplifies extreme values quadratically")
    print(f"   Low vol (1% std) → 0.0001 variance")
    print(f"   High vol (5% std) → 0.0025 variance (25x higher!)")
    print(f"   This makes vol_risk excellent for risk scaling")
    
    return df

def variance_based_regime_identification(df):
    """
    Use vol_risk (variance) for more sophisticated regime identification
    Variance-based regimes are more sensitive to extreme risk periods
    """
    
    if 'vol_risk' not in df.columns:
        print("vol_risk not available for regime identification")
        return df
    
    # Variance-based regime thresholds (more sensitive than std dev)
    # These thresholds are based on variance, so they're quadratic
    vol_risk_10th = df['vol_risk'].quantile(0.10)
    vol_risk_30th = df['vol_risk'].quantile(0.30)
    vol_risk_70th = df['vol_risk'].quantile(0.70)
    vol_risk_90th = df['vol_risk'].quantile(0.90)
    
    print(f"Variance-Based Regime Thresholds:")
    print(f"   10th percentile: {vol_risk_10th:.6f} (√ = {np.sqrt(vol_risk_10th):.3f})")
    print(f"   30th percentile: {vol_risk_30th:.6f} (√ = {np.sqrt(vol_risk_30th):.3f})")
    print(f"   70th percentile: {vol_risk_70th:.6f} (√ = {np.sqrt(vol_risk_70th):.3f})")
    print(f"   90th percentile: {vol_risk_90th:.6f} (√ = {np.sqrt(vol_risk_90th):.3f})")
    
    # Create variance-based regimes (more granular than std dev based)
    df['variance_regime_ultra_low'] = (df['vol_risk'] <= vol_risk_10th).astype(int)
    df['variance_regime_low'] = ((df['vol_risk'] > vol_risk_10th) & (df['vol_risk'] <= vol_risk_30th)).astype(int)
    df['variance_regime_medium'] = ((df['vol_risk'] > vol_risk_30th) & (df['vol_risk'] <= vol_risk_70th)).astype(int)
    df['variance_regime_high'] = ((df['vol_risk'] > vol_risk_70th) & (df['vol_risk'] <= vol_risk_90th)).astype(int)
    df['variance_regime_extreme'] = (df['vol_risk'] > vol_risk_90th).astype(int)
    
    # Regime distribution
    regime_dist = {
        'Ultra Low': df['variance_regime_ultra_low'].sum(),
        'Low': df['variance_regime_low'].sum(),
        'Medium': df['variance_regime_medium'].sum(),
        'High': df['variance_regime_high'].sum(),
        'Extreme': df['variance_regime_extreme'].sum()
    }
    
    print(f"\n🏛️ Variance-Based Regime Distribution:")
    for regime, count in regime_dist.items():
        pct = count / len(df) * 100
        print(f"   {regime}: {count:,} ({pct:.1f}%)")
    
    return df

def variance_based_position_sizing(df, base_position_pct=0.1):
    """
    Use vol_risk (variance) for sophisticated position sizing
    Variance naturally penalizes high-risk periods more than std dev
    """
    
    if 'vol_risk' not in df.columns or 'q50' not in df.columns:
        print("Missing required columns for variance-based position sizing")
        return df
    
    # Method 1: Inverse Variance Scaling
    # Position size inversely proportional to variance
    # Higher variance = smaller position (quadratic penalty)
    target_variance = df['vol_risk'].median()  # Target "normal" variance level
    df['variance_scaling_factor'] = np.sqrt(target_variance / np.maximum(df['vol_risk'], target_variance * 0.1))
    
    # Method 2: Variance-Adjusted Kelly
    # Use variance directly in Kelly calculation for more accurate risk assessment
    df['expected_return'] = df['q50']  # Use Q50 as expected return
    df['variance_risk'] = df['vol_risk']  # This is already variance
    
    # Kelly formula: f = (bp - q) / b where b = odds, p = win prob, q = lose prob
    # Simplified: f = expected_return / variance (for small returns)
    df['kelly_variance_based'] = np.where(
        df['variance_risk'] > 0,
        df['expected_return'] / df['variance_risk'],
        0
    ).clip(-0.5, 0.5)  # Cap at ±50%
    
    # Method 3: Regime-Aware Variance Scaling
    # Different base positions for different variance regimes
    regime_multipliers = pd.Series(1.0, index=df.index)
    
    # Ultra low variance: can take larger positions (predictable)
    regime_multipliers += df['variance_regime_ultra_low'] * 0.5  # +50%
    
    # Low variance: slightly larger positions
    regime_multipliers += df['variance_regime_low'] * 0.2  # +20%
    
    # High variance: reduce positions
    regime_multipliers -= df['variance_regime_high'] * 0.3  # -30%
    
    # Extreme variance: heavily reduce positions
    regime_multipliers -= df['variance_regime_extreme'] * 0.6  # -60%
    
    df['regime_variance_multiplier'] = regime_multipliers.clip(0.1, 2.0)
    
    # Combined position sizing
    df['position_size_variance_based'] = (
        base_position_pct * 
        df['variance_scaling_factor'] * 
        df['regime_variance_multiplier'] * 
        np.abs(df['kelly_variance_based'])
    ).clip(0.01, 0.5)
    
    # Analysis
    print(f"\n💰 VARIANCE-BASED POSITION SIZING ANALYSIS:")
    print(f"   Average position size: {df['position_size_variance_based'].mean():.3f}")
    print(f"   Position size std: {df['position_size_variance_based'].std():.3f}")
    
    # By regime
    for regime in ['ultra_low', 'low', 'medium', 'high', 'extreme']:
        regime_col = f'variance_regime_{regime}'
        if regime_col in df.columns:
            regime_data = df[df[regime_col] == 1]
            if len(regime_data) > 0:
                avg_pos = regime_data['position_size_variance_based'].mean()
                print(f"   {regime.title()} variance regime avg position: {avg_pos:.3f}")
    
    return df

def variance_risk_metrics(df):
    """
    Create advanced risk metrics using vol_risk (variance)
    """
    
    if 'vol_risk' not in df.columns:
        return df
    
    # Risk-adjusted signal strength
    if 'q50' in df.columns:
        df['signal_to_variance_ratio'] = df['q50'].abs() / np.maximum(df['vol_risk'], 0.0001)
        df['variance_adjusted_signal'] = df['q50'] / np.sqrt(np.maximum(df['vol_risk'], 0.0001))
    
    # Variance momentum (change in risk level)
    df['variance_momentum'] = df['vol_risk'] - df['vol_risk'].shift(1)
    df['variance_acceleration'] = df['variance_momentum'] - df['variance_momentum'].shift(1)
    
    # Risk regime transitions (when variance changes significantly)
    df['variance_regime_change'] = (
        (df['variance_regime_extreme'] != df['variance_regime_extreme'].shift()) |
        (df['variance_regime_ultra_low'] != df['variance_regime_ultra_low'].shift())
    ).astype(int)
    
    # Variance percentile (rolling)
    df['variance_percentile'] = df['vol_risk'].rolling(168, min_periods=24).rank(pct=True)
    
    # Risk-adjusted returns (if we have returns)
    if 'truth' in df.columns:
        df['variance_adjusted_return'] = df['truth'] / np.sqrt(np.maximum(df['vol_risk'], 0.0001))
        df['variance_sharpe_proxy'] = df['truth'] / df['vol_risk']  # Return per unit variance
    
    print(f"\nVARIANCE RISK METRICS CREATED:")
    metrics = [
        'signal_to_variance_ratio', 'variance_adjusted_signal', 'variance_momentum',
        'variance_acceleration', 'variance_regime_change', 'variance_percentile'
    ]
    
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"   {metric}: mean={values.mean():.4f}, std={values.std():.4f}")
    
    return df

def enhanced_q50_with_variance_risk(df, transaction_cost_bps=20, base_info_ratio=1.5):
    """
    Enhanced Q50-centric approach using vol_risk (variance) for superior risk assessment
    """
    
    print(f"\nENHANCED Q50 WITH VARIANCE RISK")
    print("=" * 60)
    
    # Ensure we have required data
    if 'vol_risk' not in df.columns:
        print("vol_risk not available")
        return df
    
    # Analyze vol_risk as variance
    df = analyze_vol_risk_as_variance(df)
    
    # Create variance-based regimes
    df = variance_based_regime_identification(df)
    
    # Calculate core metrics
    df['spread'] = df['q90'] - df['q10']
    df['abs_q50'] = df['q50'].abs()
    
    # Enhanced information ratio using variance
    # Traditional: signal / spread
    # Enhanced: signal / sqrt(variance + spread²) - accounts for both prediction and market risk
    df['market_variance'] = df['vol_risk']  # Market variance from vol_risk
    df['prediction_variance'] = (df['spread'] / 2) ** 2  # Prediction variance from spread
    df['total_risk'] = np.sqrt(df['market_variance'] + df['prediction_variance'])
    df['enhanced_info_ratio'] = df['abs_q50'] / np.maximum(df['total_risk'], 0.001)
    
    print(f"Enhanced Info Ratio vs Traditional:")
    traditional_info = df['abs_q50'] / np.maximum(df['spread'], 0.001)
    print(f"   Traditional (signal/spread): {traditional_info.mean():.3f}")
    print(f"   Enhanced (signal/total_risk): {df['enhanced_info_ratio'].mean():.3f}")
    
    # Variance-based thresholds
    base_transaction_cost = transaction_cost_bps / 10000
    
    # Scale thresholds by variance (not std dev) - more sensitive to risk
    variance_multiplier = 1.0 + df['vol_risk'] * 1000  # Scale factor for variance
    df['variance_adjusted_threshold'] = base_transaction_cost * variance_multiplier
    
    # Regime-aware info ratio thresholds
    info_ratio_threshold = pd.Series(base_info_ratio, index=df.index)
    
    # Ultra low variance: can accept lower info ratios (stable environment)
    info_ratio_threshold -= df['variance_regime_ultra_low'] * 0.5
    
    # Extreme variance: require much higher info ratios (unstable environment)
    info_ratio_threshold += df['variance_regime_extreme'] * 1.0
    
    df['variance_info_ratio_threshold'] = info_ratio_threshold.clip(0.5, 3.0)
    
    # Trading conditions
    df['economically_significant'] = df['abs_q50'] > df['variance_adjusted_threshold']
    df['high_quality'] = df['enhanced_info_ratio'] > df['variance_info_ratio_threshold']
    df['tradeable'] = df['economically_significant'] & df['high_quality']
    
    # Position sizing using variance
    df = variance_based_position_sizing(df)
    
    # Risk metrics
    df = variance_risk_metrics(df)
    
    # Signal generation
    buy_mask = df['tradeable'] & (df['q50'] > 0)
    sell_mask = df['tradeable'] & (df['q50'] < 0)
    
    df['side_variance_enhanced'] = -1  # Default HOLD
    df.loc[buy_mask, 'side_variance_enhanced'] = 1   # LONG
    df.loc[sell_mask, 'side_variance_enhanced'] = 0  # SHORT
    
    # Results
    signal_counts = df['side_variance_enhanced'].value_counts()
    total = len(df)
    
    print(f"\nVARIANCE-ENHANCED SIGNALS:")
    for side, count in signal_counts.items():
        side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side]
        print(f"   {side_name}: {count:,} ({count/total*100:.1f}%)")
    
    # Quality metrics
    trading_signals = df[df['side_variance_enhanced'] != -1]
    if len(trading_signals) > 0:
        print(f"\nTrading Signal Quality:")
        print(f"   Avg Enhanced Info Ratio: {trading_signals['enhanced_info_ratio'].mean():.2f}")
        print(f"   Avg Position Size: {trading_signals['position_size_variance_based'].mean():.3f}")
        print(f"   Avg Variance Level: {trading_signals['vol_risk'].mean():.6f}")
    
    return df

def main():
    """Test the variance-based implementation"""
    
    print("VOL_RISK STRATEGIC IMPLEMENTATION")
    print("=" * 70)
    print("Using vol_risk as VARIANCE for enhanced position sizing and risk measurement")
    
    # Create test data
    np.random.seed(42)
    n = 2000
    
    # Generate realistic variance data (vol_risk)
    # Variance should have occasional spikes (fat tails)
    base_variance = np.random.gamma(2, 0.0001, n)  # Gamma distribution for realistic variance
    regime_spikes = np.random.choice([1, 3, 10], n, p=[0.85, 0.12, 0.03])  # Occasional spikes
    vol_risk = base_variance * regime_spikes
    
    # Generate Q50 that's inversely related to variance (higher variance = weaker signals)
    q50_base = np.random.normal(0, 0.01, n)
    q50 = q50_base / np.sqrt(vol_risk * 1000 + 1)  # Weaker signals in high variance
    
    # Generate spread that increases with variance
    spread = np.random.uniform(0.01, 0.03, n) * np.sqrt(vol_risk * 1000 + 1)
    
    q10 = q50 - spread * 0.4
    q90 = q50 + spread * 0.6
    
    # Create test DataFrame
    df = pd.DataFrame({
        'vol_risk': vol_risk,
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'truth': np.random.normal(0, np.sqrt(vol_risk))  # Returns scale with variance
    })
    
    print(f"Created test data with realistic variance distribution")
    print(f"   Vol_risk range: {vol_risk.min():.6f} to {vol_risk.max():.6f}")
    print(f"   Vol_risk mean: {vol_risk.mean():.6f} (√ = {np.sqrt(vol_risk.mean()):.3f})")
    
    # Test the enhanced implementation
    df_result = enhanced_q50_with_variance_risk(df)
    
    print(f"\nVARIANCE-BASED IMPLEMENTATION COMPLETE!")
    print(f"   Ready for integration with crypto_loader_optimized vol_risk feature")

if __name__ == "__main__":
    main()