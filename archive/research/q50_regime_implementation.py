#!/usr/bin/env python3
"""
Q50-centric signal generation with regime identification and vol_risk integration
Replaces the problematic threshold approach with economically meaningful logic
"""
import pandas as pd
import numpy as np

def calculate_vol_risk(df, vol_col='$realized_vol_6', rolling_window=168):
    """
    Calculate vol_risk as normalized volatility measure
    vol_risk = Std(Log(close/close_prev), 6) * Std(Log(close/close_prev), 6)
    
    Args:
        df: DataFrame with volatility data
        vol_col: Column name for realized volatility
        rolling_window: Window for quantile calculation (168 = 1 week hourly)
    """
    
    if vol_col not in df.columns:
        print(f"⚠️  {vol_col} not found, using vol_raw as fallback")
        vol_col = 'vol_raw' if 'vol_raw' in df.columns else '$realized_vol_3'
    
    # Calculate rolling quantiles for normalization
    q_low = df[vol_col].rolling(rolling_window, min_periods=24).quantile(0.01)
    q_high = df[vol_col].rolling(rolling_window, min_periods=24).quantile(0.99)
    
    # Normalize to 0-1 range (vol_risk definition)
    df['vol_risk'] = ((df[vol_col] - q_low.shift(1)) / 
                     (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)
    
    # Fill initial NaN values
    df['vol_risk'] = df['vol_risk'].fillna(0.5)  # Neutral risk level
    
    return df

def identify_market_regimes(df):
    """
    Identify market regimes using volatility and momentum features
    Creates regime interaction features for enhanced signal quality
    """
    
    # Ensure we have required volatility measures
    if 'vol_risk' not in df.columns:
        df = calculate_vol_risk(df)
    
    # Volatility regimes (using vol_risk for consistency)
    df['vol_regime_low'] = (df['vol_risk'] < 0.3).astype(int)
    df['vol_regime_medium'] = ((df['vol_risk'] >= 0.3) & (df['vol_risk'] < 0.7)).astype(int)
    df['vol_regime_high'] = (df['vol_risk'] >= 0.7).astype(int)
    
    # Momentum regimes (using existing momentum features if available)
    if 'vol_raw_momentum' in df.columns:
        momentum = df['vol_raw_momentum']
    else:
        # Calculate simple momentum if not available
        momentum = df['vol_raw'].rolling(24).mean() / df['vol_raw'].rolling(168).mean() - 1
        df['vol_raw_momentum'] = momentum
    
    df['momentum_regime_trending'] = (abs(momentum) > 0.1).astype(int)
    df['momentum_regime_ranging'] = (abs(momentum) <= 0.1).astype(int)
    
    # Combined regime classification
    df['regime_low_vol_trending'] = df['vol_regime_low'] * df['momentum_regime_trending']
    df['regime_low_vol_ranging'] = df['vol_regime_low'] * df['momentum_regime_ranging']
    df['regime_high_vol_trending'] = df['vol_regime_high'] * df['momentum_regime_trending']
    df['regime_high_vol_ranging'] = df['vol_regime_high'] * df['momentum_regime_ranging']
    
    # Regime stability (how long in current regime)
    df['regime_stability'] = df.groupby(
        (df['vol_regime_high'] != df['vol_regime_high'].shift()).cumsum()
    ).cumcount() + 1
    
    return df

def q50_regime_aware_signals(df, transaction_cost_bps=20, base_info_ratio=1.5, 
                           regime_adjustments=True, vol_risk_scaling=True):
    """
    Generate Q50-centric signals with regime-aware adjustments
    
    Args:
        df: DataFrame with quantile predictions and features
        transaction_cost_bps: Base trading costs in basis points
        base_info_ratio: Base minimum information ratio
        regime_adjustments: Whether to adjust thresholds by regime
        vol_risk_scaling: Whether to scale thresholds by vol_risk
    """
    
    df = df.copy()
    
    # Ensure we have regime features
    df = identify_market_regimes(df)
    
    # Calculate core signal metrics
    df['spread'] = df['q90'] - df['q10']
    df['abs_q50'] = df['q50'].abs()
    df['info_ratio'] = df['abs_q50'] / np.maximum(df['spread'], 0.001)
    
    # Base economic threshold
    base_transaction_cost = transaction_cost_bps / 10000
    
    # Regime-aware threshold adjustments
    if regime_adjustments:
        # Different thresholds for different regimes
        regime_multipliers = pd.Series(1.0, index=df.index)  # Default multiplier
        
        # High volatility regimes: require higher returns to compensate for risk
        regime_multipliers += df['vol_regime_high'] * 0.5  # +50% threshold in high vol
        
        # Low volatility regimes: can accept lower thresholds (more opportunities)
        regime_multipliers -= df['vol_regime_low'] * 0.2  # -20% threshold in low vol
        
        # Trending markets: lower thresholds (momentum helps)
        regime_multipliers -= df['momentum_regime_trending'] * 0.1  # -10% in trending
        
        # Ranging markets: higher thresholds (mean reversion risk)
        regime_multipliers += df['momentum_regime_ranging'] * 0.2  # +20% in ranging
        
        # Ensure multipliers stay reasonable
        regime_multipliers = regime_multipliers.clip(0.5, 2.0)
        
        df['regime_multiplier'] = regime_multipliers
    else:
        df['regime_multiplier'] = 1.0
    
    # Vol_risk scaling for additional risk adjustment
    if vol_risk_scaling:
        # Higher vol_risk = higher threshold needed
        vol_risk_multiplier = 1.0 + df['vol_risk'] * 1.0  # Scale 1.0-2.0 based on vol_risk
        df['vol_risk_multiplier'] = vol_risk_multiplier
    else:
        df['vol_risk_multiplier'] = 1.0
    
    # Combined effective threshold
    df['effective_threshold'] = (base_transaction_cost * 
                                df['regime_multiplier'] * 
                                df['vol_risk_multiplier'])
    
    # Regime-aware information ratio threshold
    regime_info_adjustments = pd.Series(base_info_ratio, index=df.index)
    
    if regime_adjustments:
        # High volatility: require higher info ratio (more selective)
        regime_info_adjustments += df['vol_regime_high'] * 0.5
        
        # Low volatility: can accept lower info ratio (more opportunities)
        regime_info_adjustments -= df['vol_regime_low'] * 0.3
        
        # Stable regimes: can be less selective
        stable_regime = (df['regime_stability'] > 10).astype(int)
        regime_info_adjustments -= stable_regime * 0.2
    
    df['effective_info_ratio_threshold'] = regime_info_adjustments.clip(0.8, 3.0)
    
    # Economic significance filter
    df['economically_significant'] = df['abs_q50'] > df['effective_threshold']
    
    # Signal quality filter (regime-aware)
    df['high_quality'] = df['info_ratio'] > df['effective_info_ratio_threshold']
    
    # Combined trading condition
    df['tradeable'] = df['economically_significant'] & df['high_quality']
    
    # Generate signals using pure Q50 logic (no complex prob_up needed!)
    df['signal_direction_q50'] = np.where(
        ~df['tradeable'], 'HOLD',
        np.where(df['q50'] > 0, 'LONG', 'SHORT')
    )
    
    # Regime-aware signal strength
    df['excess_return'] = df['abs_q50'] - df['effective_threshold']
    df['base_strength'] = np.where(
        df['tradeable'],
        df['excess_return'] * np.minimum(df['info_ratio'] / df['effective_info_ratio_threshold'], 2.0),
        0.0
    )
    
    # Regime strength adjustments
    regime_strength_bonus = pd.Series(1.0, index=df.index)
    if regime_adjustments:
        # Trending markets: boost signal strength (momentum helps)
        regime_strength_bonus += df['momentum_regime_trending'] * 0.3
        
        # Low vol stable regimes: boost strength (predictable environment)
        low_vol_stable = df['vol_regime_low'] * (df['regime_stability'] > 5).astype(int)
        regime_strength_bonus += low_vol_stable * 0.2
        
        # High vol ranging: reduce strength (unpredictable)
        high_vol_ranging = df['vol_regime_high'] * df['momentum_regime_ranging']
        regime_strength_bonus -= high_vol_ranging * 0.3
    
    df['regime_strength_multiplier'] = regime_strength_bonus.clip(0.5, 1.5)
    df['signal_strength_q50'] = df['base_strength'] * df['regime_strength_multiplier']
    
    # Confidence based on information ratio and regime stability
    base_confidence = np.minimum(df['info_ratio'] / df['effective_info_ratio_threshold'], 1.0)
    regime_confidence_bonus = (df['regime_stability'] / 20).clip(0, 0.2)  # Up to +20% for stable regimes
    df['signal_confidence_q50'] = (base_confidence + regime_confidence_bonus).clip(0, 1.0)
    
    # Side encoding (matching current format)
    df['side_q50'] = np.where(
        df['signal_direction_q50'] == 'LONG', 1,
        np.where(df['signal_direction_q50'] == 'SHORT', 0, -1)
    )
    
    # Create regime interaction features for model training
    df = create_regime_interaction_features(df)
    
    return df

def create_regime_interaction_features(df):
    """
    Create interaction features between regimes and signal characteristics
    These can be used as additional features in your quantile model
    """
    
    # Q50 × Regime interactions
    df['q50_x_low_vol'] = df['q50'] * df['vol_regime_low']
    df['q50_x_high_vol'] = df['q50'] * df['vol_regime_high']
    df['q50_x_trending'] = df['q50'] * df['momentum_regime_trending']
    df['q50_x_ranging'] = df['q50'] * df['momentum_regime_ranging']
    
    # Spread × Regime interactions (uncertainty varies by regime)
    df['spread_x_low_vol'] = df['spread'] * df['vol_regime_low']
    df['spread_x_high_vol'] = df['spread'] * df['vol_regime_high']
    
    # Vol_risk × Signal interactions
    df['vol_risk_x_abs_q50'] = df['vol_risk'] * df['abs_q50']
    df['vol_risk_x_spread'] = df['vol_risk'] * df['spread']
    
    # Information ratio × Regime (signal quality varies by regime)
    df['info_ratio_x_trending'] = df['info_ratio'] * df['momentum_regime_trending']
    df['info_ratio_x_stable'] = df['info_ratio'] * (df['regime_stability'] > 10).astype(int)
    
    # Regime transition features (regime changes can be predictive)
    df['regime_transition'] = (df['vol_regime_high'] != df['vol_regime_high'].shift()).astype(int)
    df['momentum_transition'] = (df['momentum_regime_trending'] != df['momentum_regime_trending'].shift()).astype(int)
    
    return df

def replace_current_signal_logic(df_all):
    """
    Replace the current problematic signal generation with Q50-centric approach
    This function can directly replace the existing logic in your main script
    """
    
    print("Replacing current signal logic with Q50-centric regime-aware approach...")
    
    # Generate Q50-centric signals with regime awareness
    df_all = q50_regime_aware_signals(
        df_all,
        transaction_cost_bps=20,  # Adjust based on your trading costs
        base_info_ratio=1.5,      # Adjust based on desired selectivity
        regime_adjustments=True,
        vol_risk_scaling=True
    )
    
    # Replace the old 'side' column with the new Q50-based one
    df_all['side'] = df_all['side_q50']
    
    # Keep some legacy columns for compatibility (but mark them as deprecated)
    df_all['signal_thresh_adaptive'] = df_all['effective_threshold']  # For compatibility
    df_all['signal_strength'] = df_all['signal_strength_q50']
    df_all['signal_confidence'] = df_all['signal_confidence_q50']
    
    # Add prob_up calculation for compatibility (but it's now derived from Q50)
    df_all['prob_up'] = np.where(df_all['q50'] > 0, 0.7, 0.3)  # Simple approximation
    
    print(f"Q50-centric signals generated:")
    signal_counts = df_all['side'].value_counts()
    for side, count in signal_counts.items():
        side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side]
        print(f"   {side_name}: {count:,} ({count/len(df_all)*100:.1f}%)")
    
    # Show regime distribution
    print(f"\nRegime Distribution:")
    print(f"   Low Vol: {df_all['vol_regime_low'].sum():,} ({df_all['vol_regime_low'].mean()*100:.1f}%)")
    print(f"   High Vol: {df_all['vol_regime_high'].sum():,} ({df_all['vol_regime_high'].mean()*100:.1f}%)")
    print(f"   Trending: {df_all['momentum_regime_trending'].sum():,} ({df_all['momentum_regime_trending'].mean()*100:.1f}%)")
    
    # Show signal quality improvement
    trading_signals = df_all[df_all['side'] != -1]
    if len(trading_signals) > 0:
        avg_info_ratio = trading_signals['info_ratio'].mean()
        avg_excess_return = trading_signals['excess_return'].mean()
        print(f"\nSignal Quality (trading signals only):")
        print(f"   Average Info Ratio: {avg_info_ratio:.2f}")
        print(f"   Average Excess Return: {avg_excess_return:.4f}")
        print(f"   Average Confidence: {trading_signals['signal_confidence_q50'].mean():.2f}")
    
    return df_all

def main():
    """Test the implementation"""
    
    print("Q50-CENTRIC REGIME-AWARE SIGNAL GENERATION")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n = 2000
    
    # Generate realistic quantile data with regime changes
    vol_base = np.random.uniform(0.005, 0.02, n)
    vol_regime = np.random.choice([0.5, 1.0, 2.0], n, p=[0.6, 0.3, 0.1])  # Low, medium, high vol regimes
    vol_raw = vol_base * vol_regime
    
    q50 = np.random.normal(0, 0.008, n) * vol_regime  # Returns scale with volatility
    spread_base = np.random.uniform(0.01, 0.03, n)
    spread = spread_base * vol_regime  # Uncertainty scales with volatility
    
    q10 = q50 - spread * 0.4
    q90 = q50 + spread * 0.6
    
    # Create test DataFrame
    df = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'vol_raw': vol_raw,
        '$realized_vol_6': vol_raw
    })
    
    print(f"Generated {len(df):,} test observations")
    
    # Test the Q50-centric approach
    df_result = replace_current_signal_logic(df)
    
    print(f"\nImplementation ready for integration into main script!")
    print(f"   Key new columns: side_q50, signal_strength_q50, signal_confidence_q50")
    print(f"   Regime features: vol_regime_*, momentum_regime_*, regime_*")
    print(f"   Interaction features: q50_x_*, spread_x_*, vol_risk_x_*")

if __name__ == "__main__":
    main()