import pandas as pd
import numpy as np

def simplified_adaptive_threshold_strategy(df):
    """
    Simplified version focusing on the most effective adjustments
    """
    df["abs_q50"] = df["q50"].abs()
    
    # Base threshold (validated 90th percentile)
    base_threshold = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90)
    
    # Simplified regime detection (reduce switching frequency)
    # Use longer lookback to reduce noise
    vol_ma = df['vol_risk'].rolling(7).mean()  # Smooth volatility
    
    conditions = [
        vol_ma > vol_ma.quantile(0.85),  # Only extreme high vol
        vol_ma < vol_ma.quantile(0.25),  # Only extreme low vol
    ]
    
    # More conservative adjustments
    threshold_adjustments = [
        0.87,  # Slightly lower in extreme high vol (was 0.85)
        0.93,  # Slightly higher in extreme low vol (was 0.95)
    ]
    
    df["signal_thresh_adaptive"] = np.select(
        conditions,
        [df['abs_q50'].rolling(30, min_periods=10).quantile(adj) for adj in threshold_adjustments],
        default=base_threshold
    )
    
    # Signal relative strength
    df["signal_rel"] = (df["abs_q50"] - df["signal_thresh_adaptive"]) / (df["signal_thresh_adaptive"] + 1e-12)
    
    return df

def validate_adaptive_vs_static_thresholds(df):
    """
    Test if adaptive thresholds outperform static thresholds
    """
    print("=== ADAPTIVE vs STATIC THRESHOLD VALIDATION ===")
    
    # Apply your adaptive strategy
    df = simplified_adaptive_threshold_strategy(df)
    
    # Create static threshold for comparison (your validated 90th percentile)
    df['signal_thresh_static'] = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90)
    
    # Test both approaches
    future_returns = df['truth'].shift(-1)
    
    # Static threshold signals
    static_signals = df['abs_q50'] >= df['signal_thresh_static']
    static_performance = {
        'avg_return': future_returns[static_signals].mean(),
        'hit_rate': (future_returns[static_signals] > 0).mean(),
        'count': static_signals.sum(),
        'sharpe': future_returns[static_signals].mean() / future_returns[static_signals].std() if static_signals.sum() > 10 else 0
    }
    
    # Adaptive threshold signals
    adaptive_signals = df['abs_q50'] >= df['signal_thresh_adaptive']
    adaptive_performance = {
        'avg_return': future_returns[adaptive_signals].mean(),
        'hit_rate': (future_returns[adaptive_signals] > 0).mean(),
        'count': adaptive_signals.sum(),
        'sharpe': future_returns[adaptive_signals].mean() / future_returns[adaptive_signals].std() if adaptive_signals.sum() > 10 else 0
    }
    
    print(f"Static Threshold Performance:")
    print(f"  Avg Return: {static_performance['avg_return']:.6f}")
    print(f"  Hit Rate: {static_performance['hit_rate']:.2%}")
    print(f"  Signal Count: {static_performance['count']}")
    print(f"  Sharpe: {static_performance['sharpe']:.4f}")
    
    print(f"\nAdaptive Threshold Performance:")
    print(f"  Avg Return: {adaptive_performance['avg_return']:.6f}")
    print(f"  Hit Rate: {adaptive_performance['hit_rate']:.2%}")
    print(f"  Signal Count: {adaptive_performance['count']}")
    print(f"  Sharpe: {adaptive_performance['sharpe']:.4f}")
    
    # Calculate improvements
    return_improvement = adaptive_performance['avg_return'] - static_performance['avg_return']
    hit_rate_improvement = adaptive_performance['hit_rate'] - static_performance['hit_rate']
    sharpe_improvement = adaptive_performance['sharpe'] - static_performance['sharpe']
    count_difference = adaptive_performance['count'] - static_performance['count']
    
    print(f"\nAdaptive vs Static Improvements:")
    print(f"  Return: {return_improvement:+.6f}")
    print(f"  Hit Rate: {hit_rate_improvement:+.2%}")
    print(f"  Sharpe: {sharpe_improvement:+.4f}")
    print(f"  Signal Count: {count_difference:+d}")
    
    return static_performance, adaptive_performance

def analyze_regime_specific_performance(df):
    """
    Test performance in different volatility regimes
    """
    print("\n=== REGIME-SPECIFIC PERFORMANCE ANALYSIS ===")
    
    future_returns = df['truth'].shift(-1)
    
    # Define volatility regimes
    vol_regimes = {
        'Very High Vol': df['vol_risk'] > df['vol_risk'].quantile(0.8),
        'High Vol': (df['vol_risk'] > df['vol_risk'].quantile(0.6)) & (df['vol_risk'] <= df['vol_risk'].quantile(0.8)),
        'Medium Vol': (df['vol_risk'] >= df['vol_risk'].quantile(0.3)) & (df['vol_risk'] <= df['vol_risk'].quantile(0.6)),
        'Low Vol': df['vol_risk'] < df['vol_risk'].quantile(0.3)
    }
    
    for regime_name, regime_mask in vol_regimes.items():
        print(f"\n{regime_name} Regime:")
        
        # Static performance in this regime
        static_signals_regime = (df['abs_q50'] >= df['signal_thresh_static']) & regime_mask
        if static_signals_regime.sum() > 10:
            static_return = future_returns[static_signals_regime].mean()
            static_hit_rate = (future_returns[static_signals_regime] > 0).mean()
            static_count = static_signals_regime.sum()
        else:
            static_return = static_hit_rate = static_count = 0
        
        # Adaptive performance in this regime
        adaptive_signals_regime = (df['abs_q50'] >= df['signal_thresh_adaptive']) & regime_mask
        if adaptive_signals_regime.sum() > 10:
            adaptive_return = future_returns[adaptive_signals_regime].mean()
            adaptive_hit_rate = (future_returns[adaptive_signals_regime] > 0).mean()
            adaptive_count = adaptive_signals_regime.sum()
        else:
            adaptive_return = adaptive_hit_rate = adaptive_count = 0
        
        print(f"  Static    - Return: {static_return:.6f}, Hit Rate: {static_hit_rate:.2%}, Count: {static_count}")
        print(f"  Adaptive  - Return: {adaptive_return:.6f}, Hit Rate: {adaptive_hit_rate:.2%}, Count: {adaptive_count}")
        
        if static_count > 0 and adaptive_count > 0:
            improvement = adaptive_return - static_return
            print(f"  Improvement: {improvement:+.6f}")

def test_threshold_stability(df):
    """
    Test how stable the adaptive thresholds are
    """
    print("\n=== THRESHOLD STABILITY ANALYSIS ===")
    
    # Calculate threshold changes
    df['thresh_change'] = df['signal_thresh_adaptive'].diff().abs()
    df['thresh_pct_change'] = df['signal_thresh_adaptive'].pct_change().abs()
    
    print(f"Threshold Statistics:")
    print(f"  Mean threshold: {df['signal_thresh_adaptive'].mean():.6f}")
    print(f"  Std threshold: {df['signal_thresh_adaptive'].std():.6f}")
    print(f"  Mean absolute change: {df['thresh_change'].mean():.6f}")
    print(f"  Mean % change: {df['thresh_pct_change'].mean():.2%}")
    print(f"  Max % change: {df['thresh_pct_change'].max():.2%}")
    
    # Count regime switches
    df['vol_regime'] = np.select([
        df['vol_risk'] > df['vol_risk'].quantile(0.8),
        df['vol_risk'] > df['vol_risk'].quantile(0.6),
        df['vol_risk'] < df['vol_risk'].quantile(0.3)
    ], ['very_high', 'high', 'low'], default='medium')
    
    regime_changes = (df['vol_regime'] != df['vol_regime'].shift(1)).sum()
    print(f"  Volatility regime changes: {regime_changes}")

# Enhanced version of your adaptive strategy with validation
def enhanced_adaptive_threshold_strategy(df):
    """
    Enhanced version with better regime detection
    """
    df["spread"] = df["q90"] - df["q10"]
    df["abs_q50"] = df["q50"].abs()
    
    # Base threshold (validated 90th percentile)
    base_threshold = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90)
    
    # Use vol_raw deciles for more precise regime detection
    if 'vol_raw' in df.columns:
        vol_decile = df['vol_raw'].apply(lambda x: get_vol_raw_decile(x) if 'get_vol_raw_decile' in globals() else 5)
        
        conditions = [
            vol_decile >= 8,  # Top 20% volatility
            vol_decile >= 6,  # Top 40% volatility  
            vol_decile <= 2,  # Bottom 30% volatility
        ]
        
        threshold_adjustments = [0.85, 0.88, 0.95]
    else:
        # Fallback to your original approach
        conditions = [
            df['vol_risk'] > df['vol_risk'].quantile(0.8),
            df['vol_risk'] > df['vol_risk'].quantile(0.6),
            df['vol_risk'] < df['vol_risk'].quantile(0.3),
        ]
        threshold_adjustments = [0.85, 0.88, 0.95]
    
    df["signal_thresh_adaptive"] = np.select(
        conditions, 
        [df['abs_q50'].rolling(30, min_periods=10).quantile(adj) for adj in threshold_adjustments],
        default=base_threshold
    )
    
    # Keep spread threshold simple (it's weak anyway)
    df["spread_thresh"] = df["spread"].rolling(30, min_periods=10).quantile(0.90)
    
    # Signal relative strength
    df["signal_rel"] = (df["abs_q50"] - df["signal_thresh_adaptive"]) / (df["signal_thresh_adaptive"] + 1e-12)
    
    return df

# Run the validation
def run_adaptive_threshold_validation(df):
    """
    Complete validation of adaptive threshold strategy
    """
    print("🔬 ADAPTIVE THRESHOLD STRATEGY VALIDATION")
    print("=" * 50)
    
    # 1. Basic performance comparison
    static_perf, adaptive_perf = validate_adaptive_vs_static_thresholds(df)
    
    # 2. Regime-specific analysis
    analyze_regime_specific_performance(df)
    
    # 3. Stability analysis
    test_threshold_stability(df)
    
    # 4. Recommendation
    print("\n=== RECOMMENDATION ===")
    
    if adaptive_perf['sharpe'] > static_perf['sharpe'] * 1.05:  # 5% improvement threshold
        print("✅ KEEP ADAPTIVE STRATEGY - Shows meaningful improvement")
    elif adaptive_perf['sharpe'] > static_perf['sharpe']:
        print("⚠️  MARGINAL IMPROVEMENT - Consider if complexity is worth it")
    else:
        print("❌ USE STATIC STRATEGY - Adaptive doesn't improve performance")
    
    return static_perf, adaptive_perf

# Usage:
# static_perf, adaptive_perf = run_adaptive_threshold_validation(df)

if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")
    static_perf, adaptive_perf = run_adaptive_threshold_validation(df)
