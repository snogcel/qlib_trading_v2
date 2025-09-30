"""
Test the top performing features in a simple backtest
"""

import pandas as pd
import numpy as np
from quantile_backtester import QuantileBacktester, BacktestConfig

def load_and_enhance_data():
    """Load data and add top performing features"""
    df = pd.read_csv("df_all_macro_analysis.csv")
    
    if 'instrument' in df.columns and 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['instrument', 'datetime'])
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs('BTCUSDT', level=0)
    
    df = df.sort_index()
    
    # Add top momentum features
    df["$momentum_1"] = df["truth"].rolling(1).mean().fillna(0)
    df["$momentum_3"] = df["truth"].rolling(3).mean().fillna(0)
    df["$momentum_24"] = df["truth"].rolling(24).mean().fillna(0)
    
    # Add volatility regime
    vol_window = 48
    df["vol_regime"] = (df["volatility"] / df["volatility"].rolling(vol_window).mean()).fillna(1.0)
    df["vol_mean_reversion"] = 1 / (1 + df["vol_regime"])
    
    # Add signal volatility adjustment
    df["signal_vol_adjusted"] = df["q50"] * df["vol_mean_reversion"]
    
    return df

def create_enhanced_tier_confidence(df):
    """Create enhanced tier confidence using top features"""
    
    # Use momentum and volatility-adjusted signals for better tier confidence
    momentum_strength = np.abs(df["$momentum_3"]) * 100  # Scale up
    signal_strength = np.abs(df["signal_vol_adjusted"]) * 100
    volatility_penalty = df["vol_regime"]
    
    # Combine for enhanced tier confidence
    enhanced_confidence = (momentum_strength + signal_strength) / volatility_penalty
    
    # Scale to 1-10 range
    df["enhanced_tier_confidence"] = np.clip(enhanced_confidence, 1.0, 10.0)
    
    return df

def simple_backtest_comparison():
    """Compare baseline vs enhanced features"""
    
    print("="*60)
    print("TOP FEATURES BACKTEST COMPARISON")
    print("="*60)
    
    # Load enhanced data
    df_enhanced = load_and_enhance_data()
    df_enhanced = create_enhanced_tier_confidence(df_enhanced)
    
    print(f"Loaded {len(df_enhanced)} observations")
    
    # Split data for testing
    total_len = len(df_enhanced)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.8)
    
    df_valid = df_enhanced.iloc[train_end:valid_end]
    
    print(f"Testing on {len(df_valid)} validation observations")
    
    # Test configurations
    configs = {
        'baseline': BacktestConfig(
            long_threshold=0.6, short_threshold=0.6,
            position_limit=0.5, base_position_size=0.1
        ),
        'aggressive': BacktestConfig(
            long_threshold=0.5, short_threshold=0.5,
            position_limit=0.8, base_position_size=0.15
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        try:
            # Create modified dataframe with enhanced tier confidence
            df_test = df_valid.copy()
            df_test["tier_confidence"] = df_test["enhanced_tier_confidence"]
            
            backtester = QuantileBacktester(config)
            trades_df = backtester.run_backtest(df_test, price_col='truth')
            
            results[config_name] = backtester.metrics
            
            print(f"Results:")
            print(f"   Return: {backtester.metrics['total_return']:>8.2%}")
            print(f"   Sharpe: {backtester.metrics['sharpe_ratio']:>8.3f}")
            print(f"   Max DD: {backtester.metrics['max_drawdown']:>8.2%}")
            print(f"   Trades: {backtester.metrics['total_trades']:>8d}")
            
        except Exception as e:
            print(f"Error: {e}")
            results[config_name] = {'error': str(e)}
    
    # Comparison
    print(f"\n{'='*60}")
    print("ENHANCED FEATURES PERFORMANCE")
    print(f"{'='*60}")
    
    for config_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{config_name:12} | Return: {metrics['total_return']:>7.2%} | "
                  f"Sharpe: {metrics['sharpe_ratio']:>6.2f} | "
                  f"Max DD: {metrics['max_drawdown']:>6.2%}")
    
    return results

def analyze_feature_impact():
    """Analyze the impact of individual top features"""
    
    print(f"\n{'='*60}")
    print("INDIVIDUAL FEATURE IMPACT ANALYSIS")
    print(f"{'='*60}")
    
    df_base = load_and_enhance_data()
    
    # Test individual feature correlations
    top_features = [
        '$momentum_1', '$momentum_3', 'q50', 'signal_vol_adjusted', 
        'prob_up', '$momentum_24', 'q90', 'side'
    ]
    
    print("Feature correlations with truth:")
    for feature in top_features:
        if feature in df_base.columns:
            corr = df_base[feature].corr(df_base['truth'])
            print(f"  {feature:20}: {corr:>8.4f}")
    
    # Test feature combinations
    print(f"\nFeature combination analysis:")
    
    # Momentum combination
    df_base['momentum_combo'] = (
        df_base['$momentum_1'] * 0.5 + 
        df_base['$momentum_3'] * 0.3 + 
        df_base['$momentum_24'] * 0.2
    )
    
    # Signal combination  
    df_base['signal_combo'] = (
        df_base['q50'] * 0.6 + 
        df_base['signal_vol_adjusted'] * 0.4
    )
    
    # Overall combination
    df_base['overall_combo'] = (
        df_base['momentum_combo'] * 0.6 + 
        df_base['signal_combo'] * 0.4
    )
    
    combo_features = ['momentum_combo', 'signal_combo', 'overall_combo']
    
    for feature in combo_features:
        corr = df_base[feature].corr(df_base['truth'])
        print(f"  {feature:20}: {corr:>8.4f}")
    
    return df_base

if __name__ == "__main__":
    # Run backtest comparison
    backtest_results = simple_backtest_comparison()
    
    # Analyze feature impact
    df_analysis = analyze_feature_impact()
    
    print(f"\nTop features analysis completed!")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Implement $momentum_1 and $momentum_3 immediately")
    print(f"2. Use signal_vol_adjusted instead of raw q50")
    print(f"3. Enhance tier_confidence with momentum + volatility")
    print(f"4. Test aggressive configuration for higher returns")