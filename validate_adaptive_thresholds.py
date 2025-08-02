#!/usr/bin/env python3
"""
Validate adaptive threshold effectiveness
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_adaptive_thresholds(df):
    """Test if adaptive thresholds improve signal quality"""
    
    # Debug: Check data quality first
    print("🔍 DATA QUALITY & SCALE CHECK:")
    print(f"abs_q50 range: {df['abs_q50'].min():.6f} to {df['abs_q50'].max():.6f}")
    print(f"abs_q50 NaN count: {df['abs_q50'].isna().sum()}")
    print(f"abs_q50 zero count: {(df['abs_q50'] == 0).sum()}")
    print(f"vol_raw range: {df['vol_raw'].min():.6f} to {df['vol_raw'].max():.6f}")
    
    # Scale comparison
    if 'q50' in df.columns:
        print(f"q50 range: {df['q50'].min():.6f} to {df['q50'].max():.6f}")
        print(f"abs_q50 vs q50 correlation: {df['abs_q50'].corr(df['q50'].abs()):.3f}")
    
    if 'truth' in df.columns:
        print(f"truth range: {df['truth'].min():.6f} to {df['truth'].max():.6f}")
        print(f"truth mean: {df['truth'].mean():.6f} (should be ~0 for balanced data)")
    
    print()
    
    # Calculate both static and adaptive thresholds
    static_threshold = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90)
    
    # Adaptive thresholds (updated to match main script - use vol_raw instead of vol_risk)
    conditions = [
        df['vol_raw'].rolling(168).mean() > df['vol_raw'].rolling(168*4).quantile(0.8),  # Weekly vs monthly
        df['vol_raw'].rolling(168).mean() > df['vol_raw'].rolling(168*4).quantile(0.6),
        df['vol_raw'].rolling(168).mean() < df['vol_raw'].rolling(168*4).quantile(0.3),
    ]
    
    # Adjust thresholds based on volatility regime
    # threshold_adjustments = [
    #     0.85,  # Lower threshold in very high vol (more opportunities)
    #     0.88,  # Slightly lower in high vol
    #     0.95,  # Higher threshold in low vol (be more selective)
    # ]

    # Maybe high vol periods need HIGHER thresholds (be more selective)
    threshold_adjustments = [0.90, 0.90, 0.90]  # [very_high, high, low]
    
    # Pre-calculate the rolling quantiles based on threshold_adjustments
    rolling_adj0 = df['abs_q50'].rolling(30, min_periods=10).quantile(threshold_adjustments[0])  # Very high vol
    rolling_adj1 = df['abs_q50'].rolling(30, min_periods=10).quantile(threshold_adjustments[1])  # High vol
    rolling_adj2 = df['abs_q50'].rolling(30, min_periods=10).quantile(threshold_adjustments[2])  # Low vol
    
    # Debug: Check if rolling quantiles are working
    print("🔍 ROLLING QUANTILE CHECK:")
    print(f"Threshold adjustments: {threshold_adjustments}")
    print(f"rolling_adj0 ({threshold_adjustments[0]}) NaN count: {rolling_adj0.isna().sum()}")
    print(f"rolling_adj1 ({threshold_adjustments[1]}) NaN count: {rolling_adj1.isna().sum()}")
    print(f"rolling_adj2 ({threshold_adjustments[2]}) NaN count: {rolling_adj2.isna().sum()}")
    print(f"rolling_adj0 range: {rolling_adj0.min():.6f} to {rolling_adj0.max():.6f}")
    print(f"rolling_adj1 range: {rolling_adj1.min():.6f} to {rolling_adj1.max():.6f}")
    print(f"rolling_adj2 range: {rolling_adj2.min():.6f} to {rolling_adj2.max():.6f}")
    print()
    
    # Debug: Check condition matching
    print("🔍 CONDITION MATCHING CHECK:")
    for i, cond in enumerate(conditions):
        print(f"Condition {i} matches: {cond.sum()} periods ({cond.mean():.1%})")
    print()
    
    # Use a more explicit approach instead of np.select
    adaptive_threshold = static_threshold.copy()  # Start with static as base
    
    # Apply adjustments in order (most specific first)
    adaptive_threshold.loc[conditions[2]] = rolling_adj2.loc[conditions[2]]  # Low vol: use threshold_adjustments[2]
    adaptive_threshold.loc[conditions[1]] = rolling_adj1.loc[conditions[1]]  # High vol: use threshold_adjustments[1]  
    adaptive_threshold.loc[conditions[0]] = rolling_adj0.loc[conditions[0]]  # Very high vol: use threshold_adjustments[0]
    
    # Generate signals
    static_signals = df['abs_q50'] > static_threshold
    adaptive_signals = df['abs_q50'] > adaptive_threshold
    
    # Test 1: Signal frequency by regime
    print("📊 SIGNAL FREQUENCY BY VOLATILITY REGIME:")
    print("-" * 50)
    
    vol_regimes = {
        'Very High Vol': df['vol_raw'] > df['vol_raw'].quantile(0.8),
        'High Vol': (df['vol_raw'] > df['vol_raw'].quantile(0.6)) & (df['vol_raw'] <= df['vol_raw'].quantile(0.8)),
        'Medium Vol': (df['vol_raw'] >= df['vol_raw'].quantile(0.3)) & (df['vol_raw'] <= df['vol_raw'].quantile(0.6)),
        'Low Vol': df['vol_raw'] < df['vol_raw'].quantile(0.3)
    }
    
    for regime_name, regime_mask in vol_regimes.items():
        if regime_mask.sum() > 0:
            static_freq = static_signals[regime_mask].mean()
            adaptive_freq = adaptive_signals[regime_mask].mean()
            print(f"{regime_name:12}: Static={static_freq:.3f}, Adaptive={adaptive_freq:.3f}, Diff={adaptive_freq-static_freq:+.3f}")
    
    # Test 2: Threshold values by regime
    print(f"\n🎯 THRESHOLD VALUES BY REGIME:")
    print("-" * 50)
    
    for regime_name, regime_mask in vol_regimes.items():
        if regime_mask.sum() > 0:
            static_avg = static_threshold[regime_mask].mean()
            adaptive_avg = adaptive_threshold[regime_mask].mean()
            # Handle NaN values
            if pd.isna(static_avg):
                static_avg = 0
            if pd.isna(adaptive_avg):
                adaptive_avg = 0
            diff = adaptive_avg - static_avg
            print(f"{regime_name:12}: Static={static_avg:.6f}, Adaptive={adaptive_avg:.6f}, Diff={diff:+.6f}")
    
    # Test 3: Signal performance using 'truth' column
    if 'truth' in df.columns:
        print(f"\n💰 SIGNAL PERFORMANCE (using 'truth' column):")
        print("-" * 50)
        
        # Overall performance - need to account for direction
        # For fair comparison, let's calculate directional returns for both static and adaptive
        
        # Static directional returns
        static_directional_returns = []
        for idx, row in df.iterrows():
            if static_signals.loc[idx]:  # If static signal triggered
                if 'side' in df.columns:
                    if row['side'] == 1:  # BUY
                        static_directional_returns.append(row['truth'])
                    elif row['side'] == 0:  # SELL
                        static_directional_returns.append(-row['truth'])
                    # HOLD contributes nothing
                else:
                    # Fallback: assume all signals are long if no side info
                    static_directional_returns.append(row['truth'])
        
        # Adaptive directional returns  
        adaptive_directional_returns = []
        for idx, row in df.iterrows():
            if adaptive_signals.loc[idx]:  # If adaptive signal triggered
                if 'side' in df.columns:
                    if row['side'] == 1:  # BUY
                        adaptive_directional_returns.append(row['truth'])
                    elif row['side'] == 0:  # SELL
                        adaptive_directional_returns.append(-row['truth'])
                    # HOLD contributes nothing
                else:
                    # Fallback: assume all signals are long if no side info
                    adaptive_directional_returns.append(row['truth'])
        
        static_returns = np.mean(static_directional_returns) if static_directional_returns else 0
        adaptive_returns = np.mean(adaptive_directional_returns) if adaptive_directional_returns else 0
        
        print(f"Static Signal Avg Return (Directional):   {static_returns:.6f}")
        print(f"Adaptive Signal Avg Return (Directional): {adaptive_returns:.6f}")
        print(f"Improvement:                              {adaptive_returns - static_returns:+.6f}")
        
        # Also show the old long-only comparison for reference
        static_long_only = df.loc[static_signals, 'truth'].mean()
        adaptive_long_only = df.loc[adaptive_signals, 'truth'].mean()
        print(f"\nFor comparison (Long-only assumption):")
        print(f"Static Long-only:   {static_long_only:.6f}")
        print(f"Adaptive Long-only: {adaptive_long_only:.6f}")
        
        # Side-based performance (if available)
        if 'side' in df.columns:
            print(f"\n🎯 DIRECTIONAL TRADING PERFORMANCE:")
            print("-" * 50)
            
            # Calculate directional returns
            buy_signals = df['side'] == 1
            sell_signals = df['side'] == 0
            hold_signals = df['side'] == -1
            
            if buy_signals.sum() > 0:
                buy_returns = df.loc[buy_signals, 'truth'].mean()
                buy_hit_rate = (df.loc[buy_signals, 'truth'] > 0).mean()
                print(f"BUY signals:  Avg Return={buy_returns:.6f}, Hit Rate={buy_hit_rate:.1%}, Count={buy_signals.sum()}")
            
            if sell_signals.sum() > 0:
                # For sell signals, we want negative returns to be profitable
                sell_returns = -df.loc[sell_signals, 'truth'].mean()  # Flip sign for short positions
                sell_hit_rate = (df.loc[sell_signals, 'truth'] < 0).mean()  # Profitable when truth < 0
                print(f"SELL signals: Avg Return={sell_returns:.6f}, Hit Rate={sell_hit_rate:.1%}, Count={sell_signals.sum()}")
            
            if hold_signals.sum() > 0:
                hold_returns = df.loc[hold_signals, 'truth'].mean()
                print(f"HOLD periods: Avg Return={hold_returns:.6f}, Count={hold_signals.sum()}")
            
            # Overall directional strategy performance
            directional_returns = []
            for idx, row in df.iterrows():
                if row['side'] == 1:  # BUY
                    directional_returns.append(row['truth'])
                elif row['side'] == 0:  # SELL
                    directional_returns.append(-row['truth'])  # Short position
                # HOLD contributes 0 to returns
            
            if directional_returns:
                avg_directional_return = np.mean(directional_returns)
                print(f"\nOverall Directional Strategy: {avg_directional_return:.6f}")
                print(f"vs Buy & Hold: {df['truth'].mean():.6f}")
                print(f"Directional Advantage: {avg_directional_return - df['truth'].mean():+.6f}")
        
        # Performance by regime
        print(f"\nPerformance by Volatility Regime:")
        for regime_name, regime_mask in vol_regimes.items():
            if regime_mask.sum() > 0:
                static_regime_signals = static_signals & regime_mask
                adaptive_regime_signals = adaptive_signals & regime_mask
                
                if static_regime_signals.sum() > 0:
                    static_regime_return = df.loc[static_regime_signals, 'truth'].mean()
                else:
                    static_regime_return = 0
                    
                if adaptive_regime_signals.sum() > 0:
                    adaptive_regime_return = df.loc[adaptive_regime_signals, 'truth'].mean()
                else:
                    adaptive_regime_return = 0
                
                improvement = adaptive_regime_return - static_regime_return
                print(f"  {regime_name:12}: Static={static_regime_return:.6f}, Adaptive={adaptive_regime_return:.6f}, Diff={improvement:+.6f}")
        
        # Hit rates
        print(f"\nSignal Hit Rates (% positive returns):")
        static_hit_rate = (df.loc[static_signals, 'truth'] > 0).mean()
        adaptive_hit_rate = (df.loc[adaptive_signals, 'truth'] > 0).mean()
        print(f"Static Hit Rate:  {static_hit_rate:.1%}")
        print(f"Adaptive Hit Rate: {adaptive_hit_rate:.1%}")
        print(f"Hit Rate Improvement: {adaptive_hit_rate - static_hit_rate:+.1%}")
    
    # Test 4: Regime switching frequency and duration analysis
    print(f"\n🔄 REGIME SWITCHING ANALYSIS:")
    print("-" * 50)
    
    # Create regime labels
    regime_labels = np.select(
        [conditions[0], conditions[1], conditions[2]],
        ['very_high', 'high', 'low'],
        default='medium'
    )
    
    # Count switches
    regime_changes = (pd.Series(regime_labels) != pd.Series(regime_labels).shift(1)).sum()
    total_periods = len(regime_labels)
    
    print(f"Total regime switches: {regime_changes}")
    print(f"Switch frequency: {regime_changes/total_periods:.3f} (switches per period)")
    print(f"Average regime duration: {total_periods/regime_changes:.1f} periods")
    
    # Regime distribution
    regime_dist = pd.Series(regime_labels).value_counts(normalize=True)
    print(f"\nRegime distribution:")
    for regime, pct in regime_dist.items():
        print(f"  {regime}: {pct:.1%}")
    
    # Regime duration analysis
    print(f"\nRegime Duration Analysis:")
    regime_series = pd.Series(regime_labels)
    regime_changes_idx = regime_series != regime_series.shift(1)
    regime_groups = regime_changes_idx.cumsum()
    
    duration_stats = {}
    for regime in ['very_high', 'high', 'medium', 'low']:
        regime_mask = regime_series == regime
        if regime_mask.sum() > 0:
            regime_durations = regime_mask.groupby(regime_groups).sum()
            regime_durations = regime_durations[regime_durations > 0]  # Only actual regime periods
            
            duration_stats[regime] = {
                'mean': regime_durations.mean(),
                'median': regime_durations.median(),
                'max': regime_durations.max(),
                'min': regime_durations.min()
            }
            
            print(f"  {regime:10}: Avg={duration_stats[regime]['mean']:.1f}h, "
                  f"Median={duration_stats[regime]['median']:.1f}h, "
                  f"Max={duration_stats[regime]['max']:.0f}h, "
                  f"Min={duration_stats[regime]['min']:.0f}h")
    
    # Time-based regime analysis (if datetime available)
    if 'datetime' in df.columns:
        print(f"\nRegime Evolution Over Time:")
        df_with_regime = df.copy()
        df_with_regime['regime'] = regime_labels
        df_with_regime['year'] = pd.to_datetime(df_with_regime['datetime']).dt.year
        
        yearly_regimes = df_with_regime.groupby(['year', 'regime']).size().unstack(fill_value=0)
        yearly_regimes_pct = yearly_regimes.div(yearly_regimes.sum(axis=1), axis=0) * 100
        
        for year in sorted(yearly_regimes_pct.index):
            year_data = yearly_regimes_pct.loc[year]
            print(f"  {year}: ", end="")
            for regime in ['very_high', 'high', 'medium', 'low']:
                if regime in year_data:
                    print(f"{regime}={year_data[regime]:.0f}% ", end="")
            print()
    
    # Export raw data for pivot table analysis
    print(f"\n📊 EXPORTING DATA FOR PIVOT ANALYSIS:")
    print("-" * 50)
    
    # Create comprehensive analysis dataset
    analysis_df = df.copy()
    
    # Add calculated fields
    analysis_df['static_threshold'] = static_threshold
    analysis_df['adaptive_threshold'] = adaptive_threshold
    analysis_df['static_signal'] = static_signals.astype(int)
    analysis_df['adaptive_signal'] = adaptive_signals.astype(int)
    analysis_df['regime'] = regime_labels
    
    # Add regime flags for easier filtering
    analysis_df['is_very_high_vol'] = (regime_labels == 'very_high').astype(int)
    analysis_df['is_high_vol'] = (regime_labels == 'high').astype(int)
    analysis_df['is_medium_vol'] = (regime_labels == 'medium').astype(int)
    analysis_df['is_low_vol'] = (regime_labels == 'low').astype(int)
    
    # Add time-based fields for pivot analysis
    if 'datetime' in df.columns:
        analysis_df['datetime_parsed'] = pd.to_datetime(analysis_df['datetime'])
        analysis_df['year'] = analysis_df['datetime_parsed'].dt.year
        analysis_df['month'] = analysis_df['datetime_parsed'].dt.month
        analysis_df['quarter'] = analysis_df['datetime_parsed'].dt.quarter
        analysis_df['day_of_week'] = analysis_df['datetime_parsed'].dt.dayofweek
        analysis_df['hour'] = analysis_df['datetime_parsed'].dt.hour
    
    # Add performance metrics
    if 'truth' in df.columns:
        # Calculate directional positions (long/short based on q50 sign)
        analysis_df['static_position'] = np.where(
            analysis_df['static_signal'] == 1,
            np.sign(analysis_df['q50']),  # Take position in direction of prediction
            0  # No position when no signal
        )
        analysis_df['adaptive_position'] = np.where(
            analysis_df['adaptive_signal'] == 1,
            np.sign(analysis_df['q50']),
            0
        )
        
        # Calculate returns properly (accounting for long/short)
        analysis_df['static_signal_return'] = analysis_df['truth'] * analysis_df['static_position']
        analysis_df['adaptive_signal_return'] = analysis_df['truth'] * analysis_df['adaptive_position']
        
        # Hit rates: correct directional prediction
        analysis_df['static_hit'] = ((analysis_df['truth'] * analysis_df['static_position']) > 0).astype(int)
        analysis_df['adaptive_hit'] = ((analysis_df['truth'] * analysis_df['adaptive_position']) > 0).astype(int)
    
    # Add volatility quantile buckets for analysis
    analysis_df['vol_raw_decile'] = pd.qcut(analysis_df['vol_raw'], 10, labels=False, duplicates='drop') + 1
    analysis_df['abs_q50_decile'] = pd.qcut(analysis_df['abs_q50'], 10, labels=False, duplicates='drop') + 1
    
    # Add threshold difference metrics
    analysis_df['threshold_diff'] = analysis_df['adaptive_threshold'] - analysis_df['static_threshold']
    analysis_df['threshold_diff_pct'] = (analysis_df['threshold_diff'] / analysis_df['static_threshold']) * 100
    
    # Select key columns for pivot analysis
    pivot_columns = [
        # Identifiers
        'datetime', 'year', 'month', 'quarter', 'hour', 'day_of_week',
        
        # Core features
        'abs_q50', 'vol_raw', 'truth',
        
        # Regime information
        'regime', 'is_very_high_vol', 'is_high_vol', 'is_medium_vol', 'is_low_vol',
        
        # Thresholds
        'static_threshold', 'adaptive_threshold', 'threshold_diff', 'threshold_diff_pct',
        
        # Signals
        'static_signal', 'adaptive_signal',
        
        # Performance (if available)
        'static_position', 'adaptive_position', 'static_signal_return', 'adaptive_signal_return', 'static_hit', 'adaptive_hit',
        
        # Directional trading
        'side',
        
        # Bucketed analysis
        'vol_raw_decile', 'abs_q50_decile'
    ]
    
    # Filter to only include columns that exist
    available_columns = [col for col in pivot_columns if col in analysis_df.columns]
    pivot_df = analysis_df[available_columns].copy()
    
    # Export to CSV
    output_file = 'validate_adaptive_thresholds.csv'
    pivot_df.to_csv(output_file, index=False)
    
    print(f"✅ Exported {len(pivot_df)} rows to '{output_file}'")
    print(f"📋 Columns included: {len(available_columns)}")
    print(f"🔍 Key pivot fields:")
    print(f"   • Regime analysis: regime, is_*_vol columns")
    print(f"   • Time analysis: year, month, quarter, hour")
    print(f"   • Performance: *_signal_return, *_hit columns")
    print(f"   • Threshold analysis: threshold_diff, threshold_diff_pct")
    
    # Create a summary table for quick pivot reference
    summary_data = []
    
    # Overall performance
    if 'truth' in df.columns:
        summary_data.append({
            'analysis_type': 'Overall',
            'category': 'All Data',
            'static_avg_return': df.loc[static_signals, 'truth'].mean(),
            'adaptive_avg_return': df.loc[adaptive_signals, 'truth'].mean(),
            'static_hit_rate': (df.loc[static_signals, 'truth'] > 0).mean(),
            'adaptive_hit_rate': (df.loc[adaptive_signals, 'truth'] > 0).mean(),
            'static_signal_count': static_signals.sum(),
            'adaptive_signal_count': adaptive_signals.sum(),
            'total_periods': len(df)
        })
        
        # By regime
        vol_regimes = {
            'Very High Vol': df['vol_raw'] > df['vol_raw'].quantile(0.8),
            'High Vol': (df['vol_raw'] > df['vol_raw'].quantile(0.6)) & (df['vol_raw'] <= df['vol_raw'].quantile(0.8)),
            'Medium Vol': (df['vol_raw'] >= df['vol_raw'].quantile(0.3)) & (df['vol_raw'] <= df['vol_raw'].quantile(0.6)),
            'Low Vol': df['vol_raw'] < df['vol_raw'].quantile(0.3)
        }
        
        for regime_name, regime_mask in vol_regimes.items():
            if regime_mask.sum() > 0:
                static_regime_signals = static_signals & regime_mask
                adaptive_regime_signals = adaptive_signals & regime_mask
                
                summary_data.append({
                    'analysis_type': 'By Regime',
                    'category': regime_name,
                    'static_avg_return': df.loc[static_regime_signals, 'truth'].mean() if static_regime_signals.sum() > 0 else 0,
                    'adaptive_avg_return': df.loc[adaptive_regime_signals, 'truth'].mean() if adaptive_regime_signals.sum() > 0 else 0,
                    'static_hit_rate': (df.loc[static_regime_signals, 'truth'] > 0).mean() if static_regime_signals.sum() > 0 else 0,
                    'adaptive_hit_rate': (df.loc[adaptive_regime_signals, 'truth'] > 0).mean() if adaptive_regime_signals.sum() > 0 else 0,
                    'static_signal_count': static_regime_signals.sum(),
                    'adaptive_signal_count': adaptive_regime_signals.sum(),
                    'total_periods': regime_mask.sum()
                })
    
    # Export summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = 'validate_adaptive_thresholds_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Exported summary table to '{summary_file}'")
    
    return {
        'static_threshold': static_threshold,
        'adaptive_threshold': adaptive_threshold,
        'static_signals': static_signals,
        'adaptive_signals': adaptive_signals,
        'regime_labels': regime_labels,
        'pivot_df': pivot_df,
        'summary_df': summary_df if summary_data else None
    }

if __name__ == "__main__":
    # You'd load your actual data here
    print("Run this with your actual DataFrame to validate adaptive thresholds")
    print("Example: results = validate_adaptive_thresholds(df)")
    df = pd.read_csv("df_all_macro_analysis.csv")
    validate_adaptive_thresholds(df)