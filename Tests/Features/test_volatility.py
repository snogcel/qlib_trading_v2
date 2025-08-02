import pandas as pd

def validate_volatility_features(df):
    """
    Comprehensive test of different volatility measures and transformations
    """
    import scipy.stats as stats
    from sklearn.metrics import mutual_info_score
    
    # Create all volatility variations to test
    rolling_window = 30  # Adjust based on your setting
    
    # 1. Different time horizons
    vol_variations = {}
    for period in [3, 6, 9, 12, 24]:
        col_name = f"$realized_vol_{period}"
        if col_name in df.columns:
            # Vol_scaled version
            q_low = df[col_name].rolling(rolling_window).quantile(0.01)
            q_high = df[col_name].rolling(rolling_window).quantile(0.99)
            vol_variations[f'vol_scaled_{period}'] = (
                (df[col_name] - q_low.shift(1)) / (q_high.shift(1) - q_low.shift(1))
            ).clip(0.0, 1.0)
            
            # Vol_rank version
            vol_variations[f'vol_rank_{period}'] = df[col_name].rolling(rolling_window).rank(pct=True)
            
            # Raw volatility
            vol_variations[f'vol_raw_{period}'] = df[col_name]
    
    # 2. ATR-based measures (if available)
    if '$approx_atr_24' in df.columns:
        atr_col = '$approx_atr_24'
        q_low_atr = df[atr_col].rolling(rolling_window).quantile(0.01)
        q_high_atr = df[atr_col].rolling(rolling_window).quantile(0.99)
        vol_variations['atr_scaled'] = (
            (df[atr_col] - q_low_atr.shift(1)) / (q_high_atr.shift(1) - q_low_atr.shift(1))
        ).clip(0.0, 1.0)
        vol_variations['atr_rank'] = df[atr_col].rolling(rolling_window).rank(pct=True)
        vol_variations['atr_raw'] = df[atr_col]
    
    # Test each variation
    results = {}
    future_returns = df['truth'].shift(-1)
    future_volatility = df['truth'].rolling(5).std().shift(-1)
    
    print("=== VOLATILITY FEATURE VALIDATION ===")
    print(f"{'Feature':<20} {'Return_Corr':<12} {'Vol_Corr':<10} {'Mutual_Info':<12} {'Regime_Diff':<12}")
    print("-" * 70)
    
    for feature_name, feature_values in vol_variations.items():
        if feature_values.isna().all():
            continue
            
        # Basic correlations
        return_corr = feature_values.corr(future_returns)
        vol_corr = feature_values.corr(future_volatility)
        
        # Mutual information (discretized)
        try:
            feature_bins = pd.qcut(feature_values.dropna(), 5, labels=False, duplicates='drop')
            return_bins = pd.qcut(future_returns.dropna(), 5, labels=False, duplicates='drop')
            
            # Align the series
            aligned_data = pd.DataFrame({
                'feature_bins': feature_bins,
                'return_bins': return_bins
            }).dropna()
            
            if len(aligned_data) > 10:
                mutual_info = mutual_info_score(aligned_data['feature_bins'], aligned_data['return_bins'])
            else:
                mutual_info = 0
        except:
            mutual_info = 0
        
        # Regime differentiation test
        high_vol = feature_values > feature_values.quantile(0.8)
        low_vol = feature_values < feature_values.quantile(0.2)
        
        high_vol_returns = future_returns[high_vol].mean()
        low_vol_returns = future_returns[low_vol].mean()
        regime_diff = abs(high_vol_returns - low_vol_returns)
        
        results[feature_name] = {
            'return_corr': return_corr,
            'vol_corr': vol_corr,
            'mutual_info': mutual_info,
            'regime_diff': regime_diff
        }
        
        print(f"{feature_name:<20} {return_corr:<12.4f} {vol_corr:<10.4f} {mutual_info:<12.4f} {regime_diff:<12.4f}")
    
    return results

def detailed_vol_scaled_vs_rank_test(df):
    """
    Specific comparison between vol_scaled and vol_rank approaches
    """
    rolling_window = 30
    
    # Create both versions for comparison
    vol_col = "$realized_vol_6"  # Your current choice
    
    # Vol_scaled (your current method)
    q_low = df[vol_col].rolling(rolling_window).quantile(0.01)
    q_high = df[vol_col].rolling(rolling_window).quantile(0.99)
    vol_scaled = ((df[vol_col] - q_low.shift(1)) / (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)
    
    # Vol_rank (alternative method)
    vol_rank = df[vol_col].rolling(rolling_window).rank(pct=True)
    
    future_returns = df['truth'].shift(-1)
    future_volatility = df['truth'].rolling(5).std().shift(-1)
    
    print("\n=== VOL_SCALED vs VOL_RANK DETAILED COMPARISON ===")
    
    # 1. Basic statistics
    print(f"Vol_scaled - Mean: {vol_scaled.mean():.4f}, Std: {vol_scaled.std():.4f}")
    print(f"Vol_rank   - Mean: {vol_rank.mean():.4f}, Std: {vol_rank.std():.4f}")
    
    # 2. Correlations
    print(f"\nCorrelations with future returns:")
    print(f"Vol_scaled: {vol_scaled.corr(future_returns):.4f}")
    print(f"Vol_rank:   {vol_rank.corr(future_returns):.4f}")
    
    print(f"\nCorrelations with future volatility:")
    print(f"Vol_scaled: {vol_scaled.corr(future_volatility):.4f}")
    print(f"Vol_rank:   {vol_rank.corr(future_volatility):.4f}")
    
    # 3. Decile analysis
    print(f"\n=== DECILE ANALYSIS ===")
    
    # Vol_scaled deciles
    vol_scaled_deciles = pd.qcut(vol_scaled, 10, labels=False)
    scaled_analysis = pd.DataFrame({
        'vol_scaled_decile': vol_scaled_deciles,
        'future_return': future_returns,
        'future_vol': future_volatility
    }).groupby('vol_scaled_decile').agg({
        'future_return': ['mean', 'std', 'count'],
        'future_vol': ['mean', 'std']
    }).round(4)
    
    # Vol_rank deciles  
    vol_rank_deciles = pd.qcut(vol_rank, 10, labels=False)
    rank_analysis = pd.DataFrame({
        'vol_rank_decile': vol_rank_deciles,
        'future_return': future_returns,
        'future_vol': future_volatility
    }).groupby('vol_rank_decile').agg({
        'future_return': ['mean', 'std', 'count'],
        'future_vol': ['mean', 'std']
    }).round(4)
    
    print("Vol_scaled decile analysis:")
    print(scaled_analysis)
    print("\nVol_rank decile analysis:")
    print(rank_analysis)
    
    # 4. Extreme regime detection
    print(f"\n=== EXTREME REGIME DETECTION ===")
    
    # High volatility regimes
    high_vol_scaled = vol_scaled > vol_scaled.quantile(0.9)
    high_vol_rank = vol_rank > vol_rank.quantile(0.9)
    
    print(f"High vol_scaled regime - Avg return: {future_returns[high_vol_scaled].mean():.4f}, Count: {high_vol_scaled.sum()}")
    print(f"High vol_rank regime   - Avg return: {future_returns[high_vol_rank].mean():.4f}, Count: {high_vol_rank.sum()}")
    
    # 5. Stability test (how consistent are the rankings?)
    correlation_between_methods = vol_scaled.corr(vol_rank)
    print(f"\nCorrelation between vol_scaled and vol_rank: {correlation_between_methods:.4f}")
    
    return {
        'vol_scaled_analysis': scaled_analysis,
        'vol_rank_analysis': rank_analysis,
        'method_correlation': correlation_between_methods
    }

def test_optimal_volatility_horizon(df):
    """
    Test which volatility time horizon works best
    """
    horizons = [3, 6, 9, 12, 24]
    results = {}
    
    print("\n=== OPTIMAL VOLATILITY HORIZON TEST ===")
    print(f"{'Horizon':<10} {'Return_Corr':<12} {'Vol_Pred_Corr':<15} {'Regime_Diff':<12}")
    print("-" * 50)
    
    future_returns = df['truth'].shift(-1)
    future_volatility = df['truth'].rolling(5).std().shift(-1)
    
    for horizon in horizons:
        vol_col = f"$realized_vol_{horizon}"
        if vol_col not in df.columns:
            continue
            
        # Create vol_scaled for this horizon
        rolling_window = 30
        q_low = df[vol_col].rolling(rolling_window).quantile(0.01)
        q_high = df[vol_col].rolling(rolling_window).quantile(0.99)
        vol_scaled_h = ((df[vol_col] - q_low.shift(1)) / (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)
        
        # Test performance
        return_corr = vol_scaled_h.corr(future_returns)
        vol_pred_corr = vol_scaled_h.corr(future_volatility)
        
        # Regime differentiation
        high_vol = vol_scaled_h > vol_scaled_h.quantile(0.8)
        low_vol = vol_scaled_h < vol_scaled_h.quantile(0.2)
        regime_diff = abs(future_returns[high_vol].mean() - future_returns[low_vol].mean())
        
        results[horizon] = {
            'return_corr': return_corr,
            'vol_pred_corr': vol_pred_corr,
            'regime_diff': regime_diff
        }
        
        print(f"{horizon:<10} {return_corr:<12.4f} {vol_pred_corr:<15.4f} {regime_diff:<12.4f}")
    
    return results


if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")

    # Usage
    print("Running comprehensive volatility validation...")
    vol_results = validate_volatility_features(df)
    detailed_comparison = detailed_vol_scaled_vs_rank_test(df)
    horizon_results = test_optimal_volatility_horizon(df)
