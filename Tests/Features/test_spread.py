import pandas as pd

def validate_spread_predictive_power(df):
    """
    Rigorous test of whether quantile spread has predictive value
    """
    import scipy.stats as stats
    from sklearn.metrics import mutual_info_score
    
    # Calculate spread
    df['spread'] = df['q90'] - df['q10']
    df['future_return'] = df['truth'].shift(-1)  # Next period return
    df['future_volatility'] = df['truth'].rolling(5).std().shift(-1)  # Future realized vol
    
    results = {}
    
    # Test 1: Correlation with future outcomes
    results['spread_return_corr'] = df['spread'].corr(df['future_return'])
    results['spread_vol_corr'] = df['spread'].corr(df['future_volatility'])
    
    # Test 2: Spread deciles vs future performance
    df['spread_decile'] = pd.qcut(df['spread'], 10, labels=False)
    spread_analysis = df.groupby('spread_decile').agg({
        'future_return': ['mean', 'std', 'count'],
        'future_volatility': ['mean', 'std']
    }).round(4)
    
    # Test 3: Statistical significance
    # Do high spread periods actually have different return distributions?
    high_spread = df[df['spread'] > df['spread'].quantile(0.8)]['future_return'].dropna()
    low_spread = df[df['spread'] < df['spread'].quantile(0.2)]['future_return'].dropna()
    
    t_stat, p_value = stats.ttest_ind(high_spread, low_spread)
    results['spread_ttest'] = {'t_stat': t_stat, 'p_value': p_value}
    
    # Test 4: Information content (mutual information)
    # Discretize for mutual info calculation
    spread_bins = pd.qcut(df['spread'].dropna(), 5, labels=False)
    return_bins = pd.qcut(df['future_return'].dropna(), 5, labels=False)
    
    # Align the series
    aligned_data = pd.DataFrame({
        'spread_bins': spread_bins,
        'return_bins': return_bins
    }).dropna()
    
    if len(aligned_data) > 0:
        results['mutual_info'] = mutual_info_score(
            aligned_data['spread_bins'], 
            aligned_data['return_bins']
        )
    
    # Test 5: Predictive power in different regimes
    regime_tests = {}
    for regime in ['high_vol', 'low_vol', 'fear', 'greed']:
        if regime == 'high_vol':
            mask = df['vol_scaled'] > df['vol_scaled'].quantile(0.7)
        elif regime == 'low_vol':
            mask = df['vol_scaled'] < df['vol_scaled'].quantile(0.3)
        elif regime == 'fear':
            mask = df['$fg_index'] < 0.3
        else:  # greed
            mask = df['$fg_index'] > 0.7
            
        regime_corr = df[mask]['spread'].corr(df[mask]['future_return'])
        regime_tests[regime] = regime_corr
    
    results['regime_correlations'] = regime_tests
    
    print("=== SPREAD VALIDATION RESULTS ===")
    print(f"Spread-Return Correlation: {results['spread_return_corr']:.4f}")
    print(f"Spread-Volatility Correlation: {results['spread_vol_corr']:.4f}")
    print(f"T-test p-value (high vs low spread): {results['spread_ttest']['p_value']:.4f}")
    print(f"Mutual Information: {results.get('mutual_info', 'N/A'):.4f}")
    print("\nSpread Decile Analysis:")
    print(spread_analysis)
    print("\nRegime-specific correlations:")
    for regime, corr in regime_tests.items():
        print(f"  {regime}: {corr:.4f}")
    
    return results, spread_analysis

    
if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")

    # Usage
    spread_results, spread_decile_analysis = validate_spread_predictive_power(df)

""" 
🎯 Key Insight: Spread is a Risk Measure, Not a Return Predictor
Looking at your decile analysis:

Returns are essentially flat across all spread deciles (0.0000 to 0.0002)
Volatility increases dramatically from 0.0018 (decile 0) to 0.0129 (decile 9)

Updated Recommendations:
❌ Remove spread from signal tier calculation - it doesn't predict returns
✅ Keep spread for risk management - it strongly predicts volatility
✅ Use spread in position sizing - reduce size when high uncertainty expected
✅ Simplify signal tiers - focus on abs_q50 and prob_up confidence




=== SPREAD VALIDATION RESULTS ===
Spread-Return Correlation: 0.0082
Spread-Volatility Correlation: 0.7111
T-test p-value (high vs low spread): 0.5084
Mutual Information: 0.0769

Spread Decile Analysis:
              future_return               future_volatility
                       mean     std count              mean     std
spread_decile
0                    0.0000  0.0027  5398            0.0018  0.0016
1                    0.0000  0.0038  5398            0.0027  0.0021
2                    0.0000  0.0045  5397            0.0033  0.0024
3                    0.0000  0.0048  5397            0.0037  0.0024
4                    0.0001  0.0055  5398            0.0043  0.0029
5                    0.0001  0.0058  5398            0.0049  0.0029
6                    0.0001  0.0066  5397            0.0055  0.0031
7                    0.0002  0.0072  5398            0.0063  0.0036
8                    0.0002  0.0086  5398            0.0078  0.0041
9                    0.0000  0.0145  5398            0.0129  0.0083

Regime-specific correlations:
  high_vol: 0.0425
  low_vol: 0.0139
  fear: 0.0111
  greed: -0.0046 """