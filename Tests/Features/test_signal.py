import pandas as pd

def validate_thresholds(df):
    """
    Test if your signal_thresh and spread_thresh actually improve performance
    """
    # Calculate current thresholds (your rolling percentile method)
    df['signal_thresh'] = df['abs_q50'].rolling(30, min_periods=10).quantile(0.85)
    df['spread_thresh'] = df['spread'].rolling(30, min_periods=10).quantile(0.85)
    
    # Test different threshold percentiles
    threshold_tests = {}
    percentiles = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for pct in percentiles:
        signal_thresh_test = df['abs_q50'].rolling(30, min_periods=10).quantile(pct)
        spread_thresh_test = df['spread'].rolling(30, min_periods=10).quantile(pct)
        
        # Create signal flags
        strong_signal = df['abs_q50'] >= signal_thresh_test
        tight_spread = df['spread'] < spread_thresh_test
        combined_signal = strong_signal & tight_spread
        
        # Test performance
        future_returns = df['truth'].shift(-1)
        
        # Performance when signal is triggered
        signal_performance = future_returns[combined_signal].mean()
        signal_count = combined_signal.sum()
        signal_hit_rate = (future_returns[combined_signal] > 0).mean()
        
        threshold_tests[pct] = {
            'avg_return': signal_performance,
            'count': signal_count,
            'hit_rate': signal_hit_rate,
            'sharpe': signal_performance / future_returns[combined_signal].std() if signal_count > 10 else 0
        }
    
    # Test no thresholds (baseline)
    baseline_performance = future_returns.mean()
    baseline_hit_rate = (future_returns > 0).mean()
    
    print("=== THRESHOLD VALIDATION RESULTS ===")
    print(f"Baseline (no filter): Return={baseline_performance:.4f}, Hit Rate={baseline_hit_rate:.2%}")
    print("\nThreshold Performance:")
    for pct, metrics in threshold_tests.items():
        print(f"  {pct:.0%} threshold: Return={metrics['avg_return']:.4f}, "
              f"Count={metrics['count']}, Hit Rate={metrics['hit_rate']:.2%}, "
              f"Sharpe={metrics['sharpe']:.2f}")
    
    return threshold_tests


if __name__ == "__main__":
    
    df = pd.read_csv("./df_all_macro_analysis.csv")

    # Usage
    threshold_results = validate_thresholds(df)