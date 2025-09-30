#!/usr/bin/env python3
"""
Analysis of signal threshold logic and proposed improvements
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_current_threshold_problems():
    """Analyze the fundamental issues with current threshold approach"""
    
    print("🔍 PROBLEMS WITH CURRENT THRESHOLD APPROACH")
    print("=" * 60)
    
    problems = [
        {
            "issue": "Asymmetric Logic",
            "description": "Uses abs_q50 for threshold but raw q50 for comparison",
            "example": "If q50=-0.015 and threshold=0.01, we compare -0.015 < -0.01 (False), but threshold was based on abs(q50)=0.015",
            "impact": "Inconsistent signal generation, especially for negative q50 values"
        },
        {
            "issue": "Threshold Leakage", 
            "description": "Threshold is based on future data via rolling window",
            "example": "At time t, threshold uses data from t-29 to t, but in live trading you only have data up to t-1",
            "impact": "Overfitting to future information, unrealistic backtest results"
        },
        {
            "issue": "Circular Logic",
            "description": "Threshold is 90th percentile of abs_q50, but we use it to filter q50",
            "example": "If 90% of abs_q50 values are above threshold, then 90% of signals should trigger by definition",
            "impact": "Threshold becomes meaningless - it's just a percentile filter"
        },
        {
            "issue": "No Economic Rationale",
            "description": "90th percentile is arbitrary - no connection to market dynamics or risk",
            "example": "Why 90th percentile? Why not 85th or 95th? What does this represent economically?",
            "impact": "Difficult to interpret, optimize, or adapt to different market conditions"
        },
        {
            "issue": "Ignores Prediction Uncertainty",
            "description": "Doesn't consider the spread (q90-q10) which represents prediction confidence",
            "example": "q50=0.02 with spread=0.001 vs q50=0.02 with spread=0.1 are treated the same",
            "impact": "Takes high-risk trades with uncertain predictions"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. **{problem['issue']}**")
        print(f"   Problem: {problem['description']}")
        print(f"   Example: {problem['example']}")
        print(f"   Impact: {problem['impact']}")
    
    return problems

def demonstrate_threshold_issues():
    """Create synthetic data to demonstrate the problems"""
    
    print(f"\nDEMONSTRATING THRESHOLD ISSUES")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n = 1000
    
    # Generate q50 values with different regimes
    q50_regime1 = np.random.normal(0.005, 0.002, n//3)  # Low volatility
    q50_regime2 = np.random.normal(0.000, 0.010, n//3)  # High volatility  
    q50_regime3 = np.random.normal(-0.005, 0.003, n//3) # Bearish regime
    
    q50 = np.concatenate([q50_regime1, q50_regime2, q50_regime3])
    abs_q50 = np.abs(q50)
    
    # Calculate rolling threshold (current approach)
    df = pd.DataFrame({'q50': q50, 'abs_q50': abs_q50})
    df['signal_thresh_adaptive'] = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90)
    
    # Demonstrate the problems
    print(f"Data Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   q50 range: {q50.min():.4f} to {q50.max():.4f}")
    print(f"   abs_q50 range: {abs_q50.min():.4f} to {abs_q50.max():.4f}")
    
    # Problem 1: Asymmetric logic
    valid_data = df.dropna()
    buy_signals = (valid_data['q50'] > valid_data['signal_thresh_adaptive']).sum()
    sell_signals = (valid_data['q50'] < -valid_data['signal_thresh_adaptive']).sum()
    total_signals = buy_signals + sell_signals
    
    print(f"\n🔍 Problem 1 - Asymmetric Logic:")
    print(f"   Buy signals (q50 > thresh): {buy_signals} ({buy_signals/len(valid_data)*100:.1f}%)")
    print(f"   Sell signals (q50 < -thresh): {sell_signals} ({sell_signals/len(valid_data)*100:.1f}%)")
    print(f"   Total signals: {total_signals} ({total_signals/len(valid_data)*100:.1f}%)")
    print(f"   Expected if symmetric: ~20% (10% each direction)")
    
    # Problem 2: Threshold leakage
    print(f"\n🔍 Problem 2 - Threshold Leakage:")
    print(f"   Current threshold uses future data in rolling window")
    print(f"   In live trading, threshold at time t should only use data up to t-1")
    
    # Demonstrate proper threshold (lagged)
    df['signal_thresh_proper'] = df['abs_q50'].rolling(30, min_periods=10).quantile(0.90).shift(1)
    
    valid_proper = df.dropna()
    buy_proper = (valid_proper['q50'] > valid_proper['signal_thresh_proper']).sum()
    sell_proper = (valid_proper['q50'] < -valid_proper['signal_thresh_proper']).sum()
    
    print(f"   With proper lagging:")
    print(f"   Buy signals: {buy_proper} vs {buy_signals} (difference: {buy_signals - buy_proper})")
    print(f"   Sell signals: {sell_proper} vs {sell_signals} (difference: {sell_signals - sell_proper})")
    
    # Problem 3: Circular logic
    print(f"\n🔍 Problem 3 - Circular Logic:")
    threshold_percentile = 0.90
    expected_above_thresh = (1 - threshold_percentile) * 100
    actual_above_thresh = (valid_data['abs_q50'] > valid_data['signal_thresh_adaptive']).mean() * 100
    
    print(f"   By definition, ~{expected_above_thresh:.0f}% of abs_q50 should be above 90th percentile threshold")
    print(f"   Actual: {actual_above_thresh:.1f}% (close to expected due to rolling window)")
    print(f"   This makes the threshold a tautology, not a meaningful filter")
    
    return df

def propose_better_approaches():
    """Propose better signal threshold approaches"""
    
    print(f"\n💡 BETTER SIGNAL THRESHOLD APPROACHES")
    print("=" * 60)
    
    approaches = [
        {
            "name": "Economic Significance Threshold",
            "logic": "Use fixed economic thresholds based on transaction costs and minimum profitable moves",
            "implementation": "signal_thresh = max(transaction_cost * 2, min_profit_threshold)",
            "example": "If trading costs 0.2% round-trip, require |q50| > 0.004 (0.4%) to be profitable",
            "pros": ["Economically meaningful", "No data leakage", "Interpretable"],
            "cons": ["Requires domain knowledge", "May need market-specific tuning"]
        },
        {
            "name": "Risk-Adjusted Threshold", 
            "logic": "Threshold based on prediction uncertainty (spread) and required Sharpe ratio",
            "implementation": "signal_thresh = (q90 - q10) * min_sharpe_ratio",
            "example": "If spread=0.02 and min_sharpe=1.0, require |q50| > 0.02 for signal",
            "pros": ["Adapts to prediction confidence", "Risk-aware", "No future data"],
            "cons": ["Requires Sharpe ratio assumption", "May be too conservative"]
        },
        {
            "name": "Volatility-Adjusted Threshold",
            "logic": "Scale threshold by recent market volatility using proper lag",
            "implementation": "signal_thresh = base_thresh * (current_vol / long_term_vol)",
            "example": "Base threshold 0.5%, scale by vol ratio: high vol = higher threshold",
            "pros": ["Market-adaptive", "Proper time series handling", "Intuitive"],
            "cons": ["Requires volatility estimation", "Parameter tuning needed"]
        },
        {
            "name": "Information Ratio Approach",
            "logic": "Require minimum information ratio (signal/noise) for trading",
            "implementation": "signal_thresh = abs(q50) / max(spread, min_spread) > min_info_ratio",
            "example": "Only trade if signal-to-noise ratio > 2.0",
            "pros": ["Directly measures signal quality", "No arbitrary percentiles", "Robust"],
            "cons": ["Different logic structure", "Requires ratio threshold tuning"]
        },
        {
            "name": "Regime-Aware Threshold",
            "logic": "Different thresholds for different market regimes (trending vs ranging)",
            "implementation": "signal_thresh = base_thresh * regime_multiplier[current_regime]",
            "example": "Lower threshold in trending markets, higher in ranging markets",
            "pros": ["Adapts to market conditions", "Can improve performance", "Flexible"],
            "cons": ["Complex regime detection", "More parameters", "Overfitting risk"]
        }
    ]
    
    for i, approach in enumerate(approaches, 1):
        print(f"\n{i}. **{approach['name']}**")
        print(f"   Logic: {approach['logic']}")
        print(f"   Implementation: {approach['implementation']}")
        print(f"   Example: {approach['example']}")
        print(f"   Pros: {', '.join(approach['pros'])}")
        print(f"   Cons: {', '.join(approach['cons'])}")
    
    return approaches

def create_implementation_example():
    """Create example implementation of better threshold approach"""
    
    print(f"\n🛠️ IMPLEMENTATION EXAMPLE: ECONOMIC SIGNIFICANCE")
    print("=" * 60)
    
    code_example = '''
def economic_significance_threshold(trading_costs_bps=20, min_sharpe=1.0, base_vol=0.01):
    """
    Calculate economically meaningful signal threshold
    
    Args:
        trading_costs_bps: Round-trip trading costs in basis points
        min_sharpe: Minimum required Sharpe ratio for trades
        base_vol: Expected volatility for Sharpe calculation
    """
    # Convert trading costs to decimal
    trading_costs = trading_costs_bps / 10000
    
    # Minimum move needed to cover costs
    min_profitable_move = trading_costs * 2  # 2x costs for safety margin
    
    # Sharpe-based threshold
    sharpe_threshold = min_sharpe * base_vol
    
    # Use the higher of the two
    economic_threshold = max(min_profitable_move, sharpe_threshold)
    
    return economic_threshold

def risk_adjusted_threshold(q10, q50, q90, min_info_ratio=2.0):
    """
    Calculate risk-adjusted threshold based on prediction uncertainty
    
    Args:
        q10, q50, q90: Quantile predictions
        min_info_ratio: Minimum required information ratio (signal/noise)
    """
    spread = q90 - q10
    abs_q50 = abs(q50)
    
    # Information ratio = signal strength / prediction uncertainty
    info_ratio = abs_q50 / max(spread, 0.001)  # Avoid division by zero
    
    # Only trade if information ratio exceeds minimum
    return info_ratio > min_info_ratio

def improved_signal_generation(df):
    """
    Improved signal generation without problematic thresholds
    """
    # Calculate economic threshold (fixed, no data leakage)
    economic_thresh = economic_significance_threshold()
    
    # Calculate prob_up
    df['prob_up'] = df.apply(lambda row: prob_up_piecewise(row), axis=1)
    
    # Method 1: Economic significance only
    df['buy_mask_econ'] = (df['q50'] > economic_thresh) & (df['prob_up'] > 0.5)
    df['sell_mask_econ'] = (df['q50'] < -economic_thresh) & (df['prob_up'] < 0.5)
    
    # Method 2: Risk-adjusted (no fixed threshold)
    df['trade_mask_risk'] = df.apply(
        lambda row: risk_adjusted_threshold(row['q10'], row['q50'], row['q90']), 
        axis=1
    )
    df['buy_mask_risk'] = df['trade_mask_risk'] & (df['prob_up'] > 0.5)
    df['sell_mask_risk'] = df['trade_mask_risk'] & (df['prob_up'] < 0.5)
    
    # Method 3: Combined approach
    df['buy_mask_combined'] = (
        (df['q50'] > economic_thresh) & 
        (df['prob_up'] > 0.5) & 
        df['trade_mask_risk']
    )
    df['sell_mask_combined'] = (
        (df['q50'] < -economic_thresh) & 
        (df['prob_up'] < 0.5) & 
        df['trade_mask_risk']
    )
    
    return df
'''
    
    print(code_example)
    
    return code_example

def main():
    """Main analysis function"""
    
    # Analyze problems
    problems = analyze_current_threshold_problems()
    
    # Demonstrate issues
    df = demonstrate_threshold_issues()
    
    # Propose solutions
    approaches = propose_better_approaches()
    
    # Show implementation
    code = create_implementation_example()
    
    print(f"\nRECOMMENDATIONS")
    print("=" * 60)
    print("1. **Immediate Fix**: Use economic significance threshold (e.g., 2x trading costs)")
    print("2. **Better Approach**: Implement risk-adjusted thresholds based on prediction uncertainty")
    print("3. **Advanced**: Combine economic and risk-based approaches")
    print("4. **Validation**: Backtest all approaches and compare performance")
    print("5. **Monitoring**: Track threshold effectiveness in live trading")
    
    print(f"\n📋 NEXT STEPS")
    print("1. Choose one of the proposed approaches")
    print("2. Implement and test on historical data") 
    print("3. Compare performance vs current threshold method")
    print("4. Validate that new approach doesn't have data leakage")
    print("5. Update production model with improved logic")

if __name__ == "__main__":
    main()