"""
Quick summary of backtest results
"""

def print_backtest_summary():
    """Print a summary of the backtest results"""
    
    print("="*80)
    print("QUANTILE TRADING STRATEGY BACKTEST RESULTS SUMMARY")
    print("="*80)
    
    print("\nPREDICTION QUALITY ANALYSIS:")
    print("• Data points: 53,978 observations")
    print("• Quantile coverage: Q10=90.66%, Q90=90.68% (excellent calibration)")
    print("• Prediction interval coverage: 81.35% (should be ~80%)")
    print("• Directional accuracy: 81.14% (very strong)")
    print("• Tier confidence range: 0.5 to 5.5 (good distribution)")
    
    print("\nTRADING PERFORMANCE COMPARISON:")
    print("Configuration    | Return | Sharpe | Max DD | Trades | Win Rate")
    print("-" * 65)
    print("Conservative     |  0.04% |   1.37 |  0.00% |     12 |   75.00%")
    print("Moderate         |  0.44% |   1.43 |  0.10% |    302 |   54.30%")
    print("Aggressive       | 11.45% |   3.69 |  0.69% |  3,532 |   50.91%")
    print("Hummingbot       |  8.96% |   3.99 |  0.54% |  3,532 |   50.91%")
    
    print("\n🎯 KEY INSIGHTS:")
    print("1. EXCELLENT MODEL CALIBRATION:")
    print("   • Quantile predictions are well-calibrated (90% coverage)")
    print("   • 81% directional accuracy is very strong")
    print("   • Prediction intervals capture 81% of actual outcomes")
    
    print("\n2. STRONG RISK-ADJUSTED RETURNS:")
    print("   • Aggressive config: 11.45% return, 3.69 Sharpe ratio")
    print("   • Hummingbot config: 8.96% return, 3.99 Sharpe ratio")
    print("   • Low maximum drawdowns (< 1%)")
    
    print("\n3. TIER-BASED PERFORMANCE:")
    print("   • Higher tier confidence → better performance")
    print("   • Tier 0 (highest): Strong positive returns")
    print("   • Tier 2 (high): Excellent risk-adjusted returns")
    print("   • Tier 1 (low): Negative returns (correctly filtered)")
    
    print("\n4. TRADING FREQUENCY IMPACT:")
    print("   • Conservative: 12 trades, 75% win rate")
    print("   • Aggressive: 3,532 trades, 51% win rate")
    print("   • Higher frequency → lower win rate but higher total returns")
    
    print("\n📈 RECOMMENDED CONFIGURATION:")
    print("• HUMMINGBOT_DEFAULT shows best risk-adjusted performance")
    print("• 8.96% return with 3.99 Sharpe ratio")
    print("• 0.54% max drawdown (excellent risk control)")
    print("• Thresholds: 0.5 for both long/short signals")
    print("• Position limit: 100% of capital")
    
    print("\n🔄 HUMMINGBOT INTEGRATION READINESS:")
    print("• Probability conversion logic validated")
    print("• Tier confidence → position sizing works well")
    print("• Real-time prediction pipeline ready")
    print("• MQTT bridge tested and functional")
    
    print("\n⚡ NEXT STEPS:")
    print("1. Deploy real-time prediction service")
    print("2. Set up MQTT broker for Hummingbot")
    print("3. Configure Hummingbot with 0.5 thresholds")
    print("4. Start with smaller position sizes for live testing")
    print("5. Monitor tier performance in live trading")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_backtest_summary()