#!/usr/bin/env python3
"""
Test the new signal analysis CSV output functionality
"""
import pandas as pd
import numpy as np
from src.backtesting.backtester import HummingbotQuantileBacktester

def create_test_data(n_samples=1000):
    """Create synthetic test data for demonstration"""
    
    np.random.seed(42)
    
    # Create datetime index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    
    # Generate synthetic quantile data
    q50 = np.random.normal(0, 0.01, n_samples)
    q10 = q50 - np.abs(np.random.normal(0.01, 0.005, n_samples))
    q90 = q50 + np.abs(np.random.normal(0.01, 0.005, n_samples))
    
    # Generate other features
    signal_tier = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.4, 0.3])
    realized_vol_6 = np.random.uniform(0.005, 0.015, n_samples)
    
    # Create thresholds
    abs_q50 = np.abs(q50)
    signal_thresh_adaptive = pd.Series(abs_q50).rolling(30, min_periods=10).quantile(0.90).fillna(0.01)
    spread = q90 - q10
    spread_thresh = pd.Series(spread).rolling(30, min_periods=10).quantile(0.90).fillna(0.02)
    
    # Create DataFrame
    df = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'signal_tier': signal_tier,
        '$realized_vol_6': realized_vol_6,
        'signal_thresh_adaptive': signal_thresh_adaptive,
        'spread_thresh': spread_thresh,
        'truth': np.random.normal(0, 0.008, n_samples)  # For price simulation
    }, index=dates)
    
    return df

def create_test_price_data(df):
    """Create synthetic price data aligned with test data"""
    
    # Simulate price evolution
    initial_price = 50000
    returns = df['truth'].fillna(0)
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    price_df = pd.DataFrame({
        'close': prices,
        'open': np.array(prices) * (1 + np.random.normal(0, 0.001, len(prices))),
        'high': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.002, len(prices)))),
        'low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.002, len(prices)))),
        'volume': np.random.uniform(1000, 10000, len(prices))
    }, index=df.index)
    
    return price_df

def test_signal_analysis_output():
    """Test the new signal analysis functionality"""
    
    print("🧪 TESTING SIGNAL ANALYSIS OUTPUT")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating synthetic test data...")
    df = create_test_data(500)  # Smaller sample for testing
    price_data = create_test_price_data(df)
    
    print(f"   Generated {len(df)} observations")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Configure backtester
    backtester = HummingbotQuantileBacktester(
        initial_balance=100000.0,
        trading_pair="BTCUSDT",
        long_threshold=0.6,
        short_threshold=0.6,
        max_position_pct=0.3,
        fee_rate=0.001,
        sizing_method="enhanced"
    )
    
    print("\n🚀 Running backtest...")
    results = backtester.run_backtest(df, price_data=price_data, price_col='close', return_col='truth')
    
    print("\n💾 Saving results with signal analysis...")
    backtester.save_results(results, output_dir="./test_signal_analysis_results")
    
    print("\n✅ Test completed! Check the following files:")
    print("   - test_signal_analysis_results/signal_analysis_pivot.csv")
    print("   - test_signal_analysis_results/signal_summary_stats.json")
    
    # Quick preview of the signal analysis data
    try:
        signal_df = pd.read_csv("./test_signal_analysis_results/signal_analysis_pivot.csv")
        print(f"\n📋 Signal Analysis Preview:")
        print(f"   Total observations: {len(signal_df):,}")
        print(f"   Columns available: {len(signal_df.columns)}")
        print(f"   Key columns: {list(signal_df.columns[:10])}")
        
        # Show execution rate summary
        execution_rate = signal_df['trade_executed'].mean()
        print(f"   Overall execution rate: {execution_rate:.2%}")
        
        # Show signal direction distribution
        direction_dist = signal_df['signal_direction'].value_counts()
        print(f"   Signal directions: {dict(direction_dist)}")
        
        print("\n🎯 Ready for pivot table analysis!")
        
    except Exception as e:
        print(f"   ⚠️  Could not preview results: {e}")

if __name__ == "__main__":
    test_signal_analysis_output()