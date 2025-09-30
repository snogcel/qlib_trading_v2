#!/usr/bin/env python3
"""
Test that the backtester can trade 24/7 without time restrictions
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)


import pandas as pd
import numpy as np
from src.backtesting.backtester import HummingbotQuantileBacktester

def create_strong_signal_data(n_samples=100):
    """Create test data with strong signals that should definitely execute"""
    
    np.random.seed(42)
    
    # Create datetime index covering different times including weekends
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='2h')  # Every 2 hours
    
    # Generate strong directional signals
    q50 = np.random.choice([-0.02, 0.02], n_samples)  # Strong directional signals
    q10 = q50 - 0.01  # Tight spreads
    q90 = q50 + 0.01
    
    # High confidence signals
    signal_tier = np.zeros(n_samples)  # All tier 0 (highest confidence)
    realized_vol_6 = np.full(n_samples, 0.008)
    
    # Create thresholds that should allow trading
    abs_q50 = np.abs(q50)
    signal_thresh_adaptive = np.full(n_samples, 0.005)  # Low threshold
    spread_thresh = np.full(n_samples, 0.02)
    
    # Create DataFrame
    df = pd.DataFrame({
        'q10': q10,
        'q50': q50,
        'q90': q90,
        'signal_tier': signal_tier,
        '$realized_vol_6': realized_vol_6,
        'signal_thresh_adaptive': signal_thresh_adaptive,
        'spread_thresh': spread_thresh,
        'truth': q50 * 0.5  # Correlated returns for price simulation
    }, index=dates)
    
    return df

def create_test_price_data(df):
    """Create price data that moves with the signals"""
    
    initial_price = 50000
    prices = [initial_price]
    
    for ret in df['truth'][1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    price_df = pd.DataFrame({
        'close': prices,
        'open': prices,
        'high': prices,
        'low': prices,
        'volume': np.full(len(prices), 1000)
    }, index=df.index)
    
    return price_df

def test_24_7_trading():
    """Test that trading works at all hours and days"""
    
    print("TESTING 24/7 CRYPTO TRADING")
    print("=" * 50)
    
    # Create strong signal data
    df = create_strong_signal_data(100)
    price_data = create_test_price_data(df)
    
    print(f"Created {len(df)} strong signals")
    print(f"📅 Time range: {df.index.min()} to {df.index.max()}")
    
    # Check time coverage
    hours_covered = df.index.hour.unique()
    days_covered = df.index.day_name().unique()
    
    print(f"Hours covered: {sorted(hours_covered)}")
    print(f"📅 Days covered: {list(days_covered)}")
    print(f"Weekend data: {any(day in ['Saturday', 'Sunday'] for day in days_covered)}")
    
    # Configure backtester with aggressive settings to encourage trading
    backtester = HummingbotQuantileBacktester(
        initial_balance=100000.0,
        trading_pair="BTCUSDT",
        long_threshold=0.3,  # Lower thresholds
        short_threshold=0.3,
        max_position_pct=0.5,  # Higher max position
        fee_rate=0.001,
        sizing_method="simple"  # Simple sizing for predictable behavior
    )
    
    print(f"\nRunning backtest with aggressive settings...")
    results = backtester.run_backtest(df, price_data=price_data, price_col='close', return_col='truth')
    
    # Analyze results
    total_trades = len(backtester.trades)
    total_holds = len(backtester.holds)
    execution_rate = total_trades / len(df) if len(df) > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"   Total signals: {len(df)}")
    print(f"   Trades executed: {total_trades}")
    print(f"   Holds: {total_holds}")
    print(f"   Execution rate: {execution_rate:.2%}")
    
    if total_trades > 0:
        # Analyze trade timing
        trades_df = pd.DataFrame(backtester.trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day_of_week'] = trades_df['timestamp'].dt.day_name()
        
        print(f"\nTRADE TIMING ANALYSIS:")
        print(f"   Hours with trades: {sorted(trades_df['hour'].unique())}")
        print(f"   Days with trades: {list(trades_df['day_of_week'].unique())}")
        
        weekend_trades = trades_df[trades_df['day_of_week'].isin(['Saturday', 'Sunday'])]
        print(f"   Weekend trades: {len(weekend_trades)} ({len(weekend_trades)/len(trades_df)*100:.1f}%)")
        
        # Check for any time-based patterns
        hourly_trades = trades_df['hour'].value_counts().sort_index()
        print(f"   Trades by hour: {dict(hourly_trades)}")
        
        if len(weekend_trades) > 0:
            print(f"   SUCCESS: Trading works on weekends!")
        else:
            print(f"    No weekend trades (might be due to data coverage)")
            
        if len(trades_df['hour'].unique()) > 10:
            print(f"   SUCCESS: Trading works across many hours!")
        else:
            print(f"    Limited hour coverage")
            
    else:
        print(f"   NO TRADES EXECUTED - investigating...")
        
        # Check signal characteristics
        sample_signals = []
        for i, (timestamp, row) in enumerate(df.head(5).iterrows()):
            signal = backtester.generate_hummingbot_signal(row)
            sample_signals.append({
                'timestamp': timestamp,
                'direction': signal['signal_direction'],
                'strength': signal['signal_strength'],
                'confidence': signal['confidence'],
                'target_pct': signal['target_pct'],
                'q50': row['q50'],
                'abs_q50': abs(row['q50']),
                'thresh': row['signal_thresh_adaptive']
            })
        
        print(f"\n SAMPLE SIGNALS:")
        for sig in sample_signals:
            print(f"   {sig['timestamp']}: {sig['direction']} (str={sig['strength']:.3f}, conf={sig['confidence']:.1f}, target={sig['target_pct']:.3f})")
            print(f"      q50={sig['q50']:.4f}, abs_q50={sig['abs_q50']:.4f}, thresh={sig['thresh']:.4f}")
    
    print(f"\nTest completed!")
    
    return backtester, results

if __name__ == "__main__":
    backtester, results = test_24_7_trading()