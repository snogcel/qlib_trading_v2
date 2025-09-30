"""
Test the backtest fix for JSON serialization
"""

import pandas as pd
import numpy as np
from quantile_backtester import QuantileBacktester, BacktestConfig

def test_json_serialization_fix():
    """Test that the JSON serialization fix works"""
    
    print("Testing JSON serialization fix...")
    
    # Load data
    df = pd.read_csv("df_all_macro_analysis.csv")
    
    if 'instrument' in df.columns and 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index(['instrument', 'datetime'])
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs('BTCUSDT', level=0)
    
    df = df.sort_index()
    
    # Use a small subset for quick testing
    df_test = df.iloc[1000:2000]  # 1000 observations
    
    print(f"Testing with {len(df_test)} observations")
    
    # Simple backtest configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        position_limit=0.5,
        long_threshold=0.6,
        short_threshold=0.6,
        base_position_size=0.1
    )
    
    try:
        # Run backtest
        backtester = QuantileBacktester(config)
        trades_df = backtester.run_backtest(df_test, price_col='truth')
        
        print("Backtest completed successfully")
        print(f"   Return: {backtester.metrics['total_return']:.2%}")
        print(f"   Sharpe: {backtester.metrics['sharpe_ratio']:.3f}")
        print(f"   Trades: {backtester.metrics['total_trades']}")
        
        # Test saving results (this is where the JSON error occurred)
        print("\nTesting results saving...")
        backtester.save_results(trades_df, "./test_backtest_output")
        
        print("Results saved successfully - JSON serialization fix works!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_json_serialization_fix()
    if success:
        print("\nJSON serialization fix successful!")
        print("You can now run run_backtest.py without the tuple key error.")
    else:
        print("\nFix needs more work.")