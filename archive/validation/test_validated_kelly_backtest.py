#!/usr/bin/env python3
"""
Test the validated Kelly methods in a quick backtest
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pandas as pd
from src.backtesting.backtester import HummingbotQuantileBacktester

def test_validated_kelly_backtest():
    """Test validated Kelly vs original methods"""
    
    # Load data
    df = pd.read_csv("df_all_macro_analysis.csv")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Use recent subset for quick test
    test_df = df.tail(1000)
    
    print(f"Testing validated Kelly methods with {len(test_df)} observations")
    print(f"Date range: {test_df.index.min()} to {test_df.index.max()}")
    
    # Test configurations
    configs = {
        'kelly_validated': {
            'sizing_method': 'kelly',
            'max_position_pct': 0.3,
            'long_threshold': 0.6,
            'short_threshold': 0.6
        },
        'enhanced_validated': {
            'sizing_method': 'enhanced', 
            'max_position_pct': 0.3,
            'long_threshold': 0.6,
            'short_threshold': 0.6
        },
        'volatility_baseline': {
            'sizing_method': 'volatility',
            'max_position_pct': 0.3,
            'long_threshold': 0.6,
            'short_threshold': 0.6
        }
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nTesting {name}...")
        
        backtester = HummingbotQuantileBacktester(
            initial_balance=10000.0,
            trading_pair="BTCUSDT",
            fee_rate=0.001,
            **config
        )
        
        try:
            # Run backtest
            backtest_results = backtester.run_backtest(test_df, return_col='truth')
            
            if len(backtest_results) > 0:
                metrics = backtester.calculate_metrics(backtest_results)
                
                results[name] = {
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'final_value': metrics.get('final_portfolio_value', 10000)
                }
                
                print(f"  Return: {results[name]['total_return']:.2%}")
                print(f"  Sharpe: {results[name]['sharpe_ratio']:.3f}")
                print(f"  Trades: {results[name]['total_trades']}")
                print(f"  Win Rate: {results[name]['win_rate']:.2%}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = None
    
    # Compare results
    print(f"\n{'='*60}")
    print("VALIDATED KELLY COMPARISON")
    print(f"{'='*60}")
    
    if results:
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        print(comparison_df)
        
        # Find best performer
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_sharpe = max(valid_results.keys(), 
                            key=lambda x: valid_results[x]['sharpe_ratio'])
            best_return = max(valid_results.keys(), 
                            key=lambda x: valid_results[x]['total_return'])
            
            print(f"\nBest Sharpe: {best_sharpe} ({valid_results[best_sharpe]['sharpe_ratio']:.3f})")
            print(f"Best Return: {best_return} ({valid_results[best_return]['total_return']:.2%})")

if __name__ == "__main__":
    test_validated_kelly_backtest()