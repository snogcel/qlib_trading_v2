"""
Test different position management strategies
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

def create_test_scenarios():
    """
    Create test scenarios to demonstrate position management
    """
    # Create synthetic data with different signal patterns
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    scenarios = []
    
    # Scenario 1: Strong long signal followed by neutral
    scenario1 = pd.DataFrame({
        'q10': [0.01] * 20 + [0.0] * 30 + [-0.005] * 50,
        'q50': [0.02] * 20 + [0.001] * 30 + [0.0] * 50,
        'q90': [0.03] * 20 + [0.002] * 30 + [0.005] * 50,
        'tier_confidence': [8.0] * 20 + [4.0] * 30 + [6.0] * 50,
        'truth': np.random.normal(0.01, 0.02, 100)
    }, index=dates)
    scenarios.append(("Strong Long → Neutral → Weak", scenario1))
    
    # Scenario 2: Long position with opposing signals
    scenario2 = pd.DataFrame({
        'q10': [0.01] * 30 + [-0.01] * 40 + [0.005] * 30,
        'q50': [0.02] * 30 + [-0.005] * 40 + [0.01] * 30,
        'q90': [0.03] * 30 + [0.0] * 40 + [0.015] * 30,
        'tier_confidence': [7.0] * 30 + [6.0] * 40 + [5.0] * 30,
        'truth': np.random.normal(0.005, 0.02, 100)
    }, index=dates)
    scenarios.append(("Long → Opposing Short → Recovery", scenario2))
    
    return scenarios

def test_position_management_styles():
    """
    Test different position management configurations
    """
    print("=== POSITION MANAGEMENT TESTING ===")
    
    scenarios = create_test_scenarios()
    
    # Different position management styles
    styles = {
        'hair_trigger': {
            'neutral_close_threshold': 0.5,  # Close very easily
            'min_confidence_hold': 5.0,
            'opposing_signal_threshold': 0.2
        },
        'balanced': {
            'neutral_close_threshold': 0.7,  # Standard
            'min_confidence_hold': 3.0,
            'opposing_signal_threshold': 0.4
        },
        'sticky': {
            'neutral_close_threshold': 0.9,  # Hold positions strongly
            'min_confidence_hold': 1.0,
            'opposing_signal_threshold': 0.6
        }
    }
    
    results = {}
    
    for scenario_name, data in scenarios:
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*50}")
        
        scenario_results = {}
        
        for style_name, style_config in styles.items():
            print(f"\nTesting {style_name} position management...")
            
            # Create backtester with this style
            backtester = HummingbotQuantileBacktester(
                initial_balance=100000.0,
                long_threshold=0.6,
                short_threshold=0.6,
                max_position_pct=0.3,
                **style_config
            )
            
            # Create simple price data
            price_data = pd.DataFrame({
                'close': 50000 * (1 + data['truth']).cumprod()
            }, index=data.index)
            
            # Run backtest
            results_df = backtester.run_backtest(
                df=data,
                price_data=price_data,
                price_col='close',
                return_col='truth'
            )
            
            # Analyze results
            metrics = backtester.calculate_metrics(results_df)
            
            scenario_results[style_name] = {
                'total_trades': metrics['total_trades'],
                'total_holds': metrics['total_holds'],
                'trade_frequency': metrics['trade_frequency'],
                'total_return': metrics['total_return'],
                'hold_reasons': metrics['hold_reasons']
            }
            
            print(f"  Trades: {metrics['total_trades']}, Holds: {metrics['total_holds']}")
            print(f"  Trade frequency: {metrics['trade_frequency']:.1%}")
            print(f"  Return: {metrics['total_return']:.2%}")
            
            # Show position closure reasons
            if backtester.trades:
                closure_trades = [t for t in backtester.trades if t.get('trade_type') == 'POSITION_CLOSURE']
                if closure_trades:
                    print(f"  Position closures: {len(closure_trades)}")
                    for trade in closure_trades[:3]:  # Show first 3
                        close_details = trade.get('signal', {}).get('close_details', {})
                        reason = close_details.get('close_reason', 'unknown')
                        print(f"    - {reason}")
        
        results[scenario_name] = scenario_results
    
    return results

def analyze_position_management_effectiveness():
    """
    Analyze which position management style works best
    """
    print("\n" + "="*60)
    print("POSITION MANAGEMENT EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    results = test_position_management_styles()
    
    # Create comparison table
    comparison_data = []
    for scenario, styles in results.items():
        for style, metrics in styles.items():
            comparison_data.append({
                'scenario': scenario,
                'style': style,
                'trades': metrics['total_trades'],
                'holds': metrics['total_holds'],
                'trade_freq': metrics['trade_frequency'],
                'return': metrics['total_return']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\nComparison Summary:")
    print(comparison_df.to_string(index=False))
    
    # Find best style per scenario
    print("\nBest performing style by scenario:")
    for scenario in comparison_df['scenario'].unique():
        scenario_data = comparison_df[comparison_df['scenario'] == scenario]
        best_style = scenario_data.loc[scenario_data['return'].idxmax()]
        print(f"  {scenario}: {best_style['style']} ({best_style['return']:.2%} return)")

if __name__ == "__main__":
    analyze_position_management_effectiveness()