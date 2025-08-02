"""
Run Hummingbot-compatible backtest on your quantile predictions
"""

import pandas as pd
import numpy as np
from hummingbot_backtester import HummingbotQuantileBacktester

def main(price_file: str = None):
    """
    Run comprehensive Hummingbot backtests with different configurations
    
    Args:
        price_file: Path to CSV with 60min price data (optional)
    """
    print("Loading data...")
    
    # Load prediction data
    df = pd.read_csv("df_all_macro_analysis.csv")
    
    # Set up datetime index properly
    if 'datetime' in df.columns:
        print("Converting datetime column...")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        print(f"Prediction data range: {df.index.min()} to {df.index.max()}")
    
    # Load price data if provided
    price_data = None
    if price_file:
        try:
            print(f"Loading price data from {price_file}...")
            price_data = pd.read_csv(price_file)
            
            # Handle different datetime column names
            datetime_col = None
            for col in ['datetime', 'timestamp', 'time']:
                if col in price_data.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                price_data['datetime'] = pd.to_datetime(price_data[datetime_col])
                price_data = price_data.set_index('datetime')
                print(f"Price data range: {price_data.index.min()} to {price_data.index.max()}")
                print(f"Price data columns: {list(price_data.columns)}")
            else:
                print("Warning: No datetime column found in price data")
                price_data = None
                
        except Exception as e:
            print(f"Error loading price data: {e}")
            price_data = None
    
    if price_data is None:
        print("No price data available, will simulate from returns")
    
    # Use recent data for testing
    df = df.tail(6600)  # Last 5k observations
    print(f"Using {len(df)} observations for backtest")
    
    # Test different configurations using VALIDATED sizing methods
    # Based on validation results: spread predicts volatility, thresholds are meaningful
    configs = {
        'conservative_validated': {
            'long_threshold': 0.6,
            'short_threshold': 0.6,
            'max_position_pct': 0.15,         # Small positions for safety
            'fee_rate': 0.001,
            'neutral_close_threshold': 0.6,   # Close positions readily
            'min_confidence_hold': 2.0,       # Higher confidence required
            'opposing_signal_threshold': 0.3,
            'sizing_method': 'kelly'          # Validated Kelly (best Sharpe: 3.98)
        },
        'moderate_validated': {
            'long_threshold': 0.6,
            'short_threshold': 0.6,
            'max_position_pct': 0.25,         # Medium positions
            'fee_rate': 0.001,
            'neutral_close_threshold': 0.7,
            'min_confidence_hold': 1.0,
            'opposing_signal_threshold': 0.4,
            'sizing_method': 'enhanced'       # Enhanced with validated features
        },
        'aggressive_validated': {
            'long_threshold': 0.6,
            'short_threshold': 0.6,
            'max_position_pct': 0.35,         # Larger positions
            'fee_rate': 0.001,
            'neutral_close_threshold': 0.8,   # Hold positions longer
            'min_confidence_hold': 0.5,
            'opposing_signal_threshold': 0.5,
            'sizing_method': 'volatility'     # Volatility method (highest return: 11.44%)
        },
        'hybrid_best': {
            'long_threshold': 0.6,
            'short_threshold': 0.6,
            'max_position_pct': 0.3,          # Balanced position size
            'fee_rate': 0.001,
            'neutral_close_threshold': 0.7,
            'min_confidence_hold': 1.5,       # Balanced confidence
            'opposing_signal_threshold': 0.4,
            'sizing_method': 'enhanced'       # Best of both worlds
        }
    }
    
    results_summary = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Running {config_name.upper()} Hummingbot backtest...")
        print(f"{'='*60}")
        
        # Extract sizing method from config
        sizing_method = config.pop('sizing_method', 'kelly')
        
        # Create backtester
        backtester = HummingbotQuantileBacktester(
            initial_balance=10000.0,
            trading_pair="BTCUSDT",
            sizing_method=sizing_method,
            **config
        )
        
        try:
            # Run backtest with real price data
            results = backtester.run_backtest(df, price_data=price_data, price_col='close', return_col='truth')
            
            # Calculate metrics
            metrics = backtester.calculate_metrics(results)
            
            # Print report
            print(backtester.generate_report(results))
            
            # Save results
            backtester.save_results(results, f"./hummingbot_backtest_results/{config_name}")
            
            # Store for comparison
            results_summary[config_name] = {
                'total_return': metrics.get('total_return', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': metrics.get('win_rate', 0)
            }
            
        except Exception as e:
            print(f"Error running {config_name} backtest: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison
    if results_summary:
        print(f"\n{'='*80}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame(results_summary).T
        comparison_df = comparison_df.round(4)
        print(comparison_df)
        
        # Save comparison
        comparison_df.to_csv("./hummingbot_backtest_results/configuration_comparison.csv")
        
        # Find best configuration
        best_config = max(results_summary.keys(), 
                         key=lambda x: results_summary[x]['sharpe_ratio'])
        print(f"\nBest configuration by Sharpe ratio: {best_config}")
        print(f"Sharpe: {results_summary[best_config]['sharpe_ratio']:.3f}")
        print(f"Return: {results_summary[best_config]['total_return']:.2%}")

if __name__ == "__main__":
    # Use the real BTC price data
    price_file = r"C:\Projects\qlib_trading_v2\csv_data\CRYPTODATA_RESAMPLE\60min\BTCUSDT.csv"
    main(price_file)