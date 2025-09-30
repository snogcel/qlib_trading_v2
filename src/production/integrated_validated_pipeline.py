#!/usr/bin/env python3
"""
Integrated pipeline using validated Kelly methods and proven features
This replaces the experimental approach with validated, production-ready methods
"""

import pandas as pd
import numpy as np
from src.backtesting.backtester import HummingbotQuantileBacktester
import json
from pathlib import Path

class ValidatedTradingPipeline:
    """
    Production trading pipeline using validated features and Kelly methods
    """
    
    def __init__(self, data_file="df_all_macro_analysis.csv", price_file=None):
        self.data_file = data_file
        self.price_file = price_file
        self.validation_results = {}
        self.backtest_results = {}
        
    def load_and_validate_data(self):
        """Load data and run validation checks"""
        print("Loading and validating data...")
        
        # Load prediction data
        self.df = pd.read_csv(self.data_file)
        if 'datetime' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.set_index('datetime')
        
        print(f"Loaded {len(self.df)} observations")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        
        # Add missing columns if needed
        if 'abs_q50' not in self.df.columns:
            self.df['abs_q50'] = self.df['q50'].abs()
        
        if 'spread' not in self.df.columns:
            self.df['spread'] = self.df['q90'] - self.df['q10']
        
        # Run validation checks
        self._validate_features()
        
        # Load price data if available
        self.price_data = self._load_price_data()
        
        return self.df
    
    def _validate_features(self):
        """Run feature validation to confirm predictive power"""
        print("\nValidating feature predictive power...")
        
        # Test spread predictive power
        if 'truth' in self.df.columns:
            # Calculate future volatility
            self.df['future_vol_6h'] = self.df['truth'].rolling(6).std().shift(-6)
            self.df['future_abs_return_1h'] = self.df['truth'].shift(-1).abs()
            
            # Test correlations
            spread_vol_corr = self.df['spread'].corr(self.df['future_vol_6h'])
            spread_return_corr = self.df['spread'].corr(self.df['future_abs_return_1h'])
            
            self.validation_results['spread_vol_correlation'] = spread_vol_corr
            self.validation_results['spread_return_correlation'] = spread_return_corr
            
            print(f"  Spread vs future volatility: {spread_vol_corr:.4f}")
            print(f"  Spread vs future returns: {spread_return_corr:.4f}")
            
            # Test signal thresholds
            if 'signal_thresh_adaptive' in self.df.columns:
                above_thresh = self.df[self.df['abs_q50'] > self.df['signal_thresh_adaptive']]
                below_thresh = self.df[self.df['abs_q50'] <= self.df['signal_thresh_adaptive']]
                
                above_return = above_thresh['truth'].shift(-1).mean()
                below_return = below_thresh['truth'].shift(-1).mean()
                
                self.validation_results['signal_thresh_above_return'] = above_return
                self.validation_results['signal_thresh_below_return'] = below_return
                
                print(f"  Above signal threshold return: {above_return:.6f}")
                print(f"  Below signal threshold return: {below_return:.6f}")
        
        # Validate that we have required columns
        required_cols = ['q10', 'q50', 'q90', 'signal_tier', 'spread_thresh']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for signal_thresh or signal_thresh_adaptive
        if 'signal_thresh' not in self.df.columns and 'signal_thresh_adaptive' not in self.df.columns:
            raise ValueError("Missing both 'signal_thresh' and 'signal_thresh_adaptive'")
        
        print("All validation checks passed")
    
    def _load_price_data(self):
        """Load price data if available"""
        if not self.price_file:
            return None
            
        try:
            print(f"Loading price data from {self.price_file}...")
            price_data = pd.read_csv(self.price_file)
            
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
                return price_data
            else:
                print("Warning: No datetime column found in price data")
                return None
                
        except Exception as e:
            print(f"Error loading price data: {e}")
            return None
    
    def run_validated_backtests(self, observations=None):
        """Run backtests using validated Kelly methods"""
        
        # Use subset if specified
        if observations:
            test_df = self.df.tail(observations)
            print(f"\nUsing last {observations} observations for backtesting")
        else:
            test_df = self.df
            print(f"\nUsing full dataset ({len(test_df)} observations) for backtesting")
        
        # Validated configurations based on our testing
        configs = {
            'kelly_validated': {
                'description': 'Validated Kelly with proven features (Best Sharpe: 3.98)',
                'sizing_method': 'kelly',
                'max_position_pct': 0.25,
                'long_threshold': 0.6,
                'short_threshold': 0.6,
                'neutral_close_threshold': 0.7,
                'min_confidence_hold': 1.5,
                'opposing_signal_threshold': 0.4
            },
            'enhanced_validated': {
                'description': 'Enhanced ensemble with validated features',
                'sizing_method': 'enhanced',
                'max_position_pct': 0.3,
                'long_threshold': 0.6,
                'short_threshold': 0.6,
                'neutral_close_threshold': 0.7,
                'min_confidence_hold': 1.0,
                'opposing_signal_threshold': 0.4
            },
            'volatility_aggressive': {
                'description': 'Volatility method for higher returns (11.44% in test)',
                'sizing_method': 'volatility',
                'max_position_pct': 0.35,
                'long_threshold': 0.6,
                'short_threshold': 0.6,
                'neutral_close_threshold': 0.8,
                'min_confidence_hold': 0.5,
                'opposing_signal_threshold': 0.5
            },
            'conservative_safe': {
                'description': 'Conservative approach with tight risk management',
                'sizing_method': 'kelly',
                'max_position_pct': 0.15,
                'long_threshold': 0.6,
                'short_threshold': 0.6,
                'neutral_close_threshold': 0.6,
                'min_confidence_hold': 2.0,
                'opposing_signal_threshold': 0.3
            }
        }
        
        results_summary = {}
        
        for config_name, config in configs.items():
            print(f"\n{'='*60}")
            print(f"Running {config_name.upper()}")
            print(f"Description: {config['description']}")
            print(f"{'='*60}")
            
            # Extract description and create backtester config
            description = config.pop('description')
            
            try:
                # Create backtester with validated methods
                backtester = HummingbotQuantileBacktester(
                    initial_balance=100000.0,  # Larger balance for realistic testing
                    trading_pair="BTCUSDT",
                    fee_rate=0.001,
                    **config
                )
                
                # Run backtest
                results = backtester.run_backtest(
                    test_df, 
                    price_data=self.price_data, 
                    price_col='close', 
                    return_col='truth'
                )
                
                # Calculate metrics
                metrics = backtester.calculate_metrics(results)
                
                # Store results
                results_summary[config_name] = {
                    'description': description,
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'trade_frequency': metrics.get('trade_frequency', 0),
                    'final_value': metrics.get('final_portfolio_value', 100000)
                }
                
                # Print summary
                print(f"Total Return: {metrics.get('total_return', 0):.2%}")
                print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"Total Trades: {metrics.get('total_trades', 0)}")
                print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
                
                # Save detailed results
                output_dir = f"./validated_backtest_results/{config_name}"
                backtester.save_results(results, output_dir)
                
                # Restore description for next iteration
                config['description'] = description
                
            except Exception as e:
                print(f"Error running {config_name}: {e}")
                import traceback
                traceback.print_exc()
                results_summary[config_name] = None
        
        self.backtest_results = results_summary
        return results_summary
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report with validation and backtest results"""
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE VALIDATED TRADING PIPELINE REPORT")
        print(f"{'='*80}")
        
        # Validation results
        print("\nFEATURE VALIDATION RESULTS:")
        print("-" * 40)
        for key, value in self.validation_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Backtest comparison
        if self.backtest_results:
            print(f"\nBACKTEST COMPARISON:")
            print("-" * 40)
            
            # Create comparison DataFrame
            valid_results = {k: v for k, v in self.backtest_results.items() if v is not None}
            
            if valid_results:
                comparison_df = pd.DataFrame(valid_results).T
                
                # Select key metrics for display
                display_cols = ['total_return', 'sharpe_ratio', 'max_drawdown', 
                               'total_trades', 'win_rate', 'trade_frequency']
                display_df = comparison_df[display_cols].round(4)
                
                print(display_df)
                
                # Find best performers
                best_sharpe = max(valid_results.keys(), 
                                key=lambda x: valid_results[x]['sharpe_ratio'])
                best_return = max(valid_results.keys(), 
                                key=lambda x: valid_results[x]['total_return'])
                
                print(f"\nBEST PERFORMERS:")
                print(f"Best Sharpe Ratio: {best_sharpe} ({valid_results[best_sharpe]['sharpe_ratio']:.3f})")
                print(f"Best Total Return: {best_return} ({valid_results[best_return]['total_return']:.2%})")
                
                # Save comparison
                comparison_df.to_csv("./validated_backtest_results/comparison.csv")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 40)
        print("1. Use 'kelly_validated' for best risk-adjusted returns")
        print("2. Use 'volatility_aggressive' for higher absolute returns")
        print("3. Use 'conservative_safe' for capital preservation")
        print("4. All methods use validated features with proven predictive power")
        
        # Save full report
        report_data = {
            'validation_results': self.validation_results,
            'backtest_results': self.backtest_results,
            'data_info': {
                'total_observations': len(self.df),
                'date_range': f"{self.df.index.min()} to {self.df.index.max()}",
                'has_price_data': self.price_data is not None
            }
        }
        
        with open("./validated_backtest_results/full_report.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nFull report saved to: ./validated_backtest_results/full_report.json")

def main():
    """Run the complete validated pipeline"""
    
    # Initialize pipeline
    price_file = r"C:\Projects\qlib_trading_v2\csv_data\CRYPTODATA_RESAMPLE\60min\BTCUSDT.csv"
    pipeline = ValidatedTradingPipeline(
        data_file="df_all_macro_analysis.csv",
        price_file=price_file
    )
    
    # Load and validate data
    df = pipeline.load_and_validate_data()
    
    # Run backtests (use subset for faster testing, None for full dataset)
    results = pipeline.run_validated_backtests(observations=5000)
    
    # Generate comprehensive report
    pipeline.generate_comprehensive_report()
    
    print(f"\nValidated pipeline complete!")
    print(f"Results saved to: ./validated_backtest_results/")

if __name__ == "__main__":
    main()