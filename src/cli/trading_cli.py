#!/usr/bin/env python3
"""
Trading Pipeline CLI Application

Combines model training, hyperparameter optimization, and backtesting
into a unified command-line interface.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.pipeline_manager import PipelineManager
from src.training.optuna_optimizer import OptunaOptimizer
from src.backtesting.backtester import HummingbotQuantileBacktester, load_price_data


class TradingCLI:
    """Main CLI application for the trading pipeline."""
    
    def __init__(self):
        self.pipeline_manager = None
        self.optimizer = None
        self.backtester = None
        
    def setup_argparse(self) -> argparse.ArgumentParser:
        """Set up command line argument parsing."""
        parser = argparse.ArgumentParser(
            description="Trading Pipeline CLI - Train models, optimize, and backtest",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Train model with default parameters
  python trading_cli.py train
  
  # Optimize hyperparameters with 100 trials
  python trading_cli.py optimize --trials 100
  
  # Run backtest with specific model
  python trading_cli.py backtest --model-path ./models/best_model.pkl
  
  # Full pipeline: train -> optimize -> backtest
  python trading_cli.py pipeline --trials 50 --backtest-data ./data/price_data.csv
  
  # Generate predictions only
  python trading_cli.py predict --model-path ./models/model.pkl --output ./predictions.csv
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train quantile models')
        self._add_training_args(train_parser)
        
        # Optimize command
        optimize_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
        self._add_optimization_args(optimize_parser)
        
        # Backtest command
        backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
        self._add_backtest_args(backtest_parser)
        
        # Pipeline command (full workflow)
        pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
        self._add_training_args(pipeline_parser)
        self._add_optimization_args(pipeline_parser)
        self._add_backtest_args(pipeline_parser)
        
        # Predict command
        predict_parser = subparsers.add_parser('predict', help='Generate predictions')
        self._add_prediction_args(predict_parser)
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Manage configuration')
        config_parser.add_argument('--show', action='store_true', help='Show current config')
        config_parser.add_argument('--reset', action='store_true', help='Reset to defaults')
        config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set config value')
        
        return parser
    
    def _add_training_args(self, parser):
        """Add training-specific arguments."""
        parser.add_argument('--train-start', default="2018-08-02", help='Training start date')
        parser.add_argument('--train-end', default="2023-12-31", help='Training end date')
        parser.add_argument('--valid-start', default="2024-01-01", help='Validation start date')
        parser.add_argument('--valid-end', default="2024-09-30", help='Validation end date')
        parser.add_argument('--test-start', default="2024-10-01", help='Test start date')
        parser.add_argument('--test-end', default="2025-04-01", help='Test end date')
        parser.add_argument('--provider-uri', default="/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA", 
                          help='Qlib data provider URI')
        parser.add_argument('--output-dir', default="./models", help='Model output directory')
        parser.add_argument('--save-features', action='store_true', help='Save feature data')
        parser.add_argument('--quantiles', nargs='+', type=float, default=[0.1, 0.5, 0.9], 
                          help='Quantiles to predict')
    
    def _add_optimization_args(self, parser):
        """Add optimization-specific arguments."""
        parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
        parser.add_argument('--timeout', type=int, help='Optimization timeout in seconds')
        parser.add_argument('--study-name', help='Optuna study name')
        parser.add_argument('--storage', help='Optuna storage URL')
        parser.add_argument('--optimize-quantile', choices=[0.1, 0.5, 0.9], type=float, default=0.5,
                          help='Which quantile to optimize')
    
    def _add_backtest_args(self, parser):
        """Add backtesting-specific arguments."""
        parser.add_argument('--price-data', help='Price data CSV file path')
        parser.add_argument('--initial-balance', type=float, default=100000.0, help='Initial balance')
        parser.add_argument('--max-position', type=float, default=0.3, help='Max position percentage')
        parser.add_argument('--fee-rate', type=float, default=0.001, help='Trading fee rate')
        parser.add_argument('--sizing-method', choices=['simple', 'kelly', 'volatility', 'enhanced'], 
                          default='enhanced', help='Position sizing method')
        parser.add_argument('--backtest-output', default="./backtest_results", help='Backtest output directory')
    
    def _add_prediction_args(self, parser):
        """Add prediction-specific arguments."""
        parser.add_argument('--model-path', required=True, help='Path to trained model')
        parser.add_argument('--input-data', help='Input data CSV file')
        parser.add_argument('--output', default="./predictions.csv", help='Output predictions file')
        parser.add_argument('--start-date', help='Prediction start date')
        parser.add_argument('--end-date', help='Prediction end date')
    
    def run_training(self, args):
        """Execute model training."""
        print("🚀 Starting model training...")
        
        self.pipeline_manager = PipelineManager(
            train_start=args.train_start,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            test_start=args.test_start,
            test_end=args.test_end,
            provider_uri=args.provider_uri,
            quantiles=args.quantiles
        )
        
        # Train model
        model, dataset, results = self.pipeline_manager.train_model()
        
        # Save model and results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        model_path = output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.pipeline_manager.save_model(model, model_path)
        
        # Save feature data if requested
        if args.save_features:
            features_path = output_dir / "features.pkl"
            results['features'].to_pickle(features_path)
            print(f"✅ Features saved to {features_path}")
        
        print(f"✅ Model training completed. Model saved to {model_path}")
        return model, results
    
    def run_optimization(self, args):
        """Execute hyperparameter optimization."""
        print("🔧 Starting hyperparameter optimization...")
        
        self.optimizer = OptunaOptimizer(
            train_start=args.train_start,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            provider_uri=args.provider_uri,
            quantile=args.optimize_quantile
        )
        
        # Run optimization
        study = self.optimizer.optimize(
            n_trials=args.trials,
            timeout=args.timeout,
            study_name=args.study_name,
            storage=args.storage
        )
        
        # Save best parameters
        output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else Path("./models")
        output_dir.mkdir(exist_ok=True)
        
        best_params_path = output_dir / f"best_params_{args.optimize_quantile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        print(f"✅ Optimization completed. Best parameters saved to {best_params_path}")
        print(f"Best score: {study.best_value:.6f}")
        
        return study
    
    def run_backtesting(self, args):
        """Execute backtesting."""
        print("📊 Starting backtesting...")
        
        # Load prediction data
        if hasattr(args, 'model_path') and args.model_path:
            # Load from specific model
            predictions_df = self._load_predictions_from_model(args.model_path, args)
        else:
            # Load from CSV (assume predictions already generated)
            predictions_file = getattr(args, 'predictions_file', './df_all_macro_analysis.csv')
            if not os.path.exists(predictions_file):
                print(f"❌ Predictions file not found: {predictions_file}")
                print("Please run training first or specify --model-path")
                return None
            
            predictions_df = pd.read_csv(predictions_file)
            if 'datetime' in predictions_df.columns:
                predictions_df['datetime'] = pd.to_datetime(predictions_df['datetime'])
                predictions_df = predictions_df.set_index('datetime')
        
        # Load price data
        price_data = load_price_data(args.price_data) if args.price_data else None
        
        # Configure backtester
        self.backtester = HummingbotQuantileBacktester(
            initial_balance=args.initial_balance,
            max_position_pct=args.max_position,
            fee_rate=args.fee_rate,
            sizing_method=args.sizing_method
        )
        
        # Run backtest
        results = self.backtester.run_backtest(
            predictions_df, 
            price_data=price_data, 
            price_col='close', 
            return_col='truth'
        )
        
        # Generate report
        report = self.backtester.generate_report(results)
        print(report)
        
        # Save results
        self.backtester.save_results(results, args.backtest_output)
        
        print(f"✅ Backtesting completed. Results saved to {args.backtest_output}")
        return results
    
    def run_pipeline(self, args):
        """Execute full pipeline: train -> optimize -> backtest."""
        print("🔄 Starting full pipeline...")
        
        # Step 1: Training
        model, training_results = self.run_training(args)
        
        # Step 2: Optimization (optional)
        if args.trials > 0:
            study = self.run_optimization(args)
            
            # Retrain with best parameters
            print("🔄 Retraining with optimized parameters...")
            # TODO: Implement retraining with best params
        
        # Step 3: Backtesting
        backtest_results = self.run_backtesting(args)
        
        print("✅ Full pipeline completed!")
        return {
            'model': model,
            'training_results': training_results,
            'backtest_results': backtest_results
        }
    
    def run_prediction(self, args):
        """Generate predictions from trained model."""
        print("🔮 Generating predictions...")
        
        # Load model
        if not os.path.exists(args.model_path):
            print(f"❌ Model file not found: {args.model_path}")
            return None
        
        # TODO: Implement prediction loading and generation
        print(f"Loading model from {args.model_path}")
        
        # Generate predictions
        predictions = self._generate_predictions(args)
        
        # Save predictions
        predictions.to_csv(args.output)
        print(f"✅ Predictions saved to {args.output}")
        
        return predictions
    
    def manage_config(self, args):
        """Manage configuration settings."""
        config_file = Path("./config/trading_config.json")
        
        if args.show:
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                print("Current configuration:")
                print(json.dumps(config, indent=2))
            else:
                print("No configuration file found.")
        
        elif args.reset:
            # Reset to defaults
            default_config = self._get_default_config()
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print("Configuration reset to defaults.")
        
        elif args.set:
            key, value = args.set
            # Load existing config or create new
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
            else:
                config = self._get_default_config()
            
            # Set value (with type inference)
            try:
                # Try to parse as number or boolean
                if value.lower() in ['true', 'false']:
                    config[key] = value.lower() == 'true'
                elif '.' in value:
                    config[key] = float(value)
                elif value.isdigit():
                    config[key] = int(value)
                else:
                    config[key] = value
            except:
                config[key] = value
            
            # Save config
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Set {key} = {config[key]}")
    
    def _load_predictions_from_model(self, model_path, args):
        """Load predictions from a trained model."""
        # TODO: Implement model loading and prediction generation
        print(f"Loading predictions from model: {model_path}")
        # For now, return dummy data
        return pd.DataFrame()
    
    def _generate_predictions(self, args):
        """Generate predictions using loaded model."""
        # TODO: Implement prediction generation
        return pd.DataFrame()
    
    def _get_default_config(self):
        """Get default configuration."""
        return {
            "train_start": "2018-08-02",
            "train_end": "2023-12-31",
            "valid_start": "2024-01-01",
            "valid_end": "2024-09-30",
            "provider_uri": "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA",
            "quantiles": [0.1, 0.5, 0.9],
            "initial_balance": 100000.0,
            "max_position": 0.3,
            "fee_rate": 0.001,
            "sizing_method": "enhanced"
        }
    
    def run(self):
        """Main entry point."""
        parser = self.setup_argparse()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            if args.command == 'train':
                self.run_training(args)
            elif args.command == 'optimize':
                self.run_optimization(args)
            elif args.command == 'backtest':
                self.run_backtesting(args)
            elif args.command == 'pipeline':
                self.run_pipeline(args)
            elif args.command == 'predict':
                self.run_prediction(args)
            elif args.command == 'config':
                self.manage_config(args)
            else:
                print(f"Unknown command: {args.command}")
                parser.print_help()
                
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            if os.getenv('DEBUG'):
                import traceback
                traceback.print_exc()


def main():
    """CLI entry point."""
    cli = TradingCLI()
    cli.run()


if __name__ == "__main__":
    main()