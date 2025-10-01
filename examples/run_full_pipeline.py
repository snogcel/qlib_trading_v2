#!/usr/bin/env python3
"""
Example: Full Trading Pipeline

This script demonstrates how to use the CLI programmatically
or provides examples for command-line usage.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and print output."""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    
    print("✅ Command completed successfully")
    return True

def main():
    """Run example pipeline."""
    
    # Check if CLI is available
    try:
        result = subprocess.run(["trading-cli", "--help"], capture_output=True)
        if result.returncode != 0:
            print("❌ trading-cli not found. Please install first:")
            print("   pip install -e .")
            return
    except FileNotFoundError:
        print("❌ trading-cli not found. Please install first:")
        print("   pip install -e .")
        return
    
    print("🎯 Trading Pipeline Example")
    print("This example demonstrates the full pipeline workflow")
    
    # Example 1: Quick training
    print("\n" + "="*60)
    print("EXAMPLE 1: Quick Model Training")
    print("="*60)
    
    cmd = [
        "trading-cli", "train",
        "--train-start", "2023-01-01",
        "--train-end", "2023-06-30", 
        "--valid-start", "2023-07-01",
        "--valid-end", "2023-09-30",
        "--output-dir", "./example_models",
        "--save-features"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("- Train models on 6 months of data")
    print("- Validate on 3 months")
    print("- Save models to ./example_models/")
    print("- Save engineered features")
    
    # Example 2: Hyperparameter optimization
    print("\n" + "="*60)
    print("EXAMPLE 2: Hyperparameter Optimization")
    print("="*60)
    
    cmd = [
        "trading-cli", "optimize",
        "--trials", "50",
        "--optimize-quantile", "0.5",
        "--study-name", "example_optimization",
        "--timeout", "1800"  # 30 minutes
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("- Run 50 optimization trials")
    print("- Optimize the median quantile (0.5)")
    print("- Timeout after 30 minutes")
    print("- Save results with study name")
    
    # Example 3: Backtesting
    print("\n" + "="*60)
    print("EXAMPLE 3: Backtesting")
    print("="*60)
    
    cmd = [
        "trading-cli", "backtest",
        "--price-data", "./data/btc_hourly.csv",  # You need to provide this
        "--initial-balance", "50000",
        "--max-position", "0.25",
        "--sizing-method", "enhanced",
        "--backtest-output", "./example_backtest"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("- Use price data from CSV file")
    print("- Start with $50,000")
    print("- Max 25% position size")
    print("- Use enhanced position sizing")
    print("- Save results to ./example_backtest/")
    
    # Example 4: Full pipeline
    print("\n" + "="*60)
    print("EXAMPLE 4: Complete Pipeline")
    print("="*60)
    
    cmd = [
        "trading-cli", "pipeline",
        "--train-start", "2023-01-01",
        "--train-end", "2023-08-31",
        "--valid-start", "2023-09-01", 
        "--valid-end", "2023-11-30",
        "--trials", "30",
        "--price-data", "./data/btc_hourly.csv",
        "--initial-balance", "100000",
        "--sizing-method", "kelly",
        "--output-dir", "./pipeline_results",
        "--save-features"
    ]
    
    print("Command:")
    print(" ".join(cmd))
    print("\nThis will:")
    print("- Train models on 8 months of data")
    print("- Optimize with 30 trials")
    print("- Backtest with Kelly sizing")
    print("- Save everything to ./pipeline_results/")
    
    # Example 5: Configuration management
    print("\n" + "="*60)
    print("EXAMPLE 5: Configuration Management")
    print("="*60)
    
    print("Show current config:")
    print("trading-cli config --show")
    
    print("\nSet default balance:")
    print("trading-cli config --set initial_balance 75000")
    
    print("\nSet default sizing method:")
    print("trading-cli config --set sizing_method volatility")
    
    print("\nReset to defaults:")
    print("trading-cli config --reset")
    
    # Interactive choice
    print("\n" + "="*60)
    print("INTERACTIVE EXECUTION")
    print("="*60)
    
    choice = input("\nWould you like to run one of these examples? (1-5, or 'n' for no): ")
    
    if choice == '1':
        print("\n🚀 Running Example 1: Quick Training")
        cmd = [
            "trading-cli", "train",
            "--train-start", "2023-01-01",
            "--train-end", "2023-06-30",
            "--valid-start", "2023-07-01", 
            "--valid-end", "2023-09-30",
            "--output-dir", "./example_models",
            "--save-features"
        ]
        run_command(cmd)
        
    elif choice == '2':
        print("\n🚀 Running Example 2: Optimization (reduced trials)")
        cmd = [
            "trading-cli", "optimize",
            "--trials", "10",  # Reduced for demo
            "--optimize-quantile", "0.5"
        ]
        run_command(cmd)
        
    elif choice == '3':
        price_file = input("Enter path to price data CSV (or press Enter to skip): ")
        if price_file and Path(price_file).exists():
            cmd = [
                "trading-cli", "backtest",
                "--price-data", price_file,
                "--initial-balance", "50000",
                "--max-position", "0.25",
                "--sizing-method", "enhanced"
            ]
            run_command(cmd)
        else:
            print("❌ Price data file not found or not provided")
            
    elif choice == '4':
        price_file = input("Enter path to price data CSV (or press Enter to skip): ")
        if price_file and Path(price_file).exists():
            cmd = [
                "trading-cli", "pipeline",
                "--train-start", "2023-06-01",
                "--train-end", "2023-08-31",
                "--valid-start", "2023-09-01",
                "--valid-end", "2023-10-31", 
                "--trials", "5",  # Very reduced for demo
                "--price-data", price_file,
                "--initial-balance", "100000",
                "--sizing-method", "kelly"
            ]
            run_command(cmd)
        else:
            print("❌ Price data file not found or not provided")
            
    elif choice == '5':
        print("\n🚀 Running Example 5: Configuration")
        run_command(["trading-cli", "config", "--show"])
        
    else:
        print("No example selected. Exiting.")
    
    print("\n✅ Example script completed!")
    print("\nNext steps:")
    print("1. Prepare your price data in the required CSV format")
    print("2. Adjust date ranges based on your data availability")
    print("3. Experiment with different sizing methods and parameters")
    print("4. Use the generated reports to analyze performance")

if __name__ == "__main__":
    main()