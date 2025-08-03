"""
Complete workflow for running Hummingbot backtests with proper data alignment
"""

import sys
import os

def main():
    """
    Complete workflow: validate data alignment, then run backtest
    """
    print("=== COMPLETE HUMMINGBOT BACKTEST WORKFLOW ===")
    
    # Step 1: Validate data alignment
    print("\nStep 1: Validating data alignment...")
    try:
        from validate_data_alignment import validate_data_alignment, create_aligned_dataset_for_backtest
        
        # First check alignment
        aligned_data = validate_data_alignment()
        
        if aligned_data is None or len(aligned_data) == 0:
            print("ERROR: Data alignment failed. Cannot proceed with backtest.")
            return
        
        print("✓ Data alignment validation completed")
        
        # Create aligned dataset
        print("\nStep 2: Creating aligned dataset...")
        backtest_data = create_aligned_dataset_for_backtest("aligned_backtest_data.csv")
        
        if backtest_data is None:
            print("ERROR: Failed to create aligned dataset")
            return
        
        print("✓ Aligned dataset created")
        
    except Exception as e:
        print(f"ERROR in data alignment: {e}")
        return
    
    # Step 2: Run Hummingbot backtest
    print("\nStep 3: Running Hummingbot backtest...")
    try:
        from run_hummingbot_backtest import main as run_backtest
        
        # Run with price data
        price_file = r"C:\Projects\qlib_trading_v2\csv_data\CRYPTODATA_RESAMPLE\60min\BTCUSDT.csv"
        
        if os.path.exists(price_file):
            print(f"Using price data from: {price_file}")
            run_backtest(price_file)
        else:
            print("Price file not found, running without price data")
            run_backtest(None)
        
        print("✓ Hummingbot backtest completed")
        
    except Exception as e:
        print(f"ERROR in backtest: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Analyze results
    print("\nStep 4: Analyzing results...")
    try:
        from analyze_trading_frequency import analyze_trading_patterns
        
        portfolio, trades = analyze_trading_patterns()
        
        if portfolio is not None:
            print("✓ Trading frequency analysis completed")
        
    except Exception as e:
        print(f"WARNING: Could not run trading analysis: {e}")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nResults saved in:")
    print("  - ./hummingbot_backtest_results/ (backtest results)")
    print("  - aligned_backtest_data.csv (aligned dataset)")
    print("  - data_alignment_validation.png (validation plots)")
    print("  - trading_activity_analysis.png (trading analysis)")

if __name__ == "__main__":
    main()