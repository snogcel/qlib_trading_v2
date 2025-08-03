"""
Validate alignment between prediction data and price data
"""

import pandas as pd
import matplotlib.pyplot as plt

def validate_data_alignment():
    """
    Check how well prediction data aligns with price data
    """
    print("=== DATA ALIGNMENT VALIDATION ===")
    
    # Load prediction data
    print("Loading prediction data...")
    df_pred = pd.read_csv("df_all_macro_analysis.csv")
    
    if 'datetime' in df_pred.columns:
        df_pred['datetime'] = pd.to_datetime(df_pred['datetime'])
        df_pred = df_pred.set_index('datetime')
    
    print(f"Prediction data: {len(df_pred)} observations")
    print(f"Prediction range: {df_pred.index.min()} to {df_pred.index.max()}")
    
    # Load price data
    print("\nLoading price data...")
    price_file = r"C:\Projects\qlib_trading_v2\csv_data\CRYPTODATA_RESAMPLE\60min\BTCUSDT.csv"
    
    try:
        df_price = pd.read_csv(price_file)
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        df_price = df_price.set_index('timestamp')
        
        print(f"Price data: {len(df_price)} observations")
        print(f"Price range: {df_price.index.min()} to {df_price.index.max()}")
        
        # Check overlap
        overlap_start = max(df_pred.index.min(), df_price.index.min())
        overlap_end = min(df_pred.index.max(), df_price.index.max())
        
        print(f"\nOverlap period: {overlap_start} to {overlap_end}")
        
        # Filter to overlap
        df_pred_overlap = df_pred.loc[overlap_start:overlap_end]
        df_price_overlap = df_price.loc[overlap_start:overlap_end]
        
        print(f"Overlapping predictions: {len(df_pred_overlap)}")
        print(f"Overlapping prices: {len(df_price_overlap)}")
        
        # Align data
        aligned = df_pred_overlap.join(df_price_overlap[['close']], how='inner')
        print(f"Aligned observations: {len(aligned)}")
        
        if len(aligned) > 0:
            print(f"Alignment success rate: {len(aligned)/len(df_pred_overlap):.1%}")
            
            # Show sample of aligned data
            print(f"\nSample aligned data:")
            sample_cols = ['q10', 'q50', 'q90', 'truth', 'close']
            available_cols = [col for col in sample_cols if col in aligned.columns]
            print(aligned[available_cols].head(10))
            
            # Validate price vs return relationship
            if 'truth' in aligned.columns and 'close' in aligned.columns:
                print(f"\nValidating price vs return relationship...")
                
                # Calculate actual returns from price data
                aligned['actual_return'] = aligned['close'].pct_change()
                
                # Compare with truth column (should be similar)
                valid_data = aligned.dropna(subset=['truth', 'actual_return'])
                correlation = valid_data['truth'].corr(valid_data['actual_return'])
                print(f"Correlation between 'truth' and actual returns: {correlation:.4f}")
                
                # Additional validation metrics
                mean_diff = (valid_data['truth'] - valid_data['actual_return']).abs().mean()
                print(f"Mean absolute difference: {mean_diff:.6f}")
                
                # Check for time alignment issues
                # Truth should predict NEXT period's return, not current
                aligned['next_return'] = aligned['actual_return'].shift(-1)
                next_correlation = valid_data['truth'].corr(aligned['next_return'].dropna())
                print(f"Correlation with NEXT period return: {next_correlation:.4f}")
                
                if correlation > 0.8:
                    print("✓ Good correlation - data appears aligned")
                elif correlation > 0.5:
                    print("⚠ Moderate correlation - check data alignment")
                elif next_correlation > correlation:
                    print("⚠ Better correlation with NEXT period - possible time shift issue")
                else:
                    print("✗ Poor correlation - data may be misaligned")
                
                # Check for missing data patterns
                missing_truth = aligned['truth'].isna().sum()
                missing_price = aligned['close'].isna().sum()
                print(f"\nMissing data:")
                print(f"  Truth: {missing_truth} ({missing_truth/len(aligned):.1%})")
                print(f"  Price: {missing_price} ({missing_price/len(aligned):.1%})")
                
                # Check data frequency consistency
                time_diffs = aligned.index.to_series().diff().dropna()
                most_common_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else None
                print(f"Most common time interval: {most_common_freq}")
                
                # Check for gaps in data
                expected_freq = pd.Timedelta(hours=1)  # Assuming hourly data
                gaps = time_diffs[time_diffs > expected_freq * 1.5]
                print(f"Data gaps (>1.5 hours): {len(gaps)}")
                
                if len(gaps) > 0:
                    print(f"Largest gap: {gaps.max()}")
                    print("First few gaps:")
                    for i, (timestamp, gap) in enumerate(gaps.head().items()):
                        print(f"  {timestamp}: {gap}")
                        if i >= 4:
                            break
                
                # Plot comparison
                plt.figure(figsize=(12, 8))
                
                # Plot 1: Price over time
                plt.subplot(2, 2, 1)
                plt.plot(aligned.index[-1000:], aligned['close'].iloc[-1000:])
                plt.title('BTC Price (Last 1000 observations)')
                plt.ylabel('Price ($)')
                
                # Plot 2: Returns comparison
                plt.subplot(2, 2, 2)
                sample_data = aligned.dropna().tail(500)
                plt.scatter(sample_data['actual_return'], sample_data['truth'], alpha=0.6)
                plt.xlabel('Actual Returns (from price)')
                plt.ylabel('Truth Returns (from data)')
                plt.title(f'Returns Correlation: {correlation:.3f}')
                
                # Plot 3: Quantile predictions
                plt.subplot(2, 2, 3)
                recent_data = aligned.tail(200)
                plt.plot(recent_data.index, recent_data['q10'], label='Q10', alpha=0.7)
                plt.plot(recent_data.index, recent_data['q50'], label='Q50', alpha=0.7)
                plt.plot(recent_data.index, recent_data['q90'], label='Q90', alpha=0.7)
                plt.plot(recent_data.index, recent_data['truth'], label='Actual', alpha=0.7)
                plt.title('Quantile Predictions vs Actual')
                plt.legend()
                
                # Plot 4: Portfolio simulation preview
                plt.subplot(2, 2, 4)
                portfolio_value = 100000
                portfolio_history = [portfolio_value]
                
                for i, (_, row) in enumerate(recent_data.iterrows()):
                    if i > 0:
                        # Simple buy-and-hold simulation
                        return_val = row['truth']
                        if not pd.isna(return_val):
                            portfolio_value *= (1 + return_val)
                        portfolio_history.append(portfolio_value)
                
                plt.plot(recent_data.index, portfolio_history[:-1])
                plt.title('Buy-and-Hold Portfolio Value')
                plt.ylabel('Portfolio Value ($)')
                
                plt.tight_layout()
                plt.savefig('data_alignment_validation.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"\nValidation plots saved as 'data_alignment_validation.png'")
                
                return aligned
        
        else:
            print("ERROR: No aligned data found!")
            return None
            
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None

def create_aligned_dataset_for_backtest(output_file: str = "aligned_backtest_data.csv"):
    """
    Create a properly aligned dataset for Hummingbot backtesting
    """
    print("=== CREATING ALIGNED DATASET FOR BACKTEST ===")
    
    # Load and align data
    aligned_data = validate_data_alignment()
    
    if aligned_data is None or len(aligned_data) == 0:
        print("ERROR: No aligned data available")
        return None
    
    # Prepare data for backtesting
    backtest_data = aligned_data.copy()
    
    # Ensure we have all required columns
    required_cols = ['q10', 'q50', 'q90', 'truth', 'close']
    missing_cols = [col for col in required_cols if col not in backtest_data.columns]
    
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Add additional useful columns
    backtest_data['actual_return'] = backtest_data['close'].pct_change()
    backtest_data['price_change'] = backtest_data['close'].diff()
    
    # Add tier confidence if available
    if 'tier_confidence' not in backtest_data.columns:
        # Create a simple tier confidence based on spread
        spread = backtest_data['q90'] - backtest_data['q10']
        backtest_data['tier_confidence'] = 10 - (spread / spread.quantile(0.95) * 5)
        backtest_data['tier_confidence'] = backtest_data['tier_confidence'].clip(1, 10)
    
    # Remove rows with missing critical data
    backtest_data = backtest_data.dropna(subset=['q10', 'q50', 'q90', 'close'])
    
    print(f"Final aligned dataset: {len(backtest_data)} observations")
    print(f"Date range: {backtest_data.index.min()} to {backtest_data.index.max()}")
    
    # Save the aligned dataset
    backtest_data.to_csv(output_file)
    print(f"Aligned dataset saved to: {output_file}")
    
    # Show summary statistics
    print(f"\nDataset Summary:")
    print(f"  Observations: {len(backtest_data):,}")
    print(f"  Price range: ${backtest_data['close'].min():.2f} - ${backtest_data['close'].max():.2f}")
    print(f"  Return range: {backtest_data['truth'].min():.4f} - {backtest_data['truth'].max():.4f}")
    print(f"  Q50 range: {backtest_data['q50'].min():.4f} - {backtest_data['q50'].max():.4f}")
    
    return backtest_data

def run_hummingbot_backtest_with_aligned_data():
    """
    Run Hummingbot backtest using properly aligned data
    """
    from src.backtesting.backtester import HummingbotQuantileBacktester
    
    print("=== RUNNING HUMMINGBOT BACKTEST WITH ALIGNED DATA ===")
    
    # Create aligned dataset
    aligned_data = create_aligned_dataset_for_backtest()
    
    if aligned_data is None:
        return None
    
    # Separate prediction data and price data
    prediction_cols = ['q10', 'q50', 'q90', 'truth', 'tier_confidence']
    prediction_data = aligned_data[prediction_cols].copy()
    
    # Create price data DataFrame
    price_data = aligned_data[['close']].copy()
    
    # Configure backtester
    backtester = HummingbotQuantileBacktester(
        initial_balance=100000.0,
        trading_pair="BTCUSDT",
        long_threshold=0.6,
        short_threshold=0.6,
        max_position_pct=0.3,
        fee_rate=0.001
    )
    
    # Run backtest
    results = backtester.run_backtest(
        df=prediction_data,
        price_data=price_data,
        price_col='close',
        return_col='truth'
    )
    
    # Generate report
    print(backtester.generate_report(results))
    
    # Save results
    backtester.save_results(results, "./hummingbot_backtest_results/aligned_data")
    
    return backtester, results

if __name__ == "__main__":
    # First validate alignment
    aligned_data = validate_data_alignment()
    
    # Then create aligned dataset and run backtest
    if aligned_data is not None:
        print("\n" + "="*60)
        backtester, results = run_hummingbot_backtest_with_aligned_data()