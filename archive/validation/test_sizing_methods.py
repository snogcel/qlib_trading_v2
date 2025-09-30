#!/usr/bin/env python3
"""
Test different position sizing methods to ensure they produce different results
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

def test_sizing_methods():
    """Test that different sizing methods produce different position sizes"""
    
    # Load a small sample of data
    df = pd.read_csv("df_all_macro_analysis.csv")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    
    # Get a sample row
    sample_row = df.iloc[100]  # Use a row with good data
    
    print("Testing position sizing methods with sample data:")
    print(f"q10: {sample_row['q10']:.6f}")
    print(f"q50: {sample_row['q50']:.6f}")
    print(f"q90: {sample_row['q90']:.6f}")
    print(f"signal_tier: {sample_row['signal_tier']}")
    print(f"spread_thresh: {sample_row.get('spread_thresh', 'N/A')}")
    print(f"signal_thresh_adaptive: {sample_row.get('signal_thresh_adaptive', 'N/A')}")
    
    # Test different sizing methods
    sizing_methods = ['simple', 'kelly', 'volatility', 'sharpe', 'risk_parity', 'enhanced']
    
    results = {}
    
    for method in sizing_methods:
        backtester = HummingbotQuantileBacktester(
            initial_balance=10000.0,
            max_position_pct=0.5,
            sizing_method=method
        )
        
        # Generate signal to test sizing
        signal = backtester.generate_hummingbot_signal(sample_row)
        
        results[method] = {
            'target_pct': signal['target_pct'],
            'signal_direction': signal['signal_direction'],
            'confidence': signal['confidence'],
            'signal_strength': signal['signal_strength']
        }
        
        print(f"\n{method.upper()} method:")
        print(f"  Target %: {signal['target_pct']:.4f}")
        print(f"  Direction: {signal['signal_direction']}")
        print(f"  Confidence: {signal['confidence']}")
        print(f"  Signal strength: {signal['signal_strength']:.4f}")
    
    # Check if methods produce different results
    target_pcts = [results[method]['target_pct'] for method in sizing_methods]
    unique_pcts = len(set(target_pcts))
    
    print(f"\n{'='*50}")
    print(f"SIZING METHOD COMPARISON")
    print(f"{'='*50}")
    print(f"Unique position sizes generated: {unique_pcts}/{len(sizing_methods)}")
    
    if unique_pcts == 1:
        print("⚠️  WARNING: All sizing methods produced identical results!")
        print("This suggests an issue with the sizing logic or data.")
    else:
        print("Different sizing methods are producing different results")
    
    # Show range of position sizes
    min_pct = min(target_pcts)
    max_pct = max(target_pcts)
    print(f"Position size range: {min_pct:.4f} to {max_pct:.4f}")
    
    return results

if __name__ == "__main__":
    test_sizing_methods()