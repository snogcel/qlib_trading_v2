#!/usr/bin/env python3
"""
Signal Execution Analysis Tool
Analyzes the relationship between signal generation and trade execution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_signal_analysis(results_dir: str = "./hummingbot_backtest_results"):
    """Load the signal analysis data"""
    results_path = Path(results_dir)
    
    # Load main signal analysis
    signal_df = pd.read_csv(results_path / "signal_analysis_pivot.csv")
    signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])
    
    # Load summary stats
    with open(results_path / "signal_summary_stats.json", 'r') as f:
        summary_stats = json.load(f)
    
    return signal_df, summary_stats

def analyze_signal_execution_patterns(signal_df: pd.DataFrame):
    """Analyze patterns in signal generation vs execution"""
    
    print("🔍 SIGNAL EXECUTION ANALYSIS")
    print("=" * 60)
    
    # Overall execution rate
    total_signals = len(signal_df)
    executed_trades = signal_df['trade_executed'].sum()
    execution_rate = executed_trades / total_signals
    
    print(f"Total Signals Generated: {total_signals:,}")
    print(f"Trades Executed: {executed_trades:,}")
    print(f"Overall Execution Rate: {execution_rate:.2%}")
    print()
    
    # Execution rate by signal direction
    print("EXECUTION RATE BY SIGNAL DIRECTION:")
    direction_analysis = signal_df.groupby('signal_direction').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'target_pct': 'mean',
        'signal_strength': 'mean',
        'confidence': 'mean'
    }).round(3)
    
    direction_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 
                                 'Avg_Target_Pct', 'Avg_Strength', 'Avg_Confidence']
    print(direction_analysis)
    print()
    
    # Execution rate by confidence level
    print("EXECUTION RATE BY CONFIDENCE BUCKET:")
    confidence_analysis = signal_df.groupby('confidence_bucket').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'target_pct': 'mean',
        'signal_strength': 'mean'
    }).round(3)
    
    confidence_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 
                                  'Avg_Target_Pct', 'Avg_Strength']
    print(confidence_analysis)
    print()
    
    # Threshold analysis
    print("THRESHOLD ANALYSIS:")
    threshold_analysis = signal_df.groupby('signal_above_thresh').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'signal_strength': 'mean',
        'confidence': 'mean',
        'target_pct': 'mean'
    }).round(3)
    
    threshold_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 
                                 'Avg_Strength', 'Avg_Confidence', 'Avg_Target_Pct']
    threshold_analysis.index = ['Below_Threshold', 'Above_Threshold']
    print(threshold_analysis)
    print()

def analyze_hold_patterns(signal_df: pd.DataFrame):
    """Analyze why signals result in holds vs trades"""
    
    print("🔍 HOLD PATTERN ANALYSIS")
    print("=" * 60)
    
    # Hold reason distribution
    hold_data = signal_df[signal_df['action_taken'] == 'HOLD']
    
    if len(hold_data) > 0:
        print("HOLD REASON DISTRIBUTION:")
        hold_reasons = hold_data['hold_reason'].value_counts()
        for reason, count in hold_reasons.items():
            pct = count / len(hold_data) * 100
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
        print()
        
        # Hold characteristics by reason
        print("HOLD CHARACTERISTICS BY REASON:")
        hold_analysis = hold_data.groupby('hold_reason').agg({
            'signal_strength': 'mean',
            'confidence': 'mean',
            'target_pct': 'mean',
            'prob_directional': 'mean',
            'has_position': 'mean'
        }).round(3)
        
        print(hold_analysis)
        print()
    else:
        print("No hold periods found in data")

def analyze_signal_quality_vs_execution(signal_df: pd.DataFrame):
    """Analyze relationship between signal quality and execution"""
    
    print("🔍 SIGNAL QUALITY VS EXECUTION ANALYSIS")
    print("=" * 60)
    
    # Create signal quality score buckets
    signal_df['quality_bucket'] = pd.cut(signal_df['signal_quality_score'], 
                                        bins=5, labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])
    
    quality_analysis = signal_df.groupby('quality_bucket').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'signal_strength': 'mean',
        'confidence': 'mean',
        'prob_directional': 'mean',
        'target_pct': 'mean'
    }).round(3)
    
    quality_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 
                               'Avg_Strength', 'Avg_Confidence', 'Avg_Prob_Dir', 'Avg_Target_Pct']
    
    print("EXECUTION RATE BY SIGNAL QUALITY:")
    print(quality_analysis)
    print()

def analyze_time_patterns(signal_df: pd.DataFrame):
    """Analyze time-based patterns in signal execution"""
    
    print("🔍 TIME-BASED EXECUTION PATTERNS")
    print("=" * 60)
    
    # Hour of day analysis
    print("EXECUTION RATE BY HOUR:")
    hourly_analysis = signal_df.groupby('hour').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'signal_strength': 'mean'
    }).round(3)
    
    hourly_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 'Avg_Strength']
    print(hourly_analysis.head(10))  # Show first 10 hours
    print()
    
    # Day of week analysis
    print("EXECUTION RATE BY DAY OF WEEK:")
    daily_analysis = signal_df.groupby('day_of_week').agg({
        'trade_executed': ['count', 'sum', 'mean'],
        'signal_strength': 'mean'
    }).round(3)
    
    daily_analysis.columns = ['Total_Signals', 'Executed', 'Execution_Rate', 'Avg_Strength']
    print(daily_analysis)
    print()

def create_pivot_examples(signal_df: pd.DataFrame):
    """Create example pivot tables for analysis"""
    
    print("🔍 PIVOT TABLE EXAMPLES")
    print("=" * 60)
    
    # Example 1: Signal Direction vs Confidence vs Execution Rate
    print("PIVOT: Signal Direction vs Confidence Bucket (Execution Rate)")
    pivot1 = pd.pivot_table(signal_df, 
                           values='trade_executed', 
                           index='signal_direction', 
                           columns='confidence_bucket', 
                           aggfunc='mean').round(3)
    print(pivot1)
    print()
    
    # Example 2: Time vs Signal Quality vs Execution
    print("PIVOT: Hour vs Signal Quality (Execution Rate)")
    pivot2 = pd.pivot_table(signal_df, 
                           values='trade_executed', 
                           index='hour', 
                           columns='strength_bucket', 
                           aggfunc='mean').round(3)
    print(pivot2.head(8))  # Show first 8 hours
    print()
    
    # Example 3: Threshold vs Direction vs Average Target %
    print("PIVOT: Above Threshold vs Signal Direction (Average Target %)")
    pivot3 = pd.pivot_table(signal_df, 
                           values='target_pct', 
                           index='signal_above_thresh', 
                           columns='signal_direction', 
                           aggfunc='mean').round(4)
    pivot3.index = ['Below_Threshold', 'Above_Threshold']
    print(pivot3)
    print()

def generate_insights(signal_df: pd.DataFrame, summary_stats: dict):
    """Generate key insights from the analysis"""
    
    print("💡 KEY INSIGHTS")
    print("=" * 60)
    
    insights = []
    
    # Execution rate insights
    overall_execution = signal_df['trade_executed'].mean()
    if overall_execution < 0.3:
        insights.append(f"⚠️  Low overall execution rate ({overall_execution:.1%}) - many signals not resulting in trades")
    elif overall_execution > 0.7:
        insights.append(f"High execution rate ({overall_execution:.1%}) - most signals result in trades")
    
    # Direction bias insights
    direction_rates = signal_df.groupby('signal_direction')['trade_executed'].mean()
    if 'LONG' in direction_rates and 'SHORT' in direction_rates:
        long_rate = direction_rates['LONG']
        short_rate = direction_rates['SHORT']
        if abs(long_rate - short_rate) > 0.2:
            bias_direction = "LONG" if long_rate > short_rate else "SHORT"
            insights.append(f"📈 Execution bias toward {bias_direction} signals ({long_rate:.1%} vs {short_rate:.1%})")
    
    # Confidence insights
    high_conf = signal_df[signal_df['high_confidence'] == True]['trade_executed'].mean()
    low_conf = signal_df[signal_df['high_confidence'] == False]['trade_executed'].mean()
    if high_conf - low_conf > 0.1:
        insights.append(f"🎯 High confidence signals execute more often ({high_conf:.1%} vs {low_conf:.1%})")
    
    # Threshold insights
    above_thresh = signal_df[signal_df['above_threshold'] == True]['trade_executed'].mean()
    below_thresh = signal_df[signal_df['above_threshold'] == False]['trade_executed'].mean()
    if above_thresh - below_thresh > 0.1:
        insights.append(f"🎯 Above-threshold signals execute more often ({above_thresh:.1%} vs {below_thresh:.1%})")
    
    # Hold pattern insights
    hold_data = signal_df[signal_df['action_taken'] == 'HOLD']
    if len(hold_data) > 0:
        most_common_hold = hold_data['hold_reason'].mode().iloc[0]
        hold_pct = len(hold_data) / len(signal_df) * 100
        insights.append(f"⏸️  {hold_pct:.1f}% of signals result in holds, most common: {most_common_hold}")
    
    # Time pattern insights
    weekend_exec = signal_df[signal_df['weekend'] == True]['trade_executed'].mean()
    weekday_exec = signal_df[signal_df['weekend'] == False]['trade_executed'].mean()
    if abs(weekend_exec - weekday_exec) > 0.1:
        better_time = "weekends" if weekend_exec > weekday_exec else "weekdays"
        insights.append(f"📅 Better execution on {better_time} ({max(weekend_exec, weekday_exec):.1%} vs {min(weekend_exec, weekday_exec):.1%})")
    
    # Print insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    if not insights:
        print("No significant patterns detected in current data.")
    
    print()

def main():
    """Main analysis function"""
    
    try:
        # Load data
        signal_df, summary_stats = load_signal_analysis()
        
        print(f"Loaded {len(signal_df):,} signal observations for analysis")
        print(f"📅 Date range: {signal_df['timestamp'].min()} to {signal_df['timestamp'].max()}")
        print()
        
        # Run analyses
        analyze_signal_execution_patterns(signal_df)
        analyze_hold_patterns(signal_df)
        analyze_signal_quality_vs_execution(signal_df)
        analyze_time_patterns(signal_df)
        create_pivot_examples(signal_df)
        generate_insights(signal_df, summary_stats)
        
        print("Analysis complete! Use the CSV files for detailed pivot table analysis in Excel/Google Sheets.")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find signal analysis files. Run a backtest first.")
        print(f"   Expected files in: ./hummingbot_backtest_results/")
        print(f"   Missing: {e}")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()