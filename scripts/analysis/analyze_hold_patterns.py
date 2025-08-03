"""
Analyze hold patterns in Hummingbot backtest results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_hold_patterns():
    """
    Analyze hold patterns from Hummingbot backtest results
    """
    print("=== HOLD PATTERN ANALYSIS ===")
    
    try:
        # Load portfolio history
        portfolio = pd.read_csv("hummingbot_backtest_results/moderate/portfolio_history.csv")
        
        # Load holds analysis if available
        holds_file = "hummingbot_backtest_results/moderate/holds_analysis.csv"
        try:
            holds = pd.read_csv(holds_file)
            print(f"Loaded {len(holds)} hold records")
        except FileNotFoundError:
            print("No holds_analysis.csv found, analyzing from portfolio history")
            holds = None
        
        print(f"Portfolio history: {len(portfolio)} observations")
        
        # Analyze action distribution
        if 'action_taken' in portfolio.columns:
            action_counts = portfolio['action_taken'].value_counts()
            print(f"\nAction Distribution:")
            for action, count in action_counts.items():
                pct = count / len(portfolio) * 100
                print(f"  {action}: {count:,} ({pct:.1f}%)")
        
        # Analyze hold reasons
        if 'hold_reason' in portfolio.columns:
            hold_reasons = portfolio['hold_reason'].dropna().value_counts()
            print(f"\nHold Reason Distribution:")
            for reason, count in hold_reasons.items():
                pct = count / len(portfolio) * 100
                print(f"  {reason}: {count:,} ({pct:.1f}%)")
        
        # Analyze hold patterns over time
        if holds is not None and len(holds) > 0:
            holds['timestamp'] = pd.to_datetime(holds['timestamp_formatted'])
            holds['date'] = holds['timestamp'].dt.date
            holds['hour'] = holds['timestamp'].dt.hour
            
            # Daily hold patterns
            daily_holds = holds.groupby('date').size()
            print(f"\nDaily Hold Statistics:")
            print(f"  Average holds per day: {daily_holds.mean():.1f}")
            print(f"  Max holds in a day: {daily_holds.max()}")
            print(f"  Days with holds: {len(daily_holds)}")
            
            # Hourly hold patterns
            hourly_holds = holds.groupby('hour')['hold_reason'].value_counts()
            print(f"\nTop hourly hold patterns:")
            print(hourly_holds.head(10))
        
        # Create visualizations
        create_hold_visualizations(portfolio, holds)
        
        return portfolio, holds
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Run the Hummingbot backtest first: python run_hummingbot_backtest.py")
        return None, None

def create_hold_visualizations(portfolio, holds):
    """
    Create visualizations for hold patterns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Action distribution over time
    if 'action_taken' in portfolio.columns:
        portfolio['timestamp'] = pd.to_datetime(portfolio['timestamp_formatted'])
        
        # Sample data for visualization (last 1000 points)
        sample_portfolio = portfolio.tail(1000)
        
        actions = sample_portfolio['action_taken']
        trade_mask = actions == 'TRADE'
        hold_mask = actions == 'HOLD'
        
        axes[0, 0].scatter(sample_portfolio['timestamp'][trade_mask], 
                          sample_portfolio['portfolio_value'][trade_mask], 
                          c='red', s=20, label='TRADE', alpha=0.7)
        axes[0, 0].plot(sample_portfolio['timestamp'], 
                       sample_portfolio['portfolio_value'], 
                       alpha=0.3, color='blue', label='Portfolio Value')
        axes[0, 0].set_title('Portfolio Value with Trade Points')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Action distribution pie chart
    if 'action_taken' in portfolio.columns:
        action_counts = portfolio['action_taken'].value_counts()
        axes[0, 1].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Action Distribution')
    
    # Plot 3: Hold reasons distribution
    if 'hold_reason' in portfolio.columns:
        hold_reasons = portfolio['hold_reason'].dropna().value_counts()
        if len(hold_reasons) > 0:
            axes[1, 0].bar(range(len(hold_reasons)), hold_reasons.values)
            axes[1, 0].set_xticks(range(len(hold_reasons)))
            axes[1, 0].set_xticklabels(hold_reasons.index, rotation=45, ha='right')
            axes[1, 0].set_title('Hold Reasons Distribution')
            axes[1, 0].set_ylabel('Count')
    
    # Plot 4: Position over time with hold/trade markers
    if 'position' in portfolio.columns:
        sample_portfolio = portfolio.tail(1000)
        
        axes[1, 1].plot(sample_portfolio['timestamp'], sample_portfolio['position'], 
                       label='Position', alpha=0.7)
        
        # Mark trades
        if 'action_taken' in sample_portfolio.columns:
            trade_points = sample_portfolio[sample_portfolio['action_taken'] == 'TRADE']
            axes[1, 1].scatter(trade_points['timestamp'], trade_points['position'], 
                             c='red', s=30, label='Trade Points', zorder=5)
        
        axes[1, 1].set_title('Position Over Time')
        axes[1, 1].set_ylabel('Position (BTC)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('hold_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Hold pattern analysis plots saved as 'hold_pattern_analysis.png'")

def analyze_hold_effectiveness():
    """
    Analyze the effectiveness of hold decisions
    """
    print("\n=== HOLD EFFECTIVENESS ANALYSIS ===")
    
    try:
        portfolio = pd.read_csv("hummingbot_backtest_results/moderate/portfolio_history.csv")
        
        if 'action_taken' not in portfolio.columns:
            print("No action_taken column found")
            return
        
        # Calculate returns after hold periods
        portfolio['next_return'] = portfolio['pnl'].shift(-1)
        
        hold_periods = portfolio[portfolio['action_taken'] == 'HOLD']
        trade_periods = portfolio[portfolio['action_taken'] == 'TRADE']
        
        if len(hold_periods) > 0:
            avg_return_after_hold = hold_periods['next_return'].mean()
            print(f"Average return after HOLD: {avg_return_after_hold:.6f}")
        
        if len(trade_periods) > 0:
            avg_return_after_trade = trade_periods['next_return'].mean()
            print(f"Average return after TRADE: {avg_return_after_trade:.6f}")
        
        # Analyze hold reasons effectiveness
        if 'hold_reason' in portfolio.columns:
            hold_effectiveness = portfolio.groupby('hold_reason')['next_return'].agg(['mean', 'count'])
            print(f"\nHold Reason Effectiveness:")
            print(hold_effectiveness.sort_values('mean', ascending=False))
        
    except Exception as e:
        print(f"Error in effectiveness analysis: {e}")

if __name__ == "__main__":
    portfolio, holds = analyze_hold_patterns()
    if portfolio is not None:
        analyze_hold_effectiveness()