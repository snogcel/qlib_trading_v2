"""
Analyze trading frequency and signal distribution
"""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_trading_patterns():
    """
    Analyze why there are gaps in trades and what's happening during those periods
    """
    
    # Load both files
    try:
        portfolio = pd.read_csv("hummingbot_backtest_results/moderate/portfolio_history.csv")
        trades = pd.read_csv("hummingbot_backtest_results/moderate/trades.csv")
        
        print("=== TRADING FREQUENCY ANALYSIS ===")
        print(f"Total observations: {len(portfolio):,}")
        print(f"Total trades: {len(trades):,}")
        print(f"Trading frequency: {len(trades)/len(portfolio):.2%}")
        
        # Analyze signal distribution
        if 'signal' in portfolio.columns:
            # Extract signal direction from signal column (it's stored as dict string)
            portfolio['signal_direction'] = portfolio['signal'].str.extract(r"'signal_direction': '(\w+)'")
            
            signal_counts = portfolio['signal_direction'].value_counts()
            print(f"\nSignal Distribution:")
            for signal, count in signal_counts.items():
                pct = count / len(portfolio) * 100
                print(f"  {signal}: {count:,} ({pct:.1f}%)")
        
        # Analyze time between trades
        if len(trades) > 1:
            trades['timestamp'] = pd.to_datetime(trades['timestamp_formatted'])
            trades = trades.sort_values('timestamp')
            trades['time_between_trades'] = trades['timestamp'].diff()
            
            print(f"\nTime Between Trades:")
            print(f"  Average: {trades['time_between_trades'].mean()}")
            print(f"  Median: {trades['time_between_trades'].median()}")
            print(f"  Min: {trades['time_between_trades'].min()}")
            print(f"  Max: {trades['time_between_trades'].max()}")
        
        # Analyze position holding periods
        portfolio['timestamp'] = pd.to_datetime(portfolio['timestamp_formatted'])
        portfolio = portfolio.sort_values('timestamp')
        
        # Find position changes
        portfolio['position_changed'] = portfolio['position'].diff().abs() > 0.001
        position_changes = portfolio[portfolio['position_changed']]
        
        print(f"\nPosition Analysis:")
        print(f"  Position changes: {len(position_changes):,}")
        print(f"  Average holding period: {len(portfolio) / len(position_changes):.1f} hours")
        
        # Show sample of non-trading periods
        print(f"\nSample of Non-Trading Periods:")
        non_trading = portfolio[~portfolio['trade_executed']]
        if len(non_trading) > 0:
            sample = non_trading.head(10)[['timestamp_formatted', 'signal_direction', 'position', 'portfolio_value']]
            print(sample.to_string(index=False))
        
        return portfolio, trades
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Run the Hummingbot backtest first: python run_hummingbot_backtest.py")
        return None, None

def plot_trading_activity(portfolio, trades):
    """
    Plot trading activity over time
    """
    if portfolio is None or trades is None:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Portfolio value over time with trade markers
    axes[0].plot(portfolio['timestamp'], portfolio['portfolio_value'], label='Portfolio Value', alpha=0.7)
    if len(trades) > 0:
        trade_times = pd.to_datetime(trades['timestamp_formatted'])
        trade_values = []
        for t in trade_times:
            closest_idx = (pd.to_datetime(portfolio['timestamp']) - t).abs().idxmin()
            trade_values.append(portfolio.loc[closest_idx, 'portfolio_value'])
        
        axes[0].scatter(trade_times, trade_values, color='red', s=50, label='Trades', zorder=5)
    
    axes[0].set_title('Portfolio Value Over Time (Red dots = Trades)')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Position over time
    axes[1].plot(portfolio['timestamp'], portfolio['position'], label='Position', color='green')
    axes[1].set_title('Position Over Time')
    axes[1].set_ylabel('Position (BTC)')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Trade frequency (trades per day)
    if len(trades) > 0:
        trades['date'] = pd.to_datetime(trades['timestamp_formatted']).dt.date
        daily_trades = trades.groupby('date').size()
        axes[2].bar(daily_trades.index, daily_trades.values, alpha=0.7)
        axes[2].set_title('Trades Per Day')
        axes[2].set_ylabel('Number of Trades')
        axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('trading_activity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Trading activity plot saved as 'trading_activity_analysis.png'")

if __name__ == "__main__":
    portfolio, trades = analyze_trading_patterns()
    if portfolio is not None:
        plot_trading_activity(portfolio, trades)