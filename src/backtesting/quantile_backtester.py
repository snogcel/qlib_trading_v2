"""
Quantile-based Trading Strategy Backtester
Evaluates trading performance using Q10/Q50/Q90 predictions with tier-based position sizing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    initial_capital: float = 100000.0
    position_limit: float = 1.0  # Max position as fraction of capital
    fee_rate: float = 0.001  # 0.1% trading fee
    slippage_rate: float = 0.0005  # 0.05% slippage
    
    # Tier-based thresholds
    long_threshold: float = 0.5  # Hummingbot threshold
    short_threshold: float = 0.5
    
    # Risk management
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    volatility_lookback: int = 20
    
    # Position sizing
    base_position_size: float = 0.1  # 10% of capital base size
    confidence_multiplier: float = 2.0  # Max multiplier from tier confidence

@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: pd.Timestamp
    side: str  # 'BUY', 'SELL', 'HOLD'
    position_before: float
    position_after: float
    price: float
    quantity: float
    tier: str
    tier_confidence: float
    q10: float
    q50: float
    q90: float
    prob_up: float
    prob_down: float
    prob_neutral: float
    fee_cost: float
    slippage_cost: float
    pnl: float
    cumulative_pnl: float
    portfolio_value: float
    drawdown: float

class QuantileBacktester:
    """
    Backtester for quantile-based trading strategies
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[TradeRecord] = []
        self.portfolio_history: List[Dict] = []
        self.metrics: Dict = {}
        
    def calculate_probabilities(self, q10: float, q50: float, q90: float) -> Tuple[float, float, float]:
        """
        Convert quantiles to probabilities using your piecewise logic
        Returns: (prob_down, prob_neutral, prob_up)
        """
        # Calculate probability of upside movement (price > 0)
        if q90 <= 0:
            prob_up = 0.0
        elif q10 >= 0:
            prob_up = 1.0
        elif q10 < 0 <= q50:
            # 0 lies between q10 and q50
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            prob_up = 1 - cdf0
        else:
            # 0 lies between q50 and q90
            cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
            prob_up = 1 - cdf0
        
        prob_down = 1 - prob_up
        
        # Use spread for neutral probability
        spread = q90 - q10
        spread_normalized = min(spread / 0.02, 1.0)  # normalize by typical spread
        neutral_weight = spread_normalized * 0.3  # max 30% neutral
        
        # Redistribute probabilities
        prob_up_adj = prob_up * (1 - neutral_weight)
        prob_down_adj = prob_down * (1 - neutral_weight)
        prob_neutral = neutral_weight
        
        return prob_down_adj, prob_neutral, prob_up_adj
    
    def calculate_position_size(self, tier_confidence: float, volatility: float, 
                              current_capital: float) -> float:
        """
        Calculate position size based on tier confidence and volatility
        """
        # Base position as fraction of capital
        base_size = self.config.base_position_size
        
        # Adjust by tier confidence (1-10 scale)
        confidence_adj = 1.0 + (tier_confidence - 5.0) / 5.0 * (self.config.confidence_multiplier - 1.0)
        confidence_adj = np.clip(confidence_adj, 0.1, self.config.confidence_multiplier)
        
        # Adjust by volatility (inverse relationship)
        vol_adj = 1.0 / (1.0 + volatility * 10)  # Higher vol = smaller position
        
        # Calculate final position size
        position_size = base_size * confidence_adj * vol_adj
        position_size = np.clip(position_size, 0.01, self.config.position_limit)
        
        return position_size
    
    def generate_trading_signal(self, row: pd.Series, current_capital: float) -> Tuple[str, float]:
        """
        Generate trading signal from quantile predictions
        Returns: (signal, target_position)
        """
        q10, q50, q90 = row['q10'], row['q50'], row['q90']
        tier_confidence = row.get('tier_confidence', 5.0)
        volatility = row.get('volatility', row.get('$realized_vol_10', 0.02))
        
        # Calculate probabilities
        prob_down, prob_neutral, prob_up = self.calculate_probabilities(q10, q50, q90)
        
        # Determine signal based on Hummingbot thresholds
        if prob_up > self.config.long_threshold:
            signal = 'BUY'
            position_size = self.calculate_position_size(tier_confidence, volatility, current_capital)
            target_position = position_size
        elif prob_down > self.config.short_threshold:
            signal = 'SELL'
            position_size = self.calculate_position_size(tier_confidence, volatility, current_capital)
            target_position = -position_size
        else:
            signal = 'HOLD'
            target_position = 0.0
        
        return signal, target_position
    
    def calculate_trading_costs(self, position_change: float, price: float) -> Tuple[float, float]:
        """Calculate trading fees and slippage"""
        notional = abs(position_change * price)
        fee = notional * self.config.fee_rate
        slippage = notional * self.config.slippage_rate * abs(position_change)  # Quadratic slippage
        return fee, slippage
    
    def run_backtest(self, df: pd.DataFrame, price_col: str = 'truth', 
                    data_frequency: str = 'hourly', prediction_horizon_hours: float = 1.0) -> pd.DataFrame:
        """
        Run backtest on prediction data with proper time alignment
        
        Args:
            df: DataFrame with columns ['q10', 'q50', 'q90', 'tier_confidence', 'truth', etc.]
            price_col: Column name for actual price returns
            data_frequency: 'hourly', 'daily', 'weekly', or 'monthly' for proper annualization
            
        Returns:
            DataFrame with trade records and portfolio performance
        """
        print(f"Starting backtest with {len(df)} observations...")
        print(f"Data frequency: {data_frequency}")
        print(f"Prediction horizon: {prediction_horizon_hours} hours")
        
        # CRITICAL FIX: Prevent look-ahead bias by aligning predictions with future returns
        df = df.copy()
        df['future_return'] = df[price_col].shift(-1)  # Next period's return for PnL calculation
        
        # Remove last row which has NaN future return
        df = df.dropna(subset=['future_return'])
        print(f"After alignment: {len(df)} observations")
        
        # Initialize portfolio
        capital = self.config.initial_capital
        position = 0.0
        cumulative_pnl = 0.0
        peak_value = capital
        
        self.trades = []
        self.portfolio_history = []
        
        # Store original index for proper timestamp handling
        original_index = df.index.copy()
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # 1. Calculate PnL from PREVIOUS position using CURRENT period's actual return
            if i > 0:
                current_return = row[price_col]  # This period's actual return
                pnl = position * current_return * capital
                cumulative_pnl += pnl
                capital += pnl
            else:
                pnl = 0.0
            
            # 2. Generate NEW trading signal for NEXT period using current predictions
            signal, target_position = self.generate_trading_signal(row, capital)
            
            # 3. Calculate position change and trading costs
            position_change = target_position - position
            fee, slippage = self.calculate_trading_costs(position_change, 1.0)
            
            # 4. Apply trading costs
            capital -= (fee + slippage)
            cumulative_pnl -= (fee + slippage)
            
            # 5. Update drawdown tracking
            if capital > peak_value:
                peak_value = capital
            drawdown = (peak_value - capital) / peak_value
            
            # 6. Risk management: Stop trading if max drawdown exceeded
            if drawdown > self.config.max_drawdown_limit:
                target_position = 0.0
                position_change = -position
                signal = 'LIQUIDATE'
                print(f"Max drawdown exceeded at {timestamp}, liquidating positions")
            
            # 7. Calculate probabilities for logging
            prob_down, prob_neutral, prob_up = self.calculate_probabilities(
                row['q10'], row['q50'], row['q90']
            )
            
            # 8. Create trade record with original timestamp preserved
            # Store both the original timestamp and a string representation
            original_timestamp = original_index[i] if i < len(original_index) else timestamp
            
            trade = TradeRecord(
                timestamp=original_timestamp,
                side=signal,
                position_before=position,
                position_after=target_position,
                price=1.0,  # Normalized price for return calculation
                quantity=abs(position_change),
                tier=row.get('signal_tier', 'Unknown'),
                tier_confidence=row.get('tier_confidence', 5.0),
                q10=row['q10'],
                q50=row['q50'],
                q90=row['q90'],
                prob_up=prob_up,
                prob_down=prob_down,
                prob_neutral=prob_neutral,
                fee_cost=fee,
                slippage_cost=slippage,
                pnl=pnl,
                cumulative_pnl=cumulative_pnl,
                portfolio_value=capital,
                drawdown=drawdown
            )
            
            self.trades.append(trade)
            
            # 9. Update position for next iteration
            position = target_position
            
            # 10. Log portfolio state
            self.portfolio_history.append({
                'timestamp': original_timestamp,
                'capital': capital,
                'position': position,
                'cumulative_pnl': cumulative_pnl,
                'drawdown': drawdown,
                'signal': signal
            })
            
            if i % 1000 == 0:
                print(f"Processed {i}/{len(df)} observations, Capital: ${capital:,.2f}")
        
        print(f"Backtest completed. Final capital: ${capital:,.2f}")
        
        # Convert to DataFrame with better timestamp handling
        trades_data = []
        for t in self.trades:
            trade_dict = {
                'timestamp': t.timestamp,
                'side': t.side,
                'position_before': t.position_before,
                'position_after': t.position_after,
                'tier': t.tier,
                'tier_confidence': t.tier_confidence,
                'q10': t.q10,
                'q50': t.q50,
                'q90': t.q90,
                'prob_up': t.prob_up,
                'prob_down': t.prob_down,
                'prob_neutral': t.prob_neutral,
                'fee_cost': t.fee_cost,
                'slippage_cost': t.slippage_cost,
                'pnl': t.pnl,
                'cumulative_pnl': t.cumulative_pnl,
                'portfolio_value': t.portfolio_value,
                'drawdown': t.drawdown
            }
            trades_data.append(trade_dict)
        
        trades_df = pd.DataFrame(trades_data)
        
        # Debug timestamp before setting as index
        if len(trades_df) > 0:
            print(f"DEBUG: Timestamp type in trades_df: {type(trades_df['timestamp'].iloc[0])}")
            print(f"DEBUG: First timestamp value: {trades_df['timestamp'].iloc[0]}")
        
        trades_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics with proper frequency adjustment
        self._calculate_metrics(trades_df, data_frequency, prediction_horizon_hours)
        
        # Validate results for common errors
        self._validate_backtest_results(trades_df)
        
        return trades_df
    
    def _calculate_metrics(self, trades_df: pd.DataFrame, data_frequency: str = 'hourly', 
                          prediction_horizon_hours: float = 1.0):
        """Calculate comprehensive performance metrics with proper frequency adjustment"""
        
        # Calculate annualization factor based on PREDICTION HORIZON, not just data frequency
        # This is the key fix for your 12-hour horizon issue
        
        # Base trading hours per year
        trading_hours_per_year = 252 * 24  # 252 trading days * 24 hours
        
        # Adjust for prediction horizon
        # If predicting 12 hours ahead, you get 252*24/12 = 504 trading opportunities per year
        periods_per_year = trading_hours_per_year / prediction_horizon_hours
        
        print(f"Data frequency: {data_frequency}")
        print(f"Prediction horizon: {prediction_horizon_hours} hours")
        print(f"Trading periods per year: {periods_per_year:.0f}")
        print(f"Using annualization factor: {periods_per_year:.0f}")
        
        # Basic metrics
        total_return = (trades_df['portfolio_value'].iloc[-1] / self.config.initial_capital) - 1
        total_trades = len(trades_df[trades_df['side'] != 'HOLD'])
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # Risk metrics with proper frequency adjustment
        returns = trades_df['pnl'] / self.config.initial_capital
        volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 1 else 0
        mean_return = returns.mean() * periods_per_year
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        max_drawdown = trades_df['drawdown'].max()
        
        # Proper annualized return calculation
        periods = len(trades_df)
        if periods > 0:
            annualized_return = ((1 + total_return) ** (periods_per_year / periods)) - 1
        else:
            annualized_return = 0
        
        # Trade analysis
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Tier analysis
        tier_performance = trades_df.groupby('tier').agg({
            'pnl': ['sum', 'mean', 'count'],
            'tier_confidence': 'mean'
        }).round(4)
        
        # Probability calibration
        try:
            prob_bins = pd.cut(trades_df['prob_up'], bins=10)
            actual_up = trades_df.groupby(prob_bins)['pnl'].apply(lambda x: (x > 0).mean())
        except:
            actual_up = pd.Series()
        
        # Convert to JSON-friendly format
        try:
            # Flatten tier_performance MultiIndex columns
            tier_performance_flat = tier_performance.copy()
            tier_performance_flat.columns = ['_'.join(map(str, col)).strip() for col in tier_performance.columns.values]
            tier_performance = tier_performance_flat
        except:
            tier_performance = pd.DataFrame()
        
        try:
            # Convert probability calibration to simple dict
            actual_up = actual_up.reset_index()
            actual_up['prob_up_str'] = actual_up['prob_up'].astype(str)
            actual_up = actual_up.set_index('prob_up_str')['pnl']
        except:
            actual_up = pd.Series()
        
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else np.inf,
            'data_frequency': data_frequency,
            'prediction_horizon_hours': prediction_horizon_hours,
            'periods_per_year': periods_per_year,
            'tier_performance': tier_performance,
            'probability_calibration': actual_up
        }
    
    def _validate_backtest_results(self, trades_df: pd.DataFrame):
        """Validate backtest results for common errors and biases"""
        
        print("\n" + "="*50)
        print("BACKTEST VALIDATION CHECKS")
        print("="*50)
        
        warnings = []
        
        # Check for suspiciously high returns
        daily_return_equiv = self.metrics['total_return'] / len(trades_df) * 252
        if daily_return_equiv > 1.0:  # >100% annualized equivalent
            warnings.append(f"WARNING: Very high returns ({daily_return_equiv:.1%} annualized equivalent)")
        
        # Check Sharpe ratio reasonableness
        if self.metrics['sharpe_ratio'] > 3.0:
            warnings.append(f"WARNING: Suspiciously high Sharpe ratio ({self.metrics['sharpe_ratio']:.2f})")
        elif self.metrics['sharpe_ratio'] > 2.0:
            print(f"NOTE: High but reasonable Sharpe ratio ({self.metrics['sharpe_ratio']:.2f})")
        
        # Check win rate
        if self.metrics['win_rate'] > 0.8:
            warnings.append(f"WARNING: Very high win rate ({self.metrics['win_rate']:.1%})")
        elif self.metrics['win_rate'] > 0.65:
            print(f"NOTE: High win rate ({self.metrics['win_rate']:.1%})")
        
        # Check for reasonable volatility
        if self.metrics['volatility'] < 0.05:  # <5% annualized volatility
            warnings.append(f"WARNING: Suspiciously low volatility ({self.metrics['volatility']:.1%})")
        
        # Check profit factor
        if self.metrics['profit_factor'] > 5.0:
            warnings.append(f"WARNING: Very high profit factor ({self.metrics['profit_factor']:.2f})")
        
        # Check for reasonable number of trades
        if self.metrics['total_trades'] < len(trades_df) * 0.01:  # <1% of observations result in trades
            warnings.append(f"WARNING: Very few trades ({self.metrics['total_trades']} out of {len(trades_df)} observations)")
        
        # Check drawdown reasonableness
        if self.metrics['max_drawdown'] < 0.02:  # <2% max drawdown
            warnings.append(f"WARNING: Suspiciously low max drawdown ({self.metrics['max_drawdown']:.1%})")
        
        # Print warnings
        if warnings:
            print("\nPOTENTIAL ISSUES DETECTED:")
            for warning in warnings:
                print(f"  {warning}")
            print("\nThese results may indicate:")
            print("  - Look-ahead bias (using future information)")
            print("  - Data leakage (features contain target information)")
            print("  - Overfitting to specific time period")
            print("  - Incorrect time alignment")
        else:
            print("✓ No obvious validation issues detected")
        
        print("="*50)
    
    def plot_results(self, trades_df: pd.DataFrame, save_path: Optional[str] = None):
        """Generate comprehensive performance plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Quantile Trading Strategy Backtest Results', fontsize=16)
        
        # 1. Portfolio Value Over Time
        axes[0, 0].plot(trades_df.index, trades_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # 2. Drawdown
        axes[0, 1].fill_between(trades_df.index, trades_df['drawdown'], alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # 3. Position Over Time
        axes[0, 2].plot(trades_df.index, trades_df['position_after'])
        axes[0, 2].set_title('Position Over Time')
        axes[0, 2].set_ylabel('Position Size')
        axes[0, 2].grid(True)
        
        # 4. PnL Distribution
        pnl_data = trades_df[trades_df['pnl'] != 0]['pnl']
        axes[1, 0].hist(pnl_data, bins=50, alpha=0.7)
        axes[1, 0].set_title('PnL Distribution')
        axes[1, 0].set_xlabel('PnL ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 5. Tier Performance
        tier_pnl = trades_df.groupby('tier')['pnl'].sum()
        axes[1, 1].bar(tier_pnl.index.astype(str), tier_pnl.values)
        axes[1, 1].set_title('PnL by Tier')
        axes[1, 1].set_xlabel('Tier')
        axes[1, 1].set_ylabel('Total PnL ($)')
        axes[1, 1].grid(True)
        
        # 6. Probability Calibration
        prob_bins = pd.cut(trades_df['prob_up'], bins=10)
        actual_up = trades_df.groupby(prob_bins)['pnl'].apply(lambda x: (x > 0).mean())
        predicted_up = trades_df.groupby(prob_bins)['prob_up'].mean()
        
        axes[1, 2].scatter(predicted_up, actual_up, alpha=0.7)
        axes[1, 2].plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        axes[1, 2].set_title('Probability Calibration')
        axes[1, 2].set_xlabel('Predicted Probability')
        axes[1, 2].set_ylabel('Actual Frequency')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, trades_df: pd.DataFrame) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
=== QUANTILE TRADING STRATEGY BACKTEST REPORT ===

PERFORMANCE SUMMARY:
- Total Return: {self.metrics['total_return']:.2%}
- Annualized Return: {self.metrics['annualized_return']:.2%}
- Volatility (Annualized): {self.metrics['volatility']:.2%}
- Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
- Maximum Drawdown: {self.metrics['max_drawdown']:.2%}

TRADING STATISTICS:
- Total Trades: {self.metrics['total_trades']:,}
- Win Rate: {self.metrics['win_rate']:.2%}
- Average Win: ${self.metrics['avg_win']:.2f}
- Average Loss: ${self.metrics['avg_loss']:.2f}
- Profit Factor: {self.metrics['profit_factor']:.2f}

DATA INFORMATION:
- Data Frequency: {self.metrics.get('data_frequency', 'unknown')}
- Prediction Horizon: {self.metrics.get('prediction_horizon_hours', 'unknown')} hours
- Periods Per Year: {self.metrics.get('periods_per_year', 'unknown'):.0f}
- Total Observations: {len(trades_df):,}

TIER PERFORMANCE:
{self.metrics['tier_performance']}

CONFIGURATION:
- Initial Capital: ${self.config.initial_capital:,.2f}
- Position Limit: {self.config.position_limit:.1%}
- Fee Rate: {self.config.fee_rate:.3%}
- Slippage Rate: {self.config.slippage_rate:.4%}
- Long Threshold: {self.config.long_threshold:.2f}
- Short Threshold: {self.config.short_threshold:.2f}
"""
        
        return report
    
    def save_results(self, trades_df: pd.DataFrame, output_dir: str = "./backtest_results"):
        """Save backtest results to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save trades with properly formatted timestamps and additional analysis columns
        trades_for_save = trades_df.copy()
        
        # Reset index to make timestamp a regular column
        trades_for_save = trades_for_save.reset_index()
        
        # Enhanced timestamp formatting - now should work with proper datetime index
        if 'timestamp' in trades_for_save.columns:
            print(f"DEBUG: Original timestamp type: {type(trades_for_save['timestamp'].iloc[0])}")
            print(f"DEBUG: First timestamp value: {trades_for_save['timestamp'].iloc[0]}")
            
            try:
                # Convert to pandas datetime series
                timestamp_series = pd.to_datetime(trades_for_save['timestamp'])
                
                # Format the timestamps
                trades_for_save['timestamp_formatted'] = timestamp_series.dt.strftime('%Y-%m-%d %H:%M:%S')
                trades_for_save['date'] = timestamp_series.dt.date
                trades_for_save['hour'] = timestamp_series.dt.hour
                trades_for_save['day_of_week'] = timestamp_series.dt.day_name()
                
                print(f"DEBUG: Successfully formatted timestamps. First: {trades_for_save['timestamp_formatted'].iloc[0]}")
                
            except Exception as e:
                print(f"DEBUG: Timestamp conversion failed: {e}")
                # Fallback to string representation
                trades_for_save['timestamp_formatted'] = trades_for_save['timestamp'].astype(str)
                trades_for_save['date'] = trades_for_save['timestamp'].astype(str)
                trades_for_save['hour'] = 0
                trades_for_save['day_of_week'] = 'Unknown'
        
        # Add some helpful analysis columns
        trades_for_save['position_change'] = trades_for_save['position_after'] - trades_for_save['position_before']
        trades_for_save['total_cost'] = trades_for_save['fee_cost'] + trades_for_save['slippage_cost']
        trades_for_save['is_profitable'] = trades_for_save['pnl'] > 0
        trades_for_save['spread'] = trades_for_save['q90'] - trades_for_save['q10']
        
        # Reorder columns for better readability
        column_order = [
            'timestamp_formatted', 'date', 'hour', 'day_of_week', 'side', 
            'position_before', 'position_after', 'position_change',
            'q10', 'q50', 'q90', 'spread',
            'prob_down', 'prob_neutral', 'prob_up',
            'tier', 'tier_confidence',
            'pnl', 'is_profitable', 'cumulative_pnl',
            'fee_cost', 'slippage_cost', 'total_cost',
            'portfolio_value', 'drawdown'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in trades_for_save.columns]
        remaining_columns = [col for col in trades_for_save.columns if col not in available_columns]
        final_columns = available_columns + remaining_columns
        
        trades_for_save = trades_for_save[final_columns]
        trades_for_save.to_csv(output_path / "trades.csv", index=False)
        
        # Create a daily summary for easier analysis
        if len(trades_for_save) > 0:
            daily_summary = trades_for_save.groupby('date').agg({
                'pnl': ['sum', 'count', 'mean'],
                'is_profitable': 'mean',
                'position_change': 'sum',
                'total_cost': 'sum',
                'portfolio_value': 'last',
                'drawdown': 'max'
            }).round(4)
            
            # Flatten column names
            daily_summary.columns = ['_'.join(map(str, col)).strip() for col in daily_summary.columns.values]
            daily_summary = daily_summary.reset_index()
            daily_summary.to_csv(output_path / "daily_summary.csv", index=False)
        
        # Save portfolio history with formatted timestamps
        portfolio_df = pd.DataFrame(self.portfolio_history)
        if 'timestamp' in portfolio_df.columns:
            try:
                timestamp_series = pd.to_datetime(portfolio_df['timestamp'])
                portfolio_df['timestamp_formatted'] = timestamp_series.dt.strftime('%Y-%m-%d %H:%M:%S')
                portfolio_df['date'] = timestamp_series.dt.date
                portfolio_df['hour'] = timestamp_series.dt.hour
                portfolio_df['day_of_week'] = timestamp_series.dt.day_name()
            except:
                portfolio_df['timestamp_formatted'] = portfolio_df['timestamp'].astype(str)
                portfolio_df['date'] = portfolio_df['timestamp'].astype(str)
                portfolio_df['hour'] = 0
                portfolio_df['day_of_week'] = 'Unknown'
            
            # Reorder columns
            portfolio_columns = [
                'timestamp_formatted', 'date', 'hour', 'day_of_week', 'signal',
                'capital', 'position', 'cumulative_pnl', 'drawdown'
            ]
            available_portfolio_columns = [col for col in portfolio_columns if col in portfolio_df.columns]
            remaining_portfolio_columns = [col for col in portfolio_df.columns if col not in available_portfolio_columns]
            final_portfolio_columns = available_portfolio_columns + remaining_portfolio_columns
            
            portfolio_df = portfolio_df[final_portfolio_columns]
        
        portfolio_df.to_csv(output_path / "portfolio_history.csv", index=False)
        
        # Save metrics
        import json
        with open(output_path / "metrics.json", 'w') as f:
            # Convert non-serializable objects
            metrics_serializable = {}
            for k, v in self.metrics.items():
                if isinstance(v, pd.DataFrame):
                    # Handle DataFrames with potentially complex MultiIndex columns
                    try:
                        # Flatten MultiIndex columns if present
                        if isinstance(v.columns, pd.MultiIndex):
                            v_flat = v.copy()
                            v_flat.columns = ['_'.join(map(str, col)).strip() for col in v.columns.values]
                            metrics_serializable[k] = v_flat.to_dict()
                        else:
                            metrics_serializable[k] = v.to_dict()
                    except:
                        # Fallback: convert to string representation
                        metrics_serializable[k] = str(v.to_dict())
                elif isinstance(v, pd.Series):
                    # Convert series with potentially complex index to simple dict
                    try:
                        # Handle different index types
                        if hasattr(v.index, 'categories'):  # Categorical index
                            metrics_serializable[k] = {str(idx): float(val) if pd.notna(val) else None 
                                                     for idx, val in v.items()}
                        else:
                            metrics_serializable[k] = {str(idx): float(val) if pd.notna(val) else None 
                                                     for idx, val in v.items()}
                    except:
                        # Fallback: convert to string representation
                        metrics_serializable[k] = str(v.to_dict())
                elif isinstance(v, (np.integer, np.floating)):
                    metrics_serializable[k] = float(v)
                elif isinstance(v, np.ndarray):
                    metrics_serializable[k] = v.tolist()
                else:
                    metrics_serializable[k] = v
            
            json.dump(metrics_serializable, f, indent=2, default=str)
        
        # Save report
        report = self.generate_report(trades_df)
        with open(output_path / "report.txt", 'w') as f:
            f.write(report)
        
        # Save plots
        self.plot_results(trades_df, save_path=output_path / "performance_plots.png")
        
        print(f"\nResults saved to {output_path}:")
        print(f"  - trades.csv (detailed trade records with formatted timestamps)")
        print(f"  - daily_summary.csv (daily aggregated performance)")
        print(f"  - portfolio_history.csv (portfolio value over time)")
        print(f"  - metrics.json (performance metrics)")
        print(f"  - report.txt (summary report)")
        print(f"  - performance_plots.png (visualization)")

# Example usage
def run_example_backtest():
    """Example of how to run the backtester with your data"""
    
    # Load your prediction data
    df = pd.read_csv("df_all_macro_analysis.csv", index_col=[0, 1])
    
    # Filter for BTCUSDT and validation period
    df_test = df.loc[("BTCUSDT", "2024-01-01"):("BTCUSDT", "2024-12-31")]
    
    # Configure backtester
    config = BacktestConfig(
        initial_capital=100000.0,
        position_limit=0.5,  # Max 50% position
        fee_rate=0.001,
        slippage_rate=0.0005,
        long_threshold=0.6,  # More conservative
        short_threshold=0.6,
        base_position_size=0.1
    )
    
    # Run backtest with proper frequency and horizon
    backtester = QuantileBacktester(config)
    results = backtester.run_backtest(df_test, data_frequency='hourly', prediction_horizon_hours=12.0)
    
    # Generate report
    print(backtester.generate_report(results))
    
    # Save results
    backtester.save_results(results)
    
    return backtester, results

if __name__ == "__main__":
    backtester, results = run_example_backtest()