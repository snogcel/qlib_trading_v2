"""
Data utilities for the feature test coverage system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from ..config.data_config import DataConfig


def load_test_data(
    assets: List[str],
    start_date: str,
    end_date: str,
    data_config: DataConfig,
    timeframe: str = "daily"
) -> Dict[str, pd.DataFrame]:
    """
    Load test data for specified assets and date range.
    
    Args:
        assets: List of asset symbols to load
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        data_config: Data configuration object
        timeframe: Data timeframe (daily, hourly, etc.)
        
    Returns:
        Dictionary mapping asset symbols to DataFrames
    """
    data = {}
    
    for asset in assets:
        try:
            # Try to load from primary data source
            asset_data = _load_asset_data(asset, start_date, end_date, data_config, timeframe)
            
            if asset_data is not None and not asset_data.empty:
                # Validate data quality
                quality_issues = _validate_data_quality(asset_data, data_config)
                if not quality_issues:
                    data[asset] = asset_data
                else:
                    print(f"Warning: Data quality issues for {asset}: {quality_issues}")
                    # Still include data but with warning
                    data[asset] = asset_data
            else:
                print(f"Warning: No data available for {asset}")
                
        except Exception as e:
            print(f"Error loading data for {asset}: {str(e)}")
            continue
    
    return data


def _load_asset_data(
    asset: str,
    start_date: str,
    end_date: str,
    data_config: DataConfig,
    timeframe: str
) -> Optional[pd.DataFrame]:
    """
    Load data for a single asset from available sources.
    
    Args:
        asset: Asset symbol
        start_date: Start date string
        end_date: End date string
        data_config: Data configuration
        timeframe: Data timeframe
        
    Returns:
        DataFrame with asset data or None if not found
    """
    # Try primary data source first
    primary_path = data_config.crypto_data_path / f"{asset}.csv"
    if primary_path.exists():
        try:
            df = pd.read_csv(primary_path)
            df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
            df = df.set_index('date')
            
            # Filter by date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            return df.loc[mask]
            
        except Exception as e:
            print(f"Error reading primary data for {asset}: {str(e)}")
    
    # Try backup sources
    for backup_source in data_config.backup_data_sources:
        backup_path = Path(backup_source) / f"{asset}.csv"
        if backup_path.exists():
            try:
                df = pd.read_csv(backup_path)
                df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
                df = df.set_index('date')
                
                # Filter by date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                return df.loc[mask]
                
            except Exception as e:
                print(f"Error reading backup data for {asset}: {str(e)}")
                continue
    
    return None


def _validate_data_quality(data: pd.DataFrame, data_config: DataConfig) -> List[str]:
    """
    Validate data quality against configuration requirements.
    
    Args:
        data: DataFrame to validate
        data_config: Data configuration with quality requirements
        
    Returns:
        List of quality issues found
    """
    issues = []
    
    # Check for required fields
    missing_fields = []
    for field in data_config.required_ohlcv_fields:
        if field not in data.columns:
            missing_fields.append(field)
    
    if missing_fields:
        issues.append(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check for missing data
    total_rows = len(data)
    if total_rows == 0:
        issues.append("No data available")
        return issues
    
    missing_rows = data.isnull().any(axis=1).sum()
    missing_pct = missing_rows / total_rows
    
    if missing_pct > data_config.max_missing_data_pct:
        issues.append(f"Missing data percentage ({missing_pct:.1%}) exceeds threshold")
    
    # Check volume if available
    if 'volume' in data.columns:
        avg_volume = data['volume'].mean()
        if avg_volume < data_config.min_volume_threshold:
            issues.append(f"Average volume ({avg_volume:.0f}) below threshold")
    
    # Check for data range
    data_days = (data.index.max() - data.index.min()).days
    if data_days < data_config.min_history_days:
        issues.append(f"Data range ({data_days} days) below minimum requirement")
    
    return issues


def simulate_market_regime(
    base_data: pd.DataFrame,
    regime_type: str,
    duration_days: int = 30,
    regime_config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Simulate specific market regime conditions on base data.
    
    Args:
        base_data: Base DataFrame to modify
        regime_type: Type of regime to simulate
        duration_days: Duration of regime in days
        regime_config: Optional regime-specific configuration
        
    Returns:
        Modified DataFrame with simulated regime conditions
    """
    if base_data.empty:
        return base_data
    
    # Make a copy to avoid modifying original data
    data = base_data.copy()
    
    # Ensure we have required columns
    if 'close' not in data.columns:
        raise ValueError("Data must contain 'close' column for regime simulation")
    
    # Limit to specified duration
    if len(data) > duration_days:
        data = data.tail(duration_days)
    
    if regime_type == "bull":
        data = _simulate_bull_market(data, regime_config)
    elif regime_type == "bear":
        data = _simulate_bear_market(data, regime_config)
    elif regime_type == "sideways":
        data = _simulate_sideways_market(data, regime_config)
    elif regime_type == "high_volatility":
        data = _simulate_high_volatility(data, regime_config)
    elif regime_type == "low_volatility":
        data = _simulate_low_volatility(data, regime_config)
    else:
        raise ValueError(f"Unknown regime type: {regime_type}")
    
    return data


def _simulate_bull_market(data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Simulate bull market conditions."""
    daily_return = 0.02 if not config else config.get('daily_return', 0.02)
    volatility = 0.015 if not config else config.get('volatility', 0.015)
    
    # Generate upward trending returns with some volatility
    returns = np.random.normal(daily_return, volatility, len(data))
    
    # Apply returns to close prices
    data['close'] = data['close'].iloc[0] * (1 + returns).cumprod()
    
    # Adjust OHLC if available
    if all(col in data.columns for col in ['open', 'high', 'low']):
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(data)))
    
    return data


def _simulate_bear_market(data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Simulate bear market conditions."""
    daily_return = -0.015 if not config else config.get('daily_return', -0.015)
    volatility = 0.02 if not config else config.get('volatility', 0.02)
    
    # Generate downward trending returns with higher volatility
    returns = np.random.normal(daily_return, volatility, len(data))
    
    # Apply returns to close prices
    data['close'] = data['close'].iloc[0] * (1 + returns).cumprod()
    
    # Adjust OHLC if available
    if all(col in data.columns for col in ['open', 'high', 'low']):
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.015, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.015, len(data)))
    
    return data


def _simulate_sideways_market(data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Simulate sideways market conditions."""
    volatility = 0.01 if not config else config.get('volatility', 0.01)
    
    # Generate mean-reverting returns
    returns = np.random.normal(0, volatility, len(data))
    
    # Add mean reversion
    for i in range(1, len(returns)):
        returns[i] -= 0.1 * returns[i-1]  # Mean reversion factor
    
    # Apply returns to close prices
    data['close'] = data['close'].iloc[0] * (1 + returns).cumprod()
    
    # Adjust OHLC if available
    if all(col in data.columns for col in ['open', 'high', 'low']):
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(data)))
    
    return data


def _simulate_high_volatility(data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Simulate high volatility conditions."""
    volatility_multiplier = 3.0 if not config else config.get('volatility_multiplier', 3.0)
    
    # Calculate current returns
    returns = data['close'].pct_change().fillna(0)
    
    # Amplify volatility while preserving trend
    amplified_returns = returns * volatility_multiplier
    
    # Apply amplified returns
    data['close'] = data['close'].iloc[0] * (1 + amplified_returns).cumprod()
    
    # Adjust OHLC if available
    if all(col in data.columns for col in ['open', 'high', 'low']):
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.03, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.03, len(data)))
    
    return data


def _simulate_low_volatility(data: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Simulate low volatility conditions."""
    volatility_reduction = 0.3 if not config else config.get('volatility_reduction', 0.3)
    
    # Calculate current returns
    returns = data['close'].pct_change().fillna(0)
    
    # Reduce volatility while preserving trend
    smoothed_returns = returns * volatility_reduction
    
    # Apply smoothed returns
    data['close'] = data['close'].iloc[0] * (1 + smoothed_returns).cumprod()
    
    # Adjust OHLC if available
    if all(col in data.columns for col in ['open', 'high', 'low']):
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.002, len(data)))
        data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.002, len(data)))
    
    return data


def create_synthetic_failure_data(
    base_data: pd.DataFrame,
    failure_type: str,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Create synthetic data for testing failure modes.
    
    Args:
        base_data: Base DataFrame to modify
        failure_type: Type of failure to simulate
        config: Optional failure-specific configuration
        
    Returns:
        Modified DataFrame with failure conditions
    """
    if base_data.empty:
        return base_data
    
    data = base_data.copy()
    
    if failure_type == "flat_market":
        # Create completely flat market
        data['close'] = data['close'].iloc[0]
        if 'open' in data.columns:
            data['open'] = data['close']
        if 'high' in data.columns:
            data['high'] = data['close']
        if 'low' in data.columns:
            data['low'] = data['close']
    
    elif failure_type == "data_gaps":
        # Introduce random data gaps
        gap_probability = 0.1 if not config else config.get('gap_probability', 0.1)
        gap_mask = np.random.random(len(data)) < gap_probability
        data.loc[gap_mask, ['open', 'high', 'low', 'close']] = np.nan
    
    elif failure_type == "extreme_outliers":
        # Introduce extreme price outliers
        outlier_probability = 0.05 if not config else config.get('outlier_probability', 0.05)
        outlier_magnitude = 10.0 if not config else config.get('outlier_magnitude', 10.0)
        
        outlier_mask = np.random.random(len(data)) < outlier_probability
        outlier_multipliers = np.random.choice([-outlier_magnitude, outlier_magnitude], sum(outlier_mask))
        
        data.loc[outlier_mask, 'close'] *= (1 + outlier_multipliers)
    
    elif failure_type == "zero_volume":
        # Set volume to zero
        if 'volume' in data.columns:
            data['volume'] = 0
    
    else:
        raise ValueError(f"Unknown failure type: {failure_type}")
    
    return data


def calculate_regime_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate statistics to characterize market regime.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with regime statistics
    """
    if data.empty or 'close' not in data.columns:
        return {}
    
    returns = data['close'].pct_change().dropna()
    
    stats = {
        'total_return': (data['close'].iloc[-1] / data['close'].iloc[0]) - 1,
        'volatility': returns.std() * np.sqrt(252),  # Annualized
        'mean_return': returns.mean() * 252,  # Annualized
        'max_drawdown': _calculate_max_drawdown(data['close']),
        'positive_days_pct': (returns > 0).mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }
    
    return stats


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min()