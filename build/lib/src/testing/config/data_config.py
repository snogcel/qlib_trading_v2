"""
Data configuration management for the feature test coverage system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta


@dataclass
class DataConfig:
    """
    Configuration settings for test data sources and requirements.
    
    This class manages data-related configuration including sources,
    quality requirements, and regime-specific data settings.
    """
    
    # Data source settings
    data_directory: Path = Path("data")
    crypto_data_path: Path = Path("qlib_data/CRYPTO")
    backup_data_sources: List[str] = field(default_factory=lambda: [
        "data/processed",
        "data/raw"
    ])
    
    # Historical data requirements
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    min_history_days: int = 365
    data_frequency: str = "daily"  # daily, hourly, minute
    
    # Asset coverage
    primary_assets: List[str] = field(default_factory=lambda: [
        "BTC", "ETH"
    ])
    secondary_assets: List[str] = field(default_factory=lambda: [
        "ADA", "DOT", "LINK", "UNI", "AAVE", "SOL"
    ])
    test_assets: List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "ADA"  # Subset for testing
    ])
    
    # Data quality requirements
    max_missing_data_pct: float = 0.05  # 5% max missing data
    min_volume_threshold: float = 1000.0  # Minimum daily volume
    outlier_detection_threshold: float = 5.0  # Standard deviations
    
    # Required data fields
    required_ohlcv_fields: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    optional_fields: List[str] = field(default_factory=lambda: [
        "market_cap", "circulating_supply"
    ])
    
    # Regime-specific data settings
    regime_data_requirements: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "bull": {
            "min_duration_days": 30,
            "min_return_threshold": 0.2,  # 20% minimum return
            "example_periods": ["2020-10-01:2021-04-01", "2023-10-01:2024-03-01"]
        },
        "bear": {
            "min_duration_days": 30,
            "max_return_threshold": -0.2,  # -20% maximum return
            "example_periods": ["2022-01-01:2022-06-01", "2022-06-01:2022-12-01"]
        },
        "sideways": {
            "min_duration_days": 60,
            "max_volatility": 0.3,  # 30% max volatility
            "example_periods": ["2019-04-01:2019-09-01", "2023-04-01:2023-09-01"]
        },
        "high_volatility": {
            "min_volatility": 0.5,  # 50% minimum volatility
            "example_periods": ["2020-03-01:2020-05-01", "2022-05-01:2022-07-01"]
        },
        "low_volatility": {
            "max_volatility": 0.2,  # 20% maximum volatility
            "example_periods": ["2019-07-01:2019-10-01", "2023-07-01:2023-10-01"]
        }
    })
    
    # Synthetic data settings for failure mode testing
    synthetic_data_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "flat_market": {
            "price_change_limit": 0.001,  # 0.1% max daily change
            "duration_days": 30
        },
        "synthetic_volatility": {
            "artificial_vol_multiplier": 3.0,
            "duration_days": 14
        },
        "data_gaps": {
            "gap_sizes": [1, 3, 7],  # days
            "gap_frequency": 0.1  # 10% of data points
        },
        "extreme_volatility": {
            "vol_multiplier": 10.0,
            "duration_days": 7
        }
    })
    
    # Caching settings
    enable_data_caching: bool = True
    cache_directory: Path = Path("data/cache/feature_testing")
    cache_expiry_hours: int = 24
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories exist
        self.data_directory.mkdir(parents=True, exist_ok=True)
        if self.enable_data_caching:
            self.cache_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate date format
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")
        
        # Validate thresholds
        if not 0 <= self.max_missing_data_pct <= 1:
            raise ValueError("max_missing_data_pct must be between 0 and 1")
    
    def get_date_range(self) -> tuple[datetime, datetime]:
        """
        Get the configured date range as datetime objects.
        
        Returns:
            Tuple of (start_date, end_date) as datetime objects
        """
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return start, end
    
    def get_assets_for_testing(self, test_scope: str = "standard") -> List[str]:
        """
        Get list of assets to use for testing based on scope.
        
        Args:
            test_scope: Testing scope ("minimal", "standard", "comprehensive")
            
        Returns:
            List of asset symbols to test
        """
        if test_scope == "minimal":
            return ["BTC"]
        elif test_scope == "standard":
            return self.test_assets
        elif test_scope == "comprehensive":
            return self.primary_assets + self.secondary_assets
        else:
            return self.test_assets
    
    def get_regime_periods(self, regime_type: str) -> List[tuple[datetime, datetime]]:
        """
        Get historical periods for a specific market regime.
        
        Args:
            regime_type: Type of market regime
            
        Returns:
            List of (start_date, end_date) tuples for the regime
        """
        if regime_type not in self.regime_data_requirements:
            return []
        
        regime_config = self.regime_data_requirements[regime_type]
        periods = []
        
        for period_str in regime_config.get("example_periods", []):
            start_str, end_str = period_str.split(":")
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            periods.append((start_date, end_date))
        
        return periods
    
    def validate_data_quality(self, data_stats: Dict[str, Any]) -> List[str]:
        """
        Validate data quality against configured requirements.
        
        Args:
            data_stats: Dictionary with data quality statistics
            
        Returns:
            List of quality issues found (empty if all good)
        """
        issues = []
        
        # Check missing data percentage
        missing_pct = data_stats.get("missing_data_pct", 0)
        if missing_pct > self.max_missing_data_pct:
            issues.append(f"Missing data ({missing_pct:.1%}) exceeds threshold ({self.max_missing_data_pct:.1%})")
        
        # Check required fields
        available_fields = data_stats.get("available_fields", [])
        for field in self.required_ohlcv_fields:
            if field not in available_fields:
                issues.append(f"Required field '{field}' not available")
        
        # Check data volume
        avg_volume = data_stats.get("average_volume", 0)
        if avg_volume < self.min_volume_threshold:
            issues.append(f"Average volume ({avg_volume}) below threshold ({self.min_volume_threshold})")
        
        # Check data range
        data_days = data_stats.get("total_days", 0)
        if data_days < self.min_history_days:
            issues.append(f"Data history ({data_days} days) below minimum ({self.min_history_days} days)")
        
        return issues
    
    def get_synthetic_data_params(self, scenario: str) -> Dict[str, Any]:
        """
        Get parameters for synthetic data generation.
        
        Args:
            scenario: Synthetic data scenario name
            
        Returns:
            Dictionary with generation parameters
        """
        return self.synthetic_data_config.get(scenario, {})
    
    def is_cache_valid(self, cache_file: Path) -> bool:
        """
        Check if a cached data file is still valid.
        
        Args:
            cache_file: Path to cached data file
            
        Returns:
            True if cache is valid, False if expired or missing
        """
        if not self.enable_data_caching or not cache_file.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        max_age = timedelta(hours=self.cache_expiry_hours)
        
        return file_age < max_age
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """
        Get configuration for data loader components.
        
        Returns:
            Dictionary with data loader configuration
        """
        return {
            'data_directory': str(self.data_directory),
            'crypto_data_path': str(self.crypto_data_path),
            'backup_sources': self.backup_data_sources,
            'required_fields': self.required_ohlcv_fields,
            'quality_threshold': 1.0 - self.max_missing_data_pct,
            'cache_enabled': self.enable_data_caching,
            'cache_directory': str(self.cache_directory)
        }