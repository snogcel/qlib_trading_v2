"""
Test configuration management for the feature test coverage system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


@dataclass
class TestConfig:
    """
    Configuration settings for test execution and validation.
    
    This class manages all configurable parameters for the test coverage system
    including execution settings, validation thresholds, and output preferences.
    """
    
    # File paths
    feature_template_path: Path = Path("docs/FEATURE_KNOWLEDGE_TEMPLATE.md")
    output_directory: Path = Path("test_results/feature_coverage")
    log_directory: Path = Path("logs/feature_testing")
    
    # Test execution settings
    parallel_execution: bool = True
    max_workers: int = 4
    test_timeout: float = 30.0  # seconds
    retry_failed_tests: bool = True
    max_retries: int = 2
    
    # Test filtering
    include_test_types: List[str] = field(default_factory=lambda: [
        "economic_hypothesis",
        "performance", 
        "implementation",
        "failure_mode",
        "regime_dependency"
    ])
    exclude_features: List[str] = field(default_factory=list)
    include_only_features: List[str] = field(default_factory=list)
    
    # Validation thresholds
    confidence_threshold: float = 0.6
    performance_tolerance: float = 0.05
    statistical_significance: float = 0.05
    
    # Regime testing settings
    regime_types: List[str] = field(default_factory=lambda: [
        "bull", "bear", "sideways", "high_volatility", "low_volatility"
    ])
    regime_test_duration: int = 30  # days of data per regime test
    
    # Data requirements
    min_data_points: int = 100
    data_quality_threshold: float = 0.95
    required_data_fields: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume"
    ])
    
    # Reporting settings
    generate_html_reports: bool = True
    generate_json_exports: bool = True
    include_charts: bool = True
    detailed_failure_analysis: bool = True
    
    # Feature prioritization
    tier_1_features: List[str] = field(default_factory=lambda: [
        "Q50", "vol_risk", "kelly_sizing", "regime_multiplier"
    ])
    tier_2_features: List[str] = field(default_factory=lambda: [
        "Q10", "Q90", "spread", "vol_raw", "fg_index", "btc_dom"
    ])
    
    # Alert settings
    enable_alerts: bool = True
    alert_on_critical_failures: bool = True
    alert_threshold: float = 0.8  # Alert if success rate drops below this
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Validate thresholds
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.alert_threshold <= 1:
            raise ValueError("alert_threshold must be between 0 and 1")
        
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
    
    def is_feature_included(self, feature_name: str) -> bool:
        """
        Check if a feature should be included in testing.
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature should be tested, False otherwise
        """
        # Check exclusion list first
        if feature_name in self.exclude_features:
            return False
        
        # If include_only_features is specified, only include those
        if self.include_only_features:
            return feature_name in self.include_only_features
        
        # Otherwise include all features not explicitly excluded
        return True
    
    def get_feature_priority(self, feature_name: str) -> str:
        """
        Get the priority level for a feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Priority level string ("tier_1", "tier_2", or "tier_3")
        """
        if feature_name in self.tier_1_features:
            return "tier_1"
        elif feature_name in self.tier_2_features:
            return "tier_2"
        else:
            return "tier_3"
    
    def should_generate_alerts(self, success_rate: float) -> bool:
        """
        Check if alerts should be generated based on success rate.
        
        Args:
            success_rate: Current test success rate (0-1)
            
        Returns:
            True if alerts should be generated
        """
        return (
            self.enable_alerts and 
            success_rate < self.alert_threshold
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'TestConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            TestConfig instance loaded from file
        """
        if not config_path.exists():
            # Return default configuration if file doesn't exist
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Convert path strings to Path objects
        if 'feature_template_path' in config_data:
            config_data['feature_template_path'] = Path(config_data['feature_template_path'])
        if 'output_directory' in config_data:
            config_data['output_directory'] = Path(config_data['output_directory'])
        if 'log_directory' in config_data:
            config_data['log_directory'] = Path(config_data['log_directory'])
        
        return cls(**config_data)
    
    def to_file(self, config_path: Path) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path where configuration should be saved
        """
        # Convert to dictionary and handle Path objects
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_execution_settings(self) -> Dict[str, Any]:
        """
        Get settings relevant to test execution.
        
        Returns:
            Dictionary with execution configuration
        """
        return {
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'test_timeout': self.test_timeout,
            'retry_failed_tests': self.retry_failed_tests,
            'max_retries': self.max_retries
        }
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """
        Get settings relevant to result validation.
        
        Returns:
            Dictionary with validation configuration
        """
        return {
            'confidence_threshold': self.confidence_threshold,
            'performance_tolerance': self.performance_tolerance,
            'statistical_significance': self.statistical_significance,
            'data_quality_threshold': self.data_quality_threshold
        }