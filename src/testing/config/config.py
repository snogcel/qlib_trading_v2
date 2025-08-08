"""
Main configuration module for the feature test coverage system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List

from .test_config import TestConfig
from .data_config import DataConfig


class FeatureTestConfig:
    """
    Main configuration manager for the feature test coverage system.
    
    This class provides a unified interface to all configuration settings
    and handles loading/saving of configuration files.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("config/feature_testing")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self.test_config = TestConfig.from_file(self.config_dir / "test_config.json")
        self.data_config = DataConfig()
        
        # Load data config if file exists
        data_config_path = self.config_dir / "data_config.json"
        if data_config_path.exists():
            import json
            with open(data_config_path, 'r') as f:
                data_config_dict = json.load(f)
            
            # Convert path strings back to Path objects
            for key in ['data_directory', 'crypto_data_path', 'cache_directory']:
                if key in data_config_dict:
                    data_config_dict[key] = Path(data_config_dict[key])
            
            self.data_config = DataConfig(**data_config_dict)
    
    def save_configs(self) -> None:
        """Save all configurations to files."""
        # Save test config
        self.test_config.to_file(self.config_dir / "test_config.json")
        
        # Save data config
        import json
        data_config_dict = {}
        for key, value in self.data_config.__dict__.items():
            if isinstance(value, Path):
                data_config_dict[key] = str(value)
            else:
                data_config_dict[key] = value
        
        with open(self.config_dir / "data_config.json", 'w') as f:
            json.dump(data_config_dict, f, indent=2)
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all configuration settings as a single dictionary.
        
        Returns:
            Dictionary with all configuration settings
        """
        return {
            'test_settings': self.test_config.__dict__,
            'data_settings': self.data_config.__dict__,
            'system_info': {
                'config_directory': str(self.config_dir),
                'version': '1.0.0'
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate all configuration settings.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        # Validate paths exist
        if not self.test_config.feature_template_path.exists():
            errors.append(f"Feature template not found: {self.test_config.feature_template_path}")
        
        if not self.data_config.data_directory.exists():
            errors.append(f"Data directory not found: {self.data_config.data_directory}")
        
        # Validate date ranges
        start_date, end_date = self.data_config.get_date_range()
        if start_date >= end_date:
            errors.append("Data start_date must be before end_date")
        
        # Validate test settings
        if self.test_config.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if not 0 <= self.test_config.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        return errors
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Get configuration settings needed at runtime.
        
        Returns:
            Dictionary with runtime configuration
        """
        return {
            'execution': self.test_config.get_execution_settings(),
            'validation': self.test_config.get_validation_settings(),
            'data_loader': self.data_config.get_data_loader_config(),
            'paths': {
                'template': str(self.test_config.feature_template_path),
                'output': str(self.test_config.output_directory),
                'logs': str(self.test_config.log_directory)
            }
        }
    
    def update_test_config(self, **kwargs) -> None:
        """
        Update test configuration settings.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.test_config, key):
                setattr(self.test_config, key, value)
            else:
                raise ValueError(f"Unknown test configuration parameter: {key}")
    
    def update_data_config(self, **kwargs) -> None:
        """
        Update data configuration settings.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.data_config, key):
                setattr(self.data_config, key, value)
            else:
                raise ValueError(f"Unknown data configuration parameter: {key}")
    
    def create_default_configs(self) -> None:
        """Create default configuration files if they don't exist."""
        # Create default test config
        test_config_path = self.config_dir / "test_config.json"
        if not test_config_path.exists():
            default_test_config = TestConfig()
            default_test_config.to_file(test_config_path)
        
        # Create default data config
        data_config_path = self.config_dir / "data_config.json"
        if not data_config_path.exists():
            import json
            default_data_config = DataConfig()
            
            # Convert to serializable format
            config_dict = {}
            for key, value in default_data_config.__dict__.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
            
            with open(data_config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)


# Global configuration instance
_global_config: Optional[FeatureTestConfig] = None


def get_config(config_dir: Optional[Path] = None) -> FeatureTestConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        FeatureTestConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = FeatureTestConfig(config_dir)
    
    return _global_config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _global_config
    _global_config = None