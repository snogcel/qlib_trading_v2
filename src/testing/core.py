"""
Core module for the Feature Test Coverage System.

This module provides the main entry point and orchestration for the
feature test coverage system components.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .config.config import get_config, FeatureTestConfig
from .models.feature_spec import FeatureSpec
from .models.test_case import TestCase
from .models.test_result import TestResult
from .utils.logging_utils import setup_logging, get_logger


class FeatureTestCoverageSystem:
    """
    Main orchestrator for the feature test coverage system.
    
    This class coordinates all components of the system to provide
    comprehensive test coverage for trading system features.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the feature test coverage system.
        
        Args:
            config_dir: Optional directory containing configuration files
        """
        self.config = get_config(config_dir)
        self.logger = get_logger("feature_test_system")
        
        # Initialize components (will be implemented in later tasks)
        self.parser = None
        self.generator = None
        self.executor = None
        self.validator = None
        self.reporter = None
        
        # Set up logging
        log_file = self.config.test_config.log_directory / "system.log"
        setup_logging(
            log_level="INFO",
            log_file=log_file
        )
        
        self.logger.info("Feature Test Coverage System initialized")
    
    def initialize_components(self):
        """
        Initialize all system components.
        
        This method will be implemented as components are created in later tasks.
        """
        self.logger.info("Initializing system components...")
        
        # TODO: Initialize components as they are implemented
        # self.parser = FeatureTemplateParser(self.config)
        # self.generator = TestCaseGenerator(self.config)
        # self.executor = TestExecutor(self.config)
        # self.validator = ValidationEngine(self.config)
        # self.reporter = CoverageReporter(self.config)
        
        self.logger.info("System components initialized")
    
    def validate_system_setup(self) -> List[str]:
        """
        Validate that the system is properly set up.
        
        Returns:
            List of validation errors (empty if all valid)
        """
        self.logger.info("Validating system setup...")
        
        errors = []
        
        # Validate configuration
        config_errors = self.config.validate_configuration()
        errors.extend(config_errors)
        
        # Check if feature template exists
        if not self.config.test_config.feature_template_path.exists():
            errors.append(f"Feature template not found: {self.config.test_config.feature_template_path}")
        
        # Check if output directories are writable
        try:
            self.config.test_config.output_directory.mkdir(parents=True, exist_ok=True)
            test_file = self.config.test_config.output_directory / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Output directory not writable: {e}")
        
        if errors:
            self.logger.error(f"System validation failed with {len(errors)} errors")
            for error in errors:
                self.logger.error(f"  - {error}")
        else:
            self.logger.info("System validation passed")
        
        return errors
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system configuration and status.
        
        Returns:
            Dictionary with system information
        """
        return {
            'version': '1.0.0',
            'config_directory': str(self.config.config_dir),
            'feature_template_path': str(self.config.test_config.feature_template_path),
            'output_directory': str(self.config.test_config.output_directory),
            'components_initialized': all([
                self.parser is not None,
                self.generator is not None,
                self.executor is not None,
                self.validator is not None,
                self.reporter is not None
            ]),
            'configuration': self.config.get_runtime_config()
        }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete feature test coverage suite.
        
        This method will orchestrate the full testing process once
        all components are implemented.
        
        Returns:
            Dictionary with test execution results
        """
        self.logger.info("Starting full test suite execution...")
        
        # TODO: Implement full test suite execution
        # This will be implemented as components become available
        
        results = {
            'status': 'not_implemented',
            'message': 'Full test suite execution will be implemented in later tasks',
            'timestamp': None,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'execution_time': 0.0
        }
        
        self.logger.info("Full test suite execution placeholder completed")
        return results
    
    def generate_system_report(self) -> str:
        """
        Generate a comprehensive system status report.
        
        Returns:
            Formatted system report string
        """
        info = self.get_system_info()
        validation_errors = self.validate_system_setup()
        
        report_lines = [
            "# Feature Test Coverage System Report",
            "",
            f"**Version:** {info['version']}",
            f"**Configuration Directory:** {info['config_directory']}",
            f"**Feature Template:** {info['feature_template_path']}",
            f"**Output Directory:** {info['output_directory']}",
            f"**Components Initialized:** {info['components_initialized']}",
            "",
            "## Configuration Summary",
            ""
        ]
        
        # Add configuration details
        config = info['configuration']
        for section, settings in config.items():
            report_lines.append(f"### {section.title()}")
            for key, value in settings.items():
                report_lines.append(f"- **{key}:** {value}")
            report_lines.append("")
        
        # Add validation results
        report_lines.append("## System Validation")
        if validation_errors:
            report_lines.append("**Status:** FAILED")
            report_lines.append("")
            report_lines.append("**Errors:**")
            for error in validation_errors:
                report_lines.append(f"- {error}")
        else:
            report_lines.append("**Status:** PASSED")
            report_lines.append("")
            report_lines.append("All system validation checks passed successfully.")
        
        return "\n".join(report_lines)


def create_default_system(config_dir: Optional[Path] = None) -> FeatureTestCoverageSystem:
    """
    Create a feature test coverage system with default configuration.
    
    Args:
        config_dir: Optional directory for configuration files
        
    Returns:
        Initialized FeatureTestCoverageSystem instance
    """
    system = FeatureTestCoverageSystem(config_dir)
    
    # Create default configuration files if they don't exist
    system.config.create_default_configs()
    
    return system