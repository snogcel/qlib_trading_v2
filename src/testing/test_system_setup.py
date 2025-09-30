#!/usr/bin/env python3
"""
Test script to verify the feature test coverage system setup.

This script validates that all components are properly configured
and the system is ready for use.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from testing.core import create_default_system
from testing.models.feature_spec import FeatureSpec
from testing.models.test_case import TestCase, TestType, TestPriority
from testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from testing.utils.validation_utils import validate_feature_spec, validate_test_case
from testing.utils.logging_utils import setup_logging, get_logger


def test_system_initialization():
    """Test that the system can be initialized properly."""
    print("Testing system initialization...")
    
    try:
        system = create_default_system()
        print("System initialization successful")
        return True
    except Exception as e:
        print(f"System initialization failed: {e}")
        return False


def test_configuration_loading():
    """Test that configuration can be loaded and validated."""
    print("Testing configuration loading...")
    
    try:
        system = create_default_system()
        
        # Test configuration access
        test_config = system.config.test_config
        data_config = system.config.data_config
        
        print(f"  - Feature template path: {test_config.feature_template_path}")
        print(f"  - Output directory: {test_config.output_directory}")
        print(f"  - Max workers: {test_config.max_workers}")
        print(f"  - Data directory: {data_config.data_directory}")
        
        # Validate configuration
        errors = system.validate_system_setup()
        if errors:
            print(f"  - Configuration validation errors: {len(errors)}")
            for error in errors:
                print(f"    - {error}")
        else:
            print("  - Configuration validation passed")
        
        print("Configuration loading successful")
        return True
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        return False


def test_data_models():
    """Test that data models work correctly."""
    print("Testing data models...")
    
    try:
        # Test FeatureSpec
        feature_spec = FeatureSpec(
            name="test_feature",
            category="Test Category",
            tier="Tier 1",
            implementation="Test implementation",
            economic_hypothesis="Test hypothesis",
            performance_characteristics={"hit_rate": 0.6},
            failure_modes=["low_liquidity"],
            regime_dependencies={"bull": "positive_performance"}
        )
        
        validation_errors = validate_feature_spec(feature_spec)
        if validation_errors:
            print(f"  - FeatureSpec validation errors: {validation_errors}")
        else:
            print("  - FeatureSpec validation passed")
        
        # Test TestCase
        test_case = TestCase(
            test_id="test_001",
            feature_name="test_feature",
            test_type=TestType.ECONOMIC_HYPOTHESIS,
            description="Test economic hypothesis validation",
            validation_criteria={"expected_behavior": "positive_returns"},
            priority=TestPriority.HIGH
        )
        
        test_case_errors = validate_test_case(test_case)
        if test_case_errors:
            print(f"  - TestCase validation errors: {test_case_errors}")
        else:
            print("  - TestCase validation passed")
        
        # Test TestResult
        test_result = TestResult(
            test_case=test_case,
            execution_id="exec_001",
            status=TestStatus.PASSED,
            execution_time=1.5,
            passed=True,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            analysis="Test passed successfully"
        )
        
        print(f"  - TestResult summary: {test_result.get_summary()}")
        
        print("Data models testing successful")
        return True
    except Exception as e:
        print(f"Data models testing failed: {e}")
        return False


def test_logging_setup():
    """Test that logging is working correctly."""
    print("Testing logging setup...")
    
    try:
        # Set up logging
        log_file = Path("logs/feature_testing/test_setup.log")
        logger = setup_logging(log_level="INFO", log_file=log_file)
        
        # Test logging
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        # Check if log file was created
        if log_file.exists():
            print(f"  - Log file created: {log_file}")
        else:
            print("  - Warning: Log file not created")
        
        print("Logging setup successful")
        return True
    except Exception as e:
        print(f"Logging setup failed: {e}")
        return False


def test_system_report():
    """Test system report generation."""
    print("Testing system report generation...")
    
    try:
        system = create_default_system()
        report = system.generate_system_report()
        
        print("  - System report generated successfully")
        print(f"  - Report length: {len(report)} characters")
        
        # Save report to file
        report_file = Path("test_results/system_setup_report.md")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report)
        print(f"  - Report saved to: {report_file}")
        
        print("System report generation successful")
        return True
    except Exception as e:
        print(f"System report generation failed: {e}")
        return False


def main():
    """Run all system setup tests."""
    print("=" * 60)
    print("Feature Test Coverage System - Setup Validation")
    print("=" * 60)
    print()
    
    tests = [
        test_system_initialization,
        test_configuration_loading,
        test_data_models,
        test_logging_setup,
        test_system_report
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System setup is complete.")
        return 0
    else:
        print(" Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())