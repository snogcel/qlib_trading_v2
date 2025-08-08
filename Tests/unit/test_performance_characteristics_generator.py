"""
Unit tests for the performance characteristics test generator.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from src.testing.generators.performance_characteristics_generator import (
    PerformanceCharacteristicsGenerator,
    PerformanceThresholds
)
from src.testing.models.feature_spec import FeatureSpec
from src.testing.models.test_case import TestType, TestPriority


class TestPerformanceCharacteristicsGenerator:
    """Test cases for the PerformanceCharacteristicsGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a performance characteristics generator for testing."""
        return PerformanceCharacteristicsGenerator()
    
    @pytest.fixture
    def signal_feature(self):
        """Create a signal feature specification for testing."""
        return FeatureSpec(
            name="Q50",
            category="Core Signal Features",
            tier="Tier 1",
            implementation="src/features/signal_features.py",
            economic_hypothesis="Provides directional bias in trending markets",
            performance_characteristics={
                "hit_rate": 0.55,
                "sharpe_ratio": 1.2
            },
            regime_dependencies={
                "bull_market": "strong_performance",
                "bear_market": "moderate_performance"
            },
            empirical_ranges={
                "signal_range": (-3.0, 3.0)
            }
        )
    
    @pytest.fixture
    def volatility_feature(self):
        """Create a volatility feature specification for testing."""
        return FeatureSpec(
            name="vol_risk",
            category="Risk & Volatility Features",
            tier="Tier 1",
            implementation="src/features/volatility_features.py",
            formula="Std(Log(close/Ref(close,1)), 6)^2",
            economic_hypothesis="Captures variance-based risk measures",
            empirical_ranges={
                "variance_range": (0.0, 1.0)
            }
        )
    
    @pytest.fixture
    def position_sizing_feature(self):
        """Create a position sizing feature specification for testing."""
        return FeatureSpec(
            name="kelly_sizing",
            category="Position Sizing Features",
            tier="Tier 1",
            implementation="src/features/position_sizing.py",
            economic_hypothesis="Optimal position sizing based on Kelly criterion",
            performance_characteristics={
                "max_drawdown": 0.15,
                "risk_adjusted_return": 0.12
            }
        )
    
    def test_initialization_with_default_config(self, generator):
        """Test generator initialization with default configuration."""
        assert generator.config == {}
        assert isinstance(generator.thresholds, PerformanceThresholds)
        assert generator.thresholds.hit_rate_min == 0.45
        assert generator.thresholds.sharpe_ratio_min == 0.5
        assert generator.thresholds.max_drawdown_threshold == 0.15
    
    def test_initialization_with_custom_config(self):
        """Test generator initialization with custom configuration."""
        config = {
            'hit_rate_min': 0.5,
            'sharpe_ratio_min': 0.8,
            'max_drawdown_threshold': 0.1
        }
        generator = PerformanceCharacteristicsGenerator(config)
        
        assert generator.config == config
        assert generator.thresholds.hit_rate_min == 0.5
        assert generator.thresholds.sharpe_ratio_min == 0.8
        assert generator.thresholds.max_drawdown_threshold == 0.1
    
    def test_generate_performance_characteristics_tests_signal_feature(self, generator, signal_feature):
        """Test performance test generation for signal features."""
        tests = generator.generate_performance_characteristics_tests(signal_feature)
        
        # Should generate multiple types of tests for signal features
        assert len(tests) > 0
        
        # Check for hit rate tests
        hit_rate_tests = [t for t in tests if 'hit rate' in t.description.lower()]
        assert len(hit_rate_tests) > 0
        
        # Check for Sharpe ratio tests
        sharpe_tests = [t for t in tests if 'sharpe ratio' in t.description.lower()]
        assert len(sharpe_tests) > 0
        
        # Check for regime performance tests
        regime_tests = [t for t in tests if t.regime_context is not None]
        assert len(regime_tests) > 0
        
        # Verify test properties
        for test in tests:
            assert test.feature_name == "Q50"
            assert test.test_type == TestType.PERFORMANCE
            assert test.priority in [TestPriority.CRITICAL, TestPriority.HIGH, TestPriority.MEDIUM]
            assert test.estimated_duration > 0
    
    def test_generate_hit_rate_tests(self, generator, signal_feature):
        """Test hit rate test generation."""
        tests = generator._generate_hit_rate_tests(signal_feature)
        
        assert len(tests) > 0
        
        # Check basic hit rate test
        basic_test = tests[0]
        assert basic_test.feature_name == "Q50"
        assert basic_test.test_type == TestType.PERFORMANCE
        assert 'hit rate' in basic_test.description.lower()
        assert basic_test.priority == TestPriority.HIGH
        
        # Check validation criteria
        criteria = basic_test.validation_criteria
        assert criteria['metric_type'] == 'hit_rate'
        assert criteria['min_threshold'] == generator.thresholds.hit_rate_min
        assert criteria['max_threshold'] == generator.thresholds.hit_rate_max
        
        # Check regime-specific tests
        regime_tests = [t for t in tests if t.regime_context is not None]
        assert len(regime_tests) == len(signal_feature.regime_dependencies)
    
    def test_generate_sharpe_ratio_tests(self, generator, signal_feature):
        """Test Sharpe ratio test generation."""
        tests = generator._generate_sharpe_ratio_tests(signal_feature)
        
        assert len(tests) >= 2  # Basic + rolling Sharpe tests
        
        # Check basic Sharpe ratio test
        basic_test = tests[0]
        assert basic_test.feature_name == "Q50"
        assert 'sharpe ratio' in basic_test.description.lower()
        assert basic_test.priority == TestPriority.HIGH
        
        # Check validation criteria
        criteria = basic_test.validation_criteria
        assert criteria['metric_type'] == 'sharpe_ratio'
        assert criteria['min_threshold'] == generator.thresholds.sharpe_ratio_min
        assert criteria['risk_free_rate'] == 0.02
        
        # Check rolling Sharpe test
        rolling_test = tests[1]
        assert 'rolling' in rolling_test.description.lower()
        assert rolling_test.validation_criteria['metric_type'] == 'rolling_sharpe_stability'
    
    def test_generate_empirical_range_tests_volatility_feature(self, generator, volatility_feature):
        """Test empirical range test generation for volatility features."""
        tests = generator._generate_empirical_range_tests(volatility_feature)
        
        assert len(tests) > 0
        
        # Check range validation test
        range_test = tests[0]
        assert range_test.feature_name == "vol_risk"
        assert 'empirical range' in range_test.description.lower()
        assert range_test.priority == TestPriority.HIGH
        
        # Check validation criteria
        criteria = range_test.validation_criteria
        assert criteria['metric_type'] == 'empirical_range'
        assert criteria['expected_min'] == 0.0
        assert criteria['expected_max'] == 1.0
        assert criteria['outlier_threshold'] == 0.05
    
    def test_generate_risk_adjusted_return_tests(self, generator, position_sizing_feature):
        """Test risk-adjusted return test generation."""
        tests = generator._generate_risk_adjusted_return_tests(position_sizing_feature)
        
        assert len(tests) > 0
        
        # Check basic risk-adjusted return test
        basic_test = tests[0]
        assert basic_test.feature_name == "kelly_sizing"
        assert 'risk-adjusted returns' in basic_test.description.lower()
        assert basic_test.priority == TestPriority.HIGH
        
        # Check Kelly criterion test for Kelly sizing features
        kelly_tests = [t for t in tests if 'kelly criterion' in t.description.lower()]
        assert len(kelly_tests) > 0
        
        kelly_test = kelly_tests[0]
        assert kelly_test.priority == TestPriority.CRITICAL
        assert kelly_test.validation_criteria['metric_type'] == 'kelly_criterion_validation'
    
    def test_generate_drawdown_control_tests(self, generator, position_sizing_feature):
        """Test drawdown control test generation."""
        tests = generator._generate_drawdown_control_tests(position_sizing_feature)
        
        assert len(tests) >= 2  # Max drawdown + frequency tests
        
        # Check maximum drawdown test
        max_dd_test = tests[0]
        assert max_dd_test.feature_name == "kelly_sizing"
        assert 'maximum drawdown' in max_dd_test.description.lower()
        assert max_dd_test.priority == TestPriority.HIGH
        
        criteria = max_dd_test.validation_criteria
        assert criteria['metric_type'] == 'max_drawdown_validation'
        assert criteria['max_drawdown_threshold'] == generator.thresholds.max_drawdown_threshold
        
        # Check drawdown frequency test
        freq_test = tests[1]
        assert 'frequency' in freq_test.description.lower()
        assert freq_test.validation_criteria['metric_type'] == 'drawdown_frequency_validation'
    
    def test_generate_performance_deviation_tests(self, generator, signal_feature):
        """Test performance deviation detection test generation."""
        tests = generator._generate_performance_deviation_tests(signal_feature)
        
        assert len(tests) >= 2  # Drift + anomaly tests
        
        # Check drift detection test
        drift_test = tests[0]
        assert drift_test.feature_name == "Q50"
        assert 'drift' in drift_test.description.lower()
        assert drift_test.priority == TestPriority.HIGH
        
        criteria = drift_test.validation_criteria
        assert criteria['metric_type'] == 'performance_drift_detection'
        assert criteria['drift_threshold'] == generator.thresholds.performance_deviation_threshold
        
        # Check anomaly detection test
        anomaly_test = tests[1]
        assert 'anomalies' in anomaly_test.description.lower()
        assert anomaly_test.validation_criteria['metric_type'] == 'performance_anomaly_detection'
    
    def test_feature_type_detection(self, generator, signal_feature, volatility_feature, position_sizing_feature):
        """Test feature type detection methods."""
        # Test signal feature detection
        assert generator._is_signal_feature(signal_feature) == True
        assert generator._is_volatility_feature(signal_feature) == False
        assert generator._is_position_sizing_feature(signal_feature) == False
        
        # Test volatility feature detection
        assert generator._is_signal_feature(volatility_feature) == False
        assert generator._is_volatility_feature(volatility_feature) == True
        assert generator._is_position_sizing_feature(volatility_feature) == False
        
        # Test position sizing feature detection
        assert generator._is_signal_feature(position_sizing_feature) == False
        assert generator._is_volatility_feature(position_sizing_feature) == False
        assert generator._is_position_sizing_feature(position_sizing_feature) == True
    
    def test_get_expected_ranges(self, generator, volatility_feature):
        """Test expected range retrieval."""
        ranges = generator._get_expected_ranges(volatility_feature)
        
        assert 'variance_range' in ranges
        assert ranges['variance_range'] == (0.0, 1.0)
    
    def test_get_expected_ranges_defaults(self, generator):
        """Test default range assignment for features without explicit ranges."""
        # Test vol_risk default
        vol_feature = FeatureSpec(name="vol_risk_test", category="test", tier="Tier 1", implementation="test")
        ranges = generator._get_expected_ranges(vol_feature)
        assert 'variance_range' in ranges
        assert ranges['variance_range'] == (0.0, 1.0)
        
        # Test fg_index default
        fg_feature = FeatureSpec(name="fg_index_test", category="test", tier="Tier 1", implementation="test")
        ranges = generator._get_expected_ranges(fg_feature)
        assert 'sentiment_range' in ranges
        assert ranges['sentiment_range'] == (0, 100)
        
        # Test Q50 default
        q50_feature = FeatureSpec(name="Q50_test", category="test", tier="Tier 1", implementation="test")
        ranges = generator._get_expected_ranges(q50_feature)
        assert 'quantile_range' in ranges
        assert ranges['quantile_range'] == (-3.0, 3.0)
    
    def test_comprehensive_test_generation(self, generator, signal_feature):
        """Test that comprehensive tests are generated for complex features."""
        tests = generator.generate_performance_characteristics_tests(signal_feature)
        
        # Should generate multiple test categories
        test_descriptions = [t.description.lower() for t in tests]
        
        # Check for different test types
        assert any('hit rate' in desc for desc in test_descriptions)
        assert any('sharpe ratio' in desc for desc in test_descriptions)
        assert any('drift' in desc for desc in test_descriptions)
        assert any('regime' in desc for desc in test_descriptions)
        
        # All tests should have proper metadata
        for test in tests:
            assert test.test_id != "auto"  # Should be generated
            assert test.feature_name == signal_feature.name
            assert test.test_type == TestType.PERFORMANCE
            assert test.description != ""
            assert test.rationale != ""
            assert test.estimated_duration > 0
            assert len(test.validation_criteria) > 0
    
    def test_empty_feature_handling(self, generator):
        """Test handling of features with minimal specifications."""
        minimal_feature = FeatureSpec(
            name="minimal_test",
            category="test",
            tier="Tier 2",
            implementation="test"
        )
        
        tests = generator.generate_performance_characteristics_tests(minimal_feature)
        
        # Should still generate some basic tests
        assert len(tests) > 0
        
        # Should include performance deviation tests at minimum
        deviation_tests = [t for t in tests if 'drift' in t.description.lower() or 'anomaly' in t.description.lower()]
        assert len(deviation_tests) > 0
    
    @patch('src.testing.generators.performance_characteristics_generator.logging.getLogger')
    def test_logging(self, mock_get_logger, generator, signal_feature):
        """Test that appropriate logging occurs during test generation."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Create new generator to use mocked logger
        test_generator = PerformanceCharacteristicsGenerator()
        test_generator.logger = mock_logger
        
        test_generator.generate_performance_characteristics_tests(signal_feature)
        
        # Should log the number of tests generated
        mock_logger.info.assert_called()
        call_args = mock_logger.info.call_args[0][0]
        assert 'Generated' in call_args
        assert 'performance characteristics tests' in call_args
        assert signal_feature.name in call_args


if __name__ == "__main__":
    pytest.main([__file__])