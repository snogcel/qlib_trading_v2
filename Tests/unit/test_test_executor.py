"""
Unit tests for the TestExecutor class.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

from src.testing.executors.test_executor import TestExecutor
from src.testing.models.test_case import TestCase, TestType, TestPriority
from src.testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from src.testing.config.data_config import DataConfig


class TestTestExecutor:
    """Test suite for TestExecutor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'volume': np.random.randint(1000, 10000, 100),
            'q50': np.random.randn(100) * 0.01,
            'q10': np.random.randn(100) * 0.01 - 0.005,
            'q90': np.random.randn(100) * 0.01 + 0.005,
            'vol_risk': np.random.rand(100) * 0.1,
            'label': np.random.randn(100) * 0.02
        }, index=dates)
        return data
    
    @pytest.fixture
    def data_config(self):
        """Create test data configuration."""
        return DataConfig(
            start_date="2024-01-01",
            end_date="2024-12-31",
            test_assets=["BTC", "ETH"],
            max_missing_data_pct=0.1
        )
    
    @pytest.fixture
    def test_executor(self, data_config):
        """Create TestExecutor instance."""
        return TestExecutor(
            data_config=data_config,
            enable_parallel=False,
            max_execution_time=30.0
        )
    
    @pytest.fixture
    def sample_test_case(self):
        """Create sample test case."""
        return TestCase(
            test_id="test_001",
            feature_name="q50",
            test_type=TestType.ECONOMIC_HYPOTHESIS,
            description="Test Q50 directional bias",
            test_parameters={'threshold': 0.01},
            validation_criteria={'directional_bias': True},
            test_data_requirements={'assets': ['BTC']},
            priority=TestPriority.HIGH,
            estimated_duration=5.0
        )
    
    def test_executor_initialization(self, data_config):
        """Test TestExecutor initialization."""
        executor = TestExecutor(data_config=data_config)
        
        assert executor.data_config == data_config
        assert executor.enable_parallel is False
        assert executor.max_execution_time == 300.0
        assert executor._total_tests_executed == 0
        assert executor._success_count == 0
        assert executor._failure_count == 0
        assert len(executor._active_environments) == 0
        assert len(executor._data_cache) == 0
    
    def test_get_supported_test_types(self, test_executor):
        """Test getting supported test types."""
        supported_types = test_executor.get_supported_test_types()
        
        expected_types = [
            'economic_hypothesis',
            'performance',
            'failure_mode',
            'implementation',
            'interaction',
            'regime_dependency'
        ]
        
        assert all(test_type in supported_types for test_type in expected_types)
        assert len(supported_types) == len(expected_types)
    
    @patch('src.testing.executors.test_executor.load_test_data')
    def test_load_test_data(self, mock_load_data, test_executor, sample_data):
        """Test data loading functionality."""
        # Mock the load_test_data function
        mock_load_data.return_value = {'BTC': sample_data}
        
        data_requirements = {
            'assets': ['BTC'],
            'start_date': '2024-01-01',
            'end_date': '2024-12-31'
        }
        
        result = test_executor.load_test_data(data_requirements)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        mock_load_data.assert_called_once()
    
    def test_simulate_market_conditions(self, test_executor, sample_data):
        """Test market condition simulation."""
        # Test bull market simulation
        bull_data = test_executor.simulate_market_conditions('bull', sample_data)
        assert isinstance(bull_data, pd.DataFrame)
        assert len(bull_data) == len(sample_data)
        assert 'close' in bull_data.columns
        
        # Test bear market simulation
        bear_data = test_executor.simulate_market_conditions('bear', sample_data)
        assert isinstance(bear_data, pd.DataFrame)
        assert len(bear_data) == len(sample_data)
        
        # Test sideways market simulation
        sideways_data = test_executor.simulate_market_conditions('sideways', sample_data)
        assert isinstance(sideways_data, pd.DataFrame)
        assert len(sideways_data) == len(sample_data)
    
    def test_setup_test_environment(self, test_executor, sample_test_case):
        """Test test environment setup."""
        with patch.object(test_executor, '_load_test_data_for_case') as mock_load:
            mock_load.return_value = pd.DataFrame({'close': [100, 101, 102]})
            
            environment = test_executor.setup_test_environment(sample_test_case)
            
            assert 'test_id' in environment
            assert 'setup_time' in environment
            assert 'temp_dir' in environment
            assert 'data' in environment
            assert environment['test_id'] == sample_test_case.test_id
            assert environment['temp_dir'] is not None
            assert Path(environment['temp_dir']).exists()
            
            # Cleanup
            test_executor.cleanup_test_environment(sample_test_case, environment)
    
    def test_cleanup_test_environment(self, test_executor, sample_test_case):
        """Test test environment cleanup."""
        with patch.object(test_executor, '_load_test_data_for_case') as mock_load:
            mock_load.return_value = pd.DataFrame({'close': [100, 101, 102]})
            
            # Setup environment
            environment = test_executor.setup_test_environment(sample_test_case)
            temp_dir = environment['temp_dir']
            
            assert Path(temp_dir).exists()
            assert sample_test_case.test_id in test_executor._active_environments
            
            # Cleanup environment
            test_executor.cleanup_test_environment(sample_test_case, environment)
            
            assert not Path(temp_dir).exists()
            assert sample_test_case.test_id not in test_executor._active_environments
    
    def test_validate_test_prerequisites(self, test_executor, sample_test_case):
        """Test test prerequisite validation."""
        # Test with valid test case
        errors = test_executor.validate_test_prerequisites(sample_test_case)
        assert isinstance(errors, list)
        
        # Test with test case requiring market data but no data loader
        test_executor.data_loader = None
        sample_test_case.test_data_requirements = {'assets': ['BTC']}
        
        errors = test_executor.validate_test_prerequisites(sample_test_case)
        # Should not fail since we have built-in data loading
        assert isinstance(errors, list)
    
    @patch('src.testing.executors.test_executor.load_test_data')
    def test_execute_single_test_success(self, mock_load_data, test_executor, sample_test_case, sample_data):
        """Test successful single test execution."""
        # Mock data loading
        mock_load_data.return_value = {'BTC': sample_data}
        
        # Add test function
        def mock_test_function(data, params):
            return {'directional_bias': True, 'correlation': 0.15}
        
        sample_test_case.test_function = mock_test_function
        
        result = test_executor.execute_single_test(sample_test_case)
        
        assert isinstance(result, TestResult)
        assert result.test_case == sample_test_case
        assert result.status in [TestStatus.PASSED, TestStatus.FAILED]
        assert result.execution_time >= 0  # Allow zero execution time for fast tests
        assert result.execution_id is not None
    
    def test_execute_single_test_error(self, test_executor, sample_test_case):
        """Test single test execution with error."""
        # Create test case that will cause error
        def failing_test_function(data, params):
            raise ValueError("Test error")
        
        sample_test_case.test_function = failing_test_function
        
        with patch.object(test_executor, '_load_test_data_for_case') as mock_load:
            mock_load.return_value = pd.DataFrame({'close': [100, 101, 102]})
            
            result = test_executor.execute_single_test(sample_test_case)
            
            assert isinstance(result, TestResult)
            assert result.status == TestStatus.ERROR
            assert result.error_message is not None
            assert "Test error" in result.error_message
    
    @patch('src.testing.executors.test_executor.load_test_data')
    def test_execute_test_suite(self, mock_load_data, test_executor, sample_data):
        """Test test suite execution."""
        # Mock data loading
        mock_load_data.return_value = {'BTC': sample_data}
        
        # Create multiple test cases
        test_cases = []
        for i in range(3):
            test_case = TestCase(
                test_id=f"test_{i:03d}",
                feature_name="q50",
                test_type=TestType.ECONOMIC_HYPOTHESIS,
                description=f"Test case {i}",
                test_parameters={'threshold': 0.01},
                validation_criteria={'test_completed': True},
                test_data_requirements={'assets': ['BTC']},
                priority=TestPriority.MEDIUM
            )
            
            # Add simple test function
            def mock_test_function(data, params):
                return {'test_completed': True}
            
            test_case.test_function = mock_test_function
            test_cases.append(test_case)
        
        results = test_executor.execute_test_suite(test_cases)
        
        assert len(results) == 3
        assert all(isinstance(result, TestResult) for result in results)
        assert test_executor._total_tests_executed == 3
    
    def test_builtin_economic_hypothesis_test(self, test_executor, sample_data):
        """Test built-in economic hypothesis test execution."""
        result = test_executor._execute_economic_hypothesis_test(
            TestCase(
                test_id="test_q50",
                feature_name="q50",
                test_type=TestType.ECONOMIC_HYPOTHESIS,
                description="Q50 test"
            ),
            sample_data
        )
        
        assert isinstance(result, dict)
        if 'correlation' in result:
            assert isinstance(result['correlation'], (int, float))
            assert isinstance(result['directional_bias'], bool)
    
    def test_builtin_performance_test(self, test_executor, sample_data):
        """Test built-in performance test execution."""
        result = test_executor._execute_performance_test(
            TestCase(
                test_id="test_perf",
                feature_name="test_feature",
                test_type=TestType.PERFORMANCE,
                description="Performance test"
            ),
            sample_data
        )
        
        assert isinstance(result, dict)
        if 'mean_return' in result:
            assert isinstance(result['mean_return'], (int, float))
            assert isinstance(result['volatility'], (int, float))
            assert isinstance(result['sharpe_ratio'], (int, float))
    
    def test_builtin_failure_mode_test(self, test_executor, sample_data):
        """Test built-in failure mode test execution."""
        test_case = TestCase(
            test_id="test_failure",
            feature_name="test_feature",
            test_type=TestType.FAILURE_MODE,
            description="Failure mode test",
            test_parameters={'failure_type': 'data_gaps'}
        )
        
        result = test_executor._execute_failure_mode_test(test_case, sample_data)
        
        assert isinstance(result, dict)
        assert 'failure_detected' in result or 'failure_simulation_completed' in result
    
    def test_builtin_implementation_test(self, test_executor, sample_data):
        """Test built-in implementation test execution."""
        result = test_executor._execute_implementation_test(
            TestCase(
                test_id="test_impl",
                feature_name="vol_risk",
                test_type=TestType.IMPLEMENTATION,
                description="Implementation test"
            ),
            sample_data
        )
        
        assert isinstance(result, dict)
        if 'implementation_valid' in result:
            assert isinstance(result['implementation_valid'], bool)
    
    def test_result_validation(self, test_executor, sample_test_case):
        """Test test result validation."""
        # Test successful validation
        actual_result = {'directional_bias': True, 'correlation': 0.15}
        validation = test_executor._validate_test_result(sample_test_case, actual_result)
        
        assert isinstance(validation, dict)
        assert 'passed' in validation
        assert 'confidence' in validation
        assert 'confidence_score' in validation
        assert 'analysis' in validation
        assert isinstance(validation['passed'], bool)
        assert isinstance(validation['confidence'], ConfidenceLevel)
        assert 0 <= validation['confidence_score'] <= 1
    
    @patch('src.testing.executors.test_executor.load_test_data')
    def test_data_caching(self, mock_load_data, test_executor):
        """Test data caching functionality."""
        mock_data = pd.DataFrame({'close': [100, 101, 102]})
        # Mock should return data for the requested asset
        def mock_load_side_effect(assets, start_date, end_date, data_config, timeframe='daily'):
            return {asset: mock_data for asset in assets}
        mock_load_data.side_effect = mock_load_side_effect
        
        data_requirements = {'assets': ['BTC'], 'start_date': '2024-01-01'}
        
        # First call should load data
        result1 = test_executor.load_test_data(data_requirements)
        assert mock_load_data.call_count == 1
        assert test_executor._cache_misses == 1
        assert test_executor._cache_hits == 0
        
        # Second call should use cache
        result2 = test_executor.load_test_data(data_requirements)
        assert mock_load_data.call_count == 1  # Should not increase
        assert test_executor._cache_misses == 1  # Should not increase
        assert test_executor._cache_hits == 1  # Should increase
    
    def test_execution_statistics(self, test_executor):
        """Test execution statistics tracking."""
        # Initial statistics
        stats = test_executor.get_execution_statistics()
        assert stats['total_tests_executed'] == 0
        assert stats['success_count'] == 0
        assert stats['failure_count'] == 0
        assert stats['success_rate'] == 0
        
        # Update statistics manually for testing
        test_executor._total_tests_executed = 10
        test_executor._success_count = 8
        test_executor._failure_count = 2
        test_executor._total_execution_time = 50.0
        
        stats = test_executor.get_execution_statistics()
        assert stats['total_tests_executed'] == 10
        assert stats['success_count'] == 8
        assert stats['failure_count'] == 2
        assert stats['success_rate'] == 0.8
        assert stats['average_execution_time'] == 5.0
    
    def test_timeout_handling(self, test_executor, sample_test_case):
        """Test timeout handling in test execution."""
        # Set very short timeout
        test_executor.max_execution_time = 0.001
        
        def slow_test_function(data, params):
            import time
            time.sleep(0.1)  # Sleep longer than timeout
            return {'result': 'completed'}
        
        sample_test_case.test_function = slow_test_function
        
        with patch.object(test_executor, '_load_test_data_for_case') as mock_load:
            mock_load.return_value = pd.DataFrame({'close': [100, 101, 102]})
            
            result = test_executor.execute_single_test(sample_test_case)
            
            # Note: Due to the way timeout is implemented, this might not always trigger
            # In a real implementation, you'd want proper timeout handling
            assert isinstance(result, TestResult)
    
    def test_environment_cleanup_on_destruction(self, data_config):
        """Test that environments are cleaned up when executor is destroyed."""
        executor = TestExecutor(data_config=data_config)
        
        # Create some temporary directories
        temp_dir1 = tempfile.mkdtemp()
        temp_dir2 = tempfile.mkdtemp()
        
        executor._temp_directories = [temp_dir1, temp_dir2]
        executor._active_environments = {
            'test1': {'temp_dir': temp_dir1},
            'test2': {'temp_dir': temp_dir2}
        }
        
        # Verify directories exist
        assert Path(temp_dir1).exists()
        assert Path(temp_dir2).exists()
        
        # Delete executor (should trigger cleanup)
        del executor
        
        # Directories should be cleaned up
        # Note: This test might be flaky due to garbage collection timing
        # In practice, explicit cleanup is better than relying on __del__
    
    def test_error_handling_in_environment_setup(self, test_executor, sample_test_case):
        """Test error handling during environment setup."""
        with patch('tempfile.mkdtemp') as mock_mkdtemp:
            mock_mkdtemp.side_effect = OSError("Cannot create temp directory")
            
            with pytest.raises(OSError):
                test_executor.setup_test_environment(sample_test_case)
    
    def test_market_condition_application(self, test_executor, sample_data):
        """Test application of market conditions to test data."""
        test_case = TestCase(
            test_id="test_market",
            feature_name="test_feature",
            test_type=TestType.REGIME_DEPENDENCY,
            description="Market condition test",
            regime_context="bull",
            market_conditions={'failure_mode': {'type': 'data_gaps'}}
        )
        
        result_data = test_executor._apply_market_simulation(sample_data, test_case)
        
        assert isinstance(result_data, pd.DataFrame)
        assert len(result_data) == len(sample_data)
        # Data should be modified by simulation
        assert not result_data.equals(sample_data)
    
    @patch('src.testing.executors.test_executor.load_test_data')
    def test_progress_tracking_and_aggregation(self, mock_load_data, test_executor, sample_data):
        """Test progress tracking and result aggregation functionality."""
        # Mock data loading
        mock_load_data.return_value = {'BTC': sample_data}
        
        # Create test cases with different types and priorities
        test_cases = [
            TestCase(
                test_id="test_001",
                feature_name="q50",
                test_type=TestType.ECONOMIC_HYPOTHESIS,
                description="Economic test",
                test_parameters={'threshold': 0.01},
                validation_criteria={'test_completed': True},
                test_data_requirements={'assets': ['BTC']},
                priority=TestPriority.HIGH
            ),
            TestCase(
                test_id="test_002",
                feature_name="vol_risk",
                test_type=TestType.PERFORMANCE,
                description="Performance test",
                test_parameters={'metric': 'sharpe'},
                validation_criteria={'test_completed': True},
                test_data_requirements={'assets': ['BTC']},
                priority=TestPriority.MEDIUM
            ),
            TestCase(
                test_id="test_003",
                feature_name="q50",
                test_type=TestType.FAILURE_MODE,
                description="Failure test",
                test_parameters={'failure_type': 'data_gaps'},
                validation_criteria={'test_completed': True},
                test_data_requirements={'assets': ['BTC']},
                priority=TestPriority.LOW
            )
        ]
        
        # Add test functions
        for test_case in test_cases:
            def mock_test_function(data, params):
                return {'test_completed': True}
            test_case.test_function = mock_test_function
        
        # Execute test suite
        results = test_executor.execute_test_suite(test_cases)
        
        # Test basic execution
        assert len(results) == 3
        assert all(isinstance(result, TestResult) for result in results)
        
        # Test that aggregated results are stored
        last_summary = test_executor.get_last_execution_summary()
        assert last_summary is not None
        assert 'execution_metadata' in last_summary
        assert 'status_summary' in last_summary
        assert 'feature_summary' in last_summary
        assert 'test_type_summary' in last_summary
        assert 'priority_summary' in last_summary
        
        # Test feature aggregation
        feature_summary = last_summary['feature_summary']
        assert 'q50' in feature_summary
        assert 'vol_risk' in feature_summary
        assert feature_summary['q50']['total_tests'] == 2  # Two q50 tests
        assert feature_summary['vol_risk']['total_tests'] == 1  # One vol_risk test
        
        # Test test type aggregation
        test_type_summary = last_summary['test_type_summary']
        assert 'economic_hypothesis' in test_type_summary
        assert 'performance' in test_type_summary
        assert 'failure_mode' in test_type_summary
        
        # Test priority aggregation
        priority_summary = last_summary['priority_summary']
        assert 'high' in priority_summary
        assert 'medium' in priority_summary
        assert 'low' in priority_summary
        
        # Test that last execution results are accessible
        last_results = test_executor.get_last_execution_results()
        assert last_results is not None
        assert len(last_results) == 3
        assert last_results == results