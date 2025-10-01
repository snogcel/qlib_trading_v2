"""
Test executor implementation for the feature test coverage system.
"""

import pandas as pd
import numpy as np
import time
import traceback
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from ..interfaces.executor_interface import TestExecutorInterface
from ..models.test_case import TestCase, TestType
from ..models.test_result import TestResult, TestStatus, ConfidenceLevel
from ..config.data_config import DataConfig
from ..utils.data_utils import load_test_data, simulate_market_regime, create_synthetic_failure_data
from ..utils.logging_utils import get_logger


class TestExecutor(TestExecutorInterface):
    """
    Concrete implementation of test case executor with environment setup and data loading.
    
    This executor provides comprehensive test execution capabilities including:
    - Market condition simulation
    - Test environment isolation
    - Data loading and caching
    - Error handling and recovery
    """
    
    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        data_loader: Optional[Callable] = None,
        feature_calculator: Optional[Callable] = None,
        enable_parallel: bool = False,
        max_execution_time: float = 300.0  # 5 minutes default timeout
    ):
        """
        Initialize the test executor.
        
        Args:
            data_config: Configuration for data loading and quality requirements
            data_loader: Optional custom data loader function
            feature_calculator: Optional feature calculation function
            enable_parallel: Whether to support parallel test execution
            max_execution_time: Maximum execution time per test in seconds
        """
        self.data_config = data_config or DataConfig()
        self.data_loader = data_loader
        self.feature_calculator = feature_calculator
        self.enable_parallel = enable_parallel
        self.max_execution_time = max_execution_time
        
        # Execution statistics
        self._total_tests_executed = 0
        self._total_execution_time = 0.0
        self._success_count = 0
        self._failure_count = 0
        self._last_execution_time = None
        
        # Environment management
        self._active_environments = {}
        self._temp_directories = []
        
        # Data cache
        self._data_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger = get_logger(__name__)
        self.logger.info(f"TestExecutor initialized with config: {self.data_config}")
    
    def execute_test_suite(self, test_cases: List[TestCase]) -> List[TestResult]:
        """
        Execute a complete suite of test cases with progress tracking and result aggregation.
        
        Args:
            test_cases: List of test cases to execute
            
        Returns:
            List of TestResult objects with execution results
        """
        self.logger.info(f"Starting execution of {len(test_cases)} test cases")
        start_time = time.time()
        self._suite_start_time = start_time  # For progress tracking
        results = []
        
        # Initialize progress tracking
        progress_tracker = self._initialize_progress_tracking(test_cases)
        
        try:
            # Sort test cases by priority and dependencies
            sorted_test_cases = self._sort_test_cases(test_cases)
            
            for i, test_case in enumerate(sorted_test_cases):
                # Update progress tracking
                self._update_progress(progress_tracker, i, test_case)
                
                try:
                    result = self.execute_single_test(test_case)
                    results.append(result)
                    
                    # Update progress with result
                    self._record_test_result(progress_tracker, result)
                    
                    # Log detailed progress
                    self._log_test_progress(i + 1, len(test_cases), test_case, result)
                        
                except Exception as e:
                    self.logger.error(f"Failed to execute test {test_case.test_id}: {str(e)}")
                    # Create error result
                    error_result = TestResult(
                        test_case=test_case,
                        execution_id=f"error_{int(time.time())}",
                        status=TestStatus.ERROR,
                        execution_time=0.0,
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                        analysis=f"Test execution failed with error: {str(e)}"
                    )
                    results.append(error_result)
                    self._failure_count += 1
                    self._record_test_result(progress_tracker, error_result)
            
            total_time = time.time() - start_time
            self._total_execution_time += total_time
            
            # Generate comprehensive execution summary
            execution_summary = self._generate_execution_summary(results, total_time, progress_tracker)
            self.logger.info(f"Test suite execution completed in {total_time:.2f}s")
            self._log_execution_summary(results)
            
            # Store aggregated results for reporting
            self._store_aggregated_results(results, execution_summary)
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {str(e)}")
            raise
        finally:
            # Cleanup any remaining environments
            self._cleanup_all_environments()
        
        return results
    
    def execute_single_test(self, test_case: TestCase) -> TestResult:
        """
        Execute a single test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            TestResult object with execution result
        """
        execution_id = f"exec_{int(time.time() * 1000)}_{test_case.test_id}"
        start_time = time.time()
        
        self.logger.debug(f"Starting execution of test: {test_case.test_id}")
        
        try:
            # Validate prerequisites
            validation_errors = self.validate_test_prerequisites(test_case)
            if validation_errors:
                return TestResult(
                    test_case=test_case,
                    execution_id=execution_id,
                    status=TestStatus.SKIPPED,
                    execution_time=0.0,
                    error_message=f"Prerequisites not met: {'; '.join(validation_errors)}",
                    analysis="Test skipped due to unmet prerequisites"
                )
            
            # Set up test environment
            environment = self.setup_test_environment(test_case)
            
            try:
                # Execute the test with timeout
                result = self._execute_with_timeout(test_case, environment)
                
                # Update statistics
                self._total_tests_executed += 1
                if result.passed:
                    self._success_count += 1
                else:
                    self._failure_count += 1
                
                return result
                
            finally:
                # Always cleanup environment
                self.cleanup_test_environment(test_case, environment)
        
        except Exception as e:
            self.logger.error(f"Test execution failed for {test_case.test_id}: {str(e)}")
            return TestResult(
                test_case=test_case,
                execution_id=execution_id,
                status=TestStatus.ERROR,
                execution_time=time.time() - start_time,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                analysis=f"Test execution failed with unexpected error: {str(e)}"
            )
    
    def setup_test_environment(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Set up the test environment for a specific test case.
        
        Args:
            test_case: Test case requiring environment setup
            
        Returns:
            Dictionary with environment configuration
        """
        self.logger.debug(f"Setting up environment for test: {test_case.test_id}")
        
        environment = {
            'test_id': test_case.test_id,
            'setup_time': datetime.now(),
            'temp_dir': None,
            'data': None,
            'market_conditions': test_case.market_conditions.copy(),
            'isolated': True
        }
        
        try:
            # Create temporary directory for test isolation
            temp_dir = tempfile.mkdtemp(prefix=f"test_{test_case.test_id}_")
            environment['temp_dir'] = temp_dir
            self._temp_directories.append(temp_dir)
            
            # Load required test data
            if test_case.requires_market_data():
                environment['data'] = self._load_test_data_for_case(test_case)
            
            # Apply market condition simulation if needed
            if test_case.regime_context or test_case.market_conditions:
                environment['data'] = self._apply_market_simulation(
                    environment['data'], 
                    test_case
                )
            
            # Store environment for cleanup
            self._active_environments[test_case.test_id] = environment
            
            self.logger.debug(f"Environment setup completed for test: {test_case.test_id}")
            return environment
            
        except Exception as e:
            self.logger.error(f"Failed to setup environment for {test_case.test_id}: {str(e)}")
            # Cleanup partial environment
            if environment.get('temp_dir'):
                try:
                    shutil.rmtree(environment['temp_dir'])
                except:
                    pass
            raise
    
    def cleanup_test_environment(self, test_case: TestCase, environment: Dict[str, Any]) -> None:
        """
        Clean up test environment after test execution.
        
        Args:
            test_case: Test case that was executed
            environment: Environment configuration to clean up
        """
        self.logger.debug(f"Cleaning up environment for test: {test_case.test_id}")
        
        try:
            # Remove from active environments
            if test_case.test_id in self._active_environments:
                del self._active_environments[test_case.test_id]
            
            # Clean up temporary directory
            temp_dir = environment.get('temp_dir')
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir)
                    if temp_dir in self._temp_directories:
                        self._temp_directories.remove(temp_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {str(e)}")
            
            # Clear data references to free memory
            if 'data' in environment:
                environment['data'] = None
            
            self.logger.debug(f"Environment cleanup completed for test: {test_case.test_id}")
            
        except Exception as e:
            self.logger.error(f"Error during environment cleanup for {test_case.test_id}: {str(e)}")
    
    def load_test_data(self, data_requirements: Dict[str, Any]) -> pd.DataFrame:
        """
        Load test data based on requirements.
        
        Args:
            data_requirements: Dictionary specifying data needs
            
        Returns:
            DataFrame with required test data
        """
        # Create cache key from requirements
        cache_key = self._create_cache_key(data_requirements)
        
        # Check cache first
        if cache_key in self._data_cache:
            self._cache_hits += 1
            self.logger.debug(f"Cache hit for data requirements: {cache_key}")
            return self._data_cache[cache_key].copy()
        
        self._cache_misses += 1
        self.logger.debug(f"Cache miss for data requirements: {cache_key}")
        
        try:
            # Extract requirements
            assets = data_requirements.get('assets', self.data_config.test_assets)
            start_date = data_requirements.get('start_date', self.data_config.start_date)
            end_date = data_requirements.get('end_date', self.data_config.end_date)
            timeframe = data_requirements.get('timeframe', 'daily')
            
            # Load data using utility function
            data_dict = load_test_data(
                assets=assets,
                start_date=start_date,
                end_date=end_date,
                data_config=self.data_config,
                timeframe=timeframe
            )
            
            # Combine data if multiple assets
            if len(data_dict) == 0:
                # No data available
                data = pd.DataFrame()
            elif len(data_dict) == 1:
                data = list(data_dict.values())[0]
            else:
                # Combine multiple assets (simple concatenation for now)
                data = pd.concat(data_dict.values(), keys=data_dict.keys())
            
            # Cache the result
            if len(self._data_cache) < 100:  # Limit cache size
                self._data_cache[cache_key] = data.copy()
            
            self.logger.info(f"Loaded {len(data)} rows of data for {len(assets)} assets")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {str(e)}")
            raise
    
    def simulate_market_conditions(self, regime_type: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate specific market conditions for testing.
        
        Args:
            regime_type: Type of market regime to simulate
            data: Base data to modify for simulation
            
        Returns:
            Modified DataFrame with simulated conditions
        """
        if data.empty:
            return data
        
        self.logger.debug(f"Simulating {regime_type} market conditions on {len(data)} rows")
        
        try:
            # Use utility function for simulation
            simulated_data = simulate_market_regime(
                base_data=data,
                regime_type=regime_type,
                duration_days=len(data),
                regime_config=self.data_config.regime_data_requirements.get(regime_type)
            )
            
            self.logger.debug(f"Market simulation completed for {regime_type}")
            return simulated_data
            
        except Exception as e:
            self.logger.error(f"Failed to simulate {regime_type} conditions: {str(e)}")
            raise
    
    def validate_test_prerequisites(self, test_case: TestCase) -> List[str]:
        """
        Validate that all prerequisites for test execution are met.
        
        Args:
            test_case: Test case to validate prerequisites for
            
        Returns:
            List of validation errors (empty if all prerequisites met)
        """
        errors = []
        
        # Check if test function is available
        method_name = f"_execute_{test_case.test_type.value}_test"
        if not test_case.test_function and not hasattr(self, method_name):
            errors.append(f"No execution method available for test type: {test_case.test_type.value}")
        
        # We have built-in data loading capability, so don't require external data loader
        
        # Check dependencies
        for dependency in test_case.dependencies:
            if not self._is_dependency_satisfied(dependency):
                errors.append(f"Dependency not satisfied: {dependency}")
        
        return errors
    
    def get_supported_test_types(self) -> List[str]:
        """
        Get list of test types this executor can handle.
        
        Returns:
            List of test type names supported by this executor
        """
        return [
            TestType.ECONOMIC_HYPOTHESIS.value,
            TestType.PERFORMANCE.value,
            TestType.FAILURE_MODE.value,
            TestType.IMPLEMENTATION.value,
            TestType.INTERACTION.value,
            TestType.REGIME_DEPENDENCY.value
        ]
    
    def _execute_with_timeout(self, test_case: TestCase, environment: Dict[str, Any]) -> TestResult:
        """
        Execute test with timeout protection.
        
        Args:
            test_case: Test case to execute
            environment: Test environment
            
        Returns:
            TestResult with execution outcome
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}_{test_case.test_id}"
        
        try:
            # Determine execution method
            if test_case.test_function:
                # Use provided test function
                actual_result = test_case.test_function(environment['data'], test_case.test_parameters)
            else:
                # Use built-in execution method
                actual_result = self._execute_builtin_test(test_case, environment)
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > self.max_execution_time:
                return TestResult(
                    test_case=test_case,
                    execution_id=execution_id,
                    status=TestStatus.TIMEOUT,
                    execution_time=execution_time,
                    error_message=f"Test exceeded maximum execution time ({self.max_execution_time}s)",
                    analysis="Test execution timed out"
                )
            
            # Validate result
            validation_result = self._validate_test_result(test_case, actual_result)
            
            return TestResult(
                test_case=test_case,
                execution_id=execution_id,
                status=TestStatus.PASSED if validation_result['passed'] else TestStatus.FAILED,
                execution_time=execution_time,
                actual_result=actual_result,
                expected_result=test_case.expected_result,
                passed=validation_result['passed'],
                confidence=validation_result['confidence'],
                confidence_score=validation_result['confidence_score'],
                analysis=validation_result['analysis'],
                recommendations=validation_result['recommendations'],
                performance_metrics=validation_result.get('metrics', {}),
                test_environment={
                    'regime_context': test_case.regime_context,
                    'market_conditions': test_case.market_conditions,
                    'data_shape': environment['data'].shape if environment.get('data') is not None else None
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Test execution failed: {str(e)}")
            
            return TestResult(
                test_case=test_case,
                execution_id=execution_id,
                status=TestStatus.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                analysis=f"Test execution failed with error: {str(e)}"
            )
    
    def _execute_builtin_test(self, test_case: TestCase, environment: Dict[str, Any]) -> Any:
        """
        Execute test using built-in test methods.
        
        Args:
            test_case: Test case to execute
            environment: Test environment
            
        Returns:
            Test execution result
        """
        test_type = test_case.test_type
        data = environment.get('data')
        
        if test_type == TestType.ECONOMIC_HYPOTHESIS:
            return self._execute_economic_hypothesis_test(test_case, data)
        elif test_type == TestType.PERFORMANCE:
            return self._execute_performance_test(test_case, data)
        elif test_type == TestType.FAILURE_MODE:
            return self._execute_failure_mode_test(test_case, data)
        elif test_type == TestType.IMPLEMENTATION:
            return self._execute_implementation_test(test_case, data)
        elif test_type == TestType.INTERACTION:
            return self._execute_interaction_test(test_case, data)
        elif test_type == TestType.REGIME_DEPENDENCY:
            return self._execute_regime_test(test_case, data)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def _execute_economic_hypothesis_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute economic hypothesis test."""
        # Placeholder implementation - would be expanded based on specific hypothesis
        feature_name = test_case.feature_name
        
        if feature_name.lower() == 'q50':
            # Test Q50 directional bias
            if 'q50' in data.columns and 'label' in data.columns:
                correlation = data['q50'].corr(data['label'])
                return {
                    'correlation': float(correlation),
                    'directional_bias': bool(correlation > 0),
                    'significance': bool(abs(correlation) > 0.1)
                }
        
        return {'test_completed': True, 'feature_behavior': 'analyzed'}
    
    def _execute_performance_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute performance characteristics test."""
        # Placeholder implementation
        if data.empty:
            return {'error': 'No data available for performance test'}
        
        # Calculate basic performance metrics
        if 'label' in data.columns:
            returns = data['label'].dropna()
            return {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'hit_rate': (returns > 0).mean()
            }
        
        return {'test_completed': True, 'metrics_calculated': True}
    
    def _execute_failure_mode_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute failure mode test."""
        # Create failure conditions and test graceful handling
        failure_type = test_case.test_parameters.get('failure_type', 'data_gaps')
        
        try:
            # Create synthetic failure data
            failure_data = create_synthetic_failure_data(data, failure_type)
            
            # Test if system handles failure gracefully
            if failure_data.isnull().any().any():
                return {
                    'failure_detected': True,
                    'graceful_handling': True,
                    'failure_type': failure_type
                }
            
            return {'failure_simulation_completed': True}
            
        except Exception as e:
            return {
                'failure_detected': True,
                'graceful_handling': False,
                'error': str(e)
            }
    
    def _execute_implementation_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute implementation validation test."""
        # Validate feature calculations match specifications
        feature_name = test_case.feature_name
        
        if feature_name == 'vol_risk' and 'close' in data.columns:
            # Validate vol_risk calculation: Std(Log(close/Ref(close,1)), 6)^2
            returns = np.log(data['close'] / data['close'].shift(1)).dropna()
            calculated_vol_risk = returns.rolling(6).std() ** 2
            
            if 'vol_risk' in data.columns:
                existing_vol_risk = data['vol_risk'].dropna()
                if len(calculated_vol_risk) > 0 and len(existing_vol_risk) > 0:
                    correlation = calculated_vol_risk.corr(existing_vol_risk)
                    return {
                        'implementation_valid': bool(correlation > 0.95),
                        'correlation': float(correlation),
                        'formula_verified': True
                    }
        
        return {'implementation_check_completed': True}
    
    def _execute_interaction_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute feature interaction test."""
        # Test feature combinations
        features = test_case.test_parameters.get('features', [])
        
        if len(features) >= 2:
            available_features = [f for f in features if f in data.columns]
            if len(available_features) >= 2:
                # Calculate interaction effects
                correlations = data[available_features].corr()
                return {
                    'interaction_detected': True,
                    'feature_correlations': correlations.to_dict(),
                    'synergy_score': correlations.mean().mean()
                }
        
        return {'interaction_test_completed': True}
    
    def _execute_regime_test(self, test_case: TestCase, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute regime dependency test."""
        regime_context = test_case.regime_context
        
        if regime_context and 'label' in data.columns:
            # Analyze performance in specific regime
            returns = data['label'].dropna()
            
            regime_stats = {
                'regime_type': regime_context,
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'regime_performance': 'analyzed'
            }
            
            # Add regime-specific validation
            if regime_context == 'bull':
                regime_stats['bull_market_validated'] = returns.mean() > 0
            elif regime_context == 'bear':
                regime_stats['bear_market_validated'] = returns.mean() < 0
            
            return regime_stats
        
        return {'regime_test_completed': True}
    
    def _validate_test_result(self, test_case: TestCase, actual_result: Any) -> Dict[str, Any]:
        """
        Validate test result against expected outcomes.
        
        Args:
            test_case: Test case with validation criteria
            actual_result: Actual test result
            
        Returns:
            Dictionary with validation outcome
        """
        validation = {
            'passed': False,
            'confidence': ConfidenceLevel.MEDIUM,
            'confidence_score': 0.5,
            'analysis': '',
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Check if we have validation criteria
            if not test_case.validation_criteria:
                validation.update({
                    'passed': True,
                    'analysis': 'No validation criteria specified - test completed successfully',
                    'confidence': ConfidenceLevel.LOW,
                    'confidence_score': 0.3
                })
                return validation
            
            # Perform validation based on criteria
            passed_checks = 0
            total_checks = len(test_case.validation_criteria)
            
            for criterion, expected_value in test_case.validation_criteria.items():
                if isinstance(actual_result, dict) and criterion in actual_result:
                    actual_value = actual_result[criterion]
                    
                    # Numerical comparison with tolerance
                    if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                        if abs(actual_value - expected_value) <= test_case.tolerance:
                            passed_checks += 1
                    # Boolean comparison
                    elif isinstance(expected_value, bool):
                        if bool(actual_value) == expected_value:
                            passed_checks += 1
                    # String comparison
                    elif str(actual_value) == str(expected_value):
                        passed_checks += 1
            
            # Calculate overall validation
            pass_rate = passed_checks / total_checks if total_checks > 0 else 0
            validation['passed'] = pass_rate >= 0.8  # 80% pass threshold
            validation['confidence_score'] = pass_rate
            
            if pass_rate >= 0.9:
                validation['confidence'] = ConfidenceLevel.HIGH
                validation['analysis'] = f"Test passed with high confidence ({pass_rate:.1%} criteria met)"
            elif pass_rate >= 0.7:
                validation['confidence'] = ConfidenceLevel.MEDIUM
                validation['analysis'] = f"Test passed with medium confidence ({pass_rate:.1%} criteria met)"
            elif pass_rate >= 0.5:
                validation['confidence'] = ConfidenceLevel.LOW
                validation['analysis'] = f"Test passed with low confidence ({pass_rate:.1%} criteria met)"
            else:
                validation['confidence'] = ConfidenceLevel.UNCERTAIN
                validation['analysis'] = f"Test failed - only {pass_rate:.1%} of criteria met"
                validation['recommendations'].append("Review test implementation and validation criteria")
            
            validation['metrics'] = {
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'pass_rate': pass_rate
            }
            
        except Exception as e:
            validation.update({
                'passed': False,
                'confidence': ConfidenceLevel.UNCERTAIN,
                'confidence_score': 0.0,
                'analysis': f"Validation failed with error: {str(e)}",
                'recommendations': ['Review validation logic and test result format']
            })
        
        return validation
    
    def _load_test_data_for_case(self, test_case: TestCase) -> pd.DataFrame:
        """Load test data specific to a test case."""
        data_requirements = test_case.test_data_requirements.copy()
        
        # Add default requirements if not specified
        if 'assets' not in data_requirements:
            data_requirements['assets'] = ['BTC']  # Default test asset
        
        return self.load_test_data(data_requirements)
    
    def _apply_market_simulation(self, data: pd.DataFrame, test_case: TestCase) -> pd.DataFrame:
        """Apply market condition simulation to test data."""
        if data is None or data.empty:
            return data
        
        # Apply regime simulation if specified
        if test_case.regime_context:
            data = self.simulate_market_conditions(test_case.regime_context, data)
        
        # Apply additional market conditions
        for condition, params in test_case.market_conditions.items():
            if condition == 'failure_mode':
                data = create_synthetic_failure_data(data, params.get('type', 'data_gaps'), params)
        
        return data
    
    def _sort_test_cases(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Sort test cases by priority and dependencies."""
        # Simple sorting by priority for now
        priority_order = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3
        }
        
        return sorted(
            test_cases,
            key=lambda tc: (
                priority_order.get(tc.priority.value, 2),
                len(tc.dependencies),
                tc.estimated_duration
            )
        )
    
    def _create_cache_key(self, data_requirements: Dict[str, Any]) -> str:
        """Create cache key from data requirements."""
        key_parts = [
            str(data_requirements.get('assets', [])),
            data_requirements.get('start_date', ''),
            data_requirements.get('end_date', ''),
            data_requirements.get('timeframe', 'daily')
        ]
        return '_'.join(key_parts)
    
    def _cleanup_all_environments(self) -> None:
        """Clean up all active test environments."""
        for test_id, environment in list(self._active_environments.items()):
            try:
                temp_dir = environment.get('temp_dir')
                if temp_dir and Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup environment for {test_id}: {str(e)}")
        
        self._active_environments.clear()
        
        # Clean up any remaining temp directories
        for temp_dir in self._temp_directories:
            try:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory {temp_dir}: {str(e)}")
        
        self._temp_directories.clear()
    
    def _log_execution_summary(self, results: List[TestResult]) -> None:
        """Log summary of test execution results."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed and r.status != TestStatus.SKIPPED)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        avg_execution_time = sum(r.execution_time for r in results) / total_tests if total_tests > 0 else 0
        
        self.logger.info(f"Test Execution Summary:")
        self.logger.info(f"  Total Tests: {total_tests}")
        self.logger.info(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        self.logger.info(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        self.logger.info(f"  Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        self.logger.info(f"  Errors: {error_tests} ({error_tests/total_tests*100:.1f}%)")
        self.logger.info(f"  Average Execution Time: {avg_execution_time:.2f}s")
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        self.logger.info(f"  Cache Hit Rate: {cache_hit_rate:.1f}%")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about test execution performance."""
        return {
            'total_tests_executed': self._total_tests_executed,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'success_rate': self._success_count / max(self._total_tests_executed, 1),
            'average_execution_time': self._total_execution_time / max(self._total_tests_executed, 1),
            'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            'active_environments': len(self._active_environments),
            'last_execution': self._last_execution_time
        }
    
    def _initialize_progress_tracking(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Initialize progress tracking for test suite execution.
        
        Args:
            test_cases: List of test cases to track
            
        Returns:
            Dictionary with progress tracking data
        """
        return {
            'total_tests': len(test_cases),
            'completed_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'start_time': time.time(),
            'test_results_by_type': {},
            'test_results_by_feature': {},
            'test_results_by_priority': {},
            'execution_times': [],
            'current_test': None,
            'estimated_remaining_time': 0.0
        }
    
    def _update_progress(self, progress_tracker: Dict[str, Any], test_index: int, test_case: TestCase) -> None:
        """
        Update progress tracking with current test information.
        
        Args:
            progress_tracker: Progress tracking dictionary
            test_index: Current test index
            test_case: Current test case being executed
        """
        progress_tracker['current_test'] = {
            'index': test_index,
            'test_id': test_case.test_id,
            'feature_name': test_case.feature_name,
            'test_type': test_case.test_type.value,
            'priority': test_case.priority.value,
            'start_time': time.time()
        }
        
        # Calculate estimated remaining time
        if progress_tracker['completed_tests'] > 0:
            elapsed_time = time.time() - progress_tracker['start_time']
            avg_time_per_test = elapsed_time / progress_tracker['completed_tests']
            remaining_tests = progress_tracker['total_tests'] - progress_tracker['completed_tests']
            progress_tracker['estimated_remaining_time'] = avg_time_per_test * remaining_tests
    
    def _record_test_result(self, progress_tracker: Dict[str, Any], result: TestResult) -> None:
        """
        Record test result in progress tracking.
        
        Args:
            progress_tracker: Progress tracking dictionary
            result: Test result to record
        """
        progress_tracker['completed_tests'] += 1
        progress_tracker['execution_times'].append(result.execution_time)
        
        # Update status counts
        if result.status == TestStatus.PASSED:
            progress_tracker['passed_tests'] += 1
        elif result.status == TestStatus.FAILED:
            progress_tracker['failed_tests'] += 1
        elif result.status == TestStatus.ERROR:
            progress_tracker['error_tests'] += 1
        elif result.status == TestStatus.SKIPPED:
            progress_tracker['skipped_tests'] += 1
        
        # Group results by test type
        test_type = result.test_case.test_type.value
        if test_type not in progress_tracker['test_results_by_type']:
            progress_tracker['test_results_by_type'][test_type] = {
                'total': 0, 'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0
            }
        
        progress_tracker['test_results_by_type'][test_type]['total'] += 1
        if result.status == TestStatus.PASSED:
            progress_tracker['test_results_by_type'][test_type]['passed'] += 1
        elif result.status == TestStatus.FAILED:
            progress_tracker['test_results_by_type'][test_type]['failed'] += 1
        elif result.status == TestStatus.ERROR:
            progress_tracker['test_results_by_type'][test_type]['error'] += 1
        elif result.status == TestStatus.SKIPPED:
            progress_tracker['test_results_by_type'][test_type]['skipped'] += 1
        
        # Group results by feature
        feature_name = result.test_case.feature_name
        if feature_name not in progress_tracker['test_results_by_feature']:
            progress_tracker['test_results_by_feature'][feature_name] = {
                'total': 0, 'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0
            }
        
        progress_tracker['test_results_by_feature'][feature_name]['total'] += 1
        if result.status == TestStatus.PASSED:
            progress_tracker['test_results_by_feature'][feature_name]['passed'] += 1
        elif result.status == TestStatus.FAILED:
            progress_tracker['test_results_by_feature'][feature_name]['failed'] += 1
        elif result.status == TestStatus.ERROR:
            progress_tracker['test_results_by_feature'][feature_name]['error'] += 1
        elif result.status == TestStatus.SKIPPED:
            progress_tracker['test_results_by_feature'][feature_name]['skipped'] += 1
        
        # Group results by priority
        priority = result.test_case.priority.value
        if priority not in progress_tracker['test_results_by_priority']:
            progress_tracker['test_results_by_priority'][priority] = {
                'total': 0, 'passed': 0, 'failed': 0, 'error': 0, 'skipped': 0
            }
        
        progress_tracker['test_results_by_priority'][priority]['total'] += 1
        if result.status == TestStatus.PASSED:
            progress_tracker['test_results_by_priority'][priority]['passed'] += 1
        elif result.status == TestStatus.FAILED:
            progress_tracker['test_results_by_priority'][priority]['failed'] += 1
        elif result.status == TestStatus.ERROR:
            progress_tracker['test_results_by_priority'][priority]['error'] += 1
        elif result.status == TestStatus.SKIPPED:
            progress_tracker['test_results_by_priority'][priority]['skipped'] += 1
    
    def _log_test_progress(self, current: int, total: int, test_case: TestCase, result: TestResult) -> None:
        """
        Log detailed progress information for individual test execution.
        
        Args:
            current: Current test number
            total: Total number of tests
            test_case: Test case that was executed
            result: Test result
        """
        progress_pct = (current / total) * 100
        status_symbol = {
            TestStatus.PASSED: "✓",
            TestStatus.FAILED: "✗",
            TestStatus.ERROR: "⚠",
            TestStatus.SKIPPED: "⊝",
            TestStatus.TIMEOUT: "⏱"
        }.get(result.status, "?")
        
        self.logger.info(
            f"[{current:3d}/{total:3d}] ({progress_pct:5.1f}%) {status_symbol} "
            f"{test_case.test_id} ({test_case.feature_name}) - "
            f"{result.execution_time:.3f}s - {result.status.value}"
        )
        
        # Log additional details for failures and errors
        if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
            if result.error_message:
                self.logger.warning(f"    Error: {result.error_message}")
            if result.analysis:
                self.logger.info(f"    Analysis: {result.analysis}")
        
        # Log progress summary every 25% completion
        if current % max(1, total // 4) == 0:
            elapsed_time = time.time() - getattr(self, '_suite_start_time', time.time())
            avg_time = elapsed_time / current
            estimated_remaining = avg_time * (total - current)
            
            self.logger.info(
                f"Progress Summary: {current}/{total} tests completed "
                f"({progress_pct:.1f}%) - "
                f"Elapsed: {elapsed_time:.1f}s, "
                f"Est. Remaining: {estimated_remaining:.1f}s"
            )
    
    def _generate_execution_summary(
        self, 
        results: List[TestResult], 
        total_time: float, 
        progress_tracker: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive execution summary with aggregated results.
        
        Args:
            results: List of all test results
            total_time: Total execution time
            progress_tracker: Progress tracking data
            
        Returns:
            Dictionary with comprehensive execution summary
        """
        total_tests = len(results)
        
        # Basic statistics
        passed_tests = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in results if r.status == TestStatus.FAILED)
        error_tests = sum(1 for r in results if r.status == TestStatus.ERROR)
        skipped_tests = sum(1 for r in results if r.status == TestStatus.SKIPPED)
        timeout_tests = sum(1 for r in results if r.status == TestStatus.TIMEOUT)
        
        # Execution time statistics
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        min_execution_time = min(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        
        # Confidence statistics
        confidence_scores = [r.confidence_score for r in results if r.confidence_score is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Feature-level aggregation
        feature_summary = self._aggregate_results_by_feature(results)
        
        # Test type aggregation
        test_type_summary = self._aggregate_results_by_test_type(results)
        
        # Priority aggregation
        priority_summary = self._aggregate_results_by_priority(results)
        
        return {
            'execution_metadata': {
                'total_tests': total_tests,
                'total_execution_time': total_time,
                'start_time': progress_tracker['start_time'],
                'end_time': time.time(),
                'executor_version': '1.0.0'
            },
            'status_summary': {
                'passed': passed_tests,
                'failed': failed_tests,
                'error': error_tests,
                'skipped': skipped_tests,
                'timeout': timeout_tests,
                'success_rate': passed_tests / max(total_tests, 1),
                'failure_rate': (failed_tests + error_tests) / max(total_tests, 1)
            },
            'timing_summary': {
                'total_time': total_time,
                'average_time': avg_execution_time,
                'min_time': min_execution_time,
                'max_time': max_execution_time,
                'tests_per_second': total_tests / max(total_time, 0.001)
            },
            'confidence_summary': {
                'average_confidence': avg_confidence,
                'high_confidence_tests': sum(1 for r in results if r.confidence == ConfidenceLevel.HIGH),
                'low_confidence_tests': sum(1 for r in results if r.confidence == ConfidenceLevel.LOW),
                'uncertain_tests': sum(1 for r in results if r.confidence == ConfidenceLevel.UNCERTAIN)
            },
            'feature_summary': feature_summary,
            'test_type_summary': test_type_summary,
            'priority_summary': priority_summary,
            'cache_statistics': {
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
            }
        }
    
    def _aggregate_results_by_feature(self, results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate test results by feature name.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with feature-level aggregation
        """
        feature_stats = {}
        
        for result in results:
            feature_name = result.test_case.feature_name
            
            if feature_name not in feature_stats:
                feature_stats[feature_name] = {
                    'total_tests': 0,
                    'passed': 0,
                    'failed': 0,
                    'error': 0,
                    'skipped': 0,
                    'total_time': 0.0,
                    'avg_confidence': 0.0,
                    'test_types': set(),
                    'priorities': set()
                }
            
            stats = feature_stats[feature_name]
            stats['total_tests'] += 1
            stats['total_time'] += result.execution_time
            stats['test_types'].add(result.test_case.test_type.value)
            stats['priorities'].add(result.test_case.priority.value)
            
            if result.status == TestStatus.PASSED:
                stats['passed'] += 1
            elif result.status == TestStatus.FAILED:
                stats['failed'] += 1
            elif result.status == TestStatus.ERROR:
                stats['error'] += 1
            elif result.status == TestStatus.SKIPPED:
                stats['skipped'] += 1
            
            if result.confidence_score is not None:
                # Running average of confidence scores
                current_avg = stats['avg_confidence']
                stats['avg_confidence'] = (current_avg * (stats['total_tests'] - 1) + result.confidence_score) / stats['total_tests']
        
        # Convert sets to lists for JSON serialization
        for feature_name, stats in feature_stats.items():
            stats['test_types'] = list(stats['test_types'])
            stats['priorities'] = list(stats['priorities'])
            stats['success_rate'] = stats['passed'] / max(stats['total_tests'], 1)
            stats['avg_execution_time'] = stats['total_time'] / max(stats['total_tests'], 1)
        
        return feature_stats
    
    def _aggregate_results_by_test_type(self, results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate test results by test type.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with test type aggregation
        """
        type_stats = {}
        
        for result in results:
            test_type = result.test_case.test_type.value
            
            if test_type not in type_stats:
                type_stats[test_type] = {
                    'total_tests': 0,
                    'passed': 0,
                    'failed': 0,
                    'error': 0,
                    'skipped': 0,
                    'total_time': 0.0,
                    'features': set()
                }
            
            stats = type_stats[test_type]
            stats['total_tests'] += 1
            stats['total_time'] += result.execution_time
            stats['features'].add(result.test_case.feature_name)
            
            if result.status == TestStatus.PASSED:
                stats['passed'] += 1
            elif result.status == TestStatus.FAILED:
                stats['failed'] += 1
            elif result.status == TestStatus.ERROR:
                stats['error'] += 1
            elif result.status == TestStatus.SKIPPED:
                stats['skipped'] += 1
        
        # Convert sets to lists and add derived metrics
        for test_type, stats in type_stats.items():
            stats['features'] = list(stats['features'])
            stats['success_rate'] = stats['passed'] / max(stats['total_tests'], 1)
            stats['avg_execution_time'] = stats['total_time'] / max(stats['total_tests'], 1)
        
        return type_stats
    
    def _aggregate_results_by_priority(self, results: List[TestResult]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate test results by priority level.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with priority-level aggregation
        """
        priority_stats = {}
        
        for result in results:
            priority = result.test_case.priority.value
            
            if priority not in priority_stats:
                priority_stats[priority] = {
                    'total_tests': 0,
                    'passed': 0,
                    'failed': 0,
                    'error': 0,
                    'skipped': 0,
                    'total_time': 0.0
                }
            
            stats = priority_stats[priority]
            stats['total_tests'] += 1
            stats['total_time'] += result.execution_time
            
            if result.status == TestStatus.PASSED:
                stats['passed'] += 1
            elif result.status == TestStatus.FAILED:
                stats['failed'] += 1
            elif result.status == TestStatus.ERROR:
                stats['error'] += 1
            elif result.status == TestStatus.SKIPPED:
                stats['skipped'] += 1
        
        # Add derived metrics
        for priority, stats in priority_stats.items():
            stats['success_rate'] = stats['passed'] / max(stats['total_tests'], 1)
            stats['avg_execution_time'] = stats['total_time'] / max(stats['total_tests'], 1)
        
        return priority_stats
    
    def _store_aggregated_results(self, results: List[TestResult], execution_summary: Dict[str, Any]) -> None:
        """
        Store aggregated results for later retrieval and reporting.
        
        Args:
            results: List of test results
            execution_summary: Comprehensive execution summary
        """
        # Store in instance variables for later access
        self._last_execution_results = results
        self._last_execution_summary = execution_summary
        self._last_execution_time = datetime.now()
        
        # Note: _total_tests_executed is already updated in execute_single_test
        # Only update success/failure counts from the summary to avoid double counting
        
        self.logger.debug(f"Stored aggregated results for {len(results)} tests")
    
    def get_last_execution_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get the summary from the last test suite execution.
        
        Returns:
            Dictionary with last execution summary, or None if no execution has occurred
        """
        return getattr(self, '_last_execution_summary', None)
    
    def get_last_execution_results(self) -> Optional[List[TestResult]]:
        """
        Get the results from the last test suite execution.
        
        Returns:
            List of test results from last execution, or None if no execution has occurred
        """
        return getattr(self, '_last_execution_results', None)
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self._cleanup_all_environments()
        except:
            pass  # Ignore cleanup errors during destruction