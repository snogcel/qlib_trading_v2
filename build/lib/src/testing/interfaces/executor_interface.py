"""
Interface for test case executors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd

from ..models.test_case import TestCase
from ..models.test_result import TestResult


class TestExecutorInterface(ABC):
    """
    Abstract interface for executing test cases against feature implementations.
    
    This interface defines the contract for components that can run tests
    and collect results from actual feature implementations.
    """
    
    @abstractmethod
    def execute_test_suite(self, test_cases: List[TestCase]) -> List[TestResult]:
        """
        Execute a complete suite of test cases.
        
        Args:
            test_cases: List of test cases to execute
            
        Returns:
            List of TestResult objects with execution results
        """
        pass
    
    @abstractmethod
    def execute_single_test(self, test_case: TestCase) -> TestResult:
        """
        Execute a single test case.
        
        Args:
            test_case: Test case to execute
            
        Returns:
            TestResult object with execution result
        """
        pass
    
    @abstractmethod
    def setup_test_environment(self, test_case: TestCase) -> Dict[str, Any]:
        """
        Set up the test environment for a specific test case.
        
        Args:
            test_case: Test case requiring environment setup
            
        Returns:
            Dictionary with environment configuration
        """
        pass
    
    @abstractmethod
    def cleanup_test_environment(self, test_case: TestCase, environment: Dict[str, Any]) -> None:
        """
        Clean up test environment after test execution.
        
        Args:
            test_case: Test case that was executed
            environment: Environment configuration to clean up
        """
        pass
    
    @abstractmethod
    def load_test_data(self, data_requirements: Dict[str, Any]) -> pd.DataFrame:
        """
        Load test data based on requirements.
        
        Args:
            data_requirements: Dictionary specifying data needs
            
        Returns:
            DataFrame with required test data
        """
        pass
    
    def simulate_market_conditions(self, regime_type: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate specific market conditions for testing.
        
        Args:
            regime_type: Type of market regime to simulate
            data: Base data to modify for simulation
            
        Returns:
            Modified DataFrame with simulated conditions
        """
        # Default implementation - can be overridden
        if regime_type == "bull":
            # Simulate bull market conditions
            data = data.copy()
            data['close'] = data['close'] * (1 + 0.001)  # Slight upward trend
        elif regime_type == "bear":
            # Simulate bear market conditions  
            data = data.copy()
            data['close'] = data['close'] * (1 - 0.001)  # Slight downward trend
        elif regime_type == "sideways":
            # Simulate sideways market
            data = data.copy()
            # Add some noise but no trend
            import numpy as np
            noise = np.random.normal(0, 0.0005, len(data))
            data['close'] = data['close'] * (1 + noise)
        
        return data
    
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
        if not test_case.test_function and not hasattr(self, f"_execute_{test_case.test_type.value}"):
            errors.append(f"No execution method available for test type: {test_case.test_type.value}")
        
        # Check data requirements
        if test_case.requires_market_data():
            if not hasattr(self, 'data_loader') or self.data_loader is None:
                errors.append("Market data required but no data loader configured")
        
        # Check dependencies
        for dependency in test_case.dependencies:
            if not self._is_dependency_satisfied(dependency):
                errors.append(f"Dependency not satisfied: {dependency}")
        
        return errors
    
    def _is_dependency_satisfied(self, dependency: str) -> bool:
        """
        Check if a test dependency is satisfied.
        
        Args:
            dependency: Name of the dependency to check
            
        Returns:
            True if dependency is satisfied, False otherwise
        """
        # Default implementation - should be overridden
        return True
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about test execution performance.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            'total_tests_executed': getattr(self, '_total_tests_executed', 0),
            'average_execution_time': getattr(self, '_average_execution_time', 0.0),
            'success_rate': getattr(self, '_success_rate', 0.0),
            'last_execution': getattr(self, '_last_execution_time', None)
        }
    
    @abstractmethod
    def get_supported_test_types(self) -> List[str]:
        """
        Get list of test types this executor can handle.
        
        Returns:
            List of test type names supported by this executor
        """
        pass
    
    def get_executor_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this executor implementation.
        
        Returns:
            Dictionary with executor capabilities and configuration
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'supported_test_types': self.get_supported_test_types(),
            'capabilities': {
                'parallel_execution': getattr(self, 'supports_parallel', False),
                'market_simulation': True,
                'data_loading': hasattr(self, 'data_loader'),
                'environment_isolation': True
            }
        }