"""
Interface for test case generators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models.feature_spec import FeatureSpec
from ..models.test_case import TestCase, TestType


class TestGeneratorInterface(ABC):
    """
    Abstract interface for generating test cases from feature specifications.
    
    This interface defines the contract for components that can create
    appropriate test cases based on feature characteristics.
    """
    
    @abstractmethod
    def generate_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate all appropriate test cases for a feature.
        
        Args:
            feature: Feature specification to generate tests for
            
        Returns:
            List of TestCase objects covering all necessary test scenarios
        """
        pass
    
    @abstractmethod
    def generate_economic_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate tests that validate economic hypotheses.
        
        Args:
            feature: Feature specification with economic hypothesis
            
        Returns:
            List of TestCase objects for economic validation
        """
        pass
    
    @abstractmethod
    def generate_performance_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate tests that validate performance characteristics.
        
        Args:
            feature: Feature specification with performance metrics
            
        Returns:
            List of TestCase objects for performance validation
        """
        pass
    
    @abstractmethod
    def generate_failure_mode_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate tests that probe documented failure modes.
        
        Args:
            feature: Feature specification with failure modes
            
        Returns:
            List of TestCase objects for failure mode testing
        """
        pass
    
    @abstractmethod
    def generate_implementation_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate tests that validate implementation correctness.
        
        Args:
            feature: Feature specification with implementation details
            
        Returns:
            List of TestCase objects for implementation validation
        """
        pass
    
    def generate_interaction_tests(self, features: List[FeatureSpec]) -> List[TestCase]:
        """
        Generate tests for feature interactions and dependencies.
        
        Args:
            features: List of feature specifications to test interactions
            
        Returns:
            List of TestCase objects for interaction testing
        """
        # Default implementation - can be overridden
        interaction_tests = []
        
        for feature in features:
            if feature.interactions:
                # Generate basic interaction tests
                for interaction in feature.interactions:
                    test_case = TestCase(
                        test_id="auto",
                        feature_name=feature.name,
                        test_type=TestType.INTERACTION,
                        description=f"Test interaction between {feature.name} and {interaction}",
                        rationale=f"Validate documented interaction: {interaction}"
                    )
                    interaction_tests.append(test_case)
        
        return interaction_tests
    
    def generate_regime_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate tests for different market regime scenarios.
        
        Args:
            feature: Feature specification with regime dependencies
            
        Returns:
            List of TestCase objects for regime testing
        """
        # Default implementation - can be overridden
        regime_tests = []
        
        if feature.regime_dependencies:
            for regime, behavior in feature.regime_dependencies.items():
                test_case = TestCase(
                    test_id="auto",
                    feature_name=feature.name,
                    test_type=TestType.REGIME_DEPENDENCY,
                    description=f"Test {feature.name} behavior in {regime} regime",
                    regime_context=regime,
                    rationale=f"Validate regime-specific behavior: {behavior}"
                )
                regime_tests.append(test_case)
        
        return regime_tests
    
    @abstractmethod
    def get_supported_test_types(self) -> List[TestType]:
        """
        Get list of test types this generator can create.
        
        Returns:
            List of TestType enums supported by this generator
        """
        pass
    
    def estimate_test_count(self, feature: FeatureSpec) -> int:
        """
        Estimate how many tests will be generated for a feature.
        
        Args:
            feature: Feature specification to estimate for
            
        Returns:
            Estimated number of test cases
        """
        count = 1  # At least implementation test
        
        if feature.economic_hypothesis:
            count += 2  # Basic economic tests
        
        if feature.performance_characteristics:
            count += len(feature.performance_characteristics)
        
        if feature.failure_modes:
            count += len(feature.failure_modes)
        
        if feature.regime_dependencies:
            count += len(feature.regime_dependencies)
        
        if feature.interactions:
            count += len(feature.interactions)
        
        return count
    
    def get_generator_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this generator implementation.
        
        Returns:
            Dictionary with generator capabilities and configuration
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'supported_test_types': [t.value for t in self.get_supported_test_types()],
            'capabilities': {
                'economic_tests': True,
                'performance_tests': True,
                'failure_mode_tests': True,
                'implementation_tests': True,
                'interaction_tests': True,
                'regime_tests': True
            }
        }