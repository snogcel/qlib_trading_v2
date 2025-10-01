"""
Base test case generator implementation with extensible test type support.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

from ..interfaces.generator_interface import TestGeneratorInterface
from ..models.feature_spec import FeatureSpec
from ..models.test_case import TestCase, TestType, TestPriority
from .economic_hypothesis_generator import EconomicHypothesisTestGenerator
from .performance_characteristics_generator import PerformanceCharacteristicsGenerator


class TestCaseGenerator(TestGeneratorInterface):
    """
    Base implementation of test case generator with extensible test type support.
    
    This class provides a framework for generating different types of tests
    and can be extended to support new test categories.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the test case generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize specialized generators
        self.economic_hypothesis_generator = EconomicHypothesisTestGenerator(config)
        self.performance_characteristics_generator = PerformanceCharacteristicsGenerator(config)
        
        # Registry of test generators by type
        self._test_generators: Dict[TestType, Callable[[FeatureSpec], List[TestCase]]] = {
            TestType.ECONOMIC_HYPOTHESIS: self.economic_hypothesis_generator.generate_economic_hypothesis_tests,
            TestType.PERFORMANCE: self.performance_characteristics_generator.generate_performance_characteristics_tests,
            TestType.FAILURE_MODE: self._generate_failure_mode_tests,
            TestType.IMPLEMENTATION: self._generate_implementation_tests,
            TestType.INTERACTION: self._generate_interaction_tests,
            TestType.REGIME_DEPENDENCY: self._generate_regime_dependency_tests,
        }
        
        # Test categorization system
        self._test_categories = {
            'critical': [TestType.ECONOMIC_HYPOTHESIS, TestType.IMPLEMENTATION],
            'performance': [TestType.PERFORMANCE, TestType.REGIME_DEPENDENCY],
            'robustness': [TestType.FAILURE_MODE, TestType.INTERACTION],
        }
        
        # Default test priorities by type
        self._default_priorities = {
            TestType.ECONOMIC_HYPOTHESIS: TestPriority.CRITICAL,
            TestType.IMPLEMENTATION: TestPriority.CRITICAL,
            TestType.PERFORMANCE: TestPriority.HIGH,
            TestType.REGIME_DEPENDENCY: TestPriority.HIGH,
            TestType.FAILURE_MODE: TestPriority.MEDIUM,
            TestType.INTERACTION: TestPriority.MEDIUM,
        }
    
    def generate_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate all appropriate test cases for a feature.
        
        Args:
            feature: Feature specification to generate tests for
            
        Returns:
            List of TestCase objects covering all necessary test scenarios
        """
        all_tests = []
        
        for test_type, generator_func in self._test_generators.items():
            try:
                tests = generator_func(feature)
                all_tests.extend(tests)
                self.logger.debug(f"Generated {len(tests)} {test_type.value} tests for {feature.name}")
            except Exception as e:
                self.logger.error(f"Failed to generate {test_type.value} tests for {feature.name}: {e}")
        
        # Sort tests by priority and estimated duration
        all_tests.sort(key=lambda t: (t.priority.value, t.estimated_duration))
        
        self.logger.info(f"Generated {len(all_tests)} total tests for feature {feature.name}")
        return all_tests
    
    def generate_economic_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate tests that validate economic hypotheses."""
        return self._test_generators[TestType.ECONOMIC_HYPOTHESIS](feature)
    
    def generate_performance_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate tests that validate performance characteristics."""
        return self._test_generators[TestType.PERFORMANCE](feature)
    
    def generate_failure_mode_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate tests that probe documented failure modes."""
        return self._test_generators[TestType.FAILURE_MODE](feature)
    
    def generate_implementation_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate tests that validate implementation correctness."""
        return self._test_generators[TestType.IMPLEMENTATION](feature)
    
    def get_supported_test_types(self) -> List[TestType]:
        """Get list of test types this generator can create."""
        return list(self._test_generators.keys())
    
    def register_test_generator(self, test_type: TestType, generator_func: Callable[[FeatureSpec], List[TestCase]]):
        """
        Register a custom test generator for a specific test type.
        
        Args:
            test_type: Type of test to generate
            generator_func: Function that generates tests for this type
        """
        self._test_generators[test_type] = generator_func
        self.logger.info(f"Registered custom generator for {test_type.value}")
    
    def get_tests_by_category(self, feature: FeatureSpec, category: str) -> List[TestCase]:
        """
        Generate tests for a specific category.
        
        Args:
            feature: Feature specification
            category: Test category ('critical', 'performance', 'robustness')
            
        Returns:
            List of test cases for the specified category
        """
        if category not in self._test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        tests = []
        for test_type in self._test_categories[category]:
            if test_type in self._test_generators:
                tests.extend(self._test_generators[test_type](feature))
        
        return tests
    

    

    
    def _generate_failure_mode_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate failure mode detection tests."""
        tests = []
        
        if not feature.failure_modes:
            return tests
        
        for failure_mode in feature.failure_modes:
            test_case = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.FAILURE_MODE,
                description=f"Test {failure_mode} failure mode for {feature.name}",
                priority=self._default_priorities[TestType.FAILURE_MODE],
                test_parameters={'failure_mode': failure_mode},
                validation_criteria={
                    'failure_mode': failure_mode,
                    'expected_behavior': 'graceful_degradation',
                    'alert_threshold': True
                },
                rationale=f"Ensure {feature.name} handles {failure_mode} gracefully",
                failure_impact=f"System vulnerability to {failure_mode}",
                estimated_duration=7.0
            )
            tests.append(test_case)
        
        return tests
    
    def _generate_implementation_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate implementation correctness tests."""
        tests = []
        
        # Basic implementation test
        test_case = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.IMPLEMENTATION,
            description=f"Validate implementation correctness for {feature.name}",
            priority=self._default_priorities[TestType.IMPLEMENTATION],
            validation_criteria={
                'implementation_path': feature.implementation,
                'formula_validation': True,
                'calculation_accuracy': 0.001
            },
            rationale=f"Ensure {feature.name} implementation matches specification",
            failure_impact="Implementation bugs could cause incorrect trading decisions",
            estimated_duration=4.0
        )
        tests.append(test_case)
        
        # Formula validation for features with specific formulas
        if 'vol_risk' in feature.name.lower():
            formula_test = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.IMPLEMENTATION,
                description=f"Validate vol_risk variance formula for {feature.name}",
                priority=TestPriority.CRITICAL,
                validation_criteria={
                    'formula': 'Std(Log(close/Ref(close,1)), 6)^2',
                    'calculation_method': 'variance_calculation',
                    'precision': 0.0001
                },
                rationale="vol_risk must use correct variance calculation formula",
                estimated_duration=3.0
            )
            tests.append(formula_test)
        
        return tests
    
    def _generate_interaction_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate feature interaction tests."""
        tests = []
        
        if not feature.interactions:
            return tests
        
        for interaction in feature.interactions:
            test_case = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.INTERACTION,
                description=f"Test interaction between {feature.name} and {interaction}",
                priority=self._default_priorities[TestType.INTERACTION],
                test_parameters={'interaction_feature': interaction},
                validation_criteria={
                    'interaction_type': 'positive_synergy',
                    'combined_performance': 'better_than_individual',
                    'correlation_threshold': 0.3
                },
                rationale=f"Validate documented interaction between {feature.name} and {interaction}",
                estimated_duration=8.0
            )
            tests.append(test_case)
        
        return tests
    
    def _generate_regime_dependency_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate regime dependency tests."""
        tests = []
        
        if not feature.regime_dependencies:
            return tests
        
        for regime, expected_behavior in feature.regime_dependencies.items():
            test_case = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.REGIME_DEPENDENCY,
                description=f"Test {feature.name} behavior in {regime} regime",
                priority=self._default_priorities[TestType.REGIME_DEPENDENCY],
                regime_context=regime,
                expected_result=expected_behavior,
                validation_criteria={
                    'regime': regime,
                    'expected_behavior': expected_behavior,
                    'performance_threshold': 'regime_specific'
                },
                rationale=f"Ensure {feature.name} performs as expected in {regime} market conditions",
                estimated_duration=6.0
            )
            tests.append(test_case)
        
        return tests