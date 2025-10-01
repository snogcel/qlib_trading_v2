"""
Test categorization system for organizing and prioritizing test cases.
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

from ..models.test_case import TestType, TestPriority


class TestCategory(Enum):
    """Categories for organizing test cases by purpose and importance."""
    CRITICAL = "critical"           # Must-pass tests for basic functionality
    PERFORMANCE = "performance"     # Performance and efficiency validation
    ROBUSTNESS = "robustness"      # Edge cases and failure handling
    INTEGRATION = "integration"     # Feature interaction and dependency tests
    REGRESSION = "regression"       # Tests to prevent known issues
    EXPLORATORY = "exploratory"     # Discovery tests for unknown behaviors


@dataclass
class TestCategorySpec:
    """Specification for a test category."""
    name: str
    description: str
    test_types: List[TestType]
    default_priority: TestPriority
    execution_order: int
    parallel_execution: bool = True
    timeout_multiplier: float = 1.0


class TestCategorizer:
    """
    System for categorizing and organizing test cases.
    
    Provides flexible categorization that can be extended for different
    testing strategies and organizational needs.
    """
    
    def __init__(self):
        """Initialize the test categorizer with default categories."""
        self._categories: Dict[TestCategory, TestCategorySpec] = {
            TestCategory.CRITICAL: TestCategorySpec(
                name="Critical Tests",
                description="Essential tests that must pass for basic functionality",
                test_types=[TestType.ECONOMIC_HYPOTHESIS, TestType.IMPLEMENTATION],
                default_priority=TestPriority.CRITICAL,
                execution_order=1,
                parallel_execution=False,  # Run sequentially for stability
                timeout_multiplier=2.0
            ),
            TestCategory.PERFORMANCE: TestCategorySpec(
                name="Performance Tests",
                description="Tests validating performance characteristics and metrics",
                test_types=[TestType.PERFORMANCE, TestType.REGIME_DEPENDENCY],
                default_priority=TestPriority.HIGH,
                execution_order=2,
                parallel_execution=True,
                timeout_multiplier=1.5
            ),
            TestCategory.ROBUSTNESS: TestCategorySpec(
                name="Robustness Tests",
                description="Tests for edge cases, failure modes, and error handling",
                test_types=[TestType.FAILURE_MODE],
                default_priority=TestPriority.MEDIUM,
                execution_order=3,
                parallel_execution=True,
                timeout_multiplier=3.0  # Failure tests may take longer
            ),
            TestCategory.INTEGRATION: TestCategorySpec(
                name="Integration Tests",
                description="Tests for feature interactions and dependencies",
                test_types=[TestType.INTERACTION],
                default_priority=TestPriority.MEDIUM,
                execution_order=4,
                parallel_execution=True,
                timeout_multiplier=2.0
            ),
            TestCategory.REGRESSION: TestCategorySpec(
                name="Regression Tests",
                description="Tests to prevent regression of known issues",
                test_types=[],  # Can include any type
                default_priority=TestPriority.LOW,
                execution_order=5,
                parallel_execution=True,
                timeout_multiplier=1.0
            ),
            TestCategory.EXPLORATORY: TestCategorySpec(
                name="Exploratory Tests",
                description="Discovery tests for unknown behaviors and edge cases",
                test_types=[],  # Can include any type
                default_priority=TestPriority.LOW,
                execution_order=6,
                parallel_execution=True,
                timeout_multiplier=1.0
            )
        }
        
        # Feature-specific categorization rules
        self._feature_category_rules: Dict[str, Dict[TestType, TestCategory]] = {
            'Q50': {
                TestType.ECONOMIC_HYPOTHESIS: TestCategory.CRITICAL,
                TestType.PERFORMANCE: TestCategory.CRITICAL,  # Q50 performance is critical
                TestType.REGIME_DEPENDENCY: TestCategory.PERFORMANCE,
                TestType.FAILURE_MODE: TestCategory.ROBUSTNESS,
                TestType.INTERACTION: TestCategory.INTEGRATION
            },
            'vol_risk': {
                TestType.IMPLEMENTATION: TestCategory.CRITICAL,  # Formula correctness critical
                TestType.PERFORMANCE: TestCategory.PERFORMANCE,
                TestType.FAILURE_MODE: TestCategory.ROBUSTNESS,
            },
            'kelly_sizing': {
                TestType.ECONOMIC_HYPOTHESIS: TestCategory.CRITICAL,
                TestType.IMPLEMENTATION: TestCategory.CRITICAL,
                TestType.PERFORMANCE: TestCategory.PERFORMANCE,
                TestType.FAILURE_MODE: TestCategory.ROBUSTNESS,
            }
        }
    
    def categorize_test_type(self, test_type: TestType, feature_name: str = "") -> TestCategory:
        """
        Categorize a test type, optionally considering the feature name.
        
        Args:
            test_type: Type of test to categorize
            feature_name: Name of feature being tested (optional)
            
        Returns:
            TestCategory for the test
        """
        # Check feature-specific rules first
        if feature_name and feature_name in self._feature_category_rules:
            feature_rules = self._feature_category_rules[feature_name]
            if test_type in feature_rules:
                return feature_rules[test_type]
        
        # Fall back to default categorization
        for category, spec in self._categories.items():
            if test_type in spec.test_types:
                return category
        
        # Default to performance category for unknown types
        return TestCategory.PERFORMANCE
    
    def get_category_spec(self, category: TestCategory) -> TestCategorySpec:
        """Get specification for a test category."""
        return self._categories[category]
    
    def get_execution_order(self, category: TestCategory) -> int:
        """Get execution order for a category."""
        return self._categories[category].execution_order
    
    def get_categories_by_priority(self) -> List[TestCategory]:
        """Get categories ordered by execution priority."""
        return sorted(
            self._categories.keys(),
            key=lambda c: self._categories[c].execution_order
        )
    
    def add_custom_category(self, category: TestCategory, spec: TestCategorySpec):
        """
        Add a custom test category.
        
        Args:
            category: Category enum value
            spec: Category specification
        """
        self._categories[category] = spec
    
    def add_feature_rule(self, feature_name: str, test_type: TestType, category: TestCategory):
        """
        Add a feature-specific categorization rule.
        
        Args:
            feature_name: Name of the feature
            test_type: Type of test
            category: Category to assign
        """
        if feature_name not in self._feature_category_rules:
            self._feature_category_rules[feature_name] = {}
        
        self._feature_category_rules[feature_name][test_type] = category
    
    def get_category_summary(self) -> Dict[str, Dict[str, any]]:
        """
        Get summary of all categories and their specifications.
        
        Returns:
            Dictionary with category information
        """
        summary = {}
        
        for category, spec in self._categories.items():
            summary[category.value] = {
                'name': spec.name,
                'description': spec.description,
                'test_types': [t.value for t in spec.test_types],
                'default_priority': spec.default_priority.value,
                'execution_order': spec.execution_order,
                'parallel_execution': spec.parallel_execution,
                'timeout_multiplier': spec.timeout_multiplier
            }
        
        return summary
    
    def filter_by_category(self, test_cases: List, category: TestCategory) -> List:
        """
        Filter test cases by category.
        
        Args:
            test_cases: List of test cases to filter
            category: Category to filter by
            
        Returns:
            Filtered list of test cases
        """
        filtered = []
        
        for test_case in test_cases:
            test_category = self.categorize_test_type(
                test_case.test_type, 
                test_case.feature_name
            )
            if test_category == category:
                filtered.append(test_case)
        
        return filtered
    
    def get_critical_test_types(self) -> Set[TestType]:
        """Get all test types considered critical."""
        critical_types = set()
        
        for category, spec in self._categories.items():
            if spec.default_priority == TestPriority.CRITICAL:
                critical_types.update(spec.test_types)
        
        return critical_types
    
    def estimate_execution_time(self, test_cases: List, category: TestCategory) -> float:
        """
        Estimate execution time for test cases in a category.
        
        Args:
            test_cases: List of test cases
            category: Category to estimate for
            
        Returns:
            Estimated execution time in seconds
        """
        category_tests = self.filter_by_category(test_cases, category)
        spec = self._categories[category]
        
        base_time = sum(test.estimated_duration for test in category_tests)
        adjusted_time = base_time * spec.timeout_multiplier
        
        # Adjust for parallel execution
        if spec.parallel_execution and len(category_tests) > 1:
            # Assume some parallelization benefit
            parallel_factor = min(0.6, 1.0 / len(category_tests) ** 0.5)
            adjusted_time *= (1 - parallel_factor)
        
        return adjusted_time