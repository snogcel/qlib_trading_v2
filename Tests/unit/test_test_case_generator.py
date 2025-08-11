"""
Unit tests for the TestCaseGenerator and test categorization system.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pytest
from unittest.mock import Mock, patch

from src.testing.generators.base_generator import TestCaseGenerator
from src.testing.generators.test_categorization import TestCategorizer, TestCategory
from src.testing.models.feature_spec import FeatureSpec
from src.testing.models.test_case import TestCase, TestType, TestPriority


class TestTestCaseGenerator:
    """Test cases for the TestCaseGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = TestCaseGenerator()
        
        # Create a sample feature spec
        self.sample_feature = FeatureSpec(
            name="Q50",
            category="Core Signal Features",
            tier="Tier 1",
            implementation="src/features/signal_features.py",
            economic_hypothesis="Provides directional bias in trending markets",
            performance_characteristics={
                "hit_rate": 0.55,
                "sharpe_ratio": 1.2
            },
            failure_modes=["regime_misclassification", "whipsaw_markets"],
            regime_dependencies={
                "bull_market": "strong_performance",
                "bear_market": "moderate_performance"
            },
            empirical_ranges={"min": -1.0, "max": 1.0},
            interactions=["spread", "vol_risk"]
        )
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        assert self.generator is not None
        assert len(self.generator.get_supported_test_types()) == 6
        assert TestType.ECONOMIC_HYPOTHESIS in self.generator.get_supported_test_types()
    
    def test_generate_all_tests(self):
        """Test generating all test types for a feature."""
        tests = self.generator.generate_tests(self.sample_feature)
        
        assert len(tests) > 0
        
        # Check that different test types are generated
        test_types = {test.test_type for test in tests}
        assert TestType.ECONOMIC_HYPOTHESIS in test_types
        assert TestType.PERFORMANCE in test_types
        assert TestType.FAILURE_MODE in test_types
        assert TestType.IMPLEMENTATION in test_types
        
        # Check that tests are sorted by priority
        priorities = [test.priority for test in tests]
        critical_count = sum(1 for p in priorities if p == TestPriority.CRITICAL)
        assert critical_count > 0
    
    def test_generate_economic_tests(self):
        """Test economic hypothesis test generation."""
        tests = self.generator.generate_economic_tests(self.sample_feature)
        
        assert len(tests) >= 1
        
        for test in tests:
            assert test.test_type == TestType.ECONOMIC_HYPOTHESIS
            assert test.feature_name == "Q50"
            # Updated to match new economic hypothesis test descriptions
            assert ("economic hypothesis" in test.description.lower() or 
                    "directional bias" in test.description.lower() or
                    "probabilistic median" in test.description.lower() or
                    "asymmetric" in test.description.lower() or
                    "anticipates shifts" in test.description.lower() or
                    "economic behavior" in test.description.lower())
            # Priority can vary based on specific test type
            assert test.priority in [TestPriority.CRITICAL, TestPriority.HIGH, TestPriority.MEDIUM]
    
    def test_generate_performance_tests(self):
        """Test performance characteristic test generation."""
        tests = self.generator.generate_performance_tests(self.sample_feature)
        
        assert len(tests) >= 2  # hit_rate and sharpe_ratio tests
        
        # Check hit rate test
        hit_rate_tests = [t for t in tests if "hit_rate" in t.description]
        assert len(hit_rate_tests) == 1
        assert hit_rate_tests[0].expected_result == 0.55
        
        # Check empirical range tests
        range_tests = [t for t in tests if "empirical range" in t.description]
        assert len(range_tests) >= 1
    
    def test_generate_failure_mode_tests(self):
        """Test failure mode test generation."""
        tests = self.generator.generate_failure_mode_tests(self.sample_feature)
        
        assert len(tests) == 2  # Two failure modes defined
        
        failure_modes = {test.test_parameters['failure_mode'] for test in tests}
        assert "regime_misclassification" in failure_modes
        assert "whipsaw_markets" in failure_modes
        
        for test in tests:
            assert test.test_type == TestType.FAILURE_MODE
            assert "failure mode" in test.description
    
    def test_generate_implementation_tests(self):
        """Test implementation validation test generation."""
        tests = self.generator.generate_implementation_tests(self.sample_feature)
        
        assert len(tests) >= 1
        
        for test in tests:
            assert test.test_type == TestType.IMPLEMENTATION
            assert test.priority == TestPriority.CRITICAL
            assert "implementation" in test.description.lower()
    
    def test_vol_risk_formula_test(self):
        """Test special formula validation for vol_risk features."""
        vol_risk_feature = FeatureSpec(
            name="vol_risk",
            category="Risk & Volatility Features",
            tier="Tier 1",
            implementation="src/features/volatility_features.py",
            economic_hypothesis="Captures variance-based risk measures"
        )
        
        tests = self.generator.generate_implementation_tests(vol_risk_feature)
        
        # Should have basic implementation test plus formula test
        assert len(tests) >= 2
        
        formula_tests = [t for t in tests if "formula" in t.description.lower()]
        assert len(formula_tests) == 1
        assert "Std(Log(close/Ref(close,1)), 6)^2" in formula_tests[0].validation_criteria['formula']
    
    def test_register_custom_generator(self):
        """Test registering custom test generators."""
        def custom_generator(feature: FeatureSpec):
            return [TestCase(
                test_id="custom_test",
                feature_name=feature.name,
                test_type=TestType.PERFORMANCE,
                description="Custom test"
            )]
        
        # Register custom generator
        self.generator.register_test_generator(TestType.PERFORMANCE, custom_generator)
        
        # Generate tests and verify custom generator is used
        tests = self.generator.generate_performance_tests(self.sample_feature)
        
        custom_tests = [t for t in tests if t.test_id == "custom_test"]
        assert len(custom_tests) == 1
        assert custom_tests[0].description == "Custom test"
    
    def test_get_tests_by_category(self):
        """Test getting tests by category."""
        critical_tests = self.generator.get_tests_by_category(self.sample_feature, 'critical')
        
        assert len(critical_tests) > 0
        
        # All tests should be critical types
        for test in critical_tests:
            assert test.test_type in [TestType.ECONOMIC_HYPOTHESIS, TestType.IMPLEMENTATION]
    
    def test_feature_without_optional_fields(self):
        """Test generator handles features with missing optional fields."""
        minimal_feature = FeatureSpec(
            name="minimal_feature",
            category="Test Features",
            tier="Tier 2",
            implementation="test.py"
        )
        
        tests = self.generator.generate_tests(minimal_feature)
        
        # Should still generate at least implementation tests
        assert len(tests) >= 1
        
        implementation_tests = [t for t in tests if t.test_type == TestType.IMPLEMENTATION]
        assert len(implementation_tests) >= 1


class TestTestCategorizer:
    """Test cases for the TestCategorizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.categorizer = TestCategorizer()
    
    def test_categorizer_initialization(self):
        """Test categorizer initializes with default categories."""
        categories = list(self.categorizer._categories.keys())
        
        assert TestCategory.CRITICAL in categories
        assert TestCategory.PERFORMANCE in categories
        assert TestCategory.ROBUSTNESS in categories
        assert TestCategory.INTEGRATION in categories
    
    def test_categorize_test_type(self):
        """Test test type categorization."""
        # Test default categorization
        category = self.categorizer.categorize_test_type(TestType.ECONOMIC_HYPOTHESIS)
        assert category == TestCategory.CRITICAL
        
        category = self.categorizer.categorize_test_type(TestType.FAILURE_MODE)
        assert category == TestCategory.ROBUSTNESS
    
    def test_feature_specific_categorization(self):
        """Test feature-specific categorization rules."""
        # Q50 performance should be critical
        category = self.categorizer.categorize_test_type(TestType.PERFORMANCE, "Q50")
        assert category == TestCategory.CRITICAL
        
        # Other features' performance should be in performance category
        category = self.categorizer.categorize_test_type(TestType.PERFORMANCE, "other_feature")
        assert category == TestCategory.PERFORMANCE
    
    def test_get_execution_order(self):
        """Test execution order retrieval."""
        critical_order = self.categorizer.get_execution_order(TestCategory.CRITICAL)
        performance_order = self.categorizer.get_execution_order(TestCategory.PERFORMANCE)
        
        assert critical_order < performance_order  # Critical should run first
    
    def test_get_categories_by_priority(self):
        """Test getting categories ordered by priority."""
        ordered_categories = self.categorizer.get_categories_by_priority()
        
        assert ordered_categories[0] == TestCategory.CRITICAL
        assert len(ordered_categories) == len(self.categorizer._categories)
    
    def test_add_custom_category(self):
        """Test adding custom categories."""
        from src.testing.generators.test_categorization import TestCategorySpec
        
        custom_category = TestCategory.EXPLORATORY
        custom_spec = TestCategorySpec(
            name="Custom Category",
            description="Custom test category",
            test_types=[TestType.PERFORMANCE],
            default_priority=TestPriority.LOW,
            execution_order=10
        )
        
        self.categorizer.add_custom_category(custom_category, custom_spec)
        
        retrieved_spec = self.categorizer.get_category_spec(custom_category)
        assert retrieved_spec.name == "Custom Category"
    
    def test_filter_by_category(self):
        """Test filtering test cases by category."""
        # Create sample test cases
        test_cases = [
            TestCase(
                test_id="test1",
                feature_name="Q50",
                test_type=TestType.ECONOMIC_HYPOTHESIS,
                description="Economic test"
            ),
            TestCase(
                test_id="test2",
                feature_name="Q50",
                test_type=TestType.FAILURE_MODE,
                description="Failure test"
            )
        ]
        
        critical_tests = self.categorizer.filter_by_category(test_cases, TestCategory.CRITICAL)
        robustness_tests = self.categorizer.filter_by_category(test_cases, TestCategory.ROBUSTNESS)
        
        assert len(critical_tests) == 1
        assert len(robustness_tests) == 1
        assert critical_tests[0].test_type == TestType.ECONOMIC_HYPOTHESIS
        assert robustness_tests[0].test_type == TestType.FAILURE_MODE
    
    def test_get_critical_test_types(self):
        """Test getting critical test types."""
        critical_types = self.categorizer.get_critical_test_types()
        
        assert TestType.ECONOMIC_HYPOTHESIS in critical_types
        assert TestType.IMPLEMENTATION in critical_types
    
    def test_estimate_execution_time(self):
        """Test execution time estimation."""
        test_cases = [
            TestCase(
                test_id="test1",
                feature_name="test",
                test_type=TestType.ECONOMIC_HYPOTHESIS,
                description="Test",
                estimated_duration=5.0
            ),
            TestCase(
                test_id="test2",
                feature_name="test",
                test_type=TestType.IMPLEMENTATION,
                description="Test",
                estimated_duration=3.0
            )
        ]
        
        estimated_time = self.categorizer.estimate_execution_time(test_cases, TestCategory.CRITICAL)
        
        # Should be base time (8.0) * timeout multiplier (2.0) = 16.0
        assert estimated_time > 8.0  # Should be adjusted upward
    
    def test_get_category_summary(self):
        """Test getting category summary."""
        summary = self.categorizer.get_category_summary()
        
        assert 'critical' in summary
        assert 'performance' in summary
        
        critical_info = summary['critical']
        assert critical_info['name'] == "Critical Tests"
        assert critical_info['execution_order'] == 1
        assert 'economic_hypothesis' in critical_info['test_types']


if __name__ == "__main__":
    pytest.main([__file__])