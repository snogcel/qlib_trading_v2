#!/usr/bin/env python3
"""
Demo script showing the TestCase data model and generation interfaces.

This script demonstrates the key components implemented in task 3.1:
- TestCase data model with test metadata and execution details
- Base TestCaseGenerator class with extensible test type support
- Test categorization system (economic, performance, failure, etc.)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.testing.models.feature_spec import FeatureSpec
from src.testing.models.test_case import TestCase, TestType, TestPriority
from src.testing.generators.base_generator import TestCaseGenerator
from src.testing.generators.test_categorization import TestCategorizer, TestCategory


def demo_test_case_model():
    """Demonstrate the TestCase data model capabilities."""
    print("=== TestCase Data Model Demo ===")
    
    # Create a sample test case
    test_case = TestCase(
        test_id="demo_test_001",
        feature_name="Q50",
        test_type=TestType.ECONOMIC_HYPOTHESIS,
        description="Validate Q50 directional bias in trending markets",
        priority=TestPriority.CRITICAL,
        validation_criteria={
            'metric': 'directional_accuracy',
            'threshold': 0.52,
            'confidence_level': 0.95
        },
        regime_context="trending_markets",
        estimated_duration=5.0,
        rationale="Q50 should provide directional edge in trending conditions",
        failure_impact="Loss of directional signal quality"
    )
    
    print(f"Test ID: {test_case.test_id}")
    print(f"Feature: {test_case.feature_name}")
    print(f"Type: {test_case.test_type.value}")
    print(f"Priority: {test_case.priority.value}")
    print(f"Description: {test_case.description}")
    print(f"Is Executable: {test_case.is_executable()}")
    print(f"Requires Market Data: {test_case.requires_market_data()}")
    print(f"Execution Weight: {test_case.get_execution_weight():.2f}")
    print()


def demo_test_generation():
    """Demonstrate test case generation from feature specifications."""
    print("=== Test Case Generation Demo ===")
    
    # Create a sample feature specification
    feature = FeatureSpec(
        name="Q50",
        category="Core Signal Features",
        tier="Tier 1",
        implementation="src/features/signal_features.py",
        economic_hypothesis="Provides directional bias in trending markets",
        performance_characteristics={
            "hit_rate": 0.55,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15
        },
        failure_modes=["regime_misclassification", "whipsaw_markets"],
        regime_dependencies={
            "bull_market": "strong_performance",
            "bear_market": "moderate_performance",
            "sideways_market": "reduced_effectiveness"
        },
        empirical_ranges={"min": -1.0, "max": 1.0},
        interactions=["spread", "vol_risk"]
    )
    
    # Initialize the test generator
    generator = TestCaseGenerator()
    
    print(f"Feature: {feature.name}")
    print(f"Category: {feature.category}")
    print(f"Tier: {feature.tier}")
    print(f"Supported Test Types: {[t.value for t in generator.get_supported_test_types()]}")
    print()
    
    # Generate all tests for the feature
    all_tests = generator.generate_tests(feature)
    print(f"Generated {len(all_tests)} total tests")
    
    # Show breakdown by test type
    test_type_counts = {}
    for test in all_tests:
        test_type = test.test_type.value
        test_type_counts[test_type] = test_type_counts.get(test_type, 0) + 1
    
    print("Test breakdown by type:")
    for test_type, count in test_type_counts.items():
        print(f"  {test_type}: {count} tests")
    print()
    
    # Show some example tests
    print("Example generated tests:")
    for i, test in enumerate(all_tests[:3]):  # Show first 3 tests
        print(f"  {i+1}. {test.description}")
        print(f"     Type: {test.test_type.value}, Priority: {test.priority.value}")
        print(f"     Duration: {test.estimated_duration}s")
    print()


def demo_test_categorization():
    """Demonstrate the test categorization system."""
    print("=== Test Categorization Demo ===")
    
    categorizer = TestCategorizer()
    
    # Show available categories
    print("Available test categories:")
    categories = categorizer.get_categories_by_priority()
    for category in categories:
        spec = categorizer.get_category_spec(category)
        print(f"  {category.value}: {spec.name}")
        print(f"    Description: {spec.description}")
        print(f"    Execution Order: {spec.execution_order}")
        print(f"    Test Types: {[t.value for t in spec.test_types]}")
        print()
    
    # Demonstrate categorization of different test types
    print("Test type categorization examples:")
    test_types = [TestType.ECONOMIC_HYPOTHESIS, TestType.PERFORMANCE, TestType.FAILURE_MODE]
    
    for test_type in test_types:
        category = categorizer.categorize_test_type(test_type)
        print(f"  {test_type.value} -> {category.value}")
    
    # Show feature-specific categorization
    print("\nFeature-specific categorization (Q50):")
    for test_type in test_types:
        category = categorizer.categorize_test_type(test_type, "Q50")
        print(f"  Q50 {test_type.value} -> {category.value}")
    print()


def demo_extensible_generators():
    """Demonstrate extensible test type support."""
    print("=== Extensible Test Generator Demo ===")
    
    generator = TestCaseGenerator()
    
    # Create a custom test generator
    def custom_stress_test_generator(feature: FeatureSpec):
        """Custom generator for stress tests."""
        return [
            TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.FAILURE_MODE,
                description=f"Custom stress test for {feature.name}",
                priority=TestPriority.HIGH,
                validation_criteria={
                    'stress_scenario': 'extreme_volatility',
                    'expected_behavior': 'graceful_degradation'
                },
                rationale=f"Ensure {feature.name} handles extreme market stress",
                estimated_duration=15.0
            )
        ]
    
    # Register the custom generator
    generator.register_test_generator(TestType.FAILURE_MODE, custom_stress_test_generator)
    print("Registered custom stress test generator")
    
    # Create a simple feature for testing
    feature = FeatureSpec(
        name="vol_risk",
        category="Risk & Volatility Features", 
        tier="Tier 1",
        implementation="src/features/volatility_features.py",
        failure_modes=["flat_markets", "synthetic_volatility"]
    )
    
    # Generate failure mode tests using the custom generator
    failure_tests = generator.generate_failure_mode_tests(feature)
    print(f"Generated {len(failure_tests)} failure mode tests")
    
    for test in failure_tests:
        print(f"  - {test.description}")
        if "Custom stress test" in test.description:
            print(f"    (Custom generator used - Duration: {test.estimated_duration}s)")
    print()


def demo_test_categories_filtering():
    """Demonstrate filtering tests by categories."""
    print("=== Test Category Filtering Demo ===")
    
    # Create feature and generate tests
    feature = FeatureSpec(
        name="kelly_sizing",
        category="Position Sizing Features",
        tier="Tier 1", 
        implementation="src/features/position_sizing.py",
        economic_hypothesis="Optimizes position sizes based on Kelly criterion",
        performance_characteristics={"sharpe_ratio": 1.5},
        failure_modes=["extreme_drawdown"]
    )
    
    generator = TestCaseGenerator()
    categorizer = TestCategorizer()
    
    all_tests = generator.generate_tests(feature)
    print(f"Generated {len(all_tests)} total tests for {feature.name}")
    
    # Filter by categories
    categories_to_show = [TestCategory.CRITICAL, TestCategory.PERFORMANCE, TestCategory.ROBUSTNESS]
    
    for category in categories_to_show:
        category_tests = categorizer.filter_by_category(all_tests, category)
        print(f"\n{category.value.title()} tests ({len(category_tests)}):")
        
        for test in category_tests:
            print(f"  - {test.description}")
            print(f"    Priority: {test.priority.value}, Duration: {test.estimated_duration}s")
    
    # Show execution time estimates
    print(f"\nExecution time estimates:")
    for category in categories_to_show:
        estimated_time = categorizer.estimate_execution_time(all_tests, category)
        print(f"  {category.value}: {estimated_time:.1f} seconds")


def main():
    """Run all demo functions."""
    print("TestCase Data Model and Generation Interfaces Demo")
    print("=" * 60)
    print()
    
    demo_test_case_model()
    demo_test_generation()
    demo_test_categorization()
    demo_extensible_generators()
    demo_test_categories_filtering()
    
    print("Demo completed successfully!")
    print("\nKey features demonstrated:")
    print("✓ TestCase dataclass with comprehensive metadata")
    print("✓ TestCaseGenerator with extensible test type support")
    print("✓ Test categorization system (economic, performance, failure, etc.)")
    print("✓ Custom test generator registration")
    print("✓ Test filtering and execution planning")


if __name__ == "__main__":
    main()