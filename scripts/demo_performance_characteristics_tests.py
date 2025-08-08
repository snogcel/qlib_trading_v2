#!/usr/bin/env python3
"""
Demo script for performance characteristics test generation.

This script demonstrates the comprehensive performance testing capabilities
including hit rates, Sharpe ratios, empirical ranges, risk-adjusted returns,
and drawdown control validation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testing.generators.performance_characteristics_generator import (
    PerformanceCharacteristicsGenerator,
    PerformanceThresholds
)
from testing.models.feature_spec import FeatureSpec
from testing.models.test_case import TestType, TestPriority


def create_sample_features():
    """Create sample feature specifications for demonstration."""
    
    # Core signal feature (Q50)
    q50_feature = FeatureSpec(
        name="Q50",
        category="Core Signal Features",
        tier="Tier 1",
        implementation="src/features/signal_features.py",
        economic_hypothesis="Provides directional bias in trending markets by capturing momentum",
        performance_characteristics={
            "hit_rate": 0.55,
            "sharpe_ratio": 1.2,
            "information_ratio": 0.8
        },
        regime_dependencies={
            "bull_market": "strong_performance",
            "bear_market": "moderate_performance",
            "sideways_market": "weak_performance"
        },
        empirical_ranges={
            "signal_range": (-3.0, 3.0)
        },
        interactions=["spread", "vol_risk"]
    )
    
    # Volatility risk feature
    vol_risk_feature = FeatureSpec(
        name="vol_risk",
        category="Risk & Volatility Features",
        tier="Tier 1",
        implementation="src/features/volatility_features.py",
        formula="Std(Log(close/Ref(close,1)), 6)^2",
        economic_hypothesis="Captures variance-based risk measures for position sizing",
        empirical_ranges={
            "variance_range": (0.0, 1.0),
            "normalized_range": (-2.0, 2.0)
        },
        failure_modes=["flat_markets", "synthetic_volatility", "data_gaps"]
    )
    
    # Kelly sizing feature
    kelly_feature = FeatureSpec(
        name="kelly_sizing",
        category="Position Sizing Features",
        tier="Tier 1",
        implementation="src/features/position_sizing.py",
        economic_hypothesis="Optimal position sizing based on Kelly criterion for risk-adjusted returns",
        performance_characteristics={
            "max_drawdown": 0.15,
            "risk_adjusted_return": 0.12,
            "volatility_target": 0.15
        },
        regime_dependencies={
            "high_volatility": "reduced_sizing",
            "low_volatility": "increased_sizing"
        }
    )
    
    # Sentiment feature
    fg_index_feature = FeatureSpec(
        name="fg_index",
        category="Market Sentiment Features",
        tier="Tier 2",
        implementation="src/features/sentiment_features.py",
        economic_hypothesis="Fear & Greed Index reflects market sentiment and contrarian opportunities",
        empirical_ranges={
            "sentiment_range": (0, 100)
        },
        regime_dependencies={
            "bull_market": "greed_indicator",
            "bear_market": "fear_indicator"
        }
    )
    
    return [q50_feature, vol_risk_feature, kelly_feature, fg_index_feature]


def demonstrate_performance_test_generation():
    """Demonstrate performance characteristics test generation."""
    
    print("=" * 80)
    print("PERFORMANCE CHARACTERISTICS TEST GENERATION DEMO")
    print("=" * 80)
    print()
    
    # Create generator with custom configuration
    config = {
        'hit_rate_min': 0.48,
        'sharpe_ratio_min': 0.6,
        'max_drawdown_threshold': 0.12,
        'performance_deviation_threshold': 0.15
    }
    
    generator = PerformanceCharacteristicsGenerator(config)
    
    print("Generator Configuration:")
    print(f"  Hit Rate Min: {generator.thresholds.hit_rate_min}")
    print(f"  Sharpe Ratio Min: {generator.thresholds.sharpe_ratio_min}")
    print(f"  Max Drawdown Threshold: {generator.thresholds.max_drawdown_threshold}")
    print(f"  Performance Deviation Threshold: {generator.thresholds.performance_deviation_threshold}")
    print()
    
    # Generate tests for each sample feature
    features = create_sample_features()
    
    for feature in features:
        print(f"Feature: {feature.name} ({feature.category})")
        print("-" * 60)
        
        # Generate performance tests
        tests = generator.generate_performance_characteristics_tests(feature)
        
        print(f"Generated {len(tests)} performance tests:")
        print()
        
        # Group tests by type for better presentation
        test_groups = {}
        for test in tests:
            test_desc = test.description.lower()
            
            if 'hit rate' in test_desc:
                group = 'Hit Rate Tests'
            elif 'sharpe ratio' in test_desc:
                group = 'Sharpe Ratio Tests'
            elif 'empirical range' in test_desc or 'normalization' in test_desc:
                group = 'Range & Normalization Tests'
            elif 'risk-adjusted' in test_desc or 'kelly' in test_desc:
                group = 'Risk-Adjusted Return Tests'
            elif 'drawdown' in test_desc:
                group = 'Drawdown Control Tests'
            elif 'drift' in test_desc or 'anomaly' in test_desc:
                group = 'Performance Monitoring Tests'
            elif 'regime' in test_desc:
                group = 'Regime Performance Tests'
            else:
                group = 'General Performance Tests'
            
            if group not in test_groups:
                test_groups[group] = []
            test_groups[group].append(test)
        
        # Display tests by group
        for group_name, group_tests in test_groups.items():
            print(f"  {group_name}:")
            for test in group_tests:
                priority_symbol = {
                    TestPriority.CRITICAL: "🔴",
                    TestPriority.HIGH: "🟡",
                    TestPriority.MEDIUM: "🟢",
                    TestPriority.LOW: "⚪"
                }.get(test.priority, "⚪")
                
                print(f"    {priority_symbol} {test.description}")
                print(f"      Priority: {test.priority.value}")
                print(f"      Duration: {test.estimated_duration:.1f}s")
                
                # Show key validation criteria
                if test.validation_criteria:
                    key_criteria = []
                    criteria = test.validation_criteria
                    
                    if 'min_threshold' in criteria:
                        key_criteria.append(f"Min: {criteria['min_threshold']}")
                    if 'max_threshold' in criteria:
                        key_criteria.append(f"Max: {criteria['max_threshold']}")
                    if 'expected_range' in criteria:
                        key_criteria.append(f"Range: {criteria['expected_range']}")
                    if 'regime' in criteria:
                        key_criteria.append(f"Regime: {criteria['regime']}")
                    
                    if key_criteria:
                        print(f"      Criteria: {', '.join(key_criteria)}")
                
                if test.regime_context:
                    print(f"      Regime: {test.regime_context}")
                
                print()
        
        print()


def demonstrate_feature_type_detection():
    """Demonstrate feature type detection capabilities."""
    
    print("=" * 80)
    print("FEATURE TYPE DETECTION DEMO")
    print("=" * 80)
    print()
    
    generator = PerformanceCharacteristicsGenerator()
    features = create_sample_features()
    
    print("Feature Type Detection Results:")
    print()
    
    for feature in features:
        print(f"Feature: {feature.name}")
        print(f"  Signal Feature: {generator._is_signal_feature(feature)}")
        print(f"  Volatility Feature: {generator._is_volatility_feature(feature)}")
        print(f"  Position Sizing Feature: {generator._is_position_sizing_feature(feature)}")
        print(f"  Normalized Feature: {generator._is_normalized_feature(feature)}")
        
        # Show expected ranges
        ranges = generator._get_expected_ranges(feature)
        if ranges:
            print(f"  Expected Ranges: {ranges}")
        
        print()


def demonstrate_test_prioritization():
    """Demonstrate test prioritization and categorization."""
    
    print("=" * 80)
    print("TEST PRIORITIZATION DEMO")
    print("=" * 80)
    print()
    
    generator = PerformanceCharacteristicsGenerator()
    features = create_sample_features()
    
    all_tests = []
    for feature in features:
        tests = generator.generate_performance_characteristics_tests(feature)
        all_tests.extend(tests)
    
    # Group by priority
    priority_groups = {}
    for test in all_tests:
        priority = test.priority
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(test)
    
    print(f"Total Tests Generated: {len(all_tests)}")
    print()
    
    # Show priority distribution
    for priority in [TestPriority.CRITICAL, TestPriority.HIGH, TestPriority.MEDIUM, TestPriority.LOW]:
        if priority in priority_groups:
            tests = priority_groups[priority]
            total_duration = sum(t.estimated_duration for t in tests)
            
            print(f"{priority.value.upper()} Priority Tests: {len(tests)}")
            print(f"  Estimated Total Duration: {total_duration:.1f} seconds")
            print(f"  Average Duration: {total_duration/len(tests):.1f} seconds")
            
            # Show sample tests
            print("  Sample Tests:")
            for test in tests[:3]:  # Show first 3
                print(f"    - {test.feature_name}: {test.description}")
            
            if len(tests) > 3:
                print(f"    ... and {len(tests) - 3} more")
            
            print()


def main():
    """Run the performance characteristics test generation demo."""
    
    try:
        print("Starting Performance Characteristics Test Generation Demo...")
        print()
        
        # Main demonstration
        demonstrate_performance_test_generation()
        
        # Feature type detection
        demonstrate_feature_type_detection()
        
        # Test prioritization
        demonstrate_test_prioritization()
        
        print("=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("✓ Hit rate and Sharpe ratio validation tests")
        print("✓ Empirical range verification for volatility features")
        print("✓ Risk-adjusted return and drawdown control tests")
        print("✓ Performance deviation detection and alerting")
        print("✓ Regime-specific performance validation")
        print("✓ Feature type detection and specialized test generation")
        print("✓ Test prioritization and duration estimation")
        print()
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())