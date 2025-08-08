#!/usr/bin/env python3
"""
Demo script for economic hypothesis test generation.

This script demonstrates the economic hypothesis test generation capabilities
for different types of trading system features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.testing.generators.economic_hypothesis_generator import EconomicHypothesisTestGenerator
from src.testing.models.feature_spec import FeatureSpec


def create_sample_features():
    """Create sample feature specifications for demonstration."""
    
    # Q50 Core Signal Feature
    q50_feature = FeatureSpec(
        name="Q50",
        category="Core Signal Features",
        tier="Tier 1",
        implementation="qlib_custom/custom_multi_quantile.py",
        economic_hypothesis="Q50 reflects the probabilistic median of future returns—a directional vote shaped by current feature context. In asymmetric markets where liquidity, sentiment, and volatility drive price action, a skewed Q50 implies actionable imbalance. It captures value where behavioral or structural inertia inhibits instant price response.",
        performance_characteristics={
            "sharpe_ratio": 1.8,
            "hit_rate": 0.62
        },
        regime_dependencies={
            "trending": "strong_performance",
            "sideways": "mixed_results"
        }
    )
    
    # vol_risk Volatility Feature
    vol_risk_feature = FeatureSpec(
        name="vol_risk",
        category="Risk & Volatility Features",
        tier="Tier 1",
        implementation="src/data/crypto_loader.py",
        formula="Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)",
        economic_hypothesis="Variance preserves squared deviations from the mean, retaining signal intensity of extreme movements. Unlike standard deviation, it does not compress tail events through square rooting, making it more suitable for amplifying asymmetric risk bursts. Economically, this reflects nonlinear exposure—where downside risk is often more impactful than upside volatility.",
        failure_modes=["flat_markets", "synthetic_volatility", "lag_risk"]
    )
    
    # fg_index Sentiment Feature
    fg_index_feature = FeatureSpec(
        name="fg_index",
        category="Risk & Volatility Features",
        tier="Tier 2",
        implementation="crypto_loader.py",
        economic_hypothesis="Captures aggregate market sentiment—fear at lower values, greed at higher values. Synthesizes volatility, momentum, social media activity, dominance, trends. Provides contrarian signals at sentiment extremes.",
        empirical_ranges={
            "extreme_fear": 0.20,
            "extreme_greed": 0.80
        }
    )
    
    # Custom Kelly Sizing Feature
    kelly_feature = FeatureSpec(
        name="custom_kelly",
        category="Position Sizing Features",
        tier="Tier 1",
        implementation="training_pipeline.py",
        economic_hypothesis="Position sizing should dynamically respond to signal quality and regime volatility. Traditional Kelly assumes static payoff asymmetry and constant risk preference, but this enhanced version adapts the sizing based on predictive certainty (prob_up), estimated distribution shape (q10, q90), and signal/volatility gating mechanisms. The hypothesis is that smarter sizing yields better compounding outcomes by avoiding overconfidence in noisy or weak regimes.",
        performance_characteristics={
            "sharpe_improvement": 0.077,
            "drawdown_control": "improved"
        }
    )
    
    return [q50_feature, vol_risk_feature, fg_index_feature, kelly_feature]


def demonstrate_economic_hypothesis_tests():
    """Demonstrate economic hypothesis test generation for different features."""
    
    print("=" * 80)
    print("ECONOMIC HYPOTHESIS TEST GENERATION DEMO")
    print("=" * 80)
    print()
    
    # Initialize the generator
    generator = EconomicHypothesisTestGenerator()
    
    # Create sample features
    features = create_sample_features()
    
    for feature in features:
        print(f"🎯 FEATURE: {feature.name}")
        print(f"   Category: {feature.category}")
        print(f"   Tier: {feature.tier}")
        print(f"   Economic Hypothesis: {feature.economic_hypothesis[:100]}...")
        print()
        
        # Generate economic hypothesis tests
        tests = generator.generate_economic_hypothesis_tests(feature)
        
        print(f"   Generated {len(tests)} economic hypothesis tests:")
        print()
        
        for i, test in enumerate(tests, 1):
            print(f"   {i}. {test.description}")
            print(f"      Priority: {test.priority.value}")
            print(f"      Duration: {test.estimated_duration}s")
            print(f"      Rationale: {test.rationale}")
            
            # Show key validation criteria
            if test.validation_criteria:
                key_criteria = []
                for key, value in test.validation_criteria.items():
                    if key in ['metric', 'threshold', 'expected_behavior']:
                        key_criteria.append(f"{key}={value}")
                if key_criteria:
                    print(f"      Key Criteria: {', '.join(key_criteria)}")
            
            print()
        
        print("-" * 60)
        print()


def demonstrate_feature_specific_tests():
    """Demonstrate feature-specific test generation capabilities."""
    
    print("=" * 80)
    print("FEATURE-SPECIFIC TEST GENERATION CAPABILITIES")
    print("=" * 80)
    print()
    
    generator = EconomicHypothesisTestGenerator()
    
    # Test Q50 specific tests
    print("🎯 Q50 DIRECTIONAL BIAS TESTS")
    print("-" * 40)
    
    q50_feature = FeatureSpec(
        name="Q50",
        category="Core Signal Features",
        tier="Tier 1",
        implementation="test.py",
        economic_hypothesis="Q50 provides directional bias in trending markets through probabilistic median estimation."
    )
    
    q50_tests = generator._generate_q50_tests(q50_feature)
    for test in q50_tests:
        print(f"• {test.description}")
        print(f"  Validation: {test.validation_criteria.get('metric', 'N/A')}")
    print()
    
    # Test vol_risk specific tests
    print("🎯 VOL_RISK VARIANCE-BASED TESTS")
    print("-" * 40)
    
    vol_risk_feature = FeatureSpec(
        name="vol_risk",
        category="Risk & Volatility Features",
        tier="Tier 1",
        implementation="test.py",
        economic_hypothesis="vol_risk captures nonlinear risk exposure through variance calculation."
    )
    
    vol_risk_tests = generator._generate_vol_risk_tests(vol_risk_feature)
    for test in vol_risk_tests:
        print(f"• {test.description}")
        print(f"  Validation: {test.validation_criteria.get('metric', 'N/A')}")
    print()
    
    # Test sentiment feature tests
    print("🎯 SENTIMENT FEATURE TESTS")
    print("-" * 40)
    
    sentiment_feature = FeatureSpec(
        name="fg_index",
        category="Sentiment Features",
        tier="Tier 2",
        implementation="test.py",
        economic_hypothesis="Sentiment features provide contrarian signals at market extremes."
    )
    
    sentiment_tests = generator._generate_sentiment_tests(sentiment_feature)
    for test in sentiment_tests:
        print(f"• {test.description}")
        print(f"  Validation: {test.validation_criteria.get('metric', 'N/A')}")
    print()
    
    # Test Kelly sizing tests
    print("🎯 KELLY SIZING TESTS")
    print("-" * 40)
    
    kelly_feature = FeatureSpec(
        name="kelly_sizing",
        category="Position Sizing Features",
        tier="Tier 1",
        implementation="test.py",
        economic_hypothesis="Kelly sizing optimizes position size for growth while managing risk."
    )
    
    kelly_tests = generator._generate_kelly_sizing_tests(kelly_feature)
    for test in kelly_tests:
        print(f"• {test.description}")
        print(f"  Validation: {test.validation_criteria.get('metric', 'N/A')}")
    print()


def demonstrate_test_requirements_mapping():
    """Demonstrate how economic hypotheses map to specific test requirements."""
    
    print("=" * 80)
    print("ECONOMIC HYPOTHESIS → TEST REQUIREMENTS MAPPING")
    print("=" * 80)
    print()
    
    # Requirements mapping based on task details
    requirements_mapping = {
        "2.1": "Q50 directional bias validation",
        "2.2": "vol_risk variance-based risk measure tests", 
        "2.3": "Sentiment feature behavior validation tests",
        "2.4": "Kelly sizing risk-adjustment behavior tests",
        "2.5": "General economic hypothesis validation"
    }
    
    print("Task 3.2 Requirements Coverage:")
    print("-" * 40)
    
    for req_id, description in requirements_mapping.items():
        print(f"✅ Requirement {req_id}: {description}")
    
    print()
    print("Implementation Details:")
    print("-" * 40)
    
    implementation_details = [
        "• Q50 tests validate directional bias accuracy in trending markets",
        "• Q50 tests verify probabilistic median behavior and asymmetric market response",
        "• vol_risk tests validate variance calculation preserves tail events",
        "• vol_risk tests verify nonlinear exposure detection and fragility identification",
        "• Sentiment tests validate market sentiment reflection and contrarian signals",
        "• Kelly sizing tests verify risk-adjusted position sizing and growth optimality",
        "• All tests include statistical significance validation and regime awareness",
        "• Generic tests handle features without specific economic hypothesis patterns"
    ]
    
    for detail in implementation_details:
        print(detail)
    
    print()


if __name__ == "__main__":
    print("Starting Economic Hypothesis Test Generation Demo...")
    print()
    
    try:
        # Run demonstrations
        demonstrate_economic_hypothesis_tests()
        demonstrate_feature_specific_tests()
        demonstrate_test_requirements_mapping()
        
        print("=" * 80)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()
        print("The economic hypothesis test generator successfully creates:")
        print("• Feature-specific tests for Q50, vol_risk, sentiment, and Kelly sizing")
        print("• Comprehensive validation of economic hypotheses and theoretical foundations")
        print("• Statistical significance testing and regime-aware validation")
        print("• Proper test prioritization and execution time estimation")
        print()
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)