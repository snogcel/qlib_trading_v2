#!/usr/bin/env python3
"""
Demo script for ValidationIntegrationSystem

This script demonstrates the ValidationIntegrationSystem functionality including:
- Generating validation tests from thesis statements
- Linking features to existing tests
- Validating performance claims against backtest data
- Creating monitoring alerts
- Writing generated tests to files

Usage:
    python scripts/demo_validation_integration_system.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.documentation.validation_integration_system import (
    ValidationIntegrationSystem,
    ValidationTest,
    TestLink,
    PerformanceValidation,
    Alert
)

from src.documentation.economic_rationale_generator import (
    FeatureEnhancement,
    ThesisStatement,
    EconomicRationale,
    ValidationCriterion,
    ChartExplanation,
    SupplyDemandClassification,
    SupplyDemandRole,
    MarketLayer,
    TimeHorizon
)


def create_sample_features():
    """Create sample feature enhancements for demonstration"""
    
    # Q50 Signal Feature
    q50_thesis = ThesisStatement(
        hypothesis="Q50 quantile signal predicts short-term returns by detecting supply/demand imbalances in order flow",
        economic_basis="When Q50 is negative, it indicates that the 50th percentile of expected returns is below zero, suggesting excess supply and selling pressure in the market",
        market_microstructure="Order flow imbalances create temporary price inefficiencies that can be exploited before the market fully adjusts",
        expected_behavior="Negative Q50 values should predict negative returns with 55% accuracy and positive Q50 should predict positive returns with 53% accuracy",
        failure_modes=["Low volume periods where order flow is not representative", "Market regime changes during high volatility", "News-driven events that override technical signals"]
    )
    
    q50_rationale = EconomicRationale(
        supply_factors=["Large sell orders", "Institutional selling pressure", "Profit-taking activities"],
        demand_factors=["Accumulation patterns", "Buying interest at support levels", "Short covering"],
        market_inefficiency="Order flow information is not immediately reflected in price due to market microstructure delays",
        interaction_effects=["Enhanced by Vol_Risk", "Works with Regime_Multiplier", "Confirmed by volume patterns"]
    )
    
    q50_feature = FeatureEnhancement(
        feature_name="Q50 Signal",
        category="Core Signal Features",
        existing_content={"description": "Primary quantile signal for return prediction"},
        thesis_statement=q50_thesis,
        economic_rationale=q50_rationale,
        validation_criteria=[
            ValidationCriterion(
                test_name="q50_statistical_significance",
                description="Test statistical significance of Q50 signal correlation with returns",
                success_threshold=0.05,
                test_implementation="test_q50_statistical_significance",
                frequency="daily",
                failure_action="alert_team"
            ),
            ValidationCriterion(
                test_name="q50_hit_rate_validation",
                description="Validate Q50 signal hit rate meets thesis claims",
                success_threshold=0.52,
                test_implementation="test_q50_hit_rate",
                frequency="weekly",
                failure_action="review_thesis"
            )
        ],
        supply_demand_classification=SupplyDemandClassification(
            primary_role=SupplyDemandRole.IMBALANCE_DETECTOR,
            secondary_roles=[SupplyDemandRole.SUPPLY_DETECTOR],
            market_layer=MarketLayer.MICROSTRUCTURE,
            time_horizon=TimeHorizon.INTRADAY,
            regime_sensitivity="medium",
            interaction_features=["Vol_Risk", "Regime_Multiplier"]
        ),
        chart_explanation=ChartExplanation(
            visual_description="Q50 appears as oscillating line around zero, with negative values indicating selling pressure",
            example_scenarios=["Q50 drops below -0.02 before price decline", "Q50 rises above 0.02 before price rally"],
            chart_patterns=["Divergence with price", "Support/resistance at zero line"],
            false_signals=["Whipsaws during low volume", "Noise during news events"],
            confirmation_signals=["Volume increase", "Regime alignment", "Vol_Risk confirmation"]
        ),
        dependencies=["Vol_Risk", "Regime_Classification"],
        validated=False
    )
    
    # Vol_Risk Feature
    vol_risk_thesis = ThesisStatement(
        hypothesis="Vol_Risk (variance-based risk measure) provides superior risk assessment compared to standard deviation by capturing tail risk and regime changes",
        economic_basis="Variance captures both upside and downside volatility, providing a more complete picture of market uncertainty and risk",
        market_microstructure="Variance-based measures are more sensitive to regime changes and tail events that standard deviation might miss",
        expected_behavior="Vol_Risk should increase before major market moves and decrease during stable periods, with Sharpe ratio improvement of 0.1-0.2",
        failure_modes=["Extremely low volatility regimes", "Black swan events", "Data quality issues"]
    )
    
    vol_risk_rationale = EconomicRationale(
        market_inefficiency="Risk is often mispriced in the short term, creating opportunities for risk-adjusted strategies",
        regime_dependency="More effective in volatile regimes, less useful in extremely stable periods"
    )
    
    vol_risk_feature = FeatureEnhancement(
        feature_name="Vol_Risk",
        category="Risk & Volatility Features",
        existing_content={"description": "Variance-based risk measure for position sizing"},
        thesis_statement=vol_risk_thesis,
        economic_rationale=vol_risk_rationale,
        validation_criteria=[
            ValidationCriterion(
                test_name="vol_risk_predictive_power",
                description="Test Vol_Risk ability to predict volatility changes",
                success_threshold=0.3,
                test_implementation="test_vol_risk_prediction",
                frequency="daily",
                failure_action="recalibrate_model"
            )
        ],
        supply_demand_classification=SupplyDemandClassification(
            primary_role=SupplyDemandRole.RISK_ASSESSOR,
            secondary_roles=[SupplyDemandRole.REGIME_CLASSIFIER],
            market_layer=MarketLayer.FUNDAMENTAL,
            time_horizon=TimeHorizon.DAILY,
            regime_sensitivity="high",
            interaction_features=["Q50", "Regime_Features"]
        ),
        chart_explanation=ChartExplanation(
            visual_description="Vol_Risk appears as smoothed line that spikes before major price moves",
            example_scenarios=["Vol_Risk increases 2 days before 5% price drop", "Low Vol_Risk during consolidation"],
            chart_patterns=["Leading indicator for breakouts", "Mean reversion patterns"],
            false_signals=["Noise during earnings", "Holiday periods"],
            confirmation_signals=["Price volatility increase", "Volume expansion"]
        ),
        dependencies=["Price_Data", "Volume_Data"],
        validated=False
    )
    
    return [q50_feature, vol_risk_feature]


def demonstrate_validation_integration():
    """Demonstrate ValidationIntegrationSystem functionality"""
    
    print("=" * 80)
    print("VALIDATION INTEGRATION SYSTEM DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize the system
    print("1. Initializing ValidationIntegrationSystem...")
    validation_system = ValidationIntegrationSystem(
        training_pipeline_path="src/training_pipeline.py",
        test_base_dir="tests",
        validation_config_path="config/validation_config.json"
    )
    print(f"✓ System initialized with config: {len(validation_system.config)} sections")
    print(f"✓ Training dates loaded: {validation_system.training_dates}")
    print()
    
    # Create sample features
    print("2. Creating sample feature enhancements...")
    features = create_sample_features()
    print(f"✓ Created {len(features)} sample features:")
    for feature in features:
        print(f"  - {feature.feature_name}: {feature.thesis_statement.hypothesis[:60]}...")
    print()
    
    # Generate validation tests for each feature
    print("3. Generating validation tests...")
    all_tests = []
    for feature in features:
        print(f"\n   Processing {feature.feature_name}:")
        
        try:
            tests = validation_system.create_validation_tests(
                feature.thesis_statement, 
                feature.feature_name
            )
            all_tests.extend(tests)
            
            print(f"   ✓ Generated {len(tests)} tests:")
            for test in tests:
                print(f"     - {test.test_name} ({test.test_type})")
                
        except Exception as e:
            print(f"   ✗ Error generating tests: {e}")
    
    print(f"\n✓ Total tests generated: {len(all_tests)}")
    print()
    
    # Link to existing tests
    print("4. Linking to existing tests...")
    all_links = []
    for feature in features:
        print(f"\n   Linking {feature.feature_name}:")
        
        try:
            links = validation_system.link_to_existing_tests(feature)
            all_links.extend(links)
            
            if links:
                print(f"   ✓ Found {len(links)} existing test links:")
                for link in links:
                    print(f"     - {link.test_function_name} (confidence: {link.confidence:.2f})")
            else:
                print("   - No existing test links found")
                
        except Exception as e:
            print(f"   ✗ Error linking tests: {e}")
    
    print(f"\n✓ Total test links found: {len(all_links)}")
    print()
    
    # Validate performance claims
    print("5. Validating performance claims...")
    performance_validations = []
    for feature in features:
        print(f"\n   Validating {feature.feature_name}:")
        
        try:
            validation = validation_system.validate_performance_claims(feature)
            performance_validations.append(validation)
            
            print(f"   ✓ Performance validation completed:")
            print(f"     - Validation passed: {validation.validation_passed}")
            print(f"     - Claimed performance: {validation.claimed_performance}")
            print(f"     - Actual performance: {validation.actual_performance}")
            
            if validation.recommendations:
                print(f"     - Recommendations: {len(validation.recommendations)}")
                for rec in validation.recommendations[:2]:  # Show first 2
                    print(f"       • {rec[:60]}...")
                    
        except Exception as e:
            print(f"   ✗ Error validating performance: {e}")
    
    print(f"\n✓ Performance validations completed: {len(performance_validations)}")
    print()
    
    # Create monitoring alerts
    print("6. Creating monitoring alerts...")
    all_alerts = []
    for feature in features:
        print(f"\n   Creating alerts for {feature.feature_name}:")
        
        try:
            alerts = validation_system.create_monitoring_alerts(feature)
            all_alerts.extend(alerts)
            
            print(f"   ✓ Created {len(alerts)} alerts:")
            for alert in alerts:
                print(f"     - {alert.alert_type} ({alert.severity}): {alert.message[:50]}...")
                
        except Exception as e:
            print(f"   ✗ Error creating alerts: {e}")
    
    print(f"\n✓ Total alerts created: {len(all_alerts)}")
    print()
    
    # Write test files
    print("7. Writing generated tests to files...")
    
    # Group tests by feature
    feature_tests = {}
    for test in all_tests:
        feature_name = test.test_name.split('_')[1] if '_' in test.test_name else 'general'
        if feature_name not in feature_tests:
            feature_tests[feature_name] = []
        feature_tests[feature_name].append(test)
    
    written_files = []
    for feature_name, tests in feature_tests.items():
        output_path = f"tests/generated/test_{feature_name}_validation.py"
        
        try:
            success = validation_system.write_test_file(tests, output_path)
            if success:
                written_files.append(output_path)
                print(f"   ✓ Written {len(tests)} tests to {output_path}")
            else:
                print(f"   ✗ Failed to write tests to {output_path}")
                
        except Exception as e:
            print(f"   ✗ Error writing {output_path}: {e}")
    
    print(f"\n✓ Test files written: {len(written_files)}")
    print()
    
    # Get validation summary
    print("8. Generating validation summary...")
    summary = validation_system.get_validation_summary()
    
    print("✓ Validation Integration Summary:")
    print(f"  - Generated tests: {summary['generated_tests']}")
    print(f"  - Test links: {summary['test_links']}")
    print(f"  - Performance cache size: {summary['performance_cache_size']}")
    print(f"  - Configuration sections: {len(summary['config'])}")
    print(f"  - Last updated: {summary['last_updated']}")
    print()
    
    # Display sample test content
    if all_tests:
        print("9. Sample generated test content:")
        print("-" * 60)
        sample_test = all_tests[0]
        print(f"Test Name: {sample_test.test_name}")
        print(f"Test Type: {sample_test.test_type}")
        print(f"Description: {sample_test.description}")
        print("\nTest Function (first 10 lines):")
        test_lines = sample_test.test_function.split('\n')[:10]
        for line in test_lines:
            print(f"  {line}")
        if len(sample_test.test_function.split('\n')) > 10:
            print("  ...")
        print("-" * 60)
        print()
    
    # Display sample performance validation
    if performance_validations:
        print("10. Sample performance validation:")
        print("-" * 60)
        sample_validation = performance_validations[0]
        print(f"Feature: {sample_validation.feature_name}")
        print(f"Validation Passed: {sample_validation.validation_passed}")
        print(f"Performance Gap: {sample_validation.performance_gap}")
        print(f"Recommendations:")
        for rec in sample_validation.recommendations:
            print(f"  • {rec}")
        print("-" * 60)
        print()
    
    # Display sample alert
    if all_alerts:
        print("11. Sample monitoring alert:")
        print("-" * 60)
        sample_alert = all_alerts[0]
        print(f"Alert ID: {sample_alert.alert_id}")
        print(f"Feature: {sample_alert.feature_name}")
        print(f"Type: {sample_alert.alert_type}")
        print(f"Severity: {sample_alert.severity}")
        print(f"Message: {sample_alert.message}")
        print(f"Recommended Actions:")
        for action in sample_alert.recommended_actions:
            print(f"  • {action}")
        print("-" * 60)
        print()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Key achievements:")
    print(f"✓ Generated {len(all_tests)} validation tests")
    print(f"✓ Found {len(all_links)} existing test links")
    print(f"✓ Completed {len(performance_validations)} performance validations")
    print(f"✓ Created {len(all_alerts)} monitoring alerts")
    print(f"✓ Written {len(written_files)} test files")
    print()
    print("The ValidationIntegrationSystem successfully:")
    print("• Links documentation claims to automated tests")
    print("• Generates validation tests based on thesis statements")
    print("• Validates feature performance against backtest data")
    print("• Creates monitoring alerts for performance degradation")
    print("• Integrates with existing test frameworks")
    print()


if __name__ == "__main__":
    try:
        demonstrate_validation_integration()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()