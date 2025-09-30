"""
Demonstration of Economic Rationale Generator

This script demonstrates the capabilities of the economic rationale generation framework
by generating thesis statements and economic rationale for various trading features.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from documentation.economic_rationale_generator import (
    EconomicRationaleGenerator,
    validate_economic_logic
)


def demo_feature_enhancement(generator, feature_name, category, description=""):
    """Demonstrate complete feature enhancement generation"""
    
    print(f"\n{'='*80}")
    print(f"FEATURE ENHANCEMENT DEMO: {feature_name}")
    print(f"Category: {category}")
    print(f"{'='*80}")
    
    # Generate complete enhancement
    enhancement = generator.generate_complete_enhancement(
        feature_name=feature_name,
        category=category,
        existing_content={"purpose": description} if description else {}
    )
    
    # Display thesis statement
    print(f"\n📋 THESIS STATEMENT:")
    print(f"Hypothesis: {enhancement.thesis_statement.hypothesis}")
    print(f"\nEconomic Basis: {enhancement.thesis_statement.economic_basis}")
    print(f"\nMarket Microstructure: {enhancement.thesis_statement.market_microstructure}")
    print(f"\nExpected Behavior: {enhancement.thesis_statement.expected_behavior}")
    print(f"\nFailure Modes: {', '.join(enhancement.thesis_statement.failure_modes)}")
    
    # Display economic rationale
    print(f"\n💰 ECONOMIC RATIONALE:")
    print(f"Supply Factors: {', '.join(enhancement.economic_rationale.supply_factors)}")
    print(f"Demand Factors: {', '.join(enhancement.economic_rationale.demand_factors)}")
    print(f"Market Inefficiency: {enhancement.economic_rationale.market_inefficiency}")
    print(f"Regime Dependency: {enhancement.economic_rationale.regime_dependency}")
    
    # Display chart explanation
    print(f"\nCHART EXPLANATION:")
    print(f"Visual Description: {enhancement.chart_explanation.visual_description}")
    print(f"Example Scenarios: {', '.join(enhancement.chart_explanation.example_scenarios[:2])}...")
    print(f"Chart Patterns: {', '.join(enhancement.chart_explanation.chart_patterns)}")
    
    # Display supply/demand classification
    print(f"\nSUPPLY/DEMAND CLASSIFICATION:")
    print(f"Primary Role: {enhancement.supply_demand_classification.primary_role.value}")
    print(f"Market Layer: {enhancement.supply_demand_classification.market_layer.value}")
    print(f"Time Horizon: {enhancement.supply_demand_classification.time_horizon.value}")
    print(f"Regime Sensitivity: {enhancement.supply_demand_classification.regime_sensitivity}")
    
    # Display validation criteria
    print(f"\nVALIDATION CRITERIA:")
    for i, criteria in enumerate(enhancement.validation_criteria[:3], 1):
        print(f"{i}. {criteria}")
    
    # Validate the thesis
    is_valid, issues = validate_economic_logic(enhancement.thesis_statement, "")
    print(f"\n🔍 VALIDATION RESULT: {'VALID' if is_valid else 'ISSUES FOUND'}")
    if issues:
        for issue in issues:
            print(f"   - {issue}")
    
    return enhancement


def demo_different_feature_types():
    """Demonstrate generation for different types of features"""
    
    print("ECONOMIC RATIONALE GENERATOR DEMONSTRATION")
    print("=" * 80)
    
    # Initialize generator
    generator = EconomicRationaleGenerator()
    
    # Demo different feature types
    features_to_demo = [
        ("q50", "Core Signal Features", "Primary quantile-based probability signal for trade direction"),
        ("vol_risk", "Risk & Volatility Features", "Variance-based risk assessment using volatility clustering"),
        ("regime_multiplier", "Regime & Market Features", "Regime-aware position scaling multiplier"),
        ("kelly_criterion", "Position Sizing Features", "Enhanced Kelly criterion for optimal position sizing"),
        ("momentum_hybrid", "Technical Features", "Momentum-based signal enhancement"),
        ("fear_greed_index", "Data Pipeline Features", "Sentiment indicator from market fear/greed levels")
    ]
    
    enhancements = []
    
    for feature_name, category, description in features_to_demo:
        enhancement = demo_feature_enhancement(generator, feature_name, category, description)
        enhancements.append(enhancement)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total features enhanced: {len(enhancements)}")
    
    # Count by primary role
    role_counts = {}
    for enhancement in enhancements:
        role = enhancement.supply_demand_classification.primary_role.value
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"\nFeatures by Supply/Demand Role:")
    for role, count in role_counts.items():
        print(f"  {role}: {count}")
    
    # Count by market layer
    layer_counts = {}
    for enhancement in enhancements:
        layer = enhancement.supply_demand_classification.market_layer.value
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    print(f"\nFeatures by Market Layer:")
    for layer, count in layer_counts.items():
        print(f"  {layer}: {count}")
    
    # Validation summary
    valid_count = 0
    for enhancement in enhancements:
        is_valid, _ = validate_economic_logic(enhancement.thesis_statement, "")
        if is_valid:
            valid_count += 1
    
    print(f"\nValidation Results:")
    print(f"  Valid thesis statements: {valid_count}/{len(enhancements)}")
    print(f"  Success rate: {valid_count/len(enhancements)*100:.1f}%")


def demo_template_system():
    """Demonstrate the template system for different feature types"""
    
    print(f"\n{'='*80}")
    print("🎨 TEMPLATE SYSTEM DEMONSTRATION")
    print(f"{'='*80}")
    
    generator = EconomicRationaleGenerator()
    
    print("\nAvailable Feature Type Templates:")
    for template_name in generator.supply_demand_templates.keys():
        print(f"  - {template_name}")
    
    print("\nFeature Type Mappings:")
    for feature_pattern, mapping in generator.feature_type_mappings.items():
        print(f"  {feature_pattern}: {mapping['template_type']} -> {mapping['primary_role'].value}")
    
    # Test classification for various feature names
    print(f"\n🔍 FEATURE NAME CLASSIFICATION EXAMPLES:")
    test_names = [
        "q50_signal", "volatility_risk", "regime_detector", "position_kelly",
        "momentum_trend", "sentiment_fear", "spread_analysis", "unknown_feature"
    ]
    
    for name in test_names:
        feature_type = generator._classify_feature_type(name)
        mapping = generator.feature_type_mappings.get(feature_type, {})
        role = mapping.get('primary_role', 'Unknown')
        print(f"  '{name}' -> {feature_type} -> {role.value if hasattr(role, 'value') else role}")


if __name__ == "__main__":
    try:
        # Run the main demonstration
        demo_different_feature_types()
        
        # Run template system demo
        demo_template_system()
        
        print(f"\n{'='*80}")
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("The Economic Rationale Generator is ready for use!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nERROR DURING DEMONSTRATION: {e}")
        import traceback
        traceback.print_exc()