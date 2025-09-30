#!/usr/bin/env python3
"""
Demo script for feature inventory generation functionality.

This script demonstrates the feature inventory generation capabilities
implemented in task 2.3 of the feature test coverage system.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testing.parsers.feature_template_parser import FeatureTemplateParser


def main():
    """Demonstrate feature inventory generation."""
    print(" Feature Inventory Generation Demo")
    print("=" * 50)
    
    # Initialize parser
    parser = FeatureTemplateParser()
    template_path = Path("docs/FEATURE_KNOWLEDGE_TEMPLATE.md")
    
    if not template_path.exists():
        print(f"Template file not found: {template_path}")
        return
    
    print(f"📄 Processing template: {template_path}")
    print()
    
    try:
        # Generate inventory
        print("Generating feature inventory...")
        inventory = parser.generate_feature_inventory(template_path)
        
        # Display summary
        summary = inventory.get_test_coverage_summary()
        
        print("Inventory Summary:")
        print(f"  • Total features: {summary['total_features']}")
        print(f"  • Critical features: {summary['critical_features']}")
        print(f"  • Categories: {summary['categories']}")
        print(f"  • Dependencies: {summary['dependencies']}")
        print(f"  • Completeness: {summary['coverage_completeness']:.1f}%")
        print()
        
        # Display categories
        print("📂 Feature Categories:")
        for name, category in inventory.categories.items():
            print(f"  {category.emoji} {name} ({category.priority} priority)")
            print(f"    Features: {len(category.features)}")
            if category.features:
                print(f"    Examples: {', '.join(category.features[:3])}")
                if len(category.features) > 3:
                    print(f"    ... and {len(category.features) - 3} more")
            print()
        
        # Display critical features
        critical_features = inventory.get_critical_features()
        print("⭐ Critical Features:")
        for feature in critical_features[:5]:  # Show first 5
            print(f"  • {feature.name} ({feature.category})")
            if feature.economic_hypothesis:
                hypothesis = feature.economic_hypothesis[:100]
                if len(feature.economic_hypothesis) > 100:
                    hypothesis += "..."
                print(f"    Hypothesis: {hypothesis}")
            print()
        
        if len(critical_features) > 5:
            print(f"  ... and {len(critical_features) - 5} more critical features")
        print()
        
        # Display test requirements distribution
        print("Test Requirements Distribution:")
        test_dist = summary['test_type_distribution']
        for test_type, count in sorted(test_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {test_type}: {count} features")
        print()
        
        # Display some dependencies
        if inventory.dependencies:
            print("Sample Dependencies:")
            for dep in inventory.dependencies[:5]:  # Show first 5
                print(f"  • {dep.source_feature} → {dep.target_feature}")
                print(f"    Type: {dep.dependency_type} ({dep.strength})")
                if dep.description:
                    print(f"    Description: {dep.description}")
                print()
            
            if len(inventory.dependencies) > 5:
                print(f"  ... and {len(inventory.dependencies) - 5} more dependencies")
        
        # Display validation results
        validation = inventory.validation_summary
        print("Validation Results:")
        print(f"  • Completeness score: {validation['completeness_score']:.1f}%")
        print(f"  • Consistency issues: {len(validation['consistency_issues'])}")
        print(f"  • Missing data items: {len(validation['missing_data'])}")
        print(f"  • Warnings: {len(validation['warnings'])}")
        
        if validation['consistency_issues']:
            print("\n Consistency Issues:")
            for issue in validation['consistency_issues'][:3]:
                print(f"    • {issue}")
            if len(validation['consistency_issues']) > 3:
                print(f"    ... and {len(validation['consistency_issues']) - 3} more")
        
        print("\nFeature inventory generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating inventory: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())