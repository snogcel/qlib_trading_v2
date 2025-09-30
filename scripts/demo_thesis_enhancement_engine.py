#!/usr/bin/env python3
"""
Demo script for ThesisEnhancementEngine

This script demonstrates the core functionality of the ThesisEnhancementEngine,
including feature parsing, enhancement generation, validation, and content preservation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.documentation.thesis_enhancement_engine import ThesisEnhancementEngine
from src.documentation.document_protection import DocumentProtectionSystem


def print_section(title: str, content: str = ""):
    """Print a formatted section"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if content:
        print(content)


def print_subsection(title: str, content: str = ""):
    """Print a formatted subsection"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")
    if content:
        print(content)


def demo_initialization():
    """Demo engine initialization and basic functionality"""
    print_section("THESIS ENHANCEMENT ENGINE DEMO")
    
    try:
        # Initialize engine
        print("Initializing ThesisEnhancementEngine...")
        engine = ThesisEnhancementEngine()
        print("Engine initialized successfully")
        
        # Show basic info
        print(f"📄 Feature documentation: {engine.feature_doc_path}")
        print(f"📋 Principles document: {engine.principles_path}")
        print(f"🛡️  Protection system: {type(engine.protection_system).__name__}")
        print(f"🧠 Rationale generator: {type(engine.rationale_generator).__name__}")
        
        return engine
        
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return None


def demo_feature_parsing(engine):
    """Demo feature parsing capabilities"""
    print_section("FEATURE PARSING DEMONSTRATION")
    
    try:
        # List categories
        categories = engine.list_categories()
        print(f"Found {len(categories)} feature categories:")
        for i, category in enumerate(categories, 1):
            print(f"  {i}. {category}")
        
        # Show detailed info for first few categories
        print_subsection("Category Details")
        for category_name in categories[:3]:  # Show first 3 categories
            info = engine.get_category_info(category_name)
            if info:
                print(f"\n📁 {info['name']}")
                print(f"   Features: {info['feature_count']}")
                print(f"   Enhanced: {'✅' if info['has_enhancements'] else ''}")
                print(f"   Feature list: {', '.join(info['features'][:3])}{'...' if len(info['features']) > 3 else ''}")
        
        return True
        
    except Exception as e:
        print(f"Feature parsing failed: {e}")
        return False


def demo_enhancement_status(engine):
    """Demo enhancement status tracking"""
    print_section("ENHANCEMENT STATUS")
    
    try:
        status = engine.get_enhancement_status()
        
        print(f"📈 Enhancement Progress:")
        print(f"   Total Categories: {status['total_categories']}")
        print(f"   Enhanced Categories: {status['enhanced_categories']}")
        print(f"   Completion: {status['enhancement_percentage']:.1f}%")
        
        print_subsection("Category Status")
        for cat_info in status['categories']:
            status_icon = "✅" if cat_info['enhanced'] else ""
            print(f"   {status_icon} {cat_info['name']} ({cat_info['feature_count']} features)")
        
        if status['enhancement_history']:
            print_subsection("Enhancement History")
            for entry in status['enhancement_history'][-3:]:  # Show last 3 entries
                print(f"   📅 {entry['timestamp'][:19]}: {entry['category']}")
                print(f"      Enhancements: {len(entry['enhancements'])}")
        
        return status
        
    except Exception as e:
        print(f"Status check failed: {e}")
        return None


def demo_enhancement_generation(engine):
    """Demo enhancement generation for a category"""
    print_section("ENHANCEMENT GENERATION DEMO")
    
    try:
        # Find a category to enhance (prefer one without enhancements)
        categories = engine.list_categories()
        target_category = None
        
        for category_name in categories:
            info = engine.get_category_info(category_name)
            if info and not info['has_enhancements'] and info['feature_count'] > 0:
                target_category = category_name
                break
        
        if not target_category:
            # If all are enhanced, pick the first one
            target_category = categories[0] if categories else None
        
        if not target_category:
            print("No suitable category found for enhancement demo")
            return None
        
        print(f"Enhancing category: {target_category}")
        
        # Generate enhancement
        print("Generating enhancements...")
        result = engine.enhance_feature_category(target_category, preserve_existing=True)
        
        if result.success:
            print("Enhancement generation successful!")
            
            print_subsection("Enhancement Summary")
            print(f"Category: {result.category_name}")
            print(f"Enhancements Applied: {len(result.enhancements_applied)}")
            for enhancement in result.enhancements_applied:
                print(f"  • {enhancement}")
            
            if result.warnings:
                print_subsection("Warnings")
                for warning in result.warnings:
                    print(f"  ⚠️  {warning}")
            
            print_subsection("Validation Results")
            for validation in result.validation_results:
                print(f"  📋 {validation}")
            
            # Show a sample of the enhanced content
            print_subsection("Enhanced Content Sample (first 500 chars)")
            sample = result.enhanced_content[:500]
            print(f"```\n{sample}{'...' if len(result.enhanced_content) > 500 else ''}\n```")
            
            return result
        else:
            print(f"Enhancement failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"Enhancement generation failed: {e}")
        return None


def demo_validation(engine):
    """Demo enhancement validation"""
    print_section("ENHANCEMENT VALIDATION DEMO")
    
    try:
        # Test validation with sample content
        print("🔍 Testing validation with sample enhanced content...")
        
        sample_enhanced_content = """
## Core Signal Features

### Q50 (Primary Signal)
- **Type**: Quantile-based probability
- **Purpose**: Primary directional signal based on 50th percentile probability

#### Economic Thesis
**Hypothesis**: Q50 should predict returns because it captures asymmetric probability distributions that indicate supply/demand imbalances before they fully manifest in price movements.

**Economic Basis**: Supply/Demand Analysis: Detects supply exhaustion when Q50 indicates sellers are becoming scarce at current price levels. Conversely, identifies demand accumulation when Q50 shows buyers are willing to pay higher prices.

**Market Microstructure**: Works by measuring probability distributions of price movements, capturing asymmetric order flow.

**Expected Behavior**: Expect Q50 to show higher values before upward price movements and lower values before downward movements, with asymmetric payoff profiles.

#### Supply/Demand Analysis
**Supply Factors**:
- Detects supply exhaustion when Q50 indicates sellers are becoming scarce at current price levels

**Demand Factors**:
- Identifies demand accumulation when Q50 shows buyers are willing to pay higher prices

**Market Inefficiency**: Exploits temporary imbalances between buyer and seller conviction levels

#### Chart Explainability
**Visual Description**: On price charts, Q50 appears as probability levels that anticipate price movements - high values before rallies, low values before declines.

**Example Scenarios**:
- Q50 reaches extreme high values just before major breakouts
- Q50 shows divergence with price at market tops/bottoms

#### Validation Criteria
- Sharpe ratio improvement > 0.1 when using Q50 signals
- Win rate asymmetry: higher payoffs on Q50 > 0.6 trades
"""
        
        validation_result = engine.validate_enhancement(sample_enhanced_content, "Core Signal Features")
        
        print_subsection("Validation Results")
        print(f"Valid: {validation_result.is_valid}")
        print(f"Alignment Score: {validation_result.alignment_score:.3f}")
        
        if validation_result.principle_violations:
            print_subsection("Principle Violations")
            for violation in validation_result.principle_violations:
                print(f"  ⚠️  {violation}")
        
        if validation_result.missing_elements:
            print_subsection("Missing Elements")
            for missing in validation_result.missing_elements:
                print(f"  {missing}")
        
        if validation_result.recommendations:
            print_subsection("Recommendations")
            for rec in validation_result.recommendations:
                print(f"  💡 {rec}")
        
        return validation_result
        
    except Exception as e:
        print(f"Validation demo failed: {e}")
        return None


def demo_protection_integration(engine):
    """Demo protection system integration"""
    print_section("PROTECTION SYSTEM INTEGRATION")
    
    try:
        protection = engine.protection_system
        
        # Show protection status
        print("🛡️  Protection System Status:")
        status = protection.get_protection_status(str(engine.feature_doc_path))
        
        print(f"   Protected: {'✅' if status['is_protected'] else ''}")
        print(f"   Backups: {status['backup_count']}")
        print(f"   Rollback Points: {status['rollback_points']}")
        print(f"   Version Tracking: {'✅' if status['version_tracking'] else ''}")
        
        if status['last_backup']:
            print(f"   Last Backup: {status['last_backup'][:19]}")
        
        # Show recent backups
        backups = protection.list_backups(str(engine.feature_doc_path))
        if backups:
            print_subsection("Recent Backups")
            for backup in backups[-3:]:  # Show last 3 backups
                print(f"   📁 {backup['timestamp']}: {backup.get('description', 'No description')}")
        
        # Show rollback points
        rollback_points = protection.list_rollback_points(str(engine.feature_doc_path))
        if rollback_points:
            print_subsection("Rollback Points")
            for point in rollback_points[-3:]:  # Show last 3 rollback points
                print(f"   {point.timestamp}: {point.description}")
        
        return True
        
    except Exception as e:
        print(f"Protection system demo failed: {e}")
        return False


def demo_error_handling():
    """Demo error handling capabilities"""
    print_section("ERROR HANDLING DEMO")
    
    try:
        print("Testing error handling scenarios...")
        
        # Test with non-existent files
        print_subsection("Non-existent Files")
        try:
            engine = ThesisEnhancementEngine(
                "non_existent_feature.md",
                "non_existent_principles.md"
            )
            print("Should have failed but didn't")
        except Exception as e:
            print(f"Correctly handled missing files: {type(e).__name__}")
        
        # Test with valid engine but invalid category
        print_subsection("Invalid Category Enhancement")
        engine = ThesisEnhancementEngine()
        result = engine.enhance_feature_category("Non-existent Category")
        
        if not result.success:
            print(f"Correctly handled invalid category: {result.error_message}")
        else:
            print("Should have failed for invalid category")
        
        return True
        
    except Exception as e:
        print(f"Error handling demo failed: {e}")
        return False


def main():
    """Main demo function"""
    print("Starting ThesisEnhancementEngine Demo")
    
    # Initialize engine
    engine = demo_initialization()
    if not engine:
        print("Demo aborted due to initialization failure")
        return
    
    # Run demo sections
    demos = [
        ("Feature Parsing", lambda: demo_feature_parsing(engine)),
        ("Enhancement Status", lambda: demo_enhancement_status(engine)),
        ("Enhancement Generation", lambda: demo_enhancement_generation(engine)),
        ("Validation", lambda: demo_validation(engine)),
        ("Protection Integration", lambda: demo_protection_integration(engine)),
        ("Error Handling", demo_error_handling)
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        try:
            print(f"\nRunning {demo_name} demo...")
            result = demo_func()
            results[demo_name] = result is not None and result is not False
            if results[demo_name]:
                print(f"{demo_name} demo completed successfully")
            else:
                print(f"⚠️  {demo_name} demo completed with issues")
        except Exception as e:
            print(f"{demo_name} demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print_section("DEMO SUMMARY")
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"Demo Results: {successful}/{total} successful")
    for demo_name, success in results.items():
        status = "✅" if success else ""
        print(f"   {status} {demo_name}")
    
    if successful == total:
        print("\nAll demos completed successfully!")
        print("The ThesisEnhancementEngine is working correctly.")
    else:
        print(f"\n⚠️  {total - successful} demo(s) had issues.")
        print("Check the output above for details.")
    
    print("\n📚 Next Steps:")
    print("1. Review the enhanced content samples")
    print("2. Run the unit tests: python -m pytest tests/unit/test_thesis_enhancement_engine.py")
    print("3. Use the engine to enhance your feature documentation")
    print("4. Integrate with your existing workflow")


if __name__ == "__main__":
    main()