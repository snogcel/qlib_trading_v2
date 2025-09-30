#!/usr/bin/env python3
"""
Demo script for the basic report generation system.

This script demonstrates the functionality of the report generation system
including summary reports, feature reports, HTML export, and templating.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testing.reporters.basic_reporter import BasicReporter
from testing.reporters.html_reporter import HTMLReporter
from testing.reporters.report_templates import ReportTemplates
from testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from testing.models.test_case import TestCase, TestType, TestPriority


def create_sample_test_results() -> list:
    """
    Create sample test results for demonstration.
    
    Returns:
        List of sample TestResult objects
    """
    print("Creating sample test results...")
    
    # Sample features and test types
    features = [
        "Q50", "vol_risk", "kelly_sizing", "regime_multiplier", 
        "fg_index", "btc_dom", "spread", "Q10", "Q90"
    ]
    
    test_types = [
        TestType.ECONOMIC_HYPOTHESIS,
        TestType.PERFORMANCE,
        TestType.FAILURE_MODE,
        TestType.IMPLEMENTATION,
        TestType.REGIME_DEPENDENCY
    ]
    
    priorities = [TestPriority.CRITICAL, TestPriority.HIGH, TestPriority.MEDIUM, TestPriority.LOW]
    statuses = [TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR, TestStatus.SKIPPED]
    confidences = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]
    
    results = []
    
    for i in range(50):  # Create 50 sample results
        # Create test case
        feature = random.choice(features)
        test_type = random.choice(test_types)
        priority = random.choice(priorities)
        
        test_case = TestCase(
            test_id=f"test_{i:03d}",
            feature_name=feature,
            test_type=test_type,
            priority=priority,
            description=f"Test {test_type.value} for {feature}",
            expected_result="Expected behavior validation",
            regime_context=random.choice(["bull", "bear", "sideways", None])
        )
        
        # Create test result
        status = random.choice(statuses)
        passed = status == TestStatus.PASSED
        confidence = random.choice(confidences)
        
        # Bias results to be more realistic
        if priority == TestPriority.CRITICAL:
            # Critical tests should mostly pass
            passed = random.random() > 0.1
            status = TestStatus.PASSED if passed else TestStatus.FAILED
        elif priority == TestPriority.HIGH:
            # High priority tests should usually pass
            passed = random.random() > 0.2
            status = TestStatus.PASSED if passed else TestStatus.FAILED
        
        result = TestResult(
            test_case=test_case,
            execution_id=f"exec_{i:03d}",
            status=status,
            execution_time=random.uniform(0.1, 5.0),
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 1440)),
            passed=passed,
            confidence=confidence,
            confidence_score=random.uniform(0.5, 1.0) if passed else random.uniform(0.0, 0.7),
            analysis=f"Analysis for {feature} {test_type.value}" if not passed else f"Successful validation of {feature}",
            error_message=f"Error in {feature} test" if not passed and status == TestStatus.ERROR else None,
            performance_metrics={
                "accuracy": random.uniform(0.6, 0.95),
                "precision": random.uniform(0.5, 0.9),
                "recall": random.uniform(0.4, 0.85)
            } if random.random() > 0.5 else {},
            recommendations=[
                f"Review {feature} implementation",
                f"Update {test_type.value} test criteria"
            ] if not passed else [],
            severity="high" if not passed and priority in [TestPriority.CRITICAL, TestPriority.HIGH] else "medium"
        )
        
        results.append(result)
    
    print(f"Created {len(results)} sample test results")
    return results


def demo_basic_reporter():
    """Demonstrate basic reporter functionality."""
    print("\n" + "="*60)
    print("🔍 DEMO: Basic Reporter Functionality")
    print("="*60)
    
    # Create sample results
    results = create_sample_test_results()
    
    # Initialize reporter
    reporter = BasicReporter()
    
    print("\n📋 Generating summary report...")
    summary_report = reporter.generate_summary_report(results)
    print("Summary report generated")
    
    # Save summary report
    output_dir = Path("test_results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "summary_report.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)
    print(f"💾 Summary report saved to: {summary_path}")
    
    # Show first few lines
    print("\n📄 Summary Report Preview:")
    print("-" * 40)
    lines = summary_report.split('\n')
    for line in lines[:15]:
        print(line)
    print("... (truncated)")
    
    print("\nGenerating coverage report...")
    coverage_report = reporter.generate_coverage_report(results)
    
    coverage_path = output_dir / "coverage_report.md"
    with open(coverage_path, 'w', encoding='utf-8') as f:
        f.write(coverage_report)
    print(f"💾 Coverage report saved to: {coverage_path}")
    
    # Generate feature-specific reports
    print("\nGenerating feature-specific reports...")
    features = list(set(r.test_case.feature_name for r in results))
    
    for feature in features[:3]:  # Demo first 3 features
        feature_report = reporter.generate_feature_report(feature, results)
        feature_path = output_dir / f"feature_report_{feature}.md"
        
        with open(feature_path, 'w', encoding='utf-8') as f:
            f.write(feature_report)
        print(f"💾 Feature report for {feature} saved to: {feature_path}")
    
    print("\nBasic reporter demo completed!")
    return results


def demo_html_reporter(results):
    """Demonstrate HTML reporter functionality."""
    print("\n" + "="*60)
    print("🌐 DEMO: HTML Reporter Functionality")
    print("="*60)
    
    # Initialize HTML reporter
    html_reporter = HTMLReporter()
    
    print("\n📋 Generating HTML summary report...")
    summary_report = html_reporter.generate_summary_report(results)
    
    # Export to HTML
    output_dir = Path("test_results/reports/html")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_path = output_dir / "summary_report.html"
    html_reporter.export_to_html(summary_report, html_path)
    print(f"💾 HTML summary report saved to: {html_path}")
    
    print("\nGenerating interactive dashboard...")
    dashboard_path = output_dir / "dashboard.html"
    html_reporter.generate_interactive_dashboard(results, dashboard_path)
    print(f"💾 Interactive dashboard saved to: {dashboard_path}")
    
    # Generate feature HTML reports
    print("\nGenerating HTML feature reports...")
    features = list(set(r.test_case.feature_name for r in results))
    
    for feature in features[:2]:  # Demo first 2 features
        feature_report = html_reporter.generate_feature_report(feature, results)
        feature_html_path = output_dir / f"feature_report_{feature}.html"
        
        html_reporter.export_to_html(feature_report, feature_html_path)
        print(f"💾 HTML feature report for {feature} saved to: {feature_html_path}")
    
    print("\nHTML reporter demo completed!")


def demo_report_templates(results):
    """Demonstrate report templates functionality."""
    print("\n" + "="*60)
    print("📝 DEMO: Report Templates Functionality")
    print("="*60)
    
    print("\n📋 Generating executive summary using template...")
    executive_summary = ReportTemplates.format_executive_summary(
        results, 
        report_period="Demo Period",
        system_version="1.0.0-demo"
    )
    
    output_dir = Path("test_results/reports/templates")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exec_path = output_dir / "executive_summary.md"
    with open(exec_path, 'w', encoding='utf-8') as f:
        f.write(executive_summary)
    print(f"💾 Executive summary saved to: {exec_path}")
    
    # Show preview
    print("\n📄 Executive Summary Preview:")
    print("-" * 40)
    lines = executive_summary.split('\n')
    for line in lines[:20]:
        print(line)
    print("... (truncated)")
    
    print("\nGenerating templated feature reports...")
    features = list(set(r.test_case.feature_name for r in results))
    
    for feature in features[:2]:  # Demo first 2 features
        feature_report = ReportTemplates.format_feature_report(
            feature_name=feature,
            results=results,
            feature_category="Core Signal" if feature in ["Q50", "Q10", "Q90"] else "Risk & Volatility",
            priority_level="High" if feature in ["Q50", "vol_risk"] else "Medium"
        )
        
        template_path = output_dir / f"templated_feature_{feature}.md"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(feature_report)
        print(f"💾 Templated feature report for {feature} saved to: {template_path}")
    
    print("\n🌐 Generating HTML wrapped reports...")
    html_exec_summary = ReportTemplates.wrap_in_html_template(
        executive_summary.replace('\n', '<br>\n'),
        "Executive Summary - Test Coverage"
    )
    
    html_exec_path = output_dir / "executive_summary.html"
    with open(html_exec_path, 'w', encoding='utf-8') as f:
        f.write(html_exec_summary)
    print(f"💾 HTML executive summary saved to: {html_exec_path}")
    
    print("\nReport templates demo completed!")


def demo_export_formats(results):
    """Demonstrate different export formats."""
    print("\n" + "="*60)
    print("📤 DEMO: Export Formats")
    print("="*60)
    
    reporter = BasicReporter()
    output_dir = Path("test_results/reports/exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a sample report
    summary_report = reporter.generate_summary_report(results)
    
    print("\n📄 Exporting to different formats...")
    
    # Text export
    text_path = output_dir / "report.txt"
    reporter.export_to_file(summary_report, text_path, 'text')
    print(f"💾 Text report saved to: {text_path}")
    
    # HTML export
    html_path = output_dir / "report.html"
    reporter.export_to_file(summary_report, html_path, 'html')
    print(f"💾 HTML report saved to: {html_path}")
    
    # JSON export (metadata)
    json_path = output_dir / "report_metadata.json"
    reporter.export_to_file(summary_report, json_path, 'json')
    print(f"💾 JSON metadata saved to: {json_path}")
    
    # Export results to JSON
    html_reporter = HTMLReporter()
    results_json_path = output_dir / "test_results.json"
    html_reporter.export_to_json(results, results_json_path)
    print(f"💾 Test results JSON saved to: {results_json_path}")
    
    print("\nExport formats demo completed!")


def demo_advanced_features(results):
    """Demonstrate advanced reporting features."""
    print("\n" + "="*60)
    print("DEMO: Advanced Reporting Features")
    print("="*60)
    
    reporter = BasicReporter()
    
    print("\nGenerating executive summary...")
    exec_summary = reporter.generate_executive_summary(results)
    
    output_dir = Path("test_results/reports/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exec_path = output_dir / "executive_summary.md"
    with open(exec_path, 'w', encoding='utf-8') as f:
        f.write(exec_summary)
    print(f"💾 Executive summary saved to: {exec_path}")
    
    print("\n🔍 Generating failure analysis...")
    failure_analysis = reporter.generate_failure_analysis(results)
    
    failure_path = output_dir / "failure_analysis.md"
    with open(failure_path, 'w', encoding='utf-8') as f:
        f.write(failure_analysis)
    print(f"💾 Failure analysis saved to: {failure_path}")
    
    print("\n📈 Generating regime analysis...")
    regime_analysis = reporter.generate_regime_analysis(results)
    
    regime_path = output_dir / "regime_analysis.md"
    with open(regime_path, 'w', encoding='utf-8') as f:
        f.write(regime_analysis)
    print(f"💾 Regime analysis saved to: {regime_path}")
    
    # Show some statistics
    print("\nReport Statistics:")
    print(f"- Total results analyzed: {len(results)}")
    print(f"- Unique features: {len(set(r.test_case.feature_name for r in results))}")
    print(f"- Test types covered: {len(set(r.test_case.test_type for r in results))}")
    print(f"- Success rate: {(sum(1 for r in results if r.passed) / len(results)) * 100:.1f}%")
    
    print("\nAdvanced features demo completed!")


def main():
    """Run the complete report generation demo."""
    print("Feature Test Coverage - Report Generation Demo")
    print("=" * 60)
    print("This demo showcases the basic report generation functionality")
    print("including summary reports, feature reports, HTML export, and templating.")
    print()
    
    try:
        # Demo basic reporter
        results = demo_basic_reporter()
        
        # Demo HTML reporter
        demo_html_reporter(results)
        
        # Demo report templates
        demo_report_templates(results)
        
        # Demo export formats
        demo_export_formats(results)
        
        # Demo advanced features
        demo_advanced_features(results)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📁 Generated reports can be found in:")
        print("   - test_results/reports/")
        print("   - test_results/reports/html/")
        print("   - test_results/reports/templates/")
        print("   - test_results/reports/exports/")
        print("   - test_results/reports/advanced/")
        print("\n💡 Open the HTML files in a web browser to see interactive reports!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())