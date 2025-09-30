"""
Unit tests for the ReportTemplates class.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from testing.reporters.report_templates import ReportTemplates
from testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from testing.models.test_case import TestCase, TestType, TestPriority


class TestReportTemplates(unittest.TestCase):
    """Test cases for ReportTemplates functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = self._create_sample_results()
    
    def _create_sample_results(self):
        """Create sample test results for testing."""
        results = []
        
        # Create a passed test
        passed_test_case = TestCase(
            test_id="test_001",
            feature_name="Q50",
            test_type=TestType.ECONOMIC_HYPOTHESIS,
            priority=TestPriority.HIGH,
            description="Test Q50 economic hypothesis",
            expected_result="Directional bias validation"
        )
        
        passed_result = TestResult(
            test_case=passed_test_case,
            execution_id="exec_001",
            status=TestStatus.PASSED,
            execution_time=1.5,
            passed=True,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.95,
            analysis="Q50 shows expected directional bias",
            performance_metrics={"accuracy": 0.85, "precision": 0.80}
        )
        results.append(passed_result)
        
        # Create a failed critical test
        failed_test_case = TestCase(
            test_id="test_002",
            feature_name="vol_risk",
            test_type=TestType.PERFORMANCE,
            priority=TestPriority.CRITICAL,
            description="Test vol_risk performance",
            expected_result="Variance calculation validation"
        )
        
        failed_result = TestResult(
            test_case=failed_test_case,
            execution_id="exec_002",
            status=TestStatus.FAILED,
            execution_time=2.1,
            passed=False,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.60,
            analysis="vol_risk calculation deviates from expected variance",
            error_message="Variance calculation error",
            recommendations=["Review vol_risk implementation", "Update variance formula"],
            severity="high"
        )
        results.append(failed_result)
        
        return results
    
    def test_get_executive_summary_template(self):
        """Test executive summary template retrieval."""
        template = ReportTemplates.get_executive_summary_template()
        
        # Check that template contains expected placeholders
        self.assertIn("{timestamp}", template)
        self.assertIn("{success_rate}", template)
        self.assertIn("{total_tests}", template)
        self.assertIn("{critical_failures}", template)
        self.assertIn("{critical_issues_section}", template)
        self.assertIn("{recommendations_section}", template)
        
        # Check that template has expected structure
        self.assertIn("# Executive Summary", template)
        self.assertIn("## 🎯 Key Metrics", template)
        self.assertIn("## Summary Statistics", template)
        self.assertIn("## 🚨 Critical Issues", template)
    
    def test_get_detailed_report_template(self):
        """Test detailed report template retrieval."""
        template = ReportTemplates.get_detailed_report_template()
        
        # Check that template contains expected placeholders
        self.assertIn("{timestamp}", template)
        self.assertIn("{execution_summary}", template)
        self.assertIn("{feature_analysis}", template)
        self.assertIn("{failure_analysis}", template)
        
        # Check that template has expected structure
        self.assertIn("# Detailed Test Coverage Report", template)
        self.assertIn("## 📋 Test Execution Summary", template)
        self.assertIn("## 🔍 Feature Analysis", template)
        self.assertIn("## ⚠️ Failure Analysis", template)
    
    def test_get_feature_report_template(self):
        """Test feature report template retrieval."""
        template = ReportTemplates.get_feature_report_template()
        
        # Check that template contains expected placeholders
        self.assertIn("{feature_name}", template)
        self.assertIn("{timestamp}", template)
        self.assertIn("{feature_overview}", template)
        self.assertIn("{test_type_table}", template)
        
        # Check that template has expected structure
        self.assertIn("# Feature Report: {feature_name}", template)
        self.assertIn("## Feature Overview", template)
        self.assertIn("## Test Results Summary", template)
        self.assertIn("## 💡 Recommendations", template)
    
    def test_get_coverage_matrix_template(self):
        """Test coverage matrix template retrieval."""
        template = ReportTemplates.get_coverage_matrix_template()
        
        # Check that template contains expected placeholders
        self.assertIn("{timestamp}", template)
        self.assertIn("{coverage_overview}", template)
        self.assertIn("{coverage_matrix}", template)
        
        # Check that template has expected structure
        self.assertIn("# Test Coverage Matrix", template)
        self.assertIn("## Coverage Overview", template)
        self.assertIn("## 🎯 Feature Coverage Matrix", template)
    
    def test_format_executive_summary_empty_results(self):
        """Test executive summary formatting with empty results."""
        summary = ReportTemplates.format_executive_summary([])
        self.assertEqual(summary, "No test results available for executive summary.")
    
    def test_format_executive_summary_with_results(self):
        """Test executive summary formatting with sample results."""
        summary = ReportTemplates.format_executive_summary(
            self.sample_results,
            report_period="Test Period",
            system_version="1.0.0-test"
        )
        
        # Check that summary contains expected content
        self.assertIn("# Executive Summary", summary)
        self.assertIn("Test Period", summary)
        self.assertIn("Overall Success Rate: 50.0%", summary)
        self.assertIn("Total Tests Executed | 2", summary)
        self.assertIn("Critical Failures | 1", summary)
        
        # Check status indicators
        self.assertIn("❌", summary)  # Should show critical status due to low success rate
        
        # Check sections are populated
        self.assertIn("## 🚨 Critical Issues", summary)
        self.assertIn("## 📈 Recommendations", summary)
        self.assertIn("## 📋 Next Steps", summary)
    
    def test_format_feature_report_no_results(self):
        """Test feature report formatting with no results for feature."""
        report = ReportTemplates.format_feature_report("nonexistent_feature", self.sample_results)
        self.assertIn("No test results found for feature: nonexistent_feature", report)
    
    def test_format_feature_report_with_results(self):
        """Test feature report formatting with results for specific feature."""
        report = ReportTemplates.format_feature_report(
            "Q50",
            self.sample_results,
            feature_category="Core Signal",
            priority_level="High"
        )
        
        # Check that report contains expected content
        self.assertIn("# Feature Report: Q50", report)
        self.assertIn("Core Signal", report)
        self.assertIn("High", report)
        
        # Check sections are populated
        self.assertIn("## Feature Overview", report)
        self.assertIn("## Test Results Summary", report)
        self.assertIn("## Passed Tests", report)
        self.assertIn("## 💡 Recommendations", report)
        
        # Check statistics
        self.assertIn("Total Tests: 1", report)
        self.assertIn("Success Rate: 100.0%", report)
    
    def test_format_critical_issues_no_issues(self):
        """Test critical issues formatting with no critical failures."""
        passed_results = [self.sample_results[0]]  # Only passed test
        issues = ReportTemplates._format_critical_issues(passed_results)
        
        self.assertIn("**No critical issues found.**", issues)
    
    def test_format_critical_issues_with_issues(self):
        """Test critical issues formatting with critical failures."""
        issues = ReportTemplates._format_critical_issues(self.sample_results)
        
        self.assertIn("**Critical issues requiring immediate attention:**", issues)
        self.assertIn("**vol_risk**", issues)
        self.assertIn("Variance calculation error", issues)
    
    def test_format_recommendations_no_failures(self):
        """Test recommendations formatting with no failures."""
        passed_results = [self.sample_results[0]]  # Only passed test
        recommendations = ReportTemplates._format_recommendations(passed_results)
        
        self.assertIn("**No immediate actions required.**", recommendations)
    
    def test_format_recommendations_with_failures(self):
        """Test recommendations formatting with failures."""
        recommendations = ReportTemplates._format_recommendations(self.sample_results)
        
        self.assertIn("**Key recommendations for improvement:**", recommendations)
        self.assertIn("Review vol_risk implementation", recommendations)
        self.assertIn("Update variance formula", recommendations)
    
    def test_format_next_steps_with_critical_failures(self):
        """Test next steps formatting with critical failures."""
        steps = ReportTemplates._format_next_steps(self.sample_results)
        
        self.assertIn("**IMMEDIATE:** Address all critical test failures", steps)
        self.assertIn("**URGENT:** Conduct root cause analysis", steps)
    
    def test_format_next_steps_no_failures(self):
        """Test next steps formatting with no failures."""
        passed_results = [self.sample_results[0]]  # Only passed test
        steps = ReportTemplates._format_next_steps(passed_results)
        
        self.assertIn("**MAINTAIN:** Continue monitoring test results", steps)
        self.assertIn("**IMPROVE:** Consider adding additional test coverage", steps)
    
    def test_format_feature_overview(self):
        """Test feature overview formatting."""
        overview = ReportTemplates._format_feature_overview([self.sample_results[0]])
        
        self.assertIn("**Total Tests:** 1", overview)
        self.assertIn("**Success Rate:** 100.0%", overview)
        self.assertIn("**Average Confidence:** 0.95", overview)
        self.assertIn("**Average Execution Time:** 1.50s", overview)
    
    def test_format_test_type_table(self):
        """Test test type table formatting."""
        table = ReportTemplates._format_test_type_table(self.sample_results)
        
        # Check table structure
        self.assertIn("Economic Hypothesis", table)
        self.assertIn("Performance Characteristics", table)
        
        # Check that it contains pipe characters for table formatting
        self.assertIn("|", table)
    
    def test_format_passed_tests(self):
        """Test passed tests formatting."""
        passed_section = ReportTemplates._format_passed_tests([self.sample_results[0]])
        
        self.assertIn("### Economic Hypothesis", passed_section)
        self.assertIn("**Confidence:** high", passed_section)
        self.assertIn("**Execution Time:** 1.50s", passed_section)
    
    def test_format_failed_tests(self):
        """Test failed tests formatting."""
        failed_section = ReportTemplates._format_failed_tests([self.sample_results[1]])
        
        self.assertIn("### Performance Characteristics", failed_section)
        self.assertIn("**Status:** failed", failed_section)
        self.assertIn("**Priority:** critical", failed_section)
        self.assertIn("**Error:** Variance calculation error", failed_section)
    
    def test_format_failed_tests_no_failures(self):
        """Test failed tests formatting with no failures."""
        failed_section = ReportTemplates._format_failed_tests([self.sample_results[0]])
        
        self.assertEqual(failed_section, "All tests passed for this feature.")
    
    def test_format_performance_analysis(self):
        """Test performance analysis formatting."""
        analysis = ReportTemplates._format_performance_analysis([self.sample_results[0]])
        
        self.assertIn("### Accuracy", analysis)
        self.assertIn("### Precision", analysis)
        self.assertIn("**Average:** 0.850", analysis)
        self.assertIn("**Range:** 0.850 - 0.850", analysis)
    
    def test_format_performance_analysis_no_metrics(self):
        """Test performance analysis formatting with no metrics."""
        # Create result without performance metrics
        result_without_metrics = TestResult(
            test_case=self.sample_results[0].test_case,
            execution_id="exec_test",
            status=TestStatus.PASSED,
            execution_time=1.0,
            passed=True
        )
        
        analysis = ReportTemplates._format_performance_analysis([result_without_metrics])
        
        self.assertEqual(analysis, "No performance metrics available for this feature.")
    
    def test_format_hypothesis_validation(self):
        """Test hypothesis validation formatting."""
        validation = ReportTemplates._format_hypothesis_validation([self.sample_results[0]])
        
        self.assertIn("**Hypothesis Test:**", validation)
        self.assertIn("**Result:** Hypothesis validated with high confidence", validation)
    
    def test_format_hypothesis_validation_no_tests(self):
        """Test hypothesis validation formatting with no hypothesis tests."""
        # Use performance test instead of hypothesis test
        validation = ReportTemplates._format_hypothesis_validation([self.sample_results[1]])
        
        self.assertEqual(validation, "No economic hypothesis tests found for this feature.")
    
    def test_format_risk_assessment(self):
        """Test risk assessment formatting."""
        assessment = ReportTemplates._format_risk_assessment(self.sample_results)
        
        self.assertIn("**High Risk Issues:**", assessment)
        self.assertIn("**Medium Risk Issues:**", assessment)
        self.assertIn("**Low Risk Issues:**", assessment)
        self.assertIn("**Overall Risk Level:**", assessment)
    
    def test_format_feature_recommendations_no_failures(self):
        """Test feature recommendations formatting with no failures."""
        recommendations = ReportTemplates._format_feature_recommendations([self.sample_results[0]])
        
        self.assertIn("**No specific recommendations.**", recommendations)
    
    def test_format_feature_recommendations_with_failures(self):
        """Test feature recommendations formatting with failures."""
        recommendations = ReportTemplates._format_feature_recommendations([self.sample_results[1]])
        
        self.assertIn("Review vol_risk implementation", recommendations)
        self.assertIn("Update variance formula", recommendations)
    
    def test_get_html_template_styles(self):
        """Test HTML template styles retrieval."""
        styles = ReportTemplates.get_html_template_styles()
        
        # Check that styles contain expected CSS
        self.assertIn("<style>", styles)
        self.assertIn("body {", styles)
        self.assertIn(".report-container", styles)
        self.assertIn(".success", styles)
        self.assertIn(".warning", styles)
        self.assertIn(".error", styles)
        self.assertIn("</style>", styles)
    
    def test_wrap_in_html_template(self):
        """Test HTML template wrapping."""
        content = "<h1>Test Content</h1><p>This is test content.</p>"
        html = ReportTemplates.wrap_in_html_template(content, "Test Report")
        
        # Check HTML structure
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<title>Test Report</title>", html)
        self.assertIn("<div class=\"report-container\">", html)
        self.assertIn(content, html)
        self.assertIn("Generated by Feature Test Coverage System", html)
        
        # Check that styles are included
        self.assertIn("<style>", html)
        self.assertIn("</style>", html)
    
    def test_wrap_in_html_template_default_title(self):
        """Test HTML template wrapping with default title."""
        content = "<h1>Test Content</h1>"
        html = ReportTemplates.wrap_in_html_template(content)
        
        self.assertIn("<title>Test Coverage Report</title>", html)


if __name__ == '__main__':
    unittest.main()