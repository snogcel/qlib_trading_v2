"""
Unit tests for the BasicReporter class.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from testing.reporters.basic_reporter import BasicReporter
from testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from testing.models.test_case import TestCase, TestType, TestPriority


class TestBasicReporter(unittest.TestCase):
    """Test cases for BasicReporter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = BasicReporter()
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
        
        # Create a failed test
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
        
        # Create an error test
        error_test_case = TestCase(
            test_id="test_003",
            feature_name="kelly_sizing",
            test_type=TestType.IMPLEMENTATION,
            priority=TestPriority.MEDIUM,
            description="Test kelly_sizing implementation",
            expected_result="Risk-adjusted sizing validation"
        )
        
        error_result = TestResult(
            test_case=error_test_case,
            execution_id="exec_003",
            status=TestStatus.ERROR,
            execution_time=0.5,
            passed=False,
            confidence=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.30,
            analysis="Implementation error in kelly_sizing",
            error_message="Division by zero in kelly calculation",
            recommendations=["Fix division by zero error"],
            severity="medium"
        )
        results.append(error_result)
        
        return results
    
    def test_init(self):
        """Test BasicReporter initialization."""
        reporter = BasicReporter()
        self.assertIsInstance(reporter.report_timestamp, datetime)
    
    def test_generate_summary_report_empty_results(self):
        """Test summary report generation with empty results."""
        report = self.reporter.generate_summary_report([])
        self.assertEqual(report, "No test results to report.")
    
    def test_generate_summary_report_with_results(self):
        """Test summary report generation with sample results."""
        report = self.reporter.generate_summary_report(self.sample_results)
        
        # Check that report contains expected sections
        self.assertIn("# Test Coverage Summary Report", report)
        self.assertIn("## Overall Statistics", report)
        self.assertIn("## Test Status Breakdown", report)
        self.assertIn("## Priority Breakdown", report)
        self.assertIn("## Critical Failures", report)
        self.assertIn("## Confidence Analysis", report)
        
        # Check statistics
        self.assertIn("Total Tests: 3", report)
        self.assertIn("Passed: 1", report)
        self.assertIn("Failed: 2", report)
        self.assertIn("Success Rate: 33.3%", report)
    
    def test_generate_feature_report_no_results(self):
        """Test feature report generation with no results for feature."""
        report = self.reporter.generate_feature_report("nonexistent_feature", self.sample_results)
        self.assertIn("No test results found for feature: nonexistent_feature", report)
    
    def test_generate_feature_report_with_results(self):
        """Test feature report generation with results for specific feature."""
        report = self.reporter.generate_feature_report("Q50", self.sample_results)
        
        # Check that report contains expected sections
        self.assertIn("# Feature Report: Q50", report)
        self.assertIn("## Feature Overview", report)
        self.assertIn("## Test Type Results", report)
        
        # Check feature statistics
        self.assertIn("Total Tests: 1", report)
        self.assertIn("Passed: 1", report)
        self.assertIn("Success Rate: 100.0%", report)
    
    def test_generate_coverage_report_empty_results(self):
        """Test coverage report generation with empty results."""
        report = self.reporter.generate_coverage_report([])
        self.assertIn("No test results available for coverage analysis.", report)
    
    def test_generate_coverage_report_with_results(self):
        """Test coverage report generation with sample results."""
        report = self.reporter.generate_coverage_report(self.sample_results)
        
        # Check that report contains expected sections
        self.assertIn("# Test Coverage Analysis", report)
        self.assertIn("## Coverage Overview", report)
        self.assertIn("## Feature Coverage Details", report)
        self.assertIn("## Test Type Coverage", report)
        self.assertIn("## Priority Coverage", report)
        
        # Check coverage statistics
        self.assertIn("Total Features Tested: 3", report)
        self.assertIn("Fully Covered Features: 1", report)  # Only Q50 is fully covered
    
    def test_export_to_html(self):
        """Test HTML export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            report_content = "# Test Report\n\nThis is a test report."
            
            self.reporter.export_to_html(report_content, output_path)
            
            # Check that file was created
            self.assertTrue(output_path.exists())
            
            # Check HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.assertIn("<!DOCTYPE html>", html_content)
            self.assertIn("<title>Test Coverage Report</title>", html_content)
            self.assertIn("<h1>Test Report</h1>", html_content)
            self.assertIn("This is a test report.", html_content)
    
    def test_markdown_to_html_conversion(self):
        """Test markdown to HTML conversion."""
        markdown_content = """# Header 1
## Header 2
### Header 3

- List item 1
- List item 2
  - Nested item

1. Numbered item 1
2. Numbered item 2

Regular paragraph with **bold** text.

```python
code block
```"""
        
        html_content = self.reporter._markdown_to_html(markdown_content)
        
        # Check header conversions
        self.assertIn("<h1>Header 1</h1>", html_content)
        self.assertIn("<h2>Header 2</h2>", html_content)
        self.assertIn("<h3>Header 3</h3>", html_content)
        
        # Check list conversions
        self.assertIn("<ul>", html_content)
        self.assertIn("<li>List item 1</li>", html_content)
        self.assertIn("<ol>", html_content)
        self.assertIn("<li>Numbered item 1</li>", html_content)
        
        # Check code block
        self.assertIn("<pre>", html_content)
        self.assertIn("code block", html_content)
    
    def test_get_supported_formats(self):
        """Test supported formats list."""
        formats = self.reporter.get_supported_formats()
        expected_formats = ['text', 'html', 'json']
        
        self.assertEqual(formats, expected_formats)
    
    def test_export_to_file_text(self):
        """Test export to text file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.txt"
            report_content = "Test report content"
            
            self.reporter.export_to_file(report_content, output_path, 'text')
            
            self.assertTrue(output_path.exists())
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.assertEqual(content, report_content)
    
    def test_export_to_file_json(self):
        """Test export to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.json"
            report_content = "Test report content"
            
            self.reporter.export_to_file(report_content, output_path, 'json')
            
            self.assertTrue(output_path.exists())
            
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            self.assertIn('report_type', json_data)
            self.assertIn('timestamp', json_data)
            self.assertIn('content', json_data)
            self.assertEqual(json_data['content'], report_content)
    
    def test_generate_executive_summary_empty_results(self):
        """Test executive summary generation with empty results."""
        summary = self.reporter.generate_executive_summary([])
        self.assertEqual(summary, "No test results available.")
    
    def test_generate_executive_summary_with_results(self):
        """Test executive summary generation with sample results."""
        summary = self.reporter.generate_executive_summary(self.sample_results)
        
        # Check that summary contains expected sections
        self.assertIn("# Executive Summary", summary)
        self.assertIn("## Key Metrics", summary)
        self.assertIn("## Overall Status:", summary)
        
        # Check metrics
        self.assertIn("Overall Success Rate: 33.3%", summary)
        self.assertIn("Critical Failures: 1", summary)  # vol_risk is critical and failed
        self.assertIn("High Priority Failures: 0", summary)  # Q50 passed, so no high priority failures
    
    def test_generate_executive_summary_status_assessment(self):
        """Test executive summary status assessment logic."""
        # Create results with high success rate
        high_success_results = [self.sample_results[0]]  # Only the passed test
        summary = self.reporter.generate_executive_summary(high_success_results)
        self.assertIn("🟢 EXCELLENT", summary)
        
        # Test with all sample results (low success rate)
        summary = self.reporter.generate_executive_summary(self.sample_results)
        self.assertIn("🔴 CRITICAL", summary)
    
    def test_regime_analysis(self):
        """Test regime analysis functionality."""
        # Add regime context to sample results
        self.sample_results[0].test_case.regime_context = "bull"
        self.sample_results[1].test_case.regime_context = "bear"
        self.sample_results[2].test_case.regime_context = "bull"
        
        analysis = self.reporter.generate_regime_analysis(self.sample_results)
        
        self.assertIn("# Market Regime Analysis", analysis)
        self.assertIn("## Bull Market Regime", analysis)
        self.assertIn("## Bear Market Regime", analysis)
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        # Create historical results (two time periods)
        historical_results = [
            self.sample_results,  # Current period
            [self.sample_results[0]]  # Previous period (only passed test)
        ]
        
        analysis = self.reporter.generate_trend_report(historical_results)
        
        self.assertIn("# Test Result Trends", analysis)
        self.assertIn("Overall trend:", analysis)
        self.assertIn("Latest success rate:", analysis)
    
    def test_failure_analysis(self):
        """Test failure analysis functionality."""
        analysis = self.reporter.generate_failure_analysis(self.sample_results)
        
        self.assertIn("# Failure Analysis", analysis)
        self.assertIn("Total failures: 2", analysis)
        self.assertIn("## vol_risk", analysis)
        self.assertIn("## kelly_sizing", analysis)
    
    def test_failure_analysis_no_failures(self):
        """Test failure analysis with no failures."""
        passed_results = [self.sample_results[0]]  # Only passed test
        analysis = self.reporter.generate_failure_analysis(passed_results)
        
        self.assertEqual(analysis, "No test failures to analyze.")


if __name__ == '__main__':
    unittest.main()