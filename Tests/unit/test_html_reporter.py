"""
Unit tests for the HTMLReporter class.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from datetime import datetime
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from testing.reporters.html_reporter import HTMLReporter
from testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from testing.models.test_case import TestCase, TestType, TestPriority


class TestHTMLReporter(unittest.TestCase):
    """Test cases for HTMLReporter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = HTMLReporter()
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
            expected_result="Directional bias validation",
            regime_context="bull"
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
            expected_result="Variance calculation validation",
            regime_context="bear"
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
    
    def test_init(self):
        """Test HTMLReporter initialization."""
        reporter = HTMLReporter()
        self.assertIsInstance(reporter.report_timestamp, datetime)
        self.assertTrue(reporter.include_charts)
        self.assertTrue(reporter.include_interactive_features)
    
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
            self.assertIn("Interactive Test Coverage Report", html_content)
            self.assertIn("report-container", html_content)
    
    def test_generate_interactive_dashboard_empty_results(self):
        """Test interactive dashboard generation with empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_dashboard.html"
            
            self.reporter.generate_interactive_dashboard([], output_path)
            
            # Check that file was created
            self.assertTrue(output_path.exists())
            
            # Check HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.assertIn("No Test Results Available", html_content)
            self.assertIn("empty-state", html_content)
    
    def test_generate_interactive_dashboard_with_results(self):
        """Test interactive dashboard generation with sample results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dashboard.html"
            
            self.reporter.generate_interactive_dashboard(self.sample_results, output_path)
            
            # Check that file was created
            self.assertTrue(output_path.exists())
            
            # Check HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for dashboard elements
            self.assertIn("Feature Test Coverage Dashboard", html_content)
            self.assertIn("summary-cards", html_content)
            self.assertIn("charts-container", html_content)
            self.assertIn("Chart.js", html_content)
            
            # Check for data
            self.assertIn("Total Tests", html_content)
            self.assertIn("Success Rate", html_content)
            self.assertIn("Failed Tests", html_content)
            self.assertIn("Critical Failures", html_content)
    
    def test_prepare_dashboard_data(self):
        """Test dashboard data preparation."""
        data = self.reporter._prepare_dashboard_data(self.sample_results)
        
        # Check summary data
        self.assertIn('summary', data)
        summary = data['summary']
        self.assertEqual(summary['total_tests'], 2)
        self.assertEqual(summary['passed_tests'], 1)
        self.assertEqual(summary['failed_tests'], 1)
        self.assertEqual(summary['success_rate'], 50.0)
        self.assertEqual(summary['critical_failures'], 1)
        
        # Check distributions
        self.assertIn('status_distribution', data)
        self.assertIn('feature_stats', data)
        self.assertIn('test_type_stats', data)
        self.assertIn('priority_stats', data)
        self.assertIn('confidence_stats', data)
        
        # Check failed tests data
        self.assertIn('failed_tests', data)
        self.assertEqual(len(data['failed_tests']), 1)
        
        # Check critical failures data
        self.assertIn('critical_failures', data)
        self.assertEqual(len(data['critical_failures']), 1)
    
    def test_serialize_test_result(self):
        """Test test result serialization."""
        result = self.sample_results[0]
        serialized = self.reporter._serialize_test_result(result)
        
        # Check required fields
        self.assertIn('feature_name', serialized)
        self.assertIn('test_type', serialized)
        self.assertIn('priority', serialized)
        self.assertIn('status', serialized)
        self.assertIn('confidence', serialized)
        self.assertIn('confidence_score', serialized)
        self.assertIn('execution_time', serialized)
        self.assertIn('timestamp', serialized)
        
        # Check values
        self.assertEqual(serialized['feature_name'], 'Q50')
        self.assertEqual(serialized['test_type'], 'economic_hypothesis')
        self.assertEqual(serialized['priority'], 'high')
        self.assertEqual(serialized['status'], 'passed')
    
    def test_generate_dashboard_html(self):
        """Test dashboard HTML generation."""
        data = self.reporter._prepare_dashboard_data(self.sample_results)
        html_content = self.reporter._generate_dashboard_html(data, self.sample_results)
        
        # Check HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Feature Test Coverage Dashboard", html_content)
        self.assertIn("Chart.js", html_content)
        
        # Check for JavaScript data
        self.assertIn("dashboardData", html_content)
        self.assertIn("initializeCharts", html_content)
        self.assertIn("populateTestLists", html_content)
        
        # Check for interactive elements
        self.assertIn("showTab", html_content)
        self.assertIn("exportToJSON", html_content)
        self.assertIn("printReport", html_content)
    
    def test_get_dashboard_css(self):
        """Test dashboard CSS generation."""
        css = self.reporter._get_dashboard_css()
        
        # Check for key CSS classes
        self.assertIn(".dashboard-container", css)
        self.assertIn(".summary-cards", css)
        self.assertIn(".charts-container", css)
        self.assertIn(".card", css)
        self.assertIn(".success", css)
        self.assertIn(".warning", css)
        self.assertIn(".error", css)
        
        # Check for responsive design
        self.assertIn("@media", css)
        self.assertIn("max-width: 768px", css)
    
    def test_get_dashboard_javascript(self):
        """Test dashboard JavaScript generation."""
        js = self.reporter._get_dashboard_javascript()
        
        # Check for key functions
        self.assertIn("initializeCharts", js)
        self.assertIn("populateTestLists", js)
        self.assertIn("createTestElement", js)
        self.assertIn("showTab", js)
        self.assertIn("exportToJSON", js)
        self.assertIn("printReport", js)
        
        # Check for Chart.js usage
        self.assertIn("new Chart", js)
        self.assertIn("doughnut", js)
        self.assertIn("bar", js)
    
    def test_generate_interactive_html(self):
        """Test interactive HTML generation."""
        report_content = "# Test Report\n\nThis is a test report."
        html_content = self.reporter._generate_interactive_html(report_content)
        
        # Check HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Interactive Test Coverage Report", html_content)
        self.assertIn("report-container", html_content)
        
        # Check for interactive elements
        self.assertIn("report-controls", html_content)
        self.assertIn("toggleSection", html_content)
        self.assertIn("exportReport", html_content)
        
        # Check for content
        self.assertIn("<h1>Test Report</h1>", html_content)
        self.assertIn("This is a test report.", html_content)
    
    def test_get_interactive_css(self):
        """Test interactive CSS generation."""
        css = self.reporter._get_interactive_css()
        
        # Check for key CSS classes
        self.assertIn(".report-container", css)
        self.assertIn(".report-header", css)
        self.assertIn(".report-controls", css)
        self.assertIn(".control-btn", css)
        self.assertIn(".collapsible", css)
        
        # Check for responsive design
        self.assertIn("@media", css)
        self.assertIn("max-width: 768px", css)
    
    def test_get_interactive_javascript(self):
        """Test interactive JavaScript generation."""
        js = self.reporter._get_interactive_javascript()
        
        # Check for key functions
        self.assertIn("toggleSection", js)
        self.assertIn("exportReport", js)
        self.assertIn("DOMContentLoaded", js)
        
        # Check for collapsible functionality
        self.assertIn("collapsible", js)
        self.assertIn("addEventListener", js)
    
    def test_get_supported_formats(self):
        """Test supported formats list."""
        formats = self.reporter.get_supported_formats()
        expected_formats = ['html', 'json', 'interactive_html', 'dashboard']
        
        self.assertEqual(formats, expected_formats)
    
    def test_inheritance_from_basic_reporter(self):
        """Test that HTMLReporter inherits from BasicReporter."""
        from testing.reporters.basic_reporter import BasicReporter
        
        self.assertIsInstance(self.reporter, BasicReporter)
        
        # Test that inherited methods work
        summary = self.reporter.generate_summary_report(self.sample_results)
        self.assertIn("# Test Coverage Summary Report", summary)
        
        feature_report = self.reporter.generate_feature_report("Q50", self.sample_results)
        self.assertIn("# Feature Report: Q50", feature_report)
    
    def test_dashboard_data_completeness(self):
        """Test that dashboard data contains all required fields."""
        data = self.reporter._prepare_dashboard_data(self.sample_results)
        
        # Check all required top-level keys
        required_keys = [
            'summary', 'status_distribution', 'feature_stats',
            'test_type_stats', 'priority_stats', 'confidence_stats',
            'failed_tests', 'critical_failures', 'timestamp'
        ]
        
        for key in required_keys:
            self.assertIn(key, data, f"Missing required key: {key}")
        
        # Check summary completeness
        summary_keys = [
            'total_tests', 'passed_tests', 'failed_tests',
            'success_rate', 'avg_execution_time', 'critical_failures'
        ]
        
        for key in summary_keys:
            self.assertIn(key, data['summary'], f"Missing summary key: {key}")
    
    def test_empty_dashboard_generation(self):
        """Test empty dashboard generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_dashboard.html"
            
            self.reporter._generate_empty_dashboard(output_path)
            
            # Check that file was created
            self.assertTrue(output_path.exists())
            
            # Check HTML content
            with open(output_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            self.assertIn("No Test Results Available", html_content)
            self.assertIn("empty-state", html_content)
            self.assertIn("Refresh Page", html_content)


if __name__ == '__main__':
    unittest.main()