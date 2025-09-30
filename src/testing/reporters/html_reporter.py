"""
HTML report generator with interactive features.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from .basic_reporter import BasicReporter
from ..models.test_result import TestResult, TestStatus, ConfidenceLevel
from ..models.test_case import TestType, TestPriority


class HTMLReporter(BasicReporter):
    """
    Advanced HTML report generator with interactive features.
    
    Extends BasicReporter to provide rich HTML reports with:
    - Interactive charts and graphs
    - Drill-down capabilities
    - Responsive design
    - Export functionality
    """
    
    def __init__(self):
        """Initialize the HTML reporter."""
        super().__init__()
        self.include_charts = True
        self.include_interactive_features = True
    
    def export_to_html(self, report_content: str, output_path: Path) -> None:
        """
        Export report content to interactive HTML format.
        
        Args:
            report_content: Formatted report content
            output_path: Path where HTML file should be saved
        """
        # For HTML reporter, we'll generate a more sophisticated HTML structure
        html_content = self._generate_interactive_html(report_content)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_interactive_dashboard(self, results: List[TestResult], output_path: Path) -> None:
        """
        Generate a comprehensive interactive dashboard.
        
        Args:
            results: List of TestResult objects
            output_path: Path where HTML dashboard should be saved
        """
        if not results:
            self._generate_empty_dashboard(output_path)
            return
        
        # Prepare data for dashboard
        dashboard_data = self._prepare_dashboard_data(results)
        
        # Generate HTML dashboard
        html_content = self._generate_dashboard_html(dashboard_data, results)
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _prepare_dashboard_data(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Prepare data structures for the interactive dashboard.
        
        Args:
            results: List of TestResult objects
            
        Returns:
            Dictionary with organized data for dashboard
        """
        # Overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Status distribution
        status_counts = {}
        for status in TestStatus:
            status_counts[status.value] = sum(1 for r in results if r.status == status)
        
        # Feature statistics
        feature_stats = {}
        for result in results:
            feature = result.test_case.feature_name
            if feature not in feature_stats:
                feature_stats[feature] = {'total': 0, 'passed': 0, 'failed': 0}
            
            feature_stats[feature]['total'] += 1
            if result.passed:
                feature_stats[feature]['passed'] += 1
            else:
                feature_stats[feature]['failed'] += 1
        
        # Test type distribution
        test_type_stats = {}
        for test_type in TestType:
            type_results = [r for r in results if r.test_case.test_type == test_type]
            if type_results:
                passed = sum(1 for r in type_results if r.passed)
                test_type_stats[test_type.value] = {
                    'total': len(type_results),
                    'passed': passed,
                    'failed': len(type_results) - passed,
                    'success_rate': (passed / len(type_results)) * 100
                }
        
        # Priority distribution
        priority_stats = {}
        for priority in TestPriority:
            priority_results = [r for r in results if r.test_case.priority == priority]
            if priority_results:
                passed = sum(1 for r in priority_results if r.passed)
                priority_stats[priority.value] = {
                    'total': len(priority_results),
                    'passed': passed,
                    'failed': len(priority_results) - passed,
                    'success_rate': (passed / len(priority_results)) * 100
                }
        
        # Confidence distribution
        confidence_stats = {}
        for confidence in ConfidenceLevel:
            confidence_stats[confidence.value] = sum(1 for r in results if r.confidence == confidence)
        
        # Execution time statistics
        execution_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Failed tests details
        failed_tests = [r for r in results if not r.passed]
        critical_failures = [r for r in failed_tests if r.test_case.priority == TestPriority.CRITICAL]
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'avg_execution_time': avg_execution_time,
                'critical_failures': len(critical_failures)
            },
            'status_distribution': status_counts,
            'feature_stats': feature_stats,
            'test_type_stats': test_type_stats,
            'priority_stats': priority_stats,
            'confidence_stats': confidence_stats,
            'failed_tests': [self._serialize_test_result(r) for r in failed_tests],
            'critical_failures': [self._serialize_test_result(r) for r in critical_failures],
            'timestamp': self.report_timestamp.isoformat()
        }
    
    def _serialize_test_result(self, result: TestResult) -> Dict[str, Any]:
        """
        Serialize a test result for JSON embedding in HTML.
        
        Args:
            result: TestResult to serialize
            
        Returns:
            Dictionary representation suitable for JSON
        """
        return {
            'feature_name': result.test_case.feature_name,
            'test_type': result.test_case.test_type.value,
            'priority': result.test_case.priority.value,
            'status': result.status.value,
            'confidence': result.confidence.value,
            'confidence_score': result.confidence_score,
            'execution_time': result.execution_time,
            'analysis': result.analysis,
            'error_message': result.error_message,
            'recommendations': result.recommendations,
            'severity': result.severity,
            'timestamp': result.timestamp.isoformat()
        }
    
    def _generate_dashboard_html(self, data: Dict[str, Any], results: List[TestResult]) -> str:
        """
        Generate the complete HTML dashboard.
        
        Args:
            data: Prepared dashboard data
            results: Original test results
            
        Returns:
            Complete HTML dashboard content
        """
        # Convert data to JSON for JavaScript
        data_json = json.dumps(data, indent=2)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Test Coverage Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._get_dashboard_css()}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>Feature Test Coverage Dashboard</h1>
            <div class="timestamp">Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>
        
        <div class="summary-cards">
            <div class="card success">
                <div class="card-title">Total Tests</div>
                <div class="card-value">{data['summary']['total_tests']}</div>
            </div>
            <div class="card {'success' if data['summary']['success_rate'] >= 90 else 'warning' if data['summary']['success_rate'] >= 70 else 'error'}">
                <div class="card-title">Success Rate</div>
                <div class="card-value">{data['summary']['success_rate']:.1f}%</div>
            </div>
            <div class="card {'success' if data['summary']['failed_tests'] == 0 else 'error'}">
                <div class="card-title">Failed Tests</div>
                <div class="card-value">{data['summary']['failed_tests']}</div>
            </div>
            <div class="card {'success' if data['summary']['critical_failures'] == 0 else 'error'}">
                <div class="card-title">Critical Failures</div>
                <div class="card-value">{data['summary']['critical_failures']}</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-section">
                <h2>Test Status Distribution</h2>
                <canvas id="statusChart"></canvas>
            </div>
            
            <div class="chart-section">
                <h2>Feature Coverage</h2>
                <canvas id="featureChart"></canvas>
            </div>
            
            <div class="chart-section">
                <h2>Test Type Performance</h2>
                <canvas id="testTypeChart"></canvas>
            </div>
            
            <div class="chart-section">
                <h2>Priority Distribution</h2>
                <canvas id="priorityChart"></canvas>
            </div>
        </div>
        
        <div class="details-section">
            <div class="tabs">
                <button class="tab-button active" onclick="showTab('failed-tests')">Failed Tests</button>
                <button class="tab-button" onclick="showTab('critical-failures')">Critical Failures</button>
                <button class="tab-button" onclick="showTab('feature-details')">Feature Details</button>
            </div>
            
            <div id="failed-tests" class="tab-content active">
                <h3>Failed Tests</h3>
                <div id="failed-tests-list"></div>
            </div>
            
            <div id="critical-failures" class="tab-content">
                <h3>Critical Failures</h3>
                <div id="critical-failures-list"></div>
            </div>
            
            <div id="feature-details" class="tab-content">
                <h3>Feature Details</h3>
                <div id="feature-details-list"></div>
            </div>
        </div>
        
        <div class="export-section">
            <button onclick="exportToJSON()" class="export-btn">Export to JSON</button>
            <button onclick="printReport()" class="export-btn">Print Report</button>
        </div>
    </div>
    
    <script>
        // Dashboard data
        const dashboardData = {data_json};
        
        {self._get_dashboard_javascript()}
    </script>
</body>
</html>"""
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styles for the dashboard."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa;
            color: #2d3748;
            line-height: 1.6;
        }
        
        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dashboard-header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .dashboard-header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .timestamp {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
        }
        
        .card.success {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
        }
        
        .card.warning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            color: white;
        }
        
        .card.error {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
            color: white;
        }
        
        .card-title {
            font-size: 1rem;
            opacity: 0.9;
            margin-bottom: 10px;
        }
        
        .card-value {
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .chart-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .chart-section h2 {
            margin-bottom: 20px;
            color: #2d3748;
            font-size: 1.3rem;
        }
        
        .details-section {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .tab-button {
            padding: 15px 25px;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 1rem;
            color: #718096;
            transition: all 0.2s;
        }
        
        .tab-button.active {
            color: #4299e1;
            border-bottom: 2px solid #4299e1;
        }
        
        .tab-button:hover {
            background-color: #f7fafc;
        }
        
        .tab-content {
            display: none;
            padding: 25px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .test-item {
            padding: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            margin-bottom: 15px;
            transition: all 0.2s;
        }
        
        .test-item:hover {
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .test-item.failed {
            border-left: 4px solid #f56565;
        }
        
        .test-item.critical {
            border-left: 4px solid #e53e3e;
            background-color: #fed7d7;
        }
        
        .test-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .test-title {
            font-weight: bold;
            color: #2d3748;
        }
        
        .test-status {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .status-failed {
            background-color: #fed7d7;
            color: #c53030;
        }
        
        .status-error {
            background-color: #fbb6ce;
            color: #97266d;
        }
        
        .test-details {
            color: #718096;
            font-size: 0.9rem;
        }
        
        .export-section {
            text-align: center;
            padding: 20px;
        }
        
        .export-btn {
            padding: 12px 24px;
            margin: 0 10px;
            border: none;
            border-radius: 6px;
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.2s;
        }
        
        .export-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        @media (max-width: 768px) {
            .dashboard-container {
                padding: 10px;
            }
            
            .charts-container {
                grid-template-columns: 1fr;
            }
            
            .summary-cards {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .dashboard-header h1 {
                font-size: 2rem;
            }
        }
        """
    
    def _get_dashboard_javascript(self) -> str:
        """Get JavaScript code for dashboard interactivity."""
        return """
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            populateTestLists();
        });
        
        function initializeCharts() {
            // Status distribution chart
            const statusCtx = document.getElementById('statusChart').getContext('2d');
            new Chart(statusCtx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(dashboardData.status_distribution),
                    datasets: [{
                        data: Object.values(dashboardData.status_distribution),
                        backgroundColor: [
                            '#48bb78', // passed
                            '#f56565', // failed
                            '#ed8936', // error
                            '#a0aec0', // skipped
                            '#9f7aea'  // timeout
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
            
            // Feature coverage chart
            const featureCtx = document.getElementById('featureChart').getContext('2d');
            const featureLabels = Object.keys(dashboardData.feature_stats);
            const featureSuccessRates = featureLabels.map(feature => {
                const stats = dashboardData.feature_stats[feature];
                return (stats.passed / stats.total) * 100;
            });
            
            new Chart(featureCtx, {
                type: 'bar',
                data: {
                    labels: featureLabels,
                    datasets: [{
                        label: 'Success Rate (%)',
                        data: featureSuccessRates,
                        backgroundColor: featureSuccessRates.map(rate => 
                            rate >= 90 ? '#48bb78' : rate >= 70 ? '#ed8936' : '#f56565'
                        )
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Test type performance chart
            const testTypeCtx = document.getElementById('testTypeChart').getContext('2d');
            const testTypeLabels = Object.keys(dashboardData.test_type_stats);
            const testTypeSuccessRates = testTypeLabels.map(type => 
                dashboardData.test_type_stats[type].success_rate
            );
            
            new Chart(testTypeCtx, {
                type: 'horizontalBar',
                data: {
                    labels: testTypeLabels.map(label => label.replace('_', ' ')),
                    datasets: [{
                        label: 'Success Rate (%)',
                        data: testTypeSuccessRates,
                        backgroundColor: '#4299e1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Priority distribution chart
            const priorityCtx = document.getElementById('priorityChart').getContext('2d');
            const priorityLabels = Object.keys(dashboardData.priority_stats);
            const priorityData = priorityLabels.map(priority => ({
                label: priority,
                passed: dashboardData.priority_stats[priority].passed,
                failed: dashboardData.priority_stats[priority].failed
            }));
            
            new Chart(priorityCtx, {
                type: 'bar',
                data: {
                    labels: priorityLabels,
                    datasets: [{
                        label: 'Passed',
                        data: priorityData.map(d => d.passed),
                        backgroundColor: '#48bb78'
                    }, {
                        label: 'Failed',
                        data: priorityData.map(d => d.failed),
                        backgroundColor: '#f56565'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            stacked: true
                        },
                        y: {
                            stacked: true
                        }
                    }
                }
            });
        }
        
        function populateTestLists() {
            // Populate failed tests
            const failedTestsList = document.getElementById('failed-tests-list');
            dashboardData.failed_tests.forEach(test => {
                const testElement = createTestElement(test, 'failed');
                failedTestsList.appendChild(testElement);
            });
            
            // Populate critical failures
            const criticalFailuresList = document.getElementById('critical-failures-list');
            dashboardData.critical_failures.forEach(test => {
                const testElement = createTestElement(test, 'critical');
                criticalFailuresList.appendChild(testElement);
            });
            
            // Populate feature details
            const featureDetailsList = document.getElementById('feature-details-list');
            Object.entries(dashboardData.feature_stats).forEach(([feature, stats]) => {
                const featureElement = createFeatureElement(feature, stats);
                featureDetailsList.appendChild(featureElement);
            });
        }
        
        function createTestElement(test, type) {
            const div = document.createElement('div');
            div.className = `test-item ${type}`;
            
            div.innerHTML = `
                <div class="test-header">
                    <div class="test-title">${test.feature_name} - ${test.test_type.replace('_', ' ')}</div>
                    <div class="test-status status-${test.status}">${test.status.toUpperCase()}</div>
                </div>
                <div class="test-details">
                    <p><strong>Priority:</strong> ${test.priority}</p>
                    <p><strong>Confidence:</strong> ${test.confidence} (${(test.confidence_score * 100).toFixed(1)}%)</p>
                    <p><strong>Execution Time:</strong> ${test.execution_time.toFixed(2)}s</p>
                    ${test.error_message ? `<p><strong>Error:</strong> ${test.error_message}</p>` : ''}
                    ${test.analysis ? `<p><strong>Analysis:</strong> ${test.analysis}</p>` : ''}
                    ${test.recommendations.length > 0 ? `
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            ${test.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    ` : ''}
                </div>
            `;
            
            return div;
        }
        
        function createFeatureElement(feature, stats) {
            const div = document.createElement('div');
            div.className = 'test-item';
            
            const successRate = (stats.passed / stats.total) * 100;
            const statusClass = successRate >= 90 ? 'success' : successRate >= 70 ? 'warning' : 'error';
            
            div.innerHTML = `
                <div class="test-header">
                    <div class="test-title">${feature}</div>
                    <div class="test-status status-${statusClass}">${successRate.toFixed(1)}%</div>
                </div>
                <div class="test-details">
                    <p><strong>Total Tests:</strong> ${stats.total}</p>
                    <p><strong>Passed:</strong> ${stats.passed}</p>
                    <p><strong>Failed:</strong> ${stats.failed}</p>
                </div>
            `;
            
            return div;
        }
        
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tab buttons
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
        
        function exportToJSON() {
            const dataStr = JSON.stringify(dashboardData, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `test-coverage-report-${new Date().toISOString().split('T')[0]}.json`;
            link.click();
            
            URL.revokeObjectURL(url);
        }
        
        function printReport() {
            window.print();
        }
        """
    
    def _generate_interactive_html(self, report_content: str) -> str:
        """
        Generate interactive HTML from basic report content.
        
        Args:
            report_content: Basic report content
            
        Returns:
            Enhanced HTML with interactive features
        """
        # Enhanced version of the basic HTML conversion
        html_content = self._markdown_to_html(report_content)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Test Coverage Report</title>
    <style>
        {self._get_interactive_css()}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>Interactive Test Coverage Report</h1>
            <div class="report-controls">
                <button onclick="toggleSection('summary')" class="control-btn">Toggle Summary</button>
                <button onclick="toggleSection('details')" class="control-btn">Toggle Details</button>
                <button onclick="exportReport()" class="control-btn">Export</button>
            </div>
        </div>
        
        <div class="report-content">
            {html_content}
        </div>
        
        <div class="report-footer">
            <p>Generated by Feature Test Coverage System</p>
            <p>Timestamp: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        {self._get_interactive_javascript()}
    </script>
</body>
</html>"""
    
    def _get_interactive_css(self) -> str:
        """Get CSS for interactive HTML reports."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        
        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            min-height: 100vh;
        }
        
        .report-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .report-header h1 {
            margin: 0;
            font-size: 2rem;
        }
        
        .report-controls {
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            padding: 8px 16px;
            border: 2px solid white;
            background: transparent;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .control-btn:hover {
            background: white;
            color: #667eea;
        }
        
        .report-content {
            padding: 30px;
        }
        
        .report-footer {
            background-color: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        h1 {
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        
        .success {
            color: #27ae60;
            font-weight: bold;
        }
        
        .warning {
            color: #f39c12;
            font-weight: bold;
        }
        
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
        
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background-color: #f1f1f1;
            border: none;
            text-align: left;
            outline: none;
            font-size: 15px;
            transition: 0.3s;
        }
        
        .collapsible:hover {
            background-color: #ddd;
        }
        
        .collapsible-content {
            padding: 0 18px;
            display: none;
            overflow: hidden;
            background-color: #f9f9f9;
        }
        
        .collapsible-content.active {
            display: block;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .highlight {
            background-color: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .report-header {
                flex-direction: column;
                gap: 20px;
            }
            
            .report-content {
                padding: 20px;
            }
            
            .control-btn {
                padding: 6px 12px;
                font-size: 0.9rem;
            }
        }
        """
    
    def _get_interactive_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return """
        function toggleSection(sectionType) {
            const elements = document.querySelectorAll(`[data-section="${sectionType}"]`);
            elements.forEach(element => {
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            });
        }
        
        function exportReport() {
            const reportContent = document.querySelector('.report-content').innerHTML;
            const blob = new Blob([reportContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = `test-report-${new Date().toISOString().split('T')[0]}.html`;
            link.click();
            
            URL.revokeObjectURL(url);
        }
        
        // Add collapsible functionality
        document.addEventListener('DOMContentLoaded', function() {
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(collapsible => {
                collapsible.addEventListener('click', function() {
                    this.classList.toggle('active');
                    const content = this.nextElementSibling;
                    content.classList.toggle('active');
                });
            });
        });
        """
    
    def _generate_empty_dashboard(self, output_path: Path) -> None:
        """
        Generate an empty dashboard when no results are available.
        
        Args:
            output_path: Path where HTML file should be saved
        """
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Feature Test Coverage Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .empty-state {{
            text-align: center;
            padding: 60px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 500px;
        }}
        .empty-icon {{
            font-size: 4rem;
            margin-bottom: 20px;
        }}
        .empty-title {{
            font-size: 1.5rem;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        .empty-message {{
            color: #718096;
            margin-bottom: 30px;
        }}
        .empty-action {{
            padding: 12px 24px;
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1rem;
        }}
    </style>
</head>
<body>
    <div class="empty-state">
        <div class="empty-icon">📊</div>
        <h1 class="empty-title">No Test Results Available</h1>
        <p class="empty-message">
            Run some tests to see your coverage dashboard here.
        </p>
        <button class="empty-action" onclick="window.location.reload()">
            Refresh Page
        </button>
        <p style="margin-top: 20px; color: #a0aec0; font-size: 0.9rem;">
            Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
</body>
</html>"""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of output formats supported by this reporter.
        
        Returns:
            List of supported format names
        """
        return ['html', 'json', 'interactive_html', 'dashboard']