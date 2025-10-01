"""
Basic report generator for test coverage results.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from ..interfaces.reporter_interface import ReporterInterface
from ..models.test_result import TestResult, TestStatus, ConfidenceLevel
from ..models.test_case import TestType, TestPriority


class BasicReporter(ReporterInterface):
    """
    Basic implementation of test result reporting.
    
    Generates text-based reports with pass/fail statistics,
    feature-level details, and coverage analysis.
    """
    
    def __init__(self):
        """Initialize the basic reporter."""
        self.report_timestamp = datetime.now()
    
    def generate_summary_report(self, results: List[TestResult]) -> str:
        """
        Generate a high-level summary report of test results.
        
        Args:
            results: List of TestResult objects to summarize
            
        Returns:
            String containing formatted summary report
        """
        if not results:
            return "No test results to report."
        
        # Calculate overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate statistics by status
        status_counts = {}
        for status in TestStatus:
            status_counts[status] = sum(1 for r in results if r.status == status)
        
        # Calculate statistics by priority
        priority_stats = {}
        for priority in TestPriority:
            priority_results = [r for r in results if r.test_case.priority == priority]
            if priority_results:
                priority_passed = sum(1 for r in priority_results if r.passed)
                priority_stats[priority] = {
                    'total': len(priority_results),
                    'passed': priority_passed,
                    'success_rate': (priority_passed / len(priority_results)) * 100
                }
        
        # Calculate average execution time
        avg_execution_time = sum(r.execution_time for r in results) / len(results)
        
        # Identify critical failures
        critical_failures = [
            r for r in results 
            if not r.passed and r.test_case.priority in [TestPriority.CRITICAL, TestPriority.HIGH]
        ]
        
        # Build report
        report_lines = [
            "# Test Coverage Summary Report",
            f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Overall Statistics",
            f"- Total Tests: {total_tests}",
            f"- Passed: {passed_tests}",
            f"- Failed: {failed_tests}",
            f"- Success Rate: {success_rate:.1f}%",
            f"- Average Execution Time: {avg_execution_time:.2f}s",
            ""
        ]
        
        # Add status breakdown
        report_lines.extend([
            "## Test Status Breakdown",
            f"- Passed: {status_counts[TestStatus.PASSED]}",
            f"- Failed: {status_counts[TestStatus.FAILED]}",
            f"- Errors: {status_counts[TestStatus.ERROR]}",
            f"- Skipped: {status_counts[TestStatus.SKIPPED]}",
            f"- Timeouts: {status_counts[TestStatus.TIMEOUT]}",
            ""
        ])
        
        # Add priority breakdown
        if priority_stats:
            report_lines.append("## Priority Breakdown")
            for priority, stats in priority_stats.items():
                report_lines.append(
                    f"- {priority.value.title()}: {stats['passed']}/{stats['total']} "
                    f"({stats['success_rate']:.1f}%)"
                )
            report_lines.append("")
        
        # Add critical failures section
        if critical_failures:
            report_lines.extend([
                "## Critical Failures",
                f"Found {len(critical_failures)} critical/high priority failures:",
                ""
            ])
            
            for failure in critical_failures:
                report_lines.extend([
                    f"### {failure.test_case.feature_name} - {failure.test_case.test_type.value}",
                    f"- Priority: {failure.test_case.priority.value}",
                    f"- Error: {failure.error_message or 'No error message'}",
                    f"- Analysis: {failure.analysis or 'No analysis available'}",
                    ""
                ])
                
                if failure.recommendations:
                    report_lines.append("- Recommendations:")
                    for rec in failure.recommendations:
                        report_lines.append(f"  - {rec}")
                    report_lines.append("")
        
        # Add confidence analysis
        confidence_stats = {}
        for confidence in ConfidenceLevel:
            confidence_stats[confidence] = sum(1 for r in results if r.confidence == confidence)
        
        report_lines.extend([
            "## Confidence Analysis",
            f"- High Confidence: {confidence_stats[ConfidenceLevel.HIGH]}",
            f"- Medium Confidence: {confidence_stats[ConfidenceLevel.MEDIUM]}",
            f"- Low Confidence: {confidence_stats[ConfidenceLevel.LOW]}",
            f"- Uncertain: {confidence_stats[ConfidenceLevel.UNCERTAIN]}",
            ""
        ])
        
        # Add recommendations section
        all_recommendations = []
        for result in results:
            if not result.passed and result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            # Get unique recommendations
            unique_recommendations = list(set(all_recommendations))
            report_lines.extend([
                "## Key Recommendations",
                ""
            ])
            for i, rec in enumerate(unique_recommendations[:10], 1):  # Top 10
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_feature_report(self, feature_name: str, results: List[TestResult]) -> str:
        """
        Generate a detailed report for a specific feature.
        
        Args:
            feature_name: Name of feature to report on
            results: List of TestResult objects for this feature
            
        Returns:
            String containing formatted feature report
        """
        # Filter results for this feature
        feature_results = [r for r in results if r.test_case.feature_name == feature_name]
        
        if not feature_results:
            return f"No test results found for feature: {feature_name}"
        
        # Calculate feature statistics
        total_tests = len(feature_results)
        passed_tests = sum(1 for r in feature_results if r.passed)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Group by test type
        test_type_stats = {}
        for test_type in TestType:
            type_results = [r for r in feature_results if r.test_case.test_type == test_type]
            if type_results:
                type_passed = sum(1 for r in type_results if r.passed)
                test_type_stats[test_type] = {
                    'total': len(type_results),
                    'passed': type_passed,
                    'success_rate': (type_passed / len(type_results)) * 100,
                    'results': type_results
                }
        
        # Calculate average confidence score
        avg_confidence = sum(r.confidence_score for r in feature_results) / len(feature_results)
        
        # Build report
        report_lines = [
            f"# Feature Report: {feature_name}",
            f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Feature Overview",
            f"- Total Tests: {total_tests}",
            f"- Passed: {passed_tests}",
            f"- Failed: {total_tests - passed_tests}",
            f"- Success Rate: {success_rate:.1f}%",
            f"- Average Confidence: {avg_confidence:.2f}",
            ""
        ]
        
        # Add test type breakdown
        if test_type_stats:
            report_lines.append("## Test Type Results")
            for test_type, stats in test_type_stats.items():
                report_lines.extend([
                    f"### {test_type.value.replace('_', ' ').title()}",
                    f"- Tests: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)",
                    ""
                ])
                
                # Add details for failed tests
                failed_tests = [r for r in stats['results'] if not r.passed]
                if failed_tests:
                    report_lines.append("**Failed Tests:**")
                    for failed in failed_tests:
                        report_lines.extend([
                            f"- Status: {failed.status.value}",
                            f"- Analysis: {failed.analysis or 'No analysis'}",
                            f"- Error: {failed.error_message or 'No error message'}",
                            ""
                        ])
                        
                        if failed.recommendations:
                            report_lines.append("  Recommendations:")
                            for rec in failed.recommendations:
                                report_lines.append(f"  - {rec}")
                            report_lines.append("")
        
        # Add performance metrics if available
        performance_metrics = {}
        for result in feature_results:
            for metric, value in result.performance_metrics.items():
                if metric not in performance_metrics:
                    performance_metrics[metric] = []
                performance_metrics[metric].append(value)
        
        if performance_metrics:
            report_lines.append("## Performance Metrics")
            for metric, values in performance_metrics.items():
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
                report_lines.extend([
                    f"### {metric.replace('_', ' ').title()}",
                    f"- Average: {avg_value:.3f}",
                    f"- Range: {min_value:.3f} - {max_value:.3f}",
                    ""
                ])
        
        # Add data quality issues
        all_issues = []
        for result in feature_results:
            all_issues.extend(result.data_quality_issues)
        
        if all_issues:
            unique_issues = list(set(all_issues))
            report_lines.extend([
                "## Data Quality Issues",
                ""
            ])
            for issue in unique_issues:
                report_lines.append(f"- {issue}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_coverage_report(self, results: List[TestResult]) -> str:
        """
        Generate a test coverage analysis report.
        
        Args:
            results: List of TestResult objects to analyze coverage
            
        Returns:
            String containing formatted coverage report
        """
        if not results:
            return "No test results available for coverage analysis."
        
        # Group results by feature
        feature_coverage = {}
        for result in results:
            feature = result.test_case.feature_name
            if feature not in feature_coverage:
                feature_coverage[feature] = {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'test_types': set(),
                    'priorities': set(),
                    'results': []
                }
            
            feature_coverage[feature]['total_tests'] += 1
            if result.passed:
                feature_coverage[feature]['passed_tests'] += 1
            feature_coverage[feature]['test_types'].add(result.test_case.test_type)
            feature_coverage[feature]['priorities'].add(result.test_case.priority)
            feature_coverage[feature]['results'].append(result)
        
        # Calculate coverage statistics
        total_features = len(feature_coverage)
        fully_covered_features = sum(
            1 for stats in feature_coverage.values() 
            if stats['passed_tests'] == stats['total_tests']
        )
        
        # Build report
        report_lines = [
            "# Test Coverage Analysis",
            f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Coverage Overview",
            f"- Total Features Tested: {total_features}",
            f"- Fully Covered Features: {fully_covered_features}",
            f"- Coverage Rate: {(fully_covered_features / total_features) * 100:.1f}%",
            ""
        ]
        
        # Add feature-by-feature coverage
        report_lines.append("## Feature Coverage Details")
        
        # Sort features by coverage (worst first)
        sorted_features = sorted(
            feature_coverage.items(),
            key=lambda x: x[1]['passed_tests'] / x[1]['total_tests']
        )
        
        for feature_name, stats in sorted_features:
            success_rate = (stats['passed_tests'] / stats['total_tests']) * 100
            coverage_status = "" if success_rate == 100 else "" if success_rate >= 80 else ""
            
            report_lines.extend([
                f"### {coverage_status} {feature_name}",
                f"- Tests: {stats['passed_tests']}/{stats['total_tests']} ({success_rate:.1f}%)",
                f"- Test Types: {len(stats['test_types'])} ({', '.join(t.value for t in stats['test_types'])})",
                f"- Priorities: {', '.join(p.value for p in stats['priorities'])}",
                ""
            ])
            
            # Add failed test details for poorly covered features
            if success_rate < 100:
                failed_tests = [r for r in stats['results'] if not r.passed]
                if failed_tests:
                    report_lines.append("**Failed Tests:**")
                    for failed in failed_tests:
                        report_lines.append(
                            f"- {failed.test_case.test_type.value}: {failed.analysis or 'No analysis'}"
                        )
                    report_lines.append("")
        
        # Add test type coverage analysis
        test_type_coverage = {}
        for test_type in TestType:
            type_results = [r for r in results if r.test_case.test_type == test_type]
            if type_results:
                passed = sum(1 for r in type_results if r.passed)
                test_type_coverage[test_type] = {
                    'total': len(type_results),
                    'passed': passed,
                    'success_rate': (passed / len(type_results)) * 100
                }
        
        if test_type_coverage:
            report_lines.extend([
                "## Test Type Coverage",
                ""
            ])
            
            for test_type, stats in test_type_coverage.items():
                report_lines.append(
                    f"- {test_type.value.replace('_', ' ').title()}: "
                    f"{stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)"
                )
            report_lines.append("")
        
        # Add priority coverage analysis
        priority_coverage = {}
        for priority in TestPriority:
            priority_results = [r for r in results if r.test_case.priority == priority]
            if priority_results:
                passed = sum(1 for r in priority_results if r.passed)
                priority_coverage[priority] = {
                    'total': len(priority_results),
                    'passed': passed,
                    'success_rate': (passed / len(priority_results)) * 100
                }
        
        if priority_coverage:
            report_lines.extend([
                "## Priority Coverage",
                ""
            ])
            
            for priority, stats in priority_coverage.items():
                status_emoji = "" if stats['success_rate'] == 100 else "" if stats['success_rate'] >= 80 else ""
                report_lines.append(
                    f"- {status_emoji} {priority.value.title()}: "
                    f"{stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)"
                )
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_to_html(self, report_content: str, output_path: Path) -> None:
        """
        Export report content to HTML format.
        
        Args:
            report_content: Formatted report content (markdown)
            output_path: Path where HTML file should be saved
        """
        # Convert markdown to HTML (basic conversion)
        html_content = self._markdown_to_html(report_content)
        
        # Wrap in HTML template
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Coverage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .stats {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .critical-failure {{
            background-color: #fdf2f2;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    
    def _markdown_to_html(self, markdown_content: str) -> str:
        """
        Convert basic markdown to HTML.
        
        Args:
            markdown_content: Markdown formatted text
            
        Returns:
            HTML formatted text
        """
        lines = markdown_content.split('\n')
        html_lines = []
        in_code_block = False
        
        for line in lines:
            # Handle code blocks
            if line.startswith('```'):
                if in_code_block:
                    html_lines.append('</pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre>')
                    in_code_block = True
                continue
            
            if in_code_block:
                html_lines.append(line)
                continue
            
            # Handle headers
            if line.startswith('# '):
                html_lines.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_lines.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_lines.append(f'<h3>{line[4:]}</h3>')
            # Handle lists
            elif line.startswith('- '):
                if not html_lines or not html_lines[-1].startswith('<ul>'):
                    html_lines.append('<ul>')
                html_lines.append(f'<li>{line[2:]}</li>')
            elif line.startswith('  - '):
                html_lines.append(f'<li style="margin-left: 20px;">{line[4:]}</li>')
            # Handle numbered lists
            elif line.strip() and line[0].isdigit() and '. ' in line:
                if not html_lines or not html_lines[-1].startswith('<ol>'):
                    html_lines.append('<ol>')
                content = line.split('. ', 1)[1] if '. ' in line else line
                html_lines.append(f'<li>{content}</li>')
            # Handle empty lines and close lists
            elif not line.strip():
                if html_lines and html_lines[-1].startswith('<li>'):
                    if '<ul>' in '\n'.join(html_lines[-10:]):
                        html_lines.append('</ul>')
                    elif '<ol>' in '\n'.join(html_lines[-10:]):
                        html_lines.append('</ol>')
                html_lines.append('<br>')
            # Handle regular paragraphs
            else:
                # Add some styling based on content
                styled_line = line
                if '' in line or 'PASS' in line:
                    styled_line = f'<span class="success">{line}</span>'
                elif '' in line or 'WARNING' in line:
                    styled_line = f'<span class="warning">{line}</span>'
                elif '' in line or 'FAIL' in line or 'ERROR' in line:
                    styled_line = f'<span class="error">{line}</span>'
                elif 'Generated:' in line:
                    styled_line = f'<span class="timestamp">{line}</span>'
                
                html_lines.append(f'<p>{styled_line}</p>')
        
        # Close any remaining lists
        if html_lines and html_lines[-1].startswith('<li>'):
            if '<ul>' in '\n'.join(html_lines[-10:]):
                html_lines.append('</ul>')
            elif '<ol>' in '\n'.join(html_lines[-10:]):
                html_lines.append('</ol>')
        
        return '\n'.join(html_lines)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of output formats supported by this reporter.
        
        Returns:
            List of supported format names
        """
        return ['text', 'html', 'json']
    
    def export_to_file(self, report_content: str, output_path: Path, format_type: str = 'text') -> None:
        """
        Export report to file in specified format.
        
        Args:
            report_content: Report content to export
            output_path: Path where file should be saved
            format_type: Format type ('text', 'html', 'json')
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type.lower() == 'html':
            self.export_to_html(report_content, output_path)
        elif format_type.lower() == 'json':
            # For JSON, we need the actual results, not the formatted report
            # This is a placeholder - in practice, you'd pass the results directly
            json_data = {
                'report_type': 'basic_report',
                'timestamp': self.report_timestamp.isoformat(),
                'content': report_content
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
        else:  # Default to text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
    
    def generate_executive_summary(self, results: List[TestResult]) -> str:
        """
        Generate a concise executive summary for stakeholders.
        
        Args:
            results: List of TestResult objects to summarize
            
        Returns:
            String containing executive summary
        """
        if not results:
            return "No test results available."
        
        # Calculate key metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        success_rate = (passed_tests / total_tests) * 100
        
        # Identify critical issues
        critical_failures = [
            r for r in results 
            if not r.passed and r.test_case.priority == TestPriority.CRITICAL
        ]
        
        high_priority_failures = [
            r for r in results 
            if not r.passed and r.test_case.priority == TestPriority.HIGH
        ]
        
        # Calculate risk assessment
        high_risk_count = len([r for r in results if r.get_risk_assessment()['risk_level'] == 'high'])
        
        # Build executive summary
        summary_lines = [
            "# Executive Summary",
            f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Key Metrics",
            f"- Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})",
            f"- Critical Failures: {len(critical_failures)}",
            f"- High Priority Failures: {len(high_priority_failures)}",
            f"- High Risk Issues: {high_risk_count}",
            ""
        ]
        
        # Add status assessment
        if success_rate >= 95:
            status = "🟢 EXCELLENT"
            message = "System is performing exceptionally well with minimal issues."
        elif success_rate >= 85:
            status = "🟡 GOOD"
            message = "System is performing well with some areas for improvement."
        elif success_rate >= 70:
            status = "🟠 NEEDS ATTENTION"
            message = "System has significant issues that require attention."
        else:
            status = "🔴 CRITICAL"
            message = "System has critical issues requiring immediate action."
        
        summary_lines.extend([
            f"## Overall Status: {status}",
            f"{message}",
            ""
        ])
        
        # Add critical issues if any
        if critical_failures:
            summary_lines.extend([
                "## Critical Issues Requiring Immediate Action",
                ""
            ])
            
            for failure in critical_failures[:5]:  # Top 5 critical issues
                summary_lines.extend([
                    f"- **{failure.test_case.feature_name}**: {failure.analysis or 'Critical test failure'}",
                    ""
                ])
        
        # Add recommendations
        if critical_failures or high_priority_failures:
            summary_lines.extend([
                "## Immediate Actions Required",
                ""
            ])
            
            if critical_failures:
                summary_lines.append("1. Address all critical failures before production deployment")
            if high_priority_failures:
                summary_lines.append("2. Review and fix high priority issues within current sprint")
            if high_risk_count > 0:
                summary_lines.append("3. Conduct risk assessment for all high-risk components")
            
            summary_lines.append("")
        
        return "\n".join(summary_lines)