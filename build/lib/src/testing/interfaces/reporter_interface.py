"""
Interface for test result reporters.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..models.test_result import TestResult


class ReporterInterface(ABC):
    """
    Abstract interface for generating test coverage reports.
    
    This interface defines the contract for components that can create
    comprehensive reports from test execution results.
    """
    
    @abstractmethod
    def generate_summary_report(self, results: List[TestResult]) -> str:
        """
        Generate a high-level summary report of test results.
        
        Args:
            results: List of TestResult objects to summarize
            
        Returns:
            String containing formatted summary report
        """
        pass
    
    @abstractmethod
    def generate_feature_report(self, feature_name: str, results: List[TestResult]) -> str:
        """
        Generate a detailed report for a specific feature.
        
        Args:
            feature_name: Name of feature to report on
            results: List of TestResult objects for this feature
            
        Returns:
            String containing formatted feature report
        """
        pass
    
    @abstractmethod
    def generate_coverage_report(self, results: List[TestResult]) -> str:
        """
        Generate a test coverage analysis report.
        
        Args:
            results: List of TestResult objects to analyze coverage
            
        Returns:
            String containing formatted coverage report
        """
        pass
    
    @abstractmethod
    def export_to_html(self, report_content: str, output_path: Path) -> None:
        """
        Export report content to HTML format.
        
        Args:
            report_content: Formatted report content
            output_path: Path where HTML file should be saved
        """
        pass
    
    def generate_regime_analysis(self, results: List[TestResult]) -> str:
        """
        Generate analysis of test results across market regimes.
        
        Args:
            results: List of TestResult objects with regime context
            
        Returns:
            String containing formatted regime analysis
        """
        # Group results by regime
        regime_results = {}
        for result in results:
            if result.test_case.regime_context:
                regime = result.test_case.regime_context
                if regime not in regime_results:
                    regime_results[regime] = []
                regime_results[regime].append(result)
        
        if not regime_results:
            return "No regime-specific test results found."
        
        # Generate analysis
        analysis_lines = ["# Market Regime Analysis\n"]
        
        for regime, regime_test_results in regime_results.items():
            passed = sum(1 for r in regime_test_results if r.passed)
            total = len(regime_test_results)
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            analysis_lines.append(f"## {regime.title()} Market Regime")
            analysis_lines.append(f"- Total Tests: {total}")
            analysis_lines.append(f"- Passed: {passed}")
            analysis_lines.append(f"- Success Rate: {success_rate:.1f}%")
            
            # Add failed test details
            failed_tests = [r for r in regime_test_results if not r.passed]
            if failed_tests:
                analysis_lines.append("- Failed Tests:")
                for failed in failed_tests:
                    analysis_lines.append(f"  - {failed.test_case.feature_name}: {failed.analysis}")
            
            analysis_lines.append("")
        
        return "\n".join(analysis_lines)
    
    def generate_trend_report(self, historical_results: List[List[TestResult]]) -> str:
        """
        Generate trend analysis from historical test results.
        
        Args:
            historical_results: List of result sets from different time periods
            
        Returns:
            String containing formatted trend analysis
        """
        if len(historical_results) < 2:
            return "Insufficient historical data for trend analysis."
        
        trend_lines = ["# Test Result Trends\n"]
        
        # Calculate success rates over time
        success_rates = []
        for result_set in historical_results:
            passed = sum(1 for r in result_set if r.passed)
            total = len(result_set)
            success_rate = (passed / total) * 100 if total > 0 else 0
            success_rates.append(success_rate)
        
        # Analyze trend
        if len(success_rates) >= 2:
            trend_direction = "improving" if success_rates[-1] > success_rates[0] else "declining"
            trend_lines.append(f"Overall trend: {trend_direction}")
            trend_lines.append(f"Latest success rate: {success_rates[-1]:.1f}%")
            trend_lines.append(f"Change from first measurement: {success_rates[-1] - success_rates[0]:+.1f}%")
        
        return "\n".join(trend_lines)
    
    def export_to_json(self, results: List[TestResult], output_path: Path) -> None:
        """
        Export test results to JSON format.
        
        Args:
            results: List of TestResult objects to export
            output_path: Path where JSON file should be saved
        """
        import json
        
        # Convert results to serializable format
        serializable_results = [result.to_dict() for result in results]
        
        # Add summary statistics
        export_data = {
            'summary': {
                'total_tests': len(results),
                'passed_tests': sum(1 for r in results if r.passed),
                'failed_tests': sum(1 for r in results if not r.passed),
                'success_rate': (sum(1 for r in results if r.passed) / len(results)) * 100 if results else 0
            },
            'results': serializable_results
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def generate_failure_analysis(self, results: List[TestResult]) -> str:
        """
        Generate detailed analysis of test failures.
        
        Args:
            results: List of TestResult objects to analyze failures
            
        Returns:
            String containing formatted failure analysis
        """
        failed_results = [r for r in results if not r.passed]
        
        if not failed_results:
            return "No test failures to analyze."
        
        analysis_lines = ["# Failure Analysis\n"]
        analysis_lines.append(f"Total failures: {len(failed_results)}\n")
        
        # Group failures by feature
        feature_failures = {}
        for result in failed_results:
            feature = result.test_case.feature_name
            if feature not in feature_failures:
                feature_failures[feature] = []
            feature_failures[feature].append(result)
        
        # Analyze each feature's failures
        for feature, failures in feature_failures.items():
            analysis_lines.append(f"## {feature}")
            analysis_lines.append(f"Failures: {len(failures)}")
            
            for failure in failures:
                analysis_lines.append(f"- {failure.test_case.test_type.value}: {failure.analysis}")
                if failure.recommendations:
                    for rec in failure.recommendations:
                        analysis_lines.append(f"  - Recommendation: {rec}")
            
            analysis_lines.append("")
        
        return "\n".join(analysis_lines)
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of output formats supported by this reporter.
        
        Returns:
            List of supported format names (e.g., ['html', 'json', 'markdown'])
        """
        pass
    
    def get_reporter_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this reporter implementation.
        
        Returns:
            Dictionary with reporter capabilities and configuration
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'supported_formats': self.get_supported_formats(),
            'report_types': [
                'summary',
                'feature_detail',
                'coverage_analysis',
                'regime_analysis',
                'trend_analysis',
                'failure_analysis'
            ]
        }