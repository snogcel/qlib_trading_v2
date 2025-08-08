"""
Result analysis utilities for comprehensive test result evaluation.

This module provides advanced analysis capabilities for test results,
including trend analysis, pattern detection, and recommendation generation.
"""

import statistics
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from ..models.test_result import TestResult, TestStatus, ConfidenceLevel
from ..models.test_case import TestType, TestPriority


class ResultAnalyzer:
    """
    Provides comprehensive analysis of test results including trends,
    patterns, and actionable insights.
    """
    
    def __init__(self):
        """Initialize the result analyzer."""
        self.analysis_cache = {}
    
    def analyze_test_suite_results(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """
        Analyze a complete test suite execution.
        
        Args:
            test_results: List of test results to analyze
            
        Returns:
            Comprehensive analysis dictionary
        """
        if not test_results:
            return {'error': 'No test results provided'}
        
        analysis = {
            'summary': self._generate_summary_statistics(test_results),
            'feature_analysis': self._analyze_by_feature(test_results),
            'test_type_analysis': self._analyze_by_test_type(test_results),
            'priority_analysis': self._analyze_by_priority(test_results),
            'confidence_analysis': self._analyze_confidence_distribution(test_results),
            'performance_analysis': self._analyze_performance_metrics(test_results),
            'failure_analysis': self._analyze_failures(test_results),
            'recommendations': self._generate_suite_recommendations(test_results),
            'quality_assessment': self._assess_suite_quality(test_results)
        }
        
        return analysis
    
    def _generate_summary_statistics(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate high-level summary statistics."""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Status breakdown
        status_counts = Counter(r.status for r in test_results)
        
        # Confidence breakdown
        confidence_counts = Counter(r.confidence for r in test_results)
        
        # Execution time statistics
        execution_times = [r.execution_time for r in test_results if r.execution_time > 0]
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'status_breakdown': {status.value: count for status, count in status_counts.items()},
            'confidence_breakdown': {conf.value: count for conf, count in confidence_counts.items()},
            'execution_stats': {
                'total_time': sum(execution_times),
                'average_time': statistics.mean(execution_times) if execution_times else 0,
                'median_time': statistics.median(execution_times) if execution_times else 0,
                'max_time': max(execution_times) if execution_times else 0
            }
        }
        
        return summary
    
    def _analyze_by_feature(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze results grouped by feature."""
        feature_results = defaultdict(list)
        
        for result in test_results:
            feature_results[result.test_case.feature_name].append(result)
        
        feature_analysis = {}
        
        for feature, results in feature_results.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            # Calculate average confidence
            confidence_scores = [r.confidence_score for r in results if r.confidence_score > 0]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            # Identify critical issues
            critical_failures = [r for r in results if not r.passed and r.test_case.priority == TestPriority.CRITICAL]
            
            # Performance metrics aggregation
            all_metrics = {}
            for result in results:
                for metric, value in result.performance_metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
            
            avg_metrics = {metric: statistics.mean(values) for metric, values in all_metrics.items()}
            
            feature_analysis[feature] = {
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': passed / total,
                'average_confidence': avg_confidence,
                'critical_failures': len(critical_failures),
                'average_metrics': avg_metrics,
                'status': 'healthy' if passed / total >= 0.8 else 'needs_attention'
            }
        
        return feature_analysis
    
    def _analyze_by_test_type(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze results grouped by test type."""
        type_results = defaultdict(list)
        
        for result in test_results:
            type_results[result.test_case.test_type].append(result)
        
        type_analysis = {}
        
        for test_type, results in type_results.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            # Calculate type-specific metrics
            confidence_scores = [r.confidence_score for r in results if r.confidence_score > 0]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            # Execution time for this type
            exec_times = [r.execution_time for r in results if r.execution_time > 0]
            avg_exec_time = statistics.mean(exec_times) if exec_times else 0
            
            type_analysis[test_type.value] = {
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': passed / total,
                'average_confidence': avg_confidence,
                'average_execution_time': avg_exec_time,
                'common_failures': self._identify_common_failures(results)
            }
        
        return type_analysis
    
    def _analyze_by_priority(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze results grouped by test priority."""
        priority_results = defaultdict(list)
        
        for result in test_results:
            priority_results[result.test_case.priority].append(result)
        
        priority_analysis = {}
        
        for priority, results in priority_results.items():
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            
            # Critical priority failures are especially important
            if priority == TestPriority.CRITICAL and passed < total:
                critical_failures = [r for r in results if not r.passed]
                failure_details = [r.get_failure_details() for r in critical_failures]
            else:
                failure_details = []
            
            priority_analysis[priority.value] = {
                'total_tests': total,
                'passed_tests': passed,
                'pass_rate': passed / total,
                'critical_failure_details': failure_details,
                'requires_immediate_attention': priority == TestPriority.CRITICAL and passed < total
            }
        
        return priority_analysis
    
    def _analyze_confidence_distribution(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze confidence score distribution and patterns."""
        confidence_scores = [r.confidence_score for r in test_results if r.confidence_score > 0]
        
        if not confidence_scores:
            return {'error': 'No confidence scores available'}
        
        # Statistical measures
        confidence_stats = {
            'mean': statistics.mean(confidence_scores),
            'median': statistics.median(confidence_scores),
            'std_dev': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            'min': min(confidence_scores),
            'max': max(confidence_scores)
        }
        
        # Distribution by confidence level
        confidence_levels = Counter(r.confidence for r in test_results)
        
        # Low confidence tests that need attention
        low_confidence_tests = [
            r for r in test_results 
            if r.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]
        ]
        
        return {
            'statistics': confidence_stats,
            'distribution': {level.value: count for level, count in confidence_levels.items()},
            'low_confidence_count': len(low_confidence_tests),
            'low_confidence_features': list(set(r.test_case.feature_name for r in low_confidence_tests))
        }
    
    def _analyze_performance_metrics(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze performance metrics across all tests."""
        all_metrics = defaultdict(list)
        
        for result in test_results:
            for metric, value in result.performance_metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric].append(value)
        
        metric_analysis = {}
        
        for metric, values in all_metrics.items():
            if values:
                metric_analysis[metric] = {
                    'count': len(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'poor_performance_count': sum(1 for v in values if v < 0.5)
                }
        
        return metric_analysis
    
    def _analyze_failures(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Analyze failure patterns and common issues."""
        failed_results = [r for r in test_results if not r.passed]
        
        if not failed_results:
            return {'total_failures': 0, 'message': 'No failures to analyze'}
        
        # Common error patterns
        error_patterns = Counter()
        for result in failed_results:
            if result.error_message:
                # Extract key error patterns
                error_type = result.error_message.split(':')[0] if ':' in result.error_message else result.error_message
                error_patterns[error_type] += 1
        
        # Failure by feature
        feature_failures = Counter(r.test_case.feature_name for r in failed_results)
        
        # Failure by test type
        type_failures = Counter(r.test_case.test_type.value for r in failed_results)
        
        # Data quality related failures
        data_quality_failures = [r for r in failed_results if r.data_quality_issues]
        
        return {
            'total_failures': len(failed_results),
            'error_patterns': dict(error_patterns.most_common(10)),
            'failures_by_feature': dict(feature_failures.most_common(10)),
            'failures_by_type': dict(type_failures),
            'data_quality_failures': len(data_quality_failures),
            'critical_failures': len([r for r in failed_results if r.test_case.priority == TestPriority.CRITICAL])
        }
    
    def _identify_common_failures(self, results: List[TestResult]) -> List[str]:
        """Identify common failure patterns in a set of results."""
        failed_results = [r for r in results if not r.passed]
        
        if not failed_results:
            return []
        
        # Extract common error messages
        error_messages = [r.error_message for r in failed_results if r.error_message]
        
        # Find common patterns (simplified)
        common_patterns = []
        for msg in error_messages:
            if 'timeout' in msg.lower():
                common_patterns.append('Timeout issues')
            elif 'data' in msg.lower():
                common_patterns.append('Data-related errors')
            elif 'calculation' in msg.lower():
                common_patterns.append('Calculation errors')
            elif 'validation' in msg.lower():
                common_patterns.append('Validation failures')
        
        return list(set(common_patterns))
    
    def _generate_suite_recommendations(self, test_results: List[TestResult]) -> List[Dict[str, Any]]:
        """Generate recommendations for the entire test suite."""
        recommendations = []
        
        # Analyze overall pass rate
        pass_rate = sum(1 for r in test_results if r.passed) / len(test_results)
        
        if pass_rate < 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'overall_health',
                'message': f'Low overall pass rate ({pass_rate:.1%}). Investigate systematic issues.',
                'action': 'Review failed tests and identify common root causes'
            })
        
        # Check for critical failures
        critical_failures = [r for r in test_results if not r.passed and r.test_case.priority == TestPriority.CRITICAL]
        if critical_failures:
            recommendations.append({
                'priority': 'critical',
                'category': 'critical_failures',
                'message': f'{len(critical_failures)} critical test failures detected',
                'action': 'Immediate investigation and resolution required'
            })
        
        # Check confidence levels
        low_confidence = [r for r in test_results if r.confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]]
        if len(low_confidence) > len(test_results) * 0.3:
            recommendations.append({
                'priority': 'medium',
                'category': 'confidence',
                'message': f'{len(low_confidence)} tests have low confidence',
                'action': 'Improve test data quality and validation criteria'
            })
        
        # Check execution time
        long_running = [r for r in test_results if r.execution_time > 60]  # > 1 minute
        if long_running:
            recommendations.append({
                'priority': 'low',
                'category': 'performance',
                'message': f'{len(long_running)} tests taking excessive time',
                'action': 'Optimize test execution or consider parallel execution'
            })
        
        # Feature-specific recommendations
        feature_analysis = self._analyze_by_feature(test_results)
        for feature, analysis in feature_analysis.items():
            if analysis['pass_rate'] < 0.5:
                recommendations.append({
                    'priority': 'high',
                    'category': 'feature_health',
                    'message': f'Feature {feature} has low pass rate ({analysis["pass_rate"]:.1%})',
                    'action': f'Review {feature} implementation and test cases'
                })
        
        return recommendations
    
    def _assess_suite_quality(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Assess the overall quality of the test suite execution."""
        quality_scores = [r.calculate_quality_score() for r in test_results]
        overall_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        # Quality categories
        high_quality = sum(1 for score in quality_scores if score >= 0.8)
        medium_quality = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        low_quality = sum(1 for score in quality_scores if score < 0.6)
        
        # Risk assessment
        high_risk_tests = [r for r in test_results if r.get_risk_assessment()['risk_level'] == 'high']
        
        return {
            'overall_quality_score': overall_quality,
            'quality_distribution': {
                'high_quality': high_quality,
                'medium_quality': medium_quality,
                'low_quality': low_quality
            },
            'high_risk_test_count': len(high_risk_tests),
            'suite_health': self._determine_suite_health(overall_quality, len(high_risk_tests), len(test_results))
        }
    
    def _determine_suite_health(self, quality_score: float, high_risk_count: int, total_tests: int) -> str:
        """Determine overall suite health status."""
        if quality_score >= 0.8 and high_risk_count == 0:
            return 'excellent'
        elif quality_score >= 0.7 and high_risk_count <= total_tests * 0.1:
            return 'good'
        elif quality_score >= 0.6 and high_risk_count <= total_tests * 0.2:
            return 'fair'
        else:
            return 'poor'
    
    def analyze_trends(self, historical_results: List[List[TestResult]]) -> Dict[str, Any]:
        """
        Analyze trends across multiple test suite executions.
        
        Args:
            historical_results: List of test result lists, ordered chronologically
            
        Returns:
            Trend analysis dictionary
        """
        if len(historical_results) < 2:
            return {'error': 'Need at least 2 historical executions for trend analysis'}
        
        # Calculate pass rates over time
        pass_rates = []
        confidence_scores = []
        execution_times = []
        
        for results in historical_results:
            if results:
                pass_rate = sum(1 for r in results if r.passed) / len(results)
                pass_rates.append(pass_rate)
                
                conf_scores = [r.confidence_score for r in results if r.confidence_score > 0]
                avg_confidence = statistics.mean(conf_scores) if conf_scores else 0
                confidence_scores.append(avg_confidence)
                
                exec_times = [r.execution_time for r in results if r.execution_time > 0]
                avg_exec_time = statistics.mean(exec_times) if exec_times else 0
                execution_times.append(avg_exec_time)
        
        # Calculate trends
        trends = {
            'pass_rate_trend': self._calculate_trend(pass_rates),
            'confidence_trend': self._calculate_trend(confidence_scores),
            'execution_time_trend': self._calculate_trend(execution_times),
            'overall_trend': 'stable'
        }
        
        # Determine overall trend
        if trends['pass_rate_trend'] == 'improving' and trends['confidence_trend'] == 'improving':
            trends['overall_trend'] = 'improving'
        elif trends['pass_rate_trend'] == 'declining' or trends['confidence_trend'] == 'declining':
            trends['overall_trend'] = 'declining'
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'