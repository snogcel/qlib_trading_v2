"""
Unit tests for the result validation system.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pytest
from datetime import datetime
from unittest.mock import Mock

from src.testing.models.test_result import TestResult, TestStatus, ConfidenceLevel
from src.testing.models.test_case import TestCase, TestType, TestPriority
from src.testing.validation.result_validator import ResultValidator, ValidationResult
from src.testing.validation.result_analyzer import ResultAnalyzer


class TestResultValidator:
    """Test the ResultValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ResultValidator()
        
        # Create a mock test case
        self.mock_test_case = Mock(spec=TestCase)
        self.mock_test_case.test_id = "test_001"
        self.mock_test_case.feature_name = "Q50"
        self.mock_test_case.test_type = TestType.ECONOMIC_HYPOTHESIS
        self.mock_test_case.priority = TestPriority.HIGH
        self.mock_test_case.tolerance = 0.05
        self.mock_test_case.test_parameters = {
            'hypothesis': 'Q50 provides directional bias',
            'expected_behavior': {'min_directional_bias': 0.55}
        }
    
    def test_validate_economic_hypothesis_success(self):
        """Test successful economic hypothesis validation."""
        # Create test result with successful directional bias
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_001",
            status=TestStatus.PASSED,
            execution_time=1.5,
            actual_result={'directional_bias': 0.65},
            passed=True,
            confidence_score=0.8
        )
        
        validation = self.validator.validate_test_result(test_result)
        
        assert validation.passed is True
        assert validation.confidence_level == ConfidenceLevel.HIGH  # Should be HIGH for 1.0 confidence
        assert 'directional_bias' in validation.statistical_measures
        assert validation.statistical_measures['directional_bias'] == 0.65
        assert 'meets expected threshold' in validation.analysis
    
    def test_validate_economic_hypothesis_failure(self):
        """Test failed economic hypothesis validation."""
        # Create test result with insufficient directional bias
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_002",
            status=TestStatus.PASSED,
            execution_time=1.5,
            actual_result={'directional_bias': 0.45},
            passed=True,
            confidence_score=0.3
        )
        
        validation = self.validator.validate_test_result(test_result)
        
        assert validation.passed is False
        assert 'below expected threshold' in validation.analysis
        assert len(validation.recommendations) > 0
        assert 'Investigate feature calculation' in validation.recommendations[0]
    
    def test_validate_performance_characteristics(self):
        """Test performance characteristics validation."""
        # Set up performance test case
        self.mock_test_case.test_type = TestType.PERFORMANCE
        self.mock_test_case.test_parameters = {
            'metrics': {
                'sharpe_ratio': {'min': 0.5, 'max': 2.0},
                'hit_rate': {'min': 0.55, 'max': 0.8}
            },
            'thresholds': {}
        }
        
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_003",
            status=TestStatus.PASSED,
            execution_time=2.0,
            actual_result={
                'sharpe_ratio': 1.2,
                'hit_rate': 0.62
            },
            passed=True
        )
        
        validation = self.validator.validate_test_result(test_result)
        
        assert 'sharpe_ratio' in validation.performance_metrics
        assert 'hit_rate' in validation.performance_metrics
        assert validation.statistical_measures['passed_metrics'] == 2
        assert validation.statistical_measures['total_metrics'] == 2
        assert validation.statistical_measures['pass_rate'] == 1.0
        
        # The confidence score is calculated based on range position
        # sharpe_ratio: 1.2 in range [0.5, 2.0] -> position = (1.2-0.5)/(2.0-0.5) = 0.47
        # hit_rate: 0.62 in range [0.55, 0.8] -> position = (0.62-0.55)/(0.8-0.55) = 0.28
        # Average = (0.47 + 0.28) / 2 = 0.375, which is < 0.5
        # So the test should fail due to low confidence, even though metrics are in range
        assert validation.passed is False  # Fails due to low confidence score
        assert validation.confidence_score < 0.5
    
    def test_validate_failure_mode(self):
        """Test failure mode validation."""
        # Set up failure mode test case
        self.mock_test_case.test_type = TestType.FAILURE_MODE
        self.mock_test_case.test_parameters = {
            'failure_condition': 'low_liquidity',
            'expected_response': {
                'graceful_failure': True,
                'warning_generated': True
            }
        }
        
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_004",
            status=TestStatus.PASSED,
            execution_time=1.0,
            actual_result={
                'graceful_failure': True,
                'warning_generated': True,
                'error_handled': False
            },
            passed=True
        )
        
        validation = self.validator.validate_test_result(test_result)
        
        assert validation.passed is True
        assert validation.confidence_score == 1.0  # 2/2 expected responses matched
        assert 'handled appropriately' in validation.analysis
    
    def test_validate_implementation(self):
        """Test implementation validation."""
        # Set up implementation test case
        self.mock_test_case.test_type = TestType.IMPLEMENTATION
        self.mock_test_case.tolerance = 0.01
        
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_005",
            status=TestStatus.PASSED,
            execution_time=0.5,
            actual_result={
                'formula_match': 0.995,
                'calculation_error': 0.005
            },
            passed=True
        )
        
        validation = self.validator.validate_test_result(test_result)
        
        assert validation.passed is True
        assert validation.confidence_score == 0.995
        assert 'matches specification' in validation.analysis
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        test_result = TestResult(
            test_case=self.mock_test_case,
            execution_id="exec_006",
            status=TestStatus.FAILED,
            execution_time=1.0,
            passed=False,
            confidence=ConfidenceLevel.LOW,
            data_quality_issues=['missing_data', 'outliers']
        )
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.LOW,
            confidence_score=0.3,
            analysis="Test failed due to data issues",
            recommendations=["Review data quality"],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        recommendations = self.validator.generate_recommendations(test_result, validation)
        
        assert len(recommendations) > 0
        assert any('data quality' in rec.lower() for rec in recommendations)
        assert any('confidence' in rec.lower() for rec in recommendations)
    
    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        # Create multiple test results with different priorities
        results = []
        
        # Critical test - high confidence
        critical_case = Mock(spec=TestCase)
        critical_case.priority = TestPriority.CRITICAL
        results.append(TestResult(
            test_case=critical_case,
            execution_id="exec_007",
            status=TestStatus.PASSED,
            execution_time=1.0,
            confidence_score=0.9,
            passed=True
        ))
        
        # Medium test - medium confidence
        medium_case = Mock(spec=TestCase)
        medium_case.priority = TestPriority.MEDIUM
        results.append(TestResult(
            test_case=medium_case,
            execution_id="exec_008",
            status=TestStatus.PASSED,
            execution_time=1.0,
            confidence_score=0.6,
            passed=True
        ))
        
        overall_score, overall_level = self.validator.calculate_overall_confidence(results)
        
        # Should be weighted toward the critical test
        assert overall_score > 0.6  # Higher than simple average due to weighting
        assert overall_level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH]


class TestResultAnalyzer:
    """Test the ResultAnalyzer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ResultAnalyzer()
        
        # Create sample test results
        self.test_results = []
        
        for i in range(5):
            mock_case = Mock(spec=TestCase)
            mock_case.feature_name = f"feature_{i % 3}"  # 3 different features
            mock_case.test_type = TestType.PERFORMANCE
            mock_case.priority = TestPriority.HIGH if i < 2 else TestPriority.MEDIUM
            
            result = TestResult(
                test_case=mock_case,
                execution_id=f"exec_{i:03d}",
                status=TestStatus.PASSED if i < 4 else TestStatus.FAILED,
                execution_time=float(i + 1),
                passed=i < 4,
                confidence_score=0.8 if i < 4 else 0.3,
                confidence=ConfidenceLevel.HIGH if i < 4 else ConfidenceLevel.LOW,
                performance_metrics={'metric_1': 0.7 + i * 0.05}
            )
            
            self.test_results.append(result)
    
    def test_analyze_test_suite_results(self):
        """Test comprehensive test suite analysis."""
        analysis = self.analyzer.analyze_test_suite_results(self.test_results)
        
        # Check summary statistics
        assert 'summary' in analysis
        assert analysis['summary']['total_tests'] == 5
        assert analysis['summary']['passed_tests'] == 4
        assert analysis['summary']['pass_rate'] == 0.8
        
        # Check feature analysis
        assert 'feature_analysis' in analysis
        assert len(analysis['feature_analysis']) == 3  # 3 different features
        
        # Check test type analysis
        assert 'test_type_analysis' in analysis
        assert 'performance' in analysis['test_type_analysis']
        
        # Check recommendations
        assert 'recommendations' in analysis
        assert isinstance(analysis['recommendations'], list)
    
    def test_analyze_by_feature(self):
        """Test feature-specific analysis."""
        feature_analysis = self.analyzer._analyze_by_feature(self.test_results)
        
        assert len(feature_analysis) == 3
        
        for feature, analysis in feature_analysis.items():
            assert 'total_tests' in analysis
            assert 'pass_rate' in analysis
            assert 'average_confidence' in analysis
            assert 'status' in analysis
    
    def test_analyze_confidence_distribution(self):
        """Test confidence distribution analysis."""
        confidence_analysis = self.analyzer._analyze_confidence_distribution(self.test_results)
        
        assert 'statistics' in confidence_analysis
        assert 'distribution' in confidence_analysis
        assert 'low_confidence_count' in confidence_analysis
        
        # Should have statistics for confidence scores
        stats = confidence_analysis['statistics']
        assert 'mean' in stats
        assert 'median' in stats
        assert stats['mean'] > 0
    
    def test_analyze_failures(self):
        """Test failure analysis."""
        failure_analysis = self.analyzer._analyze_failures(self.test_results)
        
        assert 'total_failures' in failure_analysis
        assert failure_analysis['total_failures'] == 1  # One failed test
        
        if failure_analysis['total_failures'] > 0:
            assert 'failures_by_feature' in failure_analysis
            assert 'failures_by_type' in failure_analysis
    
    def test_generate_suite_recommendations(self):
        """Test suite-level recommendation generation."""
        recommendations = self.analyzer._generate_suite_recommendations(self.test_results)
        
        assert isinstance(recommendations, list)
        
        # Should have recommendations for the failed test
        for rec in recommendations:
            assert 'priority' in rec
            assert 'category' in rec
            assert 'message' in rec
            assert 'action' in rec
    
    def test_assess_suite_quality(self):
        """Test suite quality assessment."""
        quality_assessment = self.analyzer._assess_suite_quality(self.test_results)
        
        assert 'overall_quality_score' in quality_assessment
        assert 'quality_distribution' in quality_assessment
        assert 'suite_health' in quality_assessment
        
        # Quality score should be reasonable
        assert 0 <= quality_assessment['overall_quality_score'] <= 1
        
        # Suite health should be a valid status
        valid_health = ['excellent', 'good', 'fair', 'poor']
        assert quality_assessment['suite_health'] in valid_health
    
    def test_analyze_trends(self):
        """Test trend analysis across multiple executions."""
        # Create historical results (3 executions)
        historical_results = [
            self.test_results[:3],  # First execution - 3 tests
            self.test_results[:4],  # Second execution - 4 tests  
            self.test_results       # Third execution - 5 tests
        ]
        
        trends = self.analyzer.analyze_trends(historical_results)
        
        assert 'pass_rate_trend' in trends
        assert 'confidence_trend' in trends
        assert 'execution_time_trend' in trends
        assert 'overall_trend' in trends
        
        # Trends should be valid values
        valid_trends = ['improving', 'declining', 'stable', 'insufficient_data']
        assert trends['pass_rate_trend'] in valid_trends
        assert trends['overall_trend'] in valid_trends


class TestTestResultEnhancements:
    """Test the enhanced TestResult methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        mock_case = Mock(spec=TestCase)
        mock_case.feature_name = "test_feature"
        mock_case.test_type = TestType.PERFORMANCE
        mock_case.priority = TestPriority.HIGH
        
        self.test_result = TestResult(
            test_case=mock_case,
            execution_id="exec_test",
            status=TestStatus.PASSED,
            execution_time=2.5,
            passed=True,
            confidence_score=0.75,
            confidence=ConfidenceLevel.MEDIUM,
            performance_metrics={'metric_1': 0.8, 'metric_2': 0.6},
            analysis="Test completed successfully"
        )
    
    def test_validate_result_completeness(self):
        """Test result completeness validation."""
        issues = self.test_result.validate_result_completeness()
        
        # Should have no issues for a well-formed result
        assert len(issues) == 0
        
        # Test with missing execution ID
        self.test_result.execution_id = ""
        issues = self.test_result.validate_result_completeness()
        assert len(issues) > 0
        assert any('execution ID' in issue for issue in issues)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        quality_score = self.test_result.calculate_quality_score()
        
        assert 0 <= quality_score <= 1
        assert quality_score > 0.5  # Should be decent quality
    
    def test_get_risk_assessment(self):
        """Test risk assessment."""
        risk_assessment = self.test_result.get_risk_assessment()
        
        assert 'risk_level' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'mitigation_actions' in risk_assessment
        assert 'requires_immediate_attention' in risk_assessment
        
        # Should be low risk for a passing test
        assert risk_assessment['risk_level'] == 'low'
        assert not risk_assessment['requires_immediate_attention']
    
    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        summary = self.test_result.generate_executive_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert self.test_result.test_case.feature_name in summary
        assert '✅' in summary  # Should have pass emoji
    
    def test_compare_with_baseline(self):
        """Test baseline comparison."""
        # Create baseline result
        baseline_case = Mock(spec=TestCase)
        baseline_case.feature_name = "test_feature"
        baseline_case.test_type = TestType.PERFORMANCE
        baseline_case.priority = TestPriority.HIGH
        
        baseline_result = TestResult(
            test_case=baseline_case,
            execution_id="baseline_exec",
            status=TestStatus.PASSED,
            execution_time=2.0,
            passed=True,
            confidence_score=0.7,
            performance_metrics={'metric_1': 0.75, 'metric_2': 0.65}
        )
        
        comparison = self.test_result.compare_with_baseline(baseline_result)
        
        assert 'status_changed' in comparison
        assert 'confidence_changed' in comparison
        assert 'performance_changes' in comparison
        assert 'overall_trend' in comparison
        
        # Should detect improvement in confidence
        assert not comparison['status_changed']  # Both passed
        # The confidence difference is 0.05 (0.75 - 0.7), which is less than 0.1 threshold
        # So it should be 'stable', not 'improved'
        assert comparison['overall_trend'] == 'stable'
    
    def test_to_dict_enhanced(self):
        """Test enhanced dictionary conversion."""
        result_dict = self.test_result.to_dict()
        
        # Should include all standard fields
        assert 'test_case' in result_dict
        assert 'execution_id' in result_dict
        assert 'status' in result_dict
        
        # Should include enhanced fields
        assert 'quality_score' in result_dict
        assert 'risk_assessment' in result_dict
        
        # Quality score should be calculated
        assert isinstance(result_dict['quality_score'], float)
        assert 0 <= result_dict['quality_score'] <= 1