"""
Result validation engine for the feature test coverage system.

This module provides comprehensive validation and analysis of test results,
including pass/fail determination, confidence scoring, and recommendation generation.
"""

import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.test_result import TestResult, TestStatus, ConfidenceLevel
from ..models.test_case import TestCase, TestType, TestPriority


class ValidationCriteria(Enum):
    """Validation criteria types for different test categories."""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    THRESHOLD_COMPARISON = "threshold_comparison"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    FORMULA_VERIFICATION = "formula_verification"
    REGIME_CONSISTENCY = "regime_consistency"
    INTERACTION_VALIDATION = "interaction_validation"


@dataclass
class ValidationResult:
    """Result of validation analysis."""
    passed: bool
    confidence_level: ConfidenceLevel
    confidence_score: float
    analysis: str
    recommendations: List[str]
    severity: str
    statistical_measures: Dict[str, float]
    performance_metrics: Dict[str, float]


class ResultValidator:
    """
    Validates test results and provides comprehensive analysis.
    
    This class implements the core validation logic for determining
    test pass/fail status, confidence scoring, and recommendation generation.
    """
    
    def __init__(self):
        """Initialize the result validator with default thresholds."""
        self.confidence_thresholds = {
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MEDIUM: 0.65,
            ConfidenceLevel.LOW: 0.45,
            ConfidenceLevel.UNCERTAIN: 0.0
        }
        
        self.statistical_significance_threshold = 0.05  # p-value threshold
        self.performance_tolerance = 0.1  # 10% tolerance for performance metrics
        
    def validate_test_result(self, test_result: TestResult) -> ValidationResult:
        """
        Perform comprehensive validation of a test result.
        
        Args:
            test_result: TestResult object to validate
            
        Returns:
            ValidationResult with detailed analysis
        """
        test_case = test_result.test_case
        
        # Route to appropriate validation method based on test type
        if test_case.test_type == TestType.ECONOMIC_HYPOTHESIS:
            return self._validate_economic_hypothesis(test_result)
        elif test_case.test_type == TestType.PERFORMANCE:
            return self._validate_performance_characteristics(test_result)
        elif test_case.test_type == TestType.FAILURE_MODE:
            return self._validate_failure_mode(test_result)
        elif test_case.test_type == TestType.IMPLEMENTATION:
            return self._validate_implementation(test_result)
        elif test_case.test_type == TestType.REGIME_DEPENDENCY:
            return self._validate_regime_dependency(test_result)
        elif test_case.test_type == TestType.INTERACTION:
            return self._validate_interaction(test_result)
        else:
            return self._validate_generic(test_result)
    
    def _validate_economic_hypothesis(self, test_result: TestResult) -> ValidationResult:
        """Validate economic hypothesis test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        expected = test_result.expected_result
        
        # Initialize validation result
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status != TestStatus.PASSED:
            validation.analysis = f"Test execution failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Fix test execution issues before validating hypothesis")
            validation.severity = "high"
            return validation
        
        # Analyze hypothesis validation based on test parameters
        hypothesis_params = test_case.test_parameters.get('hypothesis', {})
        expected_behavior = test_case.test_parameters.get('expected_behavior', {})
        
        if isinstance(actual, dict) and 'directional_bias' in actual:
            # Q50 directional bias validation
            bias_strength = actual.get('directional_bias', 0.0)
            expected_bias = expected_behavior.get('min_directional_bias', 0.55)
            
            validation.statistical_measures['directional_bias'] = bias_strength
            validation.statistical_measures['expected_bias'] = expected_bias
            
            if bias_strength >= expected_bias:
                validation.passed = True
                validation.confidence_score = min(1.0, bias_strength / expected_bias)
                validation.analysis = f"Directional bias ({bias_strength:.3f}) meets expected threshold ({expected_bias:.3f})"
            else:
                validation.analysis = f"Directional bias ({bias_strength:.3f}) below expected threshold ({expected_bias:.3f})"
                validation.recommendations.append("Investigate feature calculation or market regime classification")
        
        elif isinstance(actual, dict) and 'variance_capture' in actual:
            # vol_risk variance capture validation
            variance_accuracy = actual.get('variance_capture', 0.0)
            expected_accuracy = expected_behavior.get('min_variance_accuracy', 0.7)
            
            validation.statistical_measures['variance_accuracy'] = variance_accuracy
            validation.statistical_measures['expected_accuracy'] = expected_accuracy
            
            if variance_accuracy >= expected_accuracy:
                validation.passed = True
                validation.confidence_score = variance_accuracy
                validation.analysis = f"Variance capture accuracy ({variance_accuracy:.3f}) meets expectations"
            else:
                validation.analysis = f"Variance capture accuracy ({variance_accuracy:.3f}) below threshold"
                validation.recommendations.append("Review volatility calculation methodology")
        
        elif isinstance(actual, dict) and 'sentiment_correlation' in actual:
            # Sentiment feature validation
            correlation = actual.get('sentiment_correlation', 0.0)
            expected_correlation = expected_behavior.get('min_correlation', 0.3)
            
            validation.statistical_measures['sentiment_correlation'] = correlation
            validation.statistical_measures['expected_correlation'] = expected_correlation
            
            if abs(correlation) >= expected_correlation:
                validation.passed = True
                validation.confidence_score = min(1.0, abs(correlation) / expected_correlation)
                validation.analysis = f"Sentiment correlation ({correlation:.3f}) shows expected relationship"
            else:
                validation.analysis = f"Sentiment correlation ({correlation:.3f}) weaker than expected"
                validation.recommendations.append("Verify sentiment data quality and feature calculation")
        
        # Set confidence level based on score
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        # Add general recommendations based on confidence
        if validation.confidence_level == ConfidenceLevel.LOW:
            validation.recommendations.append("Consider additional validation with different market periods")
        elif validation.confidence_level == ConfidenceLevel.UNCERTAIN:
            validation.recommendations.append("Requires manual review - insufficient evidence for automated validation")
        
        return validation
    
    def _validate_performance_characteristics(self, test_result: TestResult) -> ValidationResult:
        """Validate performance characteristics test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status != TestStatus.PASSED:
            validation.analysis = f"Test execution failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Fix test execution issues before validating performance")
            validation.severity = "high"
            return validation
        
        # Extract performance metrics and thresholds
        metrics = test_case.test_parameters.get('metrics', {})
        thresholds = test_case.test_parameters.get('thresholds', {})
        
        if not isinstance(actual, dict):
            validation.analysis = "Invalid performance test result format"
            validation.recommendations.append("Ensure performance tests return dictionary with metrics")
            return validation
        
        # Validate each performance metric
        passed_metrics = 0
        total_metrics = len(metrics)
        metric_scores = []
        
        for metric_name, expected_range in metrics.items():
            if metric_name not in actual:
                validation.recommendations.append(f"Missing performance metric: {metric_name}")
                continue
            
            actual_value = actual[metric_name]
            validation.performance_metrics[metric_name] = actual_value
            
            # Check if metric is within expected range
            if isinstance(expected_range, dict):
                min_val = expected_range.get('min', float('-inf'))
                max_val = expected_range.get('max', float('inf'))
                
                if min_val <= actual_value <= max_val:
                    passed_metrics += 1
                    # Calculate how well the metric performs within range
                    if min_val != float('-inf') and max_val != float('inf'):
                        range_position = (actual_value - min_val) / (max_val - min_val)
                        metric_scores.append(min(1.0, max(0.0, range_position)))
                    else:
                        metric_scores.append(1.0)
                else:
                    validation.recommendations.append(
                        f"{metric_name} ({actual_value:.3f}) outside expected range [{min_val}, {max_val}]"
                    )
                    metric_scores.append(0.0)
            
            elif isinstance(expected_range, (int, float)):
                # Single threshold comparison
                threshold = thresholds.get(metric_name, expected_range)
                if actual_value >= threshold:
                    passed_metrics += 1
                    metric_scores.append(min(1.0, actual_value / threshold))
                else:
                    validation.recommendations.append(
                        f"{metric_name} ({actual_value:.3f}) below threshold ({threshold:.3f})"
                    )
                    metric_scores.append(actual_value / threshold if threshold > 0 else 0.0)
        
        # Calculate overall performance score
        if total_metrics > 0:
            validation.confidence_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
            pass_rate = passed_metrics / total_metrics
            
            validation.statistical_measures['pass_rate'] = pass_rate
            validation.statistical_measures['total_metrics'] = total_metrics
            validation.statistical_measures['passed_metrics'] = passed_metrics
            
            # Test passes if majority of metrics pass and confidence is reasonable
            validation.passed = pass_rate >= 0.7 and validation.confidence_score >= 0.5
            
            if validation.passed:
                validation.analysis = f"Performance validation passed: {passed_metrics}/{total_metrics} metrics within expected ranges"
            else:
                validation.analysis = f"Performance validation failed: {passed_metrics}/{total_metrics} metrics within expected ranges"
                
                if pass_rate < 0.5:
                    validation.severity = "high"
                    validation.recommendations.append("Multiple performance metrics failing - investigate feature implementation")
        
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        return validation
    
    def _validate_failure_mode(self, test_result: TestResult) -> ValidationResult:
        """Validate failure mode test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        # For failure mode tests, we expect the system to handle failures gracefully
        failure_condition = test_case.test_parameters.get('failure_condition', '')
        expected_response = test_case.test_parameters.get('expected_response', {})
        
        if isinstance(actual, dict):
            # Check if the system responded appropriately to failure condition
            graceful_failure = actual.get('graceful_failure', False)
            warning_generated = actual.get('warning_generated', False)
            error_handled = actual.get('error_handled', False)
            
            validation.statistical_measures['graceful_failure'] = graceful_failure
            validation.statistical_measures['warning_generated'] = warning_generated
            validation.statistical_measures['error_handled'] = error_handled
            
            # Calculate confidence based on expected response
            response_score = 0.0
            response_count = 0
            
            if 'graceful_failure' in expected_response:
                response_count += 1
                if graceful_failure == expected_response['graceful_failure']:
                    response_score += 1.0
            
            if 'warning_generated' in expected_response:
                response_count += 1
                if warning_generated == expected_response['warning_generated']:
                    response_score += 1.0
            
            if 'error_handled' in expected_response:
                response_count += 1
                if error_handled == expected_response['error_handled']:
                    response_score += 1.0
            
            if response_count > 0:
                validation.confidence_score = response_score / response_count
                validation.passed = validation.confidence_score >= 0.8
                
                if validation.passed:
                    validation.analysis = f"Failure mode '{failure_condition}' handled appropriately"
                else:
                    validation.analysis = f"Failure mode '{failure_condition}' not handled as expected"
                    validation.recommendations.append("Review error handling logic for this failure condition")
            else:
                validation.analysis = "No expected response criteria defined for failure mode test"
                validation.recommendations.append("Define expected response criteria for failure mode validation")
        
        elif test_result.status == TestStatus.ERROR:
            # If test errored, check if this was expected
            if expected_response.get('should_error', False):
                validation.passed = True
                validation.confidence_score = 1.0
                validation.analysis = "Expected error occurred during failure mode test"
            else:
                validation.analysis = f"Unexpected error during failure mode test: {test_result.error_message}"
                validation.recommendations.append("Fix unexpected errors in failure mode handling")
                validation.severity = "high"
        
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        return validation
    
    def _validate_implementation(self, test_result: TestResult) -> ValidationResult:
        """Validate implementation test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        expected = test_result.expected_result
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status != TestStatus.PASSED:
            validation.analysis = f"Implementation test execution failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Fix implementation test execution issues")
            validation.severity = "high"
            return validation
        
        # Check formula verification
        if isinstance(actual, dict) and 'formula_match' in actual:
            formula_accuracy = actual.get('formula_match', 0.0)
            calculation_error = actual.get('calculation_error', float('inf'))
            
            validation.statistical_measures['formula_accuracy'] = formula_accuracy
            validation.statistical_measures['calculation_error'] = calculation_error
            
            # Implementation passes if formula matches closely
            tolerance = test_case.tolerance or 0.01
            
            if formula_accuracy >= (1.0 - tolerance) and calculation_error <= tolerance:
                validation.passed = True
                validation.confidence_score = formula_accuracy
                validation.analysis = f"Implementation matches specification (accuracy: {formula_accuracy:.4f})"
            else:
                validation.analysis = f"Implementation deviation detected (accuracy: {formula_accuracy:.4f}, error: {calculation_error:.4f})"
                validation.recommendations.append("Review implementation against documented specification")
                
                if calculation_error > 0.1:
                    validation.severity = "high"
                    validation.recommendations.append("Significant calculation error - immediate review required")
        
        # Check normalization validation
        elif isinstance(actual, dict) and 'normalization_valid' in actual:
            normalization_valid = actual.get('normalization_valid', False)
            range_compliance = actual.get('range_compliance', 0.0)
            
            validation.statistical_measures['normalization_valid'] = normalization_valid
            validation.statistical_measures['range_compliance'] = range_compliance
            
            if normalization_valid and range_compliance >= 0.95:
                validation.passed = True
                validation.confidence_score = range_compliance
                validation.analysis = "Normalization logic validated successfully"
            else:
                validation.analysis = "Normalization issues detected"
                validation.recommendations.append("Review normalization implementation and expected ranges")
        
        # Check temporal integrity
        elif isinstance(actual, dict) and 'temporal_integrity' in actual:
            temporal_valid = actual.get('temporal_integrity', False)
            causality_preserved = actual.get('causality_preserved', False)
            
            validation.statistical_measures['temporal_integrity'] = temporal_valid
            validation.statistical_measures['causality_preserved'] = causality_preserved
            
            if temporal_valid and causality_preserved:
                validation.passed = True
                validation.confidence_score = 1.0
                validation.analysis = "Temporal integrity and causality validated"
            else:
                validation.analysis = "Temporal integrity issues detected"
                validation.recommendations.append("Review temporal feature implementation for causality violations")
                validation.severity = "high"
        
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        return validation
    
    def _validate_regime_dependency(self, test_result: TestResult) -> ValidationResult:
        """Validate regime dependency test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status != TestStatus.PASSED:
            validation.analysis = f"Regime dependency test failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Fix regime dependency test execution")
            validation.severity = "high"
            return validation
        
        if isinstance(actual, dict):
            regime_performance = actual.get('regime_performance', {})
            regime_consistency = actual.get('regime_consistency', 0.0)
            adaptation_score = actual.get('adaptation_score', 0.0)
            
            validation.statistical_measures['regime_consistency'] = regime_consistency
            validation.statistical_measures['adaptation_score'] = adaptation_score
            
            # Store regime-specific performance
            for regime, performance in regime_performance.items():
                validation.performance_metrics[f'{regime}_performance'] = performance
            
            # Validate regime consistency and adaptation
            if regime_consistency >= 0.7 and adaptation_score >= 0.6:
                validation.passed = True
                validation.confidence_score = (regime_consistency + adaptation_score) / 2
                validation.analysis = f"Regime dependency validated (consistency: {regime_consistency:.3f}, adaptation: {adaptation_score:.3f})"
            else:
                validation.analysis = f"Regime dependency issues (consistency: {regime_consistency:.3f}, adaptation: {adaptation_score:.3f})"
                validation.recommendations.append("Review feature behavior across different market regimes")
                
                if regime_consistency < 0.5:
                    validation.severity = "high"
                    validation.recommendations.append("Inconsistent regime behavior - investigate regime classification")
        
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        return validation
    
    def _validate_interaction(self, test_result: TestResult) -> ValidationResult:
        """Validate feature interaction test results."""
        test_case = test_result.test_case
        actual = test_result.actual_result
        
        validation = ValidationResult(
            passed=False,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            confidence_score=0.0,
            analysis="",
            recommendations=[],
            severity="medium",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status != TestStatus.PASSED:
            validation.analysis = f"Interaction test failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Fix interaction test execution")
            validation.severity = "high"
            return validation
        
        if isinstance(actual, dict):
            synergy_score = actual.get('synergy_score', 0.0)
            conflict_detected = actual.get('conflict_detected', False)
            interaction_strength = actual.get('interaction_strength', 0.0)
            
            validation.statistical_measures['synergy_score'] = synergy_score
            validation.statistical_measures['conflict_detected'] = conflict_detected
            validation.statistical_measures['interaction_strength'] = interaction_strength
            
            # Positive interaction validation
            if synergy_score > 0.1 and not conflict_detected:
                validation.passed = True
                validation.confidence_score = min(1.0, synergy_score)
                validation.analysis = f"Positive feature interaction validated (synergy: {synergy_score:.3f})"
            elif conflict_detected:
                validation.analysis = "Feature conflict detected"
                validation.recommendations.append("Investigate negative feature interactions")
                validation.severity = "high"
            else:
                validation.analysis = f"Weak feature interaction (synergy: {synergy_score:.3f})"
                validation.recommendations.append("Consider feature combination effectiveness")
        
        validation.confidence_level = self._determine_confidence_level(validation.confidence_score)
        
        return validation
    
    def _validate_generic(self, test_result: TestResult) -> ValidationResult:
        """Generic validation for unknown test types."""
        validation = ValidationResult(
            passed=test_result.passed,
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            analysis="Generic validation - test type not specifically handled",
            recommendations=["Implement specific validation logic for this test type"],
            severity="low",
            statistical_measures={},
            performance_metrics={}
        )
        
        if test_result.status == TestStatus.PASSED:
            validation.analysis = "Test passed with generic validation"
        else:
            validation.analysis = f"Test failed: {test_result.error_message or 'Unknown error'}"
            validation.recommendations.append("Review test execution and expected results")
        
        return validation
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level based on numerical score."""
        if confidence_score >= self.confidence_thresholds[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif confidence_score >= self.confidence_thresholds[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= self.confidence_thresholds[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def generate_recommendations(self, test_result: TestResult, validation: ValidationResult) -> List[str]:
        """
        Generate specific recommendations based on test results and validation.
        
        Args:
            test_result: Original test result
            validation: Validation analysis result
            
        Returns:
            List of actionable recommendations
        """
        recommendations = validation.recommendations.copy()
        
        # Add priority-based recommendations
        if test_result.test_case.priority == TestPriority.CRITICAL and not validation.passed:
            recommendations.insert(0, "CRITICAL: Immediate attention required - core system functionality affected")
        
        # Add confidence-based recommendations
        if validation.confidence_level == ConfidenceLevel.UNCERTAIN:
            recommendations.append("Consider manual review due to uncertain validation results")
        elif validation.confidence_level == ConfidenceLevel.LOW:
            recommendations.append("Increase test coverage or data quality to improve confidence")
        
        # Add test-type specific recommendations
        test_type = test_result.test_case.test_type
        
        if test_type == TestType.ECONOMIC_HYPOTHESIS and not validation.passed:
            recommendations.append("Review economic assumptions and market data used for hypothesis testing")
        elif test_type == TestType.PERFORMANCE and not validation.passed:
            recommendations.append("Analyze performance degradation patterns and potential causes")
        elif test_type == TestType.IMPLEMENTATION and not validation.passed:
            recommendations.append("Compare implementation against specification documentation")
        
        # Add data quality recommendations
        if test_result.data_quality_issues:
            recommendations.append("Address data quality issues before re-running validation")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def calculate_overall_confidence(self, test_results: List[TestResult]) -> Tuple[float, ConfidenceLevel]:
        """
        Calculate overall confidence across multiple test results.
        
        Args:
            test_results: List of test results to analyze
            
        Returns:
            Tuple of (confidence_score, confidence_level)
        """
        if not test_results:
            return 0.0, ConfidenceLevel.UNCERTAIN
        
        # Weight confidence scores by test priority
        weighted_scores = []
        total_weight = 0
        
        for result in test_results:
            weight = self._get_priority_weight(result.test_case.priority)
            weighted_scores.append(result.confidence_score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, ConfidenceLevel.UNCERTAIN
        
        overall_score = sum(weighted_scores) / total_weight
        overall_level = self._determine_confidence_level(overall_score)
        
        return overall_score, overall_level
    
    def _get_priority_weight(self, priority: TestPriority) -> float:
        """Get weight for test priority in confidence calculations."""
        weights = {
            TestPriority.CRITICAL: 3.0,
            TestPriority.HIGH: 2.0,
            TestPriority.MEDIUM: 1.0,
            TestPriority.LOW: 0.5
        }
        return weights.get(priority, 1.0)