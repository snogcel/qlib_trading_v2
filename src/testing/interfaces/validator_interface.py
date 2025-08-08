"""
Interface for test result validators.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..models.test_result import TestResult, ConfidenceLevel


class ValidationInterface(ABC):
    """
    Abstract interface for validating test results and determining pass/fail status.
    
    This interface defines the contract for components that can analyze
    test results and provide detailed validation analysis.
    """
    
    @abstractmethod
    def validate_result(self, result: TestResult) -> TestResult:
        """
        Validate a test result and update its analysis.
        
        Args:
            result: TestResult object to validate
            
        Returns:
            Updated TestResult with validation analysis
        """
        pass
    
    @abstractmethod
    def validate_economic_hypothesis(self, result: TestResult) -> TestResult:
        """
        Validate results from economic hypothesis tests.
        
        Args:
            result: TestResult from economic hypothesis test
            
        Returns:
            Updated TestResult with economic validation analysis
        """
        pass
    
    @abstractmethod
    def validate_performance_metrics(self, result: TestResult) -> TestResult:
        """
        Validate results from performance characteristic tests.
        
        Args:
            result: TestResult from performance test
            
        Returns:
            Updated TestResult with performance validation analysis
        """
        pass
    
    @abstractmethod
    def validate_failure_handling(self, result: TestResult) -> TestResult:
        """
        Validate results from failure mode tests.
        
        Args:
            result: TestResult from failure mode test
            
        Returns:
            Updated TestResult with failure handling analysis
        """
        pass
    
    @abstractmethod
    def validate_implementation_correctness(self, result: TestResult) -> TestResult:
        """
        Validate results from implementation tests.
        
        Args:
            result: TestResult from implementation test
            
        Returns:
            Updated TestResult with implementation validation analysis
        """
        pass
    
    def calculate_confidence_score(self, result: TestResult) -> float:
        """
        Calculate numerical confidence score for a test result.
        
        Args:
            result: TestResult to calculate confidence for
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5
        
        # Adjust based on test execution success
        if result.status.value == "passed":
            base_confidence += 0.3
        elif result.status.value == "failed":
            base_confidence -= 0.2
        elif result.status.value == "error":
            base_confidence -= 0.4
        
        # Adjust based on data quality
        if result.data_quality_issues:
            base_confidence -= 0.1 * len(result.data_quality_issues)
        
        # Adjust based on statistical measures
        if result.statistical_measures:
            if 'p_value' in result.statistical_measures:
                p_value = result.statistical_measures['p_value']
                if p_value < 0.01:
                    base_confidence += 0.2
                elif p_value < 0.05:
                    base_confidence += 0.1
                elif p_value > 0.1:
                    base_confidence -= 0.1
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, base_confidence))
    
    def determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """
        Convert numerical confidence score to confidence level enum.
        
        Args:
            confidence_score: Numerical confidence between 0.0 and 1.0
            
        Returns:
            ConfidenceLevel enum value
        """
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def generate_recommendations(self, result: TestResult) -> List[str]:
        """
        Generate actionable recommendations based on test result.
        
        Args:
            result: TestResult to generate recommendations for
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not result.passed:
            recommendations.append(f"Investigate failure in {result.test_case.feature_name}")
            
            if result.error_message:
                recommendations.append("Review error message and stack trace for root cause")
            
            if result.data_quality_issues:
                recommendations.append("Address data quality issues before retesting")
        
        if result.confidence == ConfidenceLevel.UNCERTAIN:
            recommendations.append("Increase test data size or improve test methodology")
        
        if result.performance_metrics:
            for metric, value in result.performance_metrics.items():
                if isinstance(value, (int, float)) and value < 0.5:
                    recommendations.append(f"Improve {metric} performance (current: {value:.3f})")
        
        return recommendations
    
    def analyze_regime_performance(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Analyze test results across different market regimes.
        
        Args:
            results: List of TestResult objects from regime tests
            
        Returns:
            Dictionary with regime performance analysis
        """
        regime_analysis = {}
        
        # Group results by regime
        regime_results = {}
        for result in results:
            if result.test_case.regime_context:
                regime = result.test_case.regime_context
                if regime not in regime_results:
                    regime_results[regime] = []
                regime_results[regime].append(result)
        
        # Analyze each regime
        for regime, regime_test_results in regime_results.items():
            passed_tests = sum(1 for r in regime_test_results if r.passed)
            total_tests = len(regime_test_results)
            
            regime_analysis[regime] = {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
                'average_confidence': sum(r.confidence_score for r in regime_test_results) / total_tests if total_tests > 0 else 0.0,
                'issues': [r.analysis for r in regime_test_results if not r.passed]
            }
        
        return regime_analysis
    
    @abstractmethod
    def get_validation_criteria(self, test_type: str) -> Dict[str, Any]:
        """
        Get validation criteria for a specific test type.
        
        Args:
            test_type: Type of test to get criteria for
            
        Returns:
            Dictionary with validation criteria and thresholds
        """
        pass
    
    def get_validator_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this validator implementation.
        
        Returns:
            Dictionary with validator capabilities and configuration
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'supported_validations': [
                'economic_hypothesis',
                'performance_metrics', 
                'failure_handling',
                'implementation_correctness'
            ],
            'statistical_methods': [
                'confidence_scoring',
                'regime_analysis',
                'trend_detection'
            ]
        }