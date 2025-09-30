"""
Test result data model for the feature test coverage system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from .test_case import TestCase


class TestStatus(Enum):
    """Test execution status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class ConfidenceLevel(Enum):
    """Confidence level in test results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class TestResult:
    """
    Represents the result of executing a test case.
    
    This class captures all information about test execution including
    results, analysis, and recommendations for action.
    """
    
    # Basic identification
    test_case: TestCase
    execution_id: str
    
    # Execution details
    status: TestStatus
    execution_time: float  # Actual execution time in seconds
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Results
    actual_result: Any = None
    expected_result: Any = None
    
    # Analysis
    passed: bool = False
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.0  # Numerical confidence (0-1)
    
    # Detailed analysis
    analysis: str = ""
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Metrics and measurements
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    statistical_measures: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations and actions
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    severity: str = "info"  # critical, high, medium, low, info
    
    # Context information
    test_environment: Dict[str, Any] = field(default_factory=dict)
    data_quality_issues: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize the test result after initialization."""
        if not self.execution_id:
            self.execution_id = f"exec_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(self.test_case.test_id)) % 1000}"
        
        # Ensure passed status matches the status enum
        if self.status == TestStatus.PASSED:
            self.passed = True
        elif self.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]:
            self.passed = False
        
        # Set severity based on test priority and result
        if not self.passed and self.test_case.priority.value in ["critical", "high"]:
            self.severity = "high" if self.test_case.priority.value == "critical" else "medium"
    
    def is_successful(self) -> bool:
        """
        Check if the test was successful (passed with reasonable confidence).
        """
        return (
            self.passed and 
            self.status == TestStatus.PASSED and
            self.confidence != ConfidenceLevel.UNCERTAIN
        )
    
    def requires_attention(self) -> bool:
        """
        Check if this result requires immediate attention.
        """
        return (
            not self.passed or
            self.severity in ["critical", "high"] or
            self.confidence == ConfidenceLevel.UNCERTAIN or
            bool(self.data_quality_issues)
        )
    
    def get_summary(self) -> str:
        """
        Generate a concise summary of the test result.
        """
        status_symbols = {
            TestStatus.PASSED: "[PASS]",
            TestStatus.FAILED: "[FAIL]", 
            TestStatus.SKIPPED: "[SKIP]",
            TestStatus.ERROR: "[ERROR]",
            TestStatus.TIMEOUT: "[TIMEOUT]"
        }
        
        symbol = status_symbols.get(self.status, "[UNKNOWN]")
        
        summary = f"{symbol} {self.test_case.feature_name} - {self.test_case.test_type.value}"
        
        if self.confidence_score > 0:
            summary += f" (confidence: {self.confidence_score:.2f})"
        
        if self.execution_time > 0:
            summary += f" [{self.execution_time:.2f}s]"
        
        return summary
    
    def get_failure_details(self) -> Dict[str, Any]:
        """
        Get detailed failure information if the test failed.
        """
        if self.passed:
            return {}
        
        details = {
            'status': self.status.value,
            'error_message': self.error_message,
            'analysis': self.analysis,
            'recommendations': self.recommendations,
            'severity': self.severity
        }
        
        if self.stack_trace:
            details['stack_trace'] = self.stack_trace
        
        if self.data_quality_issues:
            details['data_quality_issues'] = self.data_quality_issues
        
        return details
    
    def validate_result_completeness(self) -> List[str]:
        """
        Validate that the test result has all required information.
        
        Returns:
            List of validation issues (empty if complete)
        """
        issues = []
        
        # Check basic execution information
        if not self.execution_id:
            issues.append("Missing execution ID")
        
        if self.execution_time < 0:
            issues.append("Invalid execution time")
        
        # Check result consistency
        if self.status == TestStatus.PASSED and not self.passed:
            issues.append("Status/passed flag mismatch")
        
        if self.passed and self.status in [TestStatus.FAILED, TestStatus.ERROR]:
            issues.append("Passed flag inconsistent with error status")
        
        # Check confidence scoring
        if self.confidence_score < 0 or self.confidence_score > 1:
            issues.append("Confidence score must be between 0 and 1")
        
        # Check analysis completeness for failed tests
        if not self.passed and not self.analysis:
            issues.append("Failed tests must include analysis")
        
        if not self.passed and not self.recommendations:
            issues.append("Failed tests should include recommendations")
        
        # Check error information consistency
        if self.status == TestStatus.ERROR and not self.error_message:
            issues.append("Error status requires error message")
        
        return issues
    
    def calculate_quality_score(self) -> float:
        """
        Calculate a quality score for this test result based on completeness and reliability.
        
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Penalize for validation issues
        issues = self.validate_result_completeness()
        score -= len(issues) * 0.1
        
        # Penalize for data quality issues
        score -= len(self.data_quality_issues) * 0.05
        
        # Penalize for low confidence
        if self.confidence == ConfidenceLevel.UNCERTAIN:
            score -= 0.3
        elif self.confidence == ConfidenceLevel.LOW:
            score -= 0.15
        
        # Penalize for execution errors
        if self.status in [TestStatus.ERROR, TestStatus.TIMEOUT]:
            score -= 0.2
        
        # Bonus for comprehensive analysis
        if self.analysis and len(self.analysis) > 50:
            score += 0.05
        
        if self.recommendations and len(self.recommendations) > 0:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """
        Assess the risk level of this test result.
        
        Returns:
            Dictionary with risk assessment details
        """
        risk_level = "low"
        risk_factors = []
        mitigation_actions = []
        
        # Assess based on test failure
        if not self.passed:
            if self.test_case.priority.value in ["critical", "high"]:
                risk_level = "high"
                risk_factors.append("Critical/high priority test failure")
                mitigation_actions.append("Immediate investigation and fix required")
            else:
                risk_level = "medium"
                risk_factors.append("Test failure in non-critical component")
                mitigation_actions.append("Schedule fix in next development cycle")
        
        # Assess based on confidence
        if self.confidence == ConfidenceLevel.UNCERTAIN:
            if risk_level == "low":
                risk_level = "medium"
            risk_factors.append("Uncertain test result confidence")
            mitigation_actions.append("Manual review and additional testing needed")
        
        # Assess based on data quality
        if self.data_quality_issues:
            if risk_level == "low":
                risk_level = "medium"
            risk_factors.append("Data quality issues detected")
            mitigation_actions.append("Address data quality before production deployment")
        
        # Assess based on performance metrics
        if self.performance_metrics:
            poor_metrics = [k for k, v in self.performance_metrics.items() 
                          if isinstance(v, (int, float)) and v < 0.5]
            if poor_metrics:
                if risk_level == "low":
                    risk_level = "medium"
                risk_factors.append(f"Poor performance in metrics: {', '.join(poor_metrics)}")
                mitigation_actions.append("Performance optimization required")
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'mitigation_actions': mitigation_actions,
            'requires_immediate_attention': risk_level == "high"
        }
    
    def generate_executive_summary(self) -> str:
        """
        Generate a concise executive summary of the test result.
        
        Returns:
            Executive summary string
        """
        status_emoji = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "", 
            TestStatus.SKIPPED: "⏭️",
            TestStatus.ERROR: "🚨",
            TestStatus.TIMEOUT: "⏰"
        }
        
        emoji = status_emoji.get(self.status, "❓")
        
        summary = f"{emoji} {self.test_case.feature_name} - {self.test_case.test_type.value.replace('_', ' ').title()}"
        
        if self.passed:
            summary += f" | Confidence: {self.confidence.value.title()}"
            if self.confidence_score > 0:
                summary += f" ({self.confidence_score:.1%})"
        else:
            risk = self.get_risk_assessment()
            summary += f" | Risk: {risk['risk_level'].title()}"
            if self.error_message:
                summary += f" | Error: {self.error_message[:50]}..."
        
        if self.execution_time > 0:
            summary += f" | Duration: {self.execution_time:.1f}s"
        
        return summary
    
    def compare_with_baseline(self, baseline_result: 'TestResult') -> Dict[str, Any]:
        """
        Compare this result with a baseline result to detect changes.
        
        Args:
            baseline_result: Previous test result to compare against
            
        Returns:
            Dictionary with comparison analysis
        """
        comparison = {
            'status_changed': self.status != baseline_result.status,
            'confidence_changed': abs(self.confidence_score - baseline_result.confidence_score) > 0.1,
            'performance_changes': {},
            'new_issues': [],
            'resolved_issues': [],
            'overall_trend': 'stable'
        }
        
        # Compare performance metrics
        for metric, value in self.performance_metrics.items():
            if metric in baseline_result.performance_metrics:
                baseline_value = baseline_result.performance_metrics[metric]
                change = value - baseline_value
                change_pct = (change / baseline_value * 100) if baseline_value != 0 else 0
                
                comparison['performance_changes'][metric] = {
                    'current': value,
                    'baseline': baseline_value,
                    'change': change,
                    'change_percent': change_pct
                }
        
        # Compare issues
        current_issues = set(self.data_quality_issues)
        baseline_issues = set(baseline_result.data_quality_issues)
        
        comparison['new_issues'] = list(current_issues - baseline_issues)
        comparison['resolved_issues'] = list(baseline_issues - current_issues)
        
        # Determine overall trend
        if self.passed and not baseline_result.passed:
            comparison['overall_trend'] = 'improved'
        elif not self.passed and baseline_result.passed:
            comparison['overall_trend'] = 'degraded'
        elif self.confidence_score > baseline_result.confidence_score + 0.1:
            comparison['overall_trend'] = 'improved'
        elif self.confidence_score < baseline_result.confidence_score - 0.1:
            comparison['overall_trend'] = 'degraded'
        
        return comparison
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert test result to dictionary for serialization.
        """
        return {
            'test_case': self.test_case.to_dict(),
            'execution_id': self.execution_id,
            'status': self.status.value,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'actual_result': self.actual_result,
            'expected_result': self.expected_result,
            'passed': self.passed,
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'analysis': self.analysis,
            'error_message': self.error_message,
            'performance_metrics': self.performance_metrics,
            'statistical_measures': self.statistical_measures,
            'recommendations': self.recommendations,
            'action_items': self.action_items,
            'severity': self.severity,
            'test_environment': self.test_environment,
            'data_quality_issues': self.data_quality_issues,
            'quality_score': self.calculate_quality_score(),
            'risk_assessment': self.get_risk_assessment()
        }