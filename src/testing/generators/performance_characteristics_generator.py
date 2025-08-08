"""
Performance characteristics test generator for validating feature performance metrics.

This module implements comprehensive performance testing including hit rates, Sharpe ratios,
empirical ranges, risk-adjusted returns, and drawdown control validation.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..models.feature_spec import FeatureSpec
from ..models.test_case import TestCase, TestType, TestPriority


@dataclass
class PerformanceThresholds:
    """Thresholds for performance validation."""
    hit_rate_min: float = 0.45  # Minimum acceptable hit rate
    hit_rate_max: float = 0.65  # Maximum expected hit rate
    sharpe_ratio_min: float = 0.5  # Minimum acceptable Sharpe ratio
    sharpe_ratio_max: float = 3.0  # Maximum expected Sharpe ratio
    max_drawdown_threshold: float = 0.15  # Maximum acceptable drawdown (15%)
    volatility_range_tolerance: float = 0.1  # 10% tolerance for volatility ranges
    performance_deviation_threshold: float = 0.2  # 20% deviation triggers alert


class PerformanceCharacteristicsGenerator:
    """
    Generator for performance characteristics validation tests.
    
    Creates comprehensive tests for validating feature performance metrics
    including hit rates, Sharpe ratios, empirical ranges, and risk controls.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance characteristics generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load performance thresholds from config or use defaults
        self.thresholds = PerformanceThresholds(
            hit_rate_min=self.config.get('hit_rate_min', 0.45),
            hit_rate_max=self.config.get('hit_rate_max', 0.65),
            sharpe_ratio_min=self.config.get('sharpe_ratio_min', 0.5),
            sharpe_ratio_max=self.config.get('sharpe_ratio_max', 3.0),
            max_drawdown_threshold=self.config.get('max_drawdown_threshold', 0.15),
            volatility_range_tolerance=self.config.get('volatility_range_tolerance', 0.1),
            performance_deviation_threshold=self.config.get('performance_deviation_threshold', 0.2)
        )
        
        # Feature-specific performance expectations
        self.feature_performance_specs = {
            'Q50': {
                'expected_hit_rate': 0.55,
                'expected_sharpe': 1.2,
                'regime_sensitivity': True,
                'directional_bias': True
            },
            'vol_risk': {
                'empirical_range': (0.0, 1.0),
                'normalization_check': True,
                'variance_formula': True
            },
            'kelly_sizing': {
                'risk_adjustment': True,
                'drawdown_control': True,
                'position_scaling': True
            },
            'fg_index': {
                'sentiment_correlation': True,
                'empirical_range': (0, 100),
                'regime_dependency': True
            },
            'btc_dom': {
                'market_correlation': True,
                'empirical_range': (0.3, 0.8),
                'trend_following': True
            }
        }
    
    def generate_performance_characteristics_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """
        Generate comprehensive performance characteristics tests for a feature.
        
        Args:
            feature: Feature specification to generate tests for
            
        Returns:
            List of performance test cases
        """
        tests = []
        
        # Generate hit rate and Sharpe ratio tests for signal features
        if self._is_signal_feature(feature):
            tests.extend(self._generate_hit_rate_tests(feature))
            tests.extend(self._generate_sharpe_ratio_tests(feature))
            tests.extend(self._generate_regime_performance_tests(feature))
        
        # Generate empirical range tests for volatility features
        if self._is_volatility_feature(feature):
            tests.extend(self._generate_empirical_range_tests(feature))
            tests.extend(self._generate_normalization_tests(feature))
        
        # Generate risk-adjusted return and drawdown tests for position sizing features
        if self._is_position_sizing_feature(feature):
            tests.extend(self._generate_risk_adjusted_return_tests(feature))
            tests.extend(self._generate_drawdown_control_tests(feature))
        
        # Generate performance deviation detection tests for all features
        tests.extend(self._generate_performance_deviation_tests(feature))
        
        # Generate general performance validation tests
        tests.extend(self._generate_general_performance_tests(feature))
        
        self.logger.info(f"Generated {len(tests)} performance characteristics tests for {feature.name}")
        return tests
    
    def _generate_hit_rate_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate hit rate validation tests."""
        tests = []
        
        # Basic hit rate validation
        hit_rate_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate hit rate for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'min_hit_rate': self.thresholds.hit_rate_min,
                'max_hit_rate': self.thresholds.hit_rate_max
            },
            validation_criteria={
                'metric_type': 'hit_rate',
                'min_threshold': self.thresholds.hit_rate_min,
                'max_threshold': self.thresholds.hit_rate_max,
                'calculation_method': 'directional_accuracy',
                'lookback_period': 252  # 1 year of daily data
            },
            test_parameters={
                'test_type': 'hit_rate_validation',
                'signal_threshold': 0.0,
                'forward_returns_period': 1
            },
            rationale=f"Validate that {feature.name} maintains acceptable hit rate for directional predictions",
            failure_impact="Poor hit rate indicates feature may not be predictive",
            estimated_duration=4.0
        )
        tests.append(hit_rate_test)
        
        # Regime-specific hit rate tests
        if feature.regime_dependencies:
            for regime in feature.regime_dependencies.keys():
                regime_hit_rate_test = TestCase(
                    test_id="auto",
                    feature_name=feature.name,
                    test_type=TestType.PERFORMANCE,
                    description=f"Validate hit rate for {feature.name} in {regime} regime",
                    priority=TestPriority.HIGH,
                    regime_context=regime,
                    expected_result={
                        'min_hit_rate': self.thresholds.hit_rate_min,
                        'regime_specific': True
                    },
                    validation_criteria={
                        'metric_type': 'regime_hit_rate',
                        'regime': regime,
                        'min_threshold': self.thresholds.hit_rate_min,
                        'regime_adjustment': True
                    },
                    test_parameters={
                        'test_type': 'regime_hit_rate_validation',
                        'regime_filter': regime
                    },
                    rationale=f"Ensure {feature.name} maintains hit rate in {regime} market conditions",
                    estimated_duration=5.0
                )
                tests.append(regime_hit_rate_test)
        
        return tests
    
    def _generate_sharpe_ratio_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate Sharpe ratio validation tests."""
        tests = []
        
        # Basic Sharpe ratio validation
        sharpe_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate Sharpe ratio for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'min_sharpe': self.thresholds.sharpe_ratio_min,
                'max_sharpe': self.thresholds.sharpe_ratio_max
            },
            validation_criteria={
                'metric_type': 'sharpe_ratio',
                'min_threshold': self.thresholds.sharpe_ratio_min,
                'max_threshold': self.thresholds.sharpe_ratio_max,
                'risk_free_rate': 0.02,  # 2% annual risk-free rate
                'annualization_factor': 252
            },
            test_parameters={
                'test_type': 'sharpe_ratio_validation',
                'return_calculation': 'log_returns',
                'volatility_adjustment': True
            },
            rationale=f"Validate that {feature.name} generates acceptable risk-adjusted returns",
            failure_impact="Poor Sharpe ratio indicates feature may not provide adequate risk-adjusted returns",
            estimated_duration=4.0
        )
        tests.append(sharpe_test)
        
        # Rolling Sharpe ratio stability test
        rolling_sharpe_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate rolling Sharpe ratio stability for {feature.name}",
            priority=TestPriority.MEDIUM,
            expected_result={
                'stability_threshold': 0.3,  # Max standard deviation of rolling Sharpe
                'consistency_check': True
            },
            validation_criteria={
                'metric_type': 'rolling_sharpe_stability',
                'rolling_window': 63,  # 3 months
                'stability_threshold': 0.3,
                'min_periods': 20
            },
            test_parameters={
                'test_type': 'rolling_sharpe_validation',
                'window_size': 63,
                'step_size': 21  # Monthly steps
            },
            rationale=f"Ensure {feature.name} Sharpe ratio is stable over time",
            failure_impact="Unstable Sharpe ratio indicates inconsistent performance",
            estimated_duration=6.0
        )
        tests.append(rolling_sharpe_test)
        
        return tests
    
    def _generate_empirical_range_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate empirical range verification tests for volatility features."""
        tests = []
        
        # Get expected ranges for the feature
        expected_ranges = self._get_expected_ranges(feature)
        
        for range_name, range_bounds in expected_ranges.items():
            range_test = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.PERFORMANCE,
                description=f"Validate {range_name} empirical range for {feature.name}",
                priority=TestPriority.HIGH,
                expected_result={
                    'min_value': range_bounds[0],
                    'max_value': range_bounds[1],
                    'outlier_threshold': 0.05  # 5% outliers allowed
                },
                validation_criteria={
                    'metric_type': 'empirical_range',
                    'range_type': range_name,
                    'expected_min': range_bounds[0],
                    'expected_max': range_bounds[1],
                    'tolerance': self.thresholds.volatility_range_tolerance,
                    'outlier_threshold': 0.05
                },
                test_parameters={
                    'test_type': 'empirical_range_validation',
                    'range_calculation': 'percentile_based',
                    'percentile_bounds': (1, 99)  # 1st to 99th percentile
                },
                rationale=f"Ensure {feature.name} values stay within expected {range_name} range",
                failure_impact=f"Values outside expected range may indicate data quality issues or feature drift",
                estimated_duration=3.0
            )
            tests.append(range_test)
        
        # Normalization validation for normalized features
        if self._is_normalized_feature(feature):
            normalization_test = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.PERFORMANCE,
                description=f"Validate normalization behavior for {feature.name}",
                priority=TestPriority.HIGH,
                expected_result={
                    'mean_near_zero': True,
                    'std_near_one': True,
                    'distribution_check': True
                },
                validation_criteria={
                    'metric_type': 'normalization_validation',
                    'mean_tolerance': 0.1,
                    'std_tolerance': 0.2,
                    'distribution_test': 'kolmogorov_smirnov'
                },
                test_parameters={
                    'test_type': 'normalization_validation',
                    'expected_mean': 0.0,
                    'expected_std': 1.0
                },
                rationale=f"Ensure {feature.name} normalization is working correctly",
                failure_impact="Incorrect normalization can lead to biased model predictions",
                estimated_duration=3.0
            )
            tests.append(normalization_test)
        
        return tests
    
    def _generate_normalization_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate normalization validation tests for features that should be normalized."""
        tests = []
        
        if not self._is_normalized_feature(feature):
            return tests
        
        normalization_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate normalization behavior for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'mean_near_zero': True,
                'std_near_one': True,
                'distribution_check': True
            },
            validation_criteria={
                'metric_type': 'normalization_validation',
                'mean_tolerance': 0.1,
                'std_tolerance': 0.2,
                'distribution_test': 'kolmogorov_smirnov'
            },
            test_parameters={
                'test_type': 'normalization_validation',
                'expected_mean': 0.0,
                'expected_std': 1.0
            },
            rationale=f"Ensure {feature.name} normalization is working correctly",
            failure_impact="Incorrect normalization can lead to biased model predictions",
            estimated_duration=3.0
        )
        tests.append(normalization_test)
        
        return tests
    
    def _generate_risk_adjusted_return_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate risk-adjusted return validation tests."""
        tests = []
        
        # Risk-adjusted return validation
        risk_adjusted_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate risk-adjusted returns for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'positive_risk_adjusted_return': True,
                'volatility_scaling': True,
                'risk_parity': True
            },
            validation_criteria={
                'metric_type': 'risk_adjusted_return',
                'min_return_threshold': 0.0,
                'volatility_adjustment': True,
                'risk_scaling_factor': 0.15  # Target 15% volatility
            },
            test_parameters={
                'test_type': 'risk_adjusted_return_validation',
                'position_sizing_method': 'volatility_targeting',
                'target_volatility': 0.15
            },
            rationale=f"Ensure {feature.name} generates positive risk-adjusted returns",
            failure_impact="Negative risk-adjusted returns indicate poor position sizing",
            estimated_duration=5.0
        )
        tests.append(risk_adjusted_test)
        
        # Kelly criterion validation for Kelly sizing features
        if 'kelly' in feature.name.lower():
            kelly_test = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.PERFORMANCE,
                description=f"Validate Kelly criterion implementation for {feature.name}",
                priority=TestPriority.CRITICAL,
                expected_result={
                    'kelly_formula_correct': True,
                    'position_bounds': True,
                    'risk_control': True
                },
                validation_criteria={
                    'metric_type': 'kelly_criterion_validation',
                    'formula_check': 'f = (bp - q) / b',
                    'max_position_size': 0.25,  # 25% max position
                    'min_position_size': -0.25
                },
                test_parameters={
                    'test_type': 'kelly_criterion_validation',
                    'win_rate_estimation': 'historical',
                    'payoff_ratio_estimation': 'historical'
                },
                rationale=f"Ensure {feature.name} correctly implements Kelly criterion",
                failure_impact="Incorrect Kelly implementation can lead to excessive risk-taking",
                estimated_duration=6.0
            )
            tests.append(kelly_test)
        
        return tests
    
    def _generate_drawdown_control_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate drawdown control validation tests."""
        tests = []
        
        # Maximum drawdown validation
        max_drawdown_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate maximum drawdown control for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'max_drawdown_threshold': self.thresholds.max_drawdown_threshold,
                'drawdown_duration': 'reasonable'
            },
            validation_criteria={
                'metric_type': 'max_drawdown_validation',
                'max_drawdown_threshold': self.thresholds.max_drawdown_threshold,
                'max_duration_days': 90,  # Max 3 months in drawdown
                'recovery_time_check': True
            },
            test_parameters={
                'test_type': 'drawdown_validation',
                'drawdown_calculation': 'peak_to_trough',
                'recovery_definition': 'new_high'
            },
            rationale=f"Ensure {feature.name} keeps drawdowns within acceptable limits",
            failure_impact="Excessive drawdowns can lead to significant capital loss",
            estimated_duration=4.0
        )
        tests.append(max_drawdown_test)
        
        # Drawdown frequency validation
        drawdown_frequency_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate drawdown frequency for {feature.name}",
            priority=TestPriority.MEDIUM,
            expected_result={
                'max_drawdown_frequency': 0.3,  # Max 30% of time in drawdown
                'average_drawdown_size': 0.05   # Average 5% drawdown
            },
            validation_criteria={
                'metric_type': 'drawdown_frequency_validation',
                'max_time_in_drawdown': 0.3,
                'average_drawdown_threshold': 0.05,
                'drawdown_clustering': False
            },
            test_parameters={
                'test_type': 'drawdown_frequency_validation',
                'drawdown_threshold': 0.01  # 1% minimum drawdown
            },
            rationale=f"Ensure {feature.name} doesn't experience excessive drawdown frequency",
            failure_impact="Frequent drawdowns indicate unstable performance",
            estimated_duration=4.0
        )
        tests.append(drawdown_frequency_test)
        
        return tests
    
    def _generate_performance_deviation_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate performance deviation detection and alerting tests."""
        tests = []
        
        # Performance drift detection
        drift_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Detect performance drift for {feature.name}",
            priority=TestPriority.HIGH,
            expected_result={
                'performance_stability': True,
                'drift_detection': True,
                'alert_generation': True
            },
            validation_criteria={
                'metric_type': 'performance_drift_detection',
                'drift_threshold': self.thresholds.performance_deviation_threshold,
                'lookback_window': 252,  # 1 year
                'comparison_window': 63,  # 3 months
                'statistical_test': 'mann_whitney_u'
            },
            test_parameters={
                'test_type': 'performance_drift_detection',
                'metrics_to_monitor': ['sharpe_ratio', 'hit_rate', 'max_drawdown'],
                'alert_threshold': self.thresholds.performance_deviation_threshold
            },
            rationale=f"Detect significant performance changes in {feature.name}",
            failure_impact="Undetected performance drift can lead to continued use of degraded features",
            estimated_duration=5.0
        )
        tests.append(drift_test)
        
        # Performance anomaly detection
        anomaly_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Detect performance anomalies for {feature.name}",
            priority=TestPriority.MEDIUM,
            expected_result={
                'anomaly_detection': True,
                'outlier_identification': True,
                'root_cause_analysis': True
            },
            validation_criteria={
                'metric_type': 'performance_anomaly_detection',
                'anomaly_threshold': 3.0,  # 3 standard deviations
                'detection_method': 'isolation_forest',
                'min_anomaly_duration': 5  # 5 days minimum
            },
            test_parameters={
                'test_type': 'performance_anomaly_detection',
                'features_to_monitor': ['returns', 'volatility', 'correlation'],
                'contamination_rate': 0.05  # 5% expected anomalies
            },
            rationale=f"Identify unusual performance patterns in {feature.name}",
            failure_impact="Undetected anomalies may indicate underlying issues",
            estimated_duration=6.0
        )
        tests.append(anomaly_test)
        
        return tests
    
    def _generate_regime_performance_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate regime-specific performance validation tests."""
        tests = []
        
        if not feature.regime_dependencies:
            return tests
        
        for regime, expected_behavior in feature.regime_dependencies.items():
            regime_test = TestCase(
                test_id="auto",
                feature_name=feature.name,
                test_type=TestType.PERFORMANCE,
                description=f"Validate performance in {regime} regime for {feature.name}",
                priority=TestPriority.HIGH,
                regime_context=regime,
                expected_result={
                    'regime_specific_performance': expected_behavior,
                    'performance_consistency': True
                },
                validation_criteria={
                    'metric_type': 'regime_performance_validation',
                    'regime': regime,
                    'expected_behavior': expected_behavior,
                    'performance_threshold': 'regime_adjusted',
                    'consistency_check': True
                },
                test_parameters={
                    'test_type': 'regime_performance_validation',
                    'regime_identification': 'historical_classification',
                    'performance_metrics': ['sharpe_ratio', 'hit_rate', 'max_drawdown']
                },
                rationale=f"Ensure {feature.name} performs as expected in {regime} market conditions",
                failure_impact=f"Poor performance in {regime} regime reduces overall system effectiveness",
                estimated_duration=7.0
            )
            tests.append(regime_test)
        
        return tests
    
    def _generate_general_performance_tests(self, feature: FeatureSpec) -> List[TestCase]:
        """Generate general performance validation tests."""
        tests = []
        
        # Performance consistency test
        consistency_test = TestCase(
            test_id="auto",
            feature_name=feature.name,
            test_type=TestType.PERFORMANCE,
            description=f"Validate performance consistency for {feature.name}",
            priority=TestPriority.MEDIUM,
            expected_result={
                'performance_consistency': True,
                'temporal_stability': True
            },
            validation_criteria={
                'metric_type': 'performance_consistency',
                'consistency_threshold': 0.8,  # 80% consistency
                'temporal_window': 252,  # 1 year
                'stability_measure': 'coefficient_of_variation'
            },
            test_parameters={
                'test_type': 'performance_consistency_validation',
                'rolling_window': 63,  # 3 months
                'overlap': 21  # 1 month overlap
            },
            rationale=f"Ensure {feature.name} provides consistent performance over time",
            failure_impact="Inconsistent performance makes the feature unreliable",
            estimated_duration=4.0
        )
        tests.append(consistency_test)
        
        return tests
    
    def _is_signal_feature(self, feature: FeatureSpec) -> bool:
        """Check if feature is a signal feature that should have hit rate and Sharpe tests."""
        signal_indicators = ['q50', 'q10', 'q90', 'spread', 'signal']
        return any(indicator in feature.name.lower() for indicator in signal_indicators)
    
    def _is_volatility_feature(self, feature: FeatureSpec) -> bool:
        """Check if feature is a volatility feature that should have empirical range tests."""
        volatility_indicators = ['vol_', 'volatility', 'fg_index', 'btc_dom']
        return any(indicator in feature.name.lower() for indicator in volatility_indicators)
    
    def _is_position_sizing_feature(self, feature: FeatureSpec) -> bool:
        """Check if feature is a position sizing feature that should have risk-adjusted return tests."""
        sizing_indicators = ['kelly', 'sizing', 'position']
        # Check for exact matches or specific patterns to avoid false positives
        name_lower = feature.name.lower()
        
        # Direct matches
        if any(indicator in name_lower for indicator in sizing_indicators):
            return True
        
        # Check category for position sizing
        if 'position sizing' in feature.category.lower():
            return True
            
        return False
    
    def _is_normalized_feature(self, feature: FeatureSpec) -> bool:
        """Check if feature should be normalized."""
        normalized_indicators = ['_norm', 'normalized', 'scaled']
        return any(indicator in feature.name.lower() for indicator in normalized_indicators)
    
    def _get_expected_ranges(self, feature: FeatureSpec) -> Dict[str, tuple]:
        """Get expected ranges for a feature based on its specification and type."""
        if feature.empirical_ranges:
            return {k: (v if isinstance(v, tuple) else (0, v)) for k, v in feature.empirical_ranges.items()}
        
        # Default ranges based on feature type
        feature_name = feature.name.lower()
        
        if 'vol_risk' in feature_name:
            return {'variance_range': (0.0, 1.0)}
        elif 'fg_index' in feature_name:
            return {'sentiment_range': (0, 100)}
        elif 'btc_dom' in feature_name:
            return {'dominance_range': (0.3, 0.8)}
        elif any(q in feature_name for q in ['q50', 'q10', 'q90']):
            return {'quantile_range': (-3.0, 3.0)}
        else:
            return {'default_range': (-5.0, 5.0)}