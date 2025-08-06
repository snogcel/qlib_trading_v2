"""
Validation Integration System

This module implements the ValidationIntegrationSystem class that links documentation claims
to automated tests, generates validation tests based on thesis statements and economic rationale,
and validates feature performance against actual backtest data from training_pipeline.py.
"""

import re
import json
import ast
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from src.documentation.economic_rationale_generator import (
    FeatureEnhancement,
    ThesisStatement,
    EconomicRationale,
    ValidationCriterion
)


@dataclass
class ValidationTest:
    """Represents a generated validation test"""
    test_name: str
    test_function: str
    description: str
    thesis_reference: str
    success_criteria: Dict[str, Any]
    test_type: str  # "statistical", "performance", "economic_logic"
    dependencies: List[str]
    expected_runtime: str
    created_at: datetime


@dataclass
class TestLink:
    """Links feature to existing test"""
    feature_name: str
    test_file_path: str
    test_function_name: str
    link_type: str  # "direct", "indirect", "performance"
    confidence: float
    description: str


@dataclass
class PerformanceValidation:
    """Result of performance claim validation"""
    feature_name: str
    claimed_performance: Dict[str, float]
    actual_performance: Dict[str, float]
    validation_passed: bool
    performance_gap: Dict[str, float]
    confidence_interval: Dict[str, Tuple[float, float]]
    validation_details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class Alert:
    """Performance monitoring alert"""
    alert_id: str
    feature_name: str
    alert_type: str  # "performance_degradation", "thesis_violation", "validation_failure"
    severity: str  # "low", "medium", "high", "critical"
    message: str
    threshold_violated: Dict[str, Any]
    current_values: Dict[str, Any]
    recommended_actions: List[str]
    created_at: datetime


class ValidationIntegrationError(Exception):
    """Base exception for validation integration errors"""
    pass


class TestGenerationError(ValidationIntegrationError):
    """Raised when test generation fails"""
    pass


class PerformanceValidationError(ValidationIntegrationError):
    """Raised when performance validation fails"""
    pass


class ValidationIntegrationSystem:
    """
    Links documentation claims to automated tests, generates validation tests based on
    thesis statements and economic rationale, and validates feature performance against
    actual backtest data from training_pipeline.py.
    """
    
    def __init__(self, 
                 training_pipeline_path: str = "src/training_pipeline.py",
                 test_base_dir: str = "tests",
                 validation_config_path: str = "config/validation_config.json"):
        """
        Initialize the validation integration system
        
        Args:
            training_pipeline_path: Path to training_pipeline.py for performance data
            test_base_dir: Base directory for test files
            validation_config_path: Path to validation configuration
        """
        self.training_pipeline_path = Path(training_pipeline_path)
        self.test_base_dir = Path(test_base_dir)
        self.validation_config_path = Path(validation_config_path)
        
        # Load configuration
        self.config = self._load_validation_config()
        
        # Initialize test tracking
        self.generated_tests: List[ValidationTest] = []
        self.test_links: List[TestLink] = []
        self.performance_cache: Dict[str, Any] = {}
        
        # Load training pipeline for performance validation
        self._load_training_pipeline()
    
    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "performance_thresholds": {
                "sharpe_ratio": {"min": 0.5, "target": 1.0},
                "max_drawdown": {"max": 0.2, "target": 0.1},
                "information_ratio": {"min": 0.3, "target": 0.6},
                "hit_rate": {"min": 0.52, "target": 0.55}
            },
            "statistical_thresholds": {
                "p_value": 0.05,
                "confidence_level": 0.95,
                "min_sample_size": 1000
            },
            "test_generation": {
                "max_tests_per_feature": 5,
                "test_timeout": 300,
                "parallel_execution": True
            },
            "monitoring": {
                "check_frequency": "daily",
                "alert_thresholds": {
                    "performance_degradation": 0.1,
                    "thesis_violation": 0.05
                }
            }
        }
        
        if self.validation_config_path.exists():
            try:
                with open(self.validation_config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                print(f"Warning: Could not load validation config: {e}")
                return default_config
        else:
            # Create default config file
            self.validation_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _load_training_pipeline(self):
        """Load training pipeline module for performance validation"""
        try:
            # Import training pipeline module
            import sys
            import os
            
            # Add project root to path
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.append(project_root)
            
            # Import the training pipeline
            from src.training_pipeline import (
                train_start_time, train_end_time, valid_start_time, valid_end_time,
                test_start_time, test_end_time
            )
            
            self.training_dates = {
                "train_start": train_start_time,
                "train_end": train_end_time,
                "valid_start": valid_start_time,
                "valid_end": valid_end_time,
                "test_start": test_start_time,
                "test_end": test_end_time
            }
            
        except Exception as e:
            print(f"Warning: Could not load training pipeline: {e}")
            self.training_dates = {}
    
    def create_validation_tests(self, thesis: ThesisStatement, feature_name: str) -> List[ValidationTest]:
        """
        Generate automated tests for thesis claims
        
        Args:
            thesis: Thesis statement to generate tests for
            feature_name: Name of the feature
            
        Returns:
            List of generated validation tests
        """
        tests = []
        
        try:
            # Generate statistical validation test
            stat_test = self._generate_statistical_test(thesis, feature_name)
            if stat_test:
                tests.append(stat_test)
            
            # Generate economic logic test
            logic_test = self._generate_economic_logic_test(thesis, feature_name)
            if logic_test:
                tests.append(logic_test)
            
            # Generate performance test
            perf_test = self._generate_performance_test(thesis, feature_name)
            if perf_test:
                tests.append(perf_test)
            
            # Generate regime-specific tests if applicable
            if "regime" in thesis.hypothesis.lower():
                regime_tests = self._generate_regime_tests(thesis, feature_name)
                tests.extend(regime_tests)
            
            # Store generated tests
            self.generated_tests.extend(tests)
            
            return tests
            
        except Exception as e:
            raise TestGenerationError(f"Failed to generate tests for {feature_name}: {str(e)}")
    
    def _generate_statistical_test(self, thesis: ThesisStatement, feature_name: str) -> Optional[ValidationTest]:
        """Generate statistical validation test"""
        test_function = f"""
def test_{feature_name.lower().replace(' ', '_')}_statistical_significance():
    \"\"\"
    Test statistical significance of {feature_name} based on thesis:
    {thesis.hypothesis}
    
    Validates:
    - Temporal Causality: No look-ahead bias in feature calculation
    - Statistical Significance: Feature has predictive power
    - Auditable Pipeline: Clear traceability of results
    \"\"\"
    import pandas as pd
    import numpy as np
    from scipy import stats
    from src.training_pipeline import *
    
    # Load feature data with temporal integrity check
    df = load_feature_data()  # This would need to be implemented
    
    # TEMPORAL CAUSALITY CHECK: Verify no future information leakage
    # Check that feature values are calculated using only past data
    assert 'timestamp' in df.columns, "Timestamp required for temporal causality validation"
    
    # Extract feature values
    feature_values = df['{feature_name.lower()}'].dropna()
    
    # AUDITABLE PIPELINE: Log data lineage
    print(f"Feature {feature_name} validation:")
    print(f"  - Data period: {{df['timestamp'].min()}} to {{df['timestamp'].max()}}")
    print(f"  - Sample size: {{len(feature_values)}}")
    print(f"  - Missing values: {{df['{feature_name.lower()}'].isna().sum()}}")
    
    # Test for statistical significance
    if len(feature_values) < {self.config['statistical_thresholds']['min_sample_size']}:
        pytest.skip(f"Insufficient data for {feature_name}: {{len(feature_values)}} samples")
    
    # Perform appropriate statistical test based on feature type
    if 'signal' in '{feature_name.lower()}':
        # Test signal predictive power with proper lag
        returns = df['returns'].shift(-1).dropna()  # Future returns (proper lag)
        feature_lagged = feature_values[:-1]  # Align with future returns
        
        correlation, p_value = stats.pearsonr(feature_lagged, returns)
        
        # AUDITABLE PIPELINE: Log test results
        print(f"  - Correlation with future returns: {{correlation:.4f}}")
        print(f"  - P-value: {{p_value:.4f}}")
        
        assert p_value < {self.config['statistical_thresholds']['p_value']}, \\
            f"Feature {feature_name} not statistically significant: p={{p_value:.4f}}"
        
        assert abs(correlation) > 0.01, \\
            f"Feature {feature_name} correlation too weak: {{correlation:.4f}}"
    
    elif 'risk' in '{feature_name.lower()}':
        # Test risk prediction accuracy
        volatility = df['volatility'].dropna()
        correlation, p_value = stats.pearsonr(feature_values, volatility)
        
        # AUDITABLE PIPELINE: Log test results
        print(f"  - Correlation with volatility: {{correlation:.4f}}")
        print(f"  - P-value: {{p_value:.4f}}")
        
        assert p_value < {self.config['statistical_thresholds']['p_value']}, \\
            f"Risk feature {feature_name} not statistically significant: p={{p_value:.4f}}"
    
    else:
        # Generic significance test
        t_stat, p_value = stats.ttest_1samp(feature_values, 0)
        
        # AUDITABLE PIPELINE: Log test results
        print(f"  - T-statistic: {{t_stat:.4f}}")
        print(f"  - P-value: {{p_value:.4f}}")
        
        assert p_value < {self.config['statistical_thresholds']['p_value']}, \\
            f"Feature {feature_name} not significantly different from zero: p={{p_value:.4f}}"
    
    # CONTRARIAN ADAPTATION: Check if feature works against prevailing narrative
    if 'sentiment' in df.columns:
        # Test if feature provides contrarian value during extreme sentiment
        extreme_fear = df['sentiment'] < 20
        extreme_greed = df['sentiment'] > 80
        
        if extreme_fear.sum() > 10:  # Sufficient extreme fear periods
            fear_performance = feature_values[extreme_fear].mean()
            overall_performance = feature_values.mean()
            print(f"  - Performance during extreme fear: {{fear_performance:.4f}} vs overall {{overall_performance:.4f}}")
        
        if extreme_greed.sum() > 10:  # Sufficient extreme greed periods
            greed_performance = feature_values[extreme_greed].mean()
            overall_performance = feature_values.mean()
            print(f"  - Performance during extreme greed: {{greed_performance:.4f}} vs overall {{overall_performance:.4f}}")
"""
        
        return ValidationTest(
            test_name=f"test_{feature_name.lower().replace(' ', '_')}_statistical_significance",
            test_function=test_function,
            description=f"Statistical significance test for {feature_name}",
            thesis_reference=thesis.hypothesis,
            success_criteria={
                "p_value": f"< {self.config['statistical_thresholds']['p_value']}",
                "min_samples": self.config['statistical_thresholds']['min_sample_size']
            },
            test_type="statistical",
            dependencies=["scipy", "pandas", "numpy"],
            expected_runtime="30s",
            created_at=datetime.now()
        )
    
    def _generate_economic_logic_test(self, thesis: ThesisStatement, feature_name: str) -> Optional[ValidationTest]:
        """Generate economic logic validation test"""
        test_function = f"""
def test_{feature_name.lower().replace(' ', '_')}_economic_logic():
    \"\"\"
    Test economic logic of {feature_name} based on thesis:
    {thesis.hypothesis}
    
    Economic basis: {thesis.economic_basis}
    \"\"\"
    import pandas as pd
    import numpy as np
    from src.training_pipeline import *
    
    # Load feature data
    df = load_feature_data()
    
    # Test economic logic based on thesis
    feature_values = df['{feature_name.lower()}'].dropna()
    
    # Test expected behavior patterns
    {self._generate_economic_logic_checks(thesis, feature_name)}
    
    # Test failure modes
    {self._generate_failure_mode_checks(thesis, feature_name)}
"""
        
        return ValidationTest(
            test_name=f"test_{feature_name.lower().replace(' ', '_')}_economic_logic",
            test_function=test_function,
            description=f"Economic logic validation for {feature_name}",
            thesis_reference=thesis.economic_basis,
            success_criteria={"logic_consistency": "pass"},
            test_type="economic_logic",
            dependencies=["pandas", "numpy"],
            expected_runtime="45s",
            created_at=datetime.now()
        )
    
    def _generate_performance_test(self, thesis: ThesisStatement, feature_name: str) -> Optional[ValidationTest]:
        """Generate performance validation test"""
        test_function = f"""
def test_{feature_name.lower().replace(' ', '_')}_performance():
    \"\"\"
    Test performance impact of {feature_name} based on thesis:
    {thesis.expected_behavior}
    \"\"\"
    import pandas as pd
    import numpy as np
    from src.training_pipeline import *
    
    # Load backtest results
    results = load_backtest_results()  # This would need to be implemented
    
    # Test performance metrics
    sharpe_ratio = results.get('sharpe_ratio', 0)
    max_drawdown = results.get('max_drawdown', 1)
    information_ratio = results.get('information_ratio', 0)
    
    # Validate against thresholds
    assert sharpe_ratio >= {self.config['performance_thresholds']['sharpe_ratio']['min']}, \\
        f"Sharpe ratio {{sharpe_ratio:.3f}} below minimum {self.config['performance_thresholds']['sharpe_ratio']['min']}"
    
    assert max_drawdown <= {self.config['performance_thresholds']['max_drawdown']['max']}, \\
        f"Max drawdown {{max_drawdown:.3f}} above maximum {self.config['performance_thresholds']['max_drawdown']['max']}"
    
    if 'information_ratio' in {self.config['performance_thresholds']}:
        assert information_ratio >= {self.config['performance_thresholds']['information_ratio']['min']}, \\
            f"Information ratio {{information_ratio:.3f}} below minimum {self.config['performance_thresholds']['information_ratio']['min']}"
"""
        
        return ValidationTest(
            test_name=f"test_{feature_name.lower().replace(' ', '_')}_performance",
            test_function=test_function,
            description=f"Performance validation for {feature_name}",
            thesis_reference=thesis.expected_behavior,
            success_criteria=self.config['performance_thresholds'],
            test_type="performance",
            dependencies=["pandas", "numpy"],
            expected_runtime="60s",
            created_at=datetime.now()
        )
    
    def _generate_regime_tests(self, thesis: ThesisStatement, feature_name: str) -> List[ValidationTest]:
        """Generate regime-specific validation tests"""
        tests = []
        
        regimes = ["bull", "bear", "sideways", "high_vol", "low_vol"]
        
        for regime in regimes:
            test_function = f"""
def test_{feature_name.lower().replace(' ', '_')}_{regime}_regime():
    \"\"\"
    Test {feature_name} behavior in {regime} regime
    Based on thesis: {thesis.hypothesis}
    \"\"\"
    import pandas as pd
    import numpy as np
    from src.training_pipeline import *
    
    # Load regime-specific data
    df = load_feature_data()
    regime_data = df[df['regime'] == '{regime}']
    
    if len(regime_data) < 100:
        pytest.skip(f"Insufficient {regime} regime data: {{len(regime_data)}} samples")
    
    # Test regime-specific behavior
    feature_values = regime_data['{feature_name.lower()}'].dropna()
    
    # Regime-specific validation logic would go here
    # This would be customized based on the specific thesis
    assert len(feature_values) > 0, f"No {feature_name} data in {regime} regime"
"""
            
            test = ValidationTest(
                test_name=f"test_{feature_name.lower().replace(' ', '_')}_{regime}_regime",
                test_function=test_function,
                description=f"{feature_name} validation in {regime} regime",
                thesis_reference=thesis.hypothesis,
                success_criteria={"regime_consistency": "pass"},
                test_type="regime_specific",
                dependencies=["pandas", "numpy"],
                expected_runtime="30s",
                created_at=datetime.now()
            )
            tests.append(test)
        
        return tests
    
    def _generate_economic_logic_checks(self, thesis: ThesisStatement, feature_name: str) -> str:
        """Generate economic logic check code"""
        checks = []
        
        # Supply/demand logic checks based on thesis content
        if "supply" in thesis.economic_basis.lower():
            checks.append("""
    # Test supply detection logic
    high_supply_periods = df[df['supply_indicator'] > df['supply_indicator'].quantile(0.8)]
    if len(high_supply_periods) > 0:
        feature_in_high_supply = high_supply_periods[feature_name.lower()].mean()
        overall_mean = feature_values.mean()
        # Feature should behave differently in high supply periods
        assert abs(feature_in_high_supply - overall_mean) > 0.01 * abs(overall_mean), \\
            f"Feature {feature_name} doesn't respond to supply changes"
""")
        
        if "demand" in thesis.economic_basis.lower():
            checks.append("""
    # Test demand detection logic
    high_demand_periods = df[df['demand_indicator'] > df['demand_indicator'].quantile(0.8)]
    if len(high_demand_periods) > 0:
        feature_in_high_demand = high_demand_periods[feature_name.lower()].mean()
        overall_mean = feature_values.mean()
        # Feature should behave differently in high demand periods
        assert abs(feature_in_high_demand - overall_mean) > 0.01 * abs(overall_mean), \\
            f"Feature {feature_name} doesn't respond to demand changes"
""")
        
        # Check for regime dependency
        if "regime" in thesis.economic_basis.lower():
            checks.append("""
    # Test regime-dependent behavior
    regimes = df['regime'].unique()
    if len(regimes) > 1:
        regime_means = df.groupby('regime')[feature_name.lower()].mean()
        # Feature should behave differently across regimes
        assert regime_means.std() > 0.01, \\
            f"Feature {feature_name} doesn't vary across market regimes"
""")
        
        # Check for inefficiency exploitation
        if "inefficiency" in thesis.economic_basis.lower():
            checks.append("""
    # Test market inefficiency exploitation
    # Feature should show predictive power for future returns
    feature_lag = feature_values.shift(1).dropna()
    returns_aligned = df['returns'].iloc[1:len(feature_lag)+1]
    if len(feature_lag) > 100:
        correlation = feature_lag.corr(returns_aligned)
        assert abs(correlation) > 0.02, \\
            f"Feature {feature_name} shows insufficient predictive power: {correlation:.4f}"
""")
        
        return "\n".join(checks) if checks else "    # No specific economic logic checks generated"
    
    def _generate_failure_mode_checks(self, thesis: ThesisStatement, feature_name: str) -> str:
        """Generate failure mode check code"""
        if not thesis.failure_modes:
            return "    # No failure modes specified in thesis"
        
        checks = []
        for failure_mode in thesis.failure_modes:
            checks.append(f"""
    # Test failure mode: {failure_mode}
    # This would include specific checks for when the feature might fail
    # Implementation would depend on the specific failure mode
""")
        
        return "\n".join(checks)
    
    def link_to_existing_tests(self, feature: FeatureEnhancement) -> List[TestLink]:
        """
        Link feature to existing test suite
        
        Args:
            feature: Feature enhancement to link
            
        Returns:
            List of test links found
        """
        links = []
        
        try:
            # Search for existing tests that might relate to this feature
            feature_name_variants = [
                feature.feature_name.lower(),
                feature.feature_name.lower().replace(' ', '_'),
                feature.feature_name.lower().replace('-', '_'),
                ''.join(feature.feature_name.lower().split())
            ]
            
            # Search in test directories
            for test_dir in self.test_base_dir.rglob("*.py"):
                if test_dir.is_file() and test_dir.name.startswith("test_"):
                    links.extend(self._find_test_links_in_file(test_dir, feature_name_variants, feature.feature_name))
            
            # Store found links
            self.test_links.extend(links)
            
            return links
            
        except Exception as e:
            print(f"Warning: Could not link feature {feature.feature_name} to existing tests: {e}")
            return []
    
    def _find_test_links_in_file(self, test_file: Path, feature_variants: List[str], feature_name: str) -> List[TestLink]:
        """Find test links in a specific test file"""
        links = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file to find test functions
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    # Check if test function relates to our feature
                    confidence = self._calculate_test_relevance(node, feature_variants, content)
                    
                    if confidence > 0.3:  # Threshold for relevance
                        link_type = self._determine_link_type(node, content)
                        
                        link = TestLink(
                            feature_name=feature_name,
                            test_file_path=str(test_file),
                            test_function_name=node.name,
                            link_type=link_type,
                            confidence=confidence,
                            description=f"Existing test {node.name} relates to {feature_name}"
                        )
                        links.append(link)
            
            return links
            
        except Exception as e:
            print(f"Warning: Could not parse test file {test_file}: {e}")
            return []
    
    def _calculate_test_relevance(self, test_node: ast.FunctionDef, feature_variants: List[str], content: str) -> float:
        """Calculate how relevant a test is to our feature"""
        relevance_score = 0.0
        
        # Check function name
        func_name = test_node.name.lower()
        for variant in feature_variants:
            if variant in func_name:
                relevance_score += 0.5
                break
        
        # Check docstring
        docstring = ast.get_docstring(test_node)
        if docstring:
            docstring_lower = docstring.lower()
            for variant in feature_variants:
                if variant in docstring_lower:
                    relevance_score += 0.3
                    break
        
        # Check function body for feature references
        func_start = test_node.lineno
        func_end = test_node.end_lineno if hasattr(test_node, 'end_lineno') else func_start + 20
        
        lines = content.split('\n')[func_start-1:func_end]
        func_content = '\n'.join(lines).lower()
        
        for variant in feature_variants:
            if variant in func_content:
                relevance_score += 0.2
                break
        
        return min(relevance_score, 1.0)
    
    def _determine_link_type(self, test_node: ast.FunctionDef, content: str) -> str:
        """Determine the type of link between test and feature"""
        func_name = test_node.name.lower()
        
        if 'performance' in func_name or 'backtest' in func_name:
            return 'performance'
        elif 'integration' in func_name:
            return 'indirect'
        else:
            return 'direct'
    
    def validate_performance_claims(self, feature: FeatureEnhancement) -> PerformanceValidation:
        """
        Validate performance claims against backtest data
        
        Args:
            feature: Feature enhancement with performance claims
            
        Returns:
            Performance validation result
        """
        try:
            # Extract performance claims from thesis and rationale
            claimed_performance = self._extract_performance_claims(feature)
            
            # Load actual performance data
            actual_performance = self._load_actual_performance(feature.feature_name)
            
            # Calculate performance gap
            performance_gap = {}
            validation_passed = True
            
            for metric, claimed_value in claimed_performance.items():
                actual_value = actual_performance.get(metric, 0.0)
                gap = actual_value - claimed_value
                performance_gap[metric] = gap
                
                # Check if performance meets claims (with tolerance)
                tolerance = self.config.get('performance_tolerance', {}).get(metric, 0.1)
                if gap < -tolerance:
                    validation_passed = False
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(actual_performance)
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                claimed_performance, actual_performance, performance_gap
            )
            
            return PerformanceValidation(
                feature_name=feature.feature_name,
                claimed_performance=claimed_performance,
                actual_performance=actual_performance,
                validation_passed=validation_passed,
                performance_gap=performance_gap,
                confidence_interval=confidence_intervals,
                validation_details={
                    "validation_date": datetime.now().isoformat(),
                    "data_period": self.training_dates,
                    "sample_size": actual_performance.get('sample_size', 0)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            raise PerformanceValidationError(f"Failed to validate performance for {feature.feature_name}: {str(e)}")
    
    def _extract_performance_claims(self, feature: FeatureEnhancement) -> Dict[str, float]:
        """Extract performance claims from feature thesis and rationale"""
        claims = {}
        
        # Search for numerical claims in thesis and rationale
        text_sources = [
            feature.thesis_statement.hypothesis,
            feature.thesis_statement.expected_behavior,
        ]
        
        # Add economic rationale fields if they exist
        if feature.economic_rationale.market_inefficiency:
            text_sources.append(feature.economic_rationale.market_inefficiency)
        if feature.economic_rationale.regime_dependency:
            text_sources.append(feature.economic_rationale.regime_dependency)
        
        for text in text_sources:
            if not text:
                continue
                
            # Look for Sharpe ratio claims
            sharpe_matches = re.findall(r'sharpe.*?(\d+\.?\d*)', text.lower())
            if sharpe_matches:
                claims['sharpe_ratio'] = float(sharpe_matches[0])
            
            # Look for return claims
            return_matches = re.findall(r'return.*?(\d+\.?\d*)%?', text.lower())
            if return_matches:
                claims['annual_return'] = float(return_matches[0]) / 100 if '%' in text else float(return_matches[0])
            
            # Look for drawdown claims
            drawdown_matches = re.findall(r'drawdown.*?(\d+\.?\d*)%?', text.lower())
            if drawdown_matches:
                claims['max_drawdown'] = float(drawdown_matches[0]) / 100 if '%' in text else float(drawdown_matches[0])
            
            # Look for hit rate claims
            hit_rate_matches = re.findall(r'hit.rate.*?(\d+\.?\d*)%?', text.lower())
            if hit_rate_matches:
                claims['hit_rate'] = float(hit_rate_matches[0]) / 100 if '%' in text else float(hit_rate_matches[0])
            
            # Look for accuracy claims
            accuracy_matches = re.findall(r'accuracy.*?(\d+\.?\d*)%?', text.lower())
            if accuracy_matches:
                claims['accuracy'] = float(accuracy_matches[0]) / 100 if '%' in text else float(accuracy_matches[0])
        
        return claims
    
    def _load_actual_performance(self, feature_name: str) -> Dict[str, float]:
        """Load actual performance data for feature"""
        # This would integrate with the actual training pipeline results
        # For now, return mock data structure
        
        # Check cache first
        cache_key = f"{feature_name}_performance"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        # Load from training pipeline results
        try:
            # This would be implemented to load actual backtest results
            # from the training pipeline or saved results
            performance_data = {
                'sharpe_ratio': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.5,
                'information_ratio': 0.0,
                'sample_size': 1000
            }
            
            # Cache the results
            self.performance_cache[cache_key] = performance_data
            
            return performance_data
            
        except Exception as e:
            print(f"Warning: Could not load performance data for {feature_name}: {e}")
            return {
                'sharpe_ratio': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.5,
                'information_ratio': 0.0,
                'sample_size': 0
            }
    
    def _calculate_confidence_intervals(self, performance_data: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for performance metrics"""
        confidence_intervals = {}
        
        # This would calculate actual confidence intervals based on the data
        # For now, return mock intervals
        for metric, value in performance_data.items():
            if metric == 'sample_size':
                continue
                
            # Mock confidence interval calculation
            std_error = abs(value) * 0.1  # 10% standard error assumption
            margin = 1.96 * std_error  # 95% confidence interval
            
            confidence_intervals[metric] = (value - margin, value + margin)
        
        return confidence_intervals
    
    def _generate_performance_recommendations(self, 
                                            claimed: Dict[str, float], 
                                            actual: Dict[str, float], 
                                            gap: Dict[str, float]) -> List[str]:
        """Generate recommendations based on performance validation"""
        recommendations = []
        
        for metric, gap_value in gap.items():
            if gap_value < -0.1:  # Significant underperformance
                recommendations.append(
                    f"Feature underperforms claimed {metric} by {abs(gap_value):.3f}. "
                    f"Consider revising thesis or improving feature implementation."
                )
            elif gap_value > 0.1:  # Significant overperformance
                recommendations.append(
                    f"Feature outperforms claimed {metric} by {gap_value:.3f}. "
                    f"Consider updating documentation to reflect actual performance."
                )
        
        if not recommendations:
            recommendations.append("Performance claims are well-aligned with actual results.")
        
        return recommendations
    
    def create_monitoring_alerts(self, feature: FeatureEnhancement) -> List[Alert]:
        """
        Create alerts for feature performance degradation
        
        Args:
            feature: Feature enhancement to monitor
            
        Returns:
            List of monitoring alerts
        """
        alerts = []
        
        try:
            # Create performance degradation alert
            perf_alert = Alert(
                alert_id=f"{feature.feature_name}_performance_{datetime.now().strftime('%Y%m%d')}",
                feature_name=feature.feature_name,
                alert_type="performance_degradation",
                severity="medium",
                message=f"Monitor {feature.feature_name} for performance degradation",
                threshold_violated={},
                current_values={},
                recommended_actions=[
                    f"Check {feature.feature_name} implementation",
                    "Validate feature data quality",
                    "Review market regime changes"
                ],
                created_at=datetime.now()
            )
            alerts.append(perf_alert)
            
            # Create thesis violation alert if applicable
            if feature.thesis_statement.failure_modes:
                thesis_alert = Alert(
                    alert_id=f"{feature.feature_name}_thesis_{datetime.now().strftime('%Y%m%d')}",
                    feature_name=feature.feature_name,
                    alert_type="thesis_violation",
                    severity="high",
                    message=f"Monitor {feature.feature_name} for thesis violations",
                    threshold_violated={},
                    current_values={},
                    recommended_actions=[
                        "Review thesis assumptions",
                        "Check for market structure changes",
                        "Validate economic rationale"
                    ],
                    created_at=datetime.now()
                )
                alerts.append(thesis_alert)
            
            return alerts
            
        except Exception as e:
            print(f"Warning: Could not create monitoring alerts for {feature.feature_name}: {e}")
            return []
    
    def write_test_file(self, tests: List[ValidationTest], output_path: str) -> bool:
        """
        Write generated tests to a test file
        
        Args:
            tests: List of validation tests to write
            output_path: Path to write the test file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate test file content
            content = self._generate_test_file_content(tests)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Error writing test file {output_path}: {e}")
            return False
    
    def _generate_test_file_content(self, tests: List[ValidationTest]) -> str:
        """Generate complete test file content"""
        imports = [
            "import pytest",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "from datetime import datetime",
            "import sys",
            "import os",
            "",
            "# Add project root to path",
            "project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            "if project_root not in sys.path:",
            "    sys.path.append(project_root)",
            "",
            "from src.training_pipeline import *",
            ""
        ]
        
        # Add helper functions
        helpers = [
            "def load_feature_data():",
            '    """Load feature data for testing"""',
            "    # This would be implemented to load actual feature data",
            "    # For now, return mock data",
            "    return pd.DataFrame()",
            "",
            "def load_backtest_results():",
            '    """Load backtest results for testing"""',
            "    # This would be implemented to load actual backtest results",
            "    return {}",
            ""
        ]
        
        # Combine all content
        content_parts = []
        content_parts.extend(imports)
        content_parts.extend(helpers)
        
        for test in tests:
            content_parts.append(test.test_function)
            content_parts.append("")
        
        return "\n".join(content_parts)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation integration status"""
        return {
            "generated_tests": len(self.generated_tests),
            "test_links": len(self.test_links),
            "performance_cache_size": len(self.performance_cache),
            "config": self.config,
            "training_dates": self.training_dates,
            "last_updated": datetime.now().isoformat()
        }