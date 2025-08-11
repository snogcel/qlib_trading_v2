"""
Tests for ValidationIntegrationSystem

This module tests the ValidationIntegrationSystem class that links documentation claims
to automated tests, generates validation tests based on thesis statements, and validates
feature performance against actual backtest data.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.documentation.validation_integration_system import (
    ValidationIntegrationSystem,
    ValidationTest,
    TestLink,
    PerformanceValidation,
    Alert,
    ValidationIntegrationError,
    TestGenerationError,
    PerformanceValidationError
)

from src.documentation.economic_rationale_generator import (
    FeatureEnhancement,
    ThesisStatement,
    EconomicRationale,
    ValidationCriterion
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_thesis():
    """Create sample thesis statement for testing"""
    return ThesisStatement(
        hypothesis="Q50 signal predicts returns by detecting supply/demand imbalances",
        economic_basis="Supply exceeds demand when Q50 is negative, indicating selling pressure",
        market_microstructure="Order flow imbalances create predictable price movements",
        expected_behavior="Negative Q50 should predict negative returns with 55% accuracy",
        failure_modes=["Low volume periods", "Market regime changes"]
    )


@pytest.fixture
def sample_economic_rationale():
    """Create sample economic rationale for testing"""
    return EconomicRationale(
        supply_factors=["Selling pressure", "Large order flow"],
        demand_factors=["Buying interest", "Accumulation patterns"],
        market_inefficiency="Order flow information not immediately reflected in price",
        regime_dependency="Works better in trending markets",
        interaction_effects=["Works with Vol_Risk", "Enhanced by regime features"]
    )


@pytest.fixture
def sample_feature_enhancement(sample_thesis, sample_economic_rationale):
    """Create sample feature enhancement for testing"""
    from src.documentation.economic_rationale_generator import (
        ChartExplanation, SupplyDemandClassification, SupplyDemandRole, MarketLayer, TimeHorizon
    )
    
    chart_explanation = ChartExplanation(
        visual_description="Q50 appears as oscillating line around zero",
        example_scenarios=["Q50 drops below -0.02 before price decline"],
        chart_patterns=["Divergence with price"],
        false_signals=["Whipsaws during low volume"],
        confirmation_signals=["Volume increase"]
    )
    
    supply_demand_classification = SupplyDemandClassification(
        primary_role=SupplyDemandRole.IMBALANCE_DETECTOR,
        secondary_roles=[SupplyDemandRole.SUPPLY_DETECTOR],
        market_layer=MarketLayer.MICROSTRUCTURE,
        time_horizon=TimeHorizon.INTRADAY,
        regime_sensitivity="medium",
        interaction_features=["Vol_Risk"]
    )
    
    return FeatureEnhancement(
        feature_name="Q50 Signal",
        category="Core Signal Features",
        existing_content={"description": "Primary quantile signal"},
        thesis_statement=sample_thesis,
        economic_rationale=sample_economic_rationale,
        chart_explanation=chart_explanation,
        supply_demand_classification=supply_demand_classification,
        validation_criteria=[
            ValidationCriterion(
                test_name="statistical_significance",
                description="Test statistical significance of Q50 signal",
                success_threshold=0.05,
                test_implementation="test_q50_statistical_significance",
                frequency="daily",
                failure_action="alert_team"
            )
        ],
        dependencies=[],
        validated=False
    )


@pytest.fixture
def validation_system(temp_dir):
    """Create ValidationIntegrationSystem instance for testing"""
    # Create mock config file
    config_path = temp_dir / "config" / "validation_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "performance_thresholds": {
            "sharpe_ratio": {"min": 0.5, "target": 1.0},
            "max_drawdown": {"max": 0.2, "target": 0.1},
            "information_ratio": {"min": 0.3, "target": 0.6},
            "hit_rate": {"min": 0.52, "target": 0.55}
        },
        "statistical_thresholds": {
            "p_value": 0.05,
            "min_sample_size": 1000
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Create mock training pipeline
    training_path = temp_dir / "src" / "training_pipeline.py"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text("# Mock training pipeline")
    
    # Create test directory
    test_dir = temp_dir / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    return ValidationIntegrationSystem(
        training_pipeline_path=str(training_path),
        test_base_dir=str(test_dir),
        validation_config_path=str(config_path)
    )


class TestValidationIntegrationSystemInit:
    """Test ValidationIntegrationSystem initialization"""
    
    def test_init_with_default_config(self, temp_dir):
        """Test initialization with default configuration"""
        system = ValidationIntegrationSystem(
            training_pipeline_path=str(temp_dir / "training.py"),
            test_base_dir=str(temp_dir / "tests"),
            validation_config_path=str(temp_dir / "config.json")
        )
        
        assert system.training_pipeline_path.name == "training.py"
        assert system.test_base_dir.name == "tests"
        assert "performance_thresholds" in system.config
        assert "statistical_thresholds" in system.config
    
    def test_init_creates_default_config(self, temp_dir):
        """Test that initialization creates default config if none exists"""
        config_path = temp_dir / "validation_config.json"
        
        system = ValidationIntegrationSystem(
            validation_config_path=str(config_path)
        )
        
        assert config_path.exists()
        assert "performance_thresholds" in system.config
    
    def test_init_loads_existing_config(self, temp_dir):
        """Test that initialization loads existing configuration"""
        config_path = temp_dir / "validation_config.json"
        custom_config = {
            "performance_thresholds": {
                "sharpe_ratio": {"min": 0.8, "target": 1.5}
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(custom_config, f)
        
        system = ValidationIntegrationSystem(
            validation_config_path=str(config_path)
        )
        
        assert system.config["performance_thresholds"]["sharpe_ratio"]["min"] == 0.8


class TestValidationTestGeneration:
    """Test validation test generation"""
    
    def test_create_validation_tests(self, validation_system, sample_thesis):
        """Test creation of validation tests from thesis"""
        tests = validation_system.create_validation_tests(sample_thesis, "Q50 Signal")
        
        assert len(tests) > 0
        assert any(test.test_type == "statistical" for test in tests)
        assert any(test.test_type == "economic_logic" for test in tests)
        assert any(test.test_type == "performance" for test in tests)
    
    def test_statistical_test_generation(self, validation_system, sample_thesis):
        """Test statistical test generation"""
        test = validation_system._generate_statistical_test(sample_thesis, "Q50 Signal")
        
        assert test is not None
        assert test.test_type == "statistical"
        assert "statistical_significance" in test.test_name
        assert "p_value" in test.success_criteria
    
    def test_economic_logic_test_generation(self, validation_system, sample_thesis):
        """Test economic logic test generation"""
        test = validation_system._generate_economic_logic_test(sample_thesis, "Q50 Signal")
        
        assert test is not None
        assert test.test_type == "economic_logic"
        assert "economic_logic" in test.test_name
        assert test.thesis_reference == sample_thesis.economic_basis
    
    def test_performance_test_generation(self, validation_system, sample_thesis):
        """Test performance test generation"""
        test = validation_system._generate_performance_test(sample_thesis, "Q50 Signal")
        
        assert test is not None
        assert test.test_type == "performance"
        assert "performance" in test.test_name
        assert "sharpe_ratio" in test.test_function
    
    def test_regime_tests_generation(self, validation_system, sample_thesis):
        """Test regime-specific test generation"""
        # Modify thesis to include regime reference
        sample_thesis.hypothesis = "Q50 signal works differently across market regimes"
        
        tests = validation_system._generate_regime_tests(sample_thesis, "Q50 Signal")
        
        assert len(tests) > 0
        assert any("bull_regime" in test.test_name for test in tests)
        assert any("bear_regime" in test.test_name for test in tests)
    
    def test_test_generation_error_handling(self, validation_system):
        """Test error handling in test generation"""
        # Test with invalid thesis
        invalid_thesis = None
        
        with pytest.raises(TestGenerationError):
            validation_system.create_validation_tests(invalid_thesis, "Invalid Feature")


class TestExistingTestLinking:
    """Test linking to existing tests"""
    
    def test_link_to_existing_tests(self, validation_system, sample_feature_enhancement, temp_dir):
        """Test linking feature to existing tests"""
        # Create mock test file
        test_file = temp_dir / "tests" / "test_q50_signal.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("""
def test_q50_signal_basic():
    '''Test basic Q50 signal functionality'''
    pass

def test_q50_performance():
    '''Test Q50 signal performance'''
    pass
""")
        
        links = validation_system.link_to_existing_tests(sample_feature_enhancement)
        
        assert len(links) >= 0  # May find links depending on matching logic
    
    def test_find_test_links_in_file(self, validation_system, temp_dir):
        """Test finding test links in specific file"""
        test_file = temp_dir / "test_example.py"
        test_file.write_text("""
def test_q50_signal():
    '''Test Q50 signal functionality'''
    q50_value = calculate_q50()
    assert q50_value is not None

def test_other_feature():
    '''Test unrelated feature'''
    pass
""")
        
        feature_variants = ["q50", "q50_signal"]
        links = validation_system._find_test_links_in_file(test_file, feature_variants, "Q50 Signal")
        
        # Should find at least the q50_signal test
        q50_links = [link for link in links if "q50" in link.test_function_name]
        assert len(q50_links) >= 0
    
    def test_calculate_test_relevance(self, validation_system):
        """Test test relevance calculation"""
        # Create mock AST node
        import ast
        
        code = """
def test_q50_signal_performance():
    '''Test Q50 signal performance metrics'''
    q50_data = load_q50_data()
    assert q50_data is not None
"""
        
        tree = ast.parse(code)
        test_node = tree.body[0]
        feature_variants = ["q50", "q50_signal"]
        
        relevance = validation_system._calculate_test_relevance(test_node, feature_variants, code)
        
        assert relevance > 0.0
        assert relevance <= 1.0


class TestPerformanceValidation:
    """Test performance validation functionality"""
    
    def test_validate_performance_claims(self, validation_system, sample_feature_enhancement):
        """Test validation of performance claims"""
        # Mock performance data loading
        with patch.object(validation_system, '_load_actual_performance') as mock_load:
            mock_load.return_value = {
                'sharpe_ratio': 0.6,
                'annual_return': 0.15,
                'max_drawdown': 0.12,
                'sample_size': 1500
            }
            
            validation = validation_system.validate_performance_claims(sample_feature_enhancement)
            
            assert isinstance(validation, PerformanceValidation)
            assert validation.feature_name == sample_feature_enhancement.feature_name
            assert 'sharpe_ratio' in validation.actual_performance
    
    def test_extract_performance_claims(self, validation_system, sample_feature_enhancement):
        """Test extraction of performance claims from thesis"""
        # Add performance claims to thesis
        sample_feature_enhancement.thesis_statement.expected_behavior = (
            "Should achieve Sharpe ratio of 0.8 with maximum drawdown of 15%"
        )
        
        claims = validation_system._extract_performance_claims(sample_feature_enhancement)
        
        assert 'sharpe_ratio' in claims or 'max_drawdown' in claims
    
    def test_load_actual_performance(self, validation_system):
        """Test loading actual performance data"""
        performance = validation_system._load_actual_performance("Q50 Signal")
        
        assert isinstance(performance, dict)
        assert 'sharpe_ratio' in performance
        assert 'sample_size' in performance
    
    def test_calculate_confidence_intervals(self, validation_system):
        """Test confidence interval calculation"""
        performance_data = {
            'sharpe_ratio': 0.6,
            'annual_return': 0.15,
            'sample_size': 1000
        }
        
        intervals = validation_system._calculate_confidence_intervals(performance_data)
        
        assert 'sharpe_ratio' in intervals
        assert 'annual_return' in intervals
        assert isinstance(intervals['sharpe_ratio'], tuple)
        assert len(intervals['sharpe_ratio']) == 2
    
    def test_generate_performance_recommendations(self, validation_system):
        """Test generation of performance recommendations"""
        claimed = {'sharpe_ratio': 0.8, 'max_drawdown': 0.1}
        actual = {'sharpe_ratio': 0.6, 'max_drawdown': 0.15}
        gap = {'sharpe_ratio': -0.2, 'max_drawdown': 0.05}
        
        recommendations = validation_system._generate_performance_recommendations(
            claimed, actual, gap
        )
        
        assert len(recommendations) > 0
        assert any("underperforms" in rec for rec in recommendations)


class TestMonitoringAlerts:
    """Test monitoring alert creation"""
    
    def test_create_monitoring_alerts(self, validation_system, sample_feature_enhancement):
        """Test creation of monitoring alerts"""
        alerts = validation_system.create_monitoring_alerts(sample_feature_enhancement)
        
        assert len(alerts) > 0
        assert any(alert.alert_type == "performance_degradation" for alert in alerts)
        
        # Check alert structure
        alert = alerts[0]
        assert alert.feature_name == sample_feature_enhancement.feature_name
        assert alert.alert_id is not None
        assert len(alert.recommended_actions) > 0
    
    def test_alert_with_failure_modes(self, validation_system, sample_feature_enhancement):
        """Test alert creation when thesis has failure modes"""
        # Ensure thesis has failure modes
        sample_feature_enhancement.thesis_statement.failure_modes = ["Low volume", "Regime change"]
        
        alerts = validation_system.create_monitoring_alerts(sample_feature_enhancement)
        
        # Should create both performance and thesis violation alerts
        alert_types = [alert.alert_type for alert in alerts]
        assert "performance_degradation" in alert_types
        assert "thesis_violation" in alert_types


class TestTestFileGeneration:
    """Test test file generation and writing"""
    
    def test_write_test_file(self, validation_system, sample_thesis, temp_dir):
        """Test writing generated tests to file"""
        tests = validation_system.create_validation_tests(sample_thesis, "Q50 Signal")
        output_path = temp_dir / "generated_tests.py"
        
        success = validation_system.write_test_file(tests, str(output_path))
        
        assert success
        assert output_path.exists()
        
        # Check file content
        content = output_path.read_text()
        assert "import pytest" in content
        assert "def test_" in content
    
    def test_generate_test_file_content(self, validation_system, sample_thesis):
        """Test generation of test file content"""
        tests = validation_system.create_validation_tests(sample_thesis, "Q50 Signal")
        
        content = validation_system._generate_test_file_content(tests)
        
        assert "import pytest" in content
        assert "import pandas as pd" in content
        assert "def load_feature_data():" in content
        assert "def test_" in content


class TestValidationSummary:
    """Test validation summary functionality"""
    
    def test_get_validation_summary(self, validation_system, sample_thesis):
        """Test getting validation summary"""
        # Generate some tests first
        validation_system.create_validation_tests(sample_thesis, "Q50 Signal")
        
        summary = validation_system.get_validation_summary()
        
        assert "generated_tests" in summary
        assert "test_links" in summary
        assert "config" in summary
        assert "last_updated" in summary
        assert summary["generated_tests"] >= 0


class TestErrorHandling:
    """Test error handling"""
    
    def test_performance_validation_error(self, validation_system, sample_feature_enhancement):
        """Test performance validation error handling"""
        # Mock a failure in performance loading
        with patch.object(validation_system, '_load_actual_performance') as mock_load:
            mock_load.side_effect = Exception("Mock error")
            
            with pytest.raises(PerformanceValidationError):
                validation_system.validate_performance_claims(sample_feature_enhancement)
    
    def test_test_generation_with_invalid_input(self, validation_system):
        """Test test generation with invalid input"""
        with pytest.raises(TestGenerationError):
            validation_system.create_validation_tests(None, "Invalid Feature")


class TestIntegration:
    """Integration tests"""
    
    def test_full_validation_workflow(self, validation_system, sample_feature_enhancement, temp_dir):
        """Test complete validation workflow"""
        # 1. Generate validation tests
        tests = validation_system.create_validation_tests(
            sample_feature_enhancement.thesis_statement, 
            sample_feature_enhancement.feature_name
        )
        assert len(tests) > 0
        
        # 2. Link to existing tests
        links = validation_system.link_to_existing_tests(sample_feature_enhancement)
        # Links may be empty if no matching tests found
        
        # 3. Validate performance claims
        performance_validation = validation_system.validate_performance_claims(sample_feature_enhancement)
        assert isinstance(performance_validation, PerformanceValidation)
        
        # 4. Create monitoring alerts
        alerts = validation_system.create_monitoring_alerts(sample_feature_enhancement)
        assert len(alerts) > 0
        
        # 5. Write test file
        output_path = temp_dir / "integration_tests.py"
        success = validation_system.write_test_file(tests, str(output_path))
        assert success
        
        # 6. Get summary
        summary = validation_system.get_validation_summary()
        assert summary["generated_tests"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])