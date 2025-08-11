"""
Unit tests for the economic hypothesis test generator.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pytest
from unittest.mock import Mock, patch

from src.testing.generators.economic_hypothesis_generator import EconomicHypothesisTestGenerator
from src.testing.models.feature_spec import FeatureSpec
from src.testing.models.test_case import TestType, TestPriority


class TestEconomicHypothesisTestGenerator:
    """Test cases for the EconomicHypothesisTestGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = EconomicHypothesisTestGenerator()
    
    def test_q50_test_generation(self):
        """Test Q50 directional bias test generation."""
        # Create Q50 feature spec
        q50_feature = FeatureSpec(
            name="Q50",
            category="Core Signal Features",
            tier="Tier 1",
            implementation="qlib_custom/custom_multi_quantile.py",
            economic_hypothesis="Q50 reflects the probabilistic median of future returns—a directional vote shaped by current feature context. In asymmetric markets where liquidity, sentiment, and volatility drive price action, a skewed Q50 implies actionable imbalance.",
            performance_characteristics={
                "sharpe_ratio": 1.8,
                "hit_rate": 0.62
            }
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(q50_feature)
        
        # Verify tests were generated
        assert len(tests) >= 4, "Should generate at least 4 Q50-specific tests"
        
        # Check for directional bias test
        directional_tests = [t for t in tests if "directional bias" in t.description]
        assert len(directional_tests) >= 1, "Should include directional bias test"
        
        directional_test = directional_tests[0]
        assert directional_test.priority == TestPriority.CRITICAL
        assert directional_test.validation_criteria['threshold'] == 0.52
        assert directional_test.regime_context == 'trending_markets'
        
        # Check for probabilistic median test
        median_tests = [t for t in tests if "probabilistic median" in t.description]
        assert len(median_tests) >= 1, "Should include probabilistic median test"
        
        # Check for asymmetric market test
        asymmetry_tests = [t for t in tests if "asymmetric" in t.description]
        assert len(asymmetry_tests) >= 1, "Should include asymmetric market test"
        
        # Check for behavioral inertia test
        inertia_tests = [t for t in tests if "anticipates shifts" in t.description]
        assert len(inertia_tests) >= 1, "Should include behavioral inertia test"
    
    def test_vol_risk_test_generation(self):
        """Test vol_risk variance-based test generation."""
        # Create vol_risk feature spec
        vol_risk_feature = FeatureSpec(
            name="vol_risk",
            category="Risk & Volatility Features",
            tier="Tier 1",
            implementation="src/data/crypto_loader.py",
            formula="Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)",
            economic_hypothesis="Variance preserves squared deviations from the mean, retaining signal intensity of extreme movements. Unlike standard deviation, it does not compress tail events through square rooting, making it more suitable for amplifying asymmetric risk bursts."
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(vol_risk_feature)
        
        # Verify tests were generated
        assert len(tests) >= 4, "Should generate at least 4 vol_risk-specific tests"
        
        # Check for variance calculation test
        variance_tests = [t for t in tests if "variance calculation" in t.description]
        assert len(variance_tests) >= 1, "Should include variance calculation test"
        
        variance_test = variance_tests[0]
        assert variance_test.priority == TestPriority.CRITICAL
        assert 'formula_validation' in variance_test.test_parameters
        assert variance_test.test_parameters['formula_validation'] == 'Std(Log(close/Ref(close,1)), 6)^2'
        
        # Check for nonlinear exposure test
        nonlinear_tests = [t for t in tests if "nonlinear" in t.description]
        assert len(nonlinear_tests) >= 1, "Should include nonlinear exposure test"
        
        # Check for fragility detection test
        fragility_tests = [t for t in tests if "fragility" in t.description]
        assert len(fragility_tests) >= 1, "Should include fragility detection test"
        
        # Check for gating effectiveness test
        gating_tests = [t for t in tests if "gating" in t.description]
        assert len(gating_tests) >= 1, "Should include gating effectiveness test"
    
    def test_sentiment_feature_test_generation(self):
        """Test sentiment feature test generation."""
        # Create fg_index feature spec
        fg_index_feature = FeatureSpec(
            name="fg_index",
            category="Risk & Volatility Features",
            tier="Tier 2",
            implementation="crypto_loader.py",
            economic_hypothesis="Captures aggregate market sentiment—fear at lower values, greed at higher values. Synthesizes volatility, momentum, social media activity, dominance, trends."
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(fg_index_feature)
        
        # Verify tests were generated
        assert len(tests) >= 3, "Should generate at least 3 sentiment-specific tests"
        
        # Check for sentiment-price relationship test
        sentiment_tests = [t for t in tests if "reflects market sentiment" in t.description]
        assert len(sentiment_tests) >= 1, "Should include sentiment-price relationship test"
        
        # Check for contrarian signal test
        contrarian_tests = [t for t in tests if "contrarian" in t.description]
        assert len(contrarian_tests) >= 1, "Should include contrarian signal test"
        
        # Check for narrative alignment test
        narrative_tests = [t for t in tests if "narrative" in t.description]
        assert len(narrative_tests) >= 1, "Should include narrative alignment test"
    
    def test_kelly_sizing_test_generation(self):
        """Test Kelly sizing test generation."""
        # Create Kelly sizing feature spec
        kelly_feature = FeatureSpec(
            name="custom_kelly",
            category="Position Sizing Features",
            tier="Tier 1",
            implementation="training_pipeline.py",
            economic_hypothesis="Position sizing should dynamically respond to signal quality and regime volatility. Traditional Kelly assumes static payoff asymmetry and constant risk preference, but this enhanced version adapts the sizing based on predictive certainty."
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(kelly_feature)
        
        # Verify tests were generated
        assert len(tests) >= 4, "Should generate at least 4 Kelly-specific tests"
        
        # Check for risk-adjusted sizing test
        risk_tests = [t for t in tests if "risk-adjusted" in t.description]
        assert len(risk_tests) >= 1, "Should include risk-adjusted sizing test"
        
        risk_test = risk_tests[0]
        assert risk_test.priority == TestPriority.CRITICAL
        assert 'risk_scaling' in risk_test.test_parameters
        
        # Check for growth optimality test
        growth_tests = [t for t in tests if "growth optimality" in t.description]
        assert len(growth_tests) >= 1, "Should include growth optimality test"
        
        # Check for signal quality adaptation test
        adaptation_tests = [t for t in tests if "signal quality" in t.description]
        assert len(adaptation_tests) >= 1, "Should include signal quality adaptation test"
        
        # Check for regime-aware behavior test
        regime_tests = [t for t in tests if "regime-aware" in t.description]
        assert len(regime_tests) >= 1, "Should include regime-aware behavior test"
    
    def test_generic_hypothesis_test_generation(self):
        """Test generic hypothesis test generation for unknown features."""
        # Create unknown feature spec
        unknown_feature = FeatureSpec(
            name="unknown_feature",
            category="Unknown Category",
            tier="Tier 2",
            implementation="unknown.py",
            economic_hypothesis="This is a generic economic hypothesis for testing purposes."
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(unknown_feature)
        
        # Should generate generic test plus general tests
        assert len(tests) >= 2, "Should generate at least generic and general tests"
        
        # Check for generic hypothesis test
        generic_tests = [t for t in tests if "Validate economic hypothesis" in t.description]
        assert len(generic_tests) >= 1, "Should include generic hypothesis test"
        
        generic_test = generic_tests[0]
        assert generic_test.priority == TestPriority.HIGH
        assert generic_test.test_parameters['hypothesis_text'] == unknown_feature.economic_hypothesis
    
    def test_feature_key_extraction(self):
        """Test feature key extraction for generator mapping."""
        # Test Q50 key extraction
        assert self.generator._get_feature_key("Q50") == "Q50"
        assert self.generator._get_feature_key("q50_signal") == "Q50"
        
        # Test vol_risk key extraction
        assert self.generator._get_feature_key("vol_risk") == "vol_risk"
        assert self.generator._get_feature_key("VOL_RISK_SCALED") == "vol_risk"
        
        # Test sentiment key extraction
        assert self.generator._get_feature_key("fg_index") == "fg_index"
        assert self.generator._get_feature_key("btc_dom") == "btc_dom"
        assert self.generator._get_feature_key("bitcoin_dominance") == "btc_dom"
        
        # Test Kelly key extraction
        assert self.generator._get_feature_key("kelly_sizing") == "kelly_sizing"
        assert self.generator._get_feature_key("custom_kelly") == "custom_kelly"
        assert self.generator._get_feature_key("enhanced_kelly") == "enhanced_kelly"
        
        # Test unknown feature
        assert self.generator._get_feature_key("unknown_feature") == "unknown_feature"
    
    def test_no_economic_hypothesis(self):
        """Test behavior when feature has no economic hypothesis."""
        # Create feature without economic hypothesis
        feature_no_hypothesis = FeatureSpec(
            name="test_feature",
            category="Test Category",
            tier="Tier 2",
            implementation="test.py"
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(feature_no_hypothesis)
        
        # Should return empty list with warning logged
        assert len(tests) == 0, "Should return empty list when no economic hypothesis"
    
    def test_all_tests_have_required_fields(self):
        """Test that all generated tests have required fields."""
        # Create comprehensive feature spec
        feature = FeatureSpec(
            name="Q50",
            category="Core Signal Features",
            tier="Tier 1",
            implementation="test.py",
            economic_hypothesis="Test hypothesis for comprehensive validation."
        )
        
        # Generate tests
        tests = self.generator.generate_economic_hypothesis_tests(feature)
        
        # Verify all tests have required fields
        for test in tests:
            assert test.test_id, "Test should have ID"
            assert test.feature_name == "Q50", "Test should have correct feature name"
            assert test.test_type == TestType.ECONOMIC_HYPOTHESIS, "Test should have correct type"
            assert test.description, "Test should have description"
            assert test.priority, "Test should have priority"
            assert test.rationale, "Test should have rationale"
            assert test.estimated_duration > 0, "Test should have positive duration"
            assert test.validation_criteria, "Test should have validation criteria"