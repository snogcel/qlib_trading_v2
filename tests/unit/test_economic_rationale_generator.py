"""
Tests for Economic Rationale Generator

Tests the core functionality of generating economic rationale and thesis statements
for trading system features based on supply/demand principles.
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from documentation.economic_rationale_generator import (
    EconomicRationaleGenerator,
    ThesisStatement,
    EconomicRationale,
    ChartExplanation,
    SupplyDemandClassification,
    FeatureEnhancement,
    SupplyDemandRole,
    MarketLayer,
    TimeHorizon,
    validate_economic_logic
)


class TestEconomicRationaleGenerator:
    """Test suite for EconomicRationaleGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance for testing"""
        return EconomicRationaleGenerator()
    
    def test_initialization(self, generator):
        """Test generator initializes correctly"""
        assert generator is not None
        assert hasattr(generator, 'supply_demand_templates')
        assert hasattr(generator, 'feature_type_mappings')
        assert len(generator.supply_demand_templates) > 0
        assert len(generator.feature_type_mappings) > 0
    
    def test_feature_type_classification(self, generator):
        """Test feature type classification from names"""
        
        # Test quantile signal classification
        assert generator._classify_feature_type("q50") == "quantile_signal"
        assert generator._classify_feature_type("quantile_feature") == "quantile_signal"
        assert generator._classify_feature_type("probability_signal") == "quantile_signal"
        
        # Test volatility classification
        assert generator._classify_feature_type("vol_risk") == "volatility_risk"
        assert generator._classify_feature_type("volatility_measure") == "volatility_risk"
        assert generator._classify_feature_type("variance_feature") == "volatility_risk"
        
        # Test regime classification
        assert generator._classify_feature_type("regime_multiplier") == "regime_classification"
        assert generator._classify_feature_type("market_state") == "regime_classification"
        
        # Test position sizing
        assert generator._classify_feature_type("kelly_criterion") == "position_sizing"
        assert generator._classify_feature_type("position_size") == "position_sizing"
        
        # Test momentum
        assert generator._classify_feature_type("momentum_signal") == "momentum"
        assert generator._classify_feature_type("trend_feature") == "momentum"
        
        # Test sentiment
        assert generator._classify_feature_type("fear_greed") == "sentiment"
        assert generator._classify_feature_type("sentiment_indicator") == "sentiment"
        
        # Test default case
        assert generator._classify_feature_type("unknown_feature") == "quantile_signal"
    
    def test_thesis_statement_generation(self, generator):
        """Test thesis statement generation"""
        
        # Test Q50 thesis generation
        thesis = generator.generate_thesis_statement("q50", "Primary quantile signal")
        
        assert isinstance(thesis, ThesisStatement)
        assert len(thesis.hypothesis) > 50
        assert "supply" in thesis.economic_basis.lower() or "demand" in thesis.economic_basis.lower()
        assert len(thesis.market_microstructure) > 20
        assert len(thesis.expected_behavior) > 30
        assert len(thesis.failure_modes) > 0
        assert isinstance(thesis.academic_support, list)
        
        # Check content quality
        assert "predict returns" in thesis.hypothesis.lower()
        assert "imbalance" in thesis.economic_basis.lower()
        
    def test_economic_rationale_generation(self, generator):
        """Test economic rationale generation"""
        
        rationale = generator.generate_economic_rationale("vol_risk", "Volatility-based risk measure")
        
        assert isinstance(rationale, EconomicRationale)
        assert len(rationale.supply_factors) > 0
        assert len(rationale.demand_factors) > 0
        assert len(rationale.market_inefficiency) > 20
        assert len(rationale.regime_dependency) > 20
        assert len(rationale.interaction_effects) > 0
        
        # Check supply/demand content
        assert any("supply" in factor.lower() for factor in rationale.supply_factors)
        assert any("demand" in factor.lower() for factor in rationale.demand_factors)
    
    def test_chart_explanation_generation(self, generator):
        """Test chart explanation generation"""
        
        chart_exp = generator.generate_chart_explanation("momentum_signal", "Momentum indicator")
        
        assert isinstance(chart_exp, ChartExplanation)
        assert len(chart_exp.visual_description) > 30
        assert len(chart_exp.example_scenarios) > 0
        assert len(chart_exp.chart_patterns) > 0
        assert len(chart_exp.false_signals) > 0
        assert len(chart_exp.confirmation_signals) > 0
        
        # Check content quality
        assert "chart" in chart_exp.visual_description.lower()
        assert all(len(scenario) > 10 for scenario in chart_exp.example_scenarios)
    
    def test_supply_demand_classification(self, generator):
        """Test supply/demand role classification"""
        
        classification = generator.classify_supply_demand_role("regime_multiplier", "Regime-based multiplier")
        
        assert isinstance(classification, SupplyDemandClassification)
        assert isinstance(classification.primary_role, SupplyDemandRole)
        assert isinstance(classification.secondary_roles, list)
        assert isinstance(classification.market_layer, MarketLayer)
        assert isinstance(classification.time_horizon, TimeHorizon)
        assert classification.regime_sensitivity in ["high", "medium", "low"]
        assert isinstance(classification.interaction_features, list)
    
    def test_complete_enhancement_generation(self, generator):
        """Test complete enhancement package generation"""
        
        existing_content = {
            "purpose": "Primary quantile signal for trade direction",
            "type": "Quantile-based probability",
            "status": "Production Ready"
        }
        
        enhancement = generator.generate_complete_enhancement(
            "q50", 
            "Core Signal Features", 
            existing_content
        )
        
        assert isinstance(enhancement, FeatureEnhancement)
        assert enhancement.feature_name == "q50"
        assert enhancement.category == "Core Signal Features"
        assert enhancement.existing_content == existing_content
        
        # Check all components are generated
        assert isinstance(enhancement.thesis_statement, ThesisStatement)
        assert isinstance(enhancement.economic_rationale, EconomicRationale)
        assert isinstance(enhancement.chart_explanation, ChartExplanation)
        assert isinstance(enhancement.supply_demand_classification, SupplyDemandClassification)
        assert isinstance(enhancement.validation_criteria, list)
        assert isinstance(enhancement.dependencies, list)
        
        # Check validation criteria
        assert len(enhancement.validation_criteria) > 0
        assert any("economic logic" in criteria.lower() for criteria in enhancement.validation_criteria)
    
    def test_different_feature_types(self, generator):
        """Test generation for different feature types"""
        
        feature_types = [
            ("q50", "quantile_signal"),
            ("vol_risk", "volatility_risk"),
            ("regime_multiplier", "regime_classification"),
            ("kelly_criterion", "position_sizing"),
            ("momentum_hybrid", "momentum"),
            ("fear_greed_index", "sentiment")
        ]
        
        for feature_name, expected_type in feature_types:
            # Test classification
            assert generator._classify_feature_type(feature_name) == expected_type
            
            # Test thesis generation
            thesis = generator.generate_thesis_statement(feature_name)
            assert isinstance(thesis, ThesisStatement)
            assert len(thesis.hypothesis) > 30
            
            # Test rationale generation
            rationale = generator.generate_economic_rationale(feature_name)
            assert isinstance(rationale, EconomicRationale)
            assert len(rationale.supply_factors) > 0
    
    def test_validation_criteria_generation(self, generator):
        """Test validation criteria generation"""
        
        thesis = generator.generate_thesis_statement("test_feature")
        criteria = generator._generate_validation_criteria("test_feature", thesis)
        
        assert isinstance(criteria, list)
        assert len(criteria) > 0
        
        # Check for key validation types
        criteria_text = " ".join(criteria).lower()
        assert "economic logic" in criteria_text
        assert "statistical" in criteria_text
        assert "regime" in criteria_text
        assert "chart" in criteria_text
    
    def test_dependency_determination(self, generator):
        """Test dependency determination"""
        
        existing_content = {"dependencies": ["price_data", "volume_data"]}
        deps = generator._determine_dependencies("vol_risk", existing_content)
        
        assert isinstance(deps, list)
        assert "price_data" in deps
        assert "volume_data" in deps
        
        # Should add common dependencies
        assert len(deps) >= 2
    
    def test_regime_sensitivity_assignment(self, generator):
        """Test regime sensitivity assignment"""
        
        # High sensitivity features
        assert generator._determine_regime_sensitivity("q50", "quantile_signal") == "high"
        assert generator._determine_regime_sensitivity("momentum", "momentum") == "high"
        
        # Medium sensitivity features
        assert generator._determine_regime_sensitivity("vol_risk", "volatility_risk") == "medium"
        
        # Low sensitivity features (should be stable)
        assert generator._determine_regime_sensitivity("regime_classifier", "regime_classification") == "low"
    
    def test_interaction_features_determination(self, generator):
        """Test interaction features determination"""
        
        interactions = generator._determine_interaction_features("q50", "quantile_signal")
        
        assert isinstance(interactions, list)
        assert len(interactions) > 0
        
        # Should include common interaction features
        interaction_text = " ".join(interactions).lower()
        assert any(term in interaction_text for term in ["regime", "vol", "kelly"])


class TestThesisValidation:
    """Test thesis validation functionality"""
    
    def test_valid_thesis_validation(self):
        """Test validation of a valid thesis"""
        
        valid_thesis = ThesisStatement(
            hypothesis="This feature should predict returns because it captures supply/demand imbalances through quantile analysis",
            economic_basis="Supply/Demand Analysis: Feature detects supply exhaustion when sellers become scarce. Conversely, it identifies demand accumulation when buyers are willing to pay higher prices.",
            market_microstructure="Works by measuring probability distributions of price movements, capturing asymmetric order flow",
            expected_behavior="Expect higher values before upward price movements and lower values before downward movements",
            failure_modes=["Regime changes", "Market microstructure changes"],
            academic_support=["Quantile regression literature"]
        )
        
        is_valid, issues = validate_economic_logic(valid_thesis, "thesis-first development")
        
        assert is_valid
        assert len(issues) == 0
    
    def test_invalid_thesis_validation(self):
        """Test validation of invalid thesis"""
        
        invalid_thesis = ThesisStatement(
            hypothesis="Short",  # Too brief
            economic_basis="Brief",  # Too brief
            market_microstructure="Brief",  # Too brief
            expected_behavior="Brief",  # Too brief
            failure_modes=[],
            academic_support=[]
        )
        
        is_valid, issues = validate_economic_logic(invalid_thesis, "thesis-first development")
        
        assert not is_valid
        assert len(issues) > 0
        
        # Check that we get brief content issues
        issue_text = " ".join(issues).lower()
        assert "brief" in issue_text
    
    def test_partial_validation_issues(self):
        """Test validation with adequate content passes"""
        
        partial_thesis = ThesisStatement(
            hypothesis="This feature should predict returns because it captures market dynamics through advanced analysis techniques",
            economic_basis="The feature works by analyzing market patterns and identifying key supply/demand signals",
            market_microstructure="Uses complex mathematical models to understand market behavior and order flow dynamics",
            expected_behavior="Should show predictive power in various market conditions with regime awareness",
            failure_modes=["Model breakdown"],
            academic_support=["Academic research"]
        )
        
        is_valid, issues = validate_economic_logic(partial_thesis, "principles document")
        
        # Should pass with adequate content length
        assert is_valid
        assert len(issues) == 0


class TestFeatureEnhancement:
    """Test FeatureEnhancement data structure"""
    
    def test_feature_enhancement_creation(self):
        """Test creating FeatureEnhancement objects"""
        
        thesis = ThesisStatement(
            hypothesis="Test hypothesis",
            economic_basis="Test basis",
            market_microstructure="Test microstructure",
            expected_behavior="Test behavior",
            failure_modes=["Test failure"],
            academic_support=["Test support"]
        )
        
        rationale = EconomicRationale(
            supply_factors=["Supply factor"],
            demand_factors=["Demand factor"],
            market_inefficiency="Test inefficiency",
            regime_dependency="Test dependency",
            interaction_effects=["Test interaction"]
        )
        
        chart_exp = ChartExplanation(
            visual_description="Test description",
            example_scenarios=["Test scenario"],
            chart_patterns=["Test pattern"],
            false_signals=["Test false signal"],
            confirmation_signals=["Test confirmation"]
        )
        
        classification = SupplyDemandClassification(
            primary_role=SupplyDemandRole.IMBALANCE_DETECTOR,
            secondary_roles=[SupplyDemandRole.SUPPLY_DETECTOR],
            market_layer=MarketLayer.TECHNICAL,
            time_horizon=TimeHorizon.DAILY,
            regime_sensitivity="medium",
            interaction_features=["test_feature"]
        )
        
        enhancement = FeatureEnhancement(
            feature_name="test_feature",
            category="Test Category",
            existing_content={"test": "content"},
            thesis_statement=thesis,
            economic_rationale=rationale,
            chart_explanation=chart_exp,
            supply_demand_classification=classification,
            validation_criteria=["Test criteria"],
            dependencies=["test_dep"],
            validated=False
        )
        
        assert enhancement.feature_name == "test_feature"
        assert enhancement.category == "Test Category"
        assert not enhancement.validated
        assert isinstance(enhancement.thesis_statement, ThesisStatement)
        assert isinstance(enhancement.economic_rationale, EconomicRationale)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])