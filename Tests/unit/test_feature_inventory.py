"""
Unit tests for feature inventory generation functionality.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.testing.parsers.feature_inventory import (
    FeatureInventoryGenerator, 
    FeatureInventory,
    FeatureCategory,
    FeatureDependency
)
from src.testing.parsers.feature_template_parser import FeatureTemplateParser
from src.testing.models.feature_spec import FeatureSpec


class TestFeatureInventoryGenerator:
    """Test the FeatureInventoryGenerator class."""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return [
            FeatureSpec(
                name="Q50",
                category="Core Signal Features",
                tier="Tier 1",
                implementation="qlib_custom/custom_multi_quantile.py",
                economic_hypothesis="Primary directional signal",
                performance_characteristics={"sharpe": 1.5, "hit_rate": 0.62},
                failure_modes=["regime misclassification", "whipsaw environments"],
                regime_dependencies={"bull": "strong", "bear": "medium"},
                interactions=["Q90-Q10 spread", "vol_risk"],
                synergies=["spread confidence measure"],
                validation_priority="high"
            ),
            FeatureSpec(
                name="vol_risk",
                category="Risk & Volatility Features", 
                tier="Tier 1",
                implementation="src/data/crypto_loader.py",
                formula="Std(Log($close / Ref($close, 1)), 6)^2",
                economic_hypothesis="Variance-based risk measure",
                performance_characteristics={"volatility_capture": 0.85},
                failure_modes=["flat markets", "synthetic volatility"],
                interactions=["Kelly sizing", "Q50"],
                validation_priority="high"
            ),
            FeatureSpec(
                name="spread",
                category="Core Signal Features",
                tier="Tier 1", 
                implementation="df['spread'] = df['q90'] - df['q10']",
                economic_hypothesis="Uncertainty measure",
                performance_characteristics={"correlation_with_vol": 0.7},
                interactions=["Q50", "vol_risk"],
                validation_priority="high"
            )
        ]
    
    @pytest.fixture
    def mock_template_content(self):
        """Mock template content for testing."""
        return """
# Feature Knowledge Template

## 🎯 Core Signal Features

### Q50 (Primary Signal)
**Implementation**: qlib_custom/custom_multi_quantile.py
- Economic Hypothesis: Primary directional signal
- Interaction Effects: Amplified by Q90–Q10 spread, works synergistically with vol_risk

### Spread (Q90 - Q10)
**Implementation**: df["spread"] = df["q90"] - df["q10"]
- Economic Hypothesis: Uncertainty measure
- Uses Q50 for directional bias

## Risk & Volatility Features

### Vol_Risk (Variance-Based)
**Formula**: Std(Log($close / Ref($close, 1)), 6)^2
- Economic Hypothesis: Variance-based risk measure
- Enhanced by Kelly sizing logic
- Works with Q50 for trade sizing
"""
    
    def test_generate_inventory_basic(self, sample_features, mock_template_content):
        """Test basic inventory generation."""
        generator = FeatureInventoryGenerator()
        
        # Mock the parser
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        # Mock template file
        mock_path = Mock()
        mock_path.read_text.return_value = mock_template_content
        
        inventory = generator.generate_inventory(mock_path)
        
        assert isinstance(inventory, FeatureInventory)
        assert len(inventory.features) == 3
        assert "Q50" in inventory.features
        assert "vol_risk" in inventory.features
        assert "spread" in inventory.features
    
    def test_category_generation(self, sample_features, mock_template_content):
        """Test category generation from template sections."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = mock_template_content
        
        inventory = generator.generate_inventory(mock_path)
        
        # Check categories were created
        assert len(inventory.categories) >= 2
        assert "Core Signal Features" in inventory.categories
        assert "Risk & Volatility Features" in inventory.categories
        
        # Check category properties
        core_category = inventory.categories["Core Signal Features"]
        assert core_category.priority == "high"
        assert core_category.emoji == "🎯"
        assert "Q50" in core_category.features
        assert "spread" in core_category.features
        
        risk_category = inventory.categories["Risk & Volatility Features"]
        assert risk_category.priority == "high"
        assert risk_category.emoji == "📊"
        assert "vol_risk" in risk_category.features
    
    def test_dependency_detection(self, sample_features, mock_template_content):
        """Test dependency detection from template content."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = mock_template_content
        
        inventory = generator.generate_inventory(mock_path)
        
        # Check dependencies were detected
        assert len(inventory.dependencies) > 0
        
        # Check for specific dependencies
        dep_pairs = [(d.source_feature, d.target_feature) for d in inventory.dependencies]
        
        # Should detect synergy between vol_risk and Q50
        synergy_deps = [d for d in inventory.dependencies if d.dependency_type == 'synergy']
        assert len(synergy_deps) > 0
    
    def test_test_requirements_generation(self, sample_features):
        """Test test requirements generation."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = ""
        
        inventory = generator.generate_inventory(mock_path)
        
        # Check test requirements were generated
        assert len(inventory.test_requirements) == 3
        
        # Check Q50 requirements (should include regime and performance tests)
        q50_reqs = inventory.test_requirements["Q50"]
        assert "implementation" in q50_reqs
        assert "economic_hypothesis" in q50_reqs
        assert "regime_dependency" in q50_reqs
        assert "performance" in q50_reqs
        
        # Check vol_risk requirements (should include failure modes)
        vol_risk_reqs = inventory.test_requirements["vol_risk"]
        assert "implementation" in vol_risk_reqs
        assert "failure_modes" in vol_risk_reqs
        assert "empirical_ranges" in vol_risk_reqs
    
    def test_inventory_validation(self, sample_features):
        """Test inventory validation functionality."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = ""
        
        inventory = generator.generate_inventory(mock_path)
        
        # Check validation results
        validation = inventory.validation_summary
        assert "completeness_score" in validation
        assert "consistency_issues" in validation
        assert "missing_data" in validation
        assert "warnings" in validation
        
        # Should have high completeness for well-formed features
        assert validation["completeness_score"] > 50
    
    def test_critical_features_identification(self, sample_features):
        """Test identification of critical features."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = ""
        
        inventory = generator.generate_inventory(mock_path)
        
        critical_features = inventory.get_critical_features()
        
        # All sample features are Tier 1, so should all be critical
        assert len(critical_features) == 3
        assert all(f.is_critical_feature() for f in critical_features)
    
    def test_dependency_chain_analysis(self, sample_features):
        """Test dependency chain analysis."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = ""
        
        inventory = generator.generate_inventory(mock_path)
        
        # Test dependency chain retrieval
        chain = inventory.get_dependency_chain("Q50")
        assert isinstance(chain, list)
        
        # Should handle circular dependencies gracefully
        chain = inventory.get_dependency_chain("nonexistent_feature")
        assert chain == []
    
    def test_test_coverage_summary(self, sample_features):
        """Test test coverage summary generation."""
        generator = FeatureInventoryGenerator()
        
        mock_parser = Mock()
        mock_parser.parse_template.return_value = sample_features
        generator.parser = mock_parser
        
        mock_path = Mock()
        mock_path.read_text.return_value = ""
        
        inventory = generator.generate_inventory(mock_path)
        
        summary = inventory.get_test_coverage_summary()
        
        assert summary["total_features"] == 3
        assert summary["critical_features"] == 3
        assert "test_type_distribution" in summary
        assert "coverage_completeness" in summary
        
        # Check test type distribution
        test_dist = summary["test_type_distribution"]
        assert "implementation" in test_dist
        assert test_dist["implementation"] == 3  # All features need implementation tests


class TestFeatureTemplateParserIntegration:
    """Test integration of inventory generation with FeatureTemplateParser."""
    
    def test_parser_inventory_generation(self):
        """Test that parser can generate inventory."""
        parser = FeatureTemplateParser()
        
        # Mock the underlying methods
        with patch.object(parser, 'parse_template') as mock_parse:
            mock_parse.return_value = [
                FeatureSpec(
                    name="test_feature",
                    category="Test Category",
                    tier="Tier 2",
                    implementation="test implementation"
                )
            ]
            
            mock_path = Mock(spec=Path)
            mock_path.read_text.return_value = "## Test Category\n### test_feature\nTest content"
            
            inventory = parser.generate_feature_inventory(mock_path)
            
            assert isinstance(inventory, FeatureInventory)
            assert len(inventory.features) == 1
            assert "test_feature" in inventory.features
    
    def test_parser_inventory_error_handling(self):
        """Test error handling in inventory generation."""
        parser = FeatureTemplateParser()
        
        # Test with invalid path
        with pytest.raises(ValueError, match="Inventory generation failed"):
            with patch('src.testing.parsers.feature_inventory.FeatureInventoryGenerator') as mock_gen_class:
                mock_gen = Mock()
                mock_gen.generate_inventory.side_effect = Exception("Test error")
                mock_gen_class.return_value = mock_gen
                parser.generate_feature_inventory(Path("nonexistent.md"))


if __name__ == "__main__":
    pytest.main([__file__])