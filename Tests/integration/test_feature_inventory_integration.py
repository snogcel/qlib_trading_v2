"""
Integration tests for feature inventory generation with real template.
"""

import pytest
from pathlib import Path

from src.testing.parsers.feature_template_parser import FeatureTemplateParser
from src.testing.parsers.feature_inventory import FeatureInventory


class TestFeatureInventoryIntegration:
    """Integration tests for feature inventory generation."""
    
    @pytest.fixture
    def template_path(self):
        """Get path to the actual feature template."""
        return Path("docs/FEATURE_KNOWLEDGE_TEMPLATE.md")
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return FeatureTemplateParser()
    
    def test_generate_inventory_from_real_template(self, parser, template_path):
        """Test inventory generation from the actual template file."""
        if not template_path.exists():
            pytest.skip("Feature template file not found")
        
        # Generate inventory
        inventory = parser.generate_feature_inventory(template_path)
        
        # Basic validation
        assert isinstance(inventory, FeatureInventory)
        assert len(inventory.features) > 0
        assert len(inventory.categories) > 0
        
        # Check for expected categories
        category_names = list(inventory.categories.keys())
        expected_categories = [
            "Core Signal Features",
            "Risk & Volatility Features", 
            "Position Sizing Features",
            "Regime & Market Features"
        ]
        
        for expected in expected_categories:
            assert any(expected in cat for cat in category_names), f"Missing category: {expected}"
        
        # Check for expected features
        feature_names = list(inventory.features.keys())
        expected_features = ["Q50", "vol_risk", "spread"]
        
        for expected in expected_features:
            assert any(expected.lower() in feat.lower() for feat in feature_names), f"Missing feature: {expected}"
    
    def test_inventory_completeness(self, parser, template_path):
        """Test that inventory has reasonable completeness."""
        if not template_path.exists():
            pytest.skip("Feature template file not found")
        
        inventory = parser.generate_feature_inventory(template_path)
        
        # Check validation summary
        validation = inventory.validation_summary
        assert validation["completeness_score"] >= 0  # Should be non-negative
        
        # Should have some dependencies
        assert len(inventory.dependencies) > 0
        
        # Should have test requirements for all features
        assert len(inventory.test_requirements) == len(inventory.features)
    
    def test_critical_features_detection(self, parser, template_path):
        """Test detection of critical features."""
        if not template_path.exists():
            pytest.skip("Feature template file not found")
        
        inventory = parser.generate_feature_inventory(template_path)
        
        critical_features = inventory.get_critical_features()
        
        # Should have some critical features
        assert len(critical_features) > 0
        
        # Critical features should be Tier 1 or high priority
        for feature in critical_features:
            assert feature.tier == "Tier 1" or feature.validation_priority == "high"
    
    def test_test_coverage_summary(self, parser, template_path):
        """Test test coverage summary generation."""
        if not template_path.exists():
            pytest.skip("Feature template file not found")
        
        inventory = parser.generate_feature_inventory(template_path)
        
        summary = inventory.get_test_coverage_summary()
        
        # Check summary structure
        required_keys = [
            'total_features', 
            'critical_features', 
            'categories', 
            'dependencies',
            'test_type_distribution',
            'coverage_completeness'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing summary key: {key}"
        
        # Check reasonable values
        assert summary['total_features'] > 0
        assert summary['categories'] > 0
        assert isinstance(summary['test_type_distribution'], dict)
        assert 0 <= summary['coverage_completeness'] <= 100


if __name__ == "__main__":
    pytest.main([__file__])