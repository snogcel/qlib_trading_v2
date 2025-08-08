"""
Unit tests for the FeatureSpec data model and markdown parsing utilities.
"""

import pytest
from pathlib import Path
import tempfile
import os

from src.testing.models.feature_spec import FeatureSpec
from src.testing.parsers.markdown_parser import MarkdownFeatureParser, ParseError
from src.testing.utils.validation_utils import validate_feature_spec_data, validate_template_file


class TestFeatureSpec:
    """Test cases for FeatureSpec data model."""
    
    def test_basic_creation(self):
        """Test basic FeatureSpec creation."""
        feature = FeatureSpec(
            name="test_feature",
            category="Core Signal Features",
            tier="Tier 1",
            implementation="Test implementation"
        )
        
        assert feature.name == "test_feature"
        assert feature.category == "Core Signal Features"
        assert feature.tier == "Tier 1"
        assert feature.validation_priority == "high"  # Auto-set for Tier 1
        assert feature.is_critical_feature()
    
    def test_validation_priority_assignment(self):
        """Test automatic validation priority assignment based on tier."""
        tier1_feature = FeatureSpec(name="test1", category="Test", tier="Tier 1", implementation="test")
        tier2_feature = FeatureSpec(name="test2", category="Test", tier="Tier 2", implementation="test")
        tier3_feature = FeatureSpec(name="test3", category="Test", tier="Tier 3", implementation="test")
        
        assert tier1_feature.validation_priority == "high"
        assert tier2_feature.validation_priority == "medium"
        assert tier3_feature.validation_priority == "low"
    
    def test_test_requirements(self):
        """Test test requirements generation."""
        feature = FeatureSpec(
            name="test_feature",
            category="Test",
            tier="Tier 1",
            implementation="test",
            economic_hypothesis="test hypothesis",
            performance_characteristics={"metric": "value"},
            failure_modes=["mode1"],
            regime_dependencies={"bull": "good"},
            interactions=["feature1"]
        )
        
        requirements = feature.get_test_requirements()
        expected = ["implementation", "economic_hypothesis", "performance", "failure_modes", "regime_dependency", "interaction"]
        
        assert all(req in requirements for req in expected)
    
    def test_complexity_score(self):
        """Test complexity score calculation."""
        simple_feature = FeatureSpec(name="simple", category="Test", tier="Tier 1", implementation="test")
        complex_feature = FeatureSpec(
            name="complex",
            category="Test", 
            tier="Tier 1",
            implementation="test",
            formula="x = y + z",
            failure_modes=["mode1", "mode2"],
            interactions=["feat1", "feat2"],
            regime_dependencies={"bull": "good"},
            test_complexity="complex"
        )
        
        assert simple_feature.get_complexity_score() == 1
        assert complex_feature.get_complexity_score() > simple_feature.get_complexity_score()
    
    def test_validation_errors(self):
        """Test validation error handling."""
        with pytest.raises(ValueError, match="Feature name cannot be empty"):
            FeatureSpec(name="", category="Test", tier="Tier 1", implementation="test")
        
        with pytest.raises(ValueError, match="Feature category cannot be empty"):
            FeatureSpec(name="test", category="", tier="Tier 1", implementation="test")
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        feature = FeatureSpec(
            name="test_feature",
            category="Test",
            tier="Tier 1",
            implementation="test implementation"
        )
        
        feature_dict = feature.to_dict()
        
        assert feature_dict['name'] == "test_feature"
        assert feature_dict['category'] == "Test"
        assert feature_dict['tier'] == "Tier 1"
        assert feature_dict['implementation'] == "test implementation"


class TestMarkdownFeatureParser:
    """Test cases for MarkdownFeatureParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MarkdownFeatureParser()
        
        # Create a sample markdown template
        self.sample_template = """
# Feature Knowledge Template

## Core Signal Features

### Q50 (Primary Signal)
**Implementation**: Found in training pipeline, qlib_custom/custom_multi_quantile.py

- Economic Hypothesis: Q50 reflects the probabilistic median of future returns—a directional vote shaped by current feature context.

- Performance Characteristics: 
  - Sharpe ratio: ~1.2–2.4 (regime-dependent)
  - Hit rate: 58–65% in trending conditions

- Failure Modes: 
  - Regime misclassification causing overconfidence in weak signal
  - Stale or misaligned inputs leading to false positives

- Regime Dependencies: 
  - Bull market: Strong in momentum/trending environments
  - Bear market: Mixed during chop unless paired with volatility

- Interaction Effects: 
  - Amplified by Q90–Q10 spread as confidence measure
  - Works synergistically with vol_risk for trade sizing

## Risk & Volatility Features

### Vol_Risk (Variance-Based)
**Formula**: Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)
**Implementation**: src/data/crypto_loader.py

- Economic Hypothesis: Variance preserves squared deviations from the mean, retaining signal intensity of extreme movements.

- Performance Characteristics:
  - Distinguishes explosive instability from mild chop
  - Enhances RL position gating during risk spikes

**Empirical Ranges**:
- Typical range: `0.005 – 0.05` for most assets during stable regimes
- Extremes: `> 0.1` during flash crashes

- Failure Modes:
  - Flat markets: minimal signal, can amplify noise
  - Synthetic volatility: increased sensitivity to false positives
"""
    
    def test_extract_feature_sections(self):
        """Test feature section extraction."""
        sections = self.parser.extract_feature_sections(self.sample_template)
        
        assert "Q50" in sections
        assert "Vol_Risk" in sections
        assert len(sections) == 2
        
        # Check that category information is embedded
        assert "Core Signal" in sections["Q50"]
        assert "Risk & Volatility" in sections["Vol_Risk"]
    
    def test_parse_feature_details(self):
        """Test detailed feature parsing."""
        sections = self.parser.extract_feature_sections(self.sample_template)
        q50_section = sections["Q50"]
        
        feature = self.parser.parse_feature_details(q50_section, "Q50")
        
        assert feature.name == "Q50"
        assert "Core Signal" in feature.category
        assert feature.tier == "Tier 1"  # Core Signal features are Tier 1
        assert feature.economic_hypothesis
        assert len(feature.failure_modes) > 0
        assert len(feature.interactions) > 0
        assert len(feature.regime_dependencies) > 0
    
    def test_formula_extraction(self):
        """Test formula extraction."""
        sections = self.parser.extract_feature_sections(self.sample_template)
        vol_risk_section = sections["Vol_Risk"]
        
        feature = self.parser.parse_feature_details(vol_risk_section, "Vol_Risk")
        
        assert feature.formula is not None
        assert "Std(Log(" in feature.formula
    
    def test_empirical_ranges_extraction(self):
        """Test empirical ranges extraction."""
        sections = self.parser.extract_feature_sections(self.sample_template)
        vol_risk_section = sections["Vol_Risk"]
        
        feature = self.parser.parse_feature_details(vol_risk_section, "Vol_Risk")
        
        assert feature.empirical_ranges
        assert "typical_range" in feature.empirical_ranges or len(feature.empirical_ranges) > 0
    
    def test_template_validation(self):
        """Test template format validation."""
        # Create temporary valid template
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(self.sample_template)
            temp_path = Path(f.name)
        
        try:
            assert self.parser.validate_template_format(temp_path)
        finally:
            os.unlink(temp_path)
        
        # Test invalid template
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write("# Just a title\nNo features here")
            temp_path = Path(f.name)
        
        try:
            assert not self.parser.validate_template_format(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_template_file_not_found(self):
        """Test parsing non-existent template file."""
        with pytest.raises(FileNotFoundError):
            self.parser.parse_template(Path("nonexistent.md"))
    
    def test_parse_template_success(self):
        """Test successful template parsing."""
        # Create temporary template
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(self.sample_template)
            temp_path = Path(f.name)
        
        try:
            features = self.parser.parse_template(temp_path)
            assert len(features) == 2
            assert any(f.name == "Q50" for f in features)
            assert any(f.name == "Vol_Risk" for f in features)
        finally:
            os.unlink(temp_path)


class TestValidationUtils:
    """Test cases for validation utilities."""
    
    def test_validate_feature_spec_data_valid(self):
        """Test validation of valid feature spec data."""
        data = {
            'name': 'test_feature',
            'category': 'Core Signal Features',
            'tier': 'Tier 1',
            'implementation': 'test implementation'
        }
        
        errors = validate_feature_spec_data(data)
        assert len(errors) == 0
    
    def test_validate_feature_spec_data_missing_required(self):
        """Test validation with missing required fields."""
        data = {
            'category': 'Core Signal Features'
            # Missing 'name'
        }
        
        errors = validate_feature_spec_data(data)
        assert len(errors) > 0
        assert any("name" in error for error in errors)
    
    def test_validate_feature_spec_data_invalid_tier(self):
        """Test validation with invalid tier."""
        data = {
            'name': 'test_feature',
            'category': 'Core Signal Features',
            'tier': 'Tier 5'  # Invalid tier
        }
        
        errors = validate_feature_spec_data(data)
        assert len(errors) > 0
        assert any("tier" in error for error in errors)
    
    def test_validate_template_file_exists(self):
        """Test template file validation for existing file."""
        # Test with actual template file
        template_path = Path("docs/FEATURE_KNOWLEDGE_TEMPLATE.md")
        if template_path.exists():
            errors = validate_template_file(template_path)
            assert len(errors) == 0
    
    def test_validate_template_file_not_exists(self):
        """Test template file validation for non-existent file."""
        errors = validate_template_file(Path("nonexistent.md"))
        assert len(errors) > 0
        assert any("does not exist" in error for error in errors)