"""
Unit tests for ThesisEnhancementEngine

Tests the core functionality of the thesis enhancement engine including
feature parsing, enhancement generation, validation, and content preservation.
"""

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)



import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.documentation.thesis_enhancement_engine import (
    ThesisEnhancementEngine,
    FeatureCategory,
    EnhancementResult,
    ValidationResult,
    ThesisEnhancementError,
    FeatureParsingError,
    EnhancementValidationError
)
from src.documentation.economic_rationale_generator import (
    FeatureEnhancement,
    ThesisStatement,
    EconomicRationale,
    ChartExplanation,
    SupplyDemandClassification,
    SupplyDemandRole,
    MarketLayer,
    TimeHorizon
)


class TestThesisEnhancementEngine:
    """Test suite for ThesisEnhancementEngine"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_feature_doc(self):
        """Sample feature documentation content"""
        return """# Trading System Feature Documentation

## Overview
This document tracks all features in the Q50-centric, variance-aware trading system.

---

## 🎯 Core Signal Features

### Q50 (Primary Signal)
- **Type**: Quantile-based probability
- **Purpose**: Primary directional signal based on 50th percentile probability
- **Usage**: Standalone signal for trade direction decisions
- **Implementation**: `qlib_custom/custom_multi_quantile.py`
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Primary driver of returns

### Regime Multiplier
- **Type**: Unified regime-based position multiplier
- **Purpose**: Comprehensive regime-aware position scaling
- **Usage**: Position sizing with regime awareness
- **Implementation**: `qlib_custom/regime_features.py`
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Unified position scaling

---

## 📊 Risk & Volatility Features

### Vol_Risk (Variance-Based)
- **Type**: Volatility-based risk metric
- **Purpose**: Risk assessment using variance instead of standard deviation
- **Usage**: Position sizing and risk management
- **Implementation**: `vol_risk_strategic_implementation.py`
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Critical for risk-adjusted returns
"""
    
    @pytest.fixture
    def sample_principles_doc(self):
        """Sample trading principles content"""
        return """# Trading System Development Principles

## 🎯 Core Philosophy

### 1. **Thesis-First Development**
Every strategy must have a clear economic thesis explaining why it works.

### 2. **Supply & Demand Focus**
Consider supply and demand dynamics in all trading decisions.

### 3. **Rule-Based Foundation with ML Enhancement**
Start with rule-based systems and enhance with ML.

### 4. **Simplicity & Explainability**
Keep it simple - if you can't explain it on a chart, it's not good.
"""
    
    @pytest.fixture
    def engine(self, temp_dir, sample_feature_doc, sample_principles_doc):
        """Create ThesisEnhancementEngine with test files"""
        feature_doc_path = temp_dir / "FEATURE_DOCUMENTATION.md"
        principles_path = temp_dir / "TRADING_SYSTEM_PRINCIPLES.md"
        backup_dir = temp_dir / "backups"
        
        # Write test files
        feature_doc_path.write_text(sample_feature_doc, encoding='utf-8')
        principles_path.write_text(sample_principles_doc, encoding='utf-8')
        backup_dir.mkdir(exist_ok=True)
        
        return ThesisEnhancementEngine(
            str(feature_doc_path),
            str(principles_path),
            str(backup_dir)
        )
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.feature_doc_path.exists()
        assert engine.principles_path.exists()
        assert engine.protection_system is not None
        assert engine.rationale_generator is not None
        assert len(engine.feature_categories) > 0
    
    def test_load_feature_documentation(self, engine):
        """Test loading feature documentation"""
        assert "Trading System Feature Documentation" in engine.feature_content
        assert "Core Signal Features" in engine.feature_content
        assert "Q50 (Primary Signal)" in engine.feature_content
    
    def test_load_principles(self, engine):
        """Test loading trading principles"""
        assert "Trading System Development Principles" in engine.principles_content
        assert "Thesis-First Development" in engine.principles_content
        assert "Supply & Demand Focus" in engine.principles_content
    
    def test_parse_feature_categories(self, engine):
        """Test parsing of feature categories"""
        categories = engine.feature_categories
        
        # Should find categories
        assert len(categories) >= 2
        
        # Check category names
        category_names = [cat.name for cat in categories]
        assert "Core Signal Features" in category_names
        assert "Risk & Volatility Features" in category_names
        
        # Check features in categories
        core_category = next(cat for cat in categories if cat.name == "Core Signal Features")
        assert len(core_category.features) >= 2
        
        feature_names = [f['name'] for f in core_category.features]
        assert "Q50 (Primary Signal)" in feature_names
        assert "Regime Multiplier" in feature_names
    
    def test_extract_feature_info(self, engine):
        """Test extraction of feature information"""
        categories = engine.feature_categories
        core_category = next(cat for cat in categories if cat.name == "Core Signal Features")
        
        q50_feature = next(f for f in core_category.features if f['name'] == "Q50 (Primary Signal)")
        
        # Check extracted properties
        assert q50_feature['type'] == "Quantile-based probability"
        assert q50_feature['purpose'] == "Primary directional signal based on 50th percentile probability"
        assert q50_feature['status'] == "✅ Production Ready"
        assert "High" in q50_feature['performance_impact']
    
    def test_find_category(self, engine):
        """Test finding categories by name"""
        # Should find existing category
        category = engine._find_category("Core Signal Features")
        assert category is not None
        assert category.name == "Core Signal Features"
        
        # Should not find non-existent category
        category = engine._find_category("Non-existent Category")
        assert category is None
        
        # Should be case insensitive
        category = engine._find_category("core signal features")
        assert category is not None
    
    @patch('src.documentation.thesis_enhancement_engine.EconomicRationaleGenerator')
    def test_enhance_feature_category_success(self, mock_generator_class, engine):
        """Test successful feature category enhancement"""
        # Mock the rationale generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        engine.rationale_generator = mock_generator
        
        # Create mock enhancement
        mock_enhancement = FeatureEnhancement(
            feature_name="Q50 (Primary Signal)",
            category="Core Signal Features",
            existing_content={},
            thesis_statement=ThesisStatement(
                hypothesis="Q50 predicts returns through probability analysis",
                economic_basis="Supply/demand imbalances create predictable patterns",
                market_microstructure="Works through order flow analysis",
                expected_behavior="Higher values before rallies",
                failure_modes=["Regime changes", "Market structure changes"]
            ),
            economic_rationale=EconomicRationale(
                supply_factors=["Supply exhaustion signals"],
                demand_factors=["Demand accumulation signals"],
                market_inefficiency="Probability distribution imbalances",
                regime_dependency="Varies by market regime",
                interaction_effects=["Works with volatility features"]
            ),
            chart_explanation=ChartExplanation(
                visual_description="Shows as probability levels on charts",
                example_scenarios=["High values before breakouts"],
                chart_patterns=["Divergence patterns"],
                false_signals=["During sideways markets"],
                confirmation_signals=["Volume confirmation"]
            ),
            supply_demand_classification=SupplyDemandClassification(
                primary_role=SupplyDemandRole.IMBALANCE_DETECTOR,
                secondary_roles=[],
                market_layer=MarketLayer.TECHNICAL,
                time_horizon=TimeHorizon.DAILY,
                regime_sensitivity="high",
                interaction_features=["vol_risk"]
            ),
            validation_criteria=["Sharpe ratio > 1.0"],
            dependencies=[]
        )
        
        mock_generator.generate_complete_enhancement.return_value = mock_enhancement
        
        # Test enhancement
        result = engine.enhance_feature_category("Core Signal Features")
        
        # Verify results
        assert result.success
        assert result.category_name == "Core Signal Features"
        assert len(result.enhancements_applied) > 0
        assert "Enhanced Q50 (Primary Signal) with thesis statement" in result.enhancements_applied
        assert "#### Economic Thesis" in result.enhanced_content
        assert "**Hypothesis**:" in result.enhanced_content
    
    def test_enhance_feature_category_not_found(self, engine):
        """Test enhancement of non-existent category"""
        result = engine.enhance_feature_category("Non-existent Category")
        
        assert not result.success
        assert "Category not found" in result.error_message
    
    @patch('src.documentation.thesis_enhancement_engine.DocumentProtectionSystem')
    def test_enhance_feature_category_backup_failure(self, mock_protection_class, engine):
        """Test enhancement when backup creation fails"""
        # Mock protection system to fail backup
        mock_protection = Mock()
        mock_protection.create_backup.return_value = Mock(success=False, error_message="Backup failed")
        mock_protection_class.return_value = mock_protection
        engine.protection_system = mock_protection
        
        result = engine.enhance_feature_category("Core Signal Features")
        
        assert not result.success
        assert "Backup creation failed" in result.error_message
    
    def test_format_enhanced_feature(self, engine):
        """Test formatting of enhanced feature content"""
        feature = {
            'name': 'Test Feature',
            'type': 'Test Type',
            'purpose': 'Test Purpose'
        }
        
        enhancement = FeatureEnhancement(
            feature_name="Test Feature",
            category="Test Category",
            existing_content=feature,
            thesis_statement=ThesisStatement(
                hypothesis="Test hypothesis",
                economic_basis="Test economic basis",
                market_microstructure="Test microstructure",
                expected_behavior="Test behavior",
                failure_modes=["Test failure mode"]
            ),
            economic_rationale=EconomicRationale(
                supply_factors=["Test supply factor"],
                demand_factors=["Test demand factor"],
                market_inefficiency="Test inefficiency",
                regime_dependency="Test regime dependency",
                interaction_effects=["Test interaction"]
            ),
            chart_explanation=ChartExplanation(
                visual_description="Test visual description",
                example_scenarios=["Test scenario"],
                chart_patterns=["Test pattern"],
                false_signals=["Test false signal"],
                confirmation_signals=["Test confirmation"]
            ),
            supply_demand_classification=SupplyDemandClassification(
                primary_role=SupplyDemandRole.IMBALANCE_DETECTOR,
                secondary_roles=[],
                market_layer=MarketLayer.TECHNICAL,
                time_horizon=TimeHorizon.DAILY,
                regime_sensitivity="medium",
                interaction_features=[]
            ),
            validation_criteria=["Test criterion"],
            dependencies=[]
        )
        
        formatted = engine._format_enhanced_feature(feature, enhancement, True)
        
        # Check structure
        assert "### Test Feature" in formatted
        assert "#### Economic Thesis" in formatted
        assert "**Hypothesis**: Test hypothesis" in formatted
        assert "#### Supply/Demand Analysis" in formatted
        assert "#### Chart Explainability" in formatted
        assert "#### Validation Criteria" in formatted
    
    def test_validate_enhancement(self, engine):
        """Test enhancement validation"""
        # Test valid enhancement
        valid_content = """
### Test Feature
#### Economic Thesis
**Hypothesis**: Test hypothesis
**Economic Basis**: Supply and demand analysis
**Market Microstructure**: Test microstructure
#### Supply/Demand Analysis
**Supply Factors**: Test supply
**Demand Factors**: Test demand
#### Chart Explainability
**Visual Description**: Test chart description
"""
        
        result = engine.validate_enhancement(valid_content, "Test Category")
        
        assert result.is_valid
        assert result.alignment_score > 0.6
        assert len(result.missing_elements) == 0
        
        # Test invalid enhancement (missing elements)
        invalid_content = """
### Test Feature
Some basic content without thesis elements
"""
        
        result = engine.validate_enhancement(invalid_content, "Test Category")
        
        assert not result.is_valid
        assert result.alignment_score < 0.6
        assert len(result.missing_elements) > 0
    
    def test_list_categories(self, engine):
        """Test listing of categories"""
        categories = engine.list_categories()
        
        assert isinstance(categories, list)
        assert len(categories) >= 2
        assert "Core Signal Features" in categories
        assert "Risk & Volatility Features" in categories
    
    def test_get_category_info(self, engine):
        """Test getting category information"""
        info = engine.get_category_info("Core Signal Features")
        
        assert info is not None
        assert info['name'] == "Core Signal Features"
        assert info['feature_count'] >= 2
        assert "Q50 (Primary Signal)" in info['features']
        assert isinstance(info['has_enhancements'], bool)
        
        # Test non-existent category
        info = engine.get_category_info("Non-existent Category")
        assert info is None
    
    def test_get_enhancement_status(self, engine):
        """Test getting enhancement status"""
        status = engine.get_enhancement_status()
        
        assert 'total_categories' in status
        assert 'enhanced_categories' in status
        assert 'enhancement_percentage' in status
        assert 'categories' in status
        assert 'enhancement_history' in status
        
        assert status['total_categories'] >= 2
        assert isinstance(status['enhancement_percentage'], (int, float))
        assert isinstance(status['categories'], list)
    
    def test_check_has_enhancements(self, engine):
        """Test checking if category has enhancements"""
        # Create category without enhancements
        category_without = FeatureCategory(
            name="Test Category",
            section_title="## Test Category",
            features=[],
            content_start=0,
            content_end=10,
            raw_content="Basic content without thesis"
        )
        
        assert not engine._check_has_enhancements(category_without)
        
        # Create category with enhancements
        category_with = FeatureCategory(
            name="Enhanced Category",
            section_title="## Enhanced Category",
            features=[],
            content_start=0,
            content_end=10,
            raw_content="Content with #### Economic Thesis section"
        )
        
        assert engine._check_has_enhancements(category_with)
    
    def test_error_handling_missing_files(self, temp_dir):
        """Test error handling for missing files"""
        # Test missing feature documentation
        with pytest.raises(ThesisEnhancementError, match="Feature documentation not found"):
            ThesisEnhancementEngine(
                str(temp_dir / "missing_feature.md"),
                str(temp_dir / "missing_principles.md"),
                str(temp_dir / "backups")
            )
    
    def test_error_handling_invalid_content(self, temp_dir):
        """Test error handling for invalid content"""
        # Create files with invalid content
        feature_doc_path = temp_dir / "FEATURE_DOCUMENTATION.md"
        principles_path = temp_dir / "TRADING_SYSTEM_PRINCIPLES.md"
        
        # Empty or malformed content
        feature_doc_path.write_text("", encoding='utf-8')
        principles_path.write_text("", encoding='utf-8')
        
        # Should still initialize but with empty categories
        engine = ThesisEnhancementEngine(
            str(feature_doc_path),
            str(principles_path),
            str(temp_dir / "backups")
        )
        
        assert len(engine.feature_categories) == 0
    
    @patch('src.documentation.thesis_enhancement_engine.EconomicRationaleGenerator')
    def test_enhancement_with_generator_failure(self, mock_generator_class, engine):
        """Test enhancement when rationale generator fails"""
        # Mock generator to raise exception
        mock_generator = Mock()
        mock_generator.generate_complete_enhancement.side_effect = Exception("Generator failed")
        mock_generator_class.return_value = mock_generator
        engine.rationale_generator = mock_generator
        
        result = engine.enhance_feature_category("Core Signal Features")
        
        # Should still succeed but with warnings
        assert result.success
        assert len(result.warnings) > 0
        assert any("Failed to enhance" in warning for warning in result.warnings)
    
    def test_reconstruct_category_content(self, engine):
        """Test reconstruction of category content"""
        category = FeatureCategory(
            name="Test Category",
            section_title="## 🎯 Test Category",
            features=[],
            content_start=0,
            content_end=10,
            raw_content=""
        )
        
        enhanced_features = [
            "### Feature 1\nContent 1",
            "### Feature 2\nContent 2"
        ]
        
        result = engine._reconstruct_category_content(category, enhanced_features)
        
        assert "## 🎯 Test Category" in result
        assert "### Feature 1" in result
        assert "### Feature 2" in result
        assert "---" in result  # Separator between features


class TestFeatureParsingEdgeCases:
    """Test edge cases in feature parsing"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_parse_malformed_features(self, temp_dir):
        """Test parsing of malformed feature content"""
        malformed_content = """# Feature Documentation

## 🎯 Core Features

### Feature Without Properties

### Feature With Malformed Properties
- **Type** Missing colon
- **Purpose**: Valid purpose
- Invalid line without markers

## Section Without Features

### Another Feature
- **Type**: Valid type
"""
        
        feature_doc_path = temp_dir / "FEATURE_DOCUMENTATION.md"
        principles_path = temp_dir / "TRADING_SYSTEM_PRINCIPLES.md"
        
        feature_doc_path.write_text(malformed_content, encoding='utf-8')
        principles_path.write_text("# Principles", encoding='utf-8')
        
        engine = ThesisEnhancementEngine(
            str(feature_doc_path),
            str(principles_path),
            str(temp_dir / "backups")
        )
        
        # Should parse what it can
        assert len(engine.feature_categories) >= 1
        
        core_category = next((cat for cat in engine.feature_categories if cat.name == "Core Features"), None)
        assert core_category is not None
        assert len(core_category.features) >= 2
    
    def test_parse_empty_categories(self, temp_dir):
        """Test parsing of empty categories"""
        empty_content = """# Feature Documentation

## 🎯 Empty Category

## 📊 Another Empty Category

---

## ✅ Category With Content

### Actual Feature
- **Type**: Test type
"""
        
        feature_doc_path = temp_dir / "FEATURE_DOCUMENTATION.md"
        principles_path = temp_dir / "TRADING_SYSTEM_PRINCIPLES.md"
        
        feature_doc_path.write_text(empty_content, encoding='utf-8')
        principles_path.write_text("# Principles", encoding='utf-8')
        
        engine = ThesisEnhancementEngine(
            str(feature_doc_path),
            str(principles_path),
            str(temp_dir / "backups")
        )
        
        # Should handle empty categories
        categories = engine.feature_categories
        empty_category = next((cat for cat in categories if cat.name == "Empty Category"), None)
        assert empty_category is not None
        assert len(empty_category.features) == 0
        
        content_category = next((cat for cat in categories if cat.name == "Category With Content"), None)
        assert content_category is not None
        assert len(content_category.features) == 1


if __name__ == "__main__":
    pytest.main([__file__])