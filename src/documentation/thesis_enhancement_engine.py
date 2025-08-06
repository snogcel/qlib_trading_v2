"""
Thesis Enhancement Engine Core

This module implements the ThesisEnhancementEngine class that orchestrates the systematic
enhancement of feature categories with thesis statements, economic rationale, and validation
criteria while preserving existing content and ensuring alignment with trading principles.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from .document_protection import DocumentProtectionSystem, ChangeValidation
from .economic_rationale_generator import (
    EconomicRationaleGenerator, 
    FeatureEnhancement,
    ThesisStatement,
    EconomicRationale,
    ChartExplanation,
    SupplyDemandClassification
)


@dataclass
class FeatureCategory:
    """Represents a feature category from the documentation"""
    name: str
    section_title: str
    features: List[Dict[str, Any]]
    content_start: int
    content_end: int
    raw_content: str


@dataclass
class EnhancementResult:
    """Result of feature category enhancement"""
    success: bool
    category_name: str
    enhanced_content: str
    enhancements_applied: List[str]
    validation_results: List[str]
    warnings: List[str]
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of enhancement validation"""
    is_valid: bool
    alignment_score: float
    principle_violations: List[str]
    missing_elements: List[str]
    recommendations: List[str]
    error_message: Optional[str] = None


class ThesisEnhancementError(Exception):
    """Base exception for thesis enhancement errors"""
    pass


class FeatureParsingError(ThesisEnhancementError):
    """Raised when feature parsing fails"""
    pass


class EnhancementValidationError(ThesisEnhancementError):
    """Raised when enhancement validation fails"""
    pass


class ThesisEnhancementEngine:
    """
    Core orchestrator that systematically enhances feature categories with thesis statements,
    economic rationale, and validation criteria while preserving existing content and ensuring
    alignment with TRADING_SYSTEM_PRINCIPLES.md.
    """
    
    def __init__(self, 
                 feature_doc_path: str = "docs/FEATURE_DOCUMENTATION.md",
                 principles_path: str = "docs/TRADING_SYSTEM_PRINCIPLES.md",
                 backup_dir: str = "docs/research/case_study"):
        """
        Initialize the thesis enhancement engine
        
        Args:
            feature_doc_path: Path to FEATURE_DOCUMENTATION.md
            principles_path: Path to TRADING_SYSTEM_PRINCIPLES.md
            backup_dir: Directory for backups and protection
        """
        self.feature_doc_path = Path(feature_doc_path)
        self.principles_path = Path(principles_path)
        
        # Initialize subsystems
        self.protection_system = DocumentProtectionSystem(backup_dir)
        self.rationale_generator = EconomicRationaleGenerator(str(principles_path))
        
        # Load content
        self.feature_content = self._load_feature_documentation()
        self.principles_content = self._load_principles()
        
        # Parse feature structure
        self.feature_categories = self._parse_feature_categories()
        
        # Enhancement tracking
        self.enhancement_history = []
        
        # Enable protection for feature documentation
        self.protection_system.enable_version_tracking(str(self.feature_doc_path))
    
    def _load_feature_documentation(self) -> str:
        """Load feature documentation content"""
        try:
            with open(self.feature_doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise ThesisEnhancementError(f"Feature documentation not found: {self.feature_doc_path}")
        except IOError as e:
            raise ThesisEnhancementError(f"Failed to load feature documentation: {e}")
    
    def _load_principles(self) -> str:
        """Load trading system principles content"""
        try:
            with open(self.principles_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise ThesisEnhancementError(f"Principles document not found: {self.principles_path}")
        except IOError as e:
            raise ThesisEnhancementError(f"Failed to load principles: {e}")
    
    def _parse_feature_categories(self) -> List[FeatureCategory]:
        """Parse feature documentation to extract categories and features"""
        categories = []
        lines = self.feature_content.split('\n')
        
        # Find category sections (## with emoji)
        category_pattern = r'^## [🎯📊🎲🔄📈🔧📊✅📋🔄📝] (.+)$'
        current_category = None
        current_features = []
        content_start = 0
        
        for i, line in enumerate(lines):
            category_match = re.match(category_pattern, line)
            
            if category_match:
                # Save previous category if exists
                if current_category:
                    categories.append(FeatureCategory(
                        name=current_category,
                        section_title=line,
                        features=current_features,
                        content_start=content_start,
                        content_end=i,
                        raw_content='\n'.join(lines[content_start:i])
                    ))
                
                # Start new category
                current_category = category_match.group(1).strip()
                current_features = []
                content_start = i
            
            elif line.startswith('### ') and current_category:
                # Extract feature information
                feature_name = line.replace('### ', '').strip()
                feature_info = self._extract_feature_info(lines, i)
                current_features.append({
                    'name': feature_name,
                    'line_number': i,
                    **feature_info
                })
        
        # Add final category
        if current_category:
            categories.append(FeatureCategory(
                name=current_category,
                section_title=lines[content_start],
                features=current_features,
                content_start=content_start,
                content_end=len(lines),
                raw_content='\n'.join(lines[content_start:])
            ))
        
        return categories
    
    def _extract_feature_info(self, lines: List[str], start_idx: int) -> Dict[str, Any]:
        """Extract feature information from documentation lines"""
        feature_info = {}
        
        # Look for feature properties in the next few lines
        for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
            line = lines[i].strip()
            
            # Stop at next feature or section
            if line.startswith('### ') or line.startswith('## '):
                break
            
            # Extract key-value pairs
            if line.startswith('- **') and '**:' in line:
                key_match = re.match(r'- \*\*(.+?)\*\*:\s*(.+)', line)
                if key_match:
                    key = key_match.group(1).lower().replace(' ', '_')
                    value = key_match.group(2).strip()
                    feature_info[key] = value
        
        return feature_info
    
    def enhance_feature_category(self, category_name: str, 
                               preserve_existing: bool = True) -> EnhancementResult:
        """
        Enhance a specific feature category with thesis statements
        
        Args:
            category_name: Name of the category to enhance
            preserve_existing: Whether to preserve existing content
            
        Returns:
            EnhancementResult with enhancement details
        """
        try:
            # Find the category
            category = self._find_category(category_name)
            if not category:
                return EnhancementResult(
                    success=False,
                    category_name=category_name,
                    enhanced_content="",
                    enhancements_applied=[],
                    validation_results=[],
                    warnings=[],
                    error_message=f"Category not found: {category_name}"
                )
            
            # Create backup before enhancement
            backup_result = self.protection_system.create_backup(
                str(self.feature_doc_path),
                f"Pre-enhancement backup for {category_name}"
            )
            
            if not backup_result.success:
                return EnhancementResult(
                    success=False,
                    category_name=category_name,
                    enhanced_content="",
                    enhancements_applied=[],
                    validation_results=[],
                    warnings=[],
                    error_message=f"Backup creation failed: {backup_result.error_message}"
                )
            
            # Generate enhancements for each feature in category
            enhanced_features = []
            enhancements_applied = []
            warnings = []
            
            for feature in category.features:
                try:
                    enhancement = self.rationale_generator.generate_complete_enhancement(
                        feature['name'],
                        category_name,
                        feature
                    )
                    
                    enhanced_feature_content = self._format_enhanced_feature(
                        feature, enhancement, preserve_existing
                    )
                    
                    enhanced_features.append(enhanced_feature_content)
                    enhancements_applied.append(f"Enhanced {feature['name']} with thesis statement")
                    
                except Exception as e:
                    warnings.append(f"Failed to enhance {feature['name']}: {str(e)}")
                    # Keep original feature content
                    enhanced_features.append(self._format_original_feature(feature))
            
            # Reconstruct category content
            enhanced_content = self._reconstruct_category_content(
                category, enhanced_features
            )
            
            # Validate enhancement
            validation = self.validate_enhancement(enhanced_content, category_name)
            validation_results = []
            
            if validation.is_valid:
                validation_results.append(f"Enhancement validation passed (score: {validation.alignment_score:.2f})")
            else:
                validation_results.append(f"Enhancement validation failed: {validation.error_message}")
                validation_results.extend(validation.principle_violations)
            
            return EnhancementResult(
                success=True,
                category_name=category_name,
                enhanced_content=enhanced_content,
                enhancements_applied=enhancements_applied,
                validation_results=validation_results,
                warnings=warnings
            )
            
        except Exception as e:
            return EnhancementResult(
                success=False,
                category_name=category_name,
                enhanced_content="",
                enhancements_applied=[],
                validation_results=[],
                warnings=[],
                error_message=f"Enhancement failed: {str(e)}"
            )
    
    def _find_category(self, category_name: str) -> Optional[FeatureCategory]:
        """Find category by name"""
        for category in self.feature_categories:
            if category.name.lower() == category_name.lower():
                return category
        return None
    
    def _format_enhanced_feature(self, feature: Dict[str, Any], 
                                enhancement: FeatureEnhancement,
                                preserve_existing: bool) -> str:
        """Format enhanced feature with thesis statement"""
        lines = []
        
        # Feature title
        lines.append(f"### {feature['name']}")
        
        # Preserve existing properties if requested
        if preserve_existing:
            for key, value in feature.items():
                if key not in ['name', 'line_number'] and value:
                    formatted_key = key.replace('_', ' ').title()
                    lines.append(f"- **{formatted_key}**: {value}")
        
        # Add thesis statement section
        lines.append("")
        lines.append("#### Economic Thesis")
        lines.append(f"**Hypothesis**: {enhancement.thesis_statement.hypothesis}")
        lines.append("")
        lines.append(f"**Economic Basis**: {enhancement.thesis_statement.economic_basis}")
        lines.append("")
        lines.append(f"**Market Microstructure**: {enhancement.thesis_statement.market_microstructure}")
        lines.append("")
        lines.append(f"**Expected Behavior**: {enhancement.thesis_statement.expected_behavior}")
        
        # Add failure modes
        if enhancement.thesis_statement.failure_modes:
            lines.append("")
            lines.append("**Failure Modes**:")
            for mode in enhancement.thesis_statement.failure_modes:
                lines.append(f"- {mode}")
        
        # Add supply/demand analysis
        lines.append("")
        lines.append("#### Supply/Demand Analysis")
        lines.append("**Supply Factors**:")
        for factor in enhancement.economic_rationale.supply_factors:
            lines.append(f"- {factor}")
        
        lines.append("")
        lines.append("**Demand Factors**:")
        for factor in enhancement.economic_rationale.demand_factors:
            lines.append(f"- {factor}")
        
        lines.append("")
        lines.append(f"**Market Inefficiency**: {enhancement.economic_rationale.market_inefficiency}")
        
        # Add chart explainability
        lines.append("")
        lines.append("#### Chart Explainability")
        lines.append(f"**Visual Description**: {enhancement.chart_explanation.visual_description}")
        
        if enhancement.chart_explanation.example_scenarios:
            lines.append("")
            lines.append("**Example Scenarios**:")
            for scenario in enhancement.chart_explanation.example_scenarios:
                lines.append(f"- {scenario}")
        
        # Add validation criteria
        if enhancement.validation_criteria:
            lines.append("")
            lines.append("#### Validation Criteria")
            for criterion in enhancement.validation_criteria:
                lines.append(f"- {criterion}")
        
        return '\n'.join(lines)
    
    def _format_original_feature(self, feature: Dict[str, Any]) -> str:
        """Format original feature without enhancements"""
        lines = []
        lines.append(f"### {feature['name']}")
        
        for key, value in feature.items():
            if key not in ['name', 'line_number'] and value:
                formatted_key = key.replace('_', ' ').title()
                lines.append(f"- **{formatted_key}**: {value}")
        
        return '\n'.join(lines)
    
    def _reconstruct_category_content(self, category: FeatureCategory, 
                                    enhanced_features: List[str]) -> str:
        """Reconstruct category content with enhanced features"""
        lines = []
        
        # Add category header
        lines.append(category.section_title)
        lines.append("")
        
        # Add enhanced features
        for i, feature_content in enumerate(enhanced_features):
            lines.append(feature_content)
            if i < len(enhanced_features) - 1:
                lines.append("")
                lines.append("---")
                lines.append("")
        
        return '\n'.join(lines)
    
    def validate_enhancement(self, enhanced_content: str, 
                           category_name: str) -> ValidationResult:
        """
        Validate enhancement against trading system principles
        
        Args:
            enhanced_content: Enhanced category content
            category_name: Name of the category
            
        Returns:
            ValidationResult with validation details
        """
        try:
            principle_violations = []
            missing_elements = []
            recommendations = []
            alignment_score = 0.0
            
            # Check for required thesis elements
            required_elements = [
                ("Economic Thesis", "#### Economic Thesis"),
                ("Hypothesis", "**Hypothesis**:"),
                ("Economic Basis", "**Economic Basis**:"),
                ("Market Microstructure", "**Market Microstructure**:"),
                ("Supply/Demand Analysis", "#### Supply/Demand Analysis"),
                ("Chart Explainability", "#### Chart Explainability")
            ]
            
            elements_found = 0
            for element_name, element_marker in required_elements:
                if element_marker in enhanced_content:
                    elements_found += 1
                else:
                    missing_elements.append(f"Missing {element_name} section")
            
            alignment_score += (elements_found / len(required_elements)) * 0.4
            
            # Check alignment with trading principles
            principle_checks = [
                ("thesis-first", "hypothesis", "Thesis-first development principle"),
                ("supply", "supply", "Supply/demand focus principle"),
                ("demand", "demand", "Supply/demand focus principle"),
                ("chart", "chart", "Simplicity & explainability principle"),
                ("economic", "economic", "Economic rationale requirement")
            ]
            
            principles_found = 0
            for check_term, search_term, principle_name in principle_checks:
                if search_term.lower() in enhanced_content.lower():
                    principles_found += 1
                else:
                    principle_violations.append(f"Weak alignment with {principle_name}")
            
            alignment_score += (principles_found / len(principle_checks)) * 0.3
            
            # Check for economic logic
            economic_terms = ["supply", "demand", "imbalance", "inefficiency", "microstructure"]
            economic_found = sum(1 for term in economic_terms if term in enhanced_content.lower())
            alignment_score += min(economic_found / len(economic_terms), 1.0) * 0.3
            
            # Generate recommendations
            if alignment_score < 0.7:
                recommendations.append("Consider adding more economic rationale")
            if not any("chart" in content.lower() for content in [enhanced_content]):
                recommendations.append("Add chart explainability section")
            if len(missing_elements) > 0:
                recommendations.append("Complete all required thesis sections")
            
            is_valid = alignment_score >= 0.6 and len(missing_elements) < 3
            
            return ValidationResult(
                is_valid=is_valid,
                alignment_score=alignment_score,
                principle_violations=principle_violations,
                missing_elements=missing_elements,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                alignment_score=0.0,
                principle_violations=[],
                missing_elements=[],
                recommendations=[],
                error_message=f"Validation failed: {str(e)}"
            )
    
    def apply_enhancement(self, enhancement_result: EnhancementResult) -> bool:
        """
        Apply enhancement to the actual documentation file
        
        Args:
            enhancement_result: Result from enhance_feature_category
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not enhancement_result.success:
                raise ThesisEnhancementError("Cannot apply failed enhancement")
            
            # Create rollback point
            rollback_point = self.protection_system.create_rollback_point(
                str(self.feature_doc_path),
                f"Before applying {enhancement_result.category_name} enhancement"
            )
            
            # Find category in current content
            category = self._find_category(enhancement_result.category_name)
            if not category:
                raise ThesisEnhancementError(f"Category not found: {enhancement_result.category_name}")
            
            # Replace category content
            lines = self.feature_content.split('\n')
            new_lines = (
                lines[:category.content_start] +
                enhancement_result.enhanced_content.split('\n') +
                lines[category.content_end:]
            )
            
            new_content = '\n'.join(new_lines)
            
            # Validate changes
            change_validation = self.protection_system.validate_changes(
                self.feature_content, new_content, str(self.feature_doc_path)
            )
            
            if not change_validation.is_valid:
                raise ThesisEnhancementError(f"Change validation failed: {change_validation.error_message}")
            
            # Apply changes
            with open(self.feature_doc_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Update internal content
            self.feature_content = new_content
            self.feature_categories = self._parse_feature_categories()
            
            # Record enhancement
            self.enhancement_history.append({
                'timestamp': datetime.now().isoformat(),
                'category': enhancement_result.category_name,
                'enhancements': enhancement_result.enhancements_applied,
                'rollback_point': rollback_point.id
            })
            
            return True
            
        except Exception as e:
            print(f"Failed to apply enhancement: {e}")
            return False
    
    def list_categories(self) -> List[str]:
        """List all available feature categories"""
        return [category.name for category in self.feature_categories]
    
    def get_category_info(self, category_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific category"""
        category = self._find_category(category_name)
        if not category:
            return None
        
        return {
            'name': category.name,
            'section_title': category.section_title,
            'feature_count': len(category.features),
            'features': [f['name'] for f in category.features],
            'has_enhancements': self._check_has_enhancements(category)
        }
    
    def _check_has_enhancements(self, category: FeatureCategory) -> bool:
        """Check if category already has thesis enhancements"""
        return "#### Economic Thesis" in category.raw_content
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get overall enhancement status"""
        total_categories = len(self.feature_categories)
        enhanced_categories = sum(1 for cat in self.feature_categories if self._check_has_enhancements(cat))
        
        return {
            'total_categories': total_categories,
            'enhanced_categories': enhanced_categories,
            'enhancement_percentage': (enhanced_categories / total_categories * 100) if total_categories > 0 else 0,
            'categories': [
                {
                    'name': cat.name,
                    'enhanced': self._check_has_enhancements(cat),
                    'feature_count': len(cat.features)
                }
                for cat in self.feature_categories
            ],
            'enhancement_history': self.enhancement_history
        }