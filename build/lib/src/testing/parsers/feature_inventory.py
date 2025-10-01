"""
Feature Inventory Generation for the test coverage system.

This module provides functionality to generate comprehensive feature inventories
with categorization, dependency detection, and test requirements mapping.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging

from ..models.feature_spec import FeatureSpec


@dataclass
class FeatureDependency:
    """Represents a dependency relationship between features."""
    source_feature: str
    target_feature: str
    dependency_type: str  # 'input', 'enhancement', 'conflict', 'synergy'
    strength: str  # 'strong', 'medium', 'weak'
    description: str = ""


@dataclass
class FeatureCategory:
    """Represents a category of features with metadata."""
    name: str
    emoji: str
    description: str
    features: List[str] = field(default_factory=list)
    priority: str = "medium"  # high, medium, low
    test_complexity: str = "standard"  # simple, standard, complex


@dataclass
class FeatureInventory:
    """Comprehensive inventory of all features with categorization and dependencies."""
    categories: Dict[str, FeatureCategory] = field(default_factory=dict)
    features: Dict[str, FeatureSpec] = field(default_factory=dict)
    dependencies: List[FeatureDependency] = field(default_factory=list)
    test_requirements: Dict[str, List[str]] = field(default_factory=dict)
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_features_by_category(self, category_name: str) -> List[FeatureSpec]:
        """Get all features in a specific category."""
        if category_name not in self.categories:
            return []
        
        feature_names = self.categories[category_name].features
        return [self.features[name] for name in feature_names if name in self.features]
    
    def get_critical_features(self) -> List[FeatureSpec]:
        """Get all features marked as critical (Tier 1 or high priority)."""
        return [feature for feature in self.features.values() if feature.is_critical_feature()]
    
    def get_dependency_chain(self, feature_name: str) -> List[str]:
        """Get the dependency chain for a specific feature."""
        chain = []
        visited = set()
        
        def _build_chain(name: str):
            if name in visited:
                return
            visited.add(name)
            
            # Find dependencies
            for dep in self.dependencies:
                if dep.target_feature == name:
                    _build_chain(dep.source_feature)
                    if dep.source_feature not in chain:
                        chain.append(dep.source_feature)
        
        _build_chain(feature_name)
        return chain
    
    def get_test_coverage_summary(self) -> Dict[str, Any]:
        """Generate a summary of test coverage requirements."""
        total_features = len(self.features)
        critical_features = len(self.get_critical_features())
        
        test_type_counts = {}
        for requirements in self.test_requirements.values():
            for req in requirements:
                test_type_counts[req] = test_type_counts.get(req, 0) + 1
        
        return {
            'total_features': total_features,
            'critical_features': critical_features,
            'categories': len(self.categories),
            'dependencies': len(self.dependencies),
            'test_type_distribution': test_type_counts,
            'coverage_completeness': self.validation_summary.get('completeness_score', 0)
        }


class FeatureInventoryGenerator:
    """
    Generates comprehensive feature inventories with categorization and dependency analysis.
    
    This class implements the functionality required by task 2.3:
    - Feature categorization logic based on template sections
    - Feature dependency detection and mapping
    - Comprehensive feature inventory with test requirements
    - Validation for completeness and consistency
    """
    
    def __init__(self, parser=None):
        """Initialize the inventory generator."""
        self.logger = logging.getLogger(__name__)
        self.parser = parser
        
        # Category mapping based on template structure
        self.category_patterns = {
            "Core Signal Features": {
                "priority": "high",
                "test_complexity": "complex",
                "description": "Primary directional signals and quantile-based features"
            },
            "Risk & Volatility Features": {
                "priority": "high", 
                "test_complexity": "standard",
                "description": "Volatility measures and risk assessment features"
            },
            "🎲 Position Sizing Features": {
                "priority": "high",
                "test_complexity": "complex", 
                "description": "Kelly sizing and position management features"
            },
            "Regime & Market Features": {
                "priority": "medium",
                "test_complexity": "standard",
                "description": "Market regime detection and classification features"
            },
            "Interaction & Enhancement Features": {
                "priority": "medium",
                "test_complexity": "complex",
                "description": "Feature interaction and enhancement mechanisms"
            },
            "Performance & Validation Features": {
                "priority": "low",
                "test_complexity": "simple",
                "description": "Performance monitoring and validation features"
            }
        }
        
        # Dependency detection patterns
        self.dependency_keywords = {
            'input': ['based on', 'uses', 'calculated from', 'derived from'],
            'enhancement': ['enhanced by', 'amplified by', 'works with', 'synergistic'],
            'conflict': ['conflicts with', 'contradicts', 'opposed to', 'negative interaction'],
            'synergy': ['synergy', 'positive interaction', 'reinforced by', 'combined with']
        }
    
    def generate_inventory(self, template_path: Path) -> FeatureInventory:
        """
        Generate a comprehensive feature inventory from the template.
        
        Args:
            template_path: Path to the Feature Knowledge Template
            
        Returns:
            FeatureInventory with complete categorization and dependencies
            
        Raises:
            ValueError: If template cannot be parsed or is invalid
        """
        self.logger.info(f"Generating feature inventory from: {template_path}")
        
        # Parse features from template
        try:
            if self.parser:
                features = self.parser.parse_template(template_path)
            else:
                # Import here to avoid circular import
                from .feature_template_parser import FeatureTemplateParser
                parser = FeatureTemplateParser()
                features = parser.parse_template(template_path)
            
            self.logger.info(f"Parsed {len(features)} features from template")
        except Exception as e:
            raise ValueError(f"Failed to parse template: {e}")
        
        if not features:
            raise ValueError("No features found in template")
        
        # Create inventory
        inventory = FeatureInventory()
        
        # Build feature mapping
        for feature in features:
            inventory.features[feature.name] = feature
        
        # Generate categories
        inventory.categories = self._generate_categories(features, template_path)
        
        # Detect dependencies
        inventory.dependencies = self._detect_dependencies(features, template_path)
        
        # Generate test requirements
        inventory.test_requirements = self._generate_test_requirements(features)
        
        # Validate inventory
        inventory.validation_summary = self._validate_inventory(inventory)
        
        self.logger.info(f"Generated inventory with {len(inventory.features)} features, "
                        f"{len(inventory.categories)} categories, "
                        f"{len(inventory.dependencies)} dependencies")
        
        return inventory
    
    def _generate_categories(self, features: List[FeatureSpec], template_path: Path) -> Dict[str, FeatureCategory]:
        """Generate feature categories based on template sections."""
        self.logger.debug("Generating feature categories")
        
        categories = {}
        
        # Read template content to extract category structure
        try:
            content = template_path.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read template for category extraction: {e}")
            return self._generate_default_categories(features)
        
        # Extract category sections
        category_sections = self._extract_category_sections(content)
        
        for section_name, section_content in category_sections.items():
            # Clean category name (remove emoji and formatting)
            clean_name = re.sub(r'^[^\w\s]+\s*', '', section_name).strip()
            
            # Extract emoji
            emoji_match = re.match(r'^([^\w\s]+)', section_name)
            emoji = emoji_match.group(1) if emoji_match else "📋"
            
            # Get category metadata
            metadata = self.category_patterns.get(section_name, {
                "priority": "medium",
                "test_complexity": "standard", 
                "description": f"Features in {clean_name} category"
            })
            
            # Find features in this category
            category_features = [f.name for f in features if f.category == clean_name]
            
            categories[clean_name] = FeatureCategory(
                name=clean_name,
                emoji=emoji,
                description=metadata["description"],
                features=category_features,
                priority=metadata["priority"],
                test_complexity=metadata["test_complexity"]
            )
        
        # Handle uncategorized features
        categorized_features = set()
        for category in categories.values():
            categorized_features.update(category.features)
        
        uncategorized = [f.name for f in features if f.name not in categorized_features]
        if uncategorized:
            categories["Uncategorized"] = FeatureCategory(
                name="Uncategorized",
                emoji="❓",
                description="Features without clear category assignment",
                features=uncategorized,
                priority="low",
                test_complexity="simple"
            )
        
        return categories
    
    def _extract_category_sections(self, content: str) -> Dict[str, str]:
        """Extract category sections from template content."""
        sections = {}
        
        # Find all category headers (## level)
        category_pattern = r'^##\s+(.+)$'
        lines = content.split('\n')
        
        current_category = None
        current_content = []
        
        for line in lines:
            category_match = re.match(category_pattern, line)
            
            if category_match:
                # Save previous category
                if current_category:
                    sections[current_category] = '\n'.join(current_content)
                
                # Start new category
                current_category = category_match.group(1).strip()
                current_content = []
            elif current_category:
                current_content.append(line)
        
        # Save last category
        if current_category:
            sections[current_category] = '\n'.join(current_content)
        
        return sections
    
    def _generate_default_categories(self, features: List[FeatureSpec]) -> Dict[str, FeatureCategory]:
        """Generate default categories when template parsing fails."""
        self.logger.warning("Using default category generation")
        
        # Group by existing category field
        category_groups = {}
        for feature in features:
            category = feature.category or "Uncategorized"
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(feature.name)
        
        categories = {}
        for category_name, feature_names in category_groups.items():
            categories[category_name] = FeatureCategory(
                name=category_name,
                emoji="📋",
                description=f"Features in {category_name} category",
                features=feature_names,
                priority="medium",
                test_complexity="standard"
            )
        
        return categories
    
    def _detect_dependencies(self, features: List[FeatureSpec], template_path: Path) -> List[FeatureDependency]:
        """Detect feature dependencies from template content and feature specifications."""
        self.logger.debug("Detecting feature dependencies")
        
        dependencies = []
        
        # Read template content for dependency analysis
        try:
            content = template_path.read_text(encoding='utf-8')
        except Exception as e:
            self.logger.error(f"Failed to read template for dependency detection: {e}")
            return self._detect_basic_dependencies(features)
        
        # Create feature name mapping for case-insensitive matching
        feature_names = {f.name.lower(): f.name for f in features}
        
        # Analyze each feature for dependencies
        for feature in features:
            feature_deps = self._analyze_feature_dependencies(
                feature, content, feature_names
            )
            dependencies.extend(feature_deps)
        
        # Add interaction-based dependencies
        interaction_deps = self._detect_interaction_dependencies(features)
        dependencies.extend(interaction_deps)
        
        # Remove duplicates
        unique_deps = []
        seen = set()
        for dep in dependencies:
            key = (dep.source_feature, dep.target_feature, dep.dependency_type)
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        self.logger.debug(f"Detected {len(unique_deps)} unique dependencies")
        return unique_deps
    
    def _analyze_feature_dependencies(self, feature: FeatureSpec, content: str, 
                                    feature_names: Dict[str, str]) -> List[FeatureDependency]:
        """Analyze a single feature for dependencies."""
        dependencies = []
        
        # Get feature section from content
        feature_section = self._extract_feature_section(feature.name, content)
        if not feature_section:
            return dependencies
        
        # Search for dependency keywords
        for dep_type, keywords in self.dependency_keywords.items():
            for keyword in keywords:
                if keyword.lower() in feature_section.lower():
                    # Find mentioned feature names
                    mentioned_features = self._find_mentioned_features(
                        feature_section, feature_names, feature.name
                    )
                    
                    for mentioned_feature in mentioned_features:
                        dependencies.append(FeatureDependency(
                            source_feature=mentioned_feature,
                            target_feature=feature.name,
                            dependency_type=dep_type,
                            strength=self._assess_dependency_strength(
                                feature_section, keyword, mentioned_feature
                            ),
                            description=f"Detected via keyword '{keyword}'"
                        ))
        
        return dependencies
    
    def _extract_feature_section(self, feature_name: str, content: str) -> str:
        """Extract the section content for a specific feature."""
        # Find feature header
        pattern = rf'^###\s+{re.escape(feature_name)}.*?(?=^###|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(0)
        
        # Try partial matching
        pattern = rf'^###\s+.*{re.escape(feature_name)}.*?(?=^###|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        return match.group(0) if match else ""
    
    def _find_mentioned_features(self, text: str, feature_names: Dict[str, str], 
                               exclude_feature: str) -> List[str]:
        """Find feature names mentioned in text."""
        mentioned = []
        
        for name_lower, name_actual in feature_names.items():
            if name_actual.lower() == exclude_feature.lower():
                continue
                
            # Look for exact matches and common variations
            patterns = [
                rf'\b{re.escape(name_lower)}\b',
                rf'\b{re.escape(name_actual)}\b',
                rf'`{re.escape(name_actual)}`',
                rf'\${re.escape(name_actual)}'
            ]
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    mentioned.append(name_actual)
                    break
        
        return mentioned
    
    def _assess_dependency_strength(self, text: str, keyword: str, feature_name: str) -> str:
        """Assess the strength of a dependency based on context."""
        # Count mentions and context
        mentions = len(re.findall(rf'\b{re.escape(feature_name)}\b', text, re.IGNORECASE))
        
        # Strong indicators
        strong_indicators = ['critical', 'essential', 'primary', 'core', 'fundamental']
        if any(indicator in text.lower() for indicator in strong_indicators):
            return 'strong'
        
        # Multiple mentions suggest stronger dependency
        if mentions >= 3:
            return 'strong'
        elif mentions >= 2:
            return 'medium'
        else:
            return 'weak'
    
    def _detect_interaction_dependencies(self, features: List[FeatureSpec]) -> List[FeatureDependency]:
        """Detect dependencies based on feature interaction specifications."""
        dependencies = []
        
        for feature in features:
            # Process interactions list
            for interaction in feature.interactions:
                # Try to match interaction to actual feature names
                for other_feature in features:
                    if (other_feature.name != feature.name and 
                        other_feature.name.lower() in interaction.lower()):
                        
                        dependencies.append(FeatureDependency(
                            source_feature=other_feature.name,
                            target_feature=feature.name,
                            dependency_type='interaction',
                            strength='medium',
                            description=f"Listed in interactions: {interaction}"
                        ))
            
            # Process synergies
            for synergy in feature.synergies:
                for other_feature in features:
                    if (other_feature.name != feature.name and 
                        other_feature.name.lower() in synergy.lower()):
                        
                        dependencies.append(FeatureDependency(
                            source_feature=other_feature.name,
                            target_feature=feature.name,
                            dependency_type='synergy',
                            strength='strong',
                            description=f"Listed in synergies: {synergy}"
                        ))
            
            # Process conflicts
            for conflict in feature.conflicts:
                for other_feature in features:
                    if (other_feature.name != feature.name and 
                        other_feature.name.lower() in conflict.lower()):
                        
                        dependencies.append(FeatureDependency(
                            source_feature=other_feature.name,
                            target_feature=feature.name,
                            dependency_type='conflict',
                            strength='medium',
                            description=f"Listed in conflicts: {conflict}"
                        ))
        
        return dependencies
    
    def _detect_basic_dependencies(self, features: List[FeatureSpec]) -> List[FeatureDependency]:
        """Fallback dependency detection using only feature specifications."""
        dependencies = []
        
        for feature in features:
            # Use implementation field to detect dependencies
            if feature.implementation:
                for other_feature in features:
                    if (other_feature.name != feature.name and 
                        other_feature.name.lower() in feature.implementation.lower()):
                        
                        dependencies.append(FeatureDependency(
                            source_feature=other_feature.name,
                            target_feature=feature.name,
                            dependency_type='input',
                            strength='medium',
                            description="Detected in implementation description"
                        ))
        
        return dependencies
    
    def _generate_test_requirements(self, features: List[FeatureSpec]) -> Dict[str, List[str]]:
        """Generate test requirements for each feature."""
        self.logger.debug("Generating test requirements")
        
        requirements = {}
        
        for feature in features:
            feature_requirements = feature.get_test_requirements()
            
            # Add category-specific requirements
            if "Core Signal" in feature.category:
                feature_requirements.extend(['regime_dependency', 'performance'])
            
            if "Risk & Volatility" in feature.category:
                feature_requirements.extend(['empirical_ranges', 'failure_modes'])
            
            if "Position Sizing" in feature.category:
                feature_requirements.extend(['risk_adjustment', 'drawdown_control'])
            
            # Add complexity-based requirements
            if feature.test_complexity == "complex":
                feature_requirements.extend(['interaction', 'stress_test'])
            
            # Remove duplicates and sort
            requirements[feature.name] = sorted(list(set(feature_requirements)))
        
        return requirements
    
    def _validate_inventory(self, inventory: FeatureInventory) -> Dict[str, Any]:
        """Validate inventory completeness and consistency."""
        self.logger.debug("Validating feature inventory")
        
        validation_results = {
            'completeness_score': 0,
            'consistency_issues': [],
            'missing_data': [],
            'warnings': []
        }
        
        total_features = len(inventory.features)
        if total_features == 0:
            validation_results['consistency_issues'].append("No features found in inventory")
            return validation_results
        
        # Check completeness
        complete_features = 0
        for feature in inventory.features.values():
            score = 0
            
            # Required fields
            if feature.name: score += 1
            if feature.category: score += 1
            if feature.implementation: score += 1
            
            # Important fields
            if feature.economic_hypothesis: score += 1
            if feature.performance_characteristics: score += 1
            if feature.failure_modes: score += 1
            
            # Bonus for comprehensive features
            if feature.regime_dependencies: score += 0.5
            if feature.interactions: score += 0.5
            
            if score >= 6:  # Threshold for "complete"
                complete_features += 1
        
        validation_results['completeness_score'] = (complete_features / total_features) * 100
        
        # Check consistency
        self._validate_categories(inventory, validation_results)
        self._validate_dependencies(inventory, validation_results)
        self._validate_test_requirements(inventory, validation_results)
        
        return validation_results
    
    def _validate_categories(self, inventory: FeatureInventory, results: Dict[str, Any]):
        """Validate category consistency."""
        # Check that all features are categorized
        categorized_features = set()
        for category in inventory.categories.values():
            categorized_features.update(category.features)
        
        uncategorized = set(inventory.features.keys()) - categorized_features
        if uncategorized:
            results['missing_data'].append(f"Uncategorized features: {list(uncategorized)}")
        
        # Check for empty categories
        empty_categories = [name for name, cat in inventory.categories.items() if not cat.features]
        if empty_categories:
            results['warnings'].append(f"Empty categories: {empty_categories}")
    
    def _validate_dependencies(self, inventory: FeatureInventory, results: Dict[str, Any]):
        """Validate dependency consistency."""
        feature_names = set(inventory.features.keys())
        
        for dep in inventory.dependencies:
            # Check that referenced features exist
            if dep.source_feature not in feature_names:
                results['consistency_issues'].append(
                    f"Dependency references unknown source feature: {dep.source_feature}"
                )
            
            if dep.target_feature not in feature_names:
                results['consistency_issues'].append(
                    f"Dependency references unknown target feature: {dep.target_feature}"
                )
            
            # Check for self-dependencies
            if dep.source_feature == dep.target_feature:
                results['consistency_issues'].append(
                    f"Self-dependency detected: {dep.source_feature}"
                )
    
    def _validate_test_requirements(self, inventory: FeatureInventory, results: Dict[str, Any]):
        """Validate test requirements consistency."""
        for feature_name, requirements in inventory.test_requirements.items():
            if feature_name not in inventory.features:
                results['consistency_issues'].append(
                    f"Test requirements for unknown feature: {feature_name}"
                )
            
            # Check for minimum requirements
            if not requirements:
                results['missing_data'].append(
                    f"No test requirements for feature: {feature_name}"
                )