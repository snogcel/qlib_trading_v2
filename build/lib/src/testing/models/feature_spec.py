"""
Feature specification data model for the test coverage system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import re


@dataclass
class FeatureSpec:
    """
    Represents a feature specification parsed from the Feature Knowledge Template.
    
    This class captures all the essential information about a trading system feature
    needed to generate comprehensive test cases.
    """
    
    # Basic identification
    name: str
    category: str  # Core Signal, Risk & Volatility, etc.
    tier: str  # Tier 1, Tier 2, etc.
    
    # Implementation details
    implementation: str  # Location/formula description
    formula: Optional[str] = None  # Specific mathematical formula if available
    
    # Economic foundation
    economic_hypothesis: str = ""
    theoretical_basis: str = ""
    
    # Performance characteristics
    performance_characteristics: Dict[str, Any] = field(default_factory=dict)
    empirical_ranges: Dict[str, float] = field(default_factory=dict)
    expected_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Failure modes and edge cases
    failure_modes: List[str] = field(default_factory=list)
    edge_cases: List[str] = field(default_factory=list)
    
    # Market regime dependencies
    regime_dependencies: Dict[str, str] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Feature interactions
    interactions: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Data requirements
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    temporal_dependencies: List[str] = field(default_factory=list)
    
    # Validation metadata
    validation_priority: str = "medium"  # high, medium, low
    test_complexity: str = "standard"  # simple, standard, complex
    
    def __post_init__(self):
        """Validate and normalize the feature specification after initialization."""
        if not self.name:
            raise ValueError("Feature name cannot be empty")
        
        if not self.category:
            raise ValueError("Feature category cannot be empty")
        
        # Normalize tier information
        if not self.tier:
            self.tier = "Tier 2"  # Default tier
        
        # Set validation priority based on tier
        if self.tier == "Tier 1":
            self.validation_priority = "high"
        elif self.tier == "Tier 2":
            self.validation_priority = "medium"
        else:
            self.validation_priority = "low"
    
    def get_test_requirements(self) -> List[str]:
        """
        Return a list of test types required for this feature based on its characteristics.
        """
        requirements = ["implementation"]  # All features need implementation tests
        
        if self.economic_hypothesis:
            requirements.append("economic_hypothesis")
        
        if self.performance_characteristics:
            requirements.append("performance")
        
        if self.failure_modes:
            requirements.append("failure_modes")
        
        if self.regime_dependencies:
            requirements.append("regime_dependency")
        
        if self.interactions:
            requirements.append("interaction")
        
        return requirements
    
    def is_critical_feature(self) -> bool:
        """
        Determine if this is a critical feature that requires comprehensive testing.
        """
        return (
            self.tier == "Tier 1" or 
            self.validation_priority == "high" or
            "Core Signal" in self.category
        )
    
    def get_complexity_score(self) -> int:
        """
        Calculate a complexity score for test generation prioritization.
        """
        score = 1  # Base complexity
        
        score += len(self.failure_modes)
        score += len(self.interactions)
        score += len(self.regime_dependencies)
        
        if self.formula:
            score += 2  # Mathematical formulas add complexity
        
        if self.test_complexity == "complex":
            score *= 2
        
        return score
    
    @classmethod
    def validate_required_fields(cls, data: Dict[str, Any]) -> List[str]:
        """
        Validate that required fields are present and valid.
        
        Args:
            data: Dictionary of field values to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        required_fields = ['name', 'category']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Validate field types
        if 'name' in data and not isinstance(data['name'], str):
            errors.append("Field 'name' must be a string")
        
        if 'category' in data and not isinstance(data['category'], str):
            errors.append("Field 'category' must be a string")
        
        # Validate tier if provided
        valid_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
        if 'tier' in data and data['tier'] and data['tier'] not in valid_tiers:
            errors.append(f"Field 'tier' must be one of {valid_tiers}")
        
        # Validate priority if provided
        valid_priorities = ['high', 'medium', 'low']
        if 'validation_priority' in data and data['validation_priority'] not in valid_priorities:
            errors.append(f"Field 'validation_priority' must be one of {valid_priorities}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the FeatureSpec to a dictionary representation.
        
        Returns:
            Dictionary representation of the feature specification
        """
        return {
            'name': self.name,
            'category': self.category,
            'tier': self.tier,
            'implementation': self.implementation,
            'formula': self.formula,
            'economic_hypothesis': self.economic_hypothesis,
            'theoretical_basis': self.theoretical_basis,
            'performance_characteristics': self.performance_characteristics,
            'empirical_ranges': self.empirical_ranges,
            'expected_metrics': self.expected_metrics,
            'failure_modes': self.failure_modes,
            'edge_cases': self.edge_cases,
            'regime_dependencies': self.regime_dependencies,
            'regime_performance': self.regime_performance,
            'interactions': self.interactions,
            'synergies': self.synergies,
            'conflicts': self.conflicts,
            'data_requirements': self.data_requirements,
            'temporal_dependencies': self.temporal_dependencies,
            'validation_priority': self.validation_priority,
            'test_complexity': self.test_complexity
        }