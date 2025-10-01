"""
Test case data model for the feature test coverage system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum


class TestType(Enum):
    """Enumeration of different test types supported by the system."""
    ECONOMIC_HYPOTHESIS = "economic_hypothesis"
    PERFORMANCE = "performance"
    FAILURE_MODE = "failure_mode"
    IMPLEMENTATION = "implementation"
    INTERACTION = "interaction"
    REGIME_DEPENDENCY = "regime_dependency"


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestCase:
    """
    Represents a single test case for validating feature behavior.
    
    This class encapsulates all information needed to execute a specific test
    and validate the results against expected behavior.
    """
    
    # Basic identification
    test_id: str
    feature_name: str
    test_type: TestType
    description: str
    
    # Test execution details
    test_function: Optional[Callable] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    test_data_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Expected results and validation
    expected_result: Any = None
    validation_criteria: Dict[str, Any] = field(default_factory=dict)
    tolerance: float = 0.01  # Default tolerance for numerical comparisons
    
    # Test context
    regime_context: Optional[str] = None
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    data_quality_requirements: List[str] = field(default_factory=list)
    
    # Test metadata
    priority: TestPriority = TestPriority.MEDIUM
    estimated_duration: float = 1.0  # Estimated execution time in seconds
    dependencies: List[str] = field(default_factory=list)  # Other tests this depends on
    tags: List[str] = field(default_factory=list)
    
    # Documentation
    rationale: str = ""  # Why this test is important
    failure_impact: str = ""  # What happens if this test fails
    
    def __post_init__(self):
        """Validate the test case after initialization."""
        if not self.test_id:
            raise ValueError("Test ID cannot be empty")
        
        if not self.feature_name:
            raise ValueError("Feature name cannot be empty")
        
        if not self.description:
            raise ValueError("Test description cannot be empty")
        
        # Generate test ID if not provided
        if self.test_id == "auto":
            self.test_id = f"{self.feature_name}_{self.test_type.value}_{hash(self.description) % 10000}"
    
    def is_executable(self) -> bool:
        """
        Check if this test case has all required components for execution.
        """
        return (
            self.test_function is not None and
            bool(self.validation_criteria)
        )
    
    def requires_market_data(self) -> bool:
        """
        Check if this test case requires historical market data.
        """
        return bool(self.test_data_requirements) or self.regime_context is not None
    
    def get_execution_weight(self) -> float:
        """
        Calculate execution weight for scheduling and resource allocation.
        """
        weight = self.estimated_duration
        
        # Adjust based on priority
        priority_multipliers = {
            TestPriority.CRITICAL: 3.0,
            TestPriority.HIGH: 2.0,
            TestPriority.MEDIUM: 1.0,
            TestPriority.LOW: 0.5
        }
        
        weight *= priority_multipliers.get(self.priority, 1.0)
        
        # Adjust based on complexity
        if self.regime_context:
            weight *= 1.5
        
        if len(self.dependencies) > 0:
            weight *= 1.2
        
        return weight
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert test case to dictionary for serialization.
        """
        return {
            'test_id': self.test_id,
            'feature_name': self.feature_name,
            'test_type': self.test_type.value,
            'description': self.description,
            'test_parameters': self.test_parameters,
            'expected_result': self.expected_result,
            'validation_criteria': self.validation_criteria,
            'regime_context': self.regime_context,
            'priority': self.priority.value,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'rationale': self.rationale
        }