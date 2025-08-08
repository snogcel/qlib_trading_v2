"""
Validation module for the feature test coverage system.

This module provides comprehensive validation and analysis capabilities
for test results, including pass/fail determination, confidence scoring,
and recommendation generation.
"""

from .result_validator import (
    ResultValidator,
    ValidationResult,
    ValidationCriteria
)
from .result_analyzer import ResultAnalyzer

__all__ = [
    'ResultValidator',
    'ValidationResult', 
    'ValidationCriteria',
    'ResultAnalyzer'
]