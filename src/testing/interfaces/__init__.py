"""
Base interfaces for the Feature Test Coverage System
"""

from .parser_interface import FeatureParserInterface
from .generator_interface import TestGeneratorInterface
from .executor_interface import TestExecutorInterface
from .validator_interface import ValidationInterface
from .reporter_interface import ReporterInterface

__all__ = [
    'FeatureParserInterface',
    'TestGeneratorInterface', 
    'TestExecutorInterface',
    'ValidationInterface',
    'ReporterInterface'
]