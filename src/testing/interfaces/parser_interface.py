"""
Interface for feature template parsers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path

from ..models.feature_spec import FeatureSpec


class FeatureParserInterface(ABC):
    """
    Abstract interface for parsing feature specifications from documentation.
    
    This interface defines the contract for components that can extract
    feature information from various documentation formats.
    """
    
    @abstractmethod
    def parse_template(self, template_path: Path) -> List[FeatureSpec]:
        """
        Parse a feature template file and extract all feature specifications.
        
        Args:
            template_path: Path to the feature template file
            
        Returns:
            List of FeatureSpec objects representing all features found
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ParseError: If template format is invalid
        """
        pass
    
    @abstractmethod
    def extract_feature_sections(self, content: str) -> Dict[str, str]:
        """
        Extract individual feature sections from template content.
        
        Args:
            content: Raw template content as string
            
        Returns:
            Dictionary mapping feature names to their section content
        """
        pass
    
    @abstractmethod
    def parse_feature_details(self, section_content: str, feature_name: str) -> FeatureSpec:
        """
        Parse detailed feature information from a section.
        
        Args:
            section_content: Raw content of a feature section
            feature_name: Name of the feature being parsed
            
        Returns:
            FeatureSpec object with parsed information
            
        Raises:
            ParseError: If section format is invalid
        """
        pass
    
    @abstractmethod
    def validate_template_format(self, template_path: Path) -> bool:
        """
        Validate that a template file follows the expected format.
        
        Args:
            template_path: Path to template file to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        pass
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported template formats.
        
        Returns:
            List of supported file extensions (e.g., ['.md', '.txt'])
        """
        return ['.md']
    
    def get_parser_metadata(self) -> Dict[str, str]:
        """
        Get metadata about this parser implementation.
        
        Returns:
            Dictionary with parser name, version, and capabilities
        """
        return {
            'name': self.__class__.__name__,
            'version': '1.0.0',
            'supported_formats': ','.join(self.get_supported_formats())
        }