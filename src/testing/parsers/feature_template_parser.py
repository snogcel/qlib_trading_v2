"""
Feature Template Parser for the test coverage system.

This module provides the FeatureTemplateParser class that implements the core
parsing functionality required for extracting feature specifications from the
Feature Knowledge Template markdown file.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..models.feature_spec import FeatureSpec
from ..interfaces.parser_interface import FeatureParserInterface
from .markdown_parser import MarkdownFeatureParser, ParseError


class FeatureTemplateParser(FeatureParserInterface):
    """
    Main parser class for processing the Feature Knowledge Template.
    
    This class implements the core parsing methods required by task 2.2:
    - parse_template: Process entire template file
    - extract_feature_sections: Identify feature boundaries
    - parse_feature_details: Extract structured data from sections
    
    It builds on the MarkdownFeatureParser for the actual parsing logic
    while providing the specific interface required by the test coverage system.
    """
    
    def __init__(self):
        """Initialize the parser with logging and underlying markdown parser."""
        self.logger = logging.getLogger(__name__)
        self._markdown_parser = MarkdownFeatureParser()
        
        # Track parsing statistics
        self._parsing_stats = {
            'total_features': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'warnings': 0
        }
    
    def parse_template(self, template_path: Path) -> List[FeatureSpec]:
        """
        Parse the entire Feature Knowledge Template file and extract all feature specifications.
        
        This method processes the complete template file, identifies all feature sections,
        and converts them into structured FeatureSpec objects. It includes comprehensive
        error handling for malformed sections and missing data.
        
        Args:
            template_path: Path to the Feature Knowledge Template markdown file
            
        Returns:
            List of FeatureSpec objects representing all successfully parsed features
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ParseError: If template format is fundamentally invalid
        """
        self.logger.info(f"Starting template parsing: {template_path}")
        
        # Reset parsing statistics
        self._parsing_stats = {
            'total_features': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'warnings': 0
        }
        
        # Validate file exists and is readable
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        if not template_path.is_file():
            raise ParseError(f"Path is not a file: {template_path}")
        
        try:
            # Read template content
            content = template_path.read_text(encoding='utf-8')
            self.logger.debug(f"Read template file: {len(content)} characters")
            
        except Exception as e:
            raise ParseError(f"Failed to read template file: {e}")
        
        # Validate basic template format
        if not self._validate_template_structure(content):
            raise ParseError("Template does not contain expected structure (categories and features)")
        
        # Extract feature sections
        try:
            feature_sections = self.extract_feature_sections(content)
            self._parsing_stats['total_features'] = len(feature_sections)
            self.logger.info(f"Found {len(feature_sections)} feature sections")
            
        except Exception as e:
            raise ParseError(f"Failed to extract feature sections: {e}")
        
        # Parse each feature section with error handling
        parsed_features = []
        
        for feature_name, section_content in feature_sections.items():
            try:
                feature_spec = self.parse_feature_details(section_content, feature_name)
                parsed_features.append(feature_spec)
                self._parsing_stats['successful_parses'] += 1
                self.logger.debug(f"Successfully parsed feature: {feature_name}")
                
            except ParseError as e:
                self._parsing_stats['failed_parses'] += 1
                self.logger.error(f"Failed to parse feature '{feature_name}': {e}")
                # Continue parsing other features
                continue
                
            except Exception as e:
                self._parsing_stats['failed_parses'] += 1
                self.logger.error(f"Unexpected error parsing feature '{feature_name}': {e}")
                continue
        
        # Log final statistics
        self._log_parsing_summary()
        
        if not parsed_features:
            raise ParseError("No features could be successfully parsed from template")
        
        self.logger.info(f"Template parsing completed: {len(parsed_features)} features parsed successfully")
        return parsed_features
    
    def extract_feature_sections(self, content: str) -> Dict[str, str]:
        """
        Extract individual feature sections from template content by identifying feature boundaries.
        
        This method scans the template content to identify feature sections based on
        markdown headers and returns a mapping of feature names to their content.
        It handles various header formats and nested structures.
        
        Args:
            content: Raw template content as string
            
        Returns:
            Dictionary mapping feature names to their section content
            
        Raises:
            ParseError: If no valid feature sections are found
        """
        self.logger.debug("Extracting feature sections from template content")
        
        if not content or not content.strip():
            raise ParseError("Template content is empty")
        
        try:
            # Use the underlying markdown parser to extract sections
            sections = self._markdown_parser.extract_feature_sections(content)
            
            if not sections:
                raise ParseError("No feature sections found in template")
            
            # Validate section content
            validated_sections = {}
            for name, section_content in sections.items():
                if self._validate_section_content(section_content):
                    validated_sections[name] = section_content
                else:
                    self._parsing_stats['warnings'] += 1
                    self.logger.warning(f"Section '{name}' has minimal content, including anyway")
                    validated_sections[name] = section_content
            
            self.logger.debug(f"Extracted {len(validated_sections)} valid feature sections")
            return validated_sections
            
        except Exception as e:
            raise ParseError(f"Failed to extract feature sections: {e}")
    
    def parse_feature_details(self, section_content: str, feature_name: str) -> FeatureSpec:
        """
        Parse detailed feature information from a section and extract structured data.
        
        This method processes a single feature section and extracts all relevant
        information including economic hypotheses, performance characteristics,
        failure modes, and implementation details. It includes comprehensive
        error handling for missing or malformed data.
        
        Args:
            section_content: Raw content of a feature section
            feature_name: Name of the feature being parsed
            
        Returns:
            FeatureSpec object with parsed and validated information
            
        Raises:
            ParseError: If section format is invalid or required data is missing
        """
        self.logger.debug(f"Parsing feature details for: {feature_name}")
        
        if not section_content or not section_content.strip():
            raise ParseError(f"Empty section content for feature: {feature_name}")
        
        if not feature_name or not feature_name.strip():
            raise ParseError("Feature name cannot be empty")
        
        try:
            # Use the underlying markdown parser for detailed parsing
            feature_spec = self._markdown_parser.parse_feature_details(section_content, feature_name)
            
            # Additional validation and enhancement
            self._enhance_feature_spec(feature_spec, section_content)
            
            # Final validation
            self._validate_parsed_feature(feature_spec)
            
            return feature_spec
            
        except ParseError:
            # Re-raise ParseError as-is
            raise
            
        except Exception as e:
            raise ParseError(f"Failed to parse feature details for '{feature_name}': {e}")
    
    def validate_template_format(self, template_path: Path) -> bool:
        """
        Validate that a template file follows the expected format.
        
        Args:
            template_path: Path to template file to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        try:
            return self._markdown_parser.validate_template_format(template_path)
        except Exception:
            return False
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the last parsing operation.
        
        Returns:
            Dictionary with parsing statistics including success/failure counts
        """
        return {
            **self._parsing_stats,
            'success_rate': (
                self._parsing_stats['successful_parses'] / max(1, self._parsing_stats['total_features'])
            ) * 100
        }
    
    # Private helper methods for validation and enhancement
    
    def _validate_template_structure(self, content: str) -> bool:
        """Validate that template has basic expected structure."""
        try:
            # Check for category headers (## level)
            has_categories = bool(re.search(r'^##\s+.+', content, re.MULTILINE))
            
            # Check for feature headers (### level)
            has_features = bool(re.search(r'^###\s+.+', content, re.MULTILINE))
            
            return has_categories and has_features
            
        except Exception:
            return False
    
    def _validate_section_content(self, section_content: str) -> bool:
        """Validate that a section has meaningful content."""
        if not section_content or len(section_content.strip()) < 50:
            return False
        
        # Check for at least some structured content
        has_structure = any([
            'Economic Hypothesis' in section_content,
            'Implementation' in section_content,
            'Performance' in section_content,
            'Formula' in section_content,
            '**' in section_content,  # Bold formatting
            '-' in section_content,   # Bullet points
        ])
        
        return has_structure
    
    def _enhance_feature_spec(self, feature_spec: FeatureSpec, section_content: str) -> None:
        """Enhance feature spec with additional derived information."""
        try:
            # Set test complexity based on content analysis
            complexity_indicators = [
                len(feature_spec.failure_modes),
                len(feature_spec.interactions),
                len(feature_spec.regime_dependencies),
                1 if feature_spec.formula else 0,
                len(section_content) // 1000  # Length-based complexity
            ]
            
            complexity_score = sum(complexity_indicators)
            
            if complexity_score >= 5:
                feature_spec.test_complexity = "complex"
            elif complexity_score >= 2:
                feature_spec.test_complexity = "standard"
            else:
                feature_spec.test_complexity = "simple"
            
            # Enhance validation priority based on content
            if any(keyword in section_content.lower() for keyword in 
                   ['critical', 'primary', 'core', 'essential', 'tier 1']):
                feature_spec.validation_priority = "high"
            
            # Extract additional data requirements
            if 'volume' in section_content.lower():
                feature_spec.data_requirements['volume'] = True
            if 'price' in section_content.lower():
                feature_spec.data_requirements['price'] = True
            if 'regime' in section_content.lower():
                feature_spec.data_requirements['regime_data'] = True
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance feature spec for {feature_spec.name}: {e}")
    
    def _validate_parsed_feature(self, feature_spec: FeatureSpec) -> None:
        """Validate that parsed feature meets minimum requirements."""
        if not feature_spec.name:
            raise ParseError("Feature name is required")
        
        if not feature_spec.category:
            raise ParseError("Feature category is required")
        
        # Warn about missing important fields
        if not feature_spec.economic_hypothesis:
            self._parsing_stats['warnings'] += 1
            self.logger.warning(f"Feature '{feature_spec.name}' missing economic hypothesis")
        
        if not feature_spec.implementation:
            self._parsing_stats['warnings'] += 1
            self.logger.warning(f"Feature '{feature_spec.name}' missing implementation details")
    
    def generate_feature_inventory(self, template_path: Path):
        """
        Generate a comprehensive feature inventory with categorization and dependencies.
        
        This method implements the functionality required by task 2.3:
        - Feature categorization logic based on template sections
        - Feature dependency detection and mapping
        - Comprehensive feature inventory with test requirements
        - Validation for completeness and consistency
        
        Args:
            template_path: Path to the Feature Knowledge Template
            
        Returns:
            FeatureInventory with complete categorization, dependencies, and test requirements
            
        Raises:
            ValueError: If inventory generation fails
        """
        self.logger.info("Generating comprehensive feature inventory")
        
        try:
            # Import here to avoid circular import
            from .feature_inventory import FeatureInventoryGenerator
            
            inventory_generator = FeatureInventoryGenerator(self)
            inventory = inventory_generator.generate_inventory(template_path)
            
            # Log inventory summary
            summary = inventory.get_test_coverage_summary()
            self.logger.info(f"Generated inventory summary:")
            self.logger.info(f"  Total features: {summary['total_features']}")
            self.logger.info(f"  Critical features: {summary['critical_features']}")
            self.logger.info(f"  Categories: {summary['categories']}")
            self.logger.info(f"  Dependencies: {summary['dependencies']}")
            self.logger.info(f"  Completeness: {summary['coverage_completeness']:.1f}%")
            
            return inventory
            
        except Exception as e:
            self.logger.error(f"Failed to generate feature inventory: {e}")
            raise ValueError(f"Inventory generation failed: {e}")
    
    def _log_parsing_summary(self) -> None:
        """Log summary of parsing operation."""
        stats = self._parsing_stats
        
        self.logger.info(f"Parsing Summary:")
        self.logger.info(f"  Total features found: {stats['total_features']}")
        self.logger.info(f"  Successfully parsed: {stats['successful_parses']}")
        self.logger.info(f"  Failed to parse: {stats['failed_parses']}")
        self.logger.info(f"  Warnings: {stats['warnings']}")
        
        if stats['total_features'] > 0:
            success_rate = (stats['successful_parses'] / stats['total_features']) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}%")