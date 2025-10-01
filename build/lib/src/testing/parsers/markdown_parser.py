"""
Markdown parser for Feature Knowledge Template.

This module provides utilities for parsing the Feature Knowledge Template
markdown file and extracting structured feature specifications.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..models.feature_spec import FeatureSpec
from ..interfaces.parser_interface import FeatureParserInterface


class ParseError(Exception):
    """Exception raised when parsing fails."""
    pass


class MarkdownFeatureParser(FeatureParserInterface):
    """
    Parser for extracting feature specifications from markdown templates.
    
    This parser understands the structure of the Feature Knowledge Template
    and can extract detailed feature information including economic hypotheses,
    performance characteristics, failure modes, and implementation details.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns for parsing different sections
        self.patterns = {
            'category_header': re.compile(r'^##\s+(.+?)(?:\s+Features?)?$', re.MULTILINE),
            'feature_header': re.compile(r'^###\s+(.+?)(?:\s+\(.+?\))?$', re.MULTILINE),
            'implementation': re.compile(r'\*\*Implementation\*\*:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'formula': re.compile(r'\*\*Formula\*\*:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'economic_hypothesis': re.compile(r'-\s*Economic Hypothesis:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'performance_characteristics': re.compile(r'-\s*Performance Characteristics:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'failure_modes': re.compile(r'-\s*Failure Modes:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'regime_dependencies': re.compile(r'-\s*Regime Dependencies:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'interaction_effects': re.compile(r'-\s*Interaction Effects:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'empirical_ranges': re.compile(r'\*\*Empirical Ranges\*\*:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
            'usage_pattern': re.compile(r'-\s*Usage Pattern[s]?:\s*(.+?)(?=\n\n|\n-|\n\*\*|$)', re.DOTALL),
        }
    
    def parse_template(self, template_path: Path) -> List[FeatureSpec]:
        """
        Parse the Feature Knowledge Template and extract all feature specifications.
        
        Args:
            template_path: Path to the feature template markdown file
            
        Returns:
            List of FeatureSpec objects for all features found
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            ParseError: If template format is invalid
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        try:
            content = template_path.read_text(encoding='utf-8')
        except Exception as e:
            raise ParseError(f"Failed to read template file: {e}")
        
        # Extract feature sections
        feature_sections = self.extract_feature_sections(content)
        
        # Parse each feature section
        features = []
        for feature_name, section_content in feature_sections.items():
            try:
                feature_spec = self.parse_feature_details(section_content, feature_name)
                features.append(feature_spec)
                self.logger.info(f"Successfully parsed feature: {feature_name}")
            except Exception as e:
                self.logger.warning(f"Failed to parse feature '{feature_name}': {e}")
                continue
        
        self.logger.info(f"Parsed {len(features)} features from template")
        return features
    
    def extract_feature_sections(self, content: str) -> Dict[str, str]:
        """
        Extract individual feature sections from template content.
        
        Args:
            content: Raw template content as string
            
        Returns:
            Dictionary mapping feature names to their section content
        """
        sections = {}
        current_category = ""
        
        # Split content into lines for processing
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for category headers (## level)
            category_match = self.patterns['category_header'].match(line)
            if category_match:
                current_category = self._clean_category_name(category_match.group(1))
                i += 1
                continue
            
            # Check for feature headers (### level)
            feature_match = self.patterns['feature_header'].match(line)
            if feature_match:
                feature_name = self._clean_feature_name(feature_match.group(1))
                
                # Extract the feature section content
                section_start = i
                section_end = self._find_section_end(lines, i + 1)
                
                section_content = '\n'.join(lines[section_start:section_end])
                sections[feature_name] = {
                    'content': section_content,
                    'category': current_category
                }
                
                i = section_end
                continue
            
            i += 1
        
        # Convert to simple dict mapping names to content with category info
        result = {}
        for name, data in sections.items():
            # Embed category info in the content for parsing
            content_with_category = f"**Category**: {data['category']}\n\n{data['content']}"
            result[name] = content_with_category
        
        return result
    
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
        try:
            # Extract basic information
            category = self._extract_category(section_content)
            implementation = self._extract_implementation(section_content)
            formula = self._extract_formula(section_content)
            
            # Extract economic and theoretical information
            economic_hypothesis = self._extract_economic_hypothesis(section_content)
            
            # Extract performance information
            performance_characteristics = self._extract_performance_characteristics(section_content)
            empirical_ranges = self._extract_empirical_ranges(section_content)
            
            # Extract failure modes and edge cases
            failure_modes = self._extract_failure_modes(section_content)
            
            # Extract regime information
            regime_dependencies = self._extract_regime_dependencies(section_content)
            
            # Extract interaction information
            interactions = self._extract_interactions(section_content)
            
            # Determine tier based on category and content analysis
            tier = self._determine_tier(category, section_content)
            
            # Create and validate the feature spec
            feature_data = {
                'name': feature_name,
                'category': category,
                'tier': tier,
                'implementation': implementation,
                'formula': formula,
                'economic_hypothesis': economic_hypothesis,
                'performance_characteristics': performance_characteristics,
                'empirical_ranges': empirical_ranges,
                'failure_modes': failure_modes,
                'regime_dependencies': regime_dependencies,
                'interactions': interactions
            }
            
            # Validate required fields
            validation_errors = FeatureSpec.validate_required_fields(feature_data)
            if validation_errors:
                raise ParseError(f"Validation errors for {feature_name}: {validation_errors}")
            
            return FeatureSpec(**feature_data)
            
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
            if not template_path.exists():
                return False
            
            content = template_path.read_text(encoding='utf-8')
            
            # Check for basic structure
            has_categories = bool(self.patterns['category_header'].search(content))
            has_features = bool(self.patterns['feature_header'].search(content))
            
            return has_categories and has_features
            
        except Exception:
            return False
    
    # Helper methods for extracting specific information
    
    def _clean_category_name(self, raw_name: str) -> str:
        """Clean and normalize category names."""
        # Remove emojis and extra whitespace
        cleaned = re.sub(r'[^\w\s&-]', '', raw_name).strip()
        return cleaned
    
    def _clean_feature_name(self, raw_name: str) -> str:
        """Clean and normalize feature names."""
        # Remove parenthetical descriptions and extra whitespace
        cleaned = re.sub(r'\s*\([^)]*\)', '', raw_name).strip()
        return cleaned
    
    def _find_section_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end of a feature section."""
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            # Section ends at next ### or ## header, or end of file
            if line.startswith('###') or line.startswith('##'):
                return i
        return len(lines)
    
    def _extract_category(self, content: str) -> str:
        """Extract category from section content."""
        match = re.search(r'\*\*Category\*\*:\s*(.+)', content)
        if match:
            return match.group(1).strip()
        return "Unknown"
    
    def _extract_implementation(self, content: str) -> str:
        """Extract implementation details."""
        match = self.patterns['implementation'].search(content)
        if match:
            return self._clean_text(match.group(1))
        
        # Fallback: look for "Found in" or "Location" patterns
        location_match = re.search(r'(?:Found in|Location):\s*(.+?)(?=\n|$)', content, re.IGNORECASE)
        if location_match:
            return self._clean_text(location_match.group(1))
        
        return "Implementation details not specified"
    
    def _extract_formula(self, content: str) -> Optional[str]:
        """Extract mathematical formula if present."""
        match = self.patterns['formula'].search(content)
        if match:
            return self._clean_text(match.group(1))
        
        # Look for formula patterns in the content
        formula_patterns = [
            r'`([^`]+)`',  # Code blocks
            r'\$([^$]+)\$',  # Math notation
        ]
        
        for pattern in formula_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the longest match as it's likely the main formula
                return max(matches, key=len)
        
        return None
    
    def _extract_economic_hypothesis(self, content: str) -> str:
        """Extract economic hypothesis."""
        match = self.patterns['economic_hypothesis'].search(content)
        if match:
            return self._clean_text(match.group(1))
        return ""
    
    def _extract_performance_characteristics(self, content: str) -> Dict[str, Any]:
        """Extract performance characteristics."""
        characteristics = {}
        
        match = self.patterns['performance_characteristics'].search(content)
        if match:
            perf_text = match.group(1)
            
            # Extract specific metrics
            metrics_patterns = {
                'sharpe_ratio': r'Sharpe ratio:\s*([~]?[\d.-]+(?:–[\d.-]+)?)',
                'hit_rate': r'Hit rate:\s*([~]?[\d.-]+%?(?:–[\d.-]+%?)?)',
                'win_rate': r'win[- ]?rate:\s*([~]?[\d.-]+%?(?:–[\d.-]+%?)?)',
            }
            
            for metric, pattern in metrics_patterns.items():
                metric_match = re.search(pattern, perf_text, re.IGNORECASE)
                if metric_match:
                    characteristics[metric] = metric_match.group(1)
        
        return characteristics
    
    def _extract_empirical_ranges(self, content: str) -> Dict[str, float]:
        """Extract empirical ranges."""
        ranges = {}
        
        match = self.patterns['empirical_ranges'].search(content)
        if match:
            ranges_text = match.group(1)
            
            # Extract range patterns
            range_patterns = re.findall(r'([^:]+):\s*`?([^`\n]+)`?', ranges_text)
            
            for name, value_str in range_patterns:
                name = name.strip().lower().replace(' ', '_')
                # Try to extract numeric values
                numbers = re.findall(r'[\d.-]+', value_str)
                if numbers:
                    try:
                        ranges[name] = float(numbers[0])
                    except ValueError:
                        ranges[name] = value_str.strip()
        
        return ranges
    
    def _extract_failure_modes(self, content: str) -> List[str]:
        """Extract failure modes."""
        failure_modes = []
        
        match = self.patterns['failure_modes'].search(content)
        if match:
            failure_text = match.group(1)
            
            # Split by bullet points or line breaks
            modes = re.split(r'\n\s*[-•]\s*', failure_text)
            failure_modes = [self._clean_text(mode) for mode in modes if mode.strip()]
        
        return failure_modes
    
    def _extract_regime_dependencies(self, content: str) -> Dict[str, str]:
        """Extract regime dependencies."""
        dependencies = {}
        
        match = self.patterns['regime_dependencies'].search(content)
        if match:
            regime_text = match.group(1)
            
            # Extract regime-specific information
            regime_patterns = {
                'bull_market': r'(?:bull|trending|momentum).*?:\s*([^-\n]+)',
                'bear_market': r'(?:bear|down|decline).*?:\s*([^-\n]+)',
                'high_volatility': r'(?:high.*?vol|volatile).*?:\s*([^-\n]+)',
                'low_volatility': r'(?:low.*?vol|stable).*?:\s*([^-\n]+)',
            }
            
            for regime, pattern in regime_patterns.items():
                regime_match = re.search(pattern, regime_text, re.IGNORECASE)
                if regime_match:
                    dependencies[regime] = self._clean_text(regime_match.group(1))
        
        return dependencies
    
    def _extract_interactions(self, content: str) -> List[str]:
        """Extract feature interactions."""
        interactions = []
        
        match = self.patterns['interaction_effects'].search(content)
        if match:
            interaction_text = match.group(1)
            
            # Split by bullet points or line breaks
            items = re.split(r'\n\s*[-•]\s*', interaction_text)
            interactions = [self._clean_text(item) for item in items if item.strip()]
        
        return interactions
    
    def _determine_tier(self, category: str, content: str) -> str:
        """Determine feature tier based on category and content."""
        # Core Signal features are typically Tier 1
        if 'Core Signal' in category:
            return 'Tier 1'
        
        # Look for tier indicators in content
        tier_match = re.search(r'Tier\s+(\d+)', content, re.IGNORECASE)
        if tier_match:
            return f'Tier {tier_match.group(1)}'
        
        # Risk & Volatility features are typically Tier 1-2
        if any(keyword in category for keyword in ['Risk', 'Volatility']):
            return 'Tier 2'
        
        # Default to Tier 2
        return 'Tier 2'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove markdown formatting
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # Bold
        cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)      # Italic
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)        # Code
        
        return cleaned.strip()