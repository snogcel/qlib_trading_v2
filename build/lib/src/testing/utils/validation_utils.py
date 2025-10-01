"""
Validation utilities for the feature test coverage system.
"""

from typing import List, Dict, Any, Optional
import re

from ..models.feature_spec import FeatureSpec
from ..models.test_case import TestCase, TestType


def validate_feature_spec(feature_spec: FeatureSpec) -> List[str]:
    """
    Validate a feature specification for completeness and correctness.
    
    Args:
        feature_spec: FeatureSpec object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Basic field validation
    if not feature_spec.name or not feature_spec.name.strip():
        errors.append("Feature name cannot be empty")
    
    if not feature_spec.category or not feature_spec.category.strip():
        errors.append("Feature category cannot be empty")
    
    # Name format validation
    if feature_spec.name and not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', feature_spec.name):
        errors.append("Feature name must start with letter and contain only letters, numbers, and underscores")
    
    # Implementation validation
    if not feature_spec.implementation:
        errors.append("Feature implementation description is required")
    
    # Economic hypothesis validation
    if feature_spec.economic_hypothesis:
        if len(feature_spec.economic_hypothesis.strip()) < 10:
            errors.append("Economic hypothesis should be at least 10 characters long")
    
    # Performance characteristics validation
    if feature_spec.performance_characteristics:
        for metric, value in feature_spec.performance_characteristics.items():
            if isinstance(value, (int, float)) and (value < 0 or value > 1):
                errors.append(f"Performance metric '{metric}' should be between 0 and 1")
    
    # Empirical ranges validation
    if feature_spec.empirical_ranges:
        for range_name, range_value in feature_spec.empirical_ranges.items():
            if not isinstance(range_value, (int, float)):
                errors.append(f"Empirical range '{range_name}' must be numeric")
    
    # Tier validation
    valid_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    if feature_spec.tier not in valid_tiers:
        errors.append(f"Feature tier must be one of: {', '.join(valid_tiers)}")
    
    # Validation priority consistency
    if feature_spec.tier == "Tier 1" and feature_spec.validation_priority != "high":
        errors.append("Tier 1 features should have high validation priority")
    
    return errors


def validate_test_case(test_case: TestCase) -> List[str]:
    """
    Validate a test case for completeness and correctness.
    
    Args:
        test_case: TestCase object to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Basic field validation
    if not test_case.test_id or not test_case.test_id.strip():
        errors.append("Test ID cannot be empty")
    
    if not test_case.feature_name or not test_case.feature_name.strip():
        errors.append("Feature name cannot be empty")
    
    if not test_case.description or not test_case.description.strip():
        errors.append("Test description cannot be empty")
    
    # Test ID format validation
    if test_case.test_id and not re.match(r'^[a-zA-Z0-9_-]+$', test_case.test_id):
        errors.append("Test ID can only contain letters, numbers, underscores, and hyphens")
    
    # Test type validation
    if not isinstance(test_case.test_type, TestType):
        errors.append("Test type must be a valid TestType enum")
    
    # Validation criteria check
    if not test_case.validation_criteria:
        errors.append("Test case must have validation criteria defined")
    
    # Tolerance validation
    if test_case.tolerance < 0 or test_case.tolerance > 1:
        errors.append("Tolerance must be between 0 and 1")
    
    # Duration validation
    if test_case.estimated_duration <= 0:
        errors.append("Estimated duration must be positive")
    
    # Regime context validation
    if test_case.regime_context:
        valid_regimes = ["bull", "bear", "sideways", "high_volatility", "low_volatility"]
        if test_case.regime_context not in valid_regimes:
            errors.append(f"Regime context must be one of: {', '.join(valid_regimes)}")
    
    # Dependencies validation
    for dependency in test_case.dependencies:
        if not isinstance(dependency, str) or not dependency.strip():
            errors.append("All dependencies must be non-empty strings")
    
    return errors


def validate_test_parameters(test_parameters: Dict[str, Any], test_type: TestType) -> List[str]:
    """
    Validate test parameters based on test type requirements.
    
    Args:
        test_parameters: Dictionary of test parameters
        test_type: Type of test being validated
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if test_type == TestType.ECONOMIC_HYPOTHESIS:
        # Economic hypothesis tests should have hypothesis definition
        if 'hypothesis' not in test_parameters:
            errors.append("Economic hypothesis tests must define 'hypothesis' parameter")
        
        if 'expected_behavior' not in test_parameters:
            errors.append("Economic hypothesis tests must define 'expected_behavior' parameter")
    
    elif test_type == TestType.PERFORMANCE:
        # Performance tests should have metrics and thresholds
        if 'metrics' not in test_parameters:
            errors.append("Performance tests must define 'metrics' parameter")
        
        if 'thresholds' not in test_parameters:
            errors.append("Performance tests must define 'thresholds' parameter")
    
    elif test_type == TestType.FAILURE_MODE:
        # Failure mode tests should define failure conditions
        if 'failure_condition' not in test_parameters:
            errors.append("Failure mode tests must define 'failure_condition' parameter")
        
        if 'expected_response' not in test_parameters:
            errors.append("Failure mode tests must define 'expected_response' parameter")
    
    elif test_type == TestType.IMPLEMENTATION:
        # Implementation tests should have formula or calculation details
        if 'formula' not in test_parameters and 'calculation_method' not in test_parameters:
            errors.append("Implementation tests must define 'formula' or 'calculation_method' parameter")
    
    elif test_type == TestType.REGIME_DEPENDENCY:
        # Regime tests should specify regime and expected behavior
        if 'regime_type' not in test_parameters:
            errors.append("Regime dependency tests must define 'regime_type' parameter")
        
        if 'expected_regime_behavior' not in test_parameters:
            errors.append("Regime dependency tests must define 'expected_regime_behavior' parameter")
    
    return errors


def validate_data_requirements(data_requirements: Dict[str, Any]) -> List[str]:
    """
    Validate data requirements for test execution.
    
    Args:
        data_requirements: Dictionary of data requirements
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ['assets', 'timeframe', 'min_data_points']
    for field in required_fields:
        if field not in data_requirements:
            errors.append(f"Data requirements must include '{field}'")
    
    # Validate assets
    if 'assets' in data_requirements:
        assets = data_requirements['assets']
        if not isinstance(assets, list) or not assets:
            errors.append("Assets must be a non-empty list")
        
        for asset in assets:
            if not isinstance(asset, str) or not asset.strip():
                errors.append("All assets must be non-empty strings")
    
    # Validate timeframe
    if 'timeframe' in data_requirements:
        timeframe = data_requirements['timeframe']
        valid_timeframes = ['daily', 'hourly', '4h', '1h', '15m', '5m', '1m']
        if timeframe not in valid_timeframes:
            errors.append(f"Timeframe must be one of: {', '.join(valid_timeframes)}")
    
    # Validate min_data_points
    if 'min_data_points' in data_requirements:
        min_points = data_requirements['min_data_points']
        if not isinstance(min_points, int) or min_points <= 0:
            errors.append("min_data_points must be a positive integer")
    
    return errors


def check_feature_completeness(feature_spec: FeatureSpec) -> Dict[str, Any]:
    """
    Check the completeness of a feature specification.
    
    Args:
        feature_spec: FeatureSpec object to check
        
    Returns:
        Dictionary with completeness analysis
    """
    completeness = {
        'overall_score': 0.0,
        'missing_sections': [],
        'weak_sections': [],
        'strong_sections': [],
        'recommendations': []
    }
    
    # Define section weights
    section_weights = {
        'basic_info': 0.2,  # name, category, tier
        'implementation': 0.2,
        'economic_hypothesis': 0.15,
        'performance_characteristics': 0.15,
        'failure_modes': 0.1,
        'regime_dependencies': 0.1,
        'interactions': 0.1
    }
    
    scores = {}
    
    # Basic info score
    basic_score = 0.0
    if feature_spec.name:
        basic_score += 0.4
    if feature_spec.category:
        basic_score += 0.3
    if feature_spec.tier:
        basic_score += 0.3
    scores['basic_info'] = basic_score
    
    # Implementation score
    impl_score = 0.0
    if feature_spec.implementation:
        impl_score += 0.7
    if feature_spec.formula:
        impl_score += 0.3
    scores['implementation'] = impl_score
    
    # Economic hypothesis score
    econ_score = 1.0 if feature_spec.economic_hypothesis else 0.0
    scores['economic_hypothesis'] = econ_score
    
    # Performance characteristics score
    perf_score = 1.0 if feature_spec.performance_characteristics else 0.0
    scores['performance_characteristics'] = perf_score
    
    # Failure modes score
    failure_score = 1.0 if feature_spec.failure_modes else 0.0
    scores['failure_modes'] = failure_score
    
    # Regime dependencies score
    regime_score = 1.0 if feature_spec.regime_dependencies else 0.0
    scores['regime_dependencies'] = regime_score
    
    # Interactions score
    interaction_score = 1.0 if feature_spec.interactions else 0.0
    scores['interactions'] = interaction_score
    
    # Calculate overall score
    overall_score = sum(scores[section] * section_weights[section] for section in scores)
    completeness['overall_score'] = overall_score
    
    # Categorize sections
    for section, score in scores.items():
        if score == 0.0:
            completeness['missing_sections'].append(section)
        elif score < 0.5:
            completeness['weak_sections'].append(section)
        else:
            completeness['strong_sections'].append(section)
    
    # Generate recommendations
    if completeness['missing_sections']:
        completeness['recommendations'].append(
            f"Add missing sections: {', '.join(completeness['missing_sections'])}"
        )
    
    if completeness['weak_sections']:
        completeness['recommendations'].append(
            f"Strengthen weak sections: {', '.join(completeness['weak_sections'])}"
        )
    
    if overall_score < 0.7:
        completeness['recommendations'].append(
            "Feature specification needs significant improvement for comprehensive testing"
        )
    
    return completeness


def validate_feature_spec_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate feature specification data structure before creating FeatureSpec object.
    
    Args:
        data: Feature specification data to validate
        
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
    string_fields = ['name', 'category', 'tier', 'implementation', 'economic_hypothesis']
    for field in string_fields:
        if field in data and data[field] is not None and not isinstance(data[field], str):
            errors.append(f"Field '{field}' must be a string")
    
    # Validate list fields
    list_fields = ['failure_modes', 'interactions', 'synergies', 'conflicts', 'temporal_dependencies']
    for field in list_fields:
        if field in data and data[field] is not None and not isinstance(data[field], list):
            errors.append(f"Field '{field}' must be a list")
    
    # Validate dict fields
    dict_fields = ['performance_characteristics', 'empirical_ranges', 'regime_dependencies', 'data_requirements']
    for field in dict_fields:
        if field in data and data[field] is not None and not isinstance(data[field], dict):
            errors.append(f"Field '{field}' must be a dictionary")
    
    # Validate tier values
    valid_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    if 'tier' in data and data['tier'] and data['tier'] not in valid_tiers:
        errors.append(f"Field 'tier' must be one of {valid_tiers}")
    
    # Validate priority values
    valid_priorities = ['high', 'medium', 'low']
    if 'validation_priority' in data and data['validation_priority'] not in valid_priorities:
        errors.append(f"Field 'validation_priority' must be one of {valid_priorities}")
    
    return errors


def validate_template_file(template_path) -> List[str]:
    """
    Validate that a template file exists and has basic required structure.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        List of validation error messages (empty if valid)
    """
    from pathlib import Path
    
    errors = []
    
    path = Path(template_path)
    
    # Check file exists
    if not path.exists():
        errors.append(f"Template file does not exist: {path}")
        return errors
    
    # Check file is readable
    try:
        content = path.read_text(encoding='utf-8')
    except Exception as e:
        errors.append(f"Cannot read template file: {e}")
        return errors
    
    # Check for basic markdown structure
    if not content.strip():
        errors.append("Template file is empty")
        return errors
    
    # Check for category headers (## level)
    if not re.search(r'^##\s+.+', content, re.MULTILINE):
        errors.append("Template file missing category headers (## level)")
    
    # Check for feature headers (### level)
    if not re.search(r'^###\s+.+', content, re.MULTILINE):
        errors.append("Template file missing feature headers (### level)")
    
    return errors


def validate_parsed_features(features: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Validate a list of parsed feature specifications.
    
    Args:
        features: List of feature specification dictionaries
        
    Returns:
        Dictionary mapping feature names to their validation errors
    """
    validation_results = {}
    
    for feature in features:
        feature_name = feature.get('name', 'Unknown')
        errors = validate_feature_spec_data(feature)
        
        if errors:
            validation_results[feature_name] = errors
    
    return validation_results


def extract_numeric_values(text: str) -> List[float]:
    """
    Extract numeric values from text.
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of numeric values found
    """
    if not text:
        return []
    
    # Pattern to match numbers (including decimals and ranges)
    number_pattern = r'[-+]?(?:\d*\.?\d+)'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers