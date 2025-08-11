#!/usr/bin/env python3
"""
Quick fix for test import issues
Adds the same path setup that works in training_pipeline.py
"""

import os
import re
from pathlib import Path

def fix_test_file(file_path):
    """Add path setup to test file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if path setup already exists
    if 'sys.path.append' in content:
        print(f"[PASS] {file_path} already has path setup")
        return False
    
    # Add path setup after imports
    path_setup = '''
# Add project root to Python path for src imports
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
'''
    
    # Find the first import line
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_index = i
            break
    
    # Insert path setup before first import
    lines.insert(insert_index, path_setup)
    
    # Also fix Unicode emojis
    content_fixed = '\n'.join(lines)
    
    # Replace common Unicode emojis with ASCII equivalents
    emoji_replacements = {
        '\\U0001f52c': '[MICROSCOPE]',
        '\\U0001f3af': '[TARGET]', 
        '\\U0001f50d': '[MAGNIFYING_GLASS]',
        '\\U0001f4ca': '[BAR_CHART]',
        '\\U0001f4c8': '[CHART_UP]',
        '\\U0001f4c9': '[CHART_DOWN]',
        '\\U0001f680': '[ROCKET]',
        '\\U0001f389': '[PARTY]'
    }
    
    for unicode_char, replacement in emoji_replacements.items():
        content_fixed = content_fixed.replace(unicode_char, replacement)
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content_fixed)
    
    print(f"[FIX] Fixed: {file_path}")
    return True

def main():
    """Fix all test files"""
    
    test_files = [
        "tests/unit/test_temporal_quantile_features.py",
        "tests/unit/test_unified_regime_features.py", 
        "tests/validation/test_regime_consolidation.py",
        "Tests/Features/test_signal_classification.py",
        "Tests/Features/test_signal_transforms.py",
        "Tests/Features/test_threshold_strategy.py",
        "Tests/integration/test_q50_integration.py",
        "tests/integration/test_q50_integration.py",
        "tests/validation/test_adaptive_thresholds.py"
    ]
    
    fixed_count = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if fix_test_file(test_file):
                fixed_count += 1
        else:
            print(f"[WARN] Not found: {test_file}")
    
    print(f"\n[PASS] Fixed {fixed_count} test files")
    print("Now try running the tests again!")

if __name__ == "__main__":
    main()