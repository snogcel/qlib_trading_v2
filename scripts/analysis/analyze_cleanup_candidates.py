#!/usr/bin/env python3
"""
Analyze files for cleanup categorization
Helps identify which files to delete, archive, or keep
"""

import os
import re
from pathlib import Path
from datetime import datetime

def analyze_file_patterns():
    """Analyze current files and categorize them"""
    
    # Get all Python files in root directory
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    md_files = [f for f in os.listdir('.') if f.endswith('.md')]
    
    categories = {
        'SAFE_TO_DELETE': [],
        'ARCHIVE_RESEARCH': [],
        'ARCHIVE_ANALYSIS': [],
        'MOVE_TO_TESTS': [],
        'MOVE_TO_SCRIPTS': [],
        'KEEP_IN_ROOT': [],
        'UNCERTAIN': []
    }
    
    # Categorization patterns
    delete_patterns = [
        r'^debug_.*\.py$',
        r'^quick_.*\.py$', 
        r'^simple_.*\.py$',
        r'^minimal_.*\.py$',
        r'^temp_.*\.py$',
        r'^test_.*_fix\.py$',  # Old fix attempts
    ]
    
    archive_research_patterns = [
        r'.*_research_analysis\.md$',
        r'.*_implementation\.py$',
        r'.*_summary\.md$',
        r'.*_consolidation.*\.md$',
        r'phase\d+.*\.md$',
    ]
    
    archive_analysis_patterns = [
        r'^analyze_.*\.py$',
        r'.*_analysis\.py$',
        r'.*_features_.*\.py$',
        r'.*threshold.*analysis.*\.py$',
    ]
    
    test_patterns = [
        r'^test_.*\.py$',
        r'^validate_.*\.py$',
    ]
    
    script_patterns = [
        r'^run_.*\.py$',
        r'^update_.*\.py$',
        r'^cleanup_.*\.py$',
        r'.*_sweep.*\.py$',
    ]
    
    keep_patterns = [
        r'^src/',
        r'^config/',
        r'^quantile_backtester\.py$',
        r'^model_evaluation_suite\.py$',
        r'^integrated_.*\.py$',
    ]
    
    # Categorize Python files
    for file in py_files:
        categorized = False
        
        # Check delete patterns
        for pattern in delete_patterns:
            if re.match(pattern, file):
                categories['SAFE_TO_DELETE'].append(file)
                categorized = True
                break
        
        if categorized:
            continue
            
        # Check archive research patterns
        for pattern in archive_research_patterns:
            if re.match(pattern, file):
                categories['ARCHIVE_RESEARCH'].append(file)
                categorized = True
                break
        
        if categorized:
            continue
            
        # Check archive analysis patterns
        for pattern in archive_analysis_patterns:
            if re.match(pattern, file):
                categories['ARCHIVE_ANALYSIS'].append(file)
                categorized = True
                break
        
        if categorized:
            continue
            
        # Check test patterns
        for pattern in test_patterns:
            if re.match(pattern, file):
                categories['MOVE_TO_TESTS'].append(file)
                categorized = True
                break
        
        if categorized:
            continue
            
        # Check script patterns
        for pattern in script_patterns:
            if re.match(pattern, file):
                categories['MOVE_TO_SCRIPTS'].append(file)
                categorized = True
                break
        
        if categorized:
            continue
            
        # Check keep patterns
        for pattern in keep_patterns:
            if re.match(pattern, file):
                categories['KEEP_IN_ROOT'].append(file)
                categorized = True
                break
        
        if not categorized:
            categories['UNCERTAIN'].append(file)
    
    # Categorize markdown files
    for file in md_files:
        if any(pattern in file.lower() for pattern in ['research', 'analysis', 'summary', 'roadmap']):
            if 'research' in file.lower():
                categories['ARCHIVE_RESEARCH'].append(file)
            else:
                categories['ARCHIVE_ANALYSIS'].append(file)
        elif file in ['README.md', 'TRADING_SYSTEM_PRINCIPLES.md', 'SYSTEM_VALIDATION_SPEC.md']:
            categories['KEEP_IN_ROOT'].append(file)
        else:
            categories['UNCERTAIN'].append(file)
    
    return categories

def print_categorization(categories):
    """Print the categorization results"""
    
    print("🧹 FILE CLEANUP ANALYSIS")
    print("=" * 60)
    
    for category, files in categories.items():
        if not files:
            continue
            
        print(f"\n📂 {category.replace('_', ' ')} ({len(files)} files):")
        print("-" * 40)
        
        for file in sorted(files):
            # Add file size info
            try:
                size = os.path.getsize(file)
                size_str = f"({size:,} bytes)"
            except:
                size_str = "(size unknown)"
            
            print(f"  • {file} {size_str}")
    
    # Summary statistics
    total_files = sum(len(files) for files in categories.values())
    safe_to_delete = len(categories['SAFE_TO_DELETE'])
    to_archive = len(categories['ARCHIVE_RESEARCH']) + len(categories['ARCHIVE_ANALYSIS'])
    
    print(f"\nSUMMARY:")
    print(f"  Total files analyzed: {total_files}")
    print(f"  Safe to delete: {safe_to_delete}")
    print(f"  To archive: {to_archive}")
    print(f"  Need review: {len(categories['UNCERTAIN'])}")

def generate_cleanup_commands(categories):
    """Generate shell commands for cleanup"""
    
    commands = []
    
    # Delete commands
    if categories['SAFE_TO_DELETE']:
        commands.append("# Safe deletions")
        for file in categories['SAFE_TO_DELETE']:
            commands.append(f"rm {file}")
        commands.append("")
    
    # Archive commands
    if categories['ARCHIVE_RESEARCH']:
        commands.append("# Archive research files")
        commands.append("mkdir -p archive/research")
        for file in categories['ARCHIVE_RESEARCH']:
            commands.append(f"mv {file} archive/research/")
        commands.append("")
    
    if categories['ARCHIVE_ANALYSIS']:
        commands.append("# Archive analysis files")
        commands.append("mkdir -p archive/analysis")
        for file in categories['ARCHIVE_ANALYSIS']:
            commands.append(f"mv {file} archive/analysis/")
        commands.append("")
    
    # Move to tests
    if categories['MOVE_TO_TESTS']:
        commands.append("# Move to tests")
        commands.append("mkdir -p tests/validation tests/unit")
        for file in categories['MOVE_TO_TESTS']:
            if 'validate' in file:
                commands.append(f"mv {file} tests/validation/")
            else:
                commands.append(f"mv {file} tests/unit/")
        commands.append("")
    
    # Move to scripts
    if categories['MOVE_TO_SCRIPTS']:
        commands.append("# Move to scripts")
        commands.append("mkdir -p scripts/analysis scripts/maintenance")
        for file in categories['MOVE_TO_SCRIPTS']:
            if any(word in file for word in ['run', 'sweep', 'optimization']):
                commands.append(f"mv {file} scripts/analysis/")
            else:
                commands.append(f"mv {file} scripts/maintenance/")
        commands.append("")
    
    return commands

def main():
    """Main analysis function"""
    print("Analyzing files for cleanup...")
    
    categories = analyze_file_patterns()
    print_categorization(categories)
    
    # Generate cleanup commands
    commands = generate_cleanup_commands(categories)
    
    if commands:
        print(f"\nSUGGESTED CLEANUP COMMANDS:")
        print("=" * 60)
        for cmd in commands:
            print(cmd)
        
        # Save commands to file
        with open('cleanup_commands.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated cleanup commands\n")
            f.write("# Review before executing!\n\n")
            f.write('\n'.join(commands))
        
        print(f"\n💾 Commands saved to: cleanup_commands.sh")
        print("⚠️  Review the commands before executing!")

if __name__ == "__main__":
    main()