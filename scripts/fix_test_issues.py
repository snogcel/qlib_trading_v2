#!/usr/bin/env python3
"""
Fix Test Issues Script
Addresses import errors and encoding issues across test files
"""

import os
import re
import sys
from pathlib import Path

def fix_import_paths():
    """Fix import path issues in test files"""
    print("Fixing import path issues...")
    
    # Find all Python test files (exclude conda and other external directories)
    test_files = []
    for root, dirs, files in os.walk('.'):
        # Skip external directories
        if any(skip in root for skip in ['.conda', '.git', '__pycache__', 'node_modules', '.pytest_cache']):
            continue
        for file in files:
            if file.endswith('.py') and ('test_' in file or file.startswith('test_')):
                test_files.append(os.path.join(root, file))
    
    fixed_files = []
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common import path patterns
            patterns_to_fix = [
                # Fix relative imports that should be absolute
                (r'from src\.', r'from src.'),
                # Ensure proper path setup
                (r'project_root = os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)',
                 r'project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'),
            ]
            
            for pattern, replacement in patterns_to_fix:
                content = re.sub(pattern, replacement, content)
            
            # Ensure proper path setup exists
            if 'from src.' in content and 'project_root' not in content:
                # Add path setup at the top
                import_section = """
# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

"""
                # Insert after the docstring but before imports
                lines = content.split('\n')
                insert_idx = 0
                
                # Find where to insert (after shebang and docstring)
                in_docstring = False
                for i, line in enumerate(lines):
                    if line.startswith('"""') or line.startswith("'''"):
                        if not in_docstring:
                            in_docstring = True
                        else:
                            insert_idx = i + 1
                            break
                    elif not in_docstring and (line.startswith('import ') or line.startswith('from ')):
                        insert_idx = i
                        break
                
                if insert_idx > 0:
                    lines.insert(insert_idx, import_section)
                    content = '\n'.join(lines)
            
            # Write back if changed
            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(test_file)
                print(f"  Fixed: {test_file}")
        
        except Exception as e:
            print(f"  Error fixing {test_file}: {e}")
    
    print(f"Fixed {len(fixed_files)} files")
    return fixed_files

def remove_problematic_emojis():
    """Replace problematic emojis with safe alternatives"""
    print("Fixing encoding issues...")
    
    # Emoji replacements (problematic -> safe)
    emoji_replacements = {
        '🧪': '[TEST]',
        '': '[SEARCH]',
        '📊': '[CHART]',
        '': '[PASS]',
        '': '[FAIL]',
        '⚠️': '[WARN]',
        '🎯': '[TARGET]',
        '🔧': '[FIX]',
        '📋': '[LIST]',
        '🚀': '[START]',
        '': '[IDEA]',
        '🌪️': '[REGIME]',
        '': '[UP]',
        '📉': '[DOWN]',
        '⚖️': '[BALANCE]',
        '🎉': '[SUCCESS]',
        '💰': '[MONEY]',
        '🔄': '[CYCLE]',
        '⏰': '[TIME]',
        '💥': '[ERROR]',
        '🏗️': '[BUILD]',
        '🎪': '[TEST]',
        '🔬': '[ANALYSIS]',
    }
    
    # Find all Python files (exclude external directories)
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip external directories
        if any(skip in root for skip in ['.conda', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'archive']):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    fixed_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Replace problematic emojis
            for emoji, replacement in emoji_replacements.items():
                content = content.replace(emoji, replacement)
            
            # Write back if changed
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(py_file)
                print(f"  Fixed encoding: {py_file}")
        
        except Exception as e:
            print(f"  Error fixing {py_file}: {e}")
    
    print(f"Fixed encoding in {len(fixed_files)} files")
    return fixed_files

def create_test_runner_wrapper():
    """Create a better test runner that handles imports properly"""
    print("Creating improved test runner...")
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Improved Test Runner
Handles imports and encoding properly
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test_with_proper_imports(test_file):
    """Run a test file with proper import handling"""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Add project root to Python path
    env = os.environ.copy()
    pythonpath = str(project_root)
    if 'PYTHONPATH' in env:
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env['PYTHONPATH'] = pythonpath
    
    # Force UTF-8 encoding
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        if test_file.startswith('tests/'):
            # Use pytest for test files
            cmd = [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short']
        else:
            # Run as script with proper path
            cmd = [sys.executable, test_file]
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -2
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_runner_wrapper.py <test_file>")
        sys.exit(1)
    
    test_file = sys.argv[1]
    result = run_test_with_proper_imports(test_file)
    
    print(f"Return code: {result['returncode']}")
    if result['stdout']:
        print("STDOUT:")
        print(result['stdout'])
    if result['stderr']:
        print("STDERR:")
        print(result['stderr'])
    
    sys.exit(result['returncode'])
'''
    
    wrapper_path = 'scripts/test_runner_wrapper.py'
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_content)
    
    print(f"Created test runner wrapper: {wrapper_path}")
    return wrapper_path

def main():
    """Main fix function"""
    print("COMPREHENSIVE TEST ISSUE FIX")
    print("=" * 50)
    
    # Fix 1: Import path issues
    fixed_imports = fix_import_paths()
    
    # Fix 2: Encoding issues
    fixed_encoding = remove_problematic_emojis()
    
    # Fix 3: Create better test runner
    wrapper_path = create_test_runner_wrapper()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print(f"Fixed import paths in {len(fixed_imports)} files")
    print(f"Fixed encoding in {len(fixed_encoding)} files")
    print(f"Created test runner wrapper: {wrapper_path}")
    
    print("\nNEXT STEPS:")
    print("1. Update run_all_tests.py to use the wrapper")
    print("2. Test a few files manually to verify fixes")
    print("3. Run the full test suite")
    
    return {
        'fixed_imports': fixed_imports,
        'fixed_encoding': fixed_encoding,
        'wrapper_path': wrapper_path
    }

if __name__ == "__main__":
    main()