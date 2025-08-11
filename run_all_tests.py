#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner
Runs all tests and compares outputs with training_pipeline.py results
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

class TestSuiteRunner:
    def __init__(self):
        self.results = {}
        self.test_dirs = [
            "tests/features",
            "tests/integration", 
            "tests/unit",
            "tests/validation",
            "tests/integration"
        ]
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def find_test_files(self):
        """Find all test files in the project"""
        test_files = []
        
        for test_dir in self.test_dirs:
            if os.path.exists(test_dir):
                for root, dirs, files in os.walk(test_dir):
                    for file in files:
                        if file.startswith('test_') and file.endswith('.py'):
                            test_files.append(os.path.join(root, file))
        
        # Also check root directory for test files
        for file in os.listdir('.'):
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(file)
                
        return sorted(test_files)
    
    def _is_pytest_file(self, test_file):
        """Check if a file contains pytest-style test functions"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for pytest patterns
            pytest_patterns = [
                'def test_',
                'class Test',
                '@pytest.',
                'import pytest',
                'from pytest'
            ]
            
            return any(pattern in content for pattern in pytest_patterns)
        except Exception:
            # If we can't read the file, assume it's not a pytest file
            return False
    
    def run_single_test(self, test_file):
        """Run a single test file and capture output"""
        print(f"\n[TEST] Running: {test_file}")
        print("-" * 50)
        
        try:
            # Determine the best way to run the test
            # First check if it's a pytest-style test by looking for test functions
            is_pytest_file = self._is_pytest_file(test_file)
            
            if test_file.startswith('tests/') and is_pytest_file:
                # Use pytest for proper test files
                cmd = [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short']
            else:
                # Run as standalone script
                cmd = [sys.executable, test_file]
            
            # Run the test with proper encoding handling
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                encoding='utf-8',  # Force UTF-8 encoding
                errors='replace'   # Replace problematic characters
            )
            
            test_result = {
                'file': test_file,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'timestamp': datetime.now().isoformat()
            }
            
            if result.returncode == 0:
                print(f"[PASS] PASSED: {test_file}")
                if result.stdout:
                    print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            else:
                print(f"[FAIL] FAILED: {test_file}")
                if result.stderr:
                    print("Error:", result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
                elif result.stdout:
                    print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
                else:
                    print("No error output captured")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            print(f"[TIME] TIMEOUT: {test_file}")
            return {
                'file': test_file,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[ERROR] {test_file} - {str(e)}")
            return {
                'file': test_file,
                'returncode': -2,
                'stdout': '',
                'stderr': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def compare_with_training_pipeline(self):
        """Compare test outputs with training_pipeline.py results"""
        print("\n[SEARCH] COMPARING WITH TRAINING PIPELINE RESULTS")
        print("=" * 60)
        
        # Check if training pipeline output files exist
        expected_files = [
            "df_all_macro_analysis.csv",
            "data3/macro_features.pkl",
            "temp/correlation_matrix.csv"
        ]
        
        comparison_results = {}
        
        for file in expected_files:
            if os.path.exists(file):
                print(f"[PASS] Found: {file}")
                
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(file)
                        comparison_results[file] = {
                            'exists': True,
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'sample_data': df.head(3).to_dict() if len(df) > 0 else {}
                        }
                        print(f"   Shape: {df.shape}")
                        print(f"   Columns: {len(df.columns)}")
                    except Exception as e:
                        comparison_results[file] = {
                            'exists': True,
                            'error': str(e)
                        }
                        print(f"   [FAIL] Error reading: {e}")
                        
                elif file.endswith('.pkl'):
                    try:
                        import pickle
                        with open(file, 'rb') as f:
                            data = pickle.load(f)
                        comparison_results[file] = {
                            'exists': True,
                            'type': str(type(data)),
                            'shape': getattr(data, 'shape', 'N/A')
                        }
                        print(f"   Type: {type(data)}")
                        if hasattr(data, 'shape'):
                            print(f"   Shape: {data.shape}")
                    except Exception as e:
                        comparison_results[file] = {
                            'exists': True,
                            'error': str(e)
                        }
                        print(f"   [FAIL] Error reading: {e}")
            else:
                print(f"[FAIL] Missing: {file}")
                comparison_results[file] = {'exists': False}
        
        return comparison_results
    
    def run_all_tests(self):
        """Run all tests and generate comprehensive report"""
        print("[START] STARTING COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        # Find all test files
        test_files = self.find_test_files()
        print(f"Found {len(test_files)} test files:")
        for test_file in test_files:
            print(f"  - {test_file}")
        
        if not test_files:
            print("[WARN] No test files found!")
            return
        
        # Run each test
        all_results = []
        passed = 0
        failed = 0
        
        for test_file in test_files:
            result = self.run_single_test(test_file)
            all_results.append(result)
            
            if result['success']:
                passed += 1
            else:
                failed += 1
        
        # Compare with training pipeline
        comparison_results = self.compare_with_training_pipeline()
        
        # Generate summary report
        summary = {
            'total_tests': len(test_files),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(test_files) if test_files else 0,
            'test_results': all_results,
            'training_pipeline_comparison': comparison_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create summary report
        self.generate_summary_report(summary)
        
        return summary
    
    def generate_summary_report(self, summary):
        """Generate human-readable summary report"""
        report = f"""
# TEST SUITE EXECUTION REPORT

**Generated**: {summary['timestamp']}

## [CHART] SUMMARY STATISTICS
- **Total Tests**: {summary['total_tests']}
- **Passed**: {summary['passed']} [PASS]
- **Failed**: {summary['failed']} [FAIL]
- **Pass Rate**: {summary['pass_rate']:.1%}

## [TEST] TEST RESULTS

### [PASS] PASSED TESTS
"""
        
        for result in summary['test_results']:
            if result['success']:
                report += f"- {result['file']}\n"
        
        report += "\n### [FAIL] FAILED TESTS\n"
        
        for result in summary['test_results']:
            if not result['success']:
                report += f"- {result['file']}\n"
                report += f"  Error: {result['stderr'][:200]}...\n"
        
        report += "\n## [SEARCH] TRAINING PIPELINE COMPARISON\n"
        
        for file, data in summary['training_pipeline_comparison'].items():
            if data.get('exists'):
                report += f"[PASS] {file}\n"
                if 'shape' in data:
                    report += f"   Shape: {data['shape']}\n"
                if 'columns' in data:
                    report += f"   Columns: {len(data['columns'])}\n"
            else:
                report += f"[FAIL] {file} (missing)\n"
        
        report += f"""
## [TARGET] RECOMMENDATIONS

### If Pass Rate > 80%:
- System is in good shape after reorganization
- Focus on fixing specific failed tests
- Ready to proceed with next development phase

### If Pass Rate < 80%:
- Review import paths in failed tests
- Check for missing dependencies
- May need additional reorganization fixes

## 📁 DETAILED RESULTS
See `test_results/test_results.json` for complete output logs.
"""
        
        with open(self.output_dir / "summary_report.md", 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print("[LIST] TEST EXECUTION COMPLETE")
        print("=" * 60)
        print(f"[PASS] Passed: {summary['passed']}")
        print(f"[FAIL] Failed: {summary['failed']}")
        print(f"[CHART] Pass Rate: {summary['pass_rate']:.1%}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📄 Summary report: {self.output_dir}/summary_report.md")

def main():
    """Main execution function"""
    runner = TestSuiteRunner()
    results = runner.run_all_tests()
    
    # Print final summary
    if results['pass_rate'] >= 0.8:
        print("\n[SUCCESS] EXCELLENT! High pass rate - system is in great shape!")
    elif results['pass_rate'] >= 0.6:
        print("\n👍 GOOD! Decent pass rate - some fixes needed")
    else:
        print("\n[WARN] ATTENTION NEEDED: Low pass rate - review failed tests")

if __name__ == "__main__":
    main()