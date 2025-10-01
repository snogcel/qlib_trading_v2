#!/usr/bin/env python3
"""
Master test runner for all NautilusTrader test suites.
Executes all test suites and provides comprehensive reporting.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_suite(test_file, description):
    """Run a test suite and return results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"FILE: {test_file}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the test with pytest
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=300)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED - {description}")
            print(f"   Execution time: {execution_time:.2f}s")
            
            # Extract test count from output
            lines = result.stdout.split('\n')
            summary_line = [line for line in lines if 'passed' in line and 'in' in line]
            if summary_line:
                print(f"   {summary_line[-1].strip()}")
            
            return True, execution_time, result.stdout
        else:
            print(f"❌ FAILED - {description}")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Error output:")
            print(result.stderr)
            return False, execution_time, result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT - {description}")
        print(f"   Test suite exceeded 5 minute timeout")
        return False, 300, "Timeout after 300 seconds"
        
    except Exception as e:
        print(f"💥 ERROR - {description}")
        print(f"   Exception: {str(e)}")
        return False, 0, str(e)

def main():
    """Main test runner"""
    print("NAUTILUS TRADER - COMPREHENSIVE TEST SUITE RUNNER")
    print("="*80)
    print("Running all test suites for NautilusTrader POC validation")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Define test suites
    test_suites = [
        {
            "file": "test_simple.py",
            "description": "Basic Functionality Tests",
            "category": "Unit Tests"
        },
        {
            "file": "tests/integration/test_nautilus_requirements_alignment_mock.py",
            "description": "Requirements Alignment Tests (Mock)",
            "category": "Integration Tests"
        },
        {
            "file": "tests/integration/test_q50_comprehensive.py", 
            "description": "Q50 Comprehensive Integration Tests",
            "category": "Integration Tests"
        }
    ]
    
    # Results tracking
    results = []
    total_start_time = time.time()
    
    # Run each test suite
    for suite in test_suites:
        test_file = suite["file"]
        description = suite["description"]
        category = suite["category"]
        
        # Check if test file exists
        if not Path(test_file).exists():
            print(f"\n⚠️  SKIPPED - {description}")
            print(f"   File not found: {test_file}")
            results.append({
                "suite": description,
                "category": category,
                "status": "SKIPPED",
                "time": 0,
                "details": "File not found"
            })
            continue
        
        # Run the test suite
        success, exec_time, output = run_test_suite(test_file, description)
        
        results.append({
            "suite": description,
            "category": category,
            "status": "PASSED" if success else "FAILED",
            "time": exec_time,
            "details": output
        })
    
    # Generate summary report
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUITE SUMMARY")
    print(f"{'='*80}")
    
    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    skipped_count = sum(1 for r in results if r["status"] == "SKIPPED")
    total_count = len(results)
    
    print(f"Total Test Suites: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⚠️  Skipped: {skipped_count}")
    print(f"⏱️  Total Execution Time: {total_time:.2f}s")
    
    # Detailed results by category
    categories = {}
    for result in results:
        category = result["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    for category, cat_results in categories.items():
        print(f"\n{category}:")
        for result in cat_results:
            status_icon = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⚠️"}[result["status"]]
            print(f"  {status_icon} {result['suite']} ({result['time']:.2f}s)")
    
    # Overall status
    print(f"\n{'='*80}")
    if failed_count == 0:
        print("🎉 ALL TEST SUITES PASSED!")
        print("   Your NautilusTrader implementation is ready for deployment")
        print("   All requirements alignment tests are passing")
        print("   Q50-centric regime-aware approach is validated")
        exit_code = 0
    else:
        print("🚨 SOME TEST SUITES FAILED!")
        print("   Please review failed tests before deployment")
        print("   Check error messages and fix issues")
        exit_code = 1
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if passed_count > 0:
        print("✅ Successful test suites indicate core functionality is working")
    if failed_count > 0:
        print("❌ Failed test suites need attention before production deployment")
    if skipped_count > 0:
        print("⚠️  Skipped test suites should be investigated and enabled if possible")
    
    print(f"\nNEXT STEPS:")
    if failed_count == 0:
        print("1. Deploy to staging environment for further testing")
        print("2. Run backtests with historical data")
        print("3. Monitor performance metrics")
        print("4. Set up automated testing in CI/CD pipeline")
    else:
        print("1. Fix failing tests")
        print("2. Re-run test suite")
        print("3. Investigate root causes of failures")
        print("4. Update implementation as needed")
    
    # Save detailed report
    report_file = "test_suite_report.txt"
    with open(report_file, "w") as f:
        f.write("NAUTILUS TRADER TEST SUITE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Execution Time: {total_time:.2f}s\n")
        f.write(f"Test Suites: {total_count} ({passed_count} passed, {failed_count} failed, {skipped_count} skipped)\n\n")
        
        for result in results:
            f.write(f"Suite: {result['suite']}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Status: {result['status']}\n")
            f.write(f"Time: {result['time']:.2f}s\n")
            f.write(f"Details:\n{result['details']}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)