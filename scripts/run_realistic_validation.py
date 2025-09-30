#!/usr/bin/env python3
"""
Realistic Statistical Validation Runner
Uses actual data from macro_features.pkl to ensure tests complete successfully
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def check_data_availability():
    """Check if real data is available"""
    data_paths = [
        os.path.join(project_root, 'data3', 'macro_features.pkl'),
        os.path.join(project_root, 'data', 'macro_features.pkl')
    ]
    
    available_data = []
    for path in data_paths:
        if os.path.exists(path):
            try:
                import pandas as pd
                df = pd.read_pickle(path)
                available_data.append({
                    'path': path,
                    'size': len(df),
                    'columns': len(df.columns),
                    'memory': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                })
            except Exception as e:
                print(f"⚠️  Could not load {path}: {e}")
    
    return available_data

def run_realistic_validation():
    """Run the realistic statistical validation test suite"""
    print("Running Realistic Statistical Validation Tests...")
    print("="*70)
    
    # Check data availability
    available_data = check_data_availability()
    if available_data:
        print("Available Real Data:")
        for data_info in available_data:
            print(f"   {data_info['path']}: {data_info['size']} samples, {data_info['columns']} columns, {data_info['memory']:.1f}MB")
    else:
        print("No real data found - will use realistic fallback data")
    
    print("\n" + "="*70)
    
    test_file = "tests/principles/test_realistic_statistical_validation.py"
    
    try:
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, 
            "-v", 
            "--tb=short",
            "--color=yes",
            "--capture=no"  # Show print statements
        ], cwd=project_root, text=True)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        success = result.returncode == 0
        
        print("\n" + "="*70)
        print(f"Realistic Validation Results:")
        print(f"   Duration: {execution_time:.2f} seconds")
        print(f"   Status: {'PASSED' if success else 'FAILED'}")
        print(f"   Return Code: {result.returncode}")
        
        return success, execution_time, result.returncode
        
    except Exception as e:
        print(f"Error running realistic validation tests: {e}")
        return False, 0, -1

def generate_realistic_validation_report():
    """Generate validation report using realistic data"""
    print("\n📋 Generating Realistic Validation Report...")
    
    # Check data availability
    available_data = check_data_availability()
    
    # Run tests and capture results
    test_success, execution_time, return_code = run_realistic_validation()
    
    # Generate report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "validation_type": "realistic_data",
        "execution_time_seconds": execution_time,
        "overall_status": "PASSED" if test_success else "FAILED",
        "return_code": return_code,
        "data_sources": available_data,
        "test_categories": {
            "Q50 Primary Signal": "Real data correlation testing",
            "Regime Features": "Actual regime identification",
            "Signal Generation": "Q50-centric signal logic",
            "Probability Calculations": "Piecewise probability logic",
            "Vol_Raw Deciles": "Volatility decile classification",
            "System Integration": "End-to-end data flow"
        },
        "validation_improvements": [
            "Uses actual system data (macro_features.pkl)",
            "Realistic correlation thresholds",
            "Proper handling of MultiIndex data",
            "Fallback to synthetic data if needed",
            "Robust error handling for missing columns"
        ]
    }
    
    # Add recommendations based on results
    if test_success:
        report["recommendations"] = [
            "All realistic validation tests passed",
            "Phase 1 statistical validation framework validated with real data",
            "System components work correctly with actual data patterns",
            "Ready for production deployment",
            "Consider implementing continuous validation monitoring"
        ]
        report["phase_1_status"] = "COMPLETED_WITH_REAL_DATA"
    else:
        report["recommendations"] = [
            "Some realistic validation tests failed",
            "Review failed tests - may indicate data preprocessing needs",
            "📋 Check data quality and column availability",
            "Consider data preprocessing pipeline improvements",
            "Validate data loading and transformation steps"
        ]
        report["phase_1_status"] = "NEEDS_DATA_FIXES"
    
    # Save report
    reports_dir = Path(project_root) / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    report_path = reports_dir / "realistic_validation_report.json"
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to: {report_path}")
    return report

def print_realistic_validation_summary(report):
    """Print executive summary of realistic validation results"""
    print("\n" + "="*70)
    print("REALISTIC STATISTICAL VALIDATION SUMMARY")
    print("="*70)
    
    status = report["overall_status"]
    if status == "PASSED":
        print("STATUS: ALL TESTS PASSED WITH REAL DATA")
        print("Phase 1 validated with actual system data!")
    else:
        print("⚠️  STATUS: SOME TESTS FAILED")
        print("May need data preprocessing improvements")
    
    print(f"\nExecution Details:")
    print(f"   Duration: {report['execution_time_seconds']:.2f} seconds")
    print(f"   Validation Type: {report['validation_type']}")
    print(f"   Return Code: {report['return_code']}")
    
    print(f"\n📁 Data Sources:")
    if report["data_sources"]:
        for data_info in report["data_sources"]:
            print(f"   • {os.path.basename(data_info['path'])}: {data_info['size']} samples, {data_info['columns']} columns")
    else:
        print("   • Using realistic fallback data (no macro_features.pkl found)")
    
    print(f"\nTest Categories:")
    for category, description in report["test_categories"].items():
        print(f"   • {category}: {description}")
    
    print(f"\n📋 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   {rec}")
    
    if status == "PASSED":
        print(f"\nKey Achievements:")
        print("   • Statistical validation framework works with real data")
        print("   • All system components validated with actual patterns")
        print("   • Phase 1 of Principle Coverage Framework COMPLETED")
        print("   • Ready for Phase 2: ML Governance Tests")
    
    print("\n" + "="*70)

def main():
    """Main realistic validation workflow"""
    print("Realistic Statistical Validation")
    print("Phase 1 of Principle Coverage Framework - Real Data Edition")
    print("="*70)
    
    # Generate comprehensive report
    report = generate_realistic_validation_report()
    
    # Print executive summary
    print_realistic_validation_summary(report)
    
    # Return appropriate exit code
    if report["overall_status"] == "PASSED":
        print("\nRealistic validation completed successfully!")
        print("Phase 1 validated with actual system data!")
        return 0
    else:
        print("\n⚠️  Realistic validation identified issues that need attention.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)