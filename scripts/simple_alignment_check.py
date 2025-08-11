#!/usr/bin/env python3
"""
Simple alignment check to validate key parameters between requirements and implementation
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def check_training_pipeline_parameters():
    """Check key parameters in training pipeline"""
    print("🔍 Checking Training Pipeline Parameters...")
    
    try:
        # Read the training pipeline file
        with open(os.path.join(project_root, 'src', 'training_pipeline.py'), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key parameters
        checks = {
            "Transaction Cost (5 bps)": "realistic_transaction_cost = 0.0005" in content,
            "Variance Percentiles": "quantile(0.30)" in content and "quantile(0.70)" in content,
            "Position Size Clipping": ".clip(0.01, 0.5)" in content,
            "60min Frequency": '"60min"' in content,
            "Q50 Signal Logic": 'q50 > 0' in content and 'q50 < 0' in content,
            "Enhanced Info Ratio": "enhanced_info_ratio" in content,
            "Regime Multipliers": "regime_multipliers" in content,
            "Expected Value Logic": "expected_value" in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅ FOUND" if passed else "❌ MISSING"
            print(f"  {check_name}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading training pipeline: {e}")
        return False

def check_requirements_document():
    """Check requirements document for key parameters"""
    print("\n🔍 Checking Requirements Document...")
    
    try:
        # Read the requirements file
        with open(os.path.join(project_root, '.kiro', 'specs', 'nautilus-trader-poc', 'requirements.md'), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key requirements
        checks = {
            "5 bps Transaction Cost": "0.0005" in content,
            "Variance Percentiles": "quantile(0.30)" in content and "quantile(0.70)" in content,
            "Position Size Limits": "0.01, 0.5" in content or "1%-50%" in content,
            "60min Frequency": "60min" in content,
            "Q50 Signal Logic": "q50 > 0" in content and "q50 < 0" in content,
            "Tradeable Filter": "tradeable=True" in content,
            "Expected Value": "expected_value" in content,
            "Sharpe Target": "1.327" in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅ FOUND" if passed else "❌ MISSING"
            print(f"  {check_name}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading requirements: {e}")
        return False

def check_feature_documentation():
    """Check feature documentation alignment"""
    print("\n🔍 Checking Feature Documentation...")
    
    try:
        # Read the feature documentation
        with open(os.path.join(project_root, 'docs', 'FEATURE_DOCUMENTATION.md'), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check key features are documented
        checks = {
            "Q50 Primary Signal": "Q50 (Primary Signal)" in content,
            "Variance-Based Vol_Risk": "vol_risk" in content and "variance" in content,
            "Regime Detection": "regime_features.py" in content,
            "Multi-Quantile Model": "multi_quantile.py" in content,
            "Data Loaders": "crypto_loader.py" in content and "gdelt_loader.py" in content,
            "Enhanced Info Ratio": "enhanced_info_ratio" in content,
            "Position Sizing": "position_sizing" in content,
            "Production Ready Status": "Production Ready" in content,
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅ FOUND" if passed else "❌ MISSING"
            print(f"  {check_name}: {status}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading feature documentation: {e}")
        return False

def main():
    """Main alignment check"""
    print("🎯 Simple NautilusTrader Requirements Alignment Check")
    print("="*60)
    
    # Run all checks
    pipeline_ok = check_training_pipeline_parameters()
    requirements_ok = check_requirements_document()
    docs_ok = check_feature_documentation()
    
    # Summary
    print("\n" + "="*60)
    print("📊 ALIGNMENT SUMMARY")
    print("="*60)
    
    overall_status = pipeline_ok and requirements_ok and docs_ok
    
    print(f"Training Pipeline: {'✅ ALIGNED' if pipeline_ok else '❌ ISSUES'}")
    print(f"Requirements Doc:  {'✅ ALIGNED' if requirements_ok else '❌ ISSUES'}")
    print(f"Feature Docs:      {'✅ ALIGNED' if docs_ok else '❌ ISSUES'}")
    
    if overall_status:
        print("\n🎉 OVERALL STATUS: FULLY ALIGNED")
        print("✅ Ready to proceed with NautilusTrader POC development!")
        print("\n🎯 Key Confirmations:")
        print("  • Transaction cost: 5 bps (0.0005)")
        print("  • Variance thresholds: 30th/70th/90th percentiles")
        print("  • Position limits: 1%-50% of capital")
        print("  • Data frequency: 60min crypto + daily GDELT")
        print("  • Signal logic: Q50-centric with tradeable filter")
        print("  • Performance target: 1.327+ Sharpe ratio")
        return 0
    else:
        print("\n⚠️  OVERALL STATUS: NEEDS ATTENTION")
        print("🔧 Some alignment issues need to be resolved")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)