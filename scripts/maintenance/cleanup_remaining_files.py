#!/usr/bin/env python3
"""
Identify remaining files that can be safely deleted after reorganization
"""

import os
import glob
from pathlib import Path

def find_files_to_delete():
    """Find files that are safe to delete based on cleanup plan"""
    
    # Files that are definitely safe to delete (superseded/deprecated)
    safe_to_delete = [
        # Old versions superseded by optimized versions
        "qlib_custom/crypto_loader.py",  # → crypto_loader_optimized.py (now in src/)
        "qlib_custom/gdelt_loader.py",   # → gdelt_loader_optimized.py (now in src/)
        
        # Debugging scripts (served their purpose)
        "debug_*.py",
        
        # Quick test scripts (ad-hoc, not systematic)
        "quick_*.py",
        "simple_*.py",
        "minimal_*.py",
        
        # Redundant analysis scripts
        "feature_correlation_test.py",
        "test_momentum_comparison.py",
        
        # Experimental/exploratory files (consolidated)
        "vol_raw_momentum_analysis.py",
        "volatility_window_analysis.py", 
        "recalculate_vol_deciles.py",
        
        # PPO runner copies
        "ppo_sweep_runner copy*.py",
        "ppo_sweep_runner_backup.py",
        
        # Temporary analysis files
        "investigate_vol_risk_bounds.py",
        "remove_redundant_features.py",
    ]
    
    # Files to potentially archive (valuable but not active)
    archive_candidates = [
        "vol_momentum_hybrid_implementation.py",
        "vol_risk_strategic_implementation.py", 
        "q50_regime_implementation.py",
        "q50_centric_implementation.py",
        "comprehensive_feature_analysis.py",
        "volatility_features_impact_assessment.py",
        "regime_features_usage_analysis.py",
        "spread_validation_results_analysis.py",
    ]
    
    print("🗑️  FILES SAFE TO DELETE:")
    print("=" * 50)
    
    found_files = []
    for pattern in safe_to_delete:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.exists(match):
                found_files.append(match)
                print(f"   {match}")
    
    print(f"\n📦 FILES TO CONSIDER ARCHIVING:")
    print("=" * 50)
    
    archive_files = []
    for file in archive_candidates:
        if os.path.exists(file):
            archive_files.append(file)
            print(f"   📁 {file}")
    
    # Check for CSV/data files that might be safe to clean
    print(f"\nDATA FILES TO REVIEW:")
    print("=" * 50)
    
    data_patterns = ["*.csv", "df_*.csv", "feat_importance_*.csv", "*.xlsx"]
    data_files = []
    for pattern in data_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.exists(match) and os.path.getsize(match) > 1024*1024:  # > 1MB
                data_files.append(match)
    
    # Show largest data files
    data_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    for file in data_files[:10]:  # Top 10 largest
        size_mb = os.path.getsize(file) / (1024*1024)
        print(f"   {file} ({size_mb:.1f} MB)")
    
    print(f"\n🧹 CLEANUP COMMANDS:")
    print("=" * 50)
    
    if found_files:
        print("# Delete superseded/deprecated files:")
        for file in found_files:
            print(f"rm '{file}'")
    
    if archive_files:
        print("\n# Move valuable files to archive:")
        for file in archive_files:
            print(f"mv '{file}' archive/research/")
    
    print(f"\n📋 SUMMARY:")
    print(f"   • {len(found_files)} files safe to delete")
    print(f"   • {len(archive_files)} files to consider archiving")
    print(f"   • {len(data_files)} large data files to review")
    
    return found_files, archive_files, data_files

def main():
    print("🧹 CLEANUP ASSISTANT - Remaining Files Analysis")
    print("=" * 60)
    
    find_files_to_delete()
    
    print(f"\n RECOMMENDATIONS:")
    print("1. Delete the superseded files (they're now in src/)")
    print("2. Move valuable research files to archive/")
    print("3. Review large data files - keep only essential ones")
    print("4. Commit the cleanup: git add -A && git commit -m 'cleanup: Remove superseded files'")

if __name__ == "__main__":
    main()