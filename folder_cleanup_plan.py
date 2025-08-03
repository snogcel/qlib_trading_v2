#!/usr/bin/env python3
"""
Root Folder Cleanup Plan
Reorganizes and removes unnecessary root directories
"""

import os
import shutil
from pathlib import Path

def safe_move_folder(src, dst):
    """Safely move folder contents if source exists"""
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        try:
            # Move contents, not the folder itself
            for item in os.listdir(src):
                src_item = os.path.join(src, item)
                dst_item = os.path.join(dst, item)
                if os.path.exists(dst_item):
                    print(f"⚠ Conflict: {dst_item} already exists, skipping {src_item}")
                else:
                    shutil.move(src_item, dst_item)
                    print(f"✓ Moved: {src_item} → {dst_item}")
            # Remove empty source folder
            os.rmdir(src)
            print(f"✓ Removed empty: {src}")
            return True
        except Exception as e:
            print(f"❌ Error moving {src}: {e}")
            return False
    else:
        print(f"⚠ Not found: {src}")
        return False

def safe_delete_folder(folder_path):
    """Safely delete folder if it exists"""
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"🗑 Deleted: {folder_path}")
            return True
        except Exception as e:
            print(f"❌ Error deleting {folder_path}: {e}")
            return False
    else:
        print(f"⚠ Not found: {folder_path}")
        return False

def create_results_structure():
    """Create organized results folder structure"""
    results_folders = [
        "results/backtests/hummingbot",
        "results/backtests/validated", 
        "results/backtests/historical",
        "results/visualizations",
        "results/models",
        "results/analysis"
    ]
    
    for folder in results_folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")

def reorganize_results():
    """Reorganize results-related folders"""
    print("\n📊 Reorganizing results folders...")
    
    # Move backtest results
    safe_move_folder("backtest_results", "results/backtests/historical")
    safe_move_folder("hummingbot_backtest_results", "results/backtests/hummingbot")
    safe_move_folder("validated_backtest_results", "results/backtests/validated")
    
    # Move visualizations
    safe_move_folder("charts", "results/visualizations")
    safe_move_folder("plots", "results/visualizations")
    
    # Move model artifacts
    safe_move_folder("checkpoints", "results/models")
    safe_move_folder("models", "results/models")
    
    # Move general outputs
    safe_move_folder("outputs", "results/analysis")

def reorganize_data():
    """Reorganize data folders"""
    print("\n💾 Reorganizing data folders...")
    
    # Move CSV data to appropriate data subfolder
    safe_move_folder("csv_data", "data/processed")
    
    # Consolidate data3 into main data structure
    if os.path.exists("data3"):
        print("📁 Found data3 folder - manual review recommended")
        print("   Consider consolidating into main data/ structure")

def cleanup_old_versions():
    """Remove old versioned folders"""
    print("\n🧹 Cleaning up old versioned folders...")
    
    old_folders = [
        "logs_momentum_v2",
        "logs_momentum_v3", 
        "logs_v2",
        "logs_v3",
        "data_backup",
        "data_backup2",
        "data2_archive",
        "data3_archive"
    ]
    
    for folder in old_folders:
        safe_delete_folder(folder)

def cleanup_experimental():
    """Remove experimental/completed analysis folders"""
    print("\n🔬 Cleaning up experimental folders...")
    
    experimental_folders = [
        "feature_experiments",
        "feature_studies", 
        "comprehensive_feature_analysis",
        "v3_analysis",
        "test_hold_reason_fix_results",
        "test_signal_analysis_results"
    ]
    
    for folder in experimental_folders:
        safe_delete_folder(folder)

def cleanup_mlflow():
    """Clean up MLflow artifacts (can be regenerated)"""
    print("\n📈 Cleaning up MLflow artifacts...")
    
    mlflow_folders = ["mlruns", "runs"]
    
    for folder in mlflow_folders:
        if os.path.exists(folder):
            response = input(f"Delete {folder}? MLflow data can be regenerated (y/n): ")
            if response.lower() == 'y':
                safe_delete_folder(folder)
            else:
                print(f"⏭ Skipped: {folder}")

def analyze_remaining():
    """Analyze folders that need manual review"""
    print("\n🔍 Folders requiring manual review:")
    
    manual_review = [
        ("portfolio_analysis", "Check if analysis is complete, move key results to results/"),
        ("Projects", "Review if contains related projects worth keeping"),
        ("examples", "Keep if contains useful examples, otherwise archive"),
        ("data3", "Consolidate into main data/ structure")
    ]
    
    for folder, recommendation in manual_review:
        if os.path.exists(folder):
            print(f"  📁 {folder}: {recommendation}")

def main():
    """Execute folder cleanup"""
    print("🗂️ ROOT FOLDER CLEANUP")
    print("=" * 50)
    
    # Create organized structure first
    print("\n📁 Creating organized folder structure...")
    create_results_structure()
    
    # Reorganize results
    reorganize_results()
    
    # Reorganize data
    reorganize_data()
    
    # Clean up old versions
    cleanup_old_versions()
    
    # Clean up experimental folders
    cleanup_experimental()
    
    # Clean up MLflow (with confirmation)
    cleanup_mlflow()
    
    # Analyze remaining folders
    analyze_remaining()
    
    print("\n✅ Folder cleanup completed!")
    print("\n📋 Next steps:")
    print("1. Review manually flagged folders")
    print("2. Test that moved results are accessible")
    print("3. Update any scripts that reference old folder paths")
    print("4. Commit the organized structure")

if __name__ == "__main__":
    main()