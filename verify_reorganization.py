#!/usr/bin/env python3
"""
Verify that critical files weren't lost during reorganization
"""

import os
from pathlib import Path

# Critical files that should exist somewhere
CRITICAL_FILES = {
    # Core pipeline files
    "training_pipeline": ["src/training_pipeline.py", "ppo_sweep_optuna_tuned_v2.py"],
    "backtester": ["src/backtesting/backtester.py", "hummingbot_backtester.py"],
    "quantile_backtester": ["src/backtesting/quantile_backtester.py"],
    
    # Production files
    "realtime_predictor": ["src/production/realtime_predictor.py", "realtime_predictor.py"],
    "hummingbot_bridge": ["src/production/hummingbot_bridge.py", "hummingbot_bridge.py"],
    "model_persistence": ["src/production/model_persistence.py", "save_model_for_production.py"],
    
    # Data loaders
    "crypto_loader": ["src/data/crypto_loader.py", "qlib_custom/crypto_loader_optimized.py"],
    "gdelt_loader": ["src/data/gdelt_loader.py", "qlib_custom/gdelt_loader_optimized.py"],
    "nested_data_loader": ["src/data/nested_data_loader.py", "qlib_custom/custom_ndl.py"],
    
    # Models
    "multi_quantile": ["src/models/multi_quantile.py", "qlib_custom/custom_multi_quantile.py"],
    "signal_environment": ["src/models/signal_environment.py", "qlib_custom/custom_signal_env.py"],
    
    # Features
    "regime_features": ["src/features/regime_features.py", "qlib_custom/regime_features.py"],
    "position_sizing": ["src/features/position_sizing.py", "advanced_position_sizing.py"],
    
    # RL Execution
    "train_meta_wrapper": ["src/rl_execution/train_meta_wrapper.py", "train_meta_wrapper.py"],
    "custom_tier_logging": ["src/rl_execution/custom_tier_logging.py", "qlib_custom/custom_tier_logging.py"],
}

def check_file_exists(file_paths):
    """Check if any of the file paths exist"""
    for path in file_paths:
        if os.path.exists(path):
            return path
    return None

def main():
    print("🔍 REORGANIZATION VERIFICATION")
    print("=" * 50)
    
    missing_files = []
    found_files = []
    
    for component, paths in CRITICAL_FILES.items():
        found_path = check_file_exists(paths)
        if found_path:
            found_files.append((component, found_path))
            print(f"✅ {component}: {found_path}")
        else:
            missing_files.append((component, paths))
            print(f"❌ {component}: MISSING - Expected: {paths}")
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY:")
    print(f"   Found: {len(found_files)}")
    print(f"   Missing: {len(missing_files)}")
    
    if missing_files:
        print(f"\n⚠️  MISSING FILES DETECTED:")
        for component, paths in missing_files:
            print(f"   {component}: {paths}")
        print(f"\n🚨 RECOMMENDATION: Don't push yet - investigate missing files")
        return False
    else:
        print(f"\n✅ ALL CRITICAL FILES FOUND - Safe to push")
        return True

if __name__ == "__main__":
    main()