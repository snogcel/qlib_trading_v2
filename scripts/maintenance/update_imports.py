#!/usr/bin/env python3
"""
Update import statements after file reorganization
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update import statements in a single file"""
    
    # Import mapping for reorganized files
    import_mappings = {
        # Core system files
        'from src.training_pipeline': 'from src.training_pipeline',
        'import src.training_pipeline': 'import src.training_pipeline',
        'from src.backtesting.backtester': 'from src.backtesting.backtester',
        'import src.backtesting.backtester': 'import src.backtesting.backtester',
        
        # qlib_custom reorganization
        'from src.features.regime_features': 'from src.features.regime_features',
        'import src.features.regime_features': 'import src.features.regime_features',
        'from src.data.crypto_loader': 'from src.data.crypto_loader',
        'import src.data.crypto_loader': 'import src.data.crypto_loader',
        'from src.data.gdelt_loader': 'from src.data.gdelt_loader',
        'import src.data.gdelt_loader': 'import src.data.gdelt_loader',
        'from src.data.nested_data_loader': 'from src.data.nested_data_loader',
        'import src.data.nested_data_loader': 'import src.data.nested_data_loader',
        'from src.models.multi_quantile': 'from src.models.multi_quantile',
        'import src.models.multi_quantile': 'import src.models.multi_quantile',
        'from src.models.signal_environment': 'from src.models.signal_environment',
        'import src.models.signal_environment': 'import src.models.signal_environment',
        
        # Production files
        'from src.production.realtime_predictor': 'from src.production.realtime_predictor',
        'import src.production.realtime_predictor': 'import src.production.realtime_predictor',
        'from src.production.hummingbot_bridge': 'from src.production.hummingbot_bridge',
        'import src.production.hummingbot_bridge': 'import src.production.hummingbot_bridge',
        'from src.features.position_sizing': 'from src.features.position_sizing',
        'import src.features.position_sizing': 'import src.features.position_sizing',
        
        # RL execution
        'from src.rl_execution.meta_training': 'from src.rl_execution.meta_training',
        'import src.rl_execution.meta_training': 'import src.rl_execution.meta_training',
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply import mappings
        for old_import, new_import in import_mappings.items():
            content = content.replace(old_import, new_import)
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update imports in all Python files"""
    
    print("🔄 Updating import statements after reorganization...")
    
    # Directories to scan for Python files
    scan_dirs = [
        'src',
        'tests', 
        'Tests',  # Keep both for now
        'scripts',
        '.'  # Root directory
    ]
    
    updated_files = []
    
    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            continue
            
        for root, dirs, files in os.walk(scan_dir):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.conda']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if update_imports_in_file(file_path):
                        updated_files.append(file_path)
    
    print(f"Updated imports in {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"   • {file_path}")
    
    if not updated_files:
        print("No import updates needed - all files already use correct imports")
    
    print("\nTesting key imports...")
    
    # Test critical imports
    test_imports = [
        'src.training_pipeline',
        'src.backtesting.backtester', 
        'src.features.regime_features',
        'src.data.crypto_loader',
        'src.models.multi_quantile'
    ]
    
    for import_name in test_imports:
        try:
            __import__(import_name)
            print(f"   {import_name}")
        except ImportError as e:
            print(f"   {import_name}: {e}")
        except Exception as e:
            print(f"   ⚠️  {import_name}: {e}")
    
    print("\nImport update complete!")
    print("\nNext steps:")
    print("1. Test your main training pipeline: python -m src.training_pipeline")
    print("2. Test backtesting: python -m src.backtesting.run_backtest")
    print("3. Verify regime features: python -c 'from src.features.regime_features import RegimeFeatureEngine; print(\"OK\")'")

if __name__ == "__main__":
    main()