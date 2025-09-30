#!/usr/bin/env python3
"""
Project Cleanup Execution Script
Systematically organizes files according to PROJECT_CLEANUP_PLAN.md
"""

import os
import shutil
from pathlib import Path

def create_folder_structure():
    """Create the new organized folder structure"""
    folders = [
        # Core development
        "src/models", "src/features", "src/data", "src/backtesting", 
        "src/production", "src/rl_execution/components", "src/logging",
        
        # Tests
        "tests/unit/features", "tests/integration", "tests/validation",
        
        # Documentation
        "docs/api", "docs/guides", "docs/research", "docs/rl_execution",
        
        # Configuration
        "config/rl_execution",
        
        # Scripts
        "scripts/analysis", "scripts/deployment", "scripts/maintenance", 
        "scripts/data_management",
        
        # Archive
        "archive/research", "archive/analysis", "archive/validation",
        "archive/documentation", "archive/deprecated",
        
        # Data & Results (will be gitignored)
        "data/raw", "data/processed", "data/features",
        "results/backtests", "results/analysis", "results/models",
        "logs", "temp"
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {folder}")

def safe_move(src, dst):
    """Safely move file if source exists"""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        print(f"✓ Moved: {src} → {dst}")
        return True
    else:
        print(f"⚠ Not found: {src}")
        return False

def safe_delete(filepath):
    """Safely delete file if it exists"""
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑 Deleted: {filepath}")
        return True
    else:
        print(f"⚠ Not found: {filepath}")
        return False

def cleanup_analysis_files():
    """Move or archive analysis files"""
    analysis_moves = {
        # Analysis files to scripts/analysis/
        "analyze_volatility_features.py": "scripts/analysis/analyze_volatility_features.py",
        "analyze_signal_execution.py": "scripts/analysis/analyze_signal_execution.py", 
        "analyze_hold_patterns.py": "scripts/analysis/analyze_hold_patterns.py",
        "analyze_trading_frequency.py": "scripts/analysis/analyze_trading_frequency.py",
        "backtest_results_analysis.py": "scripts/analysis/backtest_results_analysis.py",
        "regime_feature_consolidation_analysis.py": "scripts/analysis/regime_consolidation_analysis.py",
        
        # Analysis files to archive/analysis/
        "magnitude_based_threshold_analysis.py": "archive/analysis/magnitude_based_threshold_analysis.py",
        "signal_threshold_analysis.py": "archive/analysis/signal_threshold_analysis.py",
        
        # Volatility analysis summaries to archive/analysis/
        "volatility_features_final_summary.py": "archive/analysis/volatility_features_final_summary.py",
        "volatility_features_corrected_analysis.py": "archive/analysis/volatility_features_corrected_analysis.py",
        "volatility_features_summary.py": "archive/analysis/volatility_features_summary.py",
    }
    
    for src, dst in analysis_moves.items():
        safe_move(src, dst)

def cleanup_test_files():
    """Organize test files"""
    test_moves = {
        # Validation tests
        "validate_adaptive_thresholds.py": "tests/validation/test_adaptive_thresholds.py",
        "validate_regime_consolidation_performance.py": "tests/validation/test_regime_consolidation.py",
        "validate_data_alignment.py": "tests/validation/test_data_alignment.py",
        
        # Unit tests (keep existing Tests/Features structure for now)
        "test_unified_regime_features.py": "tests/unit/test_unified_regime_features.py",
        "test_temporal_quantile_features.py": "tests/unit/test_temporal_quantile_features.py",
        "test_momentum_hybrid_features.py": "tests/unit/test_momentum_hybrid_features.py",
        
        # Integration tests
        "test_q50_integration.py": "tests/integration/test_q50_integration.py",
        
        # Archive old test files
        "test_magnitude_based_threshold.py": "archive/validation/test_magnitude_based_threshold.py",
        "test_vol_scaled_implementation.py": "archive/validation/test_vol_scaled_implementation.py",
        "test_sizing_methods.py": "archive/validation/test_sizing_methods.py",
        "test_backtester_fix.py": "archive/validation/test_backtester_fix.py",
        "test_24_7_trading.py": "archive/validation/test_24_7_trading.py",
        "test_signal_analysis_output.py": "archive/validation/test_signal_analysis_output.py",
        "test_hold_reason_fix.py": "archive/validation/test_hold_reason_fix.py",
        "test_position_sizing_methods.py": "archive/validation/test_position_sizing_methods.py",
        "test_position_management.py": "archive/validation/test_position_management.py",
        "test_regime_features_fix.py": "archive/validation/test_regime_features_fix.py",
        "test_fixed_spread_validation.py": "archive/validation/test_fixed_spread_validation.py",
    }
    
    for src, dst in test_moves.items():
        safe_move(src, dst)

def cleanup_research_files():
    """Archive research and development files"""
    research_moves = {
        # Research analysis to archive/research/
        "dqn_trading_research_analysis.md": "archive/research/dqn_trading_research_analysis.md",
        "quantile_deep_learning_research_analysis.md": "archive/research/quantile_deep_learning_research_analysis.md",
        "research_consolidation_final.md": "archive/research/research_consolidation_final.md",
        "phase1_temporal_features_summary.md": "archive/research/phase1_temporal_features_summary.md",
        
        # Feature research to archive/research/
        "regime_feature_recommendations.md": "archive/research/regime_feature_recommendations.md",
        "regime_features_fix_summary.py": "archive/research/regime_features_fix_summary.py",
        
        # Summary files to archive/documentation/
        "quantiles_to_probabilities_fix_summary.md": "archive/documentation/quantiles_to_probabilities_fix_summary.md",
        "regression_fixes_summary.md": "archive/documentation/regression_fixes_summary.md",
        "q50_regime_integration_summary.md": "archive/documentation/q50_regime_integration_summary.md",
    }
    
    for src, dst in research_moves.items():
        safe_move(src, dst)

def cleanup_utility_files():
    """Move utility and maintenance files"""
    utility_moves = {
        # Utility scripts to scripts/maintenance/
        "update_feature_removal.py": "scripts/maintenance/update_feature_removal.py",
        "update_imports.py": "scripts/maintenance/update_imports.py",
        "cleanup_remaining_files.py": "scripts/maintenance/cleanup_remaining_files.py",
        
        # Batch files to scripts/maintenance/
        "reorganize_files.bat": "scripts/maintenance/reorganize_files.bat",
        "git_commit_commands.bat": "scripts/maintenance/git_commit_commands.bat",
        "commit_reorganization.sh": "scripts/maintenance/commit_reorganization.sh",
        
        # Optimization scripts to scripts/analysis/
        "run_feature_optimization.py": "scripts/analysis/run_feature_optimization.py",
        "test_top_features.py": "scripts/analysis/test_top_features.py",
        "ppo_sweep_new_features.py": "scripts/analysis/ppo_sweep_new_features.py",
    }
    
    for src, dst in utility_moves.items():
        safe_move(src, dst)

def cleanup_core_files():
    """Move core system files to src/"""
    core_moves = {
        # Backtesting to src/backtesting/
        "quantile_backtester.py": "src/backtesting/quantile_backtester.py",
        "run_backtest.py": "src/backtesting/run_backtest.py",
        "backtest_summary.py": "src/backtesting/backtest_summary.py",
        
        # Model evaluation to src/models/
        "model_evaluation_suite.py": "src/models/model_evaluation_suite.py",
        
        # Production files to src/production/
        "integrated_validated_pipeline.py": "src/production/integrated_validated_pipeline.py",
        "validated_kelly_with_spread.py": "src/production/validated_kelly_with_spread.py",
        "regime_aware_kelly.py": "src/production/regime_aware_kelly.py",
    }
    
    for src, dst in core_moves.items():
        safe_move(src, dst)

def create_gitignore():
    """Create/update .gitignore for new structure"""
    gitignore_content = """
# Data directories
data/
results/
logs/
temp/

# Model files
*.pkl
*.joblib
*.h5
*.pt
*.pth

# Cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv
env/
venv/

# Logs
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    print("✓ Created/updated .gitignore")

def main():
    """Execute the cleanup process"""
    print("🧹 Starting Project Cleanup...")
    print("=" * 50)
    
    # Phase 1: Create folder structure
    print("\n📁 Phase 1: Creating folder structure...")
    create_folder_structure()
    
    # Phase 2: Move analysis files
    print("\nPhase 2: Organizing analysis files...")
    cleanup_analysis_files()
    
    # Phase 3: Move test files
    print("\nPhase 3: Organizing test files...")
    cleanup_test_files()
    
    # Phase 4: Archive research files
    print("\n🔬 Phase 4: Archiving research files...")
    cleanup_research_files()
    
    # Phase 5: Move utility files
    print("\nPhase 5: Organizing utility files...")
    cleanup_utility_files()
    
    # Phase 6: Move core files
    print("\n⚙️ Phase 6: Organizing core system files...")
    cleanup_core_files()
    
    # Phase 7: Create gitignore
    print("\n📝 Phase 7: Creating .gitignore...")
    create_gitignore()
    
    print("\nCleanup completed!")
    print("\n📋 Next steps:")
    print("1. Review moved files and verify functionality")
    print("2. Update import statements in moved files")
    print("3. Test critical functionality")
    print("4. Commit organized structure to Git")
    print("5. Delete empty directories if desired")

if __name__ == "__main__":
    main()