# Import Migration Tracker

## Overview
This document tracks all import path changes made during the project reorganization to ensure nothing gets lost and all connections can be restored.

**Date Created**: 2025-08-02  
**Status**: Active Migration in Progress  
**Priority**: Critical - System functionality depends on these imports

---

## 📦 Major Folder Migrations

### 1. **qlib_custom/ → src/ Structure**
**Status**: NEEDS IMPORT UPDATES

**Old Structure:**
```
qlib_custom/
├── regime_features.py
├── crypto_loader_optimized.py
├── gdelt_loader_optimized.py
├── custom_multi_quantile.py
├── custom_ndl.py
├── custom_signal_env.py
├── meta_trigger/
│   ├── meta_dqn_policy.py
│   ├── experience_buffer.py
│   └── train_meta_dqn.py
├── logger/
│   ├── tensorboard_logger.py
│   └── custom_logger_callback.py
└── [other custom components]
```

**New Structure:**
```
src/
├── features/
│   └── regime_features.py
├── data/
│   ├── crypto_loader_optimized.py
│   ├── gdelt_loader_optimized.py
│   └── custom_ndl.py
├── models/
│   ├── custom_multi_quantile.py
│   └── custom_signal_env.py
└── rl_execution/
    ├── components/
    │   ├── meta_trigger/
    │   │   ├── meta_dqn_policy.py
    │   │   ├── experience_buffer.py
    │   │   └── train_meta_dqn.py
    │   └── logger/
    │       ├── tensorboard_logger.py
    │       └── custom_logger_callback.py
    └── train_meta_wrapper.py
```

### 2. **csv_data/ → data/processed/**
**Status**: PATH UPDATED IN CLEANUP SCRIPT

**Migration:**
- `csv_data/BTC_FEAT/` → `data/processed/BTC_FEAT/`
- `csv_data/CRYPTODATA/` → `data/processed/CRYPTODATA/`
- `csv_data/GDELT_MERGED/` → `data/processed/GDELT_MERGED/`

---

## Critical Import Updates Needed

### **Primary Pipeline Flow** (HIGHEST PRIORITY)

**CURRENT PIPELINE FLOW (Single Model Focus):**

**1. Main Entry Point: `src/training_pipeline.py`**
- **Status**: LOCATED (formerly ppo_sweep_optuna_tuned_v2.py)
- **Function**: Primary entrance point, generates interim file in project root
- **Priority**: **CRITICAL** - This is the actual starting point

**2. Backtesting: `src/backtesting/run_backtest.py`**
- **Function**: Picks up interim file from training_pipeline.py, runs backtesting
- **Priority**: **HIGH** - Depends on training_pipeline.py output
- **Note**: Simplified single-model approach, focused on integration point

**3. RL Execution: `src/rl_execution/train_meta_wrapper.py`**
- **Status**: **FUTURE ITERATION** - Not currently in active pipeline
- **Function**: Advanced RL training (for later development)
- **Priority**: Medium - Future enhancement

**Current Broken Imports:**
```python
# OLD IMPORTS (BROKEN)
from qlib_custom.custom_order import Order
from qlib_custom.custom_simulator import CustomSingleAssetOrderExecutionSimple
from qlib_custom.meta_trigger.meta_dqn_policy import MetaDQNPolicy
from qlib_custom.meta_trigger.experience_buffer import ExperienceBuffer
from qlib_custom.meta_trigger.train_meta_dqn import train_meta_dqn_model
from qlib_custom.custom_logger_callback import MetaDQNCheckpointManager
from qlib_custom.logger.tensorboard_logger import TensorboardLogger
from qlib_custom.custom_train import CustomTrainer, backtest, train
```

**Required New Imports:**
```python
# NEW IMPORTS (TO IMPLEMENT)
from qlib_custom.custom_order import Order  # CHECK IF MOVED
from qlib_custom.custom_simulator import CustomSingleAssetOrderExecutionSimple  # CHECK IF MOVED
from src.rl_execution.components.meta_trigger.meta_dqn_policy import MetaDQNPolicy
from src.rl_execution.components.meta_trigger.experience_buffer import ExperienceBuffer
from src.rl_execution.components.meta_trigger.train_meta_dqn import train_meta_dqn_model
from src.rl_execution.components.logger.custom_logger_callback import MetaDQNCheckpointManager
from src.rl_execution.components.logger.tensorboard_logger import TensorboardLogger
from qlib_custom.custom_train import CustomTrainer, backtest, train  # CHECK IF MOVED
```

### **Other Files Likely Needing Updates:**

**1. Main Training Pipeline Files:**
- `ppo_sweep_optuna_tuned_v2.py` (if exists)
- `src/backtesting/run_backtest.py`
- `src/production/integrated_validated_pipeline.py`

**2. Feature Engineering Files:**
- Any files importing `from qlib_custom.regime_features import ...`
- Should become: `from src.features.regime_features import ...`

**3. Data Loading Files:**
- Any files importing crypto/GDELT loaders
- Should become: `from src.data.crypto_loader_optimized import ...`

**4. Model Files:**
- Any files importing custom models
- Should become: `from src.models.custom_multi_quantile import ...`

---

##  Files to Check for Import Issues

### **High Priority (Current Active Pipeline):**
- [ ] **`src/training_pipeline.py`** **CRITICAL** - Main entry point (formerly ppo_sweep_optuna_tuned_v2.py)
- [ ] **`src/backtesting/run_backtest.py`** **HIGH** - Depends on training_pipeline.py interim file output
- [ ] `src/backtesting/quantile_backtester.py` - Core backtesting engine
- [ ] `src/production/integrated_validated_pipeline.py` - Production integration
- [ ] `src/models/model_evaluation_suite.py` - Model evaluation

### **Medium Priority (Future Iterations):**
- [ ] `src/rl_execution/train_meta_wrapper.py` - Advanced RL training (future enhancement)
- [ ] Files in `tests/` directories
- [ ] Analysis scripts in `scripts/analysis/`

### **Medium Priority (Analysis/Scripts):**
- [ ] `scripts/analysis/run_feature_optimization.py`
- [ ] `scripts/analysis/ppo_sweep_new_features.py`
- [ ] Files in `tests/` directories

### **Low Priority (Archive/Historical):**
- [ ] Files in `archive/` (may have broken imports but not critical)

---

##  Potential Remaining qlib_custom Dependencies

**Files that might still be in qlib_custom/ and need migration:**

1. **custom_order.py** - Referenced in train_meta_wrapper.py
2. **custom_simulator.py** - Referenced in train_meta_wrapper.py  
3. **custom_train.py** - Referenced in train_meta_wrapper.py
4. **custom_data_handler.py** - Likely used in RL system
5. **custom_data_provider.py** - Likely used in RL system
6. **custom_action_interpreter.py** - RL component
7. **custom_state_interpreter.py** - RL component
8. **custom_reward.py** - RL component

**Action Required**: Check if these files still exist in `qlib_custom/` and determine their proper location in the new `src/` structure.

---

## 📋 Migration Checklist

### **Phase 1: Inventory (CURRENT)**
- [x] Document current broken imports
- [ ] Verify which files remain in `qlib_custom/`
- [ ] Identify all files that import from `qlib_custom/`

### **Phase 2: Locate Primary Entry Point**
- [x] **FOUND**: Main entry point is now `src/training_pipeline.py`
  - **Original name**: `ppo_sweep_optuna_tuned_v2.py` 
  - **New location**: `src/training_pipeline.py` (moved during cleanup per PROJECT_CLEANUP_PLAN.md)
  - **Status**: **LOCATED** - File exists and has recent cache files
- [ ] Verify it generates CSV that `train_meta_wrapper.py` consumes
- [ ] Document the complete pipeline flow

### **Phase 3: Move Remaining Files**
- [x] Move remaining `qlib_custom/` files to appropriate `src/` locations
- [x] **RL Order Execution Scripts**: Moved to `scripts/rl_order_execution/`
  - **ORGANIZED**: RL-specific scripts now properly grouped
  - **ACCESSIBLE**: `pickle_data_config.yml` and related utilities available
- [x] **Old Research Folder**: Moved to `scratch_data/research/`
  - **ROOT CLEANED**: Removed clutter from main directory
  - **PRESERVED**: Old research still accessible if needed
- [ ] Update `qlib_custom/` imports to new `src/` paths
- [ ] Test critical functionality after each move

### **Phase 4: Update Imports (Current Pipeline Priority)**
- [x] Fix **`src/training_pipeline.py`** imports (HIGHEST PRIORITY - MAIN ENTRY POINT)
  - **COMPLETED**: All imports fixed and working
  - **SUCCESSFUL RUN**: Pipeline completed with signal generation
  - **OUTPUT**: Generated 15,212 trading signals (28.2% of data)
  - **FILES**: Created `df_all_macro_analysis.csv` and `./data3/macro_features.pkl`
- [x] Fix **`src/backtesting/run_backtest.py`** imports (HIGH PRIORITY - DEPENDS ON INTERIM FILE FROM ABOVE)
  - **COMPLETED**: Files replaced from GitHub, imports working
  - **SUCCESSFUL RUN**: Backtesting completed with multiple configurations
  - **BEST PERFORMANCE**: Moderate config - 1.327 Sharpe, 17.48% return, -11.77% max drawdown
- [ ] Fix `src/backtesting/quantile_backtester.py` imports (CORE BACKTESTING)
- [ ] Fix `src/production/integrated_validated_pipeline.py` imports (PRODUCTION)
- [ ] Fix `src/rl_execution/train_meta_wrapper.py` imports (FUTURE ITERATION)
- [ ] Fix test file imports
- [ ] Fix analysis script imports

### **Phase 4: Validation**
- [ ] Test RL training pipeline
- [ ] Test main backtesting system
- [ ] Test production pipeline
- [ ] Run test suite

---

## Quick Fix Commands

**Find all files with qlib_custom imports:**
```bash
grep -r "from qlib_custom" . --include="*.py"
grep -r "import qlib_custom" . --include="*.py"
```

**Find all files with csv_data paths:**
```bash
grep -r "csv_data" . --include="*.py"
```

**Update csv_data paths (example):**
```bash
# Replace csv_data/ with data/processed/ in Python files
sed -i 's/csv_data\//data\/processed\//g' *.py
```

---

## Success Criteria

**Migration Complete When:**
- [ ] No broken imports in any active files
- [ ] `train_meta_wrapper.py` runs without import errors
- [ ] Main backtesting pipeline works
- [ ] RL training pipeline works
- [ ] All tests pass

**Files Can Be Safely Deleted:**
- [ ] `qlib_custom/` folder (after all files migrated)
- [ ] Any remaining empty directories

---

## 📞 Emergency Rollback Plan

**If imports become too complex:**
1. **Backup current state**: `git commit -m "WIP: Import migration"`
2. **Create symlinks**: Link new locations back to old import paths temporarily
3. **Gradual migration**: Move one component at a time instead of everything at once
4. **Test incrementally**: Ensure each component works before moving the next

**Temporary symlink example:**
```bash
# Create temporary symlink to maintain old import paths
ln -s src/features/regime_features.py qlib_custom/regime_features.py
```

---

**Last Updated**: 2025-08-02  
**Next Review**: After completing Phase 1 inventory  
**Owner**: Project reorganization effort