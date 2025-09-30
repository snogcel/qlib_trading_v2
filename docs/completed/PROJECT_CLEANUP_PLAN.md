# Project Cleanup & Organization Plan

## 🎯 Overview
Organize the project for clean development, efficient Git management, and future scalability.

---

## 🗑️ Files to DELETE (Safe to Remove)

### Superseded/Deprecated Files
```bash
# Old versions superseded by optimized versions
qlib_custom/crypto_loader.py                    # → crypto_loader_optimized.py
qlib_custom/gdelt_loader.py                     # → gdelt_loader_optimized.py
ppo_sweep_optuna.py                             # → ppo_sweep_optuna_tuned_v2.py

# Debugging scripts (served their purpose)
debug_hold_signals.py
debug_short_buy_issue.py
debug_volatility_calculation.py

# Quick test scripts (ad-hoc, not systematic)
quick_prob_up_test.py
simple_feature_test.py
minimal_test.py
test_data_load.py

# Redundant analysis scripts
feature_correlation_test.py                     # → comprehensive_feature_analysis.py
test_momentum_comparison.py                     # → momentum_hybrid_final_summary.py
```

### Experimental/Exploratory Files (No Longer Needed)
```bash
# Feature exploration (consolidated into regime_features.py)
advanced_regime_features.py                     # → qlib_custom/regime_features.py
robust_regime_features.py                       # → qlib_custom/regime_features.py
regime_multipliers_implementation.py            # → qlib_custom/regime_features.py

# Volatility analysis (conclusions documented)
vol_raw_momentum_analysis.py
volatility_window_analysis.py
recalculate_vol_deciles.py

# Signal analysis experiments
test_prob_up_fix.py
test_quantiles_to_probabilities.py

# Validation experiments (replaced by systematic validation)
test_vol_risk_variance_integration.py
validate_ppo_alignment.py
```

---

## 📦 Files to ARCHIVE (Move to archive/ folder)

### Research & Development History
```bash
# Create: archive/research/
vol_momentum_hybrid_implementation.py           # Valuable research insights
vol_risk_strategic_implementation.py            # Strategic implementation notes
q50_regime_implementation.py                    # Q50 development history
q50_centric_implementation.py                   # Implementation evolution

# Create: archive/analysis/
comprehensive_feature_analysis.py               # Comprehensive analysis results
feature_optimization_summary.md                 # Optimization insights
volatility_features_impact_assessment.py        # Impact assessment
regime_features_usage_analysis.py               # Usage analysis

# Create: archive/validation/
spread_validation_results_analysis.py           # Validation results
test_validated_kelly_backtest.py               # Kelly validation
validate_spread_predictive_power.py            # Predictive power analysis
```

## 🏗️ PRESERVE & ORGANIZE (RL Order Execution Components)

### RL Order Execution Infrastructure (KEEP - High Value)
```bash
# Create: src/rl_execution/ (move from rl_order_execution/)
rl_order_execution/train_meta_wrapper.py        # → src/rl_execution/meta_training.py
rl_order_execution/troubleshoot_dataloader.py   # → src/data/dataloader_diagnostics.py

# Custom RL Components (move to src/rl_execution/components/)
rl_order_execution/custom_action_interpreter.py
rl_order_execution/custom_data_handler.py
rl_order_execution/custom_data_provider.py
rl_order_execution/custom_logger_callback.py    # → src/logging/tensorboard_logger.py
rl_order_execution/custom_order.py
rl_order_execution/custom_reward.py
rl_order_execution/custom_simulator.py
rl_order_execution/custom_state_interpreter.py
rl_order_execution/custom_tier_logging.py       # → Already in qlib_custom/
rl_order_execution/custom_train.py
rl_order_execution/custom_training_vessel.py

# Configuration Templates (move to config/rl_execution/)
rl_order_execution/exp_configs/train_ppo.yml    # → config/rl_execution/train_ppo.yml
rl_order_execution/README.md                    # → docs/rl_execution/README.md
```

### Data Infrastructure (CRITICAL - Keep & Organize)
```bash
# Create: scripts/data_management/
qlib_data_import.txt                            # → scripts/data_management/import_commands.sh
# Add data validation scripts
# Add data pipeline monitoring
```

### Documentation & Summaries
```bash
# Create: archive/documentation/
q50_implementation_complete.md                  # Complete implementation guide
magnitude_based_threshold_summary.md            # Threshold strategy docs
vol_risk_variance_integration_summary.md        # Integration summary
vol_scaled_implementation_summary.md            # Implementation summary
momentum_hybrid_final_summary.py                # Momentum analysis summary
regime_features_final_recommendation.py         # Final recommendations
```

---

## 📁 New Folder Structure to CREATE

### Core Development Folders
```bash
mkdir -p src/                                   # Main source code
mkdir -p src/models/                            # Model implementations
mkdir -p src/features/                          # Feature engineering
mkdir -p src/data/                              # Data processing
mkdir -p src/backtesting/                       # Backtesting components
mkdir -p src/production/                        # Production deployment
mkdir -p src/rl_execution/                      # RL order execution system
mkdir -p src/rl_execution/components/           # Custom RL components
mkdir -p src/logging/                           # Logging infrastructure

mkdir -p tests/                                 # Organized test suite
mkdir -p tests/unit/                            # Unit tests
mkdir -p tests/integration/                     # Integration tests
mkdir -p tests/validation/                      # Validation tests

mkdir -p docs/                                  # Documentation
mkdir -p docs/api/                              # API documentation
mkdir -p docs/guides/                           # User guides
mkdir -p docs/research/                         # Research notes
mkdir -p docs/rl_execution/                     # RL execution documentation

mkdir -p config/                                # Configuration files
mkdir -p config/rl_execution/                   # RL execution configs

mkdir -p scripts/                               # Utility scripts
mkdir -p scripts/analysis/                      # Analysis scripts
mkdir -p scripts/deployment/                    # Deployment scripts
mkdir -p scripts/maintenance/                   # Maintenance scripts
mkdir -p scripts/data_management/               # Data import/export scripts
```

### Archive Folders
```bash
mkdir -p archive/                               # Historical files
mkdir -p archive/research/                      # Research & experiments
mkdir -p archive/analysis/                      # Analysis results
mkdir -p archive/validation/                    # Validation studies
mkdir -p archive/documentation/                 # Old documentation
mkdir -p archive/deprecated/                    # Deprecated code
```

### Data & Results Folders
```bash
mkdir -p data/                                  # Data files (gitignored)
mkdir -p data/raw/                              # Raw data
mkdir -p data/processed/                        # Processed data
mkdir -p data/features/                         # Feature data

mkdir -p results/                               # Results (gitignored)
mkdir -p results/backtests/                     # Backtest results
mkdir -p results/analysis/                      # Analysis results
mkdir -p results/models/                        # Trained models

mkdir -p logs/                                  # Log files (gitignored)
mkdir -p temp/                                  # Temporary files (gitignored)
```

---

## 🔄 File REORGANIZATION Plan

### Move to src/ Structure
```bash
# Core system files
mv ppo_sweep_optuna_tuned_v2.py → src/training_pipeline.py
mv hummingbot_backtester.py → src/backtesting/backtester.py
mv save_model_for_production.py → src/production/model_persistence.py
mv run_hummingbot_backtest.py → src/backtesting/run_backtest.py

# Feature engineering
mv qlib_custom/regime_features.py → src/features/regime_features.py
mv advanced_position_sizing.py → src/features/position_sizing.py

# Data processing
mv qlib_custom/crypto_loader_optimized.py → src/data/crypto_loader.py
mv qlib_custom/gdelt_loader_optimized.py → src/data/gdelt_loader.py
mv qlib_custom/custom_ndl.py → src/data/nested_data_loader.py

# Models
mv qlib_custom/custom_multi_quantile.py → src/models/multi_quantile.py
mv qlib_custom/custom_signal_env.py → src/models/signal_environment.py

# Production
mv realtime_predictor.py → src/production/realtime_predictor.py
mv hummingbot_bridge.py → src/production/hummingbot_bridge.py
mv setup_mqtt.py → src/production/mqtt_setup.py

# RL Execution System
mv rl_order_execution/train_meta_wrapper.py → src/rl_execution/meta_training.py
mv rl_order_execution/troubleshoot_dataloader.py → src/data/dataloader_diagnostics.py
mv rl_order_execution/custom_logger_callback.py → src/logging/tensorboard_logger.py

# RL Components
mv rl_order_execution/custom_*.py → src/rl_execution/components/
mv rl_order_execution/exp_configs/ → config/rl_execution/
mv rl_order_execution/README.md → docs/rl_execution/README.md

# Data Management
mv qlib_data_import.txt → scripts/data_management/import_commands.sh
```

### Move Tests to tests/ Structure
```bash
mv Tests/Features/ → tests/unit/features/
mv test_unified_regime_features.py → tests/integration/test_regime_features.py
mv validate_regime_consolidation_performance.py → tests/validation/test_performance.py
mv simple_regime_validation.py → tests/unit/test_regime_validation.py
```

### Move Analysis Scripts
```bash
mv analyze_signal_features.py → scripts/analysis/analyze_signals.py
mv regime_feature_consolidation_analysis.py → scripts/analysis/analyze_regime_consolidation.py
mv model_evaluation_suite.py → scripts/analysis/evaluate_models.py
mv backtest_results_analysis.py → scripts/analysis/analyze_backtests.py
```

---

## 📋 CLEANUP EXECUTION PLAN

### Phase 1: Safe Deletions (Week 1)
```bash
# Delete superseded files
rm qlib_custom/crypto_loader.py
rm qlib_custom/gdelt_loader.py  
rm ppo_sweep_optuna.py

# Delete debugging scripts
rm debug_*.py
rm quick_*.py
rm simple_*.py
rm minimal_*.py
```

### Phase 2: Archive Creation (Week 1)
```bash
# Create archive structure
mkdir -p archive/{research,analysis,validation,documentation,deprecated}

# Move research files
mv vol_momentum_hybrid_implementation.py archive/research/
mv vol_risk_strategic_implementation.py archive/research/
mv q50_regime_implementation.py archive/research/
# ... (continue with archive list)
```

### Phase 3: Folder Structure (Week 2)
```bash
# Create new folder structure
mkdir -p src/{models,features,data,backtesting,production}
mkdir -p tests/{unit,integration,validation}
mkdir -p docs/{api,guides,research}
mkdir -p scripts/{analysis,deployment,maintenance}
```

### Phase 4: File Reorganization (Week 2)
```bash
# Move core files to new structure
# (Execute moves as listed above)
```

### Phase 5: Update Imports & Documentation (Week 3)
```bash
# Update all import statements
# Update documentation paths
# Update configuration files
# Test all functionality
```

---

## RL Order Execution System Integration

### Key Components to Preserve
```bash
# Multi-step Processing Pipeline
train_meta_wrapper.py                          # 24h predictions → ML model pipeline
troubleshoot_dataloader.py                     # Critical data import bug fixes

# Custom RL Infrastructure
custom_action_interpreter.py                   # Action space management
custom_data_handler.py                         # Data pipeline integration
custom_data_provider.py                        # Real-time data serving
custom_logger_callback.py                      # TensorBoard integration
custom_order.py                                # Order management system
custom_reward.py                               # Reward function design
custom_simulator.py                            # Trading simulation
custom_state_interpreter.py                    # State space management
custom_tier_logging.py                         # Tier-based logging
custom_train.py                                # Training orchestration
custom_training_vessel.py                      # Training container
```

### Future Integration Opportunities
```bash
# QLib Server Integration (https://qlib.readthedocs.io/en/stable/advanced/server.html)
- Real-time data serving capabilities
- Distributed model serving
- Live prediction infrastructure
- Multi-client data distribution

# YML Configuration System
- Declarative experiment configuration
- Reproducible training setups
- Easy hyperparameter management
- Production deployment configs

# TensorBoard Integration
- Real-time training monitoring
- Performance visualization
- Model comparison dashboards
- Production metrics tracking
```

### Data Import Critical Notes
```bash
# CRITICAL: Data Import Bug Prevention
# From qlib_data_import.txt - specific import order required
# Incorrect CSV import can serve wrong data silently!

# Required import sequence:
1. BTCUSDT 60min data
2. BTCUSDT daily data  
3. BTC_FEAT daily data (btc_dom, fg_index)
4. GDELT sentiment data (CWT features)

# Use dump_fix for corrections if needed
```

---

## 🎯 Benefits of This Organization

### Development Efficiency
- **Clear separation of concerns**: Core code vs tests vs analysis
- **Easy navigation**: Logical folder structure
- **Reduced clutter**: Only active development files visible

### Git Management
- **Cleaner commits**: Organized file structure
- **Better diffs**: Related files grouped together
- **Easier reviews**: Clear component boundaries

### Future Scalability
- **Team collaboration**: Clear ownership boundaries
- **CI/CD integration**: Organized test structure
- **Documentation**: Centralized docs folder

### Maintenance
- **Easier debugging**: Clear component isolation
- **Simpler updates**: Organized dependencies
- **Better testing**: Systematic test organization

---

## 🚨 Pre-Cleanup Checklist

### Backup Strategy
- [ ] Create full project backup
- [ ] Commit current state to Git
- [ ] Tag current version (e.g., `v1.0-pre-cleanup`)
- [ ] Document current working directory structure

### Validation Plan
- [ ] Test current system functionality
- [ ] Document all import dependencies
- [ ] Identify critical file relationships
- [ ] Plan import statement updates

### Risk Mitigation
- [ ] Keep archive folder until cleanup validated
- [ ] Maintain rollback capability
- [ ] Test incrementally (don't move everything at once)
- [ ] Update documentation as you go

---

*This cleanup will transform your project from a research environment into a professional, maintainable codebase ready for production deployment and team collaboration.*