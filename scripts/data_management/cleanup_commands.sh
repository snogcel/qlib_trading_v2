#!/bin/bash
# Generated cleanup commands
# Review before executing!

# Safe deletions
rm test_backtester_fix.py
rm test_backtest_fix.py
rm test_hold_reason_fix.py
rm test_regime_features_fix.py

# Archive research files
mkdir -p archive/research
mv test_vol_scaled_implementation.py archive/research/
mv dqn_trading_research_analysis.md archive/research/
mv quantile_deep_learning_research_analysis.md archive/research/
mv research_consolidation_final.md archive/research/
mv research_synthesis_and_implementation.md archive/research/

# Archive analysis files
mkdir -p archive/analysis
mv analyze_cleanup_candidates.py archive/analysis/
mv analyze_hold_patterns.py archive/analysis/
mv analyze_signal_execution.py archive/analysis/
mv analyze_trading_frequency.py archive/analysis/
mv analyze_volatility_features.py archive/analysis/
mv backtest_results_analysis.py archive/analysis/
mv magnitude_based_threshold_analysis.py archive/analysis/
mv regime_features_fix_summary.py archive/analysis/
mv regime_feature_consolidation_analysis.py archive/analysis/
mv signal_threshold_analysis.py archive/analysis/
mv volatility_features_corrected_analysis.py archive/analysis/
mv volatility_features_final_summary.py archive/analysis/
mv volatility_features_summary.py archive/analysis/
mv HUMMINGBOT_INTEGRATION_ROADMAP.md archive/analysis/
mv phase1_temporal_features_summary.md archive/analysis/
mv q50_regime_integration_summary.md archive/analysis/
mv quantiles_to_probabilities_fix_summary.md archive/analysis/
mv QUANTILE_ENHANCEMENT_ROADMAP.md archive/analysis/
mv regression_fixes_summary.md archive/analysis/
mv resource_mapping_analysis.md archive/analysis/
mv signal_analysis_csv_guide.md archive/analysis/

# Move to tests
mkdir -p tests/validation tests/unit
mv test_24_7_trading.py tests/unit/
mv test_comprehensive_backtest.py tests/unit/
mv test_feature_optimization.py tests/unit/
mv test_fixed_spread_validation.py tests/unit/
mv test_magnitude_based_threshold.py tests/unit/
mv test_momentum_hybrid_features.py tests/unit/
mv test_position_management.py tests/unit/
mv test_position_sizing_methods.py tests/unit/
mv test_q50_integration.py tests/unit/
mv test_signal_analysis_output.py tests/unit/
mv test_sizing_methods.py tests/unit/
mv test_temporal_quantile_features.py tests/unit/
mv test_top_features.py tests/unit/
mv test_unified_regime_features.py tests/unit/
mv validate_adaptive_thresholds.py tests/validation/
mv validate_data_alignment.py tests/validation/
mv validate_regime_consolidation_performance.py tests/validation/

# Move to scripts
mkdir -p scripts/analysis scripts/maintenance
mv cleanup_remaining_files.py scripts/maintenance/
mv ppo_sweep_new_features.py scripts/analysis/
mv run_backtest.py scripts/analysis/
mv run_complete_backtest_workflow.py scripts/analysis/
mv run_feature_optimization.py scripts/analysis/
mv update_feature_removal.py scripts/maintenance/
mv update_imports.py scripts/maintenance/
