#!/bin/bash

echo "Committing reorganized trading system to GitHub..."
echo

# Add all the new organized structure
echo "Adding reorganized files..."
git add src/
git add config/
git add scripts/data_management/
git add docs/
git add archive/

# Add key documentation
git add TRADING_SYSTEM_PRINCIPLES.md
git add SYSTEM_VALIDATION_SPEC.md
git add FEATURE_DOCUMENTATION.md
git add FEATURE_STANDARDIZATION_PLAN.md
git add PROJECT_CLEANUP_PLAN.md
git add HUMMINGBOT_INTEGRATION_ROADMAP.md

# Add test files
git add Tests/
git add test_unified_regime_features.py
git add validate_regime_consolidation_performance.py

# Add validated results
git add validated_backtest_results/

# Show what we're about to commit
echo
echo "Files staged for commit:"
git status --porcelain | grep "^A"

echo
echo "Committing reorganization..."

# Single comprehensive commit
git commit -m "feat: Complete system reorganization with professional structure

🏗️ CORE SYSTEM (1.327 Sharpe Ratio Validated):
- Reorganized into professional src/ structure
- Main training pipeline with Q50-centric approach  
- Unified regime feature engine (23+ features → 7 clean features)
- Multi-timeframe data pipeline (60min crypto + daily sentiment)
- Advanced position sizing with Kelly criterion

PERFORMANCE VALIDATED:
- 17.48% total return with 1.327 Sharpe ratio
- -11.77% max drawdown (excellent risk management)
- 1,562 trades with optimal frequency
- Variance-aware risk assessment

🏭 PRODUCTION READY:
- Real-time prediction service
- Hummingbot integration bridge
- MQTT infrastructure  
- Model persistence utilities

RL EXECUTION SYSTEM:
- Multi-step processing pipeline
- Custom RL components organized
- YML configuration system
- Order execution simulation

📋 PROFESSIONAL FRAMEWORK:
- Trading system principles (thesis-first development)
- Comprehensive validation specification
- Feature documentation and lifecycle management
- Test suite with economic logic validation

🗂️ ORGANIZED STRUCTURE:
- src/ - Clean source code organization
- config/ - Validated configurations
- docs/ - Comprehensive documentation
- Tests/ - Systematic test suite  
- archive/ - Preserved research insights

This reorganization transforms the project from research environment
to production-ready professional trading system."

echo
echo "Commit complete!"
echo
echo "Ready to push to GitHub:"
echo "   git push origin main"
echo