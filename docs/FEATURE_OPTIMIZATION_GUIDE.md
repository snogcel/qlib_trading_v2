# Feature Optimization Guide

## Overview

This comprehensive feature optimization suite helps you systematically test and optimize different feature configurations and model parameters to maximize your quantile trading strategy performance.

## Quick Start

### 1. Quick Feature Test (Recommended First Step)
```bash
python run_feature_optimization.py --mode quick
```
This runs the 3 most promising feature configurations:
- Enhanced volatility features
- Multi-timeframe momentum
- Feature interactions

**Expected runtime:** 5-10 minutes  
**Output:** Quick comparison of performance metrics

### 2. Targeted Optimization
```bash
python run_feature_optimization.py --mode optimize --config enhanced_volatility
```
Deep optimization of the best performing configuration from quick test.

**Expected runtime:** 30-60 minutes  
**Output:** Fully optimized configuration ready for production

### 3. Feature Importance Analysis
```bash
python run_feature_optimization.py --mode analyze
```
Analyzes which features are most consistently important across configurations.

**Expected runtime:** 10-15 minutes  
**Output:** Feature importance rankings and consistency scores

### 4. Full Experiment Suite
```bash
python run_feature_optimization.py --mode full --trials 200
```
Runs all feature experiments + comprehensive hyperparameter optimization.

**Expected runtime:** 2-4 hours  
**Output:** Complete optimization results

## Feature Experiment Types

### 1. **Enhanced Volatility** (`enhanced_volatility`)
- **New Features:** Volatility regime detection, breakout signals, GARCH-like volatility
- **Expected Improvement:** Better volatility regime detection and risk management
- **Best For:** Volatile market conditions

### 2. **Multi-Timeframe Momentum** (`multi_timeframe_momentum`)
- **New Features:** 1h, 3h, 5h, 10h, 1d, 2d, 1w momentum + acceleration/divergence
- **Expected Improvement:** Better trend detection across timeframes
- **Best For:** Trending markets

### 3. **Enhanced GDELT** (`enhanced_gdelt`)
- **New Features:** Sentiment moving averages, volatility, event intensity, news flow
- **Expected Improvement:** Better sentiment signal processing
- **Best For:** News-driven market movements

### 4. **Microstructure** (`microstructure`)
- **New Features:** Order book features, trade intensity, liquidity scores
- **Expected Improvement:** Better short-term price prediction
- **Best For:** High-frequency trading scenarios

### 5. **Feature Interactions** (`feature_interactions`)
- **New Features:** Momentum×volatility, sentiment×momentum, regime-aware features
- **Expected Improvement:** Capture non-linear relationships
- **Best For:** Complex market dynamics

### 6. **Alternative Quantiles** (`alternative_quantiles`)
- **New Features:** 5-quantile system (Q05, Q25, Q50, Q75, Q95)
- **Expected Improvement:** Better uncertainty quantification
- **Best For:** Risk-sensitive applications

## ⚙️ Hyperparameter Optimization

The system optimizes three levels of parameters:

### 1. **Feature Engineering Parameters**
- Momentum windows
- Volatility regime thresholds
- Sentiment processing methods
- Feature interaction terms

### 2. **Signal Generation Parameters**
- Signal threshold percentiles
- Spread threshold percentiles
- Rolling window sizes
- Tier confidence calculations

### 3. **Backtesting Parameters**
- Position limits
- Fee rates and slippage
- Long/short thresholds
- Risk management settings

## 📈 Performance Metrics

Each experiment tracks:
- **Total Return:** Absolute performance
- **Sharpe Ratio:** Risk-adjusted return
- **Maximum Drawdown:** Risk control
- **Win Rate:** Trade success rate
- **Total Trades:** Strategy activity level

## 🎯 Optimization Workflow

### Recommended Sequence:

1. **Quick Test** → Identify best feature group
2. **Feature Analysis** → Understand feature importance
3. **Targeted Optimization** → Deep dive on best configuration
4. **Production Deployment** → Use optimized parameters

### Example Workflow:
```bash
# Step 1: Quick test
python run_feature_optimization.py --mode quick

# Step 2: Analyze features (parallel to step 3)
python run_feature_optimization.py --mode analyze

# Step 3: Optimize best configuration (e.g., enhanced_volatility)
python run_feature_optimization.py --mode optimize --config enhanced_volatility

# Step 4: Deploy optimized configuration
# Use the saved configuration file for production
```

## 📁 Output Files

### Results Structure:
```
./feature_experiments/          # Individual experiment results
./hyperparameter_optimization/  # Optimization results
./optimized_configurations/     # Final optimized configs
./feature_analysis/            # Feature importance analysis
```

### Key Output Files:
- `experiment_comparison.csv` - Performance comparison across all experiments
- `optimized_[config]_[timestamp].json` - Final optimized configuration
- `feature_importance_analysis_[timestamp].json` - Feature importance rankings

## 🔧 Customization

### Adding New Feature Experiments:
1. Edit `feature_experiment_suite.py`
2. Add new `FeatureConfig` in `define_feature_experiments()`
3. Implement feature engineering in `engineer_features()`

### Modifying Optimization Objectives:
1. Edit `hyperparameter_optimizer.py`
2. Modify the `objective()` function in optimization methods
3. Adjust scoring weights (e.g., Sharpe vs. return vs. drawdown)

## 🚨 Performance Expectations

Based on your baseline results:

### Current Baseline:
- **Return:** 8.96%
- **Sharpe:** 3.99
- **Max Drawdown:** 0.54%

### Expected Improvements:
- **Enhanced Volatility:** +10-20% Sharpe improvement
- **Multi-Momentum:** +15-25% return improvement
- **Feature Interactions:** +5-15% overall performance
- **Full Optimization:** +20-40% combined improvement

## 💡 Tips for Best Results

1. **Start with quick test** - Don't jump to full optimization
2. **Monitor feature importance** - Remove low-importance features
3. **Consider market regimes** - Some features work better in specific conditions
4. **Validate on out-of-sample data** - Use the test set for final validation
5. **Balance complexity vs. performance** - More features ≠ always better

## 🔄 Integration with Hummingbot

After optimization:
1. Use optimized thresholds in `hummingbot_bridge.py`
2. Update `realtime_predictor.py` with best features
3. Configure Hummingbot with optimized parameters
4. Monitor live performance vs. backtest results

## 📞 Troubleshooting

### Common Issues:
- **Memory errors:** Reduce number of trials or features
- **Slow performance:** Use quick mode first
- **Poor results:** Check data quality and feature engineering
- **Optimization fails:** Verify data format and required columns

### Debug Mode:
Add `--debug` flag to any command for verbose output and error details.