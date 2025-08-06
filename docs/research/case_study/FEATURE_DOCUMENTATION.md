# Trading System Feature Documentation

## Overview
This document tracks all features in the Q50-centric, variance-aware trading system. Each feature is documented with its purpose, implementation details, and usage patterns based on the current `src/` repository structure.

---

## 🎯 Core Signal Features

### Q50 (Primary Signal)
- **Type**: Quantile-based probability
- **Purpose**: Primary directional signal based on 50th percentile probability
- **Usage**: Standalone signal for trade direction decisions
- **Implementation**: `src/models/multi_quantile.py` - MultiQuantileModel class
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Primary driver of returns
- **Dependencies**: LightGBM quantile regression models (0.1, 0.5, 0.9 quantiles)

### Q50-Centric Signal Generation
- **Type**: Variance-aware directional signal system
- **Purpose**: Generate trading signals using Q50 with regime-aware thresholds
- **Usage**: Primary signal generation in training pipeline
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Core signal generation logic
- **Features**:
  - Economic significance testing (expected value vs transaction costs)
  - Variance-based regime identification
  - Enhanced information ratio calculation
  - Adaptive threshold scaling

### Signal Classification & Tiers
- **Type**: Signal quality classification system
- **Purpose**: Classify signal strength for position sizing and execution
- **Usage**: RL environment reward calculation and position sizing
- **Implementation**: `src/training_pipeline.py` - signal_classification() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Determines trade execution
- **Logic**: Threshold-based classification (0-3 tiers) using adaptive signal thresholds

---

## 📊 Risk & Volatility Features

### Vol_Risk (Variance-Based)
- **Type**: Volatility-based risk metric using variance (not standard deviation)
- **Purpose**: Superior risk assessment for position sizing and regime detection
- **Usage**: Core component in variance-based regime identification and position scaling
- **Implementation**: `src/data/crypto_loader.py` - crypto_dataloader_optimized.get_feature_config()
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Critical for risk-adjusted returns
- **Formula**: `Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)` (variance)
- **Dependencies**: 6-day rolling window volatility calculation

### Volatility Regime Detection
- **Type**: Multi-tier volatility classification system
- **Purpose**: Market regime detection using variance percentiles
- **Usage**: Adaptive threshold scaling and position sizing
- **Implementation**: `src/training_pipeline.py` - identify_market_regimes() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Enables regime-aware trading
- **Regimes**: 
  - Low variance (≤30th percentile): -30% threshold adjustment
  - High variance (70th-90th percentile): +40% threshold adjustment  
  - Extreme variance (>90th percentile): +80% threshold adjustment

### Enhanced Information Ratio
- **Type**: Variance-enhanced signal quality metric
- **Purpose**: Superior signal assessment combining market and prediction uncertainty
- **Usage**: Signal quality filtering and position sizing
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Improves signal selection
- **Formula**: `signal / sqrt(market_variance + prediction_variance)`

---

## 🎲 Position Sizing Features

### Enhanced Kelly Criterion (Regime-Aware)
- **Type**: Multi-factor position sizing with regime awareness
- **Purpose**: Optimal position sizing using Kelly + Vol + Sharpe + Risk Parity + Regime Multiplier
- **Usage**: Determines position size for each trade based on multiple risk factors and market regimes
- **Implementation**: `src/features/position_sizing.py` - AdvancedPositionSizer class
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Optimizes risk-adjusted returns with regime awareness
- **Logic**: 
  - Enhanced Kelly calculation with vol_risk as primary risk measure
  - Regime-based multipliers from RegimeFeatureEngine (0.1x-5.0x range)
  - Dynamic volatility adjustments based on market conditions
  - Integrated with sentiment and dominance regimes for comprehensive risk management

**Migration Note**: Replaces deprecated `kelly_with_vol_raw_deciles()` with more sophisticated regime-aware approach.

### Position Sizing Features Suite
- **Type**: Comprehensive position sizing system
- **Purpose**: Multiple position sizing approaches for different market conditions
- **Implementation**: `src/features/position_sizing.py`
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Core position management
- **Components**: Kelly criterion, risk parity, volatility scaling, regime adjustments

### Variance-Based Position Scaling
- **Type**: Inverse variance position sizing
- **Purpose**: Scale positions inversely to market variance for risk management
- **Usage**: Dynamic position scaling in volatile markets
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Risk-adjusted position sizing
- **Formula**: `base_size / max(vol_risk * 1000, 0.1)` with 0.01-0.5 limits

---

## 🔄 Regime & Market Features

### Unified Regime Feature Engine
- **Type**: Comprehensive market regime detection system
- **Purpose**: Consolidated regime classification across multiple dimensions
- **Usage**: Regime-aware trading decisions, adaptive thresholds, position scaling
- **Implementation**: `src/features/regime_features.py` - RegimeFeatureEngine class
- **Status**: ✅ Production Ready (Consolidated from 23+ scattered features)
- **Performance Impact**: High - Comprehensive regime awareness
- **Features**:
  - `regime_volatility`: categorical (ultra_low, low, medium, high, extreme)
  - `regime_sentiment`: categorical (extreme_fear, fear, neutral, greed, extreme_greed)
  - `regime_dominance`: categorical (btc_low, balanced, btc_high)
  - `regime_crisis`: binary crisis detection
  - `regime_opportunity`: binary contrarian opportunity detection
  - `regime_stability`: continuous [0,1] regime transition frequency
  - `regime_multiplier`: continuous [0.1,5.0] unified position scaling

### Market Regime Identification
- **Type**: Multi-dimensional regime classification
- **Purpose**: Identify market regimes using volatility and momentum features
- **Usage**: Regime interaction features for enhanced signal quality
- **Implementation**: `src/training_pipeline.py` - identify_market_regimes() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Enables regime-aware signal generation
- **Components**:
  - Volatility regimes (low/medium/high using vol_risk)
  - Momentum regimes (trending/ranging using vol_raw_momentum)
  - Combined regime classifications
  - Regime stability tracking

### Variance-Based Interaction Features
- **Type**: Regime-signal interaction features for model training
- **Purpose**: Capture regime-dependent signal behavior
- **Usage**: Enhanced model training with regime awareness
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Improves model predictive power
- **Features**:
  - `q50_x_low_variance`, `q50_x_high_variance`, `q50_x_extreme_variance`
  - `vol_risk_x_abs_q50`, `enhanced_info_ratio_x_trending`
  - `signal_to_variance_ratio`, `variance_adjusted_signal`

---

## 📈 Technical Features

### Crypto Technical Indicators
- **Type**: Optimized cryptocurrency technical analysis features
- **Purpose**: Price, volume, and momentum analysis for crypto markets
- **Usage**: Primary technical signal generation
- **Implementation**: `src/data/crypto_loader.py` - crypto_dataloader_optimized.get_feature_config()
- **Status**: ✅ Production Ready (Optimized - removed redundant features)
- **Performance Impact**: Critical - Core technical analysis
- **Features**:
  - Price features: OPEN1, HIGH, LOW, CLOSE, VWAP
  - Volume features: VOLUME1 (VOLUME2/3 removed due to correlation)
  - Volatility: vol_6 (vol_3/vol_9 removed due to correlation)
  - Technical indicators: ROC, STD, RSI, MACD, Bollinger Bands
  - Regime detection: regime_volatility, regime_sentiment, regime_dominance

### Quantile Spread Analysis
- **Type**: Quantile prediction uncertainty metrics
- **Purpose**: Assess prediction confidence and market uncertainty
- **Usage**: Signal quality assessment and position sizing
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Signal quality assessment
- **Formula**: `spread = q90 - q10` (prediction uncertainty range)

### Probability Calculations
- **Type**: Piecewise probability estimation from quantiles
- **Purpose**: Convert quantile predictions to directional probabilities
- **Usage**: Directional signal generation and confidence assessment
- **Implementation**: `src/training_pipeline.py` - prob_up_piecewise() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Core probability estimation
- **Logic**: Piecewise linear interpolation between quantiles to estimate P(return > 0)

---

## 🔧 Threshold & Control Features

### Magnitude-Based Economic Thresholds
- **Type**: Expected value-based trading threshold system
- **Purpose**: Ensure trades are economically significant after transaction costs
- **Usage**: Primary trade filtering mechanism
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Enables profitable trading frequency
- **Logic**:
  - Traditional: `abs_q50 > signal_thresh_adaptive`
  - Expected value: `expected_value > transaction_cost` (5 bps)
  - Combined: Expected value + minimum signal strength filter
- **Improvement**: ~40-60% more trading opportunities vs traditional thresholds

### Adaptive Regime-Aware Thresholds
- **Type**: Dynamic threshold system with regime adjustments
- **Purpose**: Adjust signal thresholds based on market volatility regimes
- **Usage**: Regime-sensitive trade entry/exit decisions
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Regime-optimized trading
- **Adjustments**:
  - Low variance: -30% threshold (more opportunities in stable markets)
  - High variance: +40% threshold (higher bar in volatile markets)
  - Extreme variance: +80% threshold (very conservative in chaos)
  - Trending markets: -10% threshold (momentum helps)

### Information Ratio Thresholds
- **Type**: Signal quality filtering system
- **Purpose**: Ensure signals meet minimum information ratio requirements
- **Usage**: Signal quality control and risk management
- **Implementation**: `src/training_pipeline.py` - q50_regime_aware_signals() function
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Signal quality assurance
- **Logic**: Enhanced info ratio must exceed regime-adjusted threshold (0.5-3.0 range)

---

## 📊 Data Pipeline Features

### GDELT Sentiment Loader
- **Type**: Optimized sentiment and market psychology data loader
- **Purpose**: Alternative data integration for sentiment analysis and regime detection
- **Usage**: Daily sentiment features for regime classification
- **Implementation**: `src/data/gdelt_loader.py` - gdelt_dataloader_optimized class
- **Status**: ✅ Production Ready (Optimized)
- **Performance Impact**: Medium - Essential for regime detection
- **Features**:
  - Fear & Greed Index: `$fg_index`, `fg_std_7d`, `fg_zscore_14d`
  - BTC Dominance: `$btc_dom`, `btc_std_7d`, `btc_zscore_14d`
  - Regime flags: `fg_extreme_greed`, `fg_extreme_fear`, `btc_dom_high`, `btc_dom_low`
- **Frequency**: Daily (merged with hourly crypto data via nested loader)

### Crypto Market Data Loader
- **Type**: Optimized cryptocurrency market data loader
- **Purpose**: Efficient crypto market data loading with correlation-based feature selection
- **Usage**: Primary data source for crypto trading (60min frequency)
- **Implementation**: `src/data/crypto_loader.py` - crypto_dataloader_optimized class
- **Status**: ✅ Production Ready (Optimized)
- **Performance Impact**: Critical - Data foundation
- **Optimizations**:
  - Removed highly correlated volatility measures (vol_3, vol_9)
  - Kept only OPEN1, VOLUME1 (removed OPEN2/3, VOLUME2/3)
  - Added essential regime detection features
  - Focused on validated predictive features

### Nested Data Loader
- **Type**: Multi-frequency data integration system
- **Purpose**: Combine 60min crypto data with daily sentiment data
- **Usage**: Unified data pipeline for training and inference
- **Implementation**: `src/data/nested_data_loader.py` - CustomNestedDataLoader class
- **Status**: ✅ Production Ready
- **Performance Impact**: Critical - Data integration backbone
- **Features**:
  - Left join of crypto (60min) and GDELT (daily) data
  - Merge_asof for time alignment
  - Handles different frequencies and instruments
  - Configurable join strategies

---

## 🤖 Reinforcement Learning Features

### Signal Environment
- **Type**: RL trading environment for signal-based trading
- **Purpose**: Train RL agents on Q50 signals with regime awareness
- **Usage**: RL agent training and evaluation
- **Implementation**: `src/models/signal_environment.py` - SignalEnv class
- **Status**: 🧪 Experimental
- **Performance Impact**: High - RL training foundation
- **Features**: Tier-weighted rewards, volatility-aware entropy, regime-based position scaling

### RL Execution Suite
- **Type**: Comprehensive RL trading execution system
- **Purpose**: RL trading system with custom components
- **Usage**: RL trading execution and backtesting
- **Implementation**: `src/rl_execution/` directory
- **Status**: 🧪 Experimental
- **Performance Impact**: Critical - RL trading system
- **Components**:
  - Custom action/state interpreters
  - Order execution environment
  - Reward calculation system
  - Training vessel and meta wrapper
  - Tier-based logging and evaluation

### Entropy-Aware PPO
- **Type**: Volatility-adaptive PPO algorithm
- **Purpose**: Adjust exploration based on market volatility
- **Usage**: RL agent training with regime awareness
- **Implementation**: `src/training_pipeline.py` - EntropyAwarePPO class
- **Status**: ✅ Production Ready
- **Performance Impact**: High - Adaptive RL training
- **Logic**: Higher entropy in low-volatility regimes, lower in high-volatility

---

## ✅ Completed Consolidations & Optimizations

### 🎉 Data Pipeline Optimization - COMPLETED

**Achievement**: Streamlined data loading with correlation-based feature selection

#### ✅ Crypto Loader Optimizations:
- **Removed redundant features**: vol_3, vol_9 (kept vol_6)
- **Simplified volume features**: OPEN1, VOLUME1 only
- **Enhanced regime detection**: Added unified regime classification system (volatility, sentiment, dominance)
- **Performance gain**: ~40% reduction in feature count, maintained predictive power

#### ✅ GDELT Loader Optimizations:
- **Core sentiment indicators**: Fear & Greed Index, BTC Dominance
- **Regime detection flags**: Extreme sentiment and dominance levels
- **Statistical features**: 7-day volatility, 14-day z-scores
- **Clean integration**: Daily frequency merged with hourly crypto data

### 🎉 Regime Feature Consolidation - COMPLETED

**Achievement**: Successfully consolidated 23+ scattered regime features into unified system

#### ✅ Implemented Unified Approach:
- **Variance-based regimes**: Using vol_risk (variance) instead of std dev
- **Multi-dimensional classification**: Volatility, sentiment, dominance, crisis, opportunity
- **Adaptive thresholds**: Regime-aware signal thresholds (-30% to +80% adjustments)
- **Interaction features**: Regime-signal combinations for enhanced model training
- **Position scaling**: Unified regime multiplier system (0.1-5.0 range)

#### 🎯 Results Achieved:
- ✅ **Simplified architecture**: From scattered features to unified system
- ✅ **Enhanced performance**: Variance-based approach superior to std dev
- ✅ **Production ready**: Integrated into training pipeline and RL system
- ✅ **Validated approach**: Tested with 53,978+ samples, backward compatible

---

## 📋 Feature Status Legend

- ✅ **Production Ready**: Fully implemented, tested, and validated
- 🧪 **Experimental**: Implemented but needs validation
- 🔍 **Needs Review**: Requires analysis and documentation
- ❌ **Deprecated**: No longer used
- 🚧 **In Development**: Currently being built

---

## 📊 Performance Impact Scale

- **High**: Directly affects returns and risk metrics
- **Medium**: Enhances system performance but not critical
- **Low**: Optimization or supplementary features
- **Critical**: System cannot function without this feature

---

## � CurreSnt System Architecture

### Training Pipeline Flow
1. **Data Loading**: Crypto (60min) + GDELT (daily) via nested loader
2. **Feature Engineering**: Q50 signals + regime detection + variance analysis
3. **Model Training**: Multi-quantile LightGBM (0.1, 0.5, 0.9 quantiles)
4. **Signal Generation**: Q50-centric with regime-aware thresholds
5. **RL Training**: PPO agent with tier-weighted rewards and volatility adaptation

### Key File Structure
```
src/
├── data/
│   ├── crypto_loader.py          # Optimized crypto market data
│   ├── gdelt_loader.py           # Sentiment & regime data
│   └── nested_data_loader.py     # Multi-frequency integration
├── features/
│   ├── regime_features.py        # Unified regime detection
│   └── position_sizing.py        # Kelly & variance-based sizing
├── models/
│   ├── multi_quantile.py         # Q10/Q50/Q90 prediction
│   └── signal_environment.py     # RL trading environment
├── rl_execution/                 # Production RL trading system
└── training_pipeline.py          # Main training orchestration
```

---

## 🔄 Development Roadmap

### ✅ Completed (Current State)
- Multi-quantile model training and validation
- Variance-based regime detection system
- Q50-centric signal generation with adaptive thresholds
- RL environment with tier-weighted rewards
- Optimized data pipeline with correlation-based feature selection

### 🚧 In Progress
- Production deployment pipeline
- Real-time data integration
- Performance monitoring and alerting

### 📋 Planned Enhancements
- Multi-asset support beyond BTC
- Alternative data source integration
- Advanced regime detection algorithms
- Portfolio-level risk management

---

## 📝 Template for New Features

```markdown
### [Feature Name]
- **Type**: [Classification/category]
- **Purpose**: [What problem does it solve?]
- **Usage**: [How is it used in the system?]
- **Implementation**: [File/module location in src/]
- **Status**: [✅ Production Ready | 🧪 Experimental | 🚧 In Development]
- **Performance Impact**: [High/Medium/Low/Critical]
- **Dependencies**: [Other features it depends on]
- **Integration**: [How it connects to training pipeline]
- **Notes**: [Additional context or considerations]
```

---

## 📊 Performance Impact Scale

- **Critical**: System cannot function without this feature (data loaders, core models)
- **High**: Directly affects returns and risk metrics (Q50 signals, regime detection)
- **Medium**: Enhances system performance but not essential (optimization features)
- **Low**: Supplementary or debugging features (logging, visualization)

---

*Last Updated: February 2025*
*Repository Structure: src/ based architecture*
*Maintainer: Trading System Development Team*