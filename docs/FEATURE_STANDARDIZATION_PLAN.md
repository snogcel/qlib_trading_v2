# Feature Standardization Plan

## 🎯 Overview
Based on analysis of the codebase, several key features need standardization to ensure consistency and maintainability. This plan addresses the most critical standardization needs.

---

## 🚨 Priority 1: Signal Strength Standardization

### Current Issues
- **Multiple implementations**: 3+ different calculation methods
- **Inconsistent scaling**: Different scale factors across modules
- **Data anomaly**: Current data shows mean=0.0000, needs investigation

### Recommended Standard
```python
def calculate_signal_strength(q50, enhanced_info_ratio=1.0, signal_thresh=None):
    """
    Standardized signal strength calculation
    
    Args:
        q50: Primary quantile signal
        enhanced_info_ratio: Information ratio multiplier (default=1.0)
        signal_thresh: Optional threshold for normalization
    
    Returns:
        float: Standardized signal strength [0, inf)
    """
    base_strength = abs(q50)
    
    if signal_thresh is not None:
        # Threshold-normalized version
        return base_strength * enhanced_info_ratio / max(signal_thresh, 1e-6)
    else:
        # Direct version with info ratio enhancement
        return base_strength * enhanced_info_ratio

# Standardized bucketing
def get_strength_bucket(signal_strength):
    """Convert signal strength to categorical bucket"""
    if signal_strength < 0.3:
        return "Very_Low"
    elif signal_strength < 0.5:
        return "Low"
    elif signal_strength < 0.7:
        return "Medium"
    elif signal_strength < 0.9:
        return "High"
    else:
        return "Very_High"
```

### Implementation Steps
1. Create standardized function in `qlib_custom/signal_utils.py`
2. 🔄 Update all modules to use standard function
3. 🔄 Investigate data anomaly (all zeros)
4. 🔄 Add validation tests

---

## 🚨 Priority 2: Signal Tier Standardization

### Current Issues
- **Multiple formats**: Numeric (0-3) vs Letter (A-D) systems
- **Inconsistent confidence mapping**: Different implementations
- **ML vs Rule-based**: Multiple classification approaches

### Recommended Standard
```python
def calculate_signal_tier(row, model=None):
    """
    Standardized signal tier calculation
    
    Args:
        row: Data row with required features
        model: Optional ML model for prediction
    
    Returns:
        dict: {'tier': int, 'confidence': float}
    """
    if model is not None:
        # ML-based prediction (preferred)
        tier = model.predict_tier(row)
    else:
        # Fallback rule-based classification
        tier = rule_based_classification(row)
    
    # Standardized confidence mapping
    confidence_map = {0: 10.0, 1: 7.0, 2: 5.0, 3: 3.0}
    confidence = confidence_map.get(tier, 3.0)
    
    return {
        'tier': int(tier),
        'confidence': float(confidence)
    }

def rule_based_classification(row):
    """Fallback rule-based tier classification"""
    # Simplified, consistent logic
    abs_q50 = abs(row.get('q50', 0))
    vol_risk = row.get('vol_risk', 0.01)
    
    if abs_q50 > 0.01 and vol_risk < 0.005:
        return 0  # Best tier
    elif abs_q50 > 0.005:
        return 1  # Good tier
    elif abs_q50 > 0.002:
        return 2  # Medium tier
    else:
        return 3  # Lowest tier
```

### Implementation Steps
1. Create standardized function in `qlib_custom/signal_utils.py`
2. 🔄 Train/validate ML model for tier prediction
3. 🔄 Update all modules to use standard format
4. 🔄 Deprecate letter-based system

---

## 🔄 Priority 3: Unified Signal Quality Score

### Proposal
Create a single, comprehensive signal quality metric that combines strength and tier:

```python
def calculate_signal_quality_score(signal_strength, tier_confidence, prob_directional=None):
    """
    Unified signal quality score combining multiple factors
    
    Args:
        signal_strength: Standardized signal strength
        tier_confidence: Tier-based confidence (1-10)
        prob_directional: Optional directional probability
    
    Returns:
        float: Unified quality score [0, 100]
    """
    base_score = signal_strength * tier_confidence
    
    if prob_directional is not None:
        # Enhance with directional confidence
        directional_boost = abs(prob_directional - 0.5) * 2  # 0-1 scale
        base_score *= (1 + directional_boost)
    
    # Scale to 0-100 range
    return min(base_score * 10, 100.0)
```

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create `qlib_custom/signal_utils.py` with standardized functions
- [ ] Investigate signal_strength data anomaly
- [ ] Write comprehensive tests for new functions

### Phase 2: Migration (Week 2)
- [ ] Update `hummingbot_backtester.py` to use standard functions
- [ ] Update `advanced_position_sizing.py` 
- [ ] Update all PPO sweep modules
- [ ] Validate no performance regression

### Phase 3: Enhancement (Week 3)
- [ ] Implement unified signal quality score
- [ ] Train ML model for tier prediction
- [ ] Add feature performance tracking
- [ ] Update documentation

### Phase 4: Cleanup (Week 4)
- [ ] Remove deprecated implementations
- [ ] Add feature lifecycle tests
- [ ] Create feature usage analytics
- [ ] Final validation and documentation

---

## Validation Plan

### Data Quality Checks
1. **Signal Strength**: Investigate why current data shows all zeros
2. **Tier Distribution**: Validate 76% tier-0 is reasonable
3. **Correlation Analysis**: Check feature correlation with returns

### Performance Validation
1. **Backtest Comparison**: Before/after standardization results
2. **Feature Importance**: Measure impact on model performance
3. **Risk Metrics**: Ensure risk-adjusted returns maintained

### Integration Testing
1. **End-to-end**: Full pipeline with standardized features
2. **Regression Tests**: Ensure no functionality breaks
3. **Performance Tests**: Check computational efficiency

---

## Success Metrics

### Code Quality
- [ ] Single implementation per feature
- [ ] 100% test coverage for new functions
- [ ] Zero deprecated code remaining

### Performance
- [ ] Maintain or improve Sharpe ratio (current: 1.327)
- [ ] Maintain or improve max drawdown (current: -11.77%)
- [ ] No significant change in trade frequency

### Maintainability
- [ ] Clear feature documentation
- [ ] Standardized naming conventions
- [ ] Automated feature validation

---

## 🚨 Risk Mitigation

### Rollback Plan
- Keep original implementations as `_legacy` functions
- Maintain feature flags for easy switching
- Comprehensive A/B testing before full migration

### Monitoring
- Track feature distributions in production
- Alert on significant changes in feature values
- Regular correlation analysis with returns

---

*This plan ensures your sophisticated trading system maintains its excellent performance while becoming more maintainable and scalable.*