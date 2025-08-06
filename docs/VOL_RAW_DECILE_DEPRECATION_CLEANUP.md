# vol_raw_decile Feature Deprecation Cleanup

**Date**: 2025-08-04  
**Reason**: Feature was commented out in pipeline but still referenced in tests and documentation, causing confusion and test failures.  
**Decision**: Clean deprecation rather than revival - the new regime system provides better volatility classification.

---

## 🧹 **Cleanup Checklist**

### ✅ **1. Strip from Feature Registry**
- [ ] Remove `vol_raw_decile` from any feature maps or auto-generation hooks
- [ ] Remove from `FEATURE_DOCUMENTATION.md` 
- [ ] Remove from any config files that reference it
- [ ] Remove from pipeline diagrams or architecture docs

### ✅ **2. Patch Test Harness**
- [ ] Remove or skip `test_kelly_criterion_vol_raw_deciles_validation`
- [ ] Remove `kelly_with_vol_raw_deciles()` test dependencies
- [ ] Update test imports to remove `get_vol_raw_decile`
- [ ] Clean up any test data that expects `vol_raw_decile` column

### ✅ **3. Doc Sanity Sweep**
- [ ] Remove mentions from `FEATURE_DOCUMENTATION.md`
- [ ] Remove from `FEATURE_KNOWLEDGE_TEMPLATE.md`
- [ ] Remove from any README files
- [ ] Remove from pipeline configuration docs
- [ ] Update any architecture diagrams

### ✅ **4. Archive Responsibly**
- [ ] Move `get_vol_raw_decile()` to archive with deprecation comment
- [ ] Move `kelly_with_vol_raw_deciles()` to archive
- [ ] Move `VOL_RAW_THRESHOLDS` to archive
- [ ] Document replacement approach (regime_volatility system)

### ✅ **5. Changelog Documentation**
- [ ] Add deprecation entry to changelog
- [ ] Document migration path to new regime system
- [ ] Note in feature evolution log
- [ ] Update version notes

---

## 📝 **Deprecation Comments Template**

```python
# =============================================================================
# DEPRECATED FEATURE: vol_raw_decile
# =============================================================================
# Date: 2025-08-04
# Reason: Feature was disabled in pipeline but still referenced in tests/docs
# Replacement: Use regime_volatility from RegimeFeatureEngine instead
# 
# The new regime system provides:
# - regime_volatility: categorical (ultra_low, low, medium, high, extreme)  
# - Better integration with sentiment and dominance regimes
# - More stable thresholds based on vol_risk percentiles
#
# Migration: Replace vol_raw_decile usage with:
#   regime_engine = RegimeFeatureEngine()
#   df['regime_volatility'] = regime_engine.calculate_regime_volatility(df)
# =============================================================================

def get_vol_raw_decile(vol_raw_value):
    """
    DEPRECATED: Convert vol_raw value to decile rank (0-9)
    
    This function is no longer used in the active pipeline.
    Use RegimeFeatureEngine.calculate_regime_volatility() instead.
    """
    # Implementation archived for reference
    pass

VOL_RAW_THRESHOLDS = [
    # DEPRECATED: Static thresholds replaced by dynamic regime classification
    # See RegimeFeatureEngine for current volatility regime logic
]

def kelly_with_vol_raw_deciles(vol_raw, signal_rel, base_size=0.1):
    """
    DEPRECATED: Kelly position sizing using vol_raw deciles
    
    This function is no longer used in the active pipeline.
    Use Enhanced Kelly Criterion from position_sizing.py instead.
    """
    # Implementation archived for reference
    pass
```

---

## 🔄 **Migration Guide**

### **Old Approach (Deprecated)**:
```python
# OLD: Static decile-based volatility classification
vol_decile = get_vol_raw_decile(vol_raw_value)
kelly_size = kelly_with_vol_raw_deciles(vol_raw, signal_rel)

# Regime flags based on deciles
vol_extreme_high = (vol_decile >= 8)
vol_high = (vol_decile >= 6)
vol_low = (vol_decile <= 2)
```

### **New Approach (Current)**:
```python
# NEW: Dynamic regime-based volatility classification
regime_engine = RegimeFeatureEngine()
regime_volatility = regime_engine.calculate_regime_volatility(df)

# Enhanced Kelly with regime awareness
from src.features.position_sizing import AdvancedPositionSizer
position_sizer = AdvancedPositionSizer()
kelly_size = position_sizer.calculate_position_size(signal, vol_risk, regime_multiplier)

# Regime-based volatility flags
vol_extreme_high = (regime_volatility == 'extreme')
vol_high = (regime_volatility.isin(['high', 'extreme']))
vol_low = (regime_volatility.isin(['ultra_low', 'low']))
```

### **Benefits of New Approach**:
- **Dynamic thresholds** adapt to market conditions vs static percentiles
- **Integrated regime system** combines volatility, sentiment, and dominance
- **Better position sizing** through regime_multiplier system
- **Fail-fast design** with proper error handling
- **Comprehensive testing** and validation framework

---

## 📋 **Files to Clean**

### **Code Files**:
- `src/training_pipeline.py` - Remove commented vol_raw_decile code
- `src/training_pipeline_optuna.py` - Remove commented vol_raw_decile code  
- `tests/principles/test_comprehensive_statistical_validation.py` - Remove failing tests
- `tests/integration/test_nautilus_requirements_alignment.py` - Remove commented tests

### **Documentation Files**:
- `docs/FEATURE_DOCUMENTATION.md` - Remove vol_raw_decile references
- `docs/research/case_study/FEATURE_DOCUMENTATION.md` - Remove vol_raw_decile references
- `docs/FEATURE_KNOWLEDGE_TEMPLATE.md` - Remove if present
- `.kiro/specs/nautilus-trader-poc/tasks.md` - Remove vol_raw_decile references

### **Test Files**:
- Remove `test_kelly_criterion_vol_raw_deciles_validation`
- Remove imports of `get_vol_raw_decile`, `kelly_with_vol_raw_deciles`
- Update any test data expectations

### **Config Files**:
- Check for any feature lists or mappings that include vol_raw_decile
- Remove from any automated feature generation configs

---

## 🎯 **Replacement Feature Mapping**

| **Deprecated Feature** | **Replacement** | **Location** |
|------------------------|-----------------|--------------|
| `vol_raw_decile` | `regime_volatility` | `src/features/regime_features.py` |
| `kelly_with_vol_raw_deciles()` | `AdvancedPositionSizer` | `src/features/position_sizing.py` |
| `VOL_RAW_THRESHOLDS` | Dynamic percentiles in `RegimeFeatureEngine` | `src/features/regime_features.py` |
| Static decile flags | `regime_volatility` categories | `src/features/regime_features.py` |

---

## 📊 **Impact Assessment**

### **Positive Impacts**:
- ✅ **Cleaner codebase** - Remove dead/commented code
- ✅ **Passing tests** - No more failures from deprecated features  
- ✅ **Clear documentation** - No confusion about active vs deprecated features
- ✅ **Better volatility classification** - Dynamic regime system vs static deciles

### **No Negative Impacts**:
- ❌ **No functionality loss** - Feature wasn't active anyway
- ❌ **No performance impact** - Feature wasn't being calculated
- ❌ **No trading impact** - System already using regime-based approach

---

## 🚀 **Post-Cleanup Validation**

### **Test Suite**:
```bash
# Verify all tests pass after cleanup
python -m pytest tests/ -v

# Specifically check that vol_raw_decile tests are gone
python -m pytest tests/ -k "vol_raw_decile" --collect-only
# Should return: "collected 0 items"
```

### **Code Search**:
```bash
# Verify no active references remain
grep -r "vol_raw_decile" src/ --exclude-dir=archive
grep -r "kelly_with_vol_raw_deciles" src/ --exclude-dir=archive
grep -r "get_vol_raw_decile" src/ --exclude-dir=archive

# Should return no results (except in archive)
```

### **Documentation Check**:
```bash
# Verify documentation is clean
grep -r "vol_raw_decile" docs/ --exclude-dir=archive
# Should return no results (except in this deprecation doc)
```

---

## 📝 **Changelog Entry**

```markdown
## [v1.5.0] - 2025-08-04

### Removed
- **vol_raw_decile feature**: Deprecated static decile-based volatility classification
  - Removed `get_vol_raw_decile()` function (archived)
  - Removed `kelly_with_vol_raw_deciles()` function (archived)  
  - Removed `VOL_RAW_THRESHOLDS` constants (archived)
  - Removed associated test cases

### Migration
- **Volatility classification**: Use `RegimeFeatureEngine.calculate_regime_volatility()` 
- **Position sizing**: Use `AdvancedPositionSizer` with regime awareness
- **Volatility flags**: Use `regime_volatility` categories instead of decile thresholds

### Benefits
- Cleaner codebase with no dead/commented code
- Dynamic volatility thresholds vs static percentiles
- Integrated regime system combining volatility, sentiment, and dominance
- All tests now pass consistently
```

---

**Next Steps**: Execute cleanup checklist systematically, then validate that all tests pass and documentation is consistent.
---


## ✅ **Cleanup Progress**

### **Completed Items**:
- ✅ **Cleaned test imports** - Removed `kelly_with_vol_raw_deciles` and `get_vol_raw_decile` from test imports
- ✅ **Archived functions** - Added deprecation comments to `get_vol_raw_decile()` and `VOL_RAW_THRESHOLDS`
- ✅ **Updated documentation** - Replaced "Kelly Criterion with Vol_Raw Deciles" with "Enhanced Kelly Criterion (Regime-Aware)"
- ✅ **Cleaned feature references** - Updated regime detection references to use new unified system

### **Migration Examples Added**:
```python
# OLD (Deprecated):
vol_decile = get_vol_raw_decile(vol_raw_value)
kelly_size = kelly_with_vol_raw_deciles(vol_raw, signal_rel)

# NEW (Current):
from src.features.regime_features import RegimeFeatureEngine
from src.features.position_sizing import AdvancedPositionSizer

regime_engine = RegimeFeatureEngine()
position_sizer = AdvancedPositionSizer()

regime_volatility = regime_engine.calculate_regime_volatility(df)
kelly_size = position_sizer.calculate_position_size(signal, vol_risk, regime_multiplier)
```

### **Documentation Updates**:
- **Enhanced Kelly Criterion** now properly documented as regime-aware system
- **Migration notes** added to guide users from old to new approach
- **Feature references** updated to reflect current unified regime system

### **Next Steps**:
- [ ] Run test suite to verify no failures from deprecated features
- [ ] Check for any remaining references in other files
- [ ] Update changelog with deprecation entry

**Status**: Major cleanup completed - system now has clean separation between deprecated and active features.---


## 🔄 **Additional Consolidation Opportunities Discovered**

### **kelly_sizing Function Consolidation**

**Issue Identified**: The `kelly_sizing()` function in `training_pipeline.py` (lines ~120-180) duplicates functionality that should be in `src/features/position_sizing.py`.

**Current Situation**:
- `kelly_sizing()` in training pipeline: Comprehensive Kelly implementation with tier confidence, spread analysis, signal quality multipliers
- `AdvancedPositionSizer` in position_sizing.py: Separate position sizing implementation
- **Result**: Code duplication and potential inconsistency between pipeline and backtester

**Consolidation Plan**:

1. **Analyze Current `kelly_sizing` Implementation**:
   ```python
   # Current features in training_pipeline.kelly_sizing():
   - Tier confidence adjustment
   - Signal threshold multipliers (1.3x boost for validated signals)
   - Spread risk multipliers (1.2x for tight spreads)
   - Combined quality bonus (1.15x for best combination)
   - Conservative Kelly scaling (0.25x base)
   - Max position limit (0.5)
   ```

2. **Enhance `AdvancedPositionSizer`**:
   ```python
   # Add to src/features/position_sizing.py:
   def calculate_kelly_position(self, 
                               q10: float, q50: float, q90: float,
                               tier_confidence: float, signal_thresh: float, 
                               prob_up: float, spread_thresh: float = None) -> float:
       """
       Consolidated Kelly sizing with all validated adjustments
       Replaces training_pipeline.kelly_sizing()
       """
       # Implement the proven logic from training pipeline
       pass
   ```

3. **Update Training Pipeline**:
   ```python
   # Replace in training_pipeline.py:
   # OLD: df_all["kelly_position_size"] = df_all.apply(kelly_sizing, axis=1)
   # NEW: 
   position_sizer = AdvancedPositionSizer()
   df_all["kelly_position_size"] = df_all.apply(
       lambda row: position_sizer.calculate_kelly_position(
           row["q10"], row["q50"], row["q90"], 
           row["signal_tier"], row["signal_thresh_adaptive"], 
           row["prob_up"]
       ), axis=1
   )
   ```

4. **Benefits of Consolidation**:
   - ✅ **Single source of truth** for position sizing logic
   - ✅ **Consistent behavior** between pipeline and backtester
   - ✅ **Easier testing** - one implementation to validate
   - ✅ **Better maintainability** - changes in one place
   - ✅ **Enhanced functionality** - combine best of both implementations

**Priority**: **HIGH** - This affects core position sizing consistency

**Implementation Steps**:
1. Document current `kelly_sizing` logic and all its features
2. Enhance `AdvancedPositionSizer` to include all validated adjustments
3. Create comprehensive tests for the consolidated implementation
4. Update training pipeline to use consolidated version
5. Validate that results are identical
6. Archive the old `kelly_sizing` function with deprecation notes

**Testing Requirements**:
- Verify identical results between old and new implementations
- Test all edge cases (zero spreads, extreme confidence values, etc.)
- Validate performance impact is minimal
- Ensure backtester consistency

---

## 🎯 **Cleanup Success Metrics**

### **Completed Cleanups**:
- ✅ **vol_raw_decile deprecation** - Clean separation of deprecated vs active features
- ✅ **Test import cleanup** - No more references to deprecated functions
- ✅ **Documentation updates** - Clear migration path documented
- ✅ **Code archiving** - Deprecated functions properly marked and explained

### **Discovered Opportunities**:
- 🔄 **kelly_sizing consolidation** - Merge training pipeline and position_sizing implementations
- 🔄 **Feature consistency audit** - Systematic review of duplicated functionality
- 🔄 **Test coverage enhancement** - Comprehensive testing of consolidated features

### **System Health Improvements**:
- **Cleaner codebase** - No confusion between deprecated and active features
- **Better documentation** - Clear separation of what's current vs archived
- **Improved maintainability** - Single source of truth for each feature
- **Enhanced testability** - Consolidated implementations easier to test

**Next Phase**: Execute kelly_sizing consolidation to complete the position sizing unification.