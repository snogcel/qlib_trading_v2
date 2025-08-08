# Test Coverage Analysis
Generated: 2025-08-08 01:50:46

## Coverage Overview
- Total Features Tested: 9
- Fully Covered Features: 1
- Coverage Rate: 11.1%

## Feature Coverage Details
### ❌ spread
- Tests: 2/5 (40.0%)
- Test Types: 4 (economic_hypothesis, regime_dependency, implementation, failure_mode)
- Priorities: high, medium, low

**Failed Tests:**
- economic_hypothesis: Analysis for spread economic_hypothesis
- failure_mode: Analysis for spread failure_mode
- regime_dependency: Analysis for spread regime_dependency

### ❌ fg_index
- Tests: 3/7 (42.9%)
- Test Types: 4 (performance, economic_hypothesis, regime_dependency, failure_mode)
- Priorities: high, critical, medium, low

**Failed Tests:**
- economic_hypothesis: Analysis for fg_index economic_hypothesis
- regime_dependency: Analysis for fg_index regime_dependency
- economic_hypothesis: Analysis for fg_index economic_hypothesis
- economic_hypothesis: Analysis for fg_index economic_hypothesis

### ❌ kelly_sizing
- Tests: 1/2 (50.0%)
- Test Types: 2 (performance, failure_mode)
- Priorities: medium, low

**Failed Tests:**
- failure_mode: Analysis for kelly_sizing failure_mode

### ❌ Q50
- Tests: 4/8 (50.0%)
- Test Types: 4 (performance, regime_dependency, implementation, economic_hypothesis)
- Priorities: medium, critical, low

**Failed Tests:**
- implementation: Analysis for Q50 implementation
- performance: Analysis for Q50 performance
- implementation: Analysis for Q50 implementation
- performance: Analysis for Q50 performance

### ❌ Q10
- Tests: 3/6 (50.0%)
- Test Types: 5 (performance, economic_hypothesis, regime_dependency, implementation, failure_mode)
- Priorities: medium, critical, high, low

**Failed Tests:**
- performance: Analysis for Q10 performance
- regime_dependency: Analysis for Q10 regime_dependency
- failure_mode: Analysis for Q10 failure_mode

### ❌ vol_risk
- Tests: 4/7 (57.1%)
- Test Types: 3 (performance, regime_dependency, economic_hypothesis)
- Priorities: medium, critical, high, low

**Failed Tests:**
- regime_dependency: Analysis for vol_risk regime_dependency
- performance: Analysis for vol_risk performance
- economic_hypothesis: Analysis for vol_risk economic_hypothesis

### ❌ btc_dom
- Tests: 4/7 (57.1%)
- Test Types: 3 (performance, implementation, failure_mode)
- Priorities: medium, critical, high, low

**Failed Tests:**
- failure_mode: Analysis for btc_dom failure_mode
- failure_mode: Analysis for btc_dom failure_mode
- failure_mode: Analysis for btc_dom failure_mode

### ❌ regime_multiplier
- Tests: 2/3 (66.7%)
- Test Types: 3 (performance, implementation, failure_mode)
- Priorities: medium, critical, low

**Failed Tests:**
- failure_mode: Analysis for regime_multiplier failure_mode

### ✅ Q90
- Tests: 5/5 (100.0%)
- Test Types: 4 (performance, regime_dependency, implementation, economic_hypothesis)
- Priorities: medium, critical, low

## Test Type Coverage

- Economic Hypothesis: 6/11 (54.5%)
- Performance: 7/11 (63.6%)
- Failure Mode: 3/10 (30.0%)
- Implementation: 8/10 (80.0%)
- Regime Dependency: 4/8 (50.0%)

## Priority Coverage

- ✅ Critical: 13/13 (100.0%)
- ⚠️ High: 7/8 (87.5%)
- ❌ Medium: 4/15 (26.7%)
- ❌ Low: 4/14 (28.6%)
