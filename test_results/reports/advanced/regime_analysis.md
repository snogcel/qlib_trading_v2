# Market Regime Analysis

## Bear Market Regime
- Total Tests: 11
- Passed: 8
- Success Rate: 72.7%
- Failed Tests:
  - kelly_sizing: Analysis for kelly_sizing economic_hypothesis
  - kelly_sizing: Analysis for kelly_sizing performance
  - vol_risk: Analysis for vol_risk economic_hypothesis

## Sideways Market Regime
- Total Tests: 12
- Passed: 7
- Success Rate: 58.3%
- Failed Tests:
  - Q10: Analysis for Q10 regime_dependency
  - regime_multiplier: Analysis for regime_multiplier performance
  - btc_dom: Analysis for btc_dom failure_mode
  - regime_multiplier: Analysis for regime_multiplier regime_dependency
  - kelly_sizing: Analysis for kelly_sizing implementation

## Bull Market Regime
- Total Tests: 15
- Passed: 6
- Success Rate: 40.0%
- Failed Tests:
  - btc_dom: Analysis for btc_dom implementation
  - Q50: Analysis for Q50 implementation
  - spread: Analysis for spread regime_dependency
  - btc_dom: Analysis for btc_dom implementation
  - Q90: Analysis for Q90 performance
  - kelly_sizing: Analysis for kelly_sizing implementation
  - Q50: Analysis for Q50 economic_hypothesis
  - Q50: Analysis for Q50 economic_hypothesis
  - vol_risk: Analysis for vol_risk failure_mode
