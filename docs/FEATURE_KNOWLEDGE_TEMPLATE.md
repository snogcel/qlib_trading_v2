# Feature Knowledge Template

**Instructions**: Please fill in the details for each feature. Add/remove features as needed. Focus on the economic intuition, performance characteristics, and failure modes you've observed.

---

## 🎯 Core Signal Features

### Q50 (Primary Signal)
**Current Documentation**: Primary directional signal based on 50th percentile probability
**Implementation**: Found in training pipeline, qlib_custom/custom_multi_quantile.py

- Economic Hypothesis: Q50 reflects the probabilistic median of future returns—a directional vote shaped by current feature context. In asymmetric markets where liquidity, sentiment, and volatility drive price action, a skewed Q50 implies actionable imbalance. It captures value where behavioral or structural inertia inhibits instant price response.

- Market Inefficiency Exploited: Latency in capital flow and participant reaction. Most actors adjust portfolios incrementally, not instantly. Q50 anticipates shifts before they're priced in, especially during volatility expansions and regime transitions.

- Performance Characteristics: 
  - Sharpe ratio: ~1.2–2.4 (regime-dependent)
  - Hit rate: 58–65% in trending conditions
  - Outperforms when volatility-adjusted gating is in place

- Regime Dependencies: 
  - Strong in momentum/trending environments
  - Mixed during chop unless paired with volatility or meta signals
  - Underperforms during macro shocks or liquidity crunches where behavior becomes nonlinear

- Failure Modes: 
  - Regime misclassification causing overconfidence in weak signal
  - Stale or misaligned inputs leading to false positives
  - Whipsaw environments where Q50 flips direction prematurely

- Interaction Effects: 
  - Amplified by Q90–Q10 spread as confidence measure
  - Works synergistically with vol_risk for trade sizing
  - Counterbalanced by macro sentiment and FG Index for narrative-aware positioning

- Time Horizon: Daily predictive horizon (1-bar forward). Framework supports multi-timeframe stacking but primary use is single-day directional bias.

- Implementation Details: 
  - Quantile regression with ensemble smoothing
  - Feature inputs normalized and lagged appropriately
  - Output passed through gating logic and thresholding before use

- Temporal Causality: 
  - All features lagged; target label shift validated
  - Leak audits performed via synthetic tests and forward-shift comparisons
  - Designed to avoid predictive feedback loops from target alignment bugs

- Auditable Pipeline: 
  - Traceability from raw .bin ingestion → feature script → final output
  - Snapshot logging with feature values, signal strength, and return attribution
  - Cross-checkable via spreadsheet overlay tools with timestamp alignment

- Contrarian Logic: 
  - High Q50 during fear regimes flags rebound setups
  - Low Q50 in greed environments suggests exhaustion or topping behavior
  - SHAP visualizations highlight narrative counterweights (volatility, sentiment, regime flags)


### Q10 & Q90 (Quantile Bounds)
**Current Documentation**: [Found in training pipeline but not well documented]
**Implementation**: Part of multi-quantile system

- Economic Hypothesis: Q10 and Q90 offer insight into the tails of the modeled return distribution. While Q50 reflects the directional bias, the bounds measure dispersion and asymmetry, contextualizing risk and conviction. They represent investor pessimism (Q10) and optimism (Q90) under prevailing conditions—akin to crowd sentiment extremes in price formation.

- Market Inefficiency Exploited: These quantiles expose latent overreaction and underreaction. In real markets, price rarely reflects "fair value" at every moment; instead, it oscillates around extremes due to emotion, leverage, and liquidity constraints. Q90 detects euphoria-driven overshoots, Q10 uncovers capitulation-based dislocations.

- Performance Characteristics:
  - Sharp shifts in Q90/Q10 often precede price acceleration or reversal
  - Wide dispersion improves directional confidence (i.e., Q50 reinforced by Q90/Q10 skew)
  - When aligned with narrative filters, bounds strengthen timing precision

- Usage Pattern:
  - Used to validate or veto Q50 signals—e.g., Q50↑ with Q90↑ = stronger conviction
  - Position sizing adapts based on dispersion; tighter bounds = smaller trades
  - Risk management uses Q10 violations as stop-loss heuristics

- Failure Modes:
  - Tails may widen artificially in volatile chop, misleading confidence
  - Regime misclassification (e.g., trend vs mean-reversion) distorts bound interpretation
  - Feature corruption or outdated lags can bias Q90/Q10—false signal alignment


### Spread (Q90 - Q10)
**Current Documentation**: Found in training pipeline as uncertainty measure
**Implementation**: `df["spread"] = df["q90"] - df["q10"]`

- Economic Hypothesis: Spread represents modeled uncertainty—wider spreads indicate dispersion in opinion or structural ambiguity, while tight spreads imply consensus or regime stability. This mirrors real-world behavior where directional conviction is stronger during clear macro narratives and weaker in indecision.

- Market Inefficiency Exploited: Spread exploits the fact that traders respond differently under uncertainty—some pull back, others overcommit. By measuring this dynamic, spread allows the model to allocate size and conviction only when risk-adjusted opportunity is high.

- **Performance Characteristics**:
  - High spread correlates with higher win-rate *when paired with directional Q50*
  - Low spread often indicates chop or regime confusion—useful for trade suppression
  - Strong additive effect in gating logic to reduce overtrading

- **Regime Dependencies**:
  - High volatility → spread expands, signaling directional opportunity
  - Low volatility → spread contracts, signals wait-state or mean reversion
  - Most reliable in identifiable trend regimes; weaker in macro chop

- **Failure Modes**:
  - Spread expansion during low-liquidity events may falsely suggest signal strength
  - Overreliance on spread without regime confirmation may lead to size inflation
  - Noise-induced spread (e.g., feature degradation) can obscure true market risk


---

## Risk & Volatility Features

### Vol_Risk (Variance-Based)
**Feature Name**: $vol_risk  
**Location in Codebase**: src/data/crypto_loader.py  
**Formula**: Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)  
→ 6-period rolling variance of log returns, calculated by squaring standard deviation

**Economic Hypothesis**  
-Variance preserves squared deviations from the mean, retaining signal intensity of extreme movements. Unlike standard deviation, it does not compress tail events through square rooting, making it more suitable for amplifying asymmetric risk bursts. Economically, this reflects nonlinear exposure—where downside risk is often more impactful than upside volatility.

**Market Inefficiency Exploited**  
-Detects underpriced fragility: liquidation spirals, panic slippage, or breakdowns in microstructure. These often escape standard deviation filters due to square-root dampening, but variance exposes them for gating and rotation logic.

**Performance Characteristics**  
- Distinguishes explosive instability from mild chop  
- Enhances RL position gating during risk spikes  
- Refines reward shaping in volatility-aware strategies  
- Syncs well with multiplier logic for dynamic aggression
- Typically enhances risk-adjusted returns when used as a gating or scaling factor for aggression. Backtests show improved drawdown control in unstable markets and better tail-risk capture during macro inflections. Works well when paired with multiplier logic.

**Regime Dependencies**  
- High Volatility: dominant signal source  
- Low Volatility: converges with std dev, weaker differentiation  
- Transition Zones: strong leading indicator for inflection gating

**Failure Modes**  
- Flat markets: minimal signal, can amplify noise  
- Synthetic volatility: increased sensitivity to false positives  
- Lag risk: fixed 6-period lookback may delay signal response  
- Requires normalization across assets to ensure comparability

**Implementation Notes**  
- Parameters:  
  - Ref($close, 1): lagged close for return calc  
  - Std(..., 6): rolling 6-period standard deviation  
- Output: squared to derive variance  
- Export: normalized per asset; exposed as $vol_risk_level  
- Downstream: used by regime classifier and RL trust filter


### Vol_Raw
**Feature Name**: $vol_raw  
**Current Documentation**: [Found in training pipeline as "Std(Log($close / Ref($close, 1)), 6)"]
**Implementation**: Rolling 6-day standard deviation of log daily returns

- Economic Hypothesis: Captures short-term realized volatility as a proxy for recent uncertainty and risk premium adjustments. Serves as a foundational input for regime detection, entropy modulation, and feature normalization.

- Market Inefficiency Exploited: Reflects latent behavioral overreactions during high volatility episodes and potential alpha decay during low volatility. Helps disentangle signal strength from noise by providing a volatility anchor.

- Performance Characteristics: Sensitive to sharp movements, regime transitions, and event clusters. Can introduce spikiness, but is auditable and temporally causal. Typically used as a baseline for scaled features, not a standalone predictor.

- Usage Pattern: Serves as an input to `vol_scaled`, entropy calibration, and conditional gating. Often accessed downstream via normalized variants for stability across regimes.

**Empirical Ranges**:
- Typical range: `0.005 – 0.05` for most assets during stable regimes
- Extremes: `> 0.1` during flash crashes, crypto liquidations, or macro shocks
- Values near `0` suggest stagnant or highly compressed price movement

**Normalization Logic**:
- Z-score normalization across rolling 120-day window for entropy application
- Optional cross-sectional ranking when used in regime classifiers
- Log transform discouraged due to distortion near zero—min-max or Gaussian mapping preferred

**Failure Modes**:
- Can become unstable around asset listing or delisting events (e.g., first few candles of a newly listed token)
- Sensitive to price anomalies (e.g., stale prints or bad ticks); requires validation against volume and spread
- Low liquidity assets may produce artificially low volatility despite fragmented price movement
- Feature death in flat regimes—flag if `vol_raw < 1e-5` for N > 3 consecutive days


### Vol_Raw_Decile
**Current Documentation**: [Found in training pipeline as "Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10"]
**Implementation**: 180-day percentile rank of 6-day rolling volatility, scaled into decile buckets (0–10)

- Economic Hypothesis: Normalizes asset-level volatility into a cross-temporal decile framework, enabling regime-aware feature scaling and comparability. Helps isolate structural shifts in realized volatility independent of raw magnitude.

- Market Inefficiency Exploited: Allows agents to condition behavior on volatility percentile rather than absolute value—beneficial in assets with nonstationary volatility profiles (e.g., newly listed tokens or leverage-driven markets). Decile ranking exposes inflection points where volatility shifts indicate possible mean reversion, forced liquidations, or hedging demand.

- Performance Characteristics: Smooths high-frequency noise in `vol_raw`, providing quantized stability and interpretability. Well-suited for gating, binning, and conditional policy selection. Can lag in fast-moving regimes (e.g., post-crash volatility recalibration), but generally robust to outliers.

- Usage Pattern: Commonly used as a conditional input for entropy modulation, reward shaping, and regime-aware decision logic. Deciles support intuitive debugging and policy slicing (e.g., "agent tends to down-weight signals in Decile 9+").

**Empirical Ranges**:
- By definition: returns decile bucket from `0` to `10`
- Values `0–2`: historically low volatility regime (compression, pre-breakout phases)
- Values `8–10`: high volatility regime (macro dislocations, crypto deleveraging, or event risk spikes)

**Normalization Logic**:
- Decile buckets eliminate need for further scaling
- For some agents: one-hot encoded or ordinal-transformed for policy conditioning
- Rarely used in cross-sectional logic unless relative to sector or cluster norms

**Failure Modes**:
- Regime bleed: if asset volatility undergoes secular shift (e.g., leverage removed), the decile anchor becomes misaligned—consider regime-conditioned re-ranking
- Asset-specific pathologies: sparse trading days can skew percentile window, especially in small caps or altcoins
- Bucket oscillation: near percentile boundaries, repeated flipping between deciles can cause policy instability—apply hysteresis or smoothing if gating behavior is volatile


### Vol_Raw_Momentum
**Current Documentation**: [Derived from vol_raw delta calculation]
**Implementation**: Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)

- Economic Hypothesis: Measures acceleration or deceleration in realized volatility—volatility is not static, and sudden changes often precede structural regime shifts. Tracks the rate of change in short-term risk appetite or perceived uncertainty.

- Market Inefficiency Exploited: Highlights behavioral overcorrection and reflexivity—markets often overreact to volatility spikes, leading to temporary mispricing or liquidity imbalances. Volatility momentum captures this adjustment period before price stabilizes.

- Performance Characteristics: Acts as an early signal for regime transition. High values suggest volatility breakout or unwinding, while low or negative values signal compression. Useful in gating logic to prevent signal usage during volatility inflection.

- Usage Pattern: Used in adaptive gating for multiplier calibration, signal trust scoring, and regime transition detection. Can be paired with sentiment inflection or dominance shift to reinforce narrative-confirmed volatility breaks.

**Empirical Ranges**:
- Normal market shifts: `-0.005 to 0.005`
- Transitional states: `> 0.01` or `< -0.01` suggest volatility dislocation or reversion
- Extreme stress: `> 0.02` seen during macro events or liquidation zones

**Normalization Logic**:
- Z-score normalization across 60–120 day window preferred
- Optionally scaled by realized volatility baseline to convert into percent-of-risk delta
- Can be clipped to remove tails above 99th percentile for policy stability

**Failure Modes**:
- Spurious spikes in illiquid assets or low-volume hours
- Can be distorted by stale data (e.g. bad ticks in close price)
- Lag sensitivity: may underreact to fast reversals if volatility snap is short-lived
- Interpreted incorrectly if `vol_raw` is low—momentum spike may be statistical noise


### Vol_Momentum_Scaled
**Current Documentation**: [Risk-adjusted velocity of volatility]
**Implementation**: (Vol_Raw_Momentum) / Std(Log($close / Ref($close, 1)), 20)

- Economic Hypothesis: Volatility momentum should not be interpreted in isolation—its market impact depends on background volatility. Scaling by longer-term realized vol allows us to contextualize how dramatic a change in volatility is, relative to recent conditions. Useful for identifying when acceleration in risk is statistically meaningful.

- Market Inefficiency Exploited: Addresses cognitive bias in interpreting raw risk—market participants tend to underweight context, reacting disproportionately to familiar but low-risk moves or underreacting to large changes in already high-risk environments. Normalized momentum flags subtle shifts in stable regimes and amplifies significance in noisy ones.

- Performance Characteristics: More stable than vol_raw_momentum in turbulent markets; resists distortion during high-volatility episodes. Elevates nuanced signals during quiet periods. Particularly helpful in multi-asset environments where volatility baselines differ.

- Usage Pattern: Ideal for signal gating and dynamic model trust scoring. Often used in volatility-adjusted portfolio rebalancing logic. Can gate feature activation or disable strategies during noise-heavy periods.

**Empirical Ranges**:
- Calm regimes: `-0.2 to 0.2`
- Transition zones: `|vol_momentum_scaled| > 0.3` indicates regime break or emerging stress
- High-alert: `> 0.5` seen during liquidation risk or macro shock

**Normalization Logic**:
- Already self-normalized by background realized volatility
- Optional z-score transformation for comparability across assets
- May be clipped for stability—|x| > 2 as discretionary filter

**Failure Modes**:
- Can invert signal in low-risk environments (division-by-small baseline)
- Sensitive to volatility calculation parameters—choice of denominator window matters
- Distortion risk if baseline `vol` is misestimated due to stale or partial price data


### Native QLib Features (Crypto Loader)

#### OPEN1
- **Definition**: `$OPEN / $close`
- **Purpose**: Captures relative positioning of open price versus current close—helps detect opening sentiment bias.
- **Note**: Higher-order OPEN2/3 excluded due to strong correlation and diminishing marginal utility.

#### VOLUME1
- **Definition**: `$volume / ($volume + 1e-12)`
- **Purpose**: Provides baseline volume measure, useful for gating features in low-liquidity environments.
- **Note**: Higher-order VOLUME2/3 removed due to autocorrelation and negligible added information.

#### RSV1 / RSV2 / RSV3
- **Definition**: `($close - Min($low, d)) / (Max($high, d) - Min($low, d) + 1e-12)` for `d = 1, 2, 3`
- **Purpose**: Rolling stochastic-style signal; identifies relative positioning of close within high-low range. Useful for short-term momentum and breakout detection.
- **Note**: RSV selected as top-performing rolling feature; others excluded to maintain loader minimalism and interpretability.

**General Loader Philosophy**
- **Correlation Screening**: Higher-order price/volume windows excluded after empirical correlation analysis.
- **Precision**: Division by small values handled safely with `+1e-12` noise to avoid computational instability.
- **Compatibility**: All features QLib-native and return normalized ratios, ideal for downstream scaling or direct model consumption.


### Vol_Scaled
**Current Documentation**: [Quantile-normalized short-term realized volatility]  
**Implementation**:
(Std(Log($close / Ref($close, 1)), 6) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01)) /  
(Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.99) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01))

- **Economic Hypothesis**:  
  Volatility behavior is inherently regime-dependent. Raw measures fluctuate with market noise and macro conditions, often misleading when compared across assets or time. By scaling realized volatility against its local empirical distribution, this feature enables regime-aware normalization—surfacing relative turbulence rather than absolute magnitude.

- **Market Inefficiency Exploited**:  
  Market participants tend to underreact to volatility normalization or compression, and overreact to spikes—especially when lacking context. This normalization counters those behavioral biases by mapping volatility into percentile space, improving gating and confidence logic.

- **Performance Characteristics**:  
  - Returns scaled float in `[0, 1]` range  
  - Smooth, interpretable signal for regime classification  
  - Enhances entropy coefficient tuning and volatility-aware policy selection  
  - More robust than z-score in the presence of outliers or heavy tails

- **Usage Pattern**:  
  Commonly used in entropy modulation, volatility-adjusted Kelly logic, and agent reward shaping. Also embedded in conditional signal activation pipelines to gate strategies during compression or breakout zones.

**Empirical Ranges**:
- Calm regimes: `0.1 – 0.3`  
- Elevated conditions: `0.5 – 0.7`  
- Turbulent zones: `>0.8` often flag liquidation risk or macro event transitions

**Normalization Logic**:
- Scaled using 30-period rolling quantile bounds (`1st` and `99th` percentile)  
- Unitless output; avoids distortion from scale or currency  
- Optionally smoothed or clipped for gating applications

**Failure Modes**:
- In illiquid assets: quantile bounds may become unstable, misrepresenting actual volatility  
- Underperforms in assets with stationary volatility regimes—may return overly flat signals  
- Sensitive to skewed distributions or stale prints; requires clean price feeds  
- Compression zones may overstate signal strength when volatility is low across entire quantile range


### Feature: $fg_index
- **Definition**: Normalized Crypto Fear & Greed Index  
- **Source Scale**: Original index ranges from 0 to 100  
- **Normalization Logic**:  
  `$fg_index = Raw_FG_Index / 100`  
  Brings signal into `[0.0, 1.0]` for consistent downstream compatibility  
- **Economic Interpretation**:  
  Captures aggregate market sentiment—fear at lower values, greed at higher values.  
  Synthesizes volatility, momentum, social media activity, dominance, trends.

- **Usage Patterns**:
  - Volatility regime overlays (e.g., gated vol_spike logic)
  - Entropy modulation in greedy regimes
  - Conditioning layer for agent action space compression
  - Tagging for regime audit panels

- **Empirical Thresholds**:
  - Extreme Fear: `≤ 0.20`  
  - Extreme Greed: `≥ 0.80`

- **Failure Modes**:
  - Sentiment may lag vs price action on rapid moves  
  - High values not always predictive of continuation—greed plateaus  
  - Underweight sources (e.g., paused surveys) may weaken coverage during anomalies

- **Implementation Notes from [alternative.me](https://alternative.me/crypto/fear-and-greed-index/) data source**:
  - Volatility (25 %) We’re measuring the current volatility and max. drawdowns of bitcoin and compare it with the corresponding average values of the last 30 days and 90 days. We argue that an unusual rise in volatility is a sign of a fearful market.

  - Market Momentum/Volume (25%) Also, we’re measuring the current volume and market momentum (again in comparison with the last 30/90 day average values) and put those two values together. Generally, when we see high buying volumes in a positive market on a daily basis, we conclude that the market acts overly greedy / too bullish.
  
  - Social Media (15%) While our reddit sentiment analysis is still not in the live index (we’re still experimenting some market-related key words in the text processing algorithm), our twitter analysis is running. There, we gather and count posts on various hashtags for each coin (publicly, we show only those for Bitcoin) and check how fast and how many interactions they receive in certain time frames. An unusual high interaction rate results in a grown public interest in the coin and in our eyes, corresponds to a greedy market behaviour.
  
  - Surveys (15%) currently paused, Together with strawpoll.com, quite a large public polling platform, we’re conducting weekly crypto polls and ask people how they see the market (disclaimer: we own this site, too). Usually, we’re seeing 2,000 - 3,000 votes on each poll, so we do get a picture of the sentiment of a group of crypto investors. We don’t give those results too much attention, but it was quite useful in the beginning of our studies. You can see some recent results here.
  
  - Dominance (10%) The dominance of a coin resembles the market cap share of the whole crypto market. Especially for Bitcoin, we think that a rise in Bitcoin dominance is caused by a fear of (and thus a reduction of) too speculative alt-coin investments, since Bitcoin is becoming more and more the safe haven of crypto. On the other side, when Bitcoin dominance shrinks, people are getting more greedy by investing in more risky alt-coins, dreaming of their chance in next big bull run. Anyhow, analyzing the dominance for a coin other than Bitcoin, you could argue the other way round, since more interest in an alt-coin may conclude a bullish/greedy behaviour for that specific coin.
  
  - Trends (10%) We pull Google Trends data for various Bitcoin related search queries and crunch those numbers, especially the change of search volumes as well as recommended other currently popular searches. For example, if you check Google Trends for "Bitcoin", you can’t get much information from the search volume. But currently, you can see that there is currently a +1,550% rise of the query „bitcoin price manipulation“ in the box of related search queries (as of 05/29/2018). This is clearly a sign of fear in the market, and we use that for our index.
  - Already normalized inline at loader level  
  - Mirror-ready for cross-feature blending (momentum, social, vol)

### Feature: $btc_dom
- **Definition**: Normalized Bitcoin Dominance Index  
- **Source Scale**: Originally expressed as a percentage (`0–100`)  
- **Normalization Logic**:  
  `$btc_dom = Raw_BTC_Dominance / 100`  
  Scaled to `[0.0, 1.0]` for compatibility across all regimes and feature blending

- **Economic Interpretation**:  
  Represents Bitcoin’s market cap share across the entire crypto ecosystem.  
  - High dominance: Flight to safety, fear of speculative altcoins  
  - Low dominance: Risk-on sentiment, broader alt-coin participation

- **Usage Patterns**:
  - Layered regime detection with sentiment signals (`$fg_index`)  
  - Alt-season tagging and allocation strategies  
  - Compression or expansion triggers in asset selection logic  
  - Agent path conditioning via dominance-aware masks

- **Empirical Thresholds**:
  - High Dominance: `≥ 0.60` → Often flags conservative or risk-off stances  
  - Low Dominance: `≤ 0.40` → Signals speculative activity or early bull rotations

- **Failure Modes**:
  - Dominance shifts can lag behind actual capital flow or layer-1 rotation  
  - Interpreting BTC dominance without alt-coin volume context may mislead  
  - Prone to distortion during stablecoin contractions or ETF flows

- **Implementation Notes**:
  - Inline normalization already applied at loader level  
  - Binary flags (`btc_dom_high`, `btc_dom_low`) consolidated into regime logic  
  - Pairs well with volatility features for full-spectrum risk mapping


### Feature: fg_std_7d
- **Definition**: 7-day rolling standard deviation of normalized Crypto Fear & Greed Index  
- **Formula**:  
  `fg_std_7d = Std($fg_index, 7)`  
  Captures sentiment volatility across a short-term window

- **Economic Interpretation**:  
  - Reflects how erratic or stable market sentiment is—not the sentiment level, but its variability.  
  - High values may indicate narrative uncertainty, shifting market mood, or speculative churn  
  - Low values often signal entrenched sentiment regimes (e.g., persistent fear during drawdown)

- **Usage Patterns**:
  - Regime volatility calibration for gating behavior  
  - Alert generation for transitions between narrative stability and panic/speculation  
  - Entropy tuning modulate decision randomness based on sentiment turbulence
  - Feed into agent conditioning logic for more cautious policy during unstable emotional regimes

  ##

- **Empirical Ranges**:
  - Stable mood: `< 0.05` (smooth sentiment curve)  
  - Transition zone: `~ 0.10 – 0.15`  
  - High emotional volatility: `> 0.20` → often aligns with macro news, liquidations, or hype phases

- **Failure Modes**:
  - False spikes from noise in the index composition (e.g. social media surges unrelated to price)  
  - Lag in reflection during fast transitions—may peak after sentiment already shifted  
  - Low dispersion during data gaps or index recalibration periods

- **Implementation Notes**:
  - Signal is volatility of sentiment, not market volatility directly  
  - Pairs well with `fg_zscore_14d` for a richer sentiment movement model  
  - Consider logging against regime tags to capture transition dynamics


### Feature: btc_std_7d
- **Definition**: 7-day rolling standard deviation of normalized Bitcoin Dominance  
- **Formula**:  
  `btc_std_7d = Std($btc_dom, 7)`  
  Measures short-term variability in Bitcoin’s market share across the crypto landscape

- **Economic Interpretation**:  
  - Captures dominance volatility, not direction—reveals the stability or churn in capital allocation preferences.  
  - High values suggest erratic rotation between BTC and altcoins  
  - Low values point to entrenched preferences or macro-driven risk stance

- **Usage Patterns**:
  - Regime rotation detection signals early-stage alt-season or flight to safety dynamics  
  - Agent strategy pivots stabilize behavior when market cap allocations fluctuate sharply  
  - Telemetry flag log spikes as potential macro or behavioral transitions  
  - Pairs with `$btc_dom` and `$fg_index` for richer risk mapping

- **Empirical Ranges**:
  - Stable dominance: `< 0.03`  
  - Rotation zone: `~ 0.05 – 0.08`  
  - High dominance churn: `> 0.10` → often driven by speculative alt flows or ETF disruptions

- **Failure Modes**:
  - False elevation from flash crashes or anomalous stablecoin moves  
  - Low dispersion in compressed dominance regimes may hide underlying flows  
  - Lag sensitivity can peak post-rotation rather than during

- **Implementation Notes**:
  - Normalized input improves comparability across time and conditions  
  - Should be smoothed before gating if used in high-frequency agents  
  - Flag telemetry spikes as regime change candidates


### Feature: fg_zscore_14d
- **Definition**: 14-day z-score of the normalized Crypto Fear & Greed Index  
- **Formula**: `fg_zscore_14d = ($fg_index - Mean($fg_index, 14)) / Std($fg_index, 14)`  
  Indicates how current sentiment deviates from recent norms

- **Economic Interpretation**:  
  - Captures sentiment shock magnitude, tells us how "unusual" today's fear or greed level is compared to the recent rolling window. High positive z-score implies unusually strong greed; negative implies extreme fear outside the baseline expectation.

- **Usage Patterns**:
  - Regime inflection detection helps catch rapid sentiment shifts even if raw level appears stable  
  - Gating trigger for strategy transitions (e.g. activate contrarian filter when z-score > 2)  
  - Telemetry marker log spikes in sentiment acceleration to audit strategy behavior  
  - May condition model policies or adjust signal weights during abnormal sentiment regimes

- **Empirical Ranges**:
  - Neutral zone: `-1 to +1`  
  - Alert levels: `> +2` (strong greed), `< -2` (strong fear)  
  - Values beyond ±3 suggest sentiment dislocation, often from macro catalysts

- **Failure Modes**:
  - Lag risk: sudden sentiment turns may take 2–3 days to reflect sharply  
  - False elevation: skewed input distribution may inflate z-score despite stable sentiment  
  - Misinterpretation: high z-score doesn’t always imply price breakout—may reflect lagging emotional reaction

- **Implementation Notes**:
  - Centered and scaled—ideal for blending with other z-normalized features  
  - Consider clipping or smoothing if used in gating logic to avoid abrupt strategy toggling  
  - Complements `fg_std_7d` by distinguishing direction from variability

### Feature: btc_zscore_14d
- **Definition**: 14-day z-score of BTC daily close price  
- **Formula**: `btc_zscore_14d = ($close_btc - Mean($close_btc, 14)) / Std($close_btc, 14)`  
  Shows relative deviation of current BTC close from short-term mean

- **Economic Interpretation**: Acts as a short-term sentiment and momentum proxy—signals whether BTC price is unusually stretched from its recent average. Crucial for detecting temporary dislocations or overextensions often tied to trader behavior, macro shocks, or liquidity spurts.

- **Usage Patterns**:
  - Gating logic: used to trigger contrarian or momentum filters  
  - Normalization anchor: re-centers volatile price signals for model ingestion  
  - Volatility regime detector: extreme z-scores often coincide with high volatility transitions  
  - Interaction metric: amplifies regime or sentiment features under stress conditions

- **Empirical Ranges**:
  - Typical range: `-1.5 to +1.5`  
  - Stretch zones: `> +2` implies overheated rally, `< -2` implies panic selloff  
  - Outside ±3 is rare and usually signals macro impact or speculative blow-off tops

- **Failure Modes**:
  - Decay blindness: may lag reversal patterns as it emphasizes mean-reversion bias  
  - False polarity: local bottoms or tops may show mild z-score due to volatility smearing  
  - Overconfidence: models may overreact to large z-scores if not contextually gated  

- **Implementation Notes**:
  - Should be paired with volatility guards (e.g., `btc_vol_7d`) to avoid false signals during low-vol drift  
  - Can be used in interaction terms with `fg_zscore_14d` to capture price-sentiment dissonance  
  - Helps distinguish structural regime change (e.g. macro or policy-driven) from sentiment noise


### LABEL0 — Forward 1-Day Return

**Definition**:  
Returns the percentage price change from day -2 to day -1 (i.e., the next day after the prediction point).

**Formula**:  
`LABEL0 = Ref($close, -2) / Ref($close, -1) - 1`

**Economic Rationale**:  
This label captures the immediate forward momentum or reversal signal and is often used in short-term prediction tasks. It reflects how effective a signal is at anticipating next-day price movement, making it a useful benchmark for features targeting daily alpha generation.

**Normalization Note**:  
Unnormalized percentage return. If used as a regression target, range standardization or regime-aware clipping may be recommended to contain extreme values.

**Failure Modes**:  
- Spikes during low liquidity or large gap opens.
- Sensitive to close price anomalies—ensure price feed integrity before labeling.
- May underrepresent sustained directional moves when used in isolation.

**Audit Hooks**:  
- Check predictive signal correlation by decile binning.
- Validate economic interpretability with feature saliency across market regimes.


### Volatility Features Suite
**Current Documentation**: Multi-window volatility metrics for regime detection
**Implementation**: `volatility_features_final_summary.py` (referenced but not found)

- **Feature List**: [What specific volatility features exist?]
- **Economic Hypothesis**: [Why multiple volatility windows?]
- **Market Inefficiency Exploited**: [What patterns do different windows capture?]
- **Performance Characteristics**: [Individual and combined performance]
- **Implementation Details**: [Window sizes, calculation methods]


## 🎲 Position Sizing Features

### Custom Kelly Sizing Model (Signal-Aware, Volatility-Calibrated)  
**Current Documentation**: Generates a risk-adjusted position size based on probabilistic signal strength, expected payoff asymmetry, and regime-aware volatility measures.  
**Implementation**: `training_pipeline.py`  

- **Economic Hypothesis**:  
  Position sizing should dynamically respond to signal quality and regime volatility. Traditional Kelly assumes static payoff asymmetry and constant risk preference, but this enhanced version adapts the sizing based on predictive certainty (`prob_up`), estimated distribution shape (`q10`, `q90`), and signal/volatility gating mechanisms. The hypothesis is that smarter sizing yields better compounding outcomes by avoiding overconfidence in noisy or weak regimes.  

- **Market Inefficiency**  
  - Forecast Asymmetry: Market signals often contain skewed forecast distributions (e.g., downside tails more predictable).  
  - Volatility Clustering: Tight spreads signal regime stability—providing better entry windows.  
  - Signal Tiering: Strong signals (high `abs_q50`) are empirically validated to outperform noise. Signal validation prevents over-allocation to spurious indicators.  

- **Performance Characteristics**:  
  - Sharpe Improvement: Dynamic thresholding boosts Sharpe from -0.002 to +0.077 in validated signal zones.  
  - Drawdown Control: Spread gating reduces excess volatility exposure—especially in wide distribution regimes.  
  - Compounding Efficiency: Conservative base sizing coupled with selective amplification ensures stability across varied regimes.  

- **Component Breakdown**  
  - `q10`, `q50`, `q90`: Predictive quantiles estimating distribution shape and used to compute `prob_up` and `spread`  
  - `prob_up`: Probability of upward price movement, derived from quantile skew  
  - `spread = q90 - q10`: Proxy for regime volatility and predictive confidence  
  - `tier_confidence`: Controls Kelly base size via confidence tiering  
  - `signal_thresh_adaptive`: Enables sizing boost if signal strength exceeds empirical Sharpe threshold  
  - `spread_thresh`: Activates bonus when spread indicates low volatility regime  
  - `combined_bonus`: Additional multiplier if both signal and spread validation pass  
  - `max_position_pct`: Ceiling to prevent overexposure, empirically scaled by regime  

**Parameter Sensitivity**

| Parameter                | Impact Level     | Calibration Notes |
|--------------------------|------------------|-------------------|
| `tier_confidence`        | **High**         | Discrete 0–10 scale; controls base Kelly sizing |
| `signal_thresh_adaptive` | **Medium–High**  | Set via Sharpe threshold percentiles |
| `spread_thresh`          | **Medium**       | Histogram-derived gating zone for spread |
| `prob_up`                | **Medium**       | Computed from quantiles; signal direction bias |
| `combined_bonus`         | **Critical**     | Use range 0.3–0.5; avoid compounding noise |
| `max_position_pct`       | **High**         | Empirical regime-based ceiling |


- **Failure Modes**  
  - Probabilistic Drift: Misestimating `prob_up` introduces systemic bias  
  - Zero or Negative Expected Loss: May yield unbounded sizing—guarded via minimum payoff override  
  - Threshold Calibration Drift: Suboptimal gating thresholds skew allocation  
  - Overfit Multipliers: Static bonuses lose validity over time unless re-learned  

- **Core Logic**  
  - Expected Payoff Setup: Long if `prob_up > 0.5`, short otherwise  
  - Base sizing scaled by `tier_confidence`  
  - +30% if `abs_q50` > `signal_thresh_adaptive`  
  - +20% if `spread < spread_thresh`  
  - +15% if both signal and spread pass → `combined_bonus` activation  
  - Apply `max_position_pct` as final cap  


### Enhanced Kelly Criterion
**Current Documentation**: Multi-factor position sizing using Kelly + Vol + Sharpe + Risk Parity
**Implementation**: `src/features/position_sizing.py` (AdvancedPositionSizer class)

- **Economic Hypothesis**: 
  Traditional Kelly criterion assumes static market conditions and uniform risk preferences. The enhanced multi-factor approach recognizes that optimal position sizing should adapt to changing market volatility, recent performance patterns, and risk-adjusted return expectations. Each factor captures a different aspect of market inefficiency: Kelly captures probability asymmetries, volatility adjustment maintains consistent risk exposure, Sharpe optimization targets risk-adjusted returns, risk parity ensures balanced risk contribution, and momentum adaptation learns from recent performance.

- **Market Inefficiency Exploited**: 
  - Volatility Clustering: Markets exhibit periods of high and low volatility that traditional Kelly ignores. Volatility adjustment exploits this by scaling positions inversely to current volatility.
  - Performance Persistence: Recent performance contains information about strategy effectiveness that static sizing misses. Momentum adaptation exploits short-term performance persistence.
  - Risk Premium Variations: Sharpe ratios vary across market regimes. Sharpe-optimized sizing exploits periods when risk premiums are elevated.
  - Risk Concentration: Single-factor sizing can lead to unintended risk concentration. Risk parity exploits diversification benefits by targeting consistent risk contributions.

- **Performance Characteristics**: 
  - Ensemble Approach: Weighted combination (Kelly 30%, Volatility 25%, Sharpe 20%, Risk Parity 15%, Momentum 10%) provides robustness across market conditions
  - Conservative Scaling: Uses 25% of full Kelly to prevent over-leveraging while maintaining growth optimality
  - Adaptive Risk Management: Volatility adjustment maintains target 15% volatility exposure, reducing drawdowns during volatile periods
  - Performance Feedback: Momentum component increases size by up to 50% after wins, decreases by up to 30% after losses

- **Component Breakdown**: 
  - Kelly Component: Uses quantile-based probability estimation and payoff ratios, scaled by tier_confidence and reduced to 25% of full Kelly for safety
  - Volatility Component: Scales position by target_volatility/current_volatility ratio, with quadratic confidence scaling and spread penalty for wide prediction intervals
  - Sharpe Component: Sizes position proportional to expected Sharpe ratio (q50/estimated_vol), with confidence boost and historical Sharpe adjustment
  - Risk Parity Component: Targets 2% portfolio volatility contribution, scaled by signal strength and confidence
  - Momentum Component: Adjusts base position by recent performance (+50% max after wins, -30% max after losses) with spread penalty

- **Failure Modes**: 
  - Missing Dependencies: Requires historical_data, portfolio_volatility, current_volatility, recent_performance, historical_sharpe - graceful degradation needed when unavailable
  - Volatility Estimation Errors: Uses (q90-q10)/3.29 approximation for volatility which may be inaccurate during regime changes
  - Performance Chasing: Momentum component can amplify losses if recent performance is misleading
  - Regime Misclassification: Fixed weights may be suboptimal across different market regimes
  - Overfitting Risk: Multiple parameters and methods increase risk of overfitting to historical data

- **Parameter Sensitivity**: 

| Parameter | Impact Level | Calibration Notes |
|-----------|--------------|-------------------|
| `max_position_pct` | **Critical** | Hard cap at 50%; prevents over-leveraging regardless of method |
| `target_volatility` | **High** | Default 15%; lower values reduce position sizes in volatile markets |
| `ensemble_weights` | **High** | Kelly 30%, Vol 25%, Sharpe 20%, Risk Parity 15%, Momentum 10% |
| `conservative_kelly_fraction` | **Medium-High** | 25% of full Kelly; balance between growth and safety |
| `confidence_scaling` | **Medium** | Linear for Kelly, quadratic for volatility, affects all methods |
| `spread_penalty_factor` | **Medium** | 50x for volatility, 30x for momentum; penalizes wide spreads |
| `performance_sensitivity` | **Medium** | 2x multiplier for wins, 1x for losses in momentum component |

**Implementation Status**: 
- **Available Methods**: Kelly, Volatility (can adapt to use vol_risk), Ensemble (partial)
- **Requires External Data**: Sharpe (historical_sharpe), Risk Parity (portfolio_volatility), Momentum (recent_performance)
- **Integration Need**: Consolidation with validated training_pipeline.py Kelly implementation recommended

### Regime-Aware Sizing
**Current Documentation**: Adjusts position sizes based on market conditions
**Implementation**: `regime_aware_kelly.py`

- **Economic Hypothesis**: [Why regime-aware position sizing?]
- **Market Inefficiency Exploited**: [What regime patterns does this exploit?]
- **Performance Characteristics**: [Improvement over static sizing]
- **Regime Classifications**: [What regimes are recognized?]
- **Adaptation Logic**: [How does sizing change across regimes?]

---

## 🔄 Regime & Market Features

### Regime Multiplier (formerly Signal Strength)  
**Current Documentation**: Unified regime-based position multiplier, range [0.1, 5.0]  
**Implementation**: `src/features/regime_features.py` – `RegimeFeatureEngine.calculate_regime_multiplier()`  
**Logic**: Volatility (0.4x–1.5x) × Sentiment (0.6x–2.0x) × Crisis (3.0x) × Opportunity (2.5x)

**Economic Hypothesis**:  
Market regime context—specifically volatility and sentiment—contains predictive signals about risk asymmetry and behavioral mispricings. By dynamically scaling exposure, the multiplier aims to increase allocation in regimes with lower perceived risk or stronger contrarian edge, and suppress size during crowded or fragile conditions.

**Market Inefficiency Exploited**:  
Static position sizing overlooks dynamic macro structures. This feature exploits behavioral overextensions, risk compression cycles, and regime-dislocation edges by modulating exposure in line with structural fragility or latent opportunity.

**Performance Characteristics**:  
Improves Sharpe by excluding trades during noisy, high-volatility regimes and amplifying edge during calm, contrarian setups. Acts as a dynamic filter on top of raw signal gating, with early backtests showing elevated hit rate consistency in favorable regime combos.

**Component Breakdown**:  
- Volatility: Reflects execution risk—lower volatility regimes imply more reliable fills and signal trust; higher volatility suggests caution.  
- Sentiment: Contrarian logic—fear signals mispricing, greed flags crowd risk.  
- Crisis: Amplifier for moments of dislocation, indicating edge from panic-driven flows or short-term inefficiencies.  
- Opportunity: Flag for regime-validated breakout setups or narrative alignment, justifying higher conviction sizing.

**Calibration Logic**:  
Initial values derived from stratified regime performance analysis. Sentiment and volatility thresholds tuned via historical risk-adjusted return heatmaps. Crisis/opportunity boosts tested across synthetic and real-world narrative events for sizing sanity. Final clipping [0.1, 5.0] prevents edge overexpression or leverage creep.

**Failure Modes**:  
- Lagged regime signals leading to premature size changes.  
- Misclassification in noisy macro conditions causing multiplier distortion.  
- Excessive compounding from stacked regimes (e.g., opportunity + crisis + fear), inflating multiplier beyond realistic expectations—best monitored via telemetry.  
- Overreliance on discrete regime flags; smoother confidence blending may improve robustness.


### Regime Classification Suite
**Current Documentation**: 7 unified regime features replacing 23+ scattered features
**Implementation**: `src/features/regime_features.py`
- `regime_volatility`: categorical (ultra_low, low, medium, high, extreme)
- `regime_sentiment`: categorical (extreme_fear, fear, neutral, greed, extreme_greed)
- `regime_dominance`: categorical (btc_low, balanced, btc_high)
- `regime_crisis`: binary crisis detection
- `regime_opportunity`: binary contrarian opportunity detection
- `regime_stability`: continuous [0,1] regime transition frequency
- `regime_multiplier`: continuous [0.1,5.0] unified position scaling

# === regime_volatility ===
regime_volatility = {
    "type": "categorical",
    "values": ["ultra_low", "low", "medium", "high", "extreme"],
    "economic_hypothesis": "Volatility clusters signal regime shifts in liquidity, leverage, and behavioral reflexivity. Classifying volatility as a discrete regime encodes investor psychology and market structure stress.",
    "market_inefficiency_exploited": "Markets underreact to volatility compression and overreact to spikes. Regime classification helps anticipate risk asymmetries and transition points.",
    "performance_characteristics": {
        "ultra_low": "Risk-on, momentum-rich; higher signal conviction",
        "low": "Mild return compression; safe to scale exposure",
        "medium": "Balanced risk-reward; neutral strategy settings",
        "high": "Elevated risk; reduce sensitivity and position size",
        "extreme": "Systemic fragility; hedging or contrarian bias preferred"
    },
    "detection_logic": "Thresholds derived from vol_risk percentiles. Conditions map across ultra_low to extreme.",
    "threshold_calibration": {
        "percentiles": [0.1, 0.3, 0.7, 0.9],
        "cache_logic": "Dynamic and cached for reproducibility and drift tolerance"
    },
    "interaction_effects": [
        "Pairs with regime_sentiment for narrative-vol amplification",
        "Confirms regime_crisis spikes",
        "Guides regime_multiplier scaling logic"
    ]
}

# === regime_sentiment ===
regime_sentiment = {
    "type": "categorical",
    "values": ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"],
    "economic_hypothesis": "Investor sentiment oscillates non-linearly around narrative catalysts. Discretization extracts reversion signals from ambient noise.",
    "market_inefficiency_exploited": "Extreme sentiment often diverges from fundamentals. Contrarian setups exploit reversion; greed phases may be momentum overextended.",
    "performance_characteristics": {
        "extreme_fear": "Contrarian setups thrive; risk premia inflated",
        "fear": "Risk-off bias; prune or hedge signals",
        "neutral": "Signals behave normally",
        "greed": "Momentum-rich setups; scale exposure",
        "extreme_greed": "Overextended markets; fade or reversion bias"
    },
    "detection_logic": "Mapped from smoothed FG Index values using static thresholds: ≤20, ≤35, ≤65, ≤80, >80",
    "threshold_calibration": {
        "thresholds": [20, 35, 65, 80],
        "rationale": "Behavioral thresholds from FG Index literature; stable and interpretable"
    },
    "interaction_effects": [
        "Pairs with regime_volatility to refine risk gating",
        "Triggers regime_opportunity logic in oversold states",
        "Informs regime_multiplier conviction scaling"
    ]
}

# === regime_dominance ===
regime_dominance = {
    "type": "categorical",
    "values": ["btc_low", "balanced", "btc_high"],
    "economic_hypothesis": "BTC dominance reflects capital concentration and market regime. Altcoin cycles behave differently depending on BTC's share of total crypto capitalization. A 'balanced' regime represents a neutral distribution of capital with no clear tilt toward Bitcoin or alts.",
    "market_inefficiency_exploited": "Dominance regimes lag narrative shifts. Detecting them enables cycle-aware capital rotation, altcoin timing, and execution gating. The 'balanced' state acts as a reset zone—neither BTC-led nor alt-led.",
    "performance_characteristics": {
        "btc_low": "Altcoin-friendly; broader speculation and dispersion",
        "balanced": "No macro tilt; favors core strategies and neutral allocation",
        "btc_high": "Risk-off preference for BTC; alt signals may decay or underperform"
    },
    "detection_logic": "BTC dominance is bucketed by historical percentiles: ≤30% → btc_low, >30% and <70% → balanced, ≥70% → btc_high.",
    "threshold_calibration": {
        "percentiles": [0.3, 0.7],
        "rationale": "Reflects extremes and midpoint in capital distribution; tuned for crypto rotation dynamics"
    },
    "interaction_effects": [
        "Modulates regime_multiplier for alt/BTC risk positioning",
        "Enhances regime_opportunity when paired with sentiment extremes",
        "Neutralizes overly directional signal logic during ambiguous market conditions"
    ]
}

# === regime_crisis ===
regime_crisis = {
    "type": "binary",
    "values": [0, 1],
    "economic_hypothesis": "Extreme volatility combined with extreme fear creates a market anomaly zone where standard execution logic may fail. These rare conditions often coincide with deleveraging events, panic-selling, or liquidity vacuums.",
    "market_inefficiency_exploited": "Markets overreact during stress regimes, mispricing risk and opportunity. By gating execution or flipping strategy profiles during crisis, agents can avoid slippage and catch mean reversion rebounds.",
    "detection_logic": "Crisis regime is active if both regime_volatility == 'extreme' AND regime_sentiment == 'extreme_fear'. Otherwise, signal is off.",
    "threshold_rationale": {
        "volatility": "Quantile-based tail condition—top 5% of historical realized vol",
        "sentiment": "Survey-derived or index-based indicator below critical level (e.g., FG Index ≤10)"
    },
    "positioning_logic": {
        "execution_gating": "Agents may pause trading or switch to protective modes",
        "regime_multiplier": "Can flip directional bias to mean-reversion if supported by volatility decay",
        "alerting": "Flag for human review or discretionary override in live systems"
    },
    "interaction_effects": [
        "Overrides regime_dominance weightings due to anomaly nature",
        "Trumps narrative-based signals unless corroborated",
        "Can trigger emergency logic across modules—slippage control, slippage awareness, bid/ask probing"
    ]
}

# === regime_opportunity ===
regime_opportunity = {
    "type": "categorical",
    "values": ["none", "alt_opportunity", "btc_opportunity"],
    "economic_hypothesis": "Certain combinations of volatility, sentiment, and dominance produce asymmetric risk-reward setups—favoring either BTC or altcoins. This includes both momentum-led and contrarian conditions.",
    "market_inefficiency_exploited": "Most agents avoid trading during extreme fear or volatility, missing high-conviction reversals and directional flows. This signal identifies those rare windows of alpha.",
    "detection_logic": {
        "momentum_conditions": {
            "alt_opportunity": "(vol=high or extreme) AND (sentiment=greed) AND (dominance=btc_low)",
            "btc_opportunity": "(vol=high or extreme) AND (sentiment=greed) AND (dominance=btc_high)"
        },
        "contrarian_conditions": {
            "btc_opportunity": "(sentiment=extreme_fear) AND ((vol=high or extreme) OR (dominance=btc_high))"
        },
        "default": "none"
    },
    "positioning_logic": {
        "regime_multiplier": "Amplifies conviction during both momentum and contrarian windows",
        "execution_gating": "Disables low-conviction strategies in 'none' regime",
        "bias_logic": "Momentum → trend-following; Contrarian → mean-reversion preference"
    },
    "feature_blend_logic": {
        "multi-factor join": "Combines regime_volatility, regime_sentiment, and regime_dominance",
        "hierarchy": ["crisis override", "opportunity momentum", "opportunity contrarian"]
    },
    "interaction_effects": [
        "In crisis=0 environments, regime_opportunity boosts allocation",
        "In crisis=1 environments, contrarian opportunity may activate protective long bias only",
        "Overrides neutral positioning during strong signal alignment"
    ]
}

# === regime_stability ===
regime_stability = {
    "type": "continuous",
    "range": [0.0, 1.0],
    "economic_hypothesis": "Stability reflects consistency in regime volatility classification. Markets with persistent volatility bands are more predictable, while frequent flips signal noise, indecision, or structural change.",
    "market_inefficiency_exploited": "Most models treat all volatility equally. This metric adds context—stable environments allow stronger inference, less overfitting, and more reliable signal blending.",
    "detection_logic": "In a rolling N-bar window, count regime_volatility changes. Then invert frequency: stability = 1 - mean(change_flag).",
    "parameters": {
        "window": 20,
        "rationale": "Captures medium-term regime persistence without overreacting to single flips. Tunable based on strategy cadence."
    },
    "positioning_logic": {
        "stability_multiplier": "Used to scale conviction scores and signal weights",
        "execution_filtering": "Suppresses low-stability zones to avoid regime whipsaws",
        "meta-gating": "Optional: disable regime-dependent features if stability < threshold"
    },
    "interaction_effects": [
        "Can gate or amplify regime_opportunity and dominance-based logic",
        "Used in score blending to weight persistent regimes more heavily",
        "Improves explainability by separating noise-induced reclassifications from true regime shifts"
    ]
}

# === regime_multiplier ===
regime_multiplier = {
    "type": "continuous",
    "range": [0.1, 5.0],
    "economic_hypothesis": "Position size should adapt to market context. Volatility, sentiment, and regime classification offer structural cues to amplify or reduce exposure. This multiplier quantifies that adaptivity in scalar form.",
    "market_inefficiency_exploited": "Static position sizing ignores macro conditions. By layering context-sensitive adjustments, this multiplier seeks to exploit windows where risk is low and edge is high, while suppressing trades during structurally fragile states.",
    "detection_logic": {
        "volatility_adjustments": {
            "ultra_low": 1.5,
            "low": 1.2,
            "medium": 1.0,
            "high": 0.7,
            "extreme": 0.4
        },
        "sentiment_adjustments": {
            "extreme_fear": 2.0,
            "fear": 1.3,
            "neutral": 1.0,
            "greed": 0.8,
            "extreme_greed": 0.6
        },
        "crisis_boost": "If regime_crisis == 1, multiplier *= 3.0",
        "opportunity_boost": "If regime_opportunity == 1, multiplier *= 2.5",
        "final_clip": "[0.1, 5.0]"
    },
    "threshold_rationale": {
        "volatility": "Inverse risk logic—low volatility increases sizing, high volatility suppresses",
        "sentiment": "Contrarian bias during fear, risk reduction during greed extremes",
        "compound stacking": "Boosts are cumulative, then clipped for guardrails"
    },
    "positioning_logic": {
        "signal amplification": "Used to adjust position sizing and gating logic",
        "alpha scaling": "Allows regime-aware strategies to respond dynamically to market stress or opportunity",
        "execution control": "Serves as meta-filter for trade eligibility and leverage"
    },
    "interaction_effects": [
        "Highly sensitive to regime overlap; stacking can lead to nonlinear jumps",
        "Can conflict with signal confidence if not properly normalized",
        "Best used downstream with telemetry panel to track regime weight contribution"
    ],
    "implementation_caveats": [
        "May create regime-induced convexity unless paired with signal damping",
        "Consider splitting multiplier into separate components for volatility vs sentiment vs regime alignment",
        "Telemetry logging advised to avoid silent sizing distortions"
    ]
}


### Temporal Quantile Features (Phase 1 Complete)
**Current Documentation**: 6 features implemented with economic justification
**Implementation**: `src/features/regime_features.py`
- `q50_momentum_3`: Information flow persistence
- `spread_momentum_3`: Market uncertainty evolution
- `q50_stability_6`: Consensus stability measure
- `q50_regime_persistence`: Behavioral momentum
- `prediction_confidence`: Risk-adjusted confidence
- `q50_direction_consistency`: Trend strength indicator

Temporal Quantile Features – Audit-Ready Breakdown

1. q50_momentum_3
- Economic Hypothesis: Price formation lags information flow; directional persistence reflects delayed signal absorption
- Market Inefficiency Exploited: Latent reaction to public/private info, slow-moving price adjustment
- Performance Characteristics: Early regime signal booster; tends to fire before consensus stabilizes
- Lookback Periods: 3 captures short-term momentum without high noise sensitivity
- Failure Modes: Can misfire in reversal regimes or news-driven whiplash

2. spread_momentum_3
- Economic Hypothesis: Spreads widen during liquidity stress or sentiment divergence
- Market Inefficiency Exploited: Frictions and fragmentation in price discovery
- Performance Characteristics: Flags uncertainty surges; useful for regime gating or downweighting signals
- Lookback Periods: 3 emphasizes short bursts of disagreement
- Failure Modes: Less informative in highly illiquid assets or sentiment-neutral conditions

3. q50_stability_6
- Economic Hypothesis: Stable mid-quantile signals reflect tight consensus and reduced strategic dispersion
- Market Inefficiency Exploited: Overreaction fade, regime lock-in
- Performance Characteristics: Improves confidence gating; helps shape weighting logic
- Lookback Periods: 6 smooths noise and avoids short-term false convergence
- Failure Modes: Can miss latent volatility under tight directional agreement

4. q50_regime_persistence
- Economic Hypothesis: Herd behavior and narrative lock-in delay regime rotation
- Market Inefficiency Exploited: Cognitive inertia, anchoring
- Performance Characteristics: Enhances regime duration gating and telegraphs behavioral inflection zones
- Lookback Periods: Dynamic via run-length encoding of directional streaks
- Failure Modes: Vulnerable to chop and fakeouts in low-volume conditions

5. prediction_confidence
- Economic Hypothesis: Position sizing should reflect signal dispersion and strength
- Market Inefficiency Exploited: Overconfidence in noisy environments
- Performance Characteristics: Core gating metric for signal weighting and reinforcement learning trust
- Lookback Periods: Instantaneous (derived feature); no rolling logic
- Failure Modes: Can overtrust flat spreads in illiquid environments or during prediction drift

6. q50_direction_consistency
- Economic Hypothesis: Consistent directional signals indicate momentum-based regimes
- Market Inefficiency Exploited: Sluggish reversal by institutional flow
- Performance Characteristics: Supports trend classifiers and sharpens exit logic
- Lookback Periods: 6 balances between identifying meaningful drift and avoiding regime noise
- Failure Modes: Fails during mean-reverting chop or synthetic signal compression

---

## Interaction & Enhancement Features

### Q50 Interaction Features
**Current Documentation**: Found in training pipeline
**Implementation**: 
- `q50_x_low_variance`: Q50 × variance regime interaction
- `q50_x_high_variance`: Q50 × high variance interaction
- `q50_x_extreme_variance`: Q50 × extreme variance interaction
- `q50_x_trending`: Q50 × trending interaction
- `spread_x_high_variance`: spread × high variance interaction
- `vol_risk_x_abs_q50`: vol_risk × abs_q50 variance interaction
- `enhanced_info_ratio_x_trending`: enhanced_info_ratio x trending

- **Economic Hypothesis**: 
  Q50 signals behave differently across market regimes, and these interactions capture regime-dependent signal effectiveness. In low variance regimes, Q50 signals are more reliable due to stable market conditions. In high/extreme variance regimes, Q50 signals may be amplified by volatility but require different interpretation. Trending regimes enhance both Q50 and enhanced_info_ratio signals due to momentum persistence, while spread interactions with high variance capture uncertainty amplification during volatile periods.

- **Market Inefficiency Exploited**: 
  - **Regime-Dependent Signal Decay**: Most models treat signals uniformly across regimes, missing that Q50 effectiveness varies with market volatility and momentum conditions
  - **Volatility-Signal Interaction**: Markets exhibit different signal-to-noise ratios across variance regimes—low variance periods have cleaner signals, extreme variance periods have amplified but noisier signals
  - **Momentum-Information Interaction**: Trending markets exhibit information persistence that enhances both directional signals (Q50) and risk-adjusted signals (enhanced_info_ratio)
  - **Uncertainty Amplification**: Spread (prediction uncertainty) becomes more significant during high variance periods, creating interaction effects

- **Performance Characteristics**: 
  - **Regime Specialization**: Each interaction captures specific market conditions where base signals perform differently
  - **Non-Linear Enhancement**: Interactions often outperform linear combinations of base features
  - **Variance-Aware Scaling**: vol_risk × abs_q50 provides variance-adjusted signal strength that adapts to market conditions
  - **Momentum Amplification**: Trending interactions boost signal strength during persistent directional moves

- **Selection Logic**: 
  - **Variance Regimes**: Low/high/extreme variance capture different volatility states with distinct signal characteristics
  - **Momentum Regimes**: Trending vs ranging markets exhibit different signal persistence patterns
  - **Risk-Signal Coupling**: vol_risk × abs_q50 combines market risk with signal strength for adaptive sizing
  - **Quality-Momentum Coupling**: enhanced_info_ratio × trending captures high-quality signals during momentum periods

- **Failure Modes**: 
  - **Regime Misclassification**: Incorrect variance or momentum regime assignment leads to wrong interaction weights
  - **Overfitting Risk**: Multiple interactions can overfit to historical regime patterns that don't persist
  - **Correlation Inflation**: Highly correlated interactions may not add independent information
  - **Regime Transition Noise**: Interactions may be unstable during regime transitions when classifications flip frequently


### Signal Enhancement Features
**Current Documentation**: Found in training pipeline with variance-based enhancements
**Implementation**: Various signal processing and enhancement features

- **Feature List**: 
  - `signal_strength`: Variance-enhanced signal using enhanced_info_ratio scaling
  - `enhanced_info_ratio`: Signal-to-total-risk ratio (market + prediction uncertainty)
  - `info_ratio`: Traditional signal-to-spread ratio (prediction uncertainty only)
  - `abs_q50`: Absolute value of Q50 for signal strength measurement
  - `position_size_suggestion`: Inverse variance-scaled position sizing
  - `base_position_size`: Raw inverse variance scaling before tradeable filtering

- **Economic Hypothesis**: 
  Raw signals need enhancement to account for varying market conditions and risk environments. Traditional information ratios only consider prediction uncertainty (spread) but ignore market volatility. Enhanced signals incorporate both market variance and prediction variance to provide risk-adjusted signal strength. Position sizing should be inversely related to market variance to maintain consistent risk exposure across different volatility regimes.

- **Market Inefficiency Exploited**: 
  - **Incomplete Risk Assessment**: Most systems use prediction uncertainty alone, ignoring market volatility context
  - **Static Signal Interpretation**: Raw signals are interpreted uniformly regardless of market conditions
  - **Variance Blindness**: Position sizing often ignores the variance structure of returns, leading to inconsistent risk exposure
  - **Threshold Misalignment**: Fixed thresholds don't adapt to changing risk environments

- **Performance Characteristics**: 
  - **Enhanced Info Ratio**: Typically 20-30% higher than traditional info ratio due to incorporating market variance
  - **Signal Strength**: Provides normalized signal quality that adapts to market conditions
  - **Position Size Suggestion**: Inverse variance scaling maintains consistent risk exposure
  - **Risk-Adjusted Performance**: Enhanced signals show better risk-adjusted returns across different volatility regimes

- **Enhancement Methods**: 
  - **Variance Integration**: Combines market variance (vol_risk) with prediction variance (spread²/4)
  - **Total Risk Calculation**: √(market_variance + prediction_variance) for comprehensive risk assessment
  - **Adaptive Scaling**: Signal strength scaled by enhanced_info_ratio relative to threshold
  - **Inverse Variance Weighting**: Position sizing inversely proportional to vol_risk for consistent risk exposure
  - **Tradeable Filtering**: Enhanced signals only applied to economically significant opportunities

### Additional Variance Risk Metrics
- **`signal_to_variance_ratio`**: `abs_q50 / max(vol_risk, 0.0001)` - Signal quality per unit of market variance
- **`variance_adjusted_signal`**: `q50 / sqrt(max(vol_risk, 0.0001))` - Risk-adjusted directional signal strength


## Performance & Validation Features

### Signal Thresholds
**Current Documentation**: Adaptive threshold system with variance-aware calibration
**Implementation**: `signal_thresh_adaptive`, `effective_info_ratio_threshold` in training_pipeline.py

- **Economic Hypothesis**: 
  Fixed thresholds fail to adapt to changing market conditions and risk environments. Adaptive thresholds should scale with transaction costs, volatility regimes, and market structure to maintain consistent economic significance. Thresholds must account for both magnitude-based (expected value) and ratio-based (information ratio) criteria to ensure trades are both economically viable and statistically significant.

- **Market Inefficiency Exploited**: 
  - **Static Threshold Inefficiency**: Fixed thresholds miss opportunities in low-volatility regimes and generate false signals in high-volatility regimes
  - **Transaction Cost Misalignment**: Thresholds that don't scale with realistic transaction costs lead to unprofitable trades
  - **Regime-Blind Filtering**: Static thresholds don't account for regime-dependent signal quality and execution risk
  - **Risk-Return Mismatching**: Thresholds that ignore volatility context create inconsistent risk-adjusted returns

- **Performance Characteristics**: 
  - **Adaptive Scaling**: Signal thresholds adjust based on variance regimes (±30% in low variance, +80% in extreme variance)
  - **Economic Significance**: Expected value approach generates 20-50% more trading opportunities than traditional thresholds
  - **Hit Rate Improvement**: Regime-aware thresholds improve hit rates by filtering out regime-inappropriate signals
  - **Transaction Cost Awareness**: 5 bps realistic cost basis prevents micro-profit trades

- **Adaptation Logic**: 
  - **Signal Threshold**: `realistic_transaction_cost × regime_multipliers × variance_multiplier`
  - **Regime Multipliers**: Low variance (-30%), High variance (+40%), Extreme variance (+80%), Trending (-10%)
  - **Variance Multiplier**: `1.0 + vol_risk × 500` to scale with market volatility
  - **Info Ratio Threshold**: Base 1.5, adjusted by variance regimes (±0.4 for low/extreme variance)
  - **Expected Value Gating**: `prob_up × potential_gain - (1-prob_up) × potential_loss > transaction_cost`

- **Calibration Method**: 
  - **Historical Analysis**: Regime multipliers derived from stratified performance analysis across volatility regimes
  - **Transaction Cost Basis**: 5 bps (0.0005) based on realistic crypto trading costs vs 20 bps theoretical
  - **Quantile-Based Regimes**: Variance regimes defined by vol_risk percentiles (30th, 70th, 90th)
  - **Dynamic Recalibration**: Thresholds can be updated based on changing market structure and transaction costs

### Information Ratio Features
**Current Documentation**: Enhanced information ratio system with variance-aware risk assessment
**Implementation**: `enhanced_info_ratio`, `info_ratio`, `signal_strength` in training_pipeline.py

- **Economic Hypothesis**: 
  Information ratio should capture signal quality relative to total risk, not just prediction uncertainty. Traditional IR (signal/spread) ignores market volatility, leading to misaligned risk assessment. Enhanced IR incorporates both market variance and prediction variance to provide a comprehensive risk-adjusted signal quality measure that adapts to changing market conditions.

- **Market Inefficiency Exploited**: 
  - **Incomplete Risk Pricing**: Markets often misprice signals during different volatility regimes because participants focus on prediction uncertainty while ignoring market risk context
  - **Regime-Blind Signal Assessment**: Traditional IR treats all signals equally regardless of underlying market volatility, missing opportunities in low-vol regimes and overestimating quality in high-vol regimes
  - **Variance Structure Ignorance**: Most systems don't decompose total risk into market and prediction components, leading to suboptimal signal weighting

- **Performance Characteristics**: 
  - **Enhanced vs Traditional**: Enhanced IR averages 20-30% higher than traditional IR due to incorporating market variance
  - **Regime Adaptation**: Enhanced IR provides better signal discrimination across different volatility regimes
  - **Signal Strength Scaling**: Used to create adaptive signal strength that scales with risk-adjusted quality
  - **Threshold Effectiveness**: Enhanced IR thresholds provide better trade filtering than traditional spread-based thresholds

- **Enhancement Logic**: 
  - **Traditional IR**: `abs_q50 / max(spread, 0.001)` - signal divided by prediction uncertainty only
  - **Enhanced IR**: `abs_q50 / max(total_risk, 0.001)` where `total_risk = √(market_variance + prediction_variance)`
  - **Market Variance**: Uses `vol_risk` (6-period rolling variance) to capture market volatility
  - **Prediction Variance**: Converts spread to variance via `(spread/2)²` assuming normal distribution
  - **Signal Strength**: `abs_q50 × min(enhanced_info_ratio / threshold, 2.0)` with 2x cap to prevent extreme scaling

- **Usage Pattern**: 
  - **Signal Quality Filter**: Enhanced IR compared against adaptive thresholds for trade eligibility
  - **Position Sizing Input**: Signal strength (enhanced IR-based) used for variance-aware position sizing
  - **Regime Gating**: Enhanced IR thresholds adjusted based on volatility regimes
  - **Performance Attribution**: Enhanced IR provides better signal quality metrics for strategy evaluation

---

## Feature Performance Hierarchy

**PLEASE FILL IN:** Based on your experience, rank features by their contribution to system performance:

### Tier 1 (Core Drivers)
**Features that drive most of the performance - system fails without these:**
- **Q50 (Primary Signal)**: [Primary directional signal - core alpha source]
- **Vol_Risk**: [Risk gating and position sizing foundation]
- **Regime Multiplier**: [Position sizing and risk management]
- **[Add other Tier 1 features based on your experience]**

### Tier 2 (Important Enhancers)
**Features that significantly improve results when combined with Tier 1:**
- **Spread (Q90-Q10)**: [Confidence measure and uncertainty quantification]
- **Kelly Sizing**: [Risk-adjusted position sizing]
- **FG Index**: [Sentiment-based regime detection]
- **BTC Dominance**: [Market structure and rotation signals]
- **[Add other Tier 2 features based on your experience]**

### Tier 3 (Useful Additions)
**Features that add value but aren't critical - nice-to-have enhancements:**
- **Vol_Raw variants**: [Additional volatility context]
- **Temporal Quantile Features**: [Time-based signal enhancements]
- **Q50 Interaction Features**: [Signal refinement]
- **[Add other Tier 3 features based on your experience]**

### Tier 4 (Experimental/Noise)
**Features that may not add value or are still being validated:**
- **[List features that haven't proven their worth yet]**
- **[Features that might be redundant or noisy]**
- **[Experimental features under evaluation]**

### Performance Impact Notes
**PLEASE FILL IN:**
- **Most Critical Dependencies**: [Which feature combinations are essential?]
- **Diminishing Returns**: [At what point do additional features stop helping?]
- **Feature Stability**: [Which features are most reliable across different market conditions?]
- **Implementation Priority**: [If rebuilding the system, what order would you implement features?]

---

## 🚨 System-Level Insights

### Feature Interactions
**PLEASE FILL IN:**
- **Positive Synergies**: [Which features work well together?]
- **Negative Interactions**: [Which features conflict or cancel out?]
- **Redundancies**: [Which features provide similar information?]

### Market Regime Dependencies
**PLEASE FILL IN:**
- **Bull Market Performance**: [Which features work best in bull markets?]
- **Bear Market Performance**: [Which features work best in bear markets?]
- **Sideways Market Performance**: [Which features work best in sideways markets?]
- **High Volatility Performance**: [Which features work best in high vol?]
- **Low Volatility Performance**: [Which features work best in low vol?]

### Implementation Challenges
**PLEASE FILL IN:**
- **Data Quality Issues**: [Which features are sensitive to data quality?]
- **Computational Complexity**: [Which features are expensive to compute?]
- **Parameter Sensitivity**: [Which features are sensitive to parameter changes?]
- **Overfitting Risks**: [Which features are prone to overfitting?]

---

## Next Steps for Documentation Enhancement

**PLEASE PRIORITIZE:**
1. **Most Important Features to Document First**: [Which features need thesis statements most urgently?]
2. **Biggest Knowledge Gaps**: [Where do you need the most help with economic rationale?]
3. **Performance Validation Priorities**: [Which features need validation tests most urgently?]
4. **Missing Features**: [What features did I miss that should be included?]

---


### **NEW** Temporal Causality Compliance
- **Look-ahead Bias Checks**
- All features computed using point-in-time data from `crypto_loader`.
- Backtests enforce strict cutoff windows—no access to future bars or metrics.
- Rolling windows anchored to historical endpoints; no forward fill or smoothing.

- **Lag Integrity**:
- Quantile features (`q10`, `q50`, `q90`) use 1-bar lag to prevent leakage.
- Volatility metrics (`vol_risk`) use trailing windows (e.g., 20-bar) with no future overlap.
- Adaptive thresholds (`signal_thresh_adaptive`) derived from historical Sharpe distributions only.

**Point-in-Time Validation**
- Backtests use timestamp-indexed snapshot data; no retroactive overwrites.
- Feature exports include timestamped hashes for auditability.
- Agent pipelines validate feature freshness before execution.

**Time-aware Features**: 
- `prob_up`: Sensitive to quantile skew; must be lagged.
- `vol_risk`: Regime-aware volatility from trailing windows only.
- `enhanced_info_ratio`: Combines lagged signal and volatility; strict timestamp alignment required.


### Auditable Pipeline Implementation

**Feature Lineage**
- Each feature includes metadata: source loader, transformation steps, dependencies.
- Feature registry tracks upstream inputs and transformation logic.
- Markdown exports include lineage trees for agent consumption.

**Decision Traceability**
- Trades link to originating signal, feature values, and gating logic.
- Audit logs include `prob_up`, `abs_q50`, `spread`, and regime filters at decision time.
- Trade metadata includes timestamped feature snapshot and sizing rationale.

**Diagnostic Logging**
- Telemetry includes feature drift, regime transitions, and sizing activations.
- Logs capture failed gating conditions, missing data, and threshold violations.
- Debug mode outputs full feature vector per trade.

**Reproducibility**
- Fixed seeds, frozen data snapshots, and versioned feature definitions.
- Pipeline outputs hash-verified and stored with config fingerprints.
- Agent workflows include reproducibility checks before deployment.

**Error Observability**
- Failure modes trigger alerts: NaN propagation, zero sizing, threshold misalignment.
- Diagnostic panels visualize feature health, volatility regime, and sizing logic.
- Exception handling includes traceback logging and auto-retry for transient errors.


### **NEW** Contrarian Adaptation in Practice
- **Narrative Indicators**: 
- BTC dominance, stablecoin flows, and funding rate skew infer prevailing sentiment.
- Optional integration with Twitter sentiment and news clustering for narrative tagging.

- **Contrarian Signals**: 
- `abs_q50` in low-volume regimes signals mispriced conviction.
- Spread compression during fear spikes indicates reversal zones.
- Regime-aware Kelly sizing suppresses overconfidence during euphoric breakouts.

- **Fear & Greed Integration**: 
- Sentiment indices gated into sizing logic—reduce exposure during greed spikes.
- Contrarian filters activate when sentiment diverges from signal strength.

- **BTC Dominance Logic**: 
- Rising dominance during alt rallies signals risk-off rotation.
- Dominance inflection points gate altcoin exposure and amplify BTC bias.

- **Crisis Opportunities**: 
- Spread gating identifies high-confidence signals during volatility spikes.
- Sizing logic favors conservative entries with asymmetric payoff in panic regimes.

- **Uncomfortable Trades**: 
- Long BTC during March 2020 crash—tight spread, strong signal, fearful sentiment.
- Short ETH during NFT mania—weak signal, wide spread, euphoric sentiment.

---

**Instructions for Completion:**
1. Fill in details for each feature based on your experience
2. Add any missing features I didn't identify
3. Correct any misunderstandings in my descriptions
4. Prioritize which features are most important for thesis development
5. Highlight areas where you'd like help with economic rationale development
6. **NEW**: Address the three principle areas (Temporal Causality, Auditable Pipeline, Contrarian Logic) for system-level validation