# Signal Analysis CSV - Pivot Table Ready Format

## Overview

The `signal_analysis_pivot.csv` file contains comprehensive data about every signal generated and whether it resulted in a trade. This format is optimized for pivot table analysis in Excel, Google Sheets, or Python.

## Key Columns for Analysis

### **Core Signal Data**
- `signal_direction`: LONG, SHORT, HOLD
- `signal_strength`: 0-1 signal strength score
- `confidence`: 0-10 confidence level
- `target_pct`: Calculated target position percentage
- `trade_executed`: Boolean - was a trade actually executed?
- `trade_outcome`: EXECUTED, HOLD_POSITION, HOLD_NO_POSITION

### **Quantile Data**
- `q10`, `q50`, `q90`: Raw quantile predictions
- `abs_q50`: Absolute value of q50 (signal strength)
- `spread`: q90 - q10 (uncertainty measure)
- `prob_up`: Calculated probability of upward movement

### 🎲 **Probability Breakdown**
- `short_prob`: Probability of downward movement
- `neutral_prob`: Probability of neutral movement  
- `long_prob`: Probability of upward movement
- `prob_directional`: Max directional probability

### 🏷️ **Categorical Buckets (Perfect for Pivot Tables)**
- `strength_bucket`: Very_Low, Low, Medium, High, Very_High
- `confidence_bucket`: Very_Low, Low, Medium, High, Very_High
- `target_bucket`: Very_Small, Small, Medium, Large, Very_Large
- `spread_bucket`: Tight, Normal, Wide, Very_Wide
- `prob_confidence_level`: Low, Medium, High

### **Time Analysis**
- `hour`: 0-23 hour of day
- `day_of_week`: Monday, Tuesday, etc.
- `month`: 1-12
- `quarter`: 1-4
- `weekend`: Boolean
- `market_hours`: Boolean (Always True for crypto - 24/7 markets)

### **Boolean Flags (Easy Filtering)**
- `signal_above_thresh`: Signal above adaptive threshold
- `high_confidence`: Confidence >= 7
- `strong_signal`: Signal strength >= 0.7
- `has_position`: Currently holding a position
- `is_long_signal`, `is_short_signal`, `is_hold_signal`: Signal type flags

### 📈 **Derived Metrics**
- `signal_quality_score`: Combined strength × confidence × directional probability
- `risk_reward_ratio`: Signal strength / spread
- `position_utilization`: Current position / max position

## Pivot Table Examples

### 1. **Signal Direction vs Confidence Analysis**
```
Rows: signal_direction
Columns: confidence_bucket  
Values: trade_executed (Average)
```
**Shows**: Which signal directions execute more at different confidence levels

### 2. **Time-Based Execution Patterns**
```
Rows: hour
Columns: signal_direction
Values: trade_executed (Count)
```
**Shows**: When different signal types are most/least likely to execute

### 3. **Signal Quality vs Execution Rate**
```
Rows: strength_bucket
Columns: above_threshold
Values: trade_executed (Average)
```
**Shows**: How signal quality affects execution probability

### 4. **Risk Analysis**
```
Rows: spread_bucket
Columns: target_bucket
Values: signal_quality_score (Average)
```
**Shows**: Relationship between uncertainty (spread) and position sizing

### 5. **Hold Pattern Analysis**
```
Rows: hold_reason
Columns: confidence_bucket
Values: signal_strength (Average)
```
**Shows**: Why signals result in holds vs trades

## Key Questions You Can Answer

### 🔍 **Signal Quality**
- Which signal types have the highest execution rate?
- Do high-confidence signals actually execute more often?
- What's the relationship between signal strength and actual trades?

### **Timing Patterns**
- Are there specific hours/days when execution rates are higher?
- Do weekend signals behave differently than weekday signals?
- Is there seasonality in signal generation or execution?

### **Threshold Effectiveness**
- Do signals above the adaptive threshold perform better?
- What's the optimal confidence level for trade execution?
- How does spread (uncertainty) affect trading decisions?

### 💰 **Position Sizing**
- What target position sizes are most common?
- Do larger target positions correlate with higher confidence?
- How does risk-reward ratio affect position sizing?

### **Hold Analysis**
- Why do signals result in holds instead of trades?
- Which hold reasons are most common?
- Do held signals have different characteristics than executed ones?

## Sample Analysis Queries

### Excel/Google Sheets Pivot Tables

**Most Effective Signal Types:**
- Rows: `signal_direction`, `confidence_bucket`
- Values: `trade_executed` (Average), `signal_strength` (Average)
- Filter: `above_threshold` = TRUE

**Time-Based Performance:**
- Rows: `day_of_week`, `hour`  
- Values: `trade_executed` (Count), `signal_quality_score` (Average)
- Columns: `signal_direction`

**Risk-Adjusted Analysis:**
- Rows: `spread_bucket`, `strength_bucket`
- Values: `target_pct` (Average), `risk_reward_ratio` (Average)
- Filter: `trade_executed` = TRUE

## Files Generated

1. **`signal_analysis_pivot.csv`** - Main analysis data (49 columns)
2. **`signal_summary_stats.json`** - Quick summary statistics
3. **`portfolio_history.csv`** - Portfolio evolution over time
4. **`trades.csv`** - Individual trade records
5. **`holds_analysis.csv`** - Detailed hold state analysis

## Usage Tips

1. **Start with summary stats** - Check `signal_summary_stats.json` for quick insights
2. **Use filters liberally** - Boolean columns make filtering easy
3. **Combine time and signal analysis** - Look for patterns across different time periods
4. **Focus on execution rates** - `trade_executed` is your key success metric
5. **Analyze hold reasons** - Understanding why signals don't execute is crucial

## Next Steps

1. Run a real backtest with your actual data
2. Load the CSV into your preferred analysis tool
3. Create pivot tables based on the examples above
4. Use the `analyze_signal_execution.py` script for automated insights
5. Iterate on signal generation logic based on findings

This format gives you complete visibility into the signal generation → trade execution pipeline, making it easy to identify optimization opportunities.