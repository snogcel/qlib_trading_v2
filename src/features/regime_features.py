#!/usr/bin/env python3
"""
Unified Regime Feature Engineering
Consolidates 23+ scattered regime features into 7 standardized features
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class RegimeFeatureEngine:
    """
    Unified regime feature engineering class
    Replaces scattered implementations with standardized approach
    """
    
    def __init__(self, 
                 vol_percentiles: Tuple[float, float, float, float] = (0.1, 0.3, 0.7, 0.9),
                 sentiment_thresholds: Tuple[float, float, float, float] = (20, 35, 65, 80),
                 dominance_percentiles: Tuple[float, float] = (0.3, 0.7)):
        """
        Initialize regime feature engine with standardized thresholds
        
        Args:
            vol_percentiles: Volatility regime thresholds (10th, 30th, 70th, 90th)
            sentiment_thresholds: Fear & Greed thresholds (extreme_fear, fear, greed, extreme_greed)
            dominance_percentiles: BTC dominance thresholds (30th, 70th)
        """
        self.vol_percentiles = vol_percentiles
        self.sentiment_thresholds = sentiment_thresholds
        self.dominance_percentiles = dominance_percentiles
        
        # Cache for dynamic thresholds
        self._vol_thresholds = None
        self._dom_thresholds = None
    
    def calculate_regime_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Unified volatility regime classification
        Replaces: vol_extreme_high, vol_high, vol_low, vol_extreme_low, variance_regime_*
        
        Returns:
            pd.Series: Categorical volatility regime
        """
        if 'vol_risk' not in df.columns:
            print("⚠️  vol_risk not found, using fallback volatility calculation")
            # Fallback to basic volatility if vol_risk not available
            vol_col = df.get('volatility', df.get('$realized_vol_6', 0.01))
        else:
            vol_col = df['vol_risk']
        
        # Calculate dynamic thresholds if not cached
        if self._vol_thresholds is None:
            self._vol_thresholds = {
                'ultra_low': vol_col.quantile(self.vol_percentiles[0]),
                'low': vol_col.quantile(self.vol_percentiles[1]),
                'high': vol_col.quantile(self.vol_percentiles[2]),
                'extreme': vol_col.quantile(self.vol_percentiles[3])
            }
        
        # Classify volatility regimes
        conditions = [
            vol_col <= self._vol_thresholds['ultra_low'],
            (vol_col > self._vol_thresholds['ultra_low']) & (vol_col <= self._vol_thresholds['low']),
            (vol_col > self._vol_thresholds['low']) & (vol_col <= self._vol_thresholds['high']),
            (vol_col > self._vol_thresholds['high']) & (vol_col <= self._vol_thresholds['extreme']),
            vol_col > self._vol_thresholds['extreme']
        ]
        
        choices = ['ultra_low', 'low', 'medium', 'high', 'extreme']
        
        return pd.Series(
            np.select(conditions, choices, default='medium'),
            index=df.index,
            name='regime_volatility'
        )
    
    def calculate_regime_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """
        Unified sentiment regime classification
        Replaces: fg_extreme_fear, fg_extreme_greed
        
        Returns:
            pd.Series: Categorical sentiment regime
        """
        if '$fg_index' not in df.columns:
            print("⚠️  $fg_index not found, using neutral sentiment")
            return pd.Series('neutral', index=df.index, name='regime_sentiment')
        
        fg_index = df['$fg_index']
        
        conditions = [
            fg_index <= self.sentiment_thresholds[0],
            (fg_index > self.sentiment_thresholds[0]) & (fg_index <= self.sentiment_thresholds[1]),
            (fg_index > self.sentiment_thresholds[1]) & (fg_index < self.sentiment_thresholds[2]),
            (fg_index >= self.sentiment_thresholds[2]) & (fg_index < self.sentiment_thresholds[3]),
            fg_index >= self.sentiment_thresholds[3]
        ]
        
        choices = ['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed']
        
        return pd.Series(
            np.select(conditions, choices, default='neutral'),
            index=df.index,
            name='regime_sentiment'
        )
    
    def calculate_regime_dominance(self, df: pd.DataFrame) -> pd.Series:
        """
        Unified BTC dominance regime classification
        Replaces: btc_dom_high, btc_dom_low
        
        Returns:
            pd.Series: Categorical dominance regime
        """
        if '$btc_dom' not in df.columns:
            print("⚠️  $btc_dom not found, using balanced dominance")
            return pd.Series('balanced', index=df.index, name='regime_dominance')
        
        btc_dom = df['$btc_dom']
        
        # Calculate dynamic thresholds if not cached
        if self._dom_thresholds is None:
            self._dom_thresholds = {
                'low': btc_dom.quantile(self.dominance_percentiles[0]),
                'high': btc_dom.quantile(self.dominance_percentiles[1])
            }
        
        conditions = [
            btc_dom <= self._dom_thresholds['low'],
            (btc_dom > self._dom_thresholds['low']) & (btc_dom < self._dom_thresholds['high']),
            btc_dom >= self._dom_thresholds['high']
        ]
        
        choices = ['btc_low', 'balanced', 'btc_high']
        
        return pd.Series(
            np.select(conditions, choices, default='balanced'),
            index=df.index,
            name='regime_dominance'
        )
    
    def calculate_regime_crisis(self, regime_volatility: pd.Series, regime_sentiment: pd.Series) -> pd.Series:
        """
        Crisis regime detection
        Replaces: crisis_signal
        
        Returns:
            pd.Series: Binary crisis indicator
        """
        crisis_conditions = (
            (regime_volatility == 'extreme') & 
            (regime_sentiment == 'extreme_fear')
        )
        
        return crisis_conditions.astype(int).rename('regime_crisis')
    
    def calculate_regime_opportunity(self, regime_volatility: pd.Series, 
                                   regime_sentiment: pd.Series,
                                   regime_dominance: pd.Series) -> pd.Series:
        """
        Contrarian opportunity detection
        Replaces: btc_flight, fear_vol_spike
        
        Returns:
            pd.Series: Binary opportunity indicator
        """
        # Contrarian opportunities: fear + high vol, or extreme fear + btc flight
        opportunity_conditions = (
            ((regime_sentiment == 'extreme_fear') & (regime_volatility.isin(['high', 'extreme']))) |
            ((regime_sentiment == 'extreme_fear') & (regime_dominance == 'btc_high'))
        )
        
        return opportunity_conditions.astype(int).rename('regime_opportunity')
    
    def calculate_regime_stability(self, regime_volatility: pd.Series, 
                                 window: int = 20) -> pd.Series:
        """
        Regime stability measurement
        Replaces: regime_stability_ratio, variance_regime_change
        
        Returns:
            pd.Series: Continuous stability measure [0, 1]
        """
        # Count regime changes in rolling window
        regime_changes = (regime_volatility != regime_volatility.shift()).astype(int)
        change_frequency = regime_changes.rolling(window=window, min_periods=1).mean()
        
        # Convert to stability (1 - change_frequency)
        stability = (1 - change_frequency).clip(0, 1)
        
        return stability.rename('regime_stability')
    
    def calculate_regime_multiplier(self, regime_volatility: pd.Series,
                                  regime_sentiment: pd.Series,
                                  regime_crisis: pd.Series,
                                  regime_opportunity: pd.Series) -> pd.Series:
        """
        Unified regime-based position multiplier
        Replaces: regime_variance_multiplier, various multipliers
        
        Returns:
            pd.Series: Position multiplier [0.1, 5.0]
        """
        # Start with base multiplier
        multiplier = pd.Series(1.0, index=regime_volatility.index)
        
        # Volatility adjustments
        vol_adjustments = {
            'ultra_low': 1.5,   # +50% in ultra low vol
            'low': 1.2,         # +20% in low vol
            'medium': 1.0,      # Neutral
            'high': 0.7,        # -30% in high vol
            'extreme': 0.4      # -60% in extreme vol
        }
        
        for vol_regime, adjustment in vol_adjustments.items():
            multiplier = np.where(
                regime_volatility == vol_regime,
                multiplier * adjustment,
                multiplier
            )
        
        # Sentiment adjustments
        sentiment_adjustments = {
            'extreme_fear': 2.0,    # Contrarian boost
            'fear': 1.3,            # Mild contrarian
            'neutral': 1.0,         # No adjustment
            'greed': 0.8,           # Reduce in greed
            'extreme_greed': 0.6    # Strong reduction
        }
        
        for sentiment, adjustment in sentiment_adjustments.items():
            multiplier = np.where(
                regime_sentiment == sentiment,
                multiplier * adjustment,
                multiplier
            )
        
        # Crisis boost (additional multiplier)
        multiplier = np.where(regime_crisis == 1, multiplier * 3.0, multiplier)
        
        # Opportunity boost
        multiplier = np.where(regime_opportunity == 1, multiplier * 2.5, multiplier)
        
        # Clip to reasonable range
        return pd.Series(multiplier, index=regime_volatility.index).clip(0.1, 5.0).rename('regime_multiplier')
    
    def add_temporal_quantile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add economically-justified temporal quantile features
        Each feature captures a specific market inefficiency with clear economic rationale
        
        Economic Thesis: Market information doesn't instantly reflect in prices,
        creating temporal patterns in quantile predictions that can be exploited.
        
        Returns:
            pd.DataFrame: df with economically-justified temporal features added
        """
        print("⏰ Adding economically-justified temporal quantile features...")
        
        # Ensure required quantile columns exist
        required_cols = ['q10', 'q50', 'q90']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️  Missing required column: {col}")
                return df
        
        # FEATURE 1: MOMENTUM PERSISTENCE (Information Flow Theory)
        # Economic Rationale: Information doesn't instantly reflect in prices
        # Supply/Demand: Persistent directional pressure indicates sustained order flow imbalance
        df['q50_momentum_3'] = df['q50'].rolling(3, min_periods=2).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0
        )
        
        # FEATURE 2: UNCERTAINTY EVOLUTION (Market Microstructure Theory)
        # Economic Rationale: Prediction uncertainty reflects market liquidity conditions
        # Supply/Demand: Widening spreads suggest increasing disagreement between buyers/sellers
        if 'spread' not in df.columns:
            df['spread'] = df['q90'] - df['q10']
        df['spread_momentum_3'] = df['spread'].rolling(3, min_periods=2).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0
        )
        
        # FEATURE 3: CONSENSUS STABILITY (Information Aggregation Theory)
        # Economic Rationale: Stable predictions indicate strong market consensus
        # Supply/Demand: Low volatility in predictions suggests balanced order flow
        df['q50_stability_6'] = df['q50'].rolling(6, min_periods=3).std().fillna(0)
        
        # FEATURE 4: REGIME PERSISTENCE (Behavioral Finance)
        # Economic Rationale: Market regimes persist due to behavioral biases and momentum
        # Supply/Demand: Persistent directional bias indicates sustained pressure
        q50_positive = (df['q50'] > 0).astype(int)
        df['q50_regime_persistence'] = q50_positive.groupby(
            (q50_positive != q50_positive.shift()).cumsum()
        ).cumcount() + 1
        
        # FEATURE 5: PREDICTION CONFIDENCE (Risk Management Theory)
        # Economic Rationale: Position size should reflect prediction confidence
        # Supply/Demand: Narrow spreads relative to signal suggest strong consensus
        df['prediction_confidence'] = 1.0 / (1.0 + df['spread'] / np.maximum(np.abs(df['q50']), 0.001))
        
        # FEATURE 6: DIRECTIONAL CONSISTENCY (Trend Following Theory)
        # Economic Rationale: Consistent directional predictions indicate trend strength
        # Supply/Demand: Persistent direction suggests sustained order flow imbalance
        df['q50_direction_consistency'] = df['q50'].rolling(6, min_periods=3).apply(
            lambda x: (np.sign(x) == np.sign(x.iloc[-1])).mean() if len(x) >= 3 else 0.5
        ).fillna(0.5)
        
        print(f"✅ Added 6 economically-justified temporal features:")
        print(f"   • q50_momentum_3: Information flow persistence")
        print(f"   • spread_momentum_3: Market uncertainty evolution") 
        print(f"   • q50_stability_6: Consensus stability measure")
        print(f"   • q50_regime_persistence: Behavioral momentum")
        print(f"   • prediction_confidence: Risk-adjusted confidence")
        print(f"   • q50_direction_consistency: Trend strength indicator")
        
        return df

    def generate_all_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all unified regime features at once
        
        Returns:
            pd.DataFrame: Original df with 7 regime features + temporal quantile features
        """
        print("🏛️  Generating unified regime features...")
        
        # Core regime classifications
        regime_volatility = self.calculate_regime_volatility(df)
        regime_sentiment = self.calculate_regime_sentiment(df)
        regime_dominance = self.calculate_regime_dominance(df)
        
        # Composite regimes
        regime_crisis = self.calculate_regime_crisis(regime_volatility, regime_sentiment)
        regime_opportunity = self.calculate_regime_opportunity(regime_volatility, regime_sentiment, regime_dominance)
        regime_stability = self.calculate_regime_stability(regime_volatility)
        regime_multiplier = self.calculate_regime_multiplier(regime_volatility, regime_sentiment, regime_crisis, regime_opportunity)
        
        # Add all features to dataframe
        result_df = df.copy()
        result_df['regime_volatility'] = regime_volatility
        result_df['regime_sentiment'] = regime_sentiment
        result_df['regime_dominance'] = regime_dominance
        result_df['regime_crisis'] = regime_crisis
        result_df['regime_opportunity'] = regime_opportunity
        result_df['regime_stability'] = regime_stability
        result_df['regime_multiplier'] = regime_multiplier
        
        # Add temporal quantile features (Phase 1 enhancement)
        result_df = self.add_temporal_quantile_features(result_df)
        
        # Print summary statistics
        self._print_regime_summary(result_df)
        
        return result_df
    
    def validate_economic_logic(self, df: pd.DataFrame) -> Dict:
        """
        Validate that temporal features behave according to economic theory
        Following thesis-first principle: each feature must have explainable behavior
        
        Returns:
            Dict: Validation results with economic rationale
        """
        print("\n🧪 VALIDATING ECONOMIC LOGIC OF TEMPORAL FEATURES:")
        
        validations = []
        
        # Test 1: Momentum should correlate with actual Q50 changes
        if 'q50_momentum_3' in df.columns:
            actual_momentum = df['q50'].diff(3)
            predicted_momentum = df['q50_momentum_3']
            
            # Remove NaN values for correlation
            valid_mask = ~(actual_momentum.isna() | predicted_momentum.isna())
            if valid_mask.sum() > 10:
                correlation = actual_momentum[valid_mask].corr(predicted_momentum[valid_mask])
                
                validations.append({
                    'test': 'Q50 momentum captures actual momentum',
                    'result': correlation > 0.7,
                    'value': f"{correlation:.3f}",
                    'economic_rationale': 'Information flow persistence theory',
                    'status': '✅' if correlation > 0.7 else '⚠️'
                })
        
        # Test 2: Prediction confidence should be lower during high volatility
        if 'prediction_confidence' in df.columns and 'vol_risk' in df.columns:
            high_vol_periods = df['vol_risk'] > df['vol_risk'].quantile(0.8)
            low_vol_periods = df['vol_risk'] < df['vol_risk'].quantile(0.2)
            
            avg_confidence_high_vol = df.loc[high_vol_periods, 'prediction_confidence'].mean()
            avg_confidence_low_vol = df.loc[low_vol_periods, 'prediction_confidence'].mean()
            
            validations.append({
                'test': 'Lower confidence during high volatility',
                'result': avg_confidence_high_vol < avg_confidence_low_vol,
                'value': f"High vol: {avg_confidence_high_vol:.3f}, Low vol: {avg_confidence_low_vol:.3f}",
                'economic_rationale': 'Market microstructure theory',
                'status': '✅' if avg_confidence_high_vol < avg_confidence_low_vol else '⚠️'
            })
        
        # Test 3: Regime persistence should be higher during trending periods
        if 'q50_regime_persistence' in df.columns and 'q50_momentum_3' in df.columns:
            high_momentum_periods = np.abs(df['q50_momentum_3']) > df['q50_momentum_3'].abs().quantile(0.8)
            low_momentum_periods = np.abs(df['q50_momentum_3']) < df['q50_momentum_3'].abs().quantile(0.2)
            
            avg_persistence_trending = df.loc[high_momentum_periods, 'q50_regime_persistence'].mean()
            avg_persistence_ranging = df.loc[low_momentum_periods, 'q50_regime_persistence'].mean()
            
            validations.append({
                'test': 'Higher persistence during trending periods',
                'result': avg_persistence_trending > avg_persistence_ranging,
                'value': f"Trending: {avg_persistence_trending:.1f}, Ranging: {avg_persistence_ranging:.1f}",
                'economic_rationale': 'Behavioral finance momentum theory',
                'status': '✅' if avg_persistence_trending > avg_persistence_ranging else '⚠️'
            })
        
        # Test 4: Spread momentum should correlate with volatility changes
        if 'spread_momentum_3' in df.columns and 'vol_risk' in df.columns:
            vol_change = df['vol_risk'].diff(3)
            spread_momentum = df['spread_momentum_3']
            
            valid_mask = ~(vol_change.isna() | spread_momentum.isna())
            if valid_mask.sum() > 10:
                correlation = vol_change[valid_mask].corr(spread_momentum[valid_mask])
                
                validations.append({
                    'test': 'Spread momentum correlates with volatility changes',
                    'result': correlation > 0.3,
                    'value': f"{correlation:.3f}",
                    'economic_rationale': 'Market microstructure uncertainty theory',
                    'status': '✅' if correlation > 0.3 else '⚠️'
                })
        
        # Print validation results
        for validation in validations:
            print(f"   {validation['status']} {validation['test']}")
            print(f"      Value: {validation['value']}")
            print(f"      Rationale: {validation['economic_rationale']}")
        
        # Summary
        passed = sum(1 for v in validations if v['result'])
        total = len(validations)
        print(f"\n📊 Economic Logic Validation: {passed}/{total} tests passed")
        
        return {
            'validations': validations,
            'passed': passed,
            'total': total,
            'success_rate': passed / total if total > 0 else 0
        }

    def _print_regime_summary(self, df: pd.DataFrame):
        """Print summary of regime feature distributions"""
        print("\n📊 REGIME FEATURE SUMMARY:")
        
        # Volatility regime distribution
        vol_dist = df['regime_volatility'].value_counts(normalize=True) * 100
        print(f"\n🌪️  Volatility Regimes:")
        for regime, pct in vol_dist.items():
            print(f"   {regime}: {pct:.1f}%")
        
        # Sentiment regime distribution
        sent_dist = df['regime_sentiment'].value_counts(normalize=True) * 100
        print(f"\n😱 Sentiment Regimes:")
        for regime, pct in sent_dist.items():
            print(f"   {regime}: {pct:.1f}%")
        
        # Dominance regime distribution
        dom_dist = df['regime_dominance'].value_counts(normalize=True) * 100
        print(f"\n₿  Dominance Regimes:")
        for regime, pct in dom_dist.items():
            print(f"   {regime}: {pct:.1f}%")
        
        # Binary regime statistics
        crisis_pct = df['regime_crisis'].mean() * 100
        opportunity_pct = df['regime_opportunity'].mean() * 100
        print(f"\n🚨 Crisis periods: {crisis_pct:.2f}%")
        print(f"🎯 Opportunity periods: {opportunity_pct:.2f}%")
        
        # Stability and multiplier statistics
        stability_mean = df['regime_stability'].mean()
        multiplier_stats = df['regime_multiplier'].describe()
        print(f"\n📈 Average regime stability: {stability_mean:.3f}")
        print(f"⚖️  Multiplier range: [{multiplier_stats['min']:.2f}, {multiplier_stats['max']:.2f}]")
        print(f"   Average multiplier: {multiplier_stats['mean']:.2f}")
        
        # Temporal feature summary if they exist
        temporal_features = ['q50_momentum_3', 'spread_momentum_3', 'q50_stability_6', 
                           'prediction_confidence', 'q50_regime_persistence', 'q50_direction_consistency']
        
        existing_temporal = [f for f in temporal_features if f in df.columns]
        if existing_temporal:
            print(f"\n⏰ Temporal Features Summary:")
            for feature in existing_temporal:
                stats = df[feature].describe()
                print(f"   {feature}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        
        # Run economic logic validation if temporal features exist
        if existing_temporal:
            self.validate_economic_logic(df)


# Convenience functions for backward compatibility
def create_regime_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate all regime features
    """
    engine = RegimeFeatureEngine(**kwargs)
    return engine.generate_all_regime_features(df)


def add_temporal_quantile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add just temporal quantile features
    Phase 1 enhancement - can be used independently
    """
    engine = RegimeFeatureEngine()
    return engine.add_temporal_quantile_features(df)


def get_regime_multiplier(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Convenience function to get just the regime multiplier
    """
    engine = RegimeFeatureEngine(**kwargs)
    regime_volatility = engine.calculate_regime_volatility(df)
    regime_sentiment = engine.calculate_regime_sentiment(df)
    regime_crisis = engine.calculate_regime_crisis(regime_volatility, regime_sentiment)
    regime_opportunity = engine.calculate_regime_opportunity(regime_volatility, regime_sentiment, 
                                                           engine.calculate_regime_dominance(df))
    
    return engine.calculate_regime_multiplier(regime_volatility, regime_sentiment, regime_crisis, regime_opportunity)


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Regime Feature Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'vol_risk': np.random.exponential(0.01, n_samples),
        '$fg_index': np.random.uniform(0, 100, n_samples),
        '$btc_dom': np.random.uniform(40, 70, n_samples),
        'q50': np.random.normal(0, 0.01, n_samples)
    })
    
    # Test regime feature generation
    engine = RegimeFeatureEngine()
    result = engine.generate_all_regime_features(sample_data)
    
    print(f"\n✅ Generated {len([col for col in result.columns if col.startswith('regime_')])} regime features")
    print("🎉 Regime Feature Engine ready for production!")