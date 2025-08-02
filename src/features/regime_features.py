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
    
    def generate_all_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all unified regime features at once
        
        Returns:
            pd.DataFrame: Original df with 7 new regime features
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
        
        # Print summary statistics
        self._print_regime_summary(result_df)
        
        return result_df
    
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


# Convenience functions for backward compatibility
def create_regime_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to generate all regime features
    """
    engine = RegimeFeatureEngine(**kwargs)
    return engine.generate_all_regime_features(df)


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