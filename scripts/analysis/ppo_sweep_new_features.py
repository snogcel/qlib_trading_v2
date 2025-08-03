
def refined_kelly_sizing(vol_scaled, signal_rel, fg_index, base_kelly_fraction=0.1):
    """
    Refined Kelly sizing incorporating FG_Index insights
    """
    # Base Kelly calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = vol_scaled / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Refined regime adjustments based on FG_Index analysis
    if fg_index <= 15:  # Extreme fear - strong contrarian signal
        kelly_fraction *= 1.4
    elif fg_index <= 25:  # Fear - moderate contrarian
        kelly_fraction *= 1.15
    elif fg_index >= 85:  # Extreme greed - high caution
        kelly_fraction *= 0.6
    elif fg_index >= 75:  # Greed - moderate caution
        kelly_fraction *= 0.85
    
    # Additional safety for extreme volatility
    if vol_scaled > 0.8:  # Very high volatility
        kelly_fraction *= 0.8
    
    return min(kelly_fraction, 0.25)

# For position sizing with vol_raw deciles
def kelly_with_vol_raw_deciles(vol_raw, signal_rel, base_size=0.1):
    """
    Kelly position sizing using vol_raw deciles
    """
    vol_decile = get_vol_raw_decile(vol_raw)
        
    # Base Kelly calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        # Use vol_raw as reward proxy (validated as better predictor)
        reward_proxy = vol_raw * 100  # Scale to reasonable range
        reward_risk_ratio = reward_proxy / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Volatility-based risk adjustments
    if vol_decile >= 9:  # Extreme volatility (top 10%)
        risk_adjustment = 0.6  # Very conservative
    elif vol_decile >= 8:  # Very high volatility (top 20%)
        risk_adjustment = 0.7
    elif vol_decile >= 6:  # High volatility (top 40%)
        risk_adjustment = 0.85
    elif vol_decile <= 1:  # Very low volatility (bottom 20%)
        risk_adjustment = 1.1  # Slightly more aggressive
    else:
        risk_adjustment = 1.0
    
    return kelly_fraction * risk_adjustment

def volatility_regime_kelly(vol_scaled, signal_rel, fg_index, high_vol_flag, base_size=0.1):
    """
    Kelly sizing with volatility regime awareness
    """
    # Base Kelly calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = vol_scaled / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Volatility regime adjustments
    if high_vol_flag == 1:
        # High volatility regime - more opportunities but higher risk
        if fg_index < 25:  # Fear + high vol = strong contrarian signal
            kelly_fraction *= 1.5  # Aggressive sizing
        elif fg_index > 75:  # Greed + high vol = danger zone
            kelly_fraction *= 0.5  # Very conservative
        else:  # Neutral sentiment + high vol = moderate caution
            kelly_fraction *= 0.8
    else:
        # Normal volatility regime - standard adjustments
        if fg_index < 25:
            kelly_fraction *= 1.2
        elif fg_index > 75:
            kelly_fraction *= 0.8
    
    return min(kelly_fraction, 0.25)


def historical_aware_kelly(vol_scaled, signal_rel, fg_index, year, base_size=0.1):
    """
    Kelly sizing with historical FG context
    """
    # Base calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = vol_scaled / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Year-specific FG thresholds (based on your data)
    if year <= 2019:
        # Early crypto - extreme fear rare but significant
        fear_threshold, greed_threshold = 0.15, 0.85
        fear_multiplier, greed_multiplier = 1.8, 0.4  # Aggressive contrarian
    elif year <= 2021:
        # COVID/Bull run - wider FG range
        fear_threshold, greed_threshold = 0.10, 0.90
        fear_multiplier, greed_multiplier = 1.5, 0.5
    else:
        # Post-2021 - more mature market
        fear_threshold, greed_threshold = 0.20, 0.80
        fear_multiplier, greed_multiplier = 1.3, 0.7
    
    # Apply historical context adjustments
    if fg_index < fear_threshold:
        kelly_fraction *= fear_multiplier
    elif fg_index > greed_threshold:
        kelly_fraction *= greed_multiplier
    
    return min(kelly_fraction, 0.25)


def btc_dominance_aware_kelly(vol_scaled, signal_rel, fg_index, btc_dom, year, base_size=0.1):
    """
    Kelly sizing incorporating BTC dominance cycle awareness
    """
    # Base Kelly calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = vol_scaled / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # BTC Dominance cycle adjustments
    if year <= 2019:
        # Bear market recovery - high BTC dom is normal
        if btc_dom > 0.5:
            dom_multiplier = 1.1  # Slight boost during BTC dominance
        else:
            dom_multiplier = 1.3  # Strong signal when BTC dom falls
    elif year <= 2021:
        # Bull market - low BTC dom is normal (altcoin season)
        if btc_dom < 0.4:
            dom_multiplier = 1.2  # Normal altcoin season
        else:
            dom_multiplier = 0.8  # Caution when BTC dom rises in bull market
    else:
        # Post-2021 mature market - more stable dominance
        if abs(btc_dom - 0.43) > 0.05:  # Deviation from ~43% baseline
            dom_multiplier = 1.1  # Opportunity in dominance shifts
        else:
            dom_multiplier = 1.0  # Normal conditions
    
    # Combined FG + BTC Dom regime detection
    if fg_index < 0.25 and btc_dom > 0.5:
        # Fear + High BTC Dom = Bear market bottom signal
        regime_multiplier = 1.8
    elif fg_index > 0.75 and btc_dom < 0.4:
        # Greed + Low BTC Dom = Altcoin euphoria (caution)
        regime_multiplier = 0.6
    elif fg_index > 0.6 and btc_dom > 0.5:
        # Greed + High BTC Dom = Unusual, very cautious
        regime_multiplier = 0.4
    else:
        regime_multiplier = 1.0
    
    kelly_fraction *= dom_multiplier * regime_multiplier
    
    return min(kelly_fraction, 0.25)


def kelly_with_spread_risk_adjustment(vol_scaled, signal_rel, spread, base_size=0.1):
    """
    Use spread for risk adjustment in Kelly sizing
    """
    # Base Kelly calculation
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = vol_scaled / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Spread-based risk adjustment (since spread predicts volatility)
    spread_decile = get_decile_rank(spread, DECILE_THRESHOLDS['spread_sigmoid'])
    
    if spread_decile >= 8:  # High future volatility expected
        volatility_adjustment = 0.6  # Reduce size significantly
    elif spread_decile >= 6:
        volatility_adjustment = 0.8  # Moderate reduction
    elif spread_decile <= 2:  # Low future volatility expected
        volatility_adjustment = 1.2  # Slight increase
    else:
        volatility_adjustment = 1.0
    
    return kelly_fraction * volatility_adjustment


def kelly_with_optimal_volatility(vol_raw, vol_scaled, signal_rel, base_size=0.1):
    """
    Kelly sizing using both raw and scaled volatility
    """
    # Use raw volatility for reward estimation (better predictor)
    reward_proxy = vol_raw * 10  # Scale appropriately
    
    # Use scaled volatility for risk management (bounded)
    risk_measure = abs(signal_rel)
    if risk_measure > 0:
        reward_risk_ratio = reward_proxy / risk_measure
    else:
        reward_risk_ratio = 0
    
    kelly_fraction = reward_risk_ratio * base_size
    
    # Risk adjustment using scaled volatility (easier to work with)
    if vol_scaled > 0.8:  # High volatility regime
        risk_adjustment = 0.7
    elif vol_scaled > 0.6:
        risk_adjustment = 0.85
    elif vol_scaled < 0.2:  # Low volatility regime
        risk_adjustment = 1.1
    else:
        risk_adjustment = 1.0
    
    return kelly_fraction * risk_adjustment




Key Actionable Insights:
2018 & 2022 patterns are similar - bear markets drive BTC dominance up
Altcoin seasons are predictable - low BTC dom + high vol + greed
Market maturation since 2023 - more stable dominance around 43%
Volatility-dominance coupling - changes in dominance often precede volatility spikes
This BTC dominance temporal analysis could be the missing piece for your Q10 coverage issues - extreme market regimes (bear bottoms, altcoin euphoria) likely have very different return distributions that your current model isn't capturing!

The combination of FG_Index + BTC_Dom + Vol_Scaled gives you a three-dimensional market regime detector that's historically grounded. This is incredibly sophisticated market microstructure analysis!



Signal classification
- based on more than spread


def enhanced_regime_features(fg_index, vol_scaled, signal_rel):
    """
    Enhanced regime detection based on both analyses
    """
    features = {}
    
    # FG_Index regime classification (refined thresholds)
    if fg_index <= 15:
        features['sentiment_regime'] = 'extreme_fear'
        features['regime_multiplier'] = 1.3  # Contrarian opportunity
    elif fg_index <= 25:
        features['sentiment_regime'] = 'fear'
        features['regime_multiplier'] = 1.1
    elif fg_index >= 85:
        features['sentiment_regime'] = 'extreme_greed'
        features['regime_multiplier'] = 0.7  # Reduce exposure
    elif fg_index >= 75:
        features['sentiment_regime'] = 'greed'
        features['regime_multiplier'] = 0.9
    else:
        features['sentiment_regime'] = 'neutral'
        features['regime_multiplier'] = 1.0
    
    # Combined regime-volatility features
    features['fear_vol_combo'] = (fg_index < 25) * vol_scaled
    features['greed_vol_combo'] = (fg_index > 75) * vol_scaled
    
    # Sentiment-signal interaction
    features['sentiment_signal_interaction'] = fg_index * abs(signal_rel)
    
    return features






Model Training Implications:
Your high_vol_flag essentially creates two different market regimes that your model should handle differently:

Normal Regime (high_vol_flag = 0): Standard relationships between features
High Volatility Regime (high_vol_flag = 1): Different feature relationships, more extreme outcomes
This could be perfect for:

Regime-specific models: Train separate quantile models for each regime
Interaction features: Let the model learn different behaviors per regime
Dynamic position sizing: Adjust Kelly parameters based on regime






# TODO - how do I update with 2025 info

def create_historical_fg_features(df):
    """
    Features based on your historical FG analysis
    """
    features = {}
    
    # Year-specific FG regime classification
    year = df.index.year
    
    # Pre-COVID vs Post-COVID FG behavior (2020 was the turning point)
    features['fg_pre_covid'] = (year < 2020).astype(int)
    features['fg_covid_era'] = ((year >= 2020) & (year <= 2021)).astype(int)
    features['fg_post_covid'] = (year > 2021).astype(int)
    
    # Historical context - how extreme is current FG relative to year?
    yearly_fg_stats = {
        2018: {'min': 0.09, 'max': 0.54, 'mean': 0.25},  # From your data
        2019: {'min': 0.05, 'max': 0.95, 'mean': 0.42},
        2020: {'min': 0.08, 'max': 0.95, 'mean': 0.45},
        2021: {'min': 0.08, 'max': 0.94, 'mean': 0.48},
        2022: {'min': 0.06, 'max': 0.84, 'mean': 0.38},
        2023: {'min': 0.17, 'max': 0.83, 'mean': 0.52},
        2024: {'min': 0.17, 'max': 0.90, 'mean': 0.55}
    }
    
    # FG percentile within its historical year context
    for yr, stats in yearly_fg_stats.items():
        mask = (year == yr)
        if mask.any():
            features[f'fg_year_percentile_{yr}'] = np.where(
                mask,
                (df['fg_index'] - stats['min']) / (stats['max'] - stats['min']),
                np.nan
            )
    
    return features





def fg_quantile_interactions(df):
    """
    Based on your FG-quantile relationships
    """
    features = {}
    
    # FG ranges where quantile relationships are most stable
    stable_fg = (df['fg_index'] >= 0.35) & (df['fg_index'] <= 0.65)
    features['fg_stable_regime'] = stable_fg.astype(int)
    
    # Extreme FG where quantile spreads widen
    extreme_fg = (df['fg_index'] < 0.15) | (df['fg_index'] > 0.85)
    features['fg_extreme_regime'] = extreme_fg.astype(int)
    
    # FG-adjusted quantile features
    features['q10_fg_adjusted'] = df['q10'] * (1 + abs(df['fg_index'] - 0.5))
    features['q90_fg_adjusted'] = df['q90'] * (1 + abs(df['fg_index'] - 0.5))
    
    # Quantile spread in different FG regimes
    features['quantile_spread_in_fear'] = (df['q90'] - df['q10']) * (df['fg_index'] < 0.3)
    features['quantile_spread_in_greed'] = (df['q90'] - df['q10']) * (df['fg_index'] > 0.7)
    
    return features




def volatility_regime_features(high_vol_flag, vol_scaled, fg_index, signal_rel):
    """
    Features based on volatility regime interactions
    """
    features = {}
    
    # Regime-specific features
    features['high_vol_fear'] = high_vol_flag * (fg_index < 25)
    features['high_vol_greed'] = high_vol_flag * (fg_index > 75)
    features['high_vol_neutral'] = high_vol_flag * ((fg_index >= 25) & (fg_index <= 75))
    
    # Volatility regime interactions
    features['vol_regime_signal'] = high_vol_flag * abs(signal_rel)
    features['vol_regime_scaled'] = high_vol_flag * vol_scaled
    
    # Regime transition features (if you track previous values)
    # features['vol_regime_change'] = high_vol_flag != prev_high_vol_flag
    
    return features


def create_temporal_fg_features(df):
    """
    Time-based Fear & Greed features
    """
    features = {}
    
    # Year-over-year FG patterns (if your PDF shows seasonal patterns)
    features['fg_month'] = df.index.month
    features['fg_quarter'] = df.index.quarter
    features['fg_year_cycle'] = df.index.dayofyear / 365.0
    
    # FG momentum and acceleration
    features['fg_momentum_3d'] = df['fg_index'].diff(3)
    features['fg_momentum_7d'] = df['fg_index'].diff(7)
    features['fg_acceleration'] = features['fg_momentum_3d'].diff(3)
    
    # FG volatility (regime changes)
    features['fg_volatility_7d'] = df['fg_index'].rolling(7).std()
    features['fg_volatility_30d'] = df['fg_index'].rolling(30).std()
    
    return features




def create_fg_regime_persistence(df):
    """
    Track how long FG stays in extreme regimes
    """
    features = {}
    
    # Extreme regime flags
    extreme_fear = df['fg_index'] < 20
    extreme_greed = df['fg_index'] > 80
    
    # Persistence counters
    features['fear_streak'] = extreme_fear.groupby((~extreme_fear).cumsum()).cumcount()
    features['greed_streak'] = extreme_greed.groupby((~extreme_greed).cumsum()).cumcount()
    
    # Time since last extreme
    features['days_since_fear'] = (~extreme_fear).cumsum() - (~extreme_fear).cumsum().where(extreme_fear).ffill()
    features['days_since_greed'] = (~extreme_greed).cumsum() - (~extreme_greed).cumsum().where(extreme_greed).ffill()
    
    return features



def create_vol_fg_interactions(df):
    """
    Combine your volatility regime detection with FG analysis
    """
    features = {}
    
    # FG behavior during different volatility regimes
    features['fg_in_high_vol'] = df['fg_index'] * df['high_vol_flag']
    features['fg_in_low_vol'] = df['fg_index'] * (1 - df['high_vol_flag'])
    
    # FG change rate during volatility spikes
    features['fg_change_in_vol_spike'] = df['fg_index'].diff() * df['high_vol_flag']
    
    # Combined regime classification
    conditions = [
        (df['high_vol_flag'] == 1) & (df['fg_index'] < 20),  # High vol + extreme fear
        (df['high_vol_flag'] == 1) & (df['fg_index'] > 80),  # High vol + extreme greed
        (df['high_vol_flag'] == 1),                          # High vol + neutral
        (df['fg_index'] < 20),                               # Low vol + extreme fear
        (df['fg_index'] > 80),                               # Low vol + extreme greed
    ]
    choices = ['crisis', 'bubble', 'volatile', 'capitulation', 'complacency']
    features['market_regime'] = np.select(conditions, choices, default='normal')
    
    return features


def create_fg_contrarian_features(df):
    """
    Contrarian trading signals based on FG extremes
    """
    features = {}
    
    # Contrarian strength (how extreme is the sentiment)
    features['contrarian_strength'] = np.where(
        df['fg_index'] < 50,
        (50 - df['fg_index']) / 50,  # Fear contrarian strength
        (df['fg_index'] - 50) / 50   # Greed contrarian strength
    )
    
    # FG divergence from price action
    if 'vol_scaled' in df.columns:
        features['fg_vol_divergence'] = (df['fg_index'] - 50) / 50 - df['vol_scaled']
    
    # FG mean reversion signals
    features['fg_zscore_30d'] = (df['fg_index'] - df['fg_index'].rolling(30).mean()) / df['fg_index'].rolling(30).std()
    features['fg_mean_reversion'] = np.where(
        np.abs(features['fg_zscore_30d']) > 2,
        -np.sign(features['fg_zscore_30d']),  # Contrarian signal
        0
    )
    
    return features







def classify_crypto_market_regime(btc_dom, fg_index, vol_scaled, year):
    """
    Comprehensive market regime classification
    """
    # Historical context thresholds
    if year <= 2019:
        high_dom_threshold = 0.5
        low_dom_threshold = 0.45
    elif year <= 2021:
        high_dom_threshold = 0.45
        low_dom_threshold = 0.35
    else:
        high_dom_threshold = 0.47
        low_dom_threshold = 0.40
    
    # Regime classification
    if btc_dom > high_dom_threshold and fg_index < 0.3:
        return "bear_market_bottom"
    elif btc_dom < low_dom_threshold and fg_index > 0.7:
        return "altcoin_euphoria"
    elif btc_dom > high_dom_threshold and vol_scaled > 0.6:
        return "flight_to_btc"
    elif btc_dom < low_dom_threshold and vol_scaled > 0.6:
        return "altcoin_breakout"
    elif abs(btc_dom - 0.43) < 0.03 and fg_index > 0.4 and fg_index < 0.6:
        return "stable_market"
    else:
        return "transition_phase"



def classify_signal_enhanced(row):
    """
    Enhanced signal classification incorporating market regime analysis
    """
    # Handle startup period
    if pd.isna(row["spread_thresh"]) or pd.isna(row["signal_thresh"]):
        return np.nan
    
    # Extract key variables
    abs_q50 = row["abs_q50"]
    spread = row["spread"]
    signal_thresh = row["signal_thresh"]
    spread_thresh = row["spread_thresh"]
    prob_up = row["prob_up"]
    fg_index = row.get("fg_index", 0.5)
    btc_dom = row.get("btc_dom", 0.43)
    vol_scaled = row.get("vol_scaled", 0.3)
    high_vol_flag = row.get("high_vol_flag", 0)
    
    # Market regime classification
    market_regime = classify_crypto_market_regime(btc_dom, fg_index, vol_scaled, 
                                                 row.name[1].year if hasattr(row.name, '__getitem__') else 2024)
    
    # Base signal strength (your original logic)
    base_tier = 0.0
    if abs_q50 >= signal_thresh and spread < spread_thresh:
        base_tier = 3.0  # Strong signal + tight spread
    elif abs_q50 >= signal_thresh:
        base_tier = 2.5  # Strong signal only
    elif spread < spread_thresh and row.get("signal_score", 0) > 0:
        base_tier = 2.0  # Tight spread + positive signal
    elif row.get("average_open", 1) > 1 and prob_up > 0.5:
        base_tier = 1.5  # Bullish alignment
    elif row.get("average_open", 1) < 1 and prob_up < 0.5:
        base_tier = 1.0  # Bearish alignment
    
    # Regime-based adjustments
    regime_multiplier = 1.0
    
    if market_regime == "bear_market_bottom":
        # High confidence in contrarian signals during bear bottoms
        if prob_up > 0.6:  # Bullish signal at bear bottom
            regime_multiplier = 1.4
        elif prob_up < 0.3:  # Bearish signal at bear bottom (fade)
            regime_multiplier = 0.6
            
    elif market_regime == "altcoin_euphoria":
        # Be more cautious during euphoria
        if prob_up > 0.7:  # Very bullish during euphoria (fade)
            regime_multiplier = 0.7
        elif prob_up < 0.4:  # Bearish during euphoria (strong signal)
            regime_multiplier = 1.3
            
    elif market_regime == "flight_to_btc":
        # BTC strength periods - boost BTC signals
        regime_multiplier = 1.2
        
    elif market_regime == "altcoin_breakout":
        # High volatility + low BTC dom - boost strong signals
        if base_tier >= 2.5:
            regime_multiplier = 1.3
            
    elif market_regime == "stable_market":
        # Normal conditions - slight boost to strong signals
        if base_tier >= 2.5:
            regime_multiplier = 1.1
    
    # Volatility regime adjustments
    if high_vol_flag == 1:
        # High volatility periods
        if base_tier >= 2.5:
            regime_multiplier *= 1.2  # Boost strong signals in high vol
        elif base_tier <= 1.0:
            regime_multiplier *= 0.8  # Reduce weak signals in high vol
    
    # FG Index extreme adjustments
    if fg_index < 0.15:  # Extreme fear
        if prob_up > 0.5:  # Contrarian bullish
            regime_multiplier *= 1.3
    elif fg_index > 0.85:  # Extreme greed
        if prob_up < 0.5:  # Contrarian bearish
            regime_multiplier *= 1.3
    
    # Calculate final tier
    final_tier = base_tier * regime_multiplier
    
    # Cap at reasonable bounds and round to nearest 0.1
    final_tier = max(0.0, min(4.0, final_tier))
    return round(final_tier * 10) / 10


        
def create_signal_tier_features(df):
    """
    Create features for ML-based signal tier classification
    """
    features = {}
    
    # Your existing signal components
    features['signal_strength'] = df['abs_q50'] / df['signal_thresh']
    features['spread_quality'] = df['spread_thresh'] / df['spread']
    features['prob_confidence'] = abs(df['prob_up'] - 0.5) * 2  # 0-1 scale
    
    # Market regime features
    features['market_regime_encoded'] = df.apply(
        lambda row: classify_crypto_market_regime(
            row['btc_dom'], row['fg_index'], row['vol_scaled'], 
            row.name[1].year if hasattr(row.name, '__getitem__') else 2024
        ), axis=1
    ).astype('category').cat.codes
    
    # Volatility regime
    features['vol_regime'] = df['high_vol_flag']
    
    # Sentiment extremes
    features['fg_extreme_fear'] = (df['fg_index'] < 0.2).astype(int)
    features['fg_extreme_greed'] = (df['fg_index'] > 0.8).astype(int)
    
    # BTC dominance regime
    features['btc_dom_high'] = (df['btc_dom'] > df['btc_dom'].quantile(0.7)).astype(int)
    features['btc_dom_low'] = (df['btc_dom'] < df['btc_dom'].quantile(0.3)).astype(int)
    
    # Interaction features
    features['signal_vol_interaction'] = features['signal_strength'] * df['vol_scaled']
    features['spread_regime_interaction'] = features['spread_quality'] * features['market_regime_encoded']
    
    return features



# Then train a simple model to predict optimal signal tiers
from sklearn.ensemble import RandomForestRegressor

def train_signal_tier_model(df, target_col='realized_return'):
    """
    Train ML model to predict optimal signal tiers based on future returns
    """
    features = create_signal_tier_features(df)
    feature_df = pd.DataFrame(features, index=df.index)
    
    # Create target: absolute future return (signal strength indicator)
    target = abs(df[target_col].shift(-1))  # Next period return
    
    # Remove NaN values
    valid_mask = ~(feature_df.isnull().any(axis=1) | target.isnull())
    X = feature_df[valid_mask]
    y = target[valid_mask]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_df.columns

def predict_signal_tier_ml(row, model, feature_columns):
    """
    Use ML model to predict signal tier
    """
    features = create_signal_tier_features(pd.DataFrame([row]))
    prediction = model.predict(features[feature_columns])[0]
    
    # Convert prediction to tier (0-4 scale)
    return min(4.0, max(0.0, prediction * 10))  # Scale appropriately







# Your decile thresholds (from the image)
DECILE_THRESHOLDS = {
    'vol_scaled': [0, 0.040228406, 0.100592005, 0.157576793, 0.219366829, 0.292197243, 0.380154782, 0.489959571, 0.650306095, 0.911945255, 1.0],
    'signal_rel': [-0.999993358, -0.91041967, -0.818028064, -0.724570109, -0.622671906, -0.513578004, -0.389794626, -0.24133338, -0.056341543, 0.170259192, 10.29764677],
    'signal_sigmoid': [0.268942684, 0.286913927, 0.30618247, 0.326387409, 0.349174012, 0.374355133, 0.403766741, 0.439957683, 0.485918339, 0.542462272, 0.952574127],
    'spread_sigmoid': [0.26804337, 0.373168477, 0.395796349, 0.412297581, 0.426721108, 0.440736902, 0.455622943, 0.472894296, 0.495122994, 0.524053256, 0.880797078],
    'prob_up': [0, 0.415979264, 0.454552117, 0.476363803, 0.492890165, 0.507965346, 0.523119476, 0.539739655, 0.560176208, 0.593712855, 1.0],
    'btc_dom': [0.358800002, 0.38939998, 0.4082, 0.4437, 0.4923, 0.5194, 0.53349996, 0.563800004, 0.6172, 0.6593, 0.71],
    'fg_index': [0.05, 0.2, 0.24, 0.3, 0.39, 0.46, 0.52, 0.61, 0.7, 0.76, 0.95]
}

def get_decile_rank(value, thresholds):
    """Convert a value to its decile rank (0-9)"""
    for i, threshold in enumerate(thresholds[1:], 1):
        if value <= threshold:
            return i - 1
    return 9

def classify_signal_with_deciles(row):
    """
    Enhanced signal classification using decile-based regime detection
    """
    # Handle startup period
    if pd.isna(row.get("spread_thresh")) or pd.isna(row.get("signal_thresh")):
        return np.nan
    
    # Get decile ranks for key variables
    vol_decile = get_decile_rank(row.get('vol_scaled', 0.3), DECILE_THRESHOLDS['vol_scaled'])
    signal_decile = get_decile_rank(row.get('signal_rel', -0.5), DECILE_THRESHOLDS['signal_rel'])
    fg_decile = get_decile_rank(row.get('fg_index', 0.5), DECILE_THRESHOLDS['fg_index'])
    btc_dom_decile = get_decile_rank(row.get('btc_dom', 0.5), DECILE_THRESHOLDS['btc_dom'])
    prob_decile = get_decile_rank(row.get('prob_up', 0.5), DECILE_THRESHOLDS['prob_up'])
    
    # Base signal strength (your original logic)
    abs_q50 = row.get("abs_q50", 0)
    
    # I have yet to see any clear usage of this q10 - q90 spread feature, I think there's something to it -- however -- I can't help but think that we're making this up entirely. What's the right way to prove that "spread" has predictive qualities? We're using it in quite a few ways through the model without ever proving it out first.
    spread = row.get("spread", 1)

    # Is it worth the overhead maintaining these thresholds? It kind of feels like we made them up as well. What's the right way to prove this out?
    signal_thresh = row.get("signal_thresh", 0.01)
    spread_thresh = row.get("spread_thresh", 0.02)
    
    base_tier = 0.0
    if abs_q50 >= signal_thresh and spread < spread_thresh:
        base_tier = 3.0
    elif abs_q50 >= signal_thresh:
        base_tier = 2.5
    elif spread < spread_thresh and row.get("signal_score", 0) > 0:
        base_tier = 2.0
    elif row.get("average_open", 1) > 1 and row.get("prob_up", 0.5) > 0.5:
        base_tier = 1.5
    elif row.get("average_open", 1) < 1 and row.get("prob_up", 0.5) < 0.5:
        base_tier = 1.0

    # I made up average_open recently, it's taking OPEN1, OPEN2, OPEN3 (which are all normalized features in crypto_loader) and using it to see if the average of the three is > 1 and if it aligns with the prediction / trade.   
    
    # Decile-based regime adjustments
    regime_multiplier = 1.0
    
    # Volatility regime (based on your decile analysis)
    if vol_decile >= 8:  # Top 20% volatility (deciles 8-9)
        if base_tier >= 2.5:
            regime_multiplier *= 1.3  # Boost strong signals in high vol
        elif base_tier <= 1.0:
            regime_multiplier *= 0.7  # Reduce weak signals in high vol
    elif vol_decile <= 1:  # Bottom 20% volatility (deciles 0-1)
        regime_multiplier *= 0.9  # Slightly reduce all signals in low vol
    
    # Signal strength regime (extreme signal_rel values)
    if signal_decile >= 8:  # Very positive signal_rel (rare!)
        regime_multiplier *= 1.4  # Strong boost for rare positive signals
    elif signal_decile <= 1:  # Very negative signal_rel (high uncertainty)
        regime_multiplier *= 0.8  # Reduce confidence in uncertain signals
    
    # Fear & Greed extremes (based on your historical analysis)
    if fg_decile <= 1:  # Extreme fear (deciles 0-1)
        if prob_decile >= 6:  # Bullish signal during fear
            regime_multiplier *= 1.4  # Strong contrarian signal
        elif prob_decile <= 3:  # Bearish signal during fear
            regime_multiplier *= 0.7  # Fade bearish signals in extreme fear
    elif fg_decile >= 8:  # Extreme greed (deciles 8-9)
        if prob_decile <= 3:  # Bearish signal during greed
            regime_multiplier *= 1.3  # Strong contrarian signal
        elif prob_decile >= 7:  # Bullish signal during greed
            regime_multiplier *= 0.6  # Fade bullish signals in extreme greed
    
    # BTC Dominance regime (based on your cycle analysis)
    if btc_dom_decile >= 8:  # High BTC dominance
        if fg_decile <= 2:  # High dom + fear = bear bottom
            regime_multiplier *= 1.5
        elif vol_decile >= 7:  # High dom + high vol = flight to safety
            regime_multiplier *= 1.2
    elif btc_dom_decile <= 2:  # Low BTC dominance
        if fg_decile >= 7:  # Low dom + greed = altcoin euphoria
            regime_multiplier *= 0.7  # Be cautious
        elif vol_decile >= 7:  # Low dom + high vol = altcoin breakout
            regime_multiplier *= 1.3
    
    # Combined extreme regime detection
    extreme_conditions = (vol_decile >= 8) + (fg_decile <= 1 or fg_decile >= 8) + (btc_dom_decile <= 1 or btc_dom_decile >= 8)
    
    if extreme_conditions >= 2:  # Multiple extreme conditions
        if base_tier >= 2.5:
            regime_multiplier *= 1.2  # Boost strong signals in extreme conditions
        else:
            regime_multiplier *= 0.8  # Reduce weak signals in extreme conditions
    
    # Calculate final tier
    final_tier = base_tier * regime_multiplier
    
    # Cap and round
    final_tier = max(0.0, min(4.0, final_tier))
    return round(final_tier * 10) / 10



def create_decile_features(df):
    """
    Create decile-based features for additional model inputs
    """
    features = {}
    
    # Convert all key variables to decile ranks
    for var_name, thresholds in DECILE_THRESHOLDS.items():
        if var_name in df.columns:
            features[f'{var_name}_decile'] = df[var_name].apply(
                lambda x: get_decile_rank(x, thresholds)
            )
    
    # Create regime interaction features
    vol_decile = features.get('vol_scaled_decile', 5)
    fg_decile = features.get('fg_index_decile', 5)
    btc_decile = features.get('btc_dom_decile', 5)
    
    # Extreme regime flags
    features['extreme_vol_regime'] = (vol_decile >= 8).astype(int)
    features['extreme_fear_regime'] = (fg_decile <= 1).astype(int)
    features['extreme_greed_regime'] = (fg_decile >= 8).astype(int)
    features['extreme_btc_dom_regime'] = ((btc_decile <= 1) | (btc_decile >= 8)).astype(int)
    
    # Combined regime score (0-4 based on number of extreme conditions)
    features['extreme_regime_score'] = (
        features['extreme_vol_regime'] + 
        features['extreme_fear_regime'] + 
        features['extreme_greed_regime'] + 
        features['extreme_btc_dom_regime']
    )
    
    # Decile-based market regime classification
    conditions = [
        (btc_decile >= 8) & (fg_decile <= 2),  # High BTC dom + fear
        (btc_decile <= 2) & (fg_decile >= 7),  # Low BTC dom + greed
        (vol_decile >= 8) & (btc_decile >= 7),  # High vol + high BTC dom
        (vol_decile >= 8) & (btc_decile <= 3),  # High vol + low BTC dom
        (vol_decile <= 2) & (fg_decile >= 4) & (fg_decile <= 6),  # Low vol + neutral sentiment
    ]
    
    choices = ['bear_bottom', 'alt_euphoria', 'flight_to_btc', 'alt_breakout', 'stable_market']
    features['market_regime_decile'] = np.select(conditions, choices, default='transition')
    
    return features



