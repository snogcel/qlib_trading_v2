import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to Python path for src imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import qlib
from qlib.constant import REG_US, REG_CN 
from qlib.utils import init_instance_by_config, flatten_dict
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, accuracy_score
from qlib.data.dataset.handler import DataHandlerLP

# Fixed imports using new src structure
from src.data.nested_data_loader import CustomNestedDataLoader
from src.models.signal_environment import SignalEnv
from src.models.multi_quantile import QuantileLGBModel, MultiQuantileModel
from src.data.crypto_loader import crypto_dataloader_optimized as crypto_dataloader
from src.features.position_sizing import AdvancedPositionSizer


# GDELT functionality now in gdelt_loader.py
from src.data.gdelt_loader import gdelt_dataloader_optimized as gdelt_dataloader

# TierLoggingCallback moved to RL execution folder
from src.rl_execution.custom_tier_logging import TierLoggingCallback

import optuna
import lightgbm as lgbm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from inspect import getfullargspec

from sklearn.model_selection import TimeSeriesSplit

train_start_time = "2018-08-02"
train_end_time = "2023-12-31"
valid_start_time = "2024-01-01"
valid_end_time = "2024-09-30"
test_start_time = "2024-10-01"
test_end_time = "2025-04-01"

fit_start_time = None
fit_end_time = None

provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA"


SEED = 42
MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_101"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)



def cross_validation_fcn(df_train, model, early_stopping_flag=False):
    """
    Performs cross-validation on a given model using KFold and returns the average
    mean squared error (MSE) score across all folds.

    Parameters:
    - X_train: the training data to use for cross-validation
    - model: the machine learning model to use for cross-validation
    - early_stopping_flag: a boolean flag to indicate whether early stopping should be used

    Returns:
    - model: the trained machine learning model
    - mean_mse: the average MSE score across all folds
    """
    
    tscv = TimeSeriesSplit(n_splits=5)
    X, y = df_train["feature"], df_train["label"]

    mse_list = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Fold {fold+1}: Train [{X_train.index[0]} to {X_train.index[-1]}], "
            f"Valid [{X_val.index[0]} to {X_val.index[-1]}]")

        # Train your model here
        if early_stopping_flag:
            # Use early stopping if enabled
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgbm.early_stopping(stopping_rounds=100, verbose=True)])
        else:
            model.fit(X_train, y_train)
            
        # Make predictions on the validation set and calculate the MSE score
        y_pred = model.predict(X_val)
        y_pred_df = pd.DataFrame(y_pred)

        y_pred_df.index = X_val.index

        mse = MSE(y_val, y_pred_df)
        mse_list.append(mse)
        
    # Return the trained model and the average MSE score
    return model, np.mean(mse_list)

def adaptive_entropy_coef(vol_scaled, base=0.005, min_coef=0.001, max_coef=0.02):
    """
    Map vol_scaled ∈ [0, 1] → entropy coef.
    More entropy in low-vol regimes, less in high-volatility ones.
    """
    inverse_vol = 1.0 - vol_scaled
    coef = base * (1 + inverse_vol * 2.0)  # amplify entropy in quiet regimes
    return float(np.clip(coef, min_coef, max_coef))

class EntropyAwarePPO(PPO):
    def __init__(self, *args, volatility_getter=None, base_entropy=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.volatility_getter = volatility_getter
        self.base_entropy = base_entropy

    def train(self):
        if callable(self.volatility_getter):
            vol_scaled = self.volatility_getter()
            self.ent_coef = adaptive_entropy_coef(vol_scaled, base=self.base_entropy)

        super().train()

def check_transform_proc(proc_l, fit_start_time, fit_end_time):
        new_l = []
        for p in proc_l:
            if not isinstance(p, Processor):
                klass, pkwargs = get_callable_kwargs(p, processor_module)
                args = getfullargspec(klass).args
                if "fit_start_time" in args and "fit_end_time" in args:
                    assert (
                        fit_start_time is not None and fit_end_time is not None
                    ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                    pkwargs.update(
                        {
                            "fit_start_time": fit_start_time,
                            "fit_end_time": fit_end_time,
                        }
                    )
                proc_config = {"class": klass.__name__, "kwargs": pkwargs}
                if isinstance(p, dict) and "module_path" in p:
                    proc_config["module_path"] = p["module_path"]
                new_l.append(proc_config)
            else:
                new_l.append(p)
        return new_l

# TODO - Remove
def prob_up_piecewise(row):
        q10, q50, q90 = row["q10"], row["q50"], row["q90"]
        if q90 <= 0:
            return 0.0
        if q10 >= 0:
            return 1.0
        # 0 lies between q10 and q50
        if q10 < 0 <= q50:
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            return 1 - cdf0
        # 0 lies between q50 and q90
        cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
        return 1 - cdf0

def kelly_sizing(row) -> float:
    """Validated Kelly Criterion sizing based on proven predictive features, pulled from hummingbot_backtester.py"""
    
    # aligned logic with hummingbot backtester
    q10, q50, q90, tier_confidence, signal_thresh, prob_up = row["q10"], row["q50"], row["q90"], row["signal_tier"], row["signal_thresh_adaptive"], row["prob_up"]

    spread_thresh = None
  
    # Calculate spread (validated as predictive of future volatility)
    spread = q90 - q10
    abs_q50 = abs(q50)

    # prob_up = self.prob_up_piecewise(q10, q50, q90)
    if prob_up is None:
        raise ValueError("prob_up cannot be None.")
    
    if prob_up > 0.5:  # Long position
        expected_win = q90
        expected_loss = abs(q10) if q10 < 0 else 0.001
        win_prob = prob_up
    else:  # Short position
        expected_win = abs(q10) if q10 < 0 else 0.001
        expected_loss = q90
        win_prob = 1 - prob_up
    
    if expected_loss <= 0:
        return 0.001
    
    # Kelly formula
    payoff_ratio = expected_win / expected_loss
    kelly_pct = (payoff_ratio * win_prob - (1 - win_prob)) / payoff_ratio
    
    # Base conservative Kelly with confidence
    confidence_adj = tier_confidence / 10.0
    base_kelly = kelly_pct * 0.25 * confidence_adj
    
    # VALIDATED ADJUSTMENTS based on spread analysis results
    
    # 1. Signal threshold adjustment (T-stat 7.10, highly significant)
    signal_quality_multiplier = 1.0
    if signal_thresh and abs_q50 > signal_thresh:
        signal_quality_multiplier = 1.3  # Boost for validated signals (Sharpe 0.077 vs -0.002)
    else:
        signal_quality_multiplier = 0.8   # Reduce for unvalidated signals
    
    # 2. Spread threshold adjustment (proven better risk-adjusted returns)
    spread_risk_multiplier = 1.0
    if spread_thresh and spread < spread_thresh:
        spread_risk_multiplier = 1.2  # Tight spread: +0.025% return vs -0.12%
    else:
        spread_risk_multiplier = 0.8   # Wide spread: higher volatility
    
    # 3. Combined quality bonus (both conditions met)
    combined_bonus = 1.0
    if (signal_thresh and abs_q50 > signal_thresh and 
        spread_thresh and spread < spread_thresh):
        combined_bonus = 1.15  # Best combination
    
    # Final Kelly calculation
    final_kelly = base_kelly * signal_quality_multiplier * spread_risk_multiplier * combined_bonus
    
    # should max_position_pct be aware of previous level, as to not exceed?
    max_position_pct = 0.5

    return max(0.001, min(final_kelly, max_position_pct))

def ensure_vol_risk_available(df):
    """
    Ensure vol_risk is available - use existing feature from crypto_loader_optimized
    vol_risk = Std(Log(close/close_prev), 6)² (VARIANCE, not std dev)
    This represents the squared volatility = variance, which is key for risk measurement
    """
    if 'vol_risk' not in df.columns:
        print(" vol_risk not found in data - this should come from crypto_loader_optimized")
        print("   vol_risk = Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)")
        
        # Fallback calculation if not available (but this shouldn't happen)
        if 'vol_raw' in df.columns:
            df['vol_risk'] = df['vol_raw'] ** 2  # Convert std dev to variance
            print("   Created vol_risk from vol_raw (vol_raw²)")
        else:
            print("   Cannot create vol_risk - missing vol_raw")
            df['vol_risk'] = 0.0001  # Small default value
    else:
        print(f"vol_risk available from crypto_loader_optimized ({df['vol_risk'].notna().sum():,} valid values)")
    
    return df

def identify_market_regimes(df):
    """
    Identify market regimes using volatility and momentum features
    Creates regime interaction features for enhanced signal quality
    """
    # Ensure we have vol_risk (variance measure from crypto_loader_optimized)
    df = ensure_vol_risk_available(df)
    
    # Volatility regimes (using vol_risk for consistency)
    df['vol_regime_low'] = (df['vol_risk'] < 0.3).astype(int)
    df['vol_regime_medium'] = ((df['vol_risk'] >= 0.3) & (df['vol_risk'] < 0.7)).astype(int)
    df['vol_regime_high'] = (df['vol_risk'] >= 0.7).astype(int)
    
    # Momentum regimes (using existing momentum features if available)
    if 'vol_raw_momentum' in df.columns:
        momentum = df['vol_raw_momentum']
    else:
        # Calculate simple momentum if not available
        momentum = df['vol_raw'].rolling(24).mean() / df['vol_raw'].rolling(168).mean() - 1
        df['vol_raw_momentum'] = momentum
    
    df['momentum_regime_trending'] = (abs(momentum) > 0.1).astype(int)
    df['momentum_regime_ranging'] = (abs(momentum) <= 0.1).astype(int)
    
    # Combined regime classification
    df['regime_low_vol_trending'] = df['vol_regime_low'] * df['momentum_regime_trending']
    df['regime_low_vol_ranging'] = df['vol_regime_low'] * df['momentum_regime_ranging']
    df['regime_high_vol_trending'] = df['vol_regime_high'] * df['momentum_regime_trending']
    df['regime_high_vol_ranging'] = df['vol_regime_high'] * df['momentum_regime_ranging']
    
    # Regime stability (how long in current regime)
    df['regime_stability'] = df.groupby(
        (df['vol_regime_high'] != df['vol_regime_high'].shift()).cumsum()
    ).cumcount() + 1
    
    return df

def q50_regime_aware_signals(df, transaction_cost_bps=20, base_info_ratio=1.5):
    """
    Generate Q50-centric signals with variance-based regime awareness
    Uses vol_risk as VARIANCE (not std dev) for superior risk assessment
    """
    df = df.copy()
    
    # Ensure we have regime features
    df = identify_market_regimes(df)
    
    # Calculate core signal metrics
    df["spread"] = df["q90"] - df["q10"]
    df["abs_q50"] = df["q50"].abs()
    
    # ENHANCED: Use vol_risk (variance) for superior risk assessment
    # Traditional info ratio: signal / spread (prediction uncertainty only)
    # Enhanced info ratio: signal / total_risk (market + prediction uncertainty)
    df['market_variance'] = df['vol_risk']  # This is already variance from crypto_loader
    df['prediction_variance'] = (df['spread'] / 2) ** 2  # Convert spread to variance
    df['total_risk'] = np.sqrt(df['market_variance'] + df['prediction_variance'])
    df['enhanced_info_ratio'] = df['abs_q50'] / np.maximum(df['total_risk'], 0.001)
    
    # Keep traditional info ratio for compatibility
    df['info_ratio'] = df['abs_q50'] / np.maximum(df['spread'], 0.001)
    
    print(f"Enhanced vs Traditional Info Ratio:")
    print(f"   Traditional (signal/spread): {df['info_ratio'].mean():.3f}")
    print(f"   Enhanced (signal/total_risk): {df['enhanced_info_ratio'].mean():.3f}")
    
    # MAGNITUDE-BASED economic threshold using quantile information
    base_transaction_cost = transaction_cost_bps / 10000
    
    # Calculate potential gains and losses from quantile distribution
    df['potential_gain'] = np.where(df['q50'] > 0, df['q90'], np.abs(df['q10']))
    df['potential_loss'] = np.where(df['q50'] > 0, np.abs(df['q10']), df['q90'])
    
    # Calculate probability-weighted expected value
    # Use existing prob_up (already calculated with prob_up_piecewise)
    if 'prob_up' not in df.columns:
        print(" prob_up not found, calculating...")
        df['prob_up'] = df.apply(prob_up_piecewise, axis=1)
    
    df['expected_value'] = (df['prob_up'] * df['potential_gain'] - 
                           (1 - df['prob_up']) * df['potential_loss'])
    
    print(f"Expected Value Analysis:")
    print(f"   Mean expected value: {df['expected_value'].mean():.4f}")
    print(f"   Positive expected value: {(df['expected_value'] > 0).mean()*100:.1f}%")
    print(f"   Mean potential gain: {df['potential_gain'].mean():.4f}")
    print(f"   Mean potential loss: {df['potential_loss'].mean():.4f}")
    
    # Use lower base cost (5 bps instead of 20 bps) - more realistic for crypto
    realistic_transaction_cost = 0.0005  # 5 bps
    
    # VARIANCE-BASED threshold scaling (more sensitive than std dev)
    # vol_risk is variance, so small changes have big impact
    variance_multiplier = 1.0 + df['vol_risk'] * 500  # Reduced multiplier for more trading
    
    # Variance-based regime identification (more granular)
    vol_risk_30th = df['vol_risk'].quantile(0.30)
    vol_risk_70th = df['vol_risk'].quantile(0.70)
    vol_risk_90th = df['vol_risk'].quantile(0.90)
    
    df['variance_regime_low'] = (df['vol_risk'] <= vol_risk_30th).astype(int)
    df['variance_regime_medium'] = ((df['vol_risk'] > vol_risk_30th) & (df['vol_risk'] <= vol_risk_70th)).astype(int)
    df['variance_regime_high'] = ((df['vol_risk'] > vol_risk_70th) & (df['vol_risk'] <= vol_risk_90th)).astype(int)
    df['variance_regime_extreme'] = (df['vol_risk'] > vol_risk_90th).astype(int)
    
    # Regime-aware threshold adjustments using variance regimes
    regime_multipliers = pd.Series(1.0, index=df.index)
    
    # Low variance: can accept lower thresholds (predictable environment)
    regime_multipliers -= df['variance_regime_low'] * 0.3  # -30% threshold
    
    # High variance: require higher thresholds
    regime_multipliers += df['variance_regime_high'] * 0.4  # +40% threshold
    
    # Extreme variance: much higher thresholds (very risky)
    regime_multipliers += df['variance_regime_extreme'] * 0.8  # +80% threshold
    
    # Trending markets: lower thresholds (momentum helps)
    regime_multipliers -= df['momentum_regime_trending'] * 0.1  # -10% in trending
    
    # Ensure multipliers stay reasonable
    regime_multipliers = regime_multipliers.clip(0.3, 3.0)
    
    # Combined effective threshold using magnitude-based approach
    # df['signal_thresh_adaptive'] = (realistic_transaction_cost * regime_multipliers * variance_multiplier)
    df['signal_thresh_adaptive'] = (realistic_transaction_cost * regime_multipliers * variance_multiplier)
    
    # Variance-aware information ratio threshold
    info_ratio_threshold = pd.Series(base_info_ratio, index=df.index)
    
    # Low variance: can accept lower info ratios (stable environment)
    info_ratio_threshold -= df['variance_regime_low'] * 0.4
    
    # Extreme variance: require much higher info ratios (unstable environment)
    info_ratio_threshold += df['variance_regime_extreme'] * 1.0
    
    df['effective_info_ratio_threshold'] = info_ratio_threshold.clip(0.5, 3.0)
    
    # MAGNITUDE-BASED trading conditions
    # Method 1: Traditional threshold approach (for comparison)
    df['economically_significant_traditional'] = df['abs_q50'] > df['signal_thresh_adaptive']
    
    # Method 2: Expected value approach (more trading opportunities)
    df['economically_significant_expected_value'] = df['expected_value'] > realistic_transaction_cost
    
    # Method 3: Combined approach - use expected value but with minimum signal strength
    min_signal_strength = df['abs_q50'].quantile(0.2)  # 20th percentile as minimum
    df['economically_significant_combined'] = (
        (df['expected_value'] > realistic_transaction_cost) & 
        (df['abs_q50'] > min_signal_strength)
    )
    
    # Choose the expected value approach for more trading opportunities
    df['economically_significant'] = df['economically_significant_expected_value']
    
    # Signal quality filter using enhanced info ratio
    df['high_quality'] = df['enhanced_info_ratio'] > df['effective_info_ratio_threshold']
    df['tradeable'] = df['economically_significant'] # & df['high_quality']
    
    # Print comparison
    trad_count = df['economically_significant_traditional'].sum()
    exp_val_count = df['economically_significant_expected_value'].sum()
    combined_count = df['economically_significant_combined'].sum()
    
    print(f"Economic Significance Comparison:")
    print(f"   Traditional threshold: {trad_count:,} ({trad_count/len(df)*100:.1f}%)")
    print(f"   Expected value: {exp_val_count:,} ({exp_val_count/len(df)*100:.1f}%)")
    print(f"   Combined approach: {combined_count:,} ({combined_count/len(df)*100:.1f}%)")
    print(f"   Improvement: {((exp_val_count/max(trad_count,1) - 1)*100):+.1f}% more opportunities")
    
    # VARIANCE-BASED interaction features for model training
    df['q50_x_low_variance'] = df['q50'] * df['variance_regime_low']
    df['q50_x_high_variance'] = df['q50'] * df['variance_regime_high']
    df['q50_x_extreme_variance'] = df['q50'] * df['variance_regime_extreme']
    df['q50_x_trending'] = df['q50'] * df['momentum_regime_trending']
    df['spread_x_high_variance'] = df['spread'] * df['variance_regime_high']
    df['vol_risk_x_abs_q50'] = df['vol_risk'] * df['abs_q50']  # Variance × signal strength
    df['enhanced_info_ratio_x_trending'] = df['enhanced_info_ratio'] * df['momentum_regime_trending']
    
    # Variance risk metrics
    df['signal_to_variance_ratio'] = df['abs_q50'] / np.maximum(df['vol_risk'], 0.0001)
    df['variance_adjusted_signal'] = df['q50'] / np.sqrt(np.maximum(df['vol_risk'], 0.0001))
    
    # Keep legacy columns for compatibility
    df["signal_rel"] = (df["abs_q50"] - df["signal_thresh_adaptive"]) / (df["signal_thresh_adaptive"] + 1e-12)
    
    # Print regime distribution
    print(f" Variance-Based Regime Distribution:")
    print(f"   Low Variance: {df['variance_regime_low'].sum():,} ({df['variance_regime_low'].mean()*100:.1f}%)")
    print(f"   High Variance: {df['variance_regime_high'].sum():,} ({df['variance_regime_high'].mean()*100:.1f}%)")
    print(f"   Extreme Variance: {df['variance_regime_extreme'].sum():,} ({df['variance_regime_extreme'].mean()*100:.1f}%)")
    
    return df

# Decile thresholds 
DECILE_THRESHOLDS = {
    'vol_scaled': [0, 0.040228406, 0.100592005, 0.157576793, 0.219366829, 0.292197243, 0.380154782, 0.489959571, 0.650306095, 0.911945255, 1.0],
    'signal_rel': [-0.999993358, -0.91041967, -0.818028064, -0.724570109, -0.622671906, -0.513578004, -0.389794626, -0.24133338, -0.056341543, 0.170259192, 10.29764677],
    'signal_sigmoid': [0.268942684, 0.286913927, 0.30618247, 0.326387409, 0.349174012, 0.374355133, 0.403766741, 0.439957683, 0.485918339, 0.542462272, 0.952574127],
    'spread_sigmoid': [0.26804337, 0.373168477, 0.395796349, 0.412297581, 0.426721108, 0.440736902, 0.455622943, 0.472894296, 0.495122994, 0.524053256, 0.880797078],
    'prob_up': [0, 0.415979264, 0.454552117, 0.476363803, 0.492890165, 0.507965346, 0.523119476, 0.539739655, 0.560176208, 0.593712855, 1.0],
    'btc_dom': [0.358800002, 0.38939998, 0.4082, 0.4437, 0.4923, 0.5194, 0.53349996, 0.563800004, 0.6172, 0.6593, 0.71],
    'fg_index': [0.05, 0.2, 0.24, 0.3, 0.39, 0.46, 0.52, 0.61, 0.7, 0.76, 0.95],
    'vol_raw': [0.000143, 0.001617, 0.002237, 0.002822, 0.003430, 0.004130, 0.004968, 0.006053, 0.007647, 0.010491, 0.127162]
}

def get_decile_rank(value, thresholds):
    """Convert a value to its decile rank (0-9)"""
    for i, threshold in enumerate(thresholds[1:], 1):
        if value <= threshold:
            return i - 1
    return 9

VOL_RAW_THRESHOLDS = [
    0.000143,   # 0th percentile
    0.001617,   # 10th percentile  
    0.002237,   # 20th percentile
    0.002822,   # 30th percentile
    0.003430,   # 40th percentile
    0.004130,   # 50th percentile
    0.004968,   # 60th percentile
    0.006053,   # 70th percentile
    0.007647,   # 80th percentile
    0.010491,   # 90th percentile
    0.127162,   # 100th percentile
]

def get_vol_raw_decile(vol_raw_value):
    """Convert vol_raw value to decile rank (0-9) - Updated for 6-day volatility"""
    for i, threshold in enumerate(VOL_RAW_THRESHOLDS[1:], 1):
        if vol_raw_value <= threshold:
            return i - 1
    return 9

# def classify_signal_corrected(row):
#     """
#     Simplified signal classification based on validation results
#     """
#     # Only use proven predictive features
#     abs_q50 = row.get("abs_q50", 0)

#     signal_thresh = row.get("signal_thresh_adaptive", 0.01)
#     prob_up = row.get("prob_up", 0.5)
    
#     # Base signal strength (validated)
#     if abs_q50 >= signal_thresh:
#         base_tier = 3.0
#     elif abs_q50 >= signal_thresh * 0.75:
#         base_tier = 2.0
#     elif abs_q50 >= signal_thresh * 0.5:
#         base_tier = 1.0
#     else:
#         base_tier = 0.0
    
#     # Directional confidence boost
#     prob_confidence = abs(prob_up - 0.5) * 2  # 0 to 1 scale
#     confidence_multiplier = 0.8 + (prob_confidence * 0.4)  # 0.8 to 1.2 range
    
#     return base_tier * confidence_multiplier


# # Helper function to add vol_raw features to your dataframe (UPDATED for optimized loader)
# def add_vol_raw_features(df):
#     """
#     Add vol_raw decile features to dataframe
#     Compatible with optimized crypto loader
#     """
    
#     # Check if vol_raw already exists (from optimized loader)
#     if 'vol_raw' not in df.columns:
#         # Fallback: use $realized_vol_6 if vol_raw not available (optimized loader uses vol_6)
#         if '$realized_vol_6' in df.columns:
#             df['vol_raw'] = df['$realized_vol_6']
#         elif '$realized_vol_3' in df.columns:
#             df['vol_raw'] = df['$realized_vol_3']
#         else:
#             print("Warning: No volatility column found for vol_raw")
#             return df
    
#     # Check if vol_raw_decile already exists (from optimized loader)
#     if 'vol_raw_decile' not in df.columns:
#         df['vol_raw_decile'] = df['vol_raw'].apply(get_vol_raw_decile)
    
#     # Check if regime flags already exist (from optimized loader)
#     if 'vol_extreme_high' not in df.columns:
#         df['vol_extreme_high'] = (df['vol_raw_decile'] >= 8).astype(int)
#     if 'vol_high' not in df.columns:
#         df['vol_high'] = (df['vol_raw_decile'] >= 6).astype(int)
#     if 'vol_low' not in df.columns:
#         df['vol_low'] = (df['vol_raw_decile'] <= 2).astype(int)
#     if 'vol_extreme_low' not in df.columns:
#         df['vol_extreme_low'] = (df['vol_raw_decile'] <= 1).astype(int)
    
#     # Check if vol_risk already exists (from optimized loader)
#     if 'vol_risk' not in df.columns:
#         # Use $realized_vol_6 instead of $realized_vol_3 (which was removed in optimization)
#         vol_col = '$realized_vol_6' if '$realized_vol_6' in df.columns else 'vol_raw'
        
#         rolling_window = 30
#         q_low = df[vol_col].rolling(rolling_window).quantile(0.01)
#         q_high = df[vol_col].rolling(rolling_window).quantile(0.99)
#         df['vol_risk'] = ((df[vol_col] - q_low.shift(1)) / 
#                          (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)
    
#     # Check if vol_raw_momentum already exists (from optimized loader)
#     if 'vol_raw_momentum' not in df.columns:
#         df['vol_raw_momentum'] = df['vol_raw'].pct_change(periods=3)
    
#     return df

def classify_signal_with_vol_raw_deciles(row):
    """
    Signal classification using optimal volatility insights
    """
    # Primary signal components (validated)
    abs_q50 = row.get("abs_q50", 0)
    signal_thresh = row.get("signal_thresh_adaptive", 0.01)
    prob_up = row.get("prob_up", 0.5)
    
    # Use raw volatility for regime detection (better predictor)
    vol_raw = row.get("vol_raw", row.get("$realized_vol_3", 0.01))
    vol_scaled = row.get("vol_scaled", row.get("vol_risk", 0.3))  # For position sizing (bounded 0-1)
    vol_decile = row.get("vol_raw_decile", -1)
   
    # Base signal strength
    if abs_q50 >= signal_thresh:
        base_tier = 3.0
    elif abs_q50 >= signal_thresh * 0.8:
        base_tier = 2.0
    elif abs_q50 >= signal_thresh * 0.6:
        base_tier = 1.0
    else:
        base_tier = 0.0
    
    # Volatility regime adjustments using deciles
    if vol_decile >= 8:  # Top 20% volatility (deciles 8-9)
        vol_regime = 'extreme_high'
        vol_multiplier = 1.3 if base_tier >= 2.0 else 0.7  # Boost strong signals, reduce weak ones
    elif vol_decile >= 6:  # High volatility (deciles 6-7)
        vol_regime = 'high'
        vol_multiplier = 1.15 if base_tier >= 2.0 else 0.85
    elif vol_decile >= 4:  # Medium-high volatility (deciles 4-5)
        vol_regime = 'medium_high'
        vol_multiplier = 1.05 if base_tier >= 2.0 else 0.95
    elif vol_decile >= 2:  # Medium-low volatility (deciles 2-3)
        vol_regime = 'medium_low'
        vol_multiplier = 1.0  # Neutral
    else:  # Low volatility (deciles 0-1)
        vol_regime = 'low'
        vol_multiplier = 0.9  # Slightly reduce all signals in low vol
    
    # Directional confidence
    prob_confidence = abs(prob_up - 0.5) * 2
    confidence_multiplier = 0.9 + (prob_confidence * 0.2)

    # Minor boost from signal_tanh (only if it adds value)
    #if base_tier > 0:
    #    tanh_boost = 1.0 + (signal_tanh * 0.05)  # Very small adjustment
    #    return base_tier * tanh_boost
    
    # return base_tier

    final_tier = base_tier * vol_multiplier * confidence_multiplier

    # print(f"final_tier: {final_tier}, abs_q50: {abs_q50}, signal_thresh: {signal_thresh}, vol_decile: {vol_decile}, base_tier: {base_tier}, vol_multiplier: {vol_multiplier}, confidence_multiplier: {confidence_multiplier}")    
    
    return {
        'signal_tier': round(final_tier * 10) / 10,
        'vol_regime': vol_regime,
        'vol_decile': vol_decile,
        'vol_multiplier': vol_multiplier
    }


def signal_classification(row):
    """
    Ultra-simple version - just use what works
    """
    abs_q50 = row.get("abs_q50", 0)
    signal_thresh = row.get("signal_thresh_adaptive", 0.01)
    
    # Just use the validated threshold approach
    if abs_q50 >= signal_thresh:
        return 3  # Strong signal
    elif abs_q50 >= signal_thresh * 0.8:
        return 2  # Medium signal
    elif abs_q50 >= signal_thresh * 0.6:
        return 1  # Weak signal
    else:
        return 0  # No signal


def quantile_loss(y_true, y_pred, quantile):
    # Step 1: Ensure both are Series or DataFrames with matching structure
    if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] == 1:
        y_pred = y_pred.iloc[:, 0]  # convert to Series

    if isinstance(y_true, pd.DataFrame) and y_true.shape[1] == 1:
        y_true = y_true.iloc[:, 0]

    # Step 2: Align index names (important in pandas!)
    y_pred.index.names = y_true.index.names

    # Step 3: Align values (intersection of shared index)
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')

    errors = y_true_aligned - y_pred_aligned
    #print("Aligned quantile_errors:\n", errors.head())  # sample output
        
    coverage = (y_true < y_pred).mean()
    #print(f"Q90 empirical coverage: {coverage:.2%}")

    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)), coverage


if __name__ == '__main__': 

    _learn_processors = [{"class": "DropnaLabel"},]
    _infer_processors = []

    infer_processors = check_transform_proc(_infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(_learn_processors, fit_start_time, fit_end_time)
    
    freq_config = {
        "feature": "60min", 
        "label": "day"
    }

    inst_processors = [
        {
            "class": "TimeRangeFlt",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "start_time": train_start_time,
                "end_time": test_end_time,
                "freq": freq_config["feature"]
            }
        }
    ]

    crypto_data_loader = {
        "class": "crypto_dataloader_optimized",
        "module_path": "src.data.crypto_loader",
        "kwargs": {
            "config": {
                "feature": crypto_dataloader.get_feature_config(),
                "label": crypto_dataloader.get_label_config(),                                                
            },                                            
            "freq": freq_config["feature"],  # "60min"
            "inst_processors": inst_processors
        }
    }

    gdelt_data_loader = {
        "class": "gdelt_dataloader_optimized",
        "module_path": "src.data.gdelt_loader",
        "kwargs": {
            "config": {
                "feature": gdelt_dataloader.get_feature_config()
            },
            "freq": freq_config["label"],  # "day"
            "inst_processors": inst_processors
        }
    }

    nested_dl = CustomNestedDataLoader(dataloader_l=[crypto_data_loader, gdelt_data_loader], join="left")    
    
    handler_config = {
        "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
        "start_time": train_start_time,
        "end_time": test_end_time,                
        "data_loader": nested_dl,        
        "infer_processors": infer_processors,
        "learn_processors": learn_processors,
        "shared_processors": [],
        "process_type": DataHandlerLP.PTYPE_A,
        "drop_raw": False 
    }

    dataset_handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": handler_config,
    }

    GENERIC_LGBM_PARAMS = {
        # Core quantile settings
        "objective": "quantile",
        "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
        "boosting_type": "gbdt",
        "device": "cpu",
        "verbose": -1,
        "random_state": 42,
        
        # Conservative learning settings for feature exploration
        # "learning_rate": 0.05,           # Moderate learning rate
        # "num_leaves": 64,                # Balanced complexity
        # "max_depth": 8,                  # Reasonable depth for GDELT features
        
        # Regularization (moderate to prevent overfitting)
        # "lambda_l1": 0.1,
        # "lambda_l2": 0.1,
        # "min_data_in_leaf": 20,
        # "feature_fraction": 0.8,         # Use 80% of features per tree
        # "bagging_fraction": 0.8,         # Use 80% of data per iteration
        # "bagging_freq": 5,
        
        # Early stopping
        "early_stopping_rounds": 50,
        "num_boost_round": 2250,         # Let early stopping decide

        # Set seed for reproducibility
        "seed": SEED
    }

    multi_quantile_params = {
        # 0.1: {'learning_rate': 0.060555113429817814, 'colsample_bytree': 0.7214813020361056, 'subsample': 0.7849919729082881, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05, 'max_depth': 10, 'num_leaves': 224, **GENERIC_LGBM_PARAMS},
        # 0.5: {'learning_rate': 0.02753370821225369, 'max_depth': -1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS},
        # 0.9: {'learning_rate': 0.09355380738420341, 'max_depth': 10, 'num_leaves': 249, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS}

        0.1: {'learning_rate': 0.026, 'max_depth': 7, **GENERIC_LGBM_PARAMS},
        0.5: {'learning_rate': 0.027, 'max_depth': 7, **GENERIC_LGBM_PARAMS},                
        0.9: {'learning_rate': 0.028, 'max_depth': 7, **GENERIC_LGBM_PARAMS} 
    }

    # "lgb_params": {
    #     0.1: {"learning_rate": 0.08971845032956545, "max_depth": 3, "n_estimators": 953, "seed": SEED},
    #     0.5: {"learning_rate": 0.022554447458683766, "max_depth": 5, "n_estimators": 556, "seed": SEED},
    #     0.9: {"learning_rate": 0.018590766014390355, "max_depth": 4, "n_estimators": 333, "seed": SEED}
    # }

    # print(multi_quantile_params[0.1])

    #"lgb_params": {
    #    0.1: {'learning_rate': 0.03915802868187673, 'colsample_bytree': 0.6224232548522113, 'subsample': 0.7322459139253197, 'lambda_l1': 6.957072141326349, 'lambda_l2': 0.004366116342801104, 'max_depth': 10, 'seed': SEED},
    #    0.5: {'learning_rate': 0.08751145729062904, 'colsample_bytree': 0.5897687601362188, 'subsample': 0.754061620932527, 'lambda_l1': 1.9808527398597983e-06, 'lambda_l2': 2.91987558633637e-05, 'max_depth': 10, 'seed': SEED},
    #    0.9: {'learning_rate': 0.028047164919345058, 'colsample_bytree': 0.841009708338563, 'subsample': 0.6210307287531586, 'lambda_l1': 2.9139063969227813e-08, 'lambda_l2': 6.363456739796053, 'max_depth': 10, 'seed': SEED}
    #}

    # 0.1 - Best hyperparameters: {'learning_rate': 0.04083809126843124, 'max_depth': 10, 'num_leaves': 224}
    # 0.1 - Best hyperparameters: {'learning_rate': 0.05628507997416036, 'max_depth': 10, 'num_leaves': 163}
    # model params:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.04083809126843124, 'max_depth': 10, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 224, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}
    # model params:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.05628507997416036, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 163, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}

    # 0.1 - Best hyperparameters: {'lambda_l1': 4.511969685016852, 'lambda_l2': 0.0006936273081692159}

    # 0.5 - Best hyperparameters: {'learning_rate': 0.022341989097031445, 'max_depth': 6, 'num_leaves': 197}
    # 0.5 - Best hyperparameters: {'learning_rate': 0.02753370821225369, 'max_depth': 6, 'num_leaves': 185}
    # model params:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.022341989097031445, 'max_depth': 6, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 197, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}
    # model params:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 1.0, 'importance_type': 'split', 'learning_rate': 0.02753370821225369, 'max_depth': -1, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 185, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0, 'subsample_for_bin': 200000, 'subsample_freq': 0}

    # 0.9 - Best hyperparameters: {'learning_rate': 0.0785666821118603, 'max_depth': 10, 'num_leaves': 161}. Best is trial 108 with value: 8.212406822184678e-05.
    # 0.9 - Best hyperparameters: {'learning_rate': 0.09355380738420341, 'max_depth': 10, 'num_leaves': 249}
    # 0.9 - Best hyperparameters: {'learning_rate': 0.10239714752158131, 'max_depth': 10, 'num_leaves': 203}


    # Best hyperparameters: {'learning_rate': 0.060555113429817814, 'colsample_bytree': 0.7214813020361056, 'subsample': 0.7849919729082881, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05, 'max_depth': 10, 'num_leaves': 224}
    # Best MSE: 0.0001
    # Number of finished trials:  100
    # Best trial:
    # Value: 8.446865714285136e-05
    # Params:
    #     learning_rate: 0.060555113429817814
    #     colsample_bytree: 0.7214813020361056
    #     subsample: 0.7849919729082881
    #     lambda_l1: 8.722794281828277e-05
    #     lambda_l2: 3.220667556916701e-05
    #     max_depth: 10
    #     num_leaves: 224
    # model params:  {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.7214813020361056, 'importance_type': 'split', 'learning_rate': 0.060555113429817814, 'max_depth': 10, 'min_child_samples': 20, 'min_child_weight': 0.001, 'min_split_gain': 0.0, 'n_estimators': 100, 'n_jobs': None, 'num_leaves': 224, 'objective': None, 'random_state': None, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.7849919729082881, 'subsample_for_bin': 200000, 'subsample_freq': 0, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05}


    # finalized model after tuning
    task_config = {        
        "model": {
            "class": "MultiQuantileModel",
            "module_path": "src.models.multi_quantile",
            "kwargs": {
                "quantiles": [0.1, 0.5, 0.9],
                "lgb_params": multi_quantile_params
            }
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",                        
            "kwargs": {
                "handler": dataset_handler_config,
                "segments": {
                    "train": (train_start_time, train_end_time),
                    "valid": (valid_start_time, valid_end_time),
                    "test": (test_start_time, test_end_time),
                },
            }
        },
        "task": {
            "model": "LGBModel",
            "dataset": "DatasetH",
            "record": [
                {
                    "class": "SignalRecord",
                    "module_path": "qlib.workflow.record_temp"
                },
                {
                    "class": "PortAnaRecord",
                    "module_path": "qlib.workflow.record_temp"
                }
            ]
        }
    }

    # define the objective function for Optuna optimization
    def objective(trial):

        params = {
            "objective": "quantile",
            "metric": ["l2", "l1"],
            "boosting_type": "gbdt",
            "device": "cpu",
            "verbose": -1,
            
            "alpha": 0.1,                      

            # Regularization (moderate to prevent overfitting)
            #"lambda_l1": 0.1,
            #"lambda_l2": 0.1,
            #"min_data_in_leaf": 20,
            #"feature_fraction": 0.8,         # Use 80% of features per tree
            #"bagging_fraction": 0.8,         # Use 80% of data per iteration
            #"bagging_freq": 5,
            
            #"colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            #"subsample": trial.suggest_float("subsample", 0.5, 1.0),

            # "learning_rate": 0.05628507997416036,
            # "max_depth": 8,            
            # "num_leaves": 163,
            # "lambda_l1": 4.511969685016852, 
            # "lambda_l2": 0.0006936273081692159,
            
            "min_data_in_leaf": 20,
            # "feature_fraction": 0.8,         # Use 80% of features per tree
            # "bagging_fraction": 0.8,         # Use 80% of data per iteration
            # "bagging_freq": 5,
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 512),
            
            # "max_depth": trial.suggest_int("max_depth", 4, 10),
           

            # Early stopping
            "early_stopping_rounds": 100,
            "num_boost_round": 1000,         # Let early stopping decide

            # Set seed for reproducibility
            "seed": SEED
        }

        # create the LightGBM regressor with the optimized parameters
        model = lgbm.LGBMRegressor(**params)

        # perform cross-validation using the optimized LightGBM regressor
        lgbm_model, mean_score = cross_validation_fcn(df_train, model, early_stopping_flag=True)

        # retrieve the best iteration of the model and store it as a user attribute in the trial object
        best_iteration = lgbm_model.best_iteration_
        trial.set_user_attr('best_iteration', best_iteration)
            
        return mean_score

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    # prepare segments
    df_train, df_valid, df_test = dataset.prepare(
        segments=["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    # split data
    X_train, y_train = df_train["feature"], df_train["label"]
    X_val, y_val = df_valid["feature"], df_valid["label"]
    #X_test, y_test = df_test["feature"], df_test["label"]

    model.fit(dataset=dataset)

    preds = model.predict(dataset, "valid")    

    feat_importance_high = model.models[0.9].get_feature_importance(importance_type='gain')    
    feature_names_high = model.models[0.9].model.feature_name()

    feat_importance_mid = model.models[0.5].get_feature_importance(importance_type='gain')
    feature_names_mid = model.models[0.5].model.feature_name()

    feat_importance_low = model.models[0.1].get_feature_importance(importance_type='gain')
    feature_names_low = model.models[0.1].model.feature_name()

    print("Feature Importance (Q90): ", feat_importance_high)
    print("Feature Importance (Q50): ", feat_importance_mid)
    print("Feature Importance (Q10): ", feat_importance_low)
    
    



    # Output Quantile Loss 
    loss, coverage = quantile_loss(y_val, preds["quantile_0.90"], 0.90)    
    feat_importance_high.to_csv(f"feat_importance_high_{loss}_{coverage}.csv")
    print(f"Quantile Loss (Q90): {loss}, coverage: {coverage:.2%}")

    loss, coverage = quantile_loss(y_val, preds["quantile_0.50"], 0.50)    
    feat_importance_mid.to_csv(f"feat_importance_mid_{loss}_{coverage}.csv")
    print(f"Quantile Loss (Q50): {loss}, coverage: {coverage:.2%}")

    loss, coverage = quantile_loss(y_val, preds["quantile_0.10"], 0.10)
    feat_importance_low.to_csv("feat_importance_low.csv")
    feat_importance_low.to_csv(f"feat_importance_low_{loss}_{coverage}.csv")
    print(f"Quantile Loss (Q10): {loss}, coverage: {coverage:.2%}")    
   

    # Filter for one instrument (e.g., BTCUSDT)
    instrument = "BTCUSDT"

    # Extract each series for this instrument
    q10 = preds["quantile_0.10"].xs(instrument, level="instrument").squeeze()
    q50 = preds["quantile_0.50"].xs(instrument, level="instrument").squeeze()
    q90 = preds["quantile_0.90"].xs(instrument, level="instrument").squeeze()
    y_true = y_val.xs(instrument, level="instrument").squeeze()

    # Sort by time for better plotting
    df_plot = pd.DataFrame({
        "Q10": q10,
        "Q50": q50,
        "Q90": q90,
        "True": y_true
    }).dropna().sort_index()

    isTuning = False
    if isTuning is True:
        # Create an optimization study with Optuna library
        study = optuna.create_study(direction="minimize",study_name="lgbm_opt")

        # Optimize the study using a user-defined objective function, for a total of 100 trials
        study.optimize(objective, n_trials=100)

        # get best hyperparameters and score
        best_params_lr = study.best_params
        best_score_lr = study.best_value

        # print best hyperparameters and score
        print(f"Best hyperparameters: {best_params_lr}")
        print(f"Best MSE: {best_score_lr:.4f}")

        # Print the number of finished trials in the study
        print("Number of finished trials: ", len(study.trials))

        # Print the best trial in the study, which represents the set of hyperparameters that yielded the lowest objective value
        print("Best trial:")
        trial = study.best_trial

        # Extract the best set of hyperparameters from the best trial and store them in a variable
        hp_lgbm = study.best_params

        # Add the best number of estimators (trees) to the set of hyperparameters
        # hp_lgbm["n_estimators"] = study.best_trial.user_attrs['best_iteration']

        # Print the objective value and the set of hyperparameters of the best trial
        print("  Value: {}".format(trial.value))
        print("  Params: ")

        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # insert identified params into final paramset for model creation
        # params = {
        #     "metric": "rmse",
        #     "early_stopping_rounds": 50,
        #     "objective": "regression",
        #     "metric": ["l2","l1"],
        #     "random_state": SEED,
        #     "colsample_bytree": trial.params["colsample_bytree"],
        #     "subsample": trial.params["subsample"],
        #     "lambda_l1": trial.params["lambda_l1"],
        #     "lambda_l2": trial.params["lambda_l2"],
        #     "learning_rate": trial.params["learning_rate"],
        #     "max_depth": trial.params["max_depth"],
        #     "num_leaves": trial.params["num_leaves"],
        #     "device": "cpu",
        #     "boosting_type": "gbdt",
        # }

        params = {
            "objective": "quantile",
            "metric": ["l2","l1"],
            "boosting_type": "gbdt",
            "device": "cpu",
            "verbose": -1,
            
            "alpha": 0.10,                      

            # Regularization (moderate to prevent overfitting)
            # "lambda_l1": 0.1,
            # "lambda_l2": 0.1,

            "lambda_l1": 4.511969685016852, 
            "lambda_l2": 0.0006936273081692159,
            
            # "min_data_in_leaf": 20,
            # "feature_fraction": 0.8,         # Use 80% of features per tree
            # "bagging_fraction": 0.8,         # Use 80% of data per iteration
            # "bagging_freq": 5,
            
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            
            #"lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            #"lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),

            "learning_rate": 0.05628507997416036,
            "max_depth": 10,
            "num_leaves": 163,

            # "max_depth": trial.params["max_depth"],

            # Early stopping
            "early_stopping_rounds": 1,
            "num_boost_round": 1000,         # Let early stopping decide

            # Set seed for reproducibility
            "seed": SEED
        }

        # Create a LightGBM regression model using the best set of hyperparameters found during the optimization process
        lgbm_model = lgbm.LGBMRegressor(**hp_lgbm)
                
        print("model params: ", lgbm_model.get_params())

        # Fit the model to the training data
        lgbm_model.fit(X_train, y_train, callbacks=[
            lgbm.log_evaluation(period=20)
        ])

        # Use the trained model to make predictions on the test data
        y_pred_lgbm = lgbm_model.predict(X_val)

        y_pred_df = pd.DataFrame(y_pred_lgbm)
        y_pred_df.index = X_val.index

        X_val.to_csv("X_val.csv")
        y_val.to_csv("y_val.csv")
        y_pred_df.to_csv("y_pred_df.csv")
   

    # fit to tuned model
    model.fit(dataset=dataset)

    preds_train = model.predict(dataset, "train")
    preds_valid = model.predict(dataset, "valid")
    #preds_test = model.predict(dataset, "test")

    y_all = pd.concat([y_train, y_val], axis=0, join='outer', ignore_index=False)
    X_all = pd.concat([X_train, X_val], axis=0, join='outer', ignore_index=False)
    preds = pd.concat([preds_train, preds_valid], axis=0, join='outer', ignore_index=False)
    
    # Include ALL features from X_all instead of just a subset
    print(f"Available features in X_all: {len(X_all.columns)}")
    print(f"Feature columns: {list(X_all.columns)}")
    
    # Start with predictions and truth
    df_all_components = [
        preds["quantile_0.10"].rename("q10"),
        preds["quantile_0.50"].rename("q50"),
        preds["quantile_0.90"].rename("q90"),
        y_all["LABEL0"].rename("truth"),
    ]
    
    # Add ALL features from X_all
    for col in X_all.columns:
        df_all_components.append(X_all[col])
    
    df_all = pd.concat(df_all_components, axis=1).dropna()
    
    print(f"Total features in df_all: {len(df_all.columns)}")
    print(f"GDELT features found: {[col for col in df_all.columns if 'cwt_' in col]}")
    print(f"Technical indicators found: {[col for col in df_all.columns if any(x in col for x in ['ROC', 'STD', 'OPEN', 'VOLUME'])]}")




    df_all.to_csv("df_all_macro_analysis_prep.csv")
    
    # build interaction / regime signals
    df_all = q50_regime_aware_signals(df_all)

    # Standalone functions for now to allow pipeline clarity
    df_all["prob_up"] = df_all.apply(prob_up_piecewise, axis=1)
    df_all["signal_tier"] = df_all.apply(signal_classification, axis=1)    
    df_all["kelly_position_size"] = df_all.apply(kelly_sizing, axis=1)    
    
    # keep
    alpha = 1.0  # controls “steepness”
    cap = 3.0
    df_all["signal_tanh"] = np.tanh(df_all["signal_rel"].clip(-cap, cap) / alpha)

    # Q50-CENTRIC SIGNAL GENERATION (replaces problematic threshold approach)
    print("Generating Q50-centric regime-aware signals...")
    
    # Generate signals using pure Q50 logic with regime awareness
    q50 = df_all["q50"]
    
    # Economic significance: must exceed regime-adjusted transaction costs
    economically_significant = df_all['economically_significant']
    
    # Signal quality: information ratio must be high enough for regime
    high_quality = df_all['high_quality']
    
    # Combined trading condition
    tradeable = economically_significant # & high_quality
    
    # Pure Q50 directional logic (no complex prob_up calculation needed!)
    buy_mask = tradeable & (q50 > 0)
    sell_mask = tradeable & (q50 < 0)
    
    # Assign side using Q50-centric approach
    df_all["side"] = -1  # default to HOLD
    df_all.loc[buy_mask, "side"] = 1   # LONG when q50 > 0 and tradeable
    df_all.loc[sell_mask, "side"] = 0  # SHORT when q50 < 0 and tradeable
    
    # prob_up is already calculated properly with prob_up_piecewise earlier in the script
    # No need to override it here - keep the original sophisticated calculation
    
    # VARIANCE-ENHANCED signal strength using enhanced info ratio
    df_all['signal_strength'] = np.where(
        tradeable,
        df_all['abs_q50'] * np.minimum(df_all['enhanced_info_ratio'] / df_all['effective_info_ratio_threshold'], 2.0),
        0.0
    )
    
    # Additional variance-based metrics for position sizing
    # Calculate for all signals, then apply tradeable filter
    base_position_size = 0.1 / np.maximum(df_all['vol_risk'] * 1000, 0.1)  # Inverse variance scaling
    df_all['position_size_suggestion'] = np.where(
        tradeable,
        base_position_size.clip(0.01, 0.5),  # Apply limits only to tradeable signals
        0.0  # Zero for non-tradeable
    )
    
    # Print signal summary
    signal_counts = df_all['side'].value_counts()
    total_signals = len(df_all)
    print(f"Q50-centric signals generated:")
    for side, count in signal_counts.items():
        side_name = {1: 'LONG', 0: 'SHORT', -1: 'HOLD'}[side]
        print(f"   {side_name}: {count:,} ({count/total_signals*100:.1f}%)")
    
    # Show signal quality for trading signals
    trading_signals = df_all[df_all['side'] != -1]
    if len(trading_signals) > 0:
        avg_info_ratio = trading_signals['info_ratio'].mean()
        avg_enhanced_info_ratio = trading_signals['enhanced_info_ratio'].mean()
        avg_abs_q50 = trading_signals['abs_q50'].mean()
        avg_vol_risk = trading_signals['vol_risk'].mean()
        avg_position_size = trading_signals['position_size_suggestion'].mean()
        
        print(f"Trading signal quality:")
        print(f"   Average Info Ratio (traditional): {avg_info_ratio:.2f}")
        print(f"   Average Enhanced Info Ratio (variance-aware): {avg_enhanced_info_ratio:.2f}")
        print(f"   Average |Q50|: {avg_abs_q50:.4f}")
        print(f"   Average Vol_Risk (variance): {avg_vol_risk:.6f} (√ = {np.sqrt(avg_vol_risk):.3f})")
        print(f"   Average Signal Strength: {trading_signals['signal_strength'].mean():.4f}")
        print(f"   Average Position Size Suggestion: {avg_position_size:.3f}")



    ML_correlation_matrix = df_all.corr()        
    ML_correlation_matrix.to_csv("ML_correlation_matrix.csv")


    # Calculate the correlation matrix

    # Xy_df = pd.concat([X_all, y_all], axis=0, join='outer', ignore_index=False)
    # correlation_matrix = Xy_df.corr()
    # correlation_matrix.to_csv("correlation_matrix.csv")
        
    # Drop redundant columns (updated for optimized loader)
    # Note: $realized_vol_3 may not exist in optimized loader (removed due to correlation)
    columns_to_drop = []
    if '$realized_vol_3' in df_all.columns:
        columns_to_drop.append('$realized_vol_3')
    #if 'abs_q50' in df_all.columns:
    #    columns_to_drop.append('abs_q50')
    #if 'signal_rel' in df_all.columns:
    #    columns_to_drop.append('signal_rel')
    #if 'spread' in df_all.columns:
    #    columns_to_drop.append('spread') # 7/30/25 -- seeing some strong correlations with other features, reducing noise.
    
    if columns_to_drop:
        df_all.drop(columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
    
    df_all.to_csv("df_all_macro_analysis.csv")



    df_cleaned = df_all.dropna(subset=["vol_risk","vol_scaled","vol_raw_momentum","signal_thresh_adaptive","signal_tanh","enhanced_info_ratio"])

    df_cleaned.to_pickle("./data3/macro_features.pkl") # pickled features used in "train_meta_wrapper.py" process


    raise SystemExit()


    df_train_rl = df_cleaned.loc[("BTCUSDT","2018-02-01"):("BTCUSDT","2023-12-31")]
    df_val_rl   = df_cleaned.loc[("BTCUSDT","2024-01-01"):("BTCUSDT","2024-09-30")]
    df_test_rl  = df_cleaned.loc[("BTCUSDT","2024-10-01"):("BTCUSDT","2025-04-01")]

    # ==========================
    # Sweep Runner
    # ==========================

    # ==========================
    # Experiment Configuration
    # ==========================
    
    reward_type = "tier_weighted"
    run_name = f"base_momentum_{reward_type}_V2"
    
    # Base PPO hyperparameters
    ppo_config = {
        "learning_rate": 3e-4, 
        "clip_range": 0.20, 
        "ent_coef": 0.005, 
        "gae_lambda": 0.95, 
        "vf_coef": 0.5
    }

    # =========================
    # Run Experiment
    # =========================
    
    print(f"\nLaunching: {run_name}")

    env_train = SignalEnv(df=df_train_rl, reward_type=reward_type)
    env_val = SignalEnv(df=df_val_rl, reward_type=reward_type, eval_mode=False)
    #env_test = SignalEnv(df=df_test_rl, reward_type=reward_type, eval_mode=False)

    vec_env = DummyVecEnv([lambda: env_train])
    vec_env.seed(SEED)
   
    callback = TierLoggingCallback(env_train, log_interval=50)

    agent = EntropyAwarePPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=ppo_config["learning_rate"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        gae_lambda=ppo_config["gae_lambda"],
        vf_coef=ppo_config["vf_coef"],
        seed=SEED, 
        verbose=1,
        volatility_getter=env_train.get_recent_vol_scaled,
        tensorboard_log=f"./logs_momentum_v3/{run_name}"
    )
    
    agent.learn(total_timesteps=384_000, callback=callback)        
    agent.save(f"./models/{run_name}")

    evaluate_agent(env_val, agent, name=run_name)



## TODO -- implement the following functions in a standlone file (used for RL)

# ==========================
# Reward Calculation Logic
# ==========================

def compute_reward(position, next_return, tier_weight, fee=0.001, slippage=0.0005, volatility_estimate=None, reward_type="tier_weighted"):
    raw_pnl = position * next_return
    if reward_type == "risk_normalized" and volatility_estimate is not None:
        raw_pnl /= volatility_estimate
    delta_position = 0  # To be filled by env
    fee_cost = fee * abs(delta_position)
    slip_cost = slippage * delta_position ** 2
    reward = raw_pnl * tier_weight - fee_cost - slip_cost
    return reward

# ==========================
# Evaluate Agent After Training
# ==========================

def evaluate_agent(env, agent, name="experiment"):
    obs = env.reset()
    done = False

    rewards, positions, tiers = [], [], []
    tier_rewards = {"A": [], "B": [], "C": [], "D": []}
    tier_positions = {"A": [], "B": [], "C": [], "D": []}

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        positions.append(env.position)
        tier = info.get("tier", "Unknown")

        if tier in tier_rewards:
            tier_rewards[tier].append(reward)
            tier_positions[tier].append(env.position)

        tiers.append(tier)

    df = pd.DataFrame({"tier": tiers, "reward": rewards, "position": positions})

    df.to_csv(f"{name}.csv")

    # Summarize tier performance
    summary = df.groupby("tier").agg({
        "reward": "sum",
        "position": lambda x: np.mean(np.abs(x))
    }).rename(columns={"reward": "total_pnl", "position": "avg_exposure"})

    summary["efficiency"] = summary["total_pnl"] / summary["avg_exposure"]
    print(f"\nEvaluation for {name}")
    print(summary.round(4))

    # Save plot of PnL per tier
    summary.to_csv(f"{name}_summary.csv")
    summary[["total_pnl", "efficiency"]].plot(kind="bar", title=f"Tier Attribution – {name}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}_tier_performance.png")
    plt.close()
