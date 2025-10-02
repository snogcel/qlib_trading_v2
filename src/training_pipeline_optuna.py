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
from src.data.gdelt_loader import gdelt_dataloader_optimized as gdelt_dataloader
from src.features.position_sizing import AdvancedPositionSizer

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

SEED = 42 # RANDOM SEED for Entropy Purposes
MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_101"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)

def cross_validation_fcn(df_train, model, early_stopping_flag=True):
    """
    Performs cross-validation on a given model using KFold and returns the average
    mean squared error (MSE) score across all folds.

    Parameters:
    - X_train: the training data to use for cross-validation
    - model: the machine learning model to use for cross-validation
    - early_stopping_flag: a boolean flag to indicate whether early stopping should be used (default is True)

    Returns:
    - model: the trained machine learning model
    - mean_mse: the average MSE score across all folds
    """
    
    tscv = TimeSeriesSplit(n_splits=5)
    X, y = df_train["feature"], df_train["label"]

    print("debug_cross_validation")
    raise SystemExit()

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
                      callbacks=[lgbm.early_stopping(stopping_rounds=1000, verbose=True)]) # increase default stopping_rounds = 1000 if flag is set to True
        else:
            model.fit(X_train, y_train)

        # model.fit(X_train, y_train)
            
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
    
    # fix: aligned the number of variables with the dataframe being passed (was previously missing signal_thresh_adaptive)
    # [54396:MainThread](2025-09-30 04:58:41,121) ERROR - qlib.workflow - [utils.py:41] - An exception has been raised[ValueError: not enough values to unpack (expected 7, got 6)].

    # aligned logic with hummingbot backtester
    q10, q50, q90, tier_confidence, signal_thresh, prob_up = row["q10"], row["q50"], row["q90"], row["signal_tier"], row["signal_thresh_adaptive"], row["prob_up"]

    spread_thresh = None
  
    # Calculate spread (validated as predictive of future volatility)
    spread = q90 - q10
    abs_q50 = abs(q50)

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

def identify_market_regimes(df):
    """
    Identify market regimes using volatility and momentum features
    Creates regime interaction features for enhanced signal quality
    """
    # Ensure we have vol_risk (variance measure from crypto_loader_optimized)
    if df['vol_risk'] is None:
        raise ValueError("vol_risk cannot be None.")    
    
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
    # Use existing prob_up (throw exception if missing)
    
    # if prob_up is None:
    #     raise ValueError("prob_up cannot be None.")
    
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

    print("_ENTER_MAIN_FUNCTION_")
    

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

    CORE_LGBM_PARAMS = {
        "objective": "quantile",
        "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
        "boosting_type": "gbdt",
        "device": "cpu",
        "verbose": 1, # set to verbose for more logs (this is outrageously complex lol)
        "random_state": 141551, # https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
        "early_stopping_rounds": 500,
        "num_boost_round": 2250,         # Let early stopping decide
        "seed": SEED
    }

    GENERIC_LGBM_PARAMS = {       
        # Conservative learning settings for feature exploration
        "learning_rate": 0.05,           # Moderate learning rate
        # "num_leaves": 64,                # Balanced complexity
        # "max_depth": 8,                  # Reasonable depth for GDELT features
        
        # Regularization (moderate to prevent overfitting)
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        
        # "min_data_in_leaf": 20, # remove constraint for tuning
        
        "feature_fraction": 0.8,         # Use 80% of features per tree
        "bagging_fraction": 0.8,         # Use 80% of data per iteration
        "bagging_freq": 5,
    }

    multi_quantile_params = {
        # 0.1: {'learning_rate': 0.060555113429817814, 'colsample_bytree': 0.7214813020361056, 'subsample': 0.7849919729082881, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05, 'max_depth': 10, 'num_leaves': 224, **GENERIC_LGBM_PARAMS},
        # 0.5: {'learning_rate': 0.02753370821225369, 'max_depth': -1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS},
        # 0.9: {'learning_rate': 0.09355380738420341, 'max_depth': 10, 'num_leaves': 249, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS}

        # 0.1: {**CORE_LGBM_PARAMS},
        # 0.5: {**CORE_LGBM_PARAMS},                
        # 0.9: {**CORE_LGBM_PARAMS} 

        0.1: {'max_depth': 25, **CORE_LGBM_PARAMS},
        0.5: {'max_depth': 25, **CORE_LGBM_PARAMS},                
        0.9: {'max_depth': 25, **CORE_LGBM_PARAMS}

    }

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
            "verbose": 1,
            
            "alpha": 0.1, # this controls which percentile the predictive model is targeting - in this case, Q10                     

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
            
            # "min_data_in_leaf": 20, # remove constraint to prevent overfitting

            # "feature_fraction": 0.8,         # Use 80% of features per tree
            # "bagging_fraction": 0.8,         # Use 80% of data per iteration
            # "bagging_freq": 5,
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "max_depth": trial.suggest_int("max_depth", 4, 50), # increase max depth to 50
            "num_leaves": trial.suggest_int("num_leaves", 20, 512),    

            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 512), # use optuna to find the optimilar data points within each point of decision tree (LGBM) - doesn't apply to LGBM model AFAIK          
            
            # "max_depth": trial.suggest_int("max_depth", 4, 10),
           

            # Early stopping
            "early_stopping_rounds": 1000, # increase from 100 to 1000 training rounds -- the variable is not being passed to cross_validation_fcn (AFAIK)
            "num_boost_round": 1000,         # Let early stopping decide

            # Set seed for reproducibility
            "seed": SEED
        }

        # create the LightGBM regressor with the optimized parameters
        model = lgbm.LGBMRegressor(**params)

        # perform cross-validation using the optimized LightGBM regressor
        lgbm_model, mean_score = cross_validation_fcn(df_train, model, early_stopping_flag=True) # test disabling early_stopping_flag

        # retrieve the best iteration of the model and store it as a user attribute in the trial object
        best_iteration = lgbm_model.best_iteration_
        
        print("best_iteration: ")
        print(best_iteration)
        
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

    print("_SPLIT_DATA_")    
    

    model.fit(dataset=dataset)

    print("_FIT_DATA_TO_LGBM_MODEL_")
    raise SystemExit()

    preds = model.predict(dataset, "valid")    

    # Calculate feature importance
    feat_importance_high = model.models[0.9].get_feature_importance(importance_type='gain')    
    feature_names_high = model.models[0.9].model.feature_name()

    feat_importance_mid = model.models[0.5].get_feature_importance(importance_type='gain')
    feature_names_mid = model.models[0.5].model.feature_name()

    feat_importance_low = model.models[0.1].get_feature_importance(importance_type='gain')
    feature_names_low = model.models[0.1].model.feature_name()

    # Output quantile Loss and Feature Importance
    loss, coverage = quantile_loss(y_val, preds["quantile_0.90"], 0.90)    
    feat_importance_high.to_csv(f"./temp/feat_importance_high_{loss}_{coverage}.csv")
    print(f"Quantile Loss (Q90): {loss}, coverage: {coverage:.2%}")

    loss, coverage = quantile_loss(y_val, preds["quantile_0.50"], 0.50)    
    feat_importance_mid.to_csv(f"./temp/feat_importance_mid_{loss}_{coverage}.csv")
    print(f"Quantile Loss (Q50): {loss}, coverage: {coverage:.2%}")

    loss, coverage = quantile_loss(y_val, preds["quantile_0.10"], 0.10)
    feat_importance_low.to_csv("./temp/feat_importance_low.csv")
    feat_importance_low.to_csv(f"./temp/feat_importance_low_{loss}_{coverage}.csv")
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
    


    # df_all.apply function calls expect the following variables:
    # q10, q50, q90, tier_confidence, signal_thresh, prob_up = row["q10"], row["q50"], row["q90"], row["signal_tier"], row["signal_thresh_adaptive"], row["prob_up"]

    # fix:
    # q10, q50, q90, tier_confidence, signal_thresh, kelly_position_size, prob_up = row["q10"], row["q50"], row["q90"], row["signal_tier"], row["signal_thresh_adaptive"], row["prob_up"]    



    # Bug: kelly_sizing function is expecting a variable to be in this dataframe: signal_thresh_adaptive
    # this variable is added to the dataframe in this function call: q50_regime_aware_signals

    # File "c:\Projects\qlib_trading_v2\src\training_pipeline_optuna.py", line 178, in kelly_sizing
    # q10, q50, q90, tier_confidence, signal_thresh, signal_thresh_adaptive, prob_up = row["q10"], row["q50"], row["q90"], row["signal_tier"], row["signal_thresh_adaptive"], row["prob_up"]

    """ Index(['q10', 'q50', 'q90', 'truth', '$btc_dom', '$fg_index', '$high_vol_flag',
       '$momentum_10', '$momentum_25', '$momentum_5', '$realized_vol_6',
       '$relative_volatility_index', 'OPEN1', 'RSV1', 'RSV2', 'RSV3',
       'VOLUME1', 'btc_std_7d', 'btc_zscore_14d', 'fg_std_7d', 'fg_zscore_14d',
       'vol_momentum_scaled', 'vol_raw', 'vol_raw_decile', 'vol_raw_momentum',
       'vol_risk', 'vol_scaled'],
      dtype='object') """
        
    # build interaction / regime signals    
    df_all["prob_up"] = df_all.apply(prob_up_piecewise, axis=1)

    # To remove rows where 'col1' has null values
    df_all.dropna(subset=['prob_up'], inplace=True)
    print("\nDataFrame after dropping rows with nulls in 'prob_up':")
    
    # df_all.to_csv("./debug.csv")
    # raise SystemExit()

    df_all = q50_regime_aware_signals(df_all)

    # Standalone functions for now to allow pipeline clarity
    
    df_all["signal_tier"] = df_all.apply(signal_classification, axis=1)    
    df_all["kelly_position_size"] = df_all.apply(kelly_sizing, axis=1)    

    
    
    alpha = 1.0  # controls “steepness”
    cap = 3.0
    df_all["signal_tanh"] = np.tanh(df_all["signal_rel"].clip(-cap, cap) / alpha)
    
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

    correlation_matrix = df_all.corr()        
    correlation_matrix.to_csv("./temp/correlation_matrix.csv")
        
    # Drop redundant columns if needed (updated for optimized loader)
    columns_to_drop = []

    if columns_to_drop:
        df_all.drop(columns_to_drop, axis=1, inplace=True)
        print(f"Dropped columns: {columns_to_drop}")
    
    df_all.to_csv("./temp/df_all_macro_analysis.csv")

    df_cleaned = df_all.dropna(subset=["vol_risk","vol_scaled","vol_raw_momentum","signal_thresh_adaptive","signal_tanh","enhanced_info_ratio"])

    df_cleaned.to_pickle("./data3/macro_features.pkl") # pickled features used in "train_meta_wrapper.py" process
