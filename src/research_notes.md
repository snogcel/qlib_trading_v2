# 0.1: {'learning_rate': 0.060555113429817814, 'colsample_bytree': 0.7214813020361056, 'subsample': 0.7849919729082881, 'lambda_l1': 8.722794281828277e-05, 'lambda_l2': 3.220667556916701e-05, 'max_depth': 10, 'num_leaves': 224, **GENERIC_LGBM_PARAMS},
# 0.5: {'learning_rate': 0.02753370821225369, 'max_depth': -1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS},
# 0.9: {'learning_rate': 0.09355380738420341, 'max_depth': 10, 'num_leaves': 249, 'lambda_l1': 0.1, 'lambda_l2': 0.1, **GENERIC_LGBM_PARAMS}

# 0.1: {'learning_rate': 0.026, 'max_depth': 7, **CORE_LGBM_PARAMS},
# 0.5: {'learning_rate': 0.027, 'max_depth': 7, **CORE_LGBM_PARAMS},                
# 0.9: {'learning_rate': 0.028, 'max_depth': 7, **CORE_LGBM_PARAMS} 



# Next, align training_pipeline_optuna with training_pipeline:

# training_pipeline.py
CORE_LGBM_PARAMS = {
        "objective": "quantile",
        "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
        "boosting_type": "gbdt",
        "device": "cpu",
        "verbose": -1,
        "random_state": 141551, # https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
        "early_stopping_rounds": 50,
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
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,         # Use 80% of features per tree
        "bagging_fraction": 0.8,         # Use 80% of data per iteration
        "bagging_freq": 5,
    }

    multi_quantile_params = {
        0.1: {'max_depth': 25, **CORE_LGBM_PARAMS},
        0.5: {'max_depth': 25, **CORE_LGBM_PARAMS},                
        0.9: {'max_depth': 25, **CORE_LGBM_PARAMS}
    }


# training_pipeline_optuna
CORE_LGBM_PARAMS = {
        "objective": "quantile",
        "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
        "boosting_type": "gbdt",
        "device": "cpu",
        "verbose": -1,
        "random_state": 141551, # https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
        "early_stopping_rounds": 500, # adjust early stopped rounds as needed
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
        "min_data_in_leaf": 20,
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


# Run training_pipeline_optuna:
1. TODO - refresh my memory on how this is transferred over to training_pipeline, and into backtester.

2. Align SEED parameter to align between optuna and training_pipeline

3. Update Objective function in training_pipeline_optuna to target parameters for optimization.


# TODO:

4. Identify the resulting updated params (goal of bringing Q10 closer to 10% coverage):

Quantile Loss (Q90): 0.0010184848780716853, coverage: 85.19%
Quantile Loss (Q50): 0.0018443741312067223, coverage: 50.49%
Quantile Loss (Q10): 0.0010294973523945455, coverage: 14.08%
{'max_depth': 25, 'objective': 'quantile', 'metric': ['l1', 'l2'], 'boosting_type': 'gbdt', 'device': 'cpu', 'verbose': -1, 'random_state': 141551, 'early_stopping_rounds': 500, 'num_boost_round': 2250, 'seed': 42}

5. 






## Trial 1 (baseline)
train_start_time = "2018-08-02"
train_end_time = "2023-12-31"
valid_start_time = "2024-01-01"
valid_end_time = "2024-09-30"
test_start_time = "2024-10-01"
test_end_time = "2025-04-01"

SEED = 42
MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_101"
FREQ = "day"

CORE_LGBM_PARAMS = {
    "objective": "quantile",
    "metric": ["l1", "l2"], # , "l2", "l1" # "rmse"
    "boosting_type": "gbdt",
    "device": "cpu",
    "verbose": -1,
    "random_state": 141551, # https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
    "early_stopping_rounds": 50,
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
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,         # Use 80% of features per tree
    "bagging_fraction": 0.8,         # Use 80% of data per iteration
    "bagging_freq": 5,
}

multi_quantile_params = {
    0.1: {'max_depth': 25, **CORE_LGBM_PARAMS},
    0.5: {'max_depth': 25, **CORE_LGBM_PARAMS},                
    0.9: {'max_depth': 25, **CORE_LGBM_PARAMS}
}

Enhanced vs Traditional Info Ratio:
   Traditional (signal/spread): 0.030
   Enhanced (signal/total_risk): 0.040
Expected Value Analysis:
   Mean expected value: 0.0002
   Positive expected value: 57.2%
   Mean potential gain: 0.0050
   Mean potential loss: 0.0048
Economic Significance Comparison:
   Traditional threshold: 3,540 (6.6%)
   Expected value: 17,701 (32.8%)
   Combined approach: 15,076 (27.9%)
   Improvement: +400.0% more opportunities
 Variance-Based Regime Distribution:
   Low Variance: 16,194 (30.0%)
   High Variance: 10,796 (20.0%)
   Extreme Variance: 5,398 (10.0%)
Q50-centric signals generated:
   HOLD: 36,277 (67.2%)
   LONG: 11,164 (20.7%)
   SHORT: 6,537 (12.1%)
Trading signal quality:
   Average Info Ratio (traditional): 0.03
   Average Enhanced Info Ratio (variance-aware): 0.04
   Average |Q50|: 0.0003
   Average Vol_Risk (variance): 0.000071 (âˆš = 0.008)
   Average Signal Strength: 0.0000
   Average Position Size Suggestion: 0.486


{
  "total_return": 0.11520824432635446,
  "annualized_return": 0.10023088213337905,
  "volatility": 0.08866812932793894,
  "sharpe_ratio": 1.1216409494847956,
  "max_drawdown": -0.05669428210002597,
  "total_trades": 3080,
  "total_holds": 6921,
  "total_periods": 10000,
  "trade_frequency": 0.308,
  "hold_frequency": 0.6921,
  "hold_reasons": {
    "HOLD_NO_POSITION": 1,
    "HOLD_POSITION": 6920
  },
  "win_rate": 0.10649350649350649,
  "final_balance": 111479.19687589817,
  "final_position": 0.0,
  "final_portfolio_value": 111520.82443263545
}
