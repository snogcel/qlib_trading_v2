import sys
from pprint import pprint
import pandas as pd
import numpy as np

import qlib
from qlib.data.dataset import DataHandlerLP, DatasetH
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import NestedDataLoader, DataLoaderDH
from qlib.data.filter import NameDFilter

from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader
from src.data.nested_data_loader import CustomNestedDataLoader

from qlib.contrib.model.gbdt import LGBModel



import optuna
# import optuna.integration.lightgbm as lgb #TODO remove, this one sucks
import qlib.contrib.model.gbdt as lgb
import lightgbm as lgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation
import shap

from lightgbm import Dataset
from optuna_integration._lightgbm_tuner import LightGBMTuner
from optuna_integration._lightgbm_tuner import LightGBMTunerCV


from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils import init_instance_by_config
from qlib.data import D
from qlib.config import C
from qlib.constant import REG_US, REG_CN 

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso


import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
#from qlib.workflow.task.utils import time_to_datetime
# from qlib.backtest import backtest, executor
from qlib.backtest import executor

from qlib.contrib.evaluate import backtest_daily
# from qlib.contrib.strategy import BaseStrategy

from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_position
from qlib.strategy.base import BaseStrategy

from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy

from qlib.backtest.signal import ModelSignal

import copy
from qlib.backtest.position import Position
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO

from qlib.utils.time import Freq
from qlib.utils import flatten_dict


provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTODATA"

MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_005"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)

""" gdelt_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    "fit_start_time": "20170818",
    "fit_end_time": "20231231",
    "instruments": ["GDELT_Feat"]
}

crypto_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    "fit_start_time": "20170818",
    "fit_end_time": "20231231",
    "instruments": ["BTCUSDT"]
    #"instruments": ["BCHUSDT","BNBUSDT","BTCUSDT","DASHUSDT","DOGEUSDT","ETHUSDT","LTCUSDT"]
    #"instruments": ["BTCUSDT","ETHUSDT","LTCUSDT"]
    #"instruments": ["AAVEUSDT","ADAUSDT","BCHUSDT","BNBUSDT","BTCUSDT","DASHUSDT","DOGEUSDT","ETHUSDT","LTCUSDT","NEARUSDT","SOLUSDT","ZECUSDT"]
    #"instruments": ["AAVEUSDT","ADAUSDT","ALGOUSDT","ATOMUSDT","AVAXUSDT","BATUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CAKEUSDT","CHZUSDT","CRVUSDT","DASHUSDT","DEXEUSDT","DOGEUSDT","DOTUSDT","ENAUSDT","ENJUSDT","EOSUSDT","ETCUSDT","ETHUSDT","FILUSDT","GRTUSDT","HBARUSDT","ICPUSDT","IOSTUSDT","IOTAUSDT","LINKUSDT","LTCUSDT","MANAUSDT","NEARUSDT","NEOUSDT","QTUMUSDT","RVNUSDT","SANDUSDT","SHIBUSDT","SOLUSDT","SUSHIUSDT","TFUELUSDT","THETAUSDT","TIAUSDT","TRXUSDT","UNIUSDT","VETUSDT","XLMUSDT","XRPUSDT","XTZUSDT","ZECUSDT","ZILUSDT"]
}

selected_instruments = crypto_handler_kwargs["instruments"] + gdelt_handler_kwargs["instruments"]
 """

n_folds = 5
seed = 2042

# create KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

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

        print("train_index: ", train_idx)
        print("val_index: ", val_idx)

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


if __name__ == '__main__': 

    gdelt_handler_kwargs = {
        "start_time": "20170818",
        "end_time": "20250401",
        "fit_start_time": "20170818",
        "fit_end_time": "20231231",
        "instruments": ["GDELT_Feat"]
    }

    crypto_handler_kwargs = {
        "start_time": "20170818",
        "end_time": "20250401",
        "fit_start_time": "20170818",
        "fit_end_time": "20231231",
        "instruments": ["BTCUSDT"]
    }

    selected_instruments = crypto_handler_kwargs["instruments"] + gdelt_handler_kwargs["instruments"]

    gdelt_dh = gdelt_handler(**gdelt_handler_kwargs)
    crypto_dh = crypto_handler(**crypto_handler_kwargs)
    
    data_loader = CustomNestedDataLoader(dataloader_l=[
            {
                "class": "qlib_custom.crypto_loader.crypto_dataloader",
                "kwargs": {
                    "config": {
                        "feature": crypto_dataloader.get_feature_config()
                    },
                    "freq": FREQ,
                    "inst_processors": [],
                }
            }, {
                "class": "qlib_custom.gdelt_loader.gdelt_dataloader",
                "kwargs": {
                    "config": {
                        "feature": gdelt_dataloader.get_feature_config(),       
                    },
                    "freq": FREQ,
                    "inst_processors": []
                }
            }
        ], join="left")   

    df = data_loader.load(instruments=selected_instruments, start_time="20170818", end_time="20250401")  

    dh = DataHandlerLP(
        instruments=selected_instruments,
        start_time="20170818",
        end_time="20250401",
        data_loader=data_loader,
        process_type="append",
        drop_raw=False
    ).from_df(df)

    dataset_conf = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": dh,
            "segments": {
                "train": ("2017-08-18", "2022-12-31"),
                "valid": ("2023-01-01", "2023-12-31"),
                "test": ("2024-01-01", "2025-04-01"),
            },
        },
    }

    # dataset = init_instance_by_config(dataset_conf)

    # causes issues with backtesting LightGBM, you can use the workaround or go manually...

    ds = DatasetH(dh, segments={"train": ("20170818", "20221231"), "valid": ("20230101", "20231231"), "test": ("20240101", "20250401")}, data_key=DataHandlerLP.DK_L)

    ds.setup_data()

    df_train, df_valid, df_test = ds.prepare(
        segments=["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    # split data
    X_train, y_train = df_train["feature"], df_train["label"]
    X_val, y_val = df_valid["feature"], df_valid["label"]
    X_test, y_test = df_test["feature"], df_test["label"]

    #dtrain = Dataset(X_train, label=y_train)
    #dval = Dataset(X_val, label=y_val)

    feature_names = list(X_train.columns)
    print("feature_names: ", feature_names)

    print(X_train.head())

    ds_train = Dataset(data=X_train, label=y_train, feature_name="auto")
    ds_val = Dataset(data=X_val, label=y_val, reference=ds_train, feature_name="auto")

    # define the objective function for Optuna optimization
    def objective(trial):

        params = {
            "early_stopping_rounds": 50,
            "random_state": seed,
            "objective": "regression",
            "metric": ["l2","l1"],
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
            #"max_depth": trial.suggest_int("max_depth", 4, 10),
            #"num_leaves": trial.suggest_int("num_leaves", 20, 256),
            "device": "cpu",
            "verbose": 1,
            "boosting_type": "gbdt",
        }

        # create the LightGBM regressor with the optimized parameters
        model = lgbm.LGBMRegressor(**params) 

        # perform cross-validation using the optimized LightGBM regressor
        lgbm_model, mean_score = cross_validation_fcn(df_train, model, early_stopping_flag=True)

        # retrieve the best iteration of the model and store it as a user attribute in the trial object
        best_iteration = lgbm_model.best_iteration_
        trial.set_user_attr('best_iteration', best_iteration)
            
        return mean_score




    """ # Create an optimization study with Optuna library
    study = optuna.create_study(direction="minimize",study_name="lgbm_opt")

    # Optimize the study using a user-defined objective function, for a total of 50 trials
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
    params = {
        "metric": "rmse",
        "early_stopping_rounds": 50,
        "objective": "regression",
        "metric": ["l2","l1"],
        "random_state": seed,
        "learning_rate": trial.params["learning_rate"],
        "colsample_bytree": trial.params["colsample_bytree"],
        "subsample": trial.params["subsample"],
        "lambda_l1": trial.params["lambda_l1"],
        "lambda_l2": trial.params["lambda_l2"],
        #"max_depth": trial.params["max_depth"],
        #"num_leaves": trial.params["num_leaves"],
        "device": "cpu",
        "boosting_type": "gbdt",
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
    y_pred_df.index = X_val.index """


    # === Train Regression Model with Optuna-Tuned Parameters (Adapted for 4h) ===
    """ model = LGBModel(
        objective="regression",
        boosting_type="gbdt",
        learning_rate=0.05,
        #n_estimators=300,
        #num_leaves=31,
        #colsample_bytree=0.9462,
        #subsample=0.9783,
        #max_depth=-1,
        lambda_l1=0.5503,
        lambda_l2=4.7943,
        #min_child_samples=20,
        #min_child_weight=0.001,
        #min_split_gain=0.0,
        #subsample_for_bin=200000,
        #subsample_freq=0,
        importance_type="gain",
        num_threads=8
    ) """

    params = {
        "objective": "quantile",   # <-- Quantile Regression
        # "alpha": 0.9,              # <-- 90th percentile
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42
    }

    class QuantileLGBModel(LGBModel):
        def __init__(self, alpha=0.9, **kwargs):
            super().__init__(loss="mse", **kwargs)  # temporary placeholder for base class
            self.params["objective"] = "quantile"
            self.params["alpha"] = alpha

    model_high = QuantileLGBModel(alpha=0.9, **params)
    model_mid = QuantileLGBModel(alpha=0.5, **params)
    model_low = QuantileLGBModel(alpha=0.1, **params)

    # === Train & Predict ===
    model_high.fit(ds)
    model_mid.fit(ds)
    model_low.fit(ds)

    pred_high = model_high.predict(ds)
    pred_mid = model_mid.predict(ds)
    pred_low = model_low.predict(ds)

    pred_high_df = pd.DataFrame(pred_high)
    pred_mid_df = pd.DataFrame(pred_mid)
    pred_low_df = pd.DataFrame(pred_low)

    model_test = LGBModel()

    feat_importance = model_high.get_feature_importance(importance_type='gain')
    #feat_importance = model_high.model.feature_importance(importance_type='gain')

    #importance = model_high.feature_importance(importance_type='gain')
    feature_names = model_high.model.feature_name()

    print(feat_importance)
    print(feature_names)



    

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
        print("Aligned quantile_errors:\n", errors.head())  # sample output
            
        coverage = (y_true < y_pred).mean()
        print(f"Q90 empirical coverage: {coverage:.2%}")

        return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))



    print("Quantile Loss (Q90):", quantile_loss(y_test, pred_high_df, 0.90))

    print("Quantile Loss (Q50):", quantile_loss(y_test, pred_mid_df, 0.50))

    print("Quantile Loss (Q10):", quantile_loss(y_test, pred_low_df, 0.10))
    

    import matplotlib.pyplot as plt

    # Filter for one instrument (e.g., BTCUSDT)
    instrument = "BTCUSDT"

    # Extract each series for this instrument
    q10 = pred_low_df.xs(instrument, level="instrument").squeeze()
    q50 = pred_mid_df.xs(instrument, level="instrument").squeeze()
    q90 = pred_high_df.xs(instrument, level="instrument").squeeze()
    y_true = y_test.xs(instrument, level="instrument").squeeze()

    # Sort by time for better plotting
    df_plot = pd.DataFrame({
        "Q10": q10,
        "Q50": q50,
        "Q90": q90,
        "True": y_true
    }).dropna().sort_index()

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df_plot.index, df_plot["True"], label="True", color="black", linewidth=1.5)
    plt.plot(df_plot.index, df_plot["Q50"], label="Q50 (Median)", color="blue", linestyle="--")
    plt.fill_between(
        df_plot.index,
        df_plot["Q10"],
        df_plot["Q90"],
        color="skyblue",
        alpha=0.3,
        label="Q10-Q90 Band"
    )

    plt.title("Prediction Interval: Q10-Q90 with Median vs. True Values")
    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    """ Q90 empirical coverage: 89.28%
    Quantile Loss (Q90): 0.0052691501181583045

    Q50 empirical coverage: 50.11%
    Quantile Loss (Q50): 0.010022451474169703

    Q10 empirical coverage: 8.53%
    Quantile Loss (Q10): 0.004958847070057403 """


    """ Q90 empirical coverage: 89.06%
    Quantile Loss (Q90): 0.005318127952459868

    Q50 empirical coverage: 51.42%
    Quantile Loss (Q50): 0.00992663853241363

    Q90 empirical coverage: 8.75%
    Quantile Loss (Q10): 0.004950012925536299 """


    # Compute the root mean squared error (RMSE) between the predicted and actual target values for the test data
    # print("RMSE:", np.sqrt(MSE(pred, y_test)))

    # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
    # print("R2: ", r2_score(y_test, pred))

    # print(pred_signal)

    


    import warnings

    
        

    from typing import Dict, List, Text, Tuple, Union
    from qlib.model.base import BaseModel
    from qlib.backtest.signal import Signal, create_signal_from

    # === Strategy: Simple Threshold Long Strategy for Single Instrument ===
    class ThresholdLongStrategy(BaseSignalStrategy):
        def __init__(
            self,
            *,
            signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
            model=None,
            dataset=None,
            risk_degree: float = 0.95,
            threshold: float = 0.001,
            trade_exchange=None,
            level_infra=None,
            common_infra=None,
            **kwargs,
        ):

            # this is creating problems with signal creation
            # super().__init__(signal=signal, model=model, dataset=dataset, **kwargs)
            # super().__init__(signal=signal, risk_degree=risk_degree, trade_exchange=None, **kwargs)

            self.risk_degree = risk_degree
            self.threshold = threshold  # Threshold for buying signal

            # This is trying to be compatible with previous version of qlib task config
            if model is not None and dataset is not None:
                warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
                signal = model, dataset

            self.signal: Signal = create_signal_from(signal)

            #print("signal passed to strategy: ", self.signal)

            print(self.signal.get_signal(start_time='2024-10-25 00:00:00', end_time='2024-10-25 23:59:59'))
            #print(self.signal)
            #print(signal)


        def generate_target_weight_position(self, score):
            pos = score.copy()
            pos["score"] = (pos["score"] > self.threshold).astype(float)
            return pos
        
        def generate_trade_decision(self, *args, **kwargs):
            # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
            trade_step = self.trade_calendar.get_trade_step()
            trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
            pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
            
            pred_score = self.signal.get_signal(start_time=trade_start_time, end_time=trade_end_time)

            #print("step trade_start_time: ", trade_start_time)
            #print("step trade_end_time: ", trade_end_time)

            #print("step pred_start_time: ", pred_start_time)
            #print("step pred_end_time: ", pred_end_time)
            #print(pred_score.columns)
            

            # Then produce a dictionary {stock: weight or shares}
            # decisions = self.generate_target_weight_position(pred_score) # Replace with your logic to calculate target positions
            
            # print(decisions)

            #return decisions # Return a TradeDecision object
        
            # return TradeDecisionWO(order_list=[], strategy=self)

            # raise SystemExit()

            # If no score, do nothing
            if pred_score is None:
                return TradeDecisionWO([], self)

            # If multiple columns, pick the first
            if isinstance(pred_score, pd.DataFrame):
                pred_score = pred_score.iloc[:, 0]
            
            print(f"{trade_start_time} to {trade_end_time} pred_score: {pred_score.values}")

            # Copy current position
            current_temp: Position = copy.deepcopy(self.trade_position)
            sell_order_list = []
            buy_order_list = []
            cash = current_temp.get_cash()
            current_stock_list = current_temp.get_stock_list()

            # Helper functions to determine sell_order_list and buy_order_list
            # ...TODO

            return TradeDecisionWO(sell_order_list + buy_order_list, self)


    # === Backtest with 4h frequency ===
    # exec = executor.SimulatorExecutor(time_per_step=FREQ, generate_portfolio_metrics=True)
    
    """ report, positions = backtest(
        pred=pred,
        strategy=strategy,
        executor=exec,
        #get_score_fn=None,
        #verbose=True
    ) """

    # signal object
    """ signal = create_signal_from(pred)

    print("create_signal_from: ", signal) """

    pred_signal = ModelSignal(model_high, ds)

    print("PRED_SIGNAL: ", pred_signal)

    STRATEGY_CONFIG = {
        #"signal": signal,
        #"threshold": 0.001,
        # pred_score, pd.Series
        #"signal": pred_signal,
        "model": model_high,
        "dataset": ds,
    }

    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }

    backtest_config = {
        "start_time": "2024-01-02",
        "end_time": "2025-03-31",
        "account": 100000000,
        "benchmark": "BTCUSDT",
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    # strategy object    
    strategy_obj = ThresholdLongStrategy(signal=pred_signal, **STRATEGY_CONFIG)

    # executor object
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)

    # backtest
    # Suppress the warning using np.errstate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        portfolio_metric_dict, indicator_dict = backtest_daily(executor=executor_obj, strategy=strategy_obj, **backtest_config)
        analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))

    # print(portfolio_metric_dict)
    # print(indicator_dict)
    
    # backtest info
    # report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    
    raise SystemExit()

    # analysis
    analysis = dict()
    analysis["excess_return_without_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"], freq=analysis_freq
    )
    analysis["excess_return_with_cost"] = risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
    )

    analysis_df = pd.concat(analysis)  # type: pd.DataFrame

    # log metrics
    analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())

    # print out results
    pprint(f"The following are analysis results of benchmark return({analysis_freq}).")
    pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
    pprint(f"The following are analysis results of the excess return without cost({analysis_freq}).")
    pprint(analysis["excess_return_without_cost"])
    pprint(f"The following are analysis results of the excess return with cost({analysis_freq}).")
    pprint(analysis["excess_return_with_cost"])


    #y_pred_df.to_csv('y_pred.csv')
    #y_val.to_csv('y_val.csv')
    
    # Compute the root mean squared error (RMSE) between the predicted and actual target values for the test data
    # print("Best rmse:", np.sqrt(MSE(y_pred_lgbm, y_val)))

    # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
    #print("R2 using LightGBM: ", r2_score(y_val, y_pred_lgbm ))

    

    

    import matplotlib.pyplot as plt
    lgbm.plot_importance(lgbm_model, max_num_features=20)
    plt.show()

    # Create a SHAP explainer object for the LightGBM model
    explainer = shap.TreeExplainer(lgbm_model)

    # Sample 10% of the test data to use for faster computation of SHAP values
    sample = X_val.sample(frac=0.1, random_state=42)

    # Calculate the SHAP values for the sampled test data using the explainer
    shap_values = explainer.shap_values(sample)

    # Create a SHAP summary plot of the feature importances based on the SHAP values
    shap.summary_plot(shap_values, sample, show=False)

    # Adjust the layout of the plot to avoid overlapping labels
    plt.tight_layout()

    # Show the plot
    plt.show()

    model = R.load_object("trained_model")  # Load best model
    import matplotlib.pyplot as plt
    lgb_model = model.model  # Qlib wraps it
    lgb.plot_importance(lgb_model, max_num_features=20)
    plt.show()


    
    

    

    ## QLib Stuff
    with R.start(experiment_name="optuna_tuning_exp"):
        model.fit(ds)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        sr = SignalRecord(model, ds, rec)
        sr.generate()

        # Load the validation metric safely
        eval_results = rec.load_object("record_evaluate_result")
        val_score = eval_results.get("valid", {}).get("l2", float("inf"))




    # fit the model
    """ model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=500,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=500),
            lgb.log_evaluation(50)
        ]
    )

    print(model) """


    

    ###################################
    # prediction, backtest & analysis
    ###################################
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": "<PRED>",
                "topk": 10,
                "n_drop": 3,
            },
        },
        "backtest": {
            "start_time": "2024-01-01",
            "end_time": "2025-04-01",
            "account": 100000000,
            "benchmark": BENCHMARK,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    #kwargs are effectively the same params as before
    params = {            
        "loss": "mse",
        "objective": "regression",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.2,
        "subsample": 0.8789,   
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,              
        "device": "cpu",
        "verbose": True
    }

    model = init_instance_by_config(
        {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "early_stopping_rounds": 50,
                "objective": "regression",
                "learning_rate": 0.05,
                "colsample_bytree": 0.8,
                "subsample": 0.8,
                "lambda_l1": 10,
                "lambda_l2": 10,
                "max_depth": 6,
                "num_leaves": 64,
                "device": "cpu",
                "num_threads": 20,
                "verbose": True
            },
        }
    )

    # start exp to train model
    with R.start(experiment_name=EXP_NAME):
        model.fit(ds)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        rid = rec.id  # save the record id

        # Inference and saving signal
        sr = SignalRecord(model, ds, rec)
        sr.generate()

    # backtest and analysis
    with R.start(experiment_name=EXP_NAME, recorder_id=rid, resume=True):
        # signal-based analysis
        rec = R.get_recorder()
        sar = SigAnaRecord(rec)
        sar.generate()

        #  portfolio-based analysis: backtest
        par = PortAnaRecord(rec, port_analysis_config, "day")
        par.generate()


    # reload saved data

    # print("Experiment Name: ", EXP_NAME)
    # Experiment ID: 754195334097261599
    # print("Recorder ID: a0168e37d51e4569857e4747d11d695f")
    # Run ID: f8fc8f46786042a4853f0966387a7f25

    # load recorder
    recorder = R.get_recorder(recorder_id=rid, experiment_name=EXP_NAME)

    # load previous results
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    # Previous Model can be loaded. but it is not used.
    loaded_model = recorder.load_object("trained_model")
    #print(loaded_model)

    from qlib.contrib.report import analysis_model, analysis_position

    #analysis_position.report_graph(report_normal_df, show_notebook=True)

    raise SystemExit()


    

    #ds.to_csv('testdata001.csv')

    #total_instruments = crypto_handler_kwargs.instruments + gdelt_handler_kwargs.instruments

    #print(total_instruments)
