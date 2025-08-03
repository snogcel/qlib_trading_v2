import sys
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
import optuna.integration.lightgbm as lgb #TODO remove, this one sucks
import lightgbm as lgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation
import shap

from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils import init_instance_by_config
from qlib.data import D
from qlib.config import C
from qlib.constant import REG_US, REG_CN 

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import matplotlib.pyplot as plt

provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTODATA"

MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_004"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)

gdelt_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    "instruments": ["GDELT_Feat"]
}

crypto_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    "instruments": ["AAVEUSDT","ADAUSDT","ALGOUSDT","ATOMUSDT","AVAXUSDT","BATUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CAKEUSDT","CHZUSDT","CRVUSDT","DASHUSDT","DEXEUSDT","DOGEUSDT","DOTUSDT","ENAUSDT","ENJUSDT","EOSUSDT","ETCUSDT","ETHUSDT","FILUSDT","GRTUSDT","HBARUSDT","ICPUSDT","IOSTUSDT","IOTAUSDT","LINKUSDT","LTCUSDT","MANAUSDT","NEARUSDT","NEOUSDT","QTUMUSDT","RVNUSDT","SANDUSDT","SHIBUSDT","SOLUSDT","SUSHIUSDT","TFUELUSDT","THETAUSDT","TIAUSDT","TRXUSDT","UNIUSDT","VETUSDT","XLMUSDT","XRPUSDT","XTZUSDT","ZECUSDT","ZILUSDT"]
}

selected_instruments = crypto_handler_kwargs["instruments"] + gdelt_handler_kwargs["instruments"]


n_folds = 5
seed = 2042

# create KFold object
kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

def cross_validation_fcn(X_train, y_train, model, early_stopping_flag=False):
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
    
    print("X_train: ", X_train)
    print("y_train: ", y_train)
    
    mse_list = []
    for train_index, val_index in kf.split(X_train, y_train):

        print("train_index: ", train_index)
        print("val_index: ", val_index)

        # X_train, y_train = df_train["feature"], df_train["label"] # X
        # X_valid, y_valid = df_valid["feature"], df_valid["label"] # y

        # Split the data into training and validation sets
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        print("X_train_fold: ", X_train_fold)
        print("X_val_fold: ", X_val_fold)
        print("y_train_fold: ", y_train_fold)
        print("y_val_fold: ", y_val_fold)

        # Train the model on the training set
        if early_stopping_flag:
            # Use early stopping if enabled
            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],
                      callbacks=[lgbm.early_stopping(stopping_rounds=250, verbose=0)])
        else:
            model.fit(X_train_fold, y_train_fold)
            
        # Make predictions on the validation set and calculate the MSE score
        y_pred = model.predict(X_val_fold)
        mse = MSE(y_val_fold, y_pred)
        mse_list.append(mse)
        
    # Return the trained model and the average MSE score
    return model, np.mean(mse_list)


if __name__ == '__main__': 

    gdelt_dh = gdelt_handler(**gdelt_handler_kwargs)
    crypto_dh = crypto_handler(**crypto_handler_kwargs)

    nd = CustomNestedDataLoader(
        dataloader_l=[
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
        ]
    )

    df = nd.load(instruments=selected_instruments, start_time="20170818", end_time="20250401")
    # df.to_csv("testdata001.csv")
    
    dh = DataHandlerLP.from_df(df)
    ds = DatasetH(dh, segments={"insample": ("20170101", "20231231"), "outsample": ("20240101", "20250401")})

    df_train = ds.prepare(segments="insample", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    df_test = ds.prepare(segments="outsample", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        
    X_train, X_test, y_train, y_test = train_test_split(df_train["feature"], df_train["label"], test_size = 0.2, random_state = seed)

    # Lightgbm need 1D array as its label
    """ if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        y_train, y_test = np.squeeze(y_train.values), np.squeeze(y_test.values)
    else:
        raise ValueError("LightGBM doesn't support multi-label training") """
    
    # Extract the best set of hyperparameters from the best trial and store them in a variable
    hp_lgbm = {
        'learning_rate': 0.08373132499143446, 
        'colsample_bytree': 0.7108727976898331, 
        'subsample': 0.7726679138009557, 
        'lambda_l1': 0.9898051828613315, 
        'lambda_l2': 2.1908593280672513, 
        'max_depth': 10, 
        'num_leaves': 172
    }
    
    # Create a LightGBM regression model using the best set of hyperparameters found during the optimization process
    lgbm_model = lgbm.LGBMRegressor(**hp_lgbm)

    # Fit the model to the training data
    """ lgbm_model.fit(X_train, y_train)

    # Use the trained model to make predictions on the test data
    y_pred_lgbm = lgbm_model.predict(X_test)
    
    # Compute the root mean squared error (RMSE) between the predicted and actual target values for the test data
    print("Best rmse:", np.sqrt(MSE(y_pred_lgbm, y_test)))

    # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
    print("R2 using LightGBM: ", r2_score(y_test, y_pred_lgbm )) """
    
    #lgb.plot_importance(lgbm_model, max_num_features=20)
    #plt.show()

    with R.start(experiment_name="optuna_tuning_exp"):
        lgbm_model.fit(X_train, y_train)
        R.save_objects(trained_model=lgbm_model)

        rec = R.get_recorder()
        sr = SignalRecord(lgbm_model, ds, rec)
        sr.generate()

        # Load the validation metric safely
        eval_results = rec.load_object("record_evaluate_result")
        val_score = eval_results.get("valid", {}).get("l2", float("inf"))


    raise SystemExit()
    

    

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
