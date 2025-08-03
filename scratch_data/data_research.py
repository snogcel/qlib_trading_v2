import sys
import pandas as pd
import numpy as np

#import lightgbm as lgb

import qlib
from qlib.data.dataset import DataHandlerLP, DatasetH
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import NestedDataLoader, DataLoaderDH
from qlib.data.filter import NameDFilter

from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader
from src.data.nested_data_loader import CustomNestedDataLoader

import optuna
from qlib.contrib.model.gbdt import LGBModel

import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation


from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.utils import init_instance_by_config
from qlib.data import D
from qlib.config import C
from qlib.constant import REG_US, REG_CN 

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



if __name__ == '__main__': 

    gdelt_dh = gdelt_handler(**gdelt_handler_kwargs)
    crypto_dh = crypto_handler(**crypto_handler_kwargs)

    print("gdelt_processors: ", gdelt_dh.get_all_processors())
    print("crypto_processors: ", crypto_dh.get_all_processors())

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
    ds = DatasetH(dh, segments={"train": ("20170101", "20221231"), "valid": ("20230101", "20231231"), "test": ("20240101", "20250401")})


    params = {
        "early_stopping_rounds": 50,
        "objective": "regression",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
        "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "device": "cpu",
        "verbose": False,
        "boosting_type": "gbdt",
    }
    
    df_train = ds.prepare(segments="train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
    df_valid = ds.prepare(segments="valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)

    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]

    # Lightgbm need 1D array as its label
    if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        y_train, y_valid = np.squeeze(y_train.values), np.squeeze(y_valid.values)
    else:
        raise ValueError("LightGBM doesn't support multi-label training")
    
    dtrain = lgb.Dataset(x_train.values, label=y_train)
    dvalid = lgb.Dataset(x_valid.values, label=y_valid)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )




    with R.start(experiment_name="optuna_tuning_exp"):
        model.fit(ds)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        sr = SignalRecord(model, ds, rec)
        sr.generate()

        # Load the validation metric safely
        eval_results = rec.load_object("record_evaluate_result")
        val_score = eval_results.get("valid", {}).get("l2", float("inf"))



    # Create and run the Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)
    print("Best validation score:", study.best_value)


    

    
    raise SystemExit()
    
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
