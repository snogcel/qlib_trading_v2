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

    df_train, df_valid = ds.prepare(
        segments=["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    # split data
    X_train, y_train = df_train["feature"], df_train["label"]
    X_val, y_val = df_valid["feature"], df_valid["label"]

    dtrain = Dataset(X_train, label=y_train)
    dval = Dataset(X_val, label=y_val)

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


        
        """ "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        #"colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
        "colsample_bytree": 0.8879,
        #"subsample": trial.suggest_uniform("subsample", 0, 1),
        "subsample": 0.8789,
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
        "max_depth": 8,
        #"num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "num_leaves": 210, """


        """ task = {
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {
                    "loss": "mse",
                    "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1),
                    "learning_rate": trial.suggest_uniform("learning_rate", 0, 1),
                    "subsample": trial.suggest_uniform("subsample", 0, 1),
                    "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 1e4),
                    "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 1e4),
                    "max_depth": 10,
                    "num_leaves": trial.suggest_int("num_leaves", 1, 1024),
                    "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                    "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                },
            },
        } """

        # create the LightGBM regressor with the optimized parameters
        model = lgbm.LGBMRegressor(**params)
        
        # model = lgbm.train(params, dtrain, valid_sets=[dval], valid_names="valid")   

        # perform cross-validation using the optimized LightGBM regressor
        lgbm_model, mean_score = cross_validation_fcn(df_train, model, early_stopping_flag=True)

        # retrieve the best iteration of the model and store it as a user attribute in the trial object
        best_iteration = lgbm_model.best_iteration_
        trial.set_user_attr('best_iteration', best_iteration)
            
        return mean_score




    # Create an optimization study with Optuna library
    study = optuna.create_study(direction="minimize",study_name="lgbm_opt")




    """ tuner = LightGBMTunerCV(
        params,
        dtrain,
        folds=KFold(n_splits=3),
        study=study,
        optuna_seed=seed,
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value)) """

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

    #print(hp_lgbm)

    # Create a LightGBM regression model using the best set of hyperparameters found during the optimization process
    lgbm_model = lgbm.LGBMRegressor(**hp_lgbm)

    

    # create the LightGBM regressor with the optimized parameters
    # lgbm_model = lgbm.train(params, dtrain, valid_sets=[dval], valid_names="valid")

    """ task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": trial.params["colsample_bytree"],
                "learning_rate": trial.params["learning_rate"],
                "subsample": trial.params["subsample"],
                "lambda_l1": trial.params["lambda_l1"],
                "lambda_l2": trial.params["lambda_l2"],
                "max_depth": trial.params["max_depth"],
                "num_leaves": trial.params["num_leaves"],
                
                # "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
                # "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
                # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            },
        },
    }
    model = init_instance_by_config(task["model"]) """

    print("model params: ", lgbm_model.get_params())

    # evals_result = dict()
    # lgbm_model.fit(ds, evals_result=evals_result)
    # print(evals_result)

    # Fit the model to the training data
    lgbm_model.fit(X_train, y_train, callbacks=[
        lgbm.log_evaluation(period=20)
    ])

    # Use the trained model to make predictions on the test data
    y_pred_lgbm = lgbm_model.predict(X_val)
    # y_pred_lgbm = lgbm_model.predict(dataset=ds,segment="valid")

    y_pred_df = pd.DataFrame(y_pred_lgbm)
    y_pred_df.index = X_val.index

    #y_pred_df.to_csv('y_pred.csv')
    #y_val.to_csv('y_val.csv')
    
    # Compute the root mean squared error (RMSE) between the predicted and actual target values for the test data
    print("Best rmse:", np.sqrt(MSE(y_pred_lgbm, y_val)))

    # Compute the R-squared (coefficient of determination) between the predicted and actual target values for the test data
    print("R2 using LightGBM: ", r2_score(y_val, y_pred_lgbm ))

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

    raise SystemExit()

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
