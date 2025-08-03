import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader
from src.data.nested_data_loader import CustomNestedDataLoader
from qlib.data.dataset import DataHandlerLP, DatasetH
from qlib.contrib.report import analysis_model, analysis_position
from qlib.constant import REG_US, REG_CN

import pandas as pd

import lightgbm as lgbm



provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTODATA"

MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_002"
FREQ = "day"

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

gdelt_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    "instruments": ["GDELT_Feat"]
}

crypto_handler_kwargs = {
    "start_time": "20170818",
    "end_time": "20250401",
    #"instruments": ["AAVEUSDT","ADAUSDT","BCHUSDT","BNBUSDT","BTCUSDT","DASHUSDT","DOGEUSDT","ETHUSDT","LTCUSDT","NEARUSDT","SOLUSDT","ZECUSDT"]
    "instruments": ["AAVEUSDT","ADAUSDT","ALGOUSDT","ATOMUSDT","AVAXUSDT","BATUSDT","BCHUSDT","BNBUSDT","BTCUSDT","CAKEUSDT","CHZUSDT","CRVUSDT","DASHUSDT","DEXEUSDT","DOGEUSDT","DOTUSDT","ENAUSDT","ENJUSDT","EOSUSDT","ETCUSDT","ETHUSDT","FILUSDT","GRTUSDT","HBARUSDT","ICPUSDT","IOSTUSDT","IOTAUSDT","LINKUSDT","LTCUSDT","MANAUSDT","NEARUSDT","NEOUSDT","QTUMUSDT","RVNUSDT","SANDUSDT","SHIBUSDT","SOLUSDT","SUSHIUSDT","TFUELUSDT","THETAUSDT","TIAUSDT","TRXUSDT","UNIUSDT","VETUSDT","XLMUSDT","XRPUSDT","XTZUSDT","ZECUSDT","ZILUSDT"]
}

selected_instruments = crypto_handler_kwargs["instruments"] + gdelt_handler_kwargs["instruments"]

qlib.init(provider_uri=provider_uri, region=REG_US, logging_level="WARNING")

experiment_id = "193365370523522933"
rid = "47e38a25997c49e2bc28fb5ee40b5db1"

""" 
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset, save_path = ".model-pytorch")
    R.save_objects(trained_model=model)
    recorder = R.get_recorder() """


if __name__ == '__main__': 
    
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

    # causes issues with backtesting LightGBM, you can use the workaround or go manually...

    ds = DatasetH(dh, segments={"train": ("20170818", "20221231"), "valid": ("20230101", "20231231"), "test": ("20240101", "20250401")}, data_key=DataHandlerLP.DK_L)

    df_train, df_valid = ds.prepare(
        segments=["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    # split data
    X_train, y_train = df_train["feature"], df_train["label"]
    X_val, y_val = df_valid["feature"], df_valid["label"]

    # Lightgbm need 1D array as its label   
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()

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

    # create the LightGBM regressor with the optimized parameters
    # model = lgbm.LGBMRegressor(**hp_lgbm)

    with R.start(experiment_id=experiment_id, recorder_id=rid, resume=True):
        recorder = R.get_recorder(recorder_id=rid)
        model = recorder.load_object("trained_model")

        print(model)

        
        """ sr = SignalRecord(model, ds, recorder)
        
        label = df_valid["label"]
        
        print("label: ", label)
        
        R.save_objects(**{"label.pkl": label}, artifact_path=None)
        
        print("pred_dataset: ", df_valid["feature"])

        pred = model.predict(df_valid["feature"])

        pred = pd.Series(pred)

        if isinstance(pred, pd.Series):
            print("-is series-")
            pred = pred.to_frame("score")

        pred.index = label.index

        print("pred: ", pred)
                
        pred.to_csv("pred.csv")
        label.to_csv("label.csv")

        R.save_objects(**{"pred.pkl": pred}, artifact_path=None)  """


        #sar = SigAnaRecord(recorder)
        #sar.generate()


        #par = PortAnaRecord(recorder, port_analysis_config, "day")
        #par.generate()


        # load previous results
        pred_df = recorder.load_object("pred.pkl")
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")


        analysis_position.report_graph(report_normal_df, show_notebook=True)
        









# backtest and analysis
""" with R.start(experiment_name=EXP_NAME, recorder_id=rid, resume=True):
    # signal-based analysis
    rec = R.get_recorder()
    sar = SigAnaRecord(rec)
    sar.generate()

    #  portfolio-based analysis: backtest
    par = PortAnaRecord(rec, port_analysis_config, "day")
    par.generate() """