import qlib
from qlib.constant import REG_US, REG_CN 
from qlib.data.dataset import DatasetH, DataHandlerLP
from qlib.data.dataset.handler import DataHandler
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from inspect import getfullargspec

from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader

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
EXP_NAME = "crypto_exp_100"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)

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

if __name__ == "__main__": 
      
    _learn_processors = []
    _infer_processors = []

    infer_processors = check_transform_proc(_infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(_learn_processors, fit_start_time, fit_end_time)

    inst_processors = [
        {
            "class": "TimeRangeFlt",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "start_time": train_start_time,
                "end_time": test_end_time,
                "freq": "60min"
            }
        }
    ]

    loader_config = {
        "feature": (
            ["Resi($close, 15)/$close", "Std(Abs($close/Ref($close, 1)-1)*$volume, 5)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, 5)+1e-12)", "Rsquare($close, 5)", "($high-$low)/$open", "Rsquare($close, 10)", "Corr($close, Log($volume+1), 5)", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 5)", "Corr($close, Log($volume+1), 10)", "Rsquare($close, 20)", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 60)", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), 10)", "Corr($close, Log($volume+1), 20)", "(Less($open, $close)-$low)/$open"], 
            ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW"],
        ),
        "label": (
            ["Ref($close, -2)/Ref($close, -1) - 1"],
            ["LABEL0"],
        )
    }

    freq_config = {
        "feature": "60min", 
        "label": "day"
    }

    data_loader_config = {
        "class": "QlibDataLoader",
        "kwargs": {
            "config": loader_config,
            "freq": freq_config,
            "inst_processors": inst_processors,                         
        },
    }

    crypto_data_loader = {
        "class": "crypto_dataloader",
        "module_path": "qlib_custom.crypto_loader",
        "kwargs": {
            "config": {
                "feature": crypto_dataloader.get_feature_config(),
                "label": crypto_dataloader.get_label_config(),                                                
            },                                            
            "freq": freq_config["feature"],  # "day"
            "inst_processors": inst_processors
        }
    }

    gdelt_data_loader = {
        "class": "gdelt_dataloader",
        "module_path": "qlib_custom.gdelt_loader",
        "kwargs": {
            "config": {
                "feature": gdelt_dataloader.get_feature_config()
            },
            "freq": freq_config["label"],  # "day"
            "inst_processors": inst_processors
        }
    }

    handler_config = {
        "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
        "start_time": train_start_time,
        "end_time": test_end_time,                
        "data_loader": crypto_data_loader,
        "infer_processors": infer_processors,
        "learn_processors": learn_processors,
        "shared_processors": [],
        "process_type": DataHandlerLP.PTYPE_A,
        "drop_raw": False,        
    }

    dataset_handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": handler_config,
    }

    segments = {
        "train": (train_start_time, train_end_time),
        "valid": (valid_start_time, valid_end_time),
        "test": (test_start_time, test_end_time),
    }       
    datasetH = DatasetH(dataset_handler_config, segments)

    # prepare segments
    df_train, df_valid, df_test = datasetH.prepare(
        segments=["train", "valid", "test"], col_set=DataHandler.CS_ALL, data_key=DataHandlerLP.DK_I
    )

    print(df_train)
    df_train.to_csv("df_train.csv")
