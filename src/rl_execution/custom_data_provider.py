import os
import time
import datetime
from typing import Optional

import qlib
from qlib import get_module_logger
from qlib.data import D
from qlib.config import REG_CN, REG_US
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc
from qlib.contrib.data.highfreq_provider import HighFreqProvider
from qlib.rl.data.native import HandlerProcessedDataProvider
from qlib.contrib.ops.high_freq import get_calendar_day, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut
import pickle as pkl
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from pathlib import Path



class MacroAwareProcessedDataProvider(HandlerProcessedDataProvider):
    def __init__(self, macro_feature_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macro_feature_path = macro_feature_path
        self.macro_features = self._load_macro_features()                

    def _load_macro_features(self):
        if self.macro_feature_path is None:
            return {}
        df = pd.read_pickle(self.macro_feature_path)
        TIER_MAP = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
        df["signal_tier"] = df["signal_tier"].map(TIER_MAP) 
        return df

    def get_data(self, stock_id, date, feature_dim, time_index):
        
        # Call base method to get today/yesterday features
        processed = super().get_data(stock_id, date, feature_dim, time_index)        

        # Inject macro features into each time step of today
        date_key = pd.Timestamp(date).floor("D")

        if isinstance(self.macro_features, pd.DataFrame) and date_key in self.macro_features.index:

            target_idx = self.macro_features.index.get_loc((date_key, stock_id))

            # Calculate the integer position of the previous row
            previous_idx = target_idx - 1

            # Check if the previous row exists (to prevent IndexError for the first row)
            if previous_idx >= 0:
                macro_row = self.macro_features.iloc[target_idx]
                macro_row_yesterday = self.macro_features.iloc[previous_idx]
            
            # Concatenate with original features            
            processed.today = processed.today.assign(**macro_row)
            processed.yesterday = processed.today.assign(**macro_row_yesterday)

        #print("DEBUG Processed: ", processed.today)
        #raise SystemExit()
        #         
        return processed

    def get_state(self, stock_id, datetime):
        state = super().get_state(stock_id, datetime)
        date_key = datetime.floor("D")
        if isinstance(self.macro_features, pd.DataFrame) and date_key in self.macro_features.index:
            macro_row = self.macro_features.loc[date_key]
            for key, val in macro_row.items():
                state[f"macro_{key}"] = val
        #print(f"[DEBUG] State at {datetime}: {state.keys()}")
        #raise SystemExit()
        return state


class CryptoHighFreqProvider(HighFreqProvider):
    def __init__(
        self,
        start_time: str,
        end_time: str,
        train_end_time: str,
        valid_start_time: str,
        valid_end_time: str,
        test_start_time: str,
        qlib_conf: dict,
        feature_conf: dict,
        label_conf: Optional[dict] = None,
        backtest_conf: dict = None,
        freq: str = "1min",
        **kwargs,
    ) -> None:
        """ super().__init__(start_time=start_time, 
                         end_time=end_time, 
                         train_end_time=train_end_time, 
                         valid_start_time=valid_start_time, 
                         valid_end_time=valid_end_time, 
                         test_start_time=test_start_time, 
                         qlib_conf=qlib_conf, 
                         feature_conf=feature_conf, 
                         label_conf=label_conf, 
                         backtest_conf=backtest_conf, 
                         freq=freq, 
                         **kwargs) """

        self.start_time = start_time
        self.end_time = end_time
        self.test_start_time = test_start_time
        self.train_end_time = train_end_time
        self.valid_start_time = valid_start_time
        self.valid_end_time = valid_end_time
        self._init_qlib(qlib_conf)
        self.feature_conf = feature_conf
        self.label_conf = label_conf
        self.backtest_conf = backtest_conf
        self.qlib_conf = qlib_conf
        self.logger = get_module_logger("CryptoHighFreqProvider")
        self.freq = freq
    
    def _init_qlib(self, qlib_conf):
        """initialize qlib"""

        qlib.init(
            region=REG_US,
            auto_mount=False,
            custom_ops=[DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut],
            expression_cache=None,
            **qlib_conf,
        )

    def _gen_dataframe(self, config, datasets=["train", "valid", "test"]):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e
        if os.path.isfile(path):
            start = time.time()
            self.logger.info(f"[{__name__}] Dataset exists, load from disk.")

            # res = dataset.prepare(['train', 'valid', 'test'])
            with open(path, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, dict):
                res = [data[i] for i in datasets]
            else:
                res = data.prepare(datasets)
            self.logger.info(f"[{__name__}] Data loaded, time cost: {time.time() - start:.2f}")
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}] Generating dataset")
            start_time = time.time()
            self._prepare_calender_cache()

            print("config: ", config)

            dataset = init_instance_by_config(config)

            trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
            data = {
                "train": trainset,
                "valid": validset,
                "test": testset,
            }
            with open(path, "wb") as f:
                pkl.dump(data, f)
            with open(path[:-4] + "train.pkl", "wb") as f:
                pkl.dump(trainset, f)
            with open(path[:-4] + "valid.pkl", "wb") as f:
                pkl.dump(validset, f)
            with open(path[:-4] + "test.pkl", "wb") as f:
                pkl.dump(testset, f)
            res = [data[i] for i in datasets]
            self.logger.info(f"[{__name__}]Data generated, time cost: {(time.time() - start_time):.2f}")
        return res

    def _gen_stock_dataset(self, config, conf_type):
        try:
            path = config.pop("path")
        except KeyError as e:
            raise ValueError("Must specify the path to save the dataset.") from e

        if os.path.isfile(path + "tmp_dataset.pkl"):
            start = time.time()
            self.logger.info(f"[{__name__}]Dataset exists, load from disk.")
        else:
            start = time.time()
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            self.logger.info(f"[{__name__}]Generating dataset")
            self._prepare_calender_cache()
            dataset = init_instance_by_config(config)
            self.logger.info(f"[{__name__}]Dataset init, time cost: {time.time() - start:.2f}")
            dataset.config(dump_all=False, recursive=True)
            dataset.to_pickle(path + "tmp_dataset.pkl")

        with open(path + "tmp_dataset.pkl", "rb") as f:
            new_dataset = pkl.load(f)

        instruments = D.instruments(market="all")
        stock_list = D.list_instruments(
            instruments=instruments, start_time=self.start_time, end_time=self.end_time, freq=self.freq, as_list=True
        )

        def generate_dataset(stock):
            if os.path.isfile(path + stock + ".pkl"):
                print("exist " + stock)
                return
            self._init_qlib(self.qlib_conf)
            new_dataset.handler.config(**{"instruments": [stock]})
            if conf_type == "backtest":
                new_dataset.handler.setup_data()
            else:
                new_dataset.handler.setup_data(init_type=DataHandlerLP.IT_LS)
            new_dataset.config(dump_all=True, recursive=True)
            new_dataset.to_pickle(path + stock + ".pkl")

            df_train, df_valid, df_test = new_dataset.prepare(segments=["train", "valid", "test"], data_key=DataHandlerLP.DK_L)
            print(df_train)
            #data = new_dataset.handler.fetch
            #print(data["train"])          

        Parallel(n_jobs=32)(delayed(generate_dataset)(stock) for stock in stock_list)    


