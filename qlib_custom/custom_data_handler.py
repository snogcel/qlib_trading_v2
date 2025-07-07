from typing import Callable, Union, Tuple, List, Iterator, Optional

import qlib
from qlib.utils import init_instance_by_config
from qlib.typehint import Literal
from qlib.data.dataset.handler import DataHandlerLP, DataHandler
from qlib.data.dataset.loader import DataLoader
from qlib.data.dataset import processor as processor_module
from qlib.contrib.data.handler import check_transform_proc
from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader

import pandas as pd

DATA_KEY_TYPE = Literal["raw", "infer", "learn"]

class CryptoHighFreqGeneralBacktestHandler(DataHandler):
    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        day_length=1440,
        freq="60min",
        columns=["$close", "$vwap", "$volume"],
        inst_processors=None,
    ):
        self.day_length = day_length
        self.columns = set(columns)        
        data_loader = {
            "class": "CustomNestedDataLoader",
            "module_path": "qlib_custom.custom_ndl",
            "kwargs": {
                "dataloader_l": [                    
                    {
                        "class": "QlibDataLoader",  
                        "module_path": "qlib.data.dataset.loader",                                                                      
                        "kwargs": {
                            "freq": freq,
                            "config": {
                                "feature": self.get_feature_config(),
                            },                        
                            "swap_level": False,                            
                            "inst_processors": inst_processors,                            
                        }
                    },
                    {
                        "class": "gdelt_dataloader",
                        "module_path": "qlib_custom.gdelt_loader",                        
                        "kwargs": {
                            "freq": "day",  # Replace with your FREQ variable
                            "config": {
                                "feature": gdelt_dataloader.get_feature_config(),                                
                            },                             
                            "swap_level": False,                            
                            "inst_processors": [],                            
                        }
                    }
                ],                
                "instruments": ["BTCUSDT", "BTC_FEAT"],
                "start_time": "20180201",
                "end_time": "20250401",                
            },            
            "join": "left",                                                          
        }
        dl = init_instance_by_config(data_loader)
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=dl,
        )

    def get_feature_config(self):
        fields = []
        names = []

        if "$close" in self.columns:
            template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"
            template_fillnan = "FFillNan({0})"
            template_if = "If(IsNull({1}), {0}, {1})"
            fields += [
                template_paused.format(template_fillnan.format("$close")),
            ]
            names += ["$close0"]

        if "$vwap" in self.columns:
            fields += [
                template_paused.format(template_if.format(template_fillnan.format("$close"), "$vwap")),
            ]
            names += ["$vwap0"]

        if "$volume" in self.columns:
            fields += [template_paused.format("If(IsNull({0}), 0, {0})".format("$volume"))]
            names += ["$volume0"]

        return fields, names


class CryptoHighFreqGeneralHandler(DataHandlerLP):
    def __init__(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
        data_loader: Union[dict, str, DataLoader] = None,
        infer_processors: List = [],
        learn_processors: List = [],   
        shared_processors: List = [],
        fit_start_time=None,
        fit_end_time=None,
        process_type="append",
        drop_raw=True,
        day_length=1440,
        freq="1min",
        columns=["$open", "$high", "$low", "$close", "$vwap"],
        inst_processors=None,
        **kwargs
    ):
        self.day_length = day_length
        self.columns = columns

        # infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        # learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # only_crypto = NameDFilter(name_rule_re='BTC_FEAT')
        # only_feats = NameDFilter(name_rule_re='BTCUSDT')

        data_loader = {
            "class": "CustomNestedDataLoader",
            "module_path": "qlib_custom.custom_ndl",
            "kwargs": {
                "dataloader_l": [                    
                    {
                        "class": "QlibDataLoader",  
                        "module_path": "qlib.data.dataset.loader",                                                                      
                        "kwargs": {
                            "freq": freq,
                            "config": {
                                "feature": self.get_feature_config(),
                            },                        
                            "swap_level": False,                            
                            "inst_processors": inst_processors,                            
                        }
                    },
                    {
                        "class": "gdelt_dataloader",
                        "module_path": "qlib_custom.gdelt_loader",                        
                        "kwargs": {
                            "freq": "day",  # Replace with your FREQ variable
                            "config": {
                                "feature": gdelt_dataloader.get_feature_config(),                                
                            },                             
                            "swap_level": False,                            
                            "inst_processors": [],                            
                        }
                    }
                ],                
                "instruments": ["BTCUSDT", "BTC_FEAT"],
                "start_time": "20180201",
                "end_time": "20250401",                
            },            
            "join": "left",                                                          
        }

        dl = init_instance_by_config(data_loader)

        print("nested_data_loader: ", dl)

        # Setup preprocessor
        self.infer_processors = []  # for lint
        self.learn_processors = []  # for lint
        self.shared_processors = []  # for lint
        for pname in "infer_processors", "learn_processors", "shared_processors":
            for proc in locals()[pname]:
                getattr(self, pname).append(
                    init_instance_by_config(
                        proc,
                        None if (isinstance(proc, dict) and "module_path" in proc) else processor_module,
                        accept_types=processor_module.Processor,
                    )
                )

        self.process_type = process_type
        self.drop_raw = drop_raw
        super().__init__(instruments, start_time, end_time, dl, **kwargs)

    def get_feature_config(self):
        fields = []
        names = []

        template_if = "If(IsNull({1}), {0}, {1})"
        template_paused = f"Cut({{0}}, {self.day_length * 2}, None)"

        def get_normalized_price_feature(price_field, shift=0):
            # norm with the close price of 237th minute of yesterday.
            if shift == 0:
                template_norm = f"{{0}}/DayLast(Ref({{1}}, {self.day_length * 2}))"
            else:
                template_norm = f"Ref({{0}}, " + str(shift) + f")/DayLast(Ref({{1}}, {self.day_length}))"

            template_fillnan = "FFillNan({0})"
            # calculate -> ffill -> remove paused
            feature_ops = template_paused.format(
                template_fillnan.format(
                    template_norm.format(template_if.format("$close", price_field), template_fillnan.format("$close"))
                )
            )
            return feature_ops

        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, 0))
            names.append(column_name)

        for column_name in self.columns:
            fields.append(get_normalized_price_feature(column_name, self.day_length))
            names.append(column_name + "_1")

        # calculate and fill nan with 0
        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format(
                    f"{{0}}/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format("$volume")
                )
            )
        ]
        names += ["$volume"]

        fields += [
            template_paused.format(
                "If(IsNull({0}), 0, {0})".format(
                    f"Ref({{0}}, {self.day_length})/Ref(DayLast(Mean({{0}}, {self.day_length * 30})), {self.day_length})".format(
                        "$volume"
                    )
                )
            )
        ]
        names += ["$volume_1"]

        return fields, names
