
import abc
import pickle
from pathlib import Path
import warnings
import pandas as pd

from qlib.data.dataset.loader import DataLoader, QlibDataLoader
from qlib.data.dataset import loader as data_loader_module
from typing import Tuple, Union, List, Dict
from qlib.utils import init_instance_by_config

from qlib.utils import load_dataset, init_instance_by_config, time_to_slc_point

class CustomNestedDataLoader(DataLoader):
    """
    We have multiple DataLoader, we can use this class to combine them.
    """

    def __init__(self, dataloader_l: List[Dict], join="left") -> None:
        """

        Parameters
        ----------
        dataloader_l : list[dict]
            A list of dataloader, for exmaple

            .. code-block:: python

                nd = NestedDataLoader(
                    dataloader_l=[
                        {
                            "class": "qlib.contrib.data.loader.Alpha158DL",
                        }, {
                            "class": "qlib.contrib.data.loader.Alpha360DL",
                            "kwargs": {
                                "config": {
                                    "label": ( ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"])
                                }
                            }
                        }
                    ]
                )
        join :
            it will pass to pd.concat when merging it.
        """
        
        super().__init__()
        self.data_loader_l = [
            (dl if isinstance(dl, DataLoader) else init_instance_by_config(dl)) for dl in dataloader_l
        ]
        self.join = join

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df_full = None
        for dl in self.data_loader_l:
            try:
                df_current = dl.load(instruments, start_time, end_time)
            except KeyError:
                warnings.warn(
                    "If the value of `instruments` cannot be processed, it will set instruments to None to get all the data."
                )
                df_current = dl.load(instruments=None, start_time=start_time, end_time=end_time)

                ## use this when working with micro
                # df_macro = self.data_loader_l[1].load(instruments=["BTCUSDT"], start_time=start_time, end_time=end_time) # loading macro_feature.pkl
                # df_macro.columns = pd.MultiIndex.from_product([['macro'], df_macro.columns])
                # df_current = df_macro.copy()

                ## use this when working with macro -- fugly
                # df_current = self.data_loader_l[1].load(instruments=["BTC_FEAT"], start_time=start_time, end_time=end_time)

            if df_full is None:
                df_full = df_current
            else:
                current_columns = df_current.columns.tolist()
                full_columns = df_full.columns.tolist()    

                columns_to_drop = [col for col in current_columns if col in full_columns]
                df_full.drop(columns=columns_to_drop, inplace=True)

                df_current = df_current.sort_index(axis=1).reset_index(level="instrument", drop=True) # BTC_FEAT, not specific to any crypto                
                df_full = df_full.sort_index(axis=1).reset_index(level="instrument", drop=False)                

                ### previously used for merging day:day timeframes, adjusted to merge_asof to allow for hour:day merge
                # df_merged = pd.merge(left=df_full, right=df_current, left_index=True, right_index=True, how=self.join)

                df_merged = pd.merge_asof(left=df_full, right=df_current, on="datetime")
                df_merged.reset_index(drop=True, inplace=True)               
                df_merged.set_index(['instrument', 'datetime'], inplace=True)

                print("df_merged: ", df_merged)
                # df_merged.to_csv("df_merged.csv")
                
        return df_merged.sort_index(axis=1)



class OLD_CustomNestedDataLoader(QlibDataLoader):
    """
    We have multiple DataLoader, we can use this class to combine them.
    """
    def __init__(self, instruments, start_time, end_time, dataloader_l: List[Dict], join="left"):       
        """
        Parameters
        ----------
        instruments :
            The stock list to retrieve.
        start_time :
            start_time of the original data.
        end_time :
            end_time of the original data.
        data_loader : Union[dict, str, DataLoader]
            data loader to load the data.
        init_data :
            initialize the original data in the constructor.
        fetch_orig : bool
            Return the original data instead of copy if possible.
        """

        print("--DATA LOADER CONFIG--")
        print(dataloader_l)

        self.data_loader_l = [
            (dl if isinstance(dl, DataLoader) else init_instance_by_config(config=dl)) for dl in dataloader_l
        ]

        #self.dataloader_1[0] = init_instance_by_config(config=dataloader_l[0], try_kwargs={"start_time": start_time, "end_time": end_time})
        #self.dataloader_1[1] = init_instance_by_config(config=dataloader_l[1], try_kwargs={"start_time": start_time, "end_time": end_time, "freq": "day"})

        
        # if isinstance(config_1, DataLoader):
        #     loader_1 = config_1
        # else:
        #     loader_1 = init_instance_by_config(config_1)
        
        # if isinstance(config_2, DataLoader):
        #     loader_2 = config_2
        # else:
        #     loader_2 = init_instance_by_config(config_2, try_kwargs={"instruments": ["BTC_FEAT"], "start_time": start_time, "end_time": end_time})        
        
        #self.dataloader_1 = [loader_1, loader_2]

        # self.dataloader_l[0] if isinstance(dataloader_l[0], DataLoader) else init_instance_by_config(dataloader_l[0], "instruments": ["BTCUSDT"], "start_time": start_time, "end_time": end_time),
        # self.dataloader_l[1] if isinstance(dataloader_l[1], DataLoader) else init_instance_by_config(dataloader_l[1], "instruments": ["BTC_FEAT"], "start_time": start_time, "end_time": end_time, "freq": "day"),
    
        self.join = join

        print(self.data_loader_l)

        #super().__init__()

        
        #self.data_loader_l = [
        #    (dl if isinstance(dl, DataLoader) else init_instance_by_config(dl)) for dl in dataloader_l
        #]
        

    def set_level_values(midx, level, values):
        """
        Replace pandas df multiindex level values with an iterable of values of the same length.

        Does allow duplicate values, which set_level_values method does not.
        
        Parameters
        ----------
        midx: pd.Multiindex
            Multilevel index or columns of pandas dataframe to change level in.
        level: str
            Name of level to change
        values: iterable
            Values to replace the original level values.
            
        Returns: pd.Multiindes
            The multivel index/columns with replaced values in given level.
        """
        full_levels = list(zip(*midx.values))
        names = midx.names
        if isinstance(level, str):
            if level not in names:
                raise ValueError(f'No level {level} in MultiIndex')
            level = names.index(level)
        if len(full_levels[level]) != len(values):
            raise ValueError('Values must be of the same size as original level')
        full_levels[level] = values
        return pd.MultiIndex.from_arrays(full_levels, names=names)

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        df_full = self.data_loader_l[0].load(instruments=["BTCUSDT"], start_time=start_time, end_time=end_time)        
        
        ## use this when working with micro
        df_macro = self.data_loader_l[1].load(instruments=["BTCUSDT"], start_time=start_time, end_time=end_time) # loading macro_feature.pkl
        df_macro.columns = pd.MultiIndex.from_product([['macro'], df_macro.columns])
        df_current = df_macro.copy()

        ## use this when working with macro -- fugly
        # df_current = self.data_loader_l[1].load(instruments=["BTC_FEAT"], start_time=start_time, end_time=end_time)



        #df.sort_index(axis=1).drop('Col1', axis=1)

        #df_current = df_current.sort_index(axis=1).reset_index()
        #df_full = df_full.sort_index(axis=1).reset_index()

        #df_current.index = df_macro.index.swaplevel("datetime", "instrument")
        #df_current.sort_index(inplace=True)


        
        
        
        print(df_full.index)
        print(df_full.columns)

        
        print(df_current.index)
        print(df_current.columns)


        print(df_full)
        print(df_current)


        df_current = df_current.sort_index(axis=1).reset_index()
        df_current = df_current.drop(columns=['instrument'])
        df_current = df_current.set_index('datetime')

        df_full = df_full.sort_index(axis=1).reset_index()
        df_full['date'] = pd.to_datetime(df_full['datetime']).dt.date
        df_full = df_full.set_index('date')


        df_merged = pd.merge(left=df_full, right=df_current, left_index=True, right_index=True, how='left')

        df_merged.reset_index(inplace=True, drop=True)
        
        df_merged = df_merged.set_index(['instrument', 'datetime'])

        #s = df_merged.pop(('label','LABEL0'))
        #df_final = pd.concat([df_merged, s], axis=1)
        
        df_final = df_merged.copy()
        
        print("df_final: ", df_final)
        df_final.to_csv("df_final.csv")

        return df_final.sort_index(axis=1)