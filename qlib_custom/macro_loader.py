import pandas as pd
from qlib.data.dataset.loader import StaticDataLoader

class MacroFeatureLoader(StaticDataLoader):
    def __init__(
            self, 
            pickle_path, 
            freq="day", 
            **kwargs
    ):
        print("MACRO FEAT pickle_path: ", pickle_path)
        super().__init__(**kwargs)
        df = pd.read_pickle(pickle_path)

        #TIER_MAP = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
        #df["signal_tier"] = df["signal_tier"].map(TIER_MAP)

        self.data = df.copy()        
        print(self.data)

    def load(self, instruments, start_time, end_time):
        df = self.data.copy()
        df = df.loc[(df.index.get_level_values("datetime") >= pd.Timestamp(start_time)) &
                    (df.index.get_level_values("datetime") <= pd.Timestamp(end_time))]
        return df
