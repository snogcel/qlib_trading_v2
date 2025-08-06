from qlib.data.dataset.loader import QlibDataLoader
from typing import Tuple, Union, List, Dict

class gdelt_dataloader_optimized(QlibDataLoader):
    """
    Optimized GDELT Dataloader - Focused on essential market sentiment features
    
    Optimizations:
    - Kept core sentiment indicators (FG Index, BTC Dominance)
    - Maintained regime detection flags
    - Removed redundant derived features
    - Focused on validated predictive features
    """

    def __init__(
        self,
        config: Tuple[list, tuple, dict],
        filter_pipe: List = None,
        swap_level: bool = True,
        freq: Union[str, dict] = "day",
        inst_processors: Union[dict, list] = None,
        **kwargs
    ):
        
        self.filter_pipe = filter_pipe
        self.swap_level = swap_level
        self.freq = freq
        self.inst_processors = inst_processors if inst_processors is not None else {}
        
        assert isinstance(
            self.inst_processors, (dict, list)
        ), f"inst_processors(={self.inst_processors}) must be dict or list"

        _config = {
            "feature": self.get_feature_config(),
        }        
        if config is not None:
            _config.update(config)

        super().__init__(config=_config, freq=freq)

    @staticmethod
    def get_feature_config():
        """
        Optimized GDELT feature configuration
        """
        
        fields = [
            '$fg_index/100', 
            '$btc_dom/100', 
            'Std($fg_index, 7)', 
            'Std($btc_dom, 7)', 
            '($fg_index - Mean($fg_index, 14)) / Std($fg_index, 14)', 
            '($btc_dom - Mean($btc_dom, 14)) / Std($btc_dom, 14)', 
            #'If($fg_index > 80, 1, 0)', 
            #'If($fg_index < 20, 1, 0)', 
            #'If($btc_dom > 60, 1, 0)', 
            #'If($btc_dom < 40, 1, 0)'
        ]
        
        names = [
            '$fg_index', # $fg_index/100
            '$btc_dom', # $btc_dom/100
            'fg_std_7d', # Std($fg_index, 7)
            'btc_std_7d', # Std($btc_dom, 7)
            'fg_zscore_14d', # ($fg_index - Mean($fg_index, 14)) / Std($fg_index, 14)
            'btc_zscore_14d', # ($btc_dom - Mean($btc_dom, 14)) / Std($btc_dom, 14)
            #'fg_extreme_greed', # REPLACED WITH regime_features.py
            #'fg_extreme_fear', # REPLACED WITH regime_features.py
            #'btc_dom_high', # REPLACED WITH regime_features.py
            #'btc_dom_low' # REPLACED WITH regime_features.py
        ]
        
        return fields, names
