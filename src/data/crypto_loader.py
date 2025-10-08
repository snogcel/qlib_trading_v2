from qlib.data.dataset.loader import QlibDataLoader
from typing import Tuple, Union, List, Dict

class crypto_dataloader_optimized(QlibDataLoader):
    """
    Optimized Crypto Dataloader - Removes redundant features based on correlation analysis
    
    Optimizations:
    - Removed highly correlated volatility measures ($realized_vol_3, $realized_vol_9)
    - Kept only OPEN1, VOLUME1 (removed OPEN2/3, VOLUME2/3)
    - Added essential regime detection features
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
            "label": kwargs.pop("label", self.get_label_config()),          
        }

        if config is not None:
            _config.update(config)

        super().__init__(config=_config, freq=freq)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},  # Use optimized kbar features
            "price": {
                "windows": [1],  # Only OPEN1 (OPEN2/3 removed due to high correlation)
                "feature": ['OPEN'],
            },
            "Volume_USDT": {
                "windows": [1],  # Only VOLUME1 (VOLUME2/3 removed due to high correlation)
            },
            "rolling": {
                "windows": [1, 2, 3],  # Reduced window set
                "include": ['RSV'],  # Keep only most valuable rolling features
            },
        }
    ):
        """
        Optimized feature configuration based on correlation analysis
        """
        fields = []
        names = []
        
        if "kbar" in config:
            # Optimized kbar features (removed redundant volatility measures)
            fields += [
                
                # realized_vol_6
                'Std(Log($close / Ref($close, 1)), 6)',
                
                # ── Relative volatility (short/long)
                'Std(Log($close / Ref($close, 1)), 5) / Std(Log($close / Ref($close, 1)), 20)', 

                # ── Rolling price momentum for directional context
                '$close / Ref($close, 5) - 1', # 5 hours
                '$close / Ref($close, 10) - 1', # 10 hours
                '$close / Ref($close, 25) - 1', # 25 hours

                # ── High-volatility regime flag
                "If(Std(Log($close / Ref($close, 1)), 5) / Std(Log($close / Ref($close, 1)), 20) > 1.25, 1, 0)", 
                
                #'If(Std(Log($close / Ref($close, 1)), 6) > Quantile(Std(Log($close / Ref($close, 1)), 6), 180, 0.9), 1, 0)', 
                #'If(Std(Log($close / Ref($close, 1)), 6) > Quantile(Std(Log($close / Ref($close, 1)), 6), 180, 0.7), 1, 0)', 
                #'If(Std(Log($close / Ref($close, 1)), 6) < Quantile(Std(Log($close / Ref($close, 1)), 6), 180, 0.3), 1, 0)', 
                #'If(Std(Log($close / Ref($close, 1)), 6) < Quantile(Std(Log($close / Ref($close, 1)), 6), 180, 0.1), 1, 0)', 
                
                'Std(Log($close / Ref($close, 1)), 6)',
                'Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10',
                'Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)',
                '(Std(Log($close / Ref($close, 1)), 6) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01)) / (Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.99) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01))',
                'Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)',
                '(Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)) * 2000'
            ]

            names += [
                
                # realized_vol_6
                '$realized_vol_6', 

                # ── Relative volatility (short/long)
                '$relative_volatility_index', 

                # ── Rolling price momentum for directional context
                '$momentum_5', 
                '$momentum_10', 
                '$momentum_25',

                # ── High-volatility regime flag
                '$high_vol_flag', 
                
                # ── replaced by Regime
                #'vol_extreme_high', # "If(Std(Log($close / Ref($close, 1)), 6) > Quantile(Std(Log($close / Ref($close, 1)), 6), 168, 0.9), 1, 0)", 
                #'vol_high', # "If(Std(Log($close / Ref($close, 1)), 6) > Quantile(Std(Log($close / Ref($close, 1)), 6), 168, 0.7), 1, 0)", 
                #'vol_low', # "If(Std(Log($close / Ref($close, 1)), 6) < Quantile(Std(Log($close / Ref($close, 1)), 6), 168, 0.3), 1, 0)", 
                #'vol_extreme_low', # "If(Std(Log($close / Ref($close, 1)), 6) < Quantile(Std(Log($close / Ref($close, 1)), 6), 168, 0.1), 1, 0)", 
                
                'vol_raw', # "Std(Log($close / Ref($close, 1)), 6)"
                'vol_raw_decile', # "Rank(Std(Log($close / Ref($close, 1)), 6), 180) / 180 * 10"
                'vol_risk', # "Std(Log($close / Ref($close, 1)), 6) * Std(Log($close / Ref($close, 1)), 6)"
                'vol_scaled', # "(Std(Log($close / Ref($close, 1)), 6) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01)) / (Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.99) - Quantile(Std(Log($close / Ref($close, 1)), 6), 30, 0.01))",
                'vol_raw_momentum', # "Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)"                 
                'vol_momentum_scaled' # "(Std(Log($close / Ref($close, 1)), 6) - Ref(Std(Log($close / Ref($close, 1)), 6), 1)) * 2000", 
            ]
            
        if "price" in config:
            windows = config["price"].get("windows", [1])
            feature = config["price"].get("feature", ["OPEN"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
                
        if "volume" in config:
            windows = config["volume"].get("windows", [1])
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
            
        if "rolling" in config:
            windows = config["rolling"].get("windows", [1, 2, 3])
            include = config["rolling"].get("include", ['RSV'])
            
            if 'RSV' in include:
                # Keep only RSV (most valuable rolling feature)
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]

        return fields, names

    @staticmethod
    def get_label_config():        
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
