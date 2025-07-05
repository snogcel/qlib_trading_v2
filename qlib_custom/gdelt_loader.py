from qlib.data.dataset.loader import QlibDataLoader
from typing import Tuple, Union, List, Dict

class gdelt_dataloader(QlibDataLoader):
    """Dataloader to return standalone GDELT Data"""

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

        # sample
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
        #fields = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_geopolitical_risk','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$cwt_social','$rawvol_back_to_the_70s','$rawvol_environment','$rawvol_geopolitical_risk','$rawvol_monetary','$rawvol_roaring_20s','$rawvol_secular_stagnation','$rawvol_social']
        #names = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_geopolitical_risk','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$cwt_social','$rawvol_back_to_the_70s','$rawvol_environment','$rawvol_geopolitical_risk','$rawvol_monetary','$rawvol_roaring_20s','$rawvol_secular_stagnation','$rawvol_social']
        
        #fields = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$rawvol_back_to_the_70s','$rawvol_roaring_20s','$rawvol_secular_stagnation','$rawvol_social']
        #names = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$rawvol_back_to_the_70s','$rawvol_roaring_20s','$rawvol_secular_stagnation','$rawvol_social']
        
        # fields = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_geopolitical_risk','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$cwt_social']
        # names = ['$cwt_back_to_the_70s','$cwt_environment','$cwt_geopolitical_risk','$cwt_monetary','$cwt_roaring_20s','$cwt_secular_stagnation','$cwt_social']

        # Using GDelt Features
        # RMSE: 0.02189432979420407
        # R2:  0.25280509520164096

        """ fields = [
            "$cwt_roaring_20s",
            "$cwt_monetary",
            "$cwt_back_to_the_70s",
            "$cwt_secular_stagnation",
            "$cwt_environment",
            "$cwt_geopolitical_risk",
            "$cwt_social",

            # Rolling z-score (window = 10)
            "($cwt_roaring_20s - Mean($cwt_roaring_20s, 10)) / Std($cwt_roaring_20s, 10)",
            "($cwt_monetary - Mean($cwt_monetary, 10)) / Std($cwt_monetary, 10)",
            "($cwt_back_to_the_70s - Mean($cwt_back_to_the_70s, 10)) / Std($cwt_back_to_the_70s, 10)",
            "($cwt_secular_stagnation - Mean($cwt_secular_stagnation, 10)) / Std($cwt_secular_stagnation, 10)",
            "($cwt_environment - Mean($cwt_environment, 10)) / Std($cwt_environment, 10)",
            "($cwt_geopolitical_risk - Mean($cwt_geopolitical_risk, 10)) / Std($cwt_geopolitical_risk, 10)",
            "($cwt_social - Mean($cwt_social, 10)) / Std($cwt_social, 10)",

            # CWT momentum (1-day delta)
            "$cwt_monetary - Ref($cwt_monetary, -1)",
            "$cwt_roaring_20s - Ref($cwt_roaring_20s, -1)",
            "$cwt_secular_stagnation - Ref($cwt_secular_stagnation, -1)",
            "$cwt_back_to_the_70s - Ref($cwt_back_to_the_70s, -1)",
            "$cwt_geopolitical_risk - Ref($cwt_geopolitical_risk, -1)",
            "$cwt_environment - Ref($cwt_environment, -1)",
            "$cwt_social - Ref($cwt_social, -1)",

            # ── Narrative turbulence (7-theme average change in CWT)
            "(Abs($cwt_monetary - Ref($cwt_monetary, -1)) + Abs($cwt_roaring_20s - Ref($cwt_roaring_20s, -1)) + Abs($cwt_secular_stagnation - Ref($cwt_secular_stagnation, -1)) + Abs($cwt_back_to_the_70s - Ref($cwt_back_to_the_70s, -1)) + Abs($cwt_geopolitical_risk - Ref($cwt_geopolitical_risk, -1)) + Abs($cwt_environment - Ref($cwt_environment, -1)) + Abs($cwt_social - Ref($cwt_social, -1))) / 7",

            # (Optional) binary anomaly flag — replace threshold dynamically if needed
            "If((Abs($cwt_monetary - Ref($cwt_monetary, -1)) + Abs($cwt_roaring_20s - Ref($cwt_roaring_20s, -1)) + Abs($cwt_secular_stagnation - Ref($cwt_secular_stagnation, -1)) + Abs($cwt_back_to_the_70s - Ref($cwt_back_to_the_70s, -1)) + Abs($cwt_geopolitical_risk - Ref($cwt_geopolitical_risk, -1)) + Abs($cwt_environment - Ref($cwt_environment, -1)) + Abs($cwt_social - Ref($cwt_social, -1))) / 7 > 0.05, 1, 0)",

            # Additional...
            # "Std([$cwt_monetary, $cwt_roaring_20s, $cwt_secular_stagnation, $cwt_back_to_the_70s, $cwt_geopolitical_risk, $cwt_environment, $cwt_social], 5)"

        ]

        names = [
            "$cwt_roaring_20s",
            "$cwt_monetary",
            "$cwt_back_to_the_70s",
            "$cwt_secular_stagnation",
            "$cwt_environment",
            "$cwt_geopolitical_risk",
            "$cwt_social",

            "$z_cwt_roaring_20s",
            "$z_cwt_monetary",
            "$z_cwt_back_to_the_70s",
            "$z_cwt_secular_stagnation",
            "$z_cwt_environment",
            "$z_cwt_geopolitical_risk",
            "$z_cwt_social",

            "$delta_cwt_monetary",
            "$delta_cwt_roaring_20s",
            "$delta_cwt_secular_stagnation",
            "$delta_cwt_back_to_the_70s",
            "$delta_cwt_geopolitical_risk",
            "$delta_cwt_environment",
            "$delta_cwt_social",

            "$cni",

            "$narrative_anomaly_flag",

        ] """

        # RMSE: 0.02188817698226591
        # R2:  0.2532249942159427
        
        print("NRAH!")

        fields = [
            "$fg_index",
            "$btc_dom",
            "$fg_index - Ref($fg_index, -3)",
            "$btc_dom - Ref($btc_dom, -3)",
            "Std($fg_index, 7)",
            "Std($btc_dom, 7)",
            "($fg_index - Mean($fg_index, 14)) / Std($fg_index, 14)",
            "($btc_dom - Mean($btc_dom, 14)) / Std($btc_dom, 14)",
            "If($fg_index > 80, 1, 0)",
            "If($fg_index < 20, 1, 0)",
            "If($btc_dom > 60, 1, 0)",
            "If($btc_dom < 40, 1, 0)",
            "$fg_index * $btc_dom",
            "$fg_index - $btc_dom",
            "Corr($fg_index, $btc_dom, 14)",
            "Mean($fg_index, 3) - Mean($fg_index, 7)",
            "Mean($btc_dom, 3) - Mean($btc_dom, 7)"
        ]

        names = [
            "fg_index",                                                 # ← original
            "btc_dom",                                                  # ← original
            "fg_momentum_3d",
            "btc_momentum_3d",
            "fg_std_7d",
            "btc_std_7d",
            "fg_zscore_14d",
            "btc_zscore_14d",
            "fg_extreme_greed",
            "fg_extreme_fear",
            "btc_dom_high",
            "btc_dom_low",
            "fg_btc_product",
            "fg_btc_spread",
            "fg_btc_corr_14d",
            "fg_ema_diff",
            "btc_ema_diff"
        ]


        return fields, names
    