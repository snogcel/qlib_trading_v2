# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from qlib.backtest.exchange import Exchange

class FeatureAwareExchange(Exchange):
    """
    An exchange that is aware of custom features.
    It loads a pre-computed feature dataframe and merges
    it with the standard observations.
    """

    def __init__(self, feature_df_path="df_cleaned.csv", **kwargs):
        super().__init__(**kwargs)
        self.feature_df = pd.read_csv(feature_df_path, index_col=0, parse_dates=True)

    def get_obs(self, trade_step: int, trade_len: int, trade_order: "Order") -> dict:
        """
        Get observation from the exchange.

        The observation should include the features required by the policy.
        """
        # Get standard observations from the base class
        obs = super().get_obs(trade_step, trade_len, trade_order)

        # Get the current timestamp
        current_time = self.get_trade_calendar().get_loc(self.trade_step)

        # Get the features for the current timestamp
        current_features = self.feature_df.loc[current_time]

        # Combine the standard observations with the custom features
        obs.update(current_features.to_dict())

        return obs
