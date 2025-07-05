# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd

from pathlib import Path

#DATA_PATH = Path(os.path.join("..", "data", "pickle", "backtest"))
#OUTPUT_PATH = Path(os.path.join("..", "data", "orders"))

MACRO_FEAT_PATH = ("/Projects/qlib_trading_v2/data/macro_features.pkl")
DATA_PATH = Path("/Projects/qlib_trading_v2/data2/pickle/backtest")
OUTPUT_PATH = Path("/Projects/qlib_trading_v2/data2/orders")

def generate_order(stock: str, start_idx: int, end_idx: int) -> bool:
    dataset = pd.read_pickle(DATA_PATH / f"{stock}.pkl")
    feat = pd.read_pickle(MACRO_FEAT_PATH)

    df = dataset.handler.fetch(level=None).reset_index()
    feat_df = feat.reset_index()
    
    if len(df) == 0 or df.isnull().values.any():        
        return False

    df["date"] = df["datetime"].dt.date.astype("datetime64[ns]")
    feat_df["date"] = feat_df["datetime"].dt.date.astype("datetime64[ns]")
    feat_df = feat_df.drop("datetime", axis=1)  

    # Convert tier label to numeric
    TIER_MAP = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}

    if "signal_tier" in feat_df:        
        feat_df["signal_tier"] = feat_df["signal_tier"].map(TIER_MAP)        

    df = pd.merge(left=df, right=feat_df, on=["date","instrument"], how="left").dropna()
    df = df.set_index(["instrument", "datetime", "date"])
    df = df.dropna()

    required_length = end_idx - start_idx

    # Count rows per date
    counts = df.groupby("date").size()

    # Filter dates with insufficient data
    incomplete_dates = counts[counts < required_length].index.tolist()

    print(f"Incomplete dates (expected ≥ {required_length} rows):")
    print(incomplete_dates) 

    valid_dates = [d for d, g in df.groupby("date") if len(g) >= (end_idx - start_idx)]
    df = df[df.index.get_level_values("date").isin(valid_dates)]
    df = df.groupby("date", group_keys=False).take(range(start_idx, end_idx))

    order_all = pd.DataFrame(df.groupby(level=(2, 0), group_keys=False).mean().dropna())
    order_all["amount"] = np.random.lognormal(-3.28, 1.14) * order_all["$volume0"]
    order_all = order_all[order_all["amount"] > 0.0]
    order_all["order_type"] = 0
    order_all = order_all.drop(columns=["$volume0"])    

    order_train = order_all[order_all.index.get_level_values(0) <= pd.Timestamp("2023-12-31")]
    order_test = order_all[order_all.index.get_level_values(0) > pd.Timestamp("2024-01-01")]
    order_valid = order_test[order_test.index.get_level_values(0) <= pd.Timestamp("2024-10-01")]
    order_test = order_test[order_test.index.get_level_values(0) > pd.Timestamp("2024-10-01")]

    for order, tag in zip((order_train, order_valid, order_test, order_all), ("train", "valid", "test", "all")):
        path = OUTPUT_PATH / tag
        os.makedirs(path, exist_ok=True)
        if len(order) > 0:
            order.to_pickle(path / f"{stock}.pkl.target")            
    return True


np.random.seed(1234)
file_list = sorted(os.listdir(DATA_PATH))
stocks = [f.replace(".pkl", "") for f in file_list]
np.random.shuffle(stocks)

# override selection of all cryptos
stocks = ['BTCUSDT']
print(stocks)

cnt = 0
for stock in stocks:
    if generate_order(stock, 0, 1440 // 60):
        cnt += 1
        if cnt == 100:
            break
