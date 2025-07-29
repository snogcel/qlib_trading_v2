import os, json
import numpy as np
import pandas as pd
from qlib.data import D
import qlib
from qlib.config import REG_US

# 1) Initialize Qlib so D.calendar() works
qlib.init(provider_uri="/Projects/qlib_trading_v2/qlib_data/CRYPTO", region=REG_US)

# 2) Load your daily calendar
cal = D.calendar(
    freq="day",
    start_time="2025-03-25",
    end_time="2025-04-10"
)

# 3) Read the quote_1d metadata (field names & order)
# meta_path = "/Projects/qlib_trading_v2/qlib_data/CRYPTO/quote_1d.json"
# with open(meta_path, "r") as f:
#     meta = json.load(f)
# fields = meta["fields"]  # e.g. ["open","high","low","close","..."]
fields = ["timestamp","open","high","low","close","volume"]

# 4) Memory‐map your ticker’s .bin file
ticker = "BTCUSDT"
bin_path = os.path.join("/Projects/qlib_trading_v2/qlib_data/CRYPTO/features/btcusdt/", "high.day.bin")
arr = np.memmap(bin_path, mode="r", dtype="float32", shape=(len(cal), len(fields)))

# 5) Wrap into a DataFrame
df = pd.DataFrame(arr, index=cal, columns=fields)
print("Last 10 closes from memmap:")
print(df["close"].tail(10))