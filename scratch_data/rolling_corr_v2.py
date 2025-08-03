from scipy.signal import correlate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Load your data
df = pd.read_csv("C://Projects//qlib_trading_v2//csv_data//BTC_FEATS//btc_supplement_feat_trim.csv", index_col='datetime', parse_dates=True)

# Convert datetime column to actual date format
# df['datetime'] = pd.to_datetime(df['datetime'])  # Simple conversion
# df.set_index('datetime')

x = df['btc_dom'].dropna()
y = df['fg_index'].dropna()

x.index = df.index
y.index = df.index

lags = np.arange(-15, 16)  # Test shifts up to +/- 30 days

# corr_values = [np.corrcoef(x.shift(lag).dropna(), y.dropna())[0, 1] for lag in lags]
""" corr_values = [
    np.corrcoef(*x.shift(lag).align(y, join="inner"))[0, 1] for lag in lags
] """

corr_values = []
for lag in lags:
    aligned_x, aligned_y = x.shift(lag), y  # Shift x but keep y unchanged
    valid_indices = aligned_x.dropna().index.intersection(y.dropna().index)  # Ensure overlap
    if len(valid_indices) > 0:  # Avoid empty correlations
        corr_values.append(np.corrcoef(aligned_x.loc[valid_indices], y.loc[valid_indices])[0, 1])
    else:
        corr_values.append(np.nan)  # Fill gaps with NaN

""" corr_values = []

for lag in lags:
    aligned_x, aligned_y = x.shift(lag).align(y, join="inner")  # Ensure matching size
    if len(aligned_x) > 0 and len(aligned_y) > 0:  # Prevent empty arrays
        corr_values.append(np.corrcoef(aligned_x, aligned_y)[0, 1])  # Extract correlation
    else:
        corr_values.append(np.nan)  # Fill gaps with NaN if alignment fails """

print(corr_values)

plt.plot(lags, corr_values, marker='o')
plt.axvline(x=0, linestyle='--', color='gray')
plt.xlabel("Lag (Days)")
plt.ylabel("Correlation Coefficient")
plt.title("BTC Dominance vs. FG Index Lag Test")
plt.show()
