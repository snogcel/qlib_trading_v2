import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("C://Projects//qlib_trading_v2//csv_data//BTC_FEATS//btc_supplement_feat.csv")

# Convert datetime column to actual date format
df['datetime'] = pd.to_datetime(df['datetime'])  # Simple conversion

# Define rolling correlation windows
windows = [30, 90, 180]

df['fg_index_lag1'] = df['fg_index'].shift(1)  # 1-day lag
df['fg_index_lead1'] = df['fg_index'].shift(-1)  # 1-day lead

# Compute rolling correlations
for window in windows:
    df[f'corr_{window}d'] = df['btc_dom'].rolling(window).corr(df['fg_index_lead1'])

# Plot rolling correlations
plt.figure(figsize=(12, 6))
for window in windows:
    plt.plot(df['datetime'], df[f'corr_{window}d'], label=f"{window}-day correlation")

plt.axhline(y=0, linestyle='--', color='gray')  # Reference line for neutrality
plt.xlabel("Date")
plt.ylabel("Correlation (BTC Dominance vs FG Index)")
plt.title("Rolling Correlation Analysis")
plt.legend()
plt.grid(True)
plt.show()