import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C://Projects//qlib_trading_v2//csv_data//BTC_FEATS//btc_supplement_feat_trim.csv")
df['datetime'] = pd.to_datetime(df['datetime'])  

# Define quantile weighting configurations
weight_configs = [
    {'Q90': 0.7, 'Q50': 0.2, 'Q10': 0.1},  # Heavy Q90
    {'Q90': 0.5, 'Q50': 0.3, 'Q10': 0.2},  # Balanced approach
    {'Q90': 0.3, 'Q50': 0.4, 'Q10': 0.3},  # More mid-range influence
]

# Compute weighted sentiment score for each setup
for config in weight_configs:
    df[f"weighted_sentiment_{config['Q90']}_{config['Q50']}_{config['Q10']}"] = (
        config['Q90'] * df['Q90'] + config['Q50'] * df['Q50'] + config['Q10'] * df['Q10']
    )

# Visualize different configurations
plt.figure(figsize=(12, 6))
for config in weight_configs:
    plt.plot(df['datetime'], df[f"weighted_sentiment_{config['Q90']}_{config['Q50']}_{config['Q10']}"], label=f"Q90-{config['Q90']}, Q50-{config['Q50']}, Q10-{config['Q10']}")

plt.xlabel("Date")
plt.ylabel("Weighted Sentiment Score")
plt.title("Quantile Weighting Experimentation")
plt.legend()
plt.grid(True)
plt.show()