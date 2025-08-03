import os
import pandas as pd

# === Configuration ===
input_folder = './csv_data/CRYPTODATA'  # Folder with your original 1-minute CSVs
output_base = './csv_data/CRYPTODATA_RESAMPLE'  # Base folder to hold all output folders
resample_configs = {
    '60min': '60T',
    '240min': '240T',
    '1d': '1D'
}

# === Create output folders ===
for folder in resample_configs.keys():
    os.makedirs(os.path.join(output_base, folder), exist_ok=True)

# === Resampling function ===
def resample_crypto_data(file_path, output_base, resample_configs):
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    dtype = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32', 'volume': 'float32'}
    df = pd.read_csv(file_path, usecols=cols, parse_dates=['timestamp'], dtype=dtype)
    
    # Parse timestamp column
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Ensure columns are correct
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{file_path} is missing required columns.")

    # Resample and save
    for label, rule in resample_configs.items():
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        output_path = os.path.join(output_base, label, os.path.basename(file_path))
        resampled.to_csv(output_path)
        print(f"Saved {output_path}")

# === Iterate through all CSV files ===
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        full_path = os.path.join(input_folder, filename)
        try:
            resample_crypto_data(full_path, output_base, resample_configs)
        except Exception as e:
            print(f"Error processing {filename}: {e}")