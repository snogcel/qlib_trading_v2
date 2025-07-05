import pandas as pd
import numpy as np
import random
import torch


import seaborn as sns

import optuna
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import lightgbm as lgb
from sklearn.metrics import mean_pinball_loss

import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_US, REG_CN 
from qlib.utils import init_instance_by_config, flatten_dict
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, accuracy_score
from qlib.data.dataset.handler import DataHandlerLP
from qlib_custom.custom_signal_env import SignalEnv
from qlib_custom.custom_multi_quantile import QuantileLGBModel
from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader

from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from qlib.contrib.model.gbdt import LGBModel

provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTODATA"

SEED = 42
MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_100"
FREQ = "day"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

qlib.init(provider_uri=provider_uri, region=REG_US)


class EntropyLogger(BaseCallback):
    def __init__(self, check_freq, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.entropies = []
        self.timesteps = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            entropy = self.model.logger.name_to_value.get("train/entropy_loss", np.nan)
            entropy = float(entropy)
            if not np.isnan(entropy):
                self.entropies.append(entropy)
                self.timesteps.append(self.num_timesteps)
                if self.verbose:
                    print(f"Step {self.num_timesteps}: Entropy = {entropy:.4f}")

            pointer = self.model.logger.name_to_value.get
            print(f"Epoch {self.pointer}: weights = {self.tier_weights}")
        return True

class PerQuantileTuner:
    def __init__(self, dataset, X_train, y_train, X_val, y_val, quantiles=[0.1, 0.5, 0.9], seed=SEED):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.quantiles = quantiles
        self.seed = seed
        self.dataset = dataset

    def tune_single_q(self, q):
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("lr", 0.01, 0.1),
                #"num_leaves": trial.suggest_int("leaves", 16, 256),
                "max_depth": trial.suggest_int("depth", 3, 10),
                "n_estimators": trial.suggest_int("trees", 300, 1000),
                "objective": "quantile",
                "alpha": q,
                "verbosity": -1,
                "seed": self.seed,
                "early_stopping_rounds": 0
            }
            model = QuantileLGBModel(**params)
            model.fit(dataset=self.dataset, verbose_eval=0)
            preds = model.predict(dataset=self.dataset, segment="valid")
            return mean_pinball_loss(self.y_val, preds, alpha=q)

        sampler = TPESampler(seed=SEED)  # Make the sampler behave in a deterministic way.        
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=50)
        return study.best_params, study.best_value

    def run(self):
        best_params = {}
        for q in self.quantiles:
            print(f"\n📐 Tuning for q={q}")
            params, loss = self.tune_single_q(q)
            print(f"  CRPS: {loss:.6f}, Params: {params}")
            best_params[q] = params
        return best_params

def plot_quantile_loss_curves(eval_results_dict, quantiles):
    """
    Plots training and validation loss curves for multiple quantile models.

    Parameters:
        eval_results_dict: dict
            Keys are quantiles (e.g., 0.1, 0.5, 0.9), values are evals_result_ from LightGBM models.
        quantiles: list
            List of quantiles (must match keys in eval_results_dict).
    """
    fig, axs = plt.subplots(len(quantiles), 1, figsize=(8, 4 * len(quantiles)), sharex=True)

    for idx, q in enumerate(quantiles):
        ax = axs[idx]
        results = eval_results_dict[q]

        train_loss = results["train"]["quantile"]
        val_loss = results["valid"]["quantile"]

        ax.plot(train_loss, label="Train", color="C0", alpha=0.7)
        ax.plot(val_loss, label="Validation", color="C1", alpha=0.9)
        ax.set_title(f"q = {q} Quantile Loss")
        ax.set_ylabel("Pinball Loss")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Boosting Round")
    plt.tight_layout()
    plt.show()

def crps_score(y_true, quantile_preds, quantiles):
    """
    Compute CRPS for a set of quantile predictions.
    
    Parameters:
    - y_true: Series of true values
    - quantile_preds: DataFrame with columns like 'quantile_0.1', 'quantile_0.5', etc.
    - quantiles: List of quantile levels (e.g., [0.1, 0.5, 0.9])
    
    Returns:
    - CRPS score (float)
    """
    crps = 0.0
    for q in quantiles:
        q_col = f"quantile_{q:.2f}"        
        y_true_aligned, q_preds_aligned = y_true.align(quantile_preds[q_col], join="inner", axis=0)
        indicator = (y_true_aligned < q_preds_aligned).astype(float)
        crps += np.mean((indicator - q) * (q_preds_aligned - y_true_aligned))
    return crps

def compute_regime_crps(y_true, preds, meta_df, quantiles):
    results = {}

    # Sentiment regimes
    results["Greed"] = crps_score(
        y_true[meta_df["fg_index"] > 60],
        preds.loc[meta_df["fg_index"] > 60],
        quantiles
    )
    results["Neutral"] = crps_score(
        y_true[(meta_df["fg_index"] >= 40) & (meta_df["fg_index"] <= 60)],
        preds.loc[(meta_df["fg_index"] >= 40) & (meta_df["fg_index"] <= 60)],
        quantiles
    )
    results["Fear"] = crps_score(
        y_true[meta_df["fg_index"] < 40],
        preds.loc[meta_df["fg_index"] < 40],
        quantiles
    )

    # Capital flow regimes
    results["BTC-Dominant"] = crps_score(
        y_true[meta_df["btc_dom"] > 60],
        preds.loc[meta_df["btc_dom"] > 60],
        quantiles
    )
    results["Altcoin-Dominant"] = crps_score(
        y_true[meta_df["btc_dom"] < 40],
        preds.loc[meta_df["btc_dom"] < 40],
        quantiles
    )

    # Volatility regimes
    vol = y_true.rolling(7).std()
    results["High Vol"] = crps_score(
        y_true[vol > vol.quantile(0.75)],
        preds.loc[vol > vol.quantile(0.75)],
        quantiles
    )
    results["Low Vol"] = crps_score(
        y_true[vol < vol.quantile(0.25)],
        preds.loc[vol < vol.quantile(0.25)],
        quantiles
    )

    return results


if __name__ == '__main__': 

    gdelt_handler_kwargs = {
        "start_time": "20180201",
        "end_time": "20250401",
        "fit_start_time": "20180201",
        "fit_end_time": "20231231",
        "instruments": ["GDELT_BTC_FEAT"]
    }

    crypto_handler_kwargs = {
        "start_time": "20180201",
        "end_time": "20250401",
        "fit_start_time": "20180201",
        "fit_end_time": "20231231",
        "instruments": ["BTCUSDT"]
    }

    task_config = {        
        "model": {
            "class": "MultiQuantileModel",
            "module_path": "qlib_custom.custom_multi_quantile",
            "kwargs": {
                "quantiles": [0.1, 0.5, 0.9],
                "lgb_params": {
                    0.1: {"learning_rate": 0.08971845032956545, "max_depth": 3, "n_estimators": 953, "seed": SEED},
                    0.5: {"learning_rate": 0.022554447458683766, "max_depth": 5, "n_estimators": 556, "seed": SEED},
                    0.9: {"learning_rate": 0.018590766014390355, "max_depth": 4, "n_estimators": 333, "seed": SEED}
                }
            }
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset",  # No need to customize this if using base
                    "kwargs": {
                        "instruments": ["BTCUSDT", "GDELT_BTC_FEAT"],
                        "start_time": "20180201",
                        "end_time": "20250401",
                        "process_type": "append",
                        "drop_raw": False,
                        "data_loader": {
                            "class": "CustomNestedDataLoader",
                            "module_path": "qlib_custom.custom_ndl",
                            "kwargs": {
                                "dataloader_l": [
                                    {
                                        "class": "crypto_dataloader",
                                        "module_path": "qlib_custom.crypto_loader",
                                        "kwargs": {
                                            "config": {
                                                "feature": crypto_dataloader.get_feature_config()
                                            },
                                            "freq": "day",  # Replace with your FREQ variable
                                            "inst_processors": []
                                        }
                                    },
                                    {
                                        "class": "gdelt_dataloader",
                                        "module_path": "qlib_custom.gdelt_loader",
                                        "kwargs": {
                                            "config": {
                                                "feature": gdelt_dataloader.get_feature_config()
                                            },
                                            "freq": "day",  # Replace with your FREQ variable
                                            "inst_processors": []
                                        }
                                    }
                                ],
                                "join": "left"
                            }
                        }
                    }
                },
                "segments": {
                    "train": ("20180201", "20231231"),
                    "valid": ("20240101", "20240930"),
                    "test": ("20241001", "20250401")
                }
            }
        },
        "task": {
            "model": "LGBModel",
            "dataset": "DatasetH",
            "record": [
                {
                    "class": "SignalRecord",
                    "module_path": "qlib.workflow.record_temp"
                },
                {
                    "class": "PortAnaRecord",
                    "module_path": "qlib.workflow.record_temp"
                }
            ]
        }
    }

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    # prepare segments
    df_train, df_valid, df_test = dataset.prepare(
        segments=["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
    )

    # split data
    X_train, y_train = df_train["feature"], df_train["label"]
    X_val, y_val = df_valid["feature"], df_valid["label"]
    X_test, y_test = df_test["feature"], df_test["label"]

    model.fit(dataset=dataset)

    plot_quantile_loss_curves(model.evals_result_dict, quantiles=[0.1, 0.5, 0.9])

    preds_train = model.predict(dataset, "train")
    preds_valid = model.predict(dataset, "valid")
    preds_test = model.predict(dataset, "test")

    y_all = pd.concat([y_train, y_val, y_test], axis=0, join='outer', ignore_index=False)
    X_all = pd.concat([X_train, X_val, X_test], axis=0, join='outer', ignore_index=False)
    preds = pd.concat([preds_train, preds_valid, preds_test], axis=0, join='outer', ignore_index=False)
    
    df_all = pd.concat([
        preds["quantile_0.10"].rename("q10"),
        preds["quantile_0.50"].rename("q50"),
        preds["quantile_0.90"].rename("q90"),
        y_all["LABEL0"].rename("truth"),
        X_all["fg_index"],  # sentiment
        X_all["btc_dom"]    # capital flow
    ], axis=1).dropna()
    
    df_all["spread"] = df_all["q90"] - df_all["q10"]
    df_all["abs_q50"] = df_all["q50"].abs()

    # Tiering    
    df_all["spread_thresh"] = df_all["spread"].rolling(180).quantile(0.5).shift(1)
    df_all["signal_thresh"] = df_all["abs_q50"].rolling(180).quantile(0.5).shift(1)
    
    def classify(row):
        if pd.isna(row["spread_thresh"]) or pd.isna(row["signal_thresh"]):
            return np.nan  # or "D" for startup grace period
        if row["abs_q50"] >= row["signal_thresh"] and row["spread"] < row["spread_thresh"]:
            return "A"
        elif row["abs_q50"] >= row["signal_thresh"]:
            return "B"
        elif row["spread"] < row["spread_thresh"]:
            return "C"
        else:
            return "D"
    
    df_all["signal_tier"] = df_all.apply(classify, axis=1)

    df_combined = df_all.copy()

    more_quantiles = [0.4, 0.5, 0.6, 0.65]

    for q in more_quantiles:
        df_combined[f"spread_q{int(q*100)}"] = df_combined["spread"].rolling(180).quantile(q).shift(1)
        df_combined[f"absq50_q{int(q*100)}"] = df_combined["abs_q50"].rolling(180).quantile(q).shift(1)
   
    df_train_rl = df_combined.loc["2018-02-01":"2023-12-31"]
    df_val_rl   = df_combined.loc["2024-01-01":"2024-09-30"]
    df_test_rl  = df_combined.loc["2024-10-01":"2025-04-01"]

    df_combined = df_combined.reset_index(level="instrument")
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot spread + rolling thresholds
    axs[0].plot(df_combined["spread"], label="Spread", color="steelblue", alpha=0.5)
    axs[0].plot(df_combined["spread_q40"], label="40th Percentile", linestyle="--", color="purple")
    axs[0].plot(df_combined["spread_q50"], label="50th Percentile", linestyle="--", color="green")
    axs[0].set_title("Spread and Rolling Thresholds")
    axs[0].legend()
    axs[0].grid(True)

    # Plot abs_q50 + rolling thresholds
    axs[1].plot(df_combined["abs_q50"], label="|q50|", color="orange", alpha=0.5)
    axs[1].plot(df_combined["absq50_q50"], label="50th Percentile", linestyle="--", color="green")
    axs[1].plot(df_combined["absq50_q60"], label="60th Percentile", linestyle="--", color="teal")
    axs[1].plot(df_combined["absq50_q65"], label="65th Percentile", linestyle="--", color="navy")
    axs[1].set_title("abs(q50) and Rolling Thresholds")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    summary = {}

    for tier in ["A", "B", "C", "D"]:
        subset = df_combined[df_combined["signal_tier"] == tier]
        if len(subset) < 10:
            continue  # skip sparse tiers

        # Hit rate
        hit_rate = (np.sign(subset["q50"]) == np.sign(subset["truth"])).mean()

        # Average return
        avg_return = subset["truth"].mean()

        # Volatility-adjusted performance
        return_std = subset["truth"].std()
        info_ratio = avg_return / return_std if return_std > 0 else np.nan

        summary[tier] = {
            "Hit Rate": hit_rate,
            "Avg Return": avg_return,
            "Info Ratio": info_ratio,
            "Count": len(subset)
        }

    # Display results
    result = pd.DataFrame(summary).T.sort_values("Info Ratio", ascending=False)
    result.to_csv("tier_perf.csv")

    # train PPO model...

    """ def objective(trial):
        # Suggest tier weights
        w_A = trial.suggest_float("tier_A", 1.00, 1.25)
        w_B = trial.suggest_float("tier_B", 0.75, 1.25)
        w_C = trial.suggest_float("tier_C", 1.00, 1.45)
        w_D = trial.suggest_float("tier_D", 0.75, 1.25)  # allow light penalty

        tier_weights = {"A": w_A, "B": w_B, "C": w_C, "D": w_D}

        # Setup training environment with these weights
        env_train = DummyVecEnv([lambda: SignalEnv(df_train_rl, tier_weights=tier_weights)])

        model = PPO("MlpPolicy", env_train, verbose=0)
        model.learn(total_timesteps=150_000)

        # Run evaluation rollout (on held-out df_test_rl)
        env_test = SignalEnv(df_val_rl)
        obs = env_test.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env_test.step(action)
            total_reward += reward

        return total_reward  # Maximize cumulative PnL on test

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial.value)
    print(study.best_params) """

    # Trial 17 finished with value: 1.0277787446975708 and parameters: {'tier_A': 1.2264196151177191, 'tier_B': 0.8946567850131599, 'tier_C': 1.0682735230212668, 'tier_D': 0.7959912079010019}. Best is trial 17 with value: 1.0277787446975708.
    custom_weights = {
        "A": 1.2264196151177191,
        "B": 0.8946567850131599,
        "C": 1.0682735230212668,
        "D": 0.7959912079010019
    }

    env_train = SignalEnv(df_train_rl, tier_weights=custom_weights)

    vec_env = DummyVecEnv([lambda: env_train])
    vec_env.seed(SEED)
    
    entropy_logger = EntropyLogger(check_freq=2048, verbose=1)  # once per PPO rollout

    model = PPO("MlpPolicy", vec_env, seed=SEED, verbose=1)
    model.learn(total_timesteps=150000, callback=entropy_logger)

    plt.figure(figsize=(10, 4))
    plt.plot(entropy_logger.timesteps, entropy_logger.entropies, marker='o')
    plt.title("Policy Entropy During PPO Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Entropy Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    # test model...
    
    env_test = SignalEnv(df_test_rl)
    obs = env_test.reset()

    rewards = []
    positions = []
    dates = []
    truths = []
    q50s = []
    tiers = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # deterministic for eval
        obs, reward, done, info = env_test.step(action)

        # Log data for analysis
        rewards.append(reward)
        positions.append(env_test.position)
        dates.append(info.get("date", None))
        truths.append(env_test.df.iloc[env_test.pointer - 1]["truth"])
        q50s.append(env_test.df.iloc[env_test.pointer - 1]["q50"])
        tiers.append(env_test.df.iloc[env_test.pointer - 1]["signal_tier"])

    results = pd.DataFrame({
        "date": dates,
        "reward": rewards,
        "position": positions,
        "truth": truths,
        "q50": q50s,
        "tier": tiers
    })

    results.index = pd.MultiIndex.from_tuples(dates, names=["datetime", "instrument"])

    # Optionally reindex by datetime alone
    results = results.reset_index(level="instrument")

    results["pnl"] = results["position"] * results["truth"]

    results["cumulative_return"] = results["pnl"].cumsum()

    step_log = pd.DataFrame({
        "tier": env_test.tier_log,
        "reward": env_test.reward_log,
        "position": env_test.position_log
    })

    # Total reward per tier
    pnl_by_tier = step_log.groupby("tier")["reward"].sum()

    # Avg absolute exposure per tier
    exposure_by_tier = step_log.groupby("tier")["position"].apply(lambda x: x.abs().mean())

    # Reward per unit exposure (efficiency)
    efficiency = pnl_by_tier / exposure_by_tier

    print(pnl_by_tier)
    print(exposure_by_tier)
    print(efficiency)

    sns.barplot(x=efficiency.index, y=efficiency.values)
    plt.title("PnL Efficiency by Tier")
    plt.show()



    # Cumulative return
    btc_results = results[results["instrument"] == "BTCUSDT"]

    plt.figure(figsize=(12, 4))
    plt.plot(btc_results.index, btc_results["cumulative_return"])
    plt.title("RD-Agent Cumulative PnL on BTCUSDT")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Position vs. Tier
    tier_colors = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4", "D": "#d62728"}
    colors = [tier_colors[t] for t in results["tier"]]

    plt.figure(figsize=(12, 3))
    plt.scatter(results.index, results["position"], c=colors, s=12, alpha=0.7)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("RD-Agent Position Over Time (Colored by Signal Tier)")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.show()


    


    raise SystemExit()


    #crps = crps_score(y_test["LABEL0"], preds, quantiles=[0.1, 0.5, 0.9])
    #print(f"CRPS: {crps:.6f}")

    crps = compute_regime_crps(y_val["LABEL0"], preds_valid, X_val, quantiles=[0.1, 0.5, 0.9])
    print("CRPS: ", crps)

    """ CRPS = { 
        'Greed': np.float64(0.014279662386042632), 
        'Neutral': np.float64(0.0170002339767871), 
        'Fear': np.float64(0.017805951986194182), 
        'BTC-Dominant': np.float64(0.017060239528393254), 
        'Altcoin-Dominant': np.nan, 
        'High Vol': np.float64(0.022327142234575794), 
        'Low Vol': np.float64(0.008411361452043485)
    } """


    # Ensure your labels are a Series
    true_labels = y_test["LABEL0"]
    pred_10 = preds["quantile_0.10"]
    pred_50 = preds["quantile_0.50"]
    pred_90 = preds["quantile_0.90"]

    # Align everything
    true_labels, pred_10 = true_labels.align(pred_10, axis=0, join='inner')
    _, pred_50 = true_labels.align(pred_50, axis=0, join='inner')
    _, pred_90 = true_labels.align(pred_90, axis=0, join='inner')

    # Compute spread and residual
    spread = (pred_90 - pred_10).rename("spread")
    residual = (pred_50 - true_labels).abs().rename("residual")

    # Join into a DataFrame for easy handling
    df = pd.concat([spread, residual], axis=1).dropna()

    # Compute correlations
    pearson_corr = df["spread"].corr(df["residual"])
    spearman_corr = df["spread"].corr(df["residual"], method="spearman")

    print(f"Pearson correlation (Spread vs. Residual): {pearson_corr:.4f}")
    print(f"Spearman correlation (Spread vs. Residual): {spearman_corr:.4f}")



    ########

    # Prep
    df = pd.concat([
        preds["quantile_0.10"].rename("q10"),
        preds["quantile_0.50"].rename("q50"),
        preds["quantile_0.90"].rename("q90"),
        y_test["LABEL0"].rename("truth")
    ], axis=1).dropna()

    # Derived columns
    df["spread"] = df["q90"] - df["q10"]
    df["abs_q50"] = df["q50"].abs()  # magnitude of signal

    # Define percentiles
    spread_thresh = df["spread"].quantile(0.5)   # median width
    signal_thresh = df["abs_q50"].quantile(0.5)  # median strength

    # Tier logic
    def classify(row):
        if row["abs_q50"] >= signal_thresh and row["spread"] < spread_thresh:
            return "A"
        elif row["abs_q50"] >= signal_thresh and row["spread"] >= spread_thresh:
            return "B"
        elif row["abs_q50"] < signal_thresh and row["spread"] < spread_thresh:
            return "C"
        else:
            return "D"

    df["signal_tier"] = df.apply(classify, axis=1)

    import matplotlib.pyplot as plt

    # Ensure your DataFrame has a datetime index (use 'datetime' level if MultiIndex)
    df = df.reset_index()  # if you’re working from a MultiIndex
    df["date"] = pd.to_datetime(df["datetime"])

    # Optional: Filter for a single instrument (like BTCUSDT)
    df = df[df["instrument"] == "BTCUSDT"]

    # Map tiers to colors
    tier_colors = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4", "D": "#d62728"}
    df["color"] = df["signal_tier"].map(tier_colors)

    """ # Plot
    plt.figure(figsize=(14, 4))
    plt.scatter(df["date"], df["q50"], c=df["color"], label=None, s=25, alpha=0.8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("🎯 Signal Tier Classification Over Time (Q50)")
    plt.ylabel("Median Prediction (Q50)")
    plt.xlabel("Date")

    # Add custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Tier {k}", 
                        markerfacecolor=v, markersize=8) for k, v in tier_colors.items()]
    plt.legend(handles=handles, title="Signal Tier")

    plt.tight_layout()
    plt.show() """

    





    # start exp
    """ with R.start(experiment_name="workflow"):
        R.log_params(**flatten_dict(task_config["task"]))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest. If users want to use backtest based on their own prediction,
        # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
        
        # par = PortAnaRecord(recorder, port_analysis_config, "day")
        # par.generate() """

