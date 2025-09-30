import pandas as pd
import numpy as np
import random
import torch

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
from src.models.signal_environment import SignalEnv
from src.models.multi_quantile import QuantileLGBModel
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




#CRPS: 0.007312, Params: {'lr': 0.05104629857953324, 'leaves': 205, 'depth': 4, 'trees': 660}
#CRPS: 0.003459, Params: {'lr': 0.08020109984455288, 'leaves': 185, 'depth': 5, 'trees': 710}

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

        train_loss = results["training"]["quantile"]
        val_loss = results["valid_0"]["quantile"]

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
                    "test": ("20241010", "20250401")
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

    """ # Assuming X_train, y_train, X_val, y_val are all preprocessed numpy arrays
    tuner = PerQuantileTuner(dataset, X_train, y_train, X_val, y_val, quantiles=[0.1, 0.5, 0.9])
    study = tuner.run()

    raise SystemExit() """

    model.fit(dataset)


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
    spread_thresh = df_all["spread"].quantile(0.5)
    signal_thresh = df_all["abs_q50"].quantile(0.5)

    def classify(row):
        if row["abs_q50"] >= signal_thresh and row["spread"] < spread_thresh:
            return "A"
        elif row["abs_q50"] >= signal_thresh:
            return "B"
        elif row["spread"] < spread_thresh:
            return "C"
        else:
            return "D"

    df_all["signal_tier"] = df_all.apply(classify, axis=1)

    df_train_rl = df_all.loc["2018-02-01":"2023-12-31"]
    df_val_rl   = df_all.loc["2024-01-01":"2024-09-30"]
    df_test_rl  = df_all.loc["2024-10-01":"2025-04-01"]

    # train model...

    def objective(trial):
        # Suggest tier weights
        w_A = trial.suggest_float("tier_A", 1.00, 1.10)   # Consistently helpful, but risky if overweighted
        w_B = trial.suggest_float("tier_B", 0.95, 1.08)   # May need subtle deweighting after poor efficiency
        w_C = trial.suggest_float("tier_C", 1.15, 1.30)   # Your superstar in recent runs—lean into it
        w_D = trial.suggest_float("tier_D", 0.50, 0.70)   # Light touch; prevents full exclusion

        tier_weights = {"A": w_A, "B": w_B, "C": w_C, "D": w_D}

        # Setup training environment with these weights
        env_train = DummyVecEnv([lambda: SignalEnv(df_train_rl, tier_weights=tier_weights)])

        model = PPO("MlpPolicy", env_train, verbose=0)
        model.learn(total_timesteps=150_000)

        # Run evaluation rollout (on held-out df_test_rl)
        env_test = SignalEnv(df_test_rl)
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
    print(study.best_params)

    raise SystemExit()


    
    custom_weights = {
        "A": 1.7,
        "B": 1.3,
        "C": 1.0,
        "D": 0.3
    }

    env_train = SignalEnv(df_train_rl, tier_weights=custom_weights)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env_train])
    vec_env.seed(SEED)
    
    entropy_logger = EntropyLogger(check_freq=2048, verbose=1)  # once per PPO rollout

    model = PPO("MlpPolicy", vec_env, seed=SEED, verbose=1)
    model.learn(total_timesteps=500_000, callback=entropy_logger)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot(entropy_logger.timesteps, entropy_logger.entropies, marker='o')
    plt.title("Policy Entropy During PPO Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Entropy Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    # total_timesteps = 300_000
    # -----------------------------------------
    # | time/                   |             |
    # |    fps                  | 981         |
    # |    iterations           | 147         |
    # |    time_elapsed         | 306         |
    # |    total_timesteps      | 301056      |
    # | train/                  |             |
    # |    approx_kl            | 0.012343295 |
    # |    clip_fraction        | 0.0995      |
    # |    clip_range           | 0.2         |
    # |    entropy_loss         | -0.662      |
    # |    explained_variance   | 0.3         |
    # |    learning_rate        | 0.0003      |
    # |    loss                 | -0.0111     |
    # |    n_updates            | 1460        |
    # |    policy_gradient_loss | -0.00135    |
    # |    std                  | 0.466       |
    # |    value_loss           | 0.00978     |
    # -----------------------------------------

    # total_timesteps = 350_000
    # -----------------------------------------
    # | time/                   |             |
    # |    fps                  | 1024        |
    # |    iterations           | 171         |
    # |    time_elapsed         | 341         |
    # |    total_timesteps      | 350208      |
    # | train/                  |             |
    # |    approx_kl            | 0.011320235 |
    # |    clip_fraction        | 0.143       |
    # |    clip_range           | 0.2         |
    # |    entropy_loss         | -0.358      |
    # |    explained_variance   | 0.326       |
    # |    learning_rate        | 0.0003      |
    # |    loss                 | 0.0246      |
    # |    n_updates            | 1700        |
    # |    policy_gradient_loss | 0.000656    |
    # |    std                  | 0.342       |
    # |    value_loss           | 0.00739     |
    # -----------------------------------------

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

    print(results)

    results["pnl"] = results["position"] * results["truth"]
    results["cumulative_return"] = results["pnl"].cumsum()

    import matplotlib.pyplot as plt

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
    plt.title("Signal Tier Classification Over Time (Q50)")
    plt.ylabel("Median Prediction (Q50)")
    plt.xlabel("Date")

    # Add custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"Tier {k}", 
                        markerfacecolor=v, markersize=8) for k, v in tier_colors.items()]
    plt.legend(handles=handles, title="Signal Tier")

    plt.tight_layout()
    plt.show() """

    summary = {}

    for tier in ["A", "B", "C", "D"]:
        subset = df[df["signal_tier"] == tier]
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
    import pandas as pd
    result = pd.DataFrame(summary).T.sort_values("Info Ratio", ascending=False)
    result.to_csv("tier_perf.csv")





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

