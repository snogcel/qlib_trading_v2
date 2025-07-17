import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import qlib
from qlib.constant import REG_US, REG_CN 
from qlib.utils import init_instance_by_config, flatten_dict
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score, accuracy_score
from qlib.data.dataset.handler import DataHandlerLP
from qlib_custom.custom_ndl import CustomNestedDataLoader
from qlib_custom.custom_signal_env import SignalEnv
from qlib_custom.custom_tier_logging import TierLoggingCallback
from qlib_custom.custom_multi_quantile import QuantileLGBModel
from qlib_custom.gdelt_handler import gdelt_handler, gdelt_dataloader
from qlib_custom.crypto_handler import crypto_handler, crypto_dataloader


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from inspect import getfullargspec

train_start_time = "2018-08-02"
train_end_time = "2023-12-31"
valid_start_time = "2024-01-01"
valid_end_time = "2024-09-30"
test_start_time = "2024-10-01"
test_end_time = "2025-04-01"

fit_start_time = None
fit_end_time = None

provider_uri = "/Projects/qlib_trading_v2/qlib_data/CRYPTO_DATA"

SEED = 42
MARKET = "all"
BENCHMARK = "BTCUSDT"
EXP_NAME = "crypto_exp_101"
FREQ = "day"

qlib.init(provider_uri=provider_uri, region=REG_US)




def adaptive_entropy_coef(vol_scaled, base=0.005, min_coef=0.001, max_coef=0.02):
    """
    Map vol_scaled ∈ [0, 1] → entropy coef.
    More entropy in low-vol regimes, less in high-volatility ones.
    """
    inverse_vol = 1.0 - vol_scaled
    coef = base * (1 + inverse_vol * 2.0)  # amplify entropy in quiet regimes
    return float(np.clip(coef, min_coef, max_coef))

class EntropyAwarePPO(PPO):
    def __init__(self, *args, volatility_getter=None, base_entropy=0.005, **kwargs):
        super().__init__(*args, **kwargs)
        self.volatility_getter = volatility_getter
        self.base_entropy = base_entropy

    def train(self):
        if callable(self.volatility_getter):
            vol_scaled = self.volatility_getter()
            self.ent_coef = adaptive_entropy_coef(vol_scaled, base=self.base_entropy)

        super().train()



def check_transform_proc(proc_l, fit_start_time, fit_end_time):
        new_l = []
        for p in proc_l:
            if not isinstance(p, Processor):
                klass, pkwargs = get_callable_kwargs(p, processor_module)
                args = getfullargspec(klass).args
                if "fit_start_time" in args and "fit_end_time" in args:
                    assert (
                        fit_start_time is not None and fit_end_time is not None
                    ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                    pkwargs.update(
                        {
                            "fit_start_time": fit_start_time,
                            "fit_end_time": fit_end_time,
                        }
                    )
                proc_config = {"class": klass.__name__, "kwargs": pkwargs}
                if isinstance(p, dict) and "module_path" in p:
                    proc_config["module_path"] = p["module_path"]
                new_l.append(proc_config)
            else:
                new_l.append(p)
        return new_l

# ==========================
# Reward Calculation Logic
# ==========================

def compute_reward(position, next_return, tier_weight, fee=0.001, slippage=0.0005, volatility_estimate=None, reward_type="tier_weighted"):
    raw_pnl = position * next_return
    if reward_type == "risk_normalized" and volatility_estimate is not None:
        raw_pnl /= volatility_estimate
    delta_position = 0  # To be filled by env
    fee_cost = fee * abs(delta_position)
    slip_cost = slippage * delta_position ** 2
    reward = raw_pnl * tier_weight - fee_cost - slip_cost
    return reward

# ==========================
# Evaluate Agent After Training
# ==========================

def evaluate_agent(env, agent, name="experiment"):
    obs = env.reset()
    done = False

    rewards, positions, tiers = [], [], []
    tier_rewards = {"A": [], "B": [], "C": [], "D": []}
    tier_positions = {"A": [], "B": [], "C": [], "D": []}

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        positions.append(env.position)
        tier = info.get("tier", "Unknown")

        if tier in tier_rewards:
            tier_rewards[tier].append(reward)
            tier_positions[tier].append(env.position)

        tiers.append(tier)

    df = pd.DataFrame({"tier": tiers, "reward": rewards, "position": positions})

    df.to_csv(f"{name}.csv")

    # Summarize tier performance
    summary = df.groupby("tier").agg({
        "reward": "sum",
        "position": lambda x: np.mean(np.abs(x))
    }).rename(columns={"reward": "total_pnl", "position": "avg_exposure"})

    summary["efficiency"] = summary["total_pnl"] / summary["avg_exposure"]
    print(f"\n🧪 Evaluation for {name}")
    print(summary.round(4))

    # Save plot of PnL per tier
    summary.to_csv(f"{name}_summary.csv")
    summary[["total_pnl", "efficiency"]].plot(kind="bar", title=f"Tier Attribution – {name}")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{name}_tier_performance.png")
    plt.close()




if __name__ == '__main__': 

    _learn_processors = [{"class": "DropnaLabel"},]
    _infer_processors = []

    infer_processors = check_transform_proc(_infer_processors, fit_start_time, fit_end_time)
    learn_processors = check_transform_proc(_learn_processors, fit_start_time, fit_end_time)

    

    # loader_config = {
    #     "feature": crypto_dataloader.get_feature_config(),
    #     "label": crypto_dataloader.get_label_config()
    # }

    # freq_config = {
    #     "feature": "day", 
    #     "label": "day"
    # }

    # data_loader_config = {
    #     "class": "QlibDataLoader",
    #     "kwargs": {
    #         "config": loader_config,
    #         "freq": freq_config,
    #         "inst_processors": inst_processors,                         
    #     },
    # }

    # generic version for later usage...

    # data_loader_config = {
    #     "class": "QlibDataLoader",
    #     "kwargs": {
    #         "config": loader_config,
    #         "freq": freq_config,
    #         "inst_processors": inst_processors,                         
    #     },
    # }
    
    freq_config = {
        "feature": "day", 
        "label": "day"
    }

    inst_processors = [
        {
            "class": "TimeRangeFlt",
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {
                "start_time": train_start_time,
                "end_time": test_end_time,
                "freq": "day"
            }
        }
    ]

    crypto_data_loader = {
        "class": "crypto_dataloader",
        "module_path": "qlib_custom.crypto_loader",
        "kwargs": {
            "config": {
                "feature": crypto_dataloader.get_feature_config(),
                "label": crypto_dataloader.get_label_config(),                                                
            },                                            
            "freq": freq_config["label"],  # "day"
            "inst_processors": inst_processors
        }
    }

    gdelt_data_loader = {
        "class": "gdelt_dataloader",
        "module_path": "qlib_custom.gdelt_loader",
        "kwargs": {
            "config": {
                "feature": gdelt_dataloader.get_feature_config()
            },
            "freq": freq_config["feature"],  # "day"
            "inst_processors": inst_processors
        }
    }

    nd = CustomNestedDataLoader(dataloader_l=[crypto_data_loader, gdelt_data_loader], join="left")
    #nd.load(instruments=["BTCUSDT", "BTC_FEAT"], start_time=train_start_time, end_time=test_end_time)
    
    handler_config = {
        "instruments": ["BTCUSDT", "BTC_FEAT"],
        "start_time": train_start_time,
        "end_time": test_end_time,                
        "data_loader": nd,        
        "infer_processors": infer_processors,
        "learn_processors": learn_processors,
        "shared_processors": [],
        "process_type": DataHandlerLP.PTYPE_A,
        "drop_raw": False 
    }

    dataset_handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": handler_config,
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
                "handler": dataset_handler_config,
                "segments": {
                    "train": (train_start_time, train_end_time),
                    "valid": (valid_start_time, valid_end_time),
                    "test": (test_start_time, test_end_time),
                },
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

    # Phase 2...
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
        X_all["fg_index"], # sentiment
        X_all["btc_dom"], # capital flow
        X_all["$momentum_5d"].rename("momentum_5d"),
        X_all["$momentum_10d"].rename("momentum_10d"),
        X_all["$realized_vol_10d"].rename("volatility"),
    ], axis=1).dropna()

    
    rolling_window = 20
    q_low = df_all["volatility"].rolling(rolling_window).quantile(0.01)
    q_high = df_all["volatility"].rolling(rolling_window).quantile(0.99)

    df_all["vol_scaled"] = ((df_all["volatility"] - q_low.shift(1)) / (q_high.shift(1) - q_low.shift(1))).clip(0.0, 1.0)
    df_all["vol_rank"] = df_all["volatility"].rank(pct=True)

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

    print(df_all)
    df_all.to_csv("df_all.csv")
    raise SystemExit()

    df_cleaned = df_all.dropna(subset=["signal_tier"])    

    df_cleaned.to_csv("df_cleaned.csv")

    df_to_pickle = df_cleaned.copy()
    

    
    #df_to_pickle.set_index(["instrument", "datetime"], inplace=True)

    TIER_MAP = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}
    df_to_pickle["signal_tier"] = df_to_pickle["signal_tier"].map(TIER_MAP)

    ## moved from train_meta_wrapper -- this is messy as fuck
    df_to_pickle["signal_strength"] = df_to_pickle["abs_q50"] / df_to_pickle["signal_thresh"]
    df_to_pickle["spread_quality"] = 1 - (df_to_pickle["spread"] / df_to_pickle["spread_thresh"])

    df_to_pickle["tier_confidence"] = np.clip(((df_to_pickle["abs_q50"] / df_to_pickle["signal_thresh"]) + (1 - (df_to_pickle["spread"] / df_to_pickle["spread_thresh"]))) / 2, 0.0, 1.0)

    # df_to_pickle["side"] = np.where(
    #     (df_to_pickle["q90"] - df_to_pickle["q50"]) > (df_to_pickle["spread_thresh"] / 2),
    #     1,  # Buy
    #     np.where(
    #         (df_to_pickle["q10"] - df_to_pickle["q50"]) < (-df_to_pickle["spread_thresh"] / 2),
    #         0,  # Sell
    #         -1  # Hold or uncertain
    #     )
    # )

    def prob_up_piecewise(row):
        q10, q50, q90 = row["q10"], row["q50"], row["q90"]
        if q90 <= 0:
            return 0.0
        if q10 >= 0:
            return 1.0
        # 0 lies between q10 and q50
        if q10 < 0 <= q50:
            cdf0 = 0.10 + 0.40 * (0 - q10) / (q50 - q10)
            return 1 - cdf0
        # 0 lies between q50 and q90
        cdf0 = 0.50 + 0.40 * (0 - q50) / (q90 - q50)
        return 1 - cdf0


    # Step 1: Compute probability of upside
    q10 = df_to_pickle["q10"]
    q50 = df_to_pickle["q50"]
    q90 = df_to_pickle["q90"]

    #den = (q90 - q10).replace(0, np.nan)
    #prob_up = ((0 - q10) / den).clip(0, 1).fillna(0.5)

    df_to_pickle["prob_up"] = df_to_pickle.apply(prob_up_piecewise, axis=1)
    prob_up = df_to_pickle["prob_up"]

    # Step 2: Define thresholds
    signal_thresh = df_to_pickle["signal_thresh"]
    # signal_thresh = 0.005

    # Step 3: Create masks
    buy_mask = (q50 > signal_thresh) & (prob_up > 0.5)
    sell_mask = (q50 < -signal_thresh) & (prob_up < 0.5)

    # Step 4: Assign side
    df_to_pickle["side"] = -1  # default to HOLD
    df_to_pickle.loc[buy_mask, "side"] = 1
    df_to_pickle.loc[sell_mask, "side"] = 0  # or -1 if you prefer SELL = -1


    print("df_to_pickle: ", df_to_pickle)
    df_to_pickle.to_csv("df_to_pickle_v9.csv")

    # Drop column 'truth' from the copied DataFrame
    df_to_pickle.drop('truth', axis=1, inplace=True)

    df_to_pickle.to_pickle("./data3/macro_features.pkl")

    raise SystemExit()



    df_train_rl = df_cleaned.loc["2018-02-01":"2023-12-31"]
    df_val_rl   = df_cleaned.loc["2024-01-01":"2024-09-30"]
    df_test_rl  = df_cleaned.loc["2024-10-01":"2025-04-01"]


    # ==========================
    # Sweep Runner
    # ==========================

    # ==========================
    # Experiment Configuration
    # ==========================
    
    reward_type = "tier_weighted"
    run_name = f"base_momentum_{reward_type}"
    
    # Base PPO hyperparameters
    ppo_config = {
        "learning_rate": 3e-4, 
        "clip_range": 0.20, 
        "ent_coef": 0.005, 
        "gae_lambda": 0.95, 
        "vf_coef": 0.5
    }

    # =========================
    # Run Experiment
    # =========================
    
    print(f"\n🚀 Launching: {run_name}")

    env_train = SignalEnv(df_train_rl, reward_type=reward_type)
    env_val = SignalEnv(df_val_rl, reward_type=reward_type, eval_mode=False)
    env_test = SignalEnv(df_test_rl, reward_type=reward_type, eval_mode=False)

    vec_env = DummyVecEnv([lambda: env_train])
    vec_env.seed(SEED)

    callback = TierLoggingCallback(env_train, log_interval=50)

    agent = EntropyAwarePPO(
        policy="MlpPolicy",
        env=env_train,
        learning_rate=ppo_config["learning_rate"],
        clip_range=ppo_config["clip_range"],
        ent_coef=ppo_config["ent_coef"],
        gae_lambda=ppo_config["gae_lambda"],
        vf_coef=ppo_config["vf_coef"],
        seed=SEED, 
        verbose=1,
        volatility_getter=env_train.get_recent_vol_scaled,
        tensorboard_log=f"./logs_v3/{run_name}"
    )
    
    agent.learn(total_timesteps=184_000, callback=callback)        
    agent.save(f"./models/{run_name}")

    evaluate_agent(env_val, agent, name=run_name)
