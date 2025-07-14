import os
import random
import sys
import warnings
import yaml
from pathlib import Path
from ruamel.yaml import YAML
from typing import cast, List, Optional

import numpy as np
import pandas as pd
import torch
from qlib.backtest import Order
from qlib.backtest.decision import OrderDir
from qlib.constant import ONE_MIN
from qlib.rl.data.native import load_handler_intraday_processed_data
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution import SingleAssetOrderExecutionSimple
from qlib.rl.reward import Reward
from qlib.rl.trainer import Checkpoint
from qlib_custom.custom_train import CustomTrainer, backtest, train
from qlib.rl.trainer.callbacks import Callback, EarlyStopping, MetricsWriter
from qlib.rl.utils.log import CsvWriter
from qlib.utils import init_instance_by_config
from tianshou.policy import BasePolicy
from torch.utils.data import Dataset

from copy import deepcopy
from qlib_custom.meta_trigger.meta_dqn_policy import MetaDQNPolicy
from qlib_custom.meta_trigger.experience_buffer import ExperienceBuffer
from qlib_custom.meta_trigger.train_meta_dqn import train_meta_dqn_model


from qlib_custom.custom_logger_callback import EpisodeLogger
from qlib_custom.logger.tensorboard_logger import TensorboardLogger
logger = TensorboardLogger(name="ppo_training")

#from qlib.rl.order_execution.trainer import PPOTrainer

OUTPUT_PATH = Path("/Projects/qlib_trading_v2/data3/selected_orders")
stock = "BTCUSDT"

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _read_orders(order_dir: Path) -> pd.DataFrame:
    if os.path.isfile(order_dir):
        return pd.read_pickle(order_dir)
    else:
        orders = []
        for file in order_dir.iterdir():
            order_data = pd.read_pickle(file)
            orders.append(order_data)
        return pd.concat(orders)

class LazyLoadDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        order_file_path: Path,
        default_start_time_index: int,
        default_end_time_index: int,
    ) -> None:
        self._default_start_time_index = default_start_time_index
        self._default_end_time_index = default_end_time_index

        self._order_df = _read_orders(order_file_path).reset_index()
        self._ticks_index: Optional[pd.DatetimeIndex] = None
        self._data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self._order_df)

    def __getitem__(self, index: int) -> Order:
        row = self._order_df.iloc[index]                
        date = pd.Timestamp(str(row["date"]))

        if self._ticks_index is None:
            # TODO: We only load ticks index once based on the assumption that ticks index of different dates
            # TODO: in one experiment are all the same. If that assumption is not hold, we need to load ticks index
            # TODO: of all dates.

            data = load_handler_intraday_processed_data(
                data_dir=self._data_dir,
                stock_id=row["instrument"],
                date=date,
                feature_columns_today=[],
                feature_columns_yesterday=[],
                backtest=True,
                index_only=True,
            )                    
            self._ticks_index = [t - date for t in data.today.index]

        order = Order(
            stock_id=row["instrument"],
            amount=row["amount"],
            direction=OrderDir(int(row["order_type"])),
            start_time=date + self._ticks_index[self._default_start_time_index],
            end_time=date + self._ticks_index[self._default_end_time_index - 1] + ONE_MIN,
        )

        return order

def train_and_test(
    env_config: dict,
    simulator_config: dict,
    trainer_config: dict,
    data_config: dict,
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    policy: BasePolicy,
    reward: Reward,
    run_training: bool,
    run_backtest: bool,
) -> None:
    order_root_path = Path(data_config["source"]["selected_order_dir"])

    data_granularity = simulator_config.get("data_granularity", 1)

    def _simulator_factory_simple(order: Order) -> SingleAssetOrderExecutionSimple:
        return SingleAssetOrderExecutionSimple(
            order=order,
            data_dir=data_config["source"]["feature_root_dir"],
            feature_columns_today=data_config["source"]["feature_columns_today"],
            feature_columns_yesterday=data_config["source"]["feature_columns_yesterday"],
            data_granularity=data_granularity,
            ticks_per_step=simulator_config["time_per_step"],
            vol_threshold=simulator_config["vol_limit"],
        )

    assert data_config["source"]["default_start_time_index"] % data_granularity == 0
    assert data_config["source"]["default_end_time_index"] % data_granularity == 0

    if run_training:
        train_dataset, valid_dataset = [
            LazyLoadDataset(
                data_dir=data_config["source"]["feature_root_dir"],
                order_file_path=order_root_path / tag,
                default_start_time_index=data_config["source"]["default_start_time_index"] // data_granularity,
                default_end_time_index=data_config["source"]["default_end_time_index"] // data_granularity,
            )
            for tag in ("train", "valid")
        ]

        callbacks: List[Callback] = []
        if "checkpoint_path" in trainer_config:                                    
            callbacks.append(MetricsWriter(dirpath=Path(trainer_config["checkpoint_path"])))
            callbacks.append(
                Checkpoint(
                    dirpath=Path(trainer_config["checkpoint_path"]) / "checkpoints",
                    every_n_iters=trainer_config.get("checkpoint_every_n_iters", 1),
                    save_latest="copy",
                ),
            )
        if "earlystop_patience" in trainer_config:
            callbacks.append(
                EarlyStopping(
                    patience=trainer_config["earlystop_patience"],
                    monitor="val/pa",
                )
            )

        train(
            simulator_fn=_simulator_factory_simple,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            policy=policy,
            reward=reward,
            initial_states=cast(List[Order], train_dataset),
            trainer_kwargs={
                "max_iters": trainer_config["max_epoch"],
                "finite_env_type": env_config["parallel_mode"],
                "concurrency": env_config["concurrency"],
                "val_every_n_iters": trainer_config.get("val_every_n_epoch", None),
                "callbacks": callbacks,
            },
            vessel_kwargs={
                "episode_per_iter": trainer_config["episode_per_collect"],
                "update_kwargs": {
                    "batch_size": trainer_config["batch_size"],
                    "repeat": trainer_config["repeat_per_collect"],
                },
                "val_initial_states": valid_dataset,
            },
        )

    if run_backtest:
        test_dataset = LazyLoadDataset(
            data_dir=data_config["source"]["feature_root_dir"],
            order_file_path=order_root_path / "test",
            default_start_time_index=data_config["source"]["default_start_time_index"] // data_granularity,
            default_end_time_index=data_config["source"]["default_end_time_index"] // data_granularity,
        )

        backtest(
            simulator_fn=_simulator_factory_simple,
            state_interpreter=state_interpreter,
            action_interpreter=action_interpreter,
            initial_states=test_dataset,
            policy=policy,
            logger=CsvWriter(Path(trainer_config["checkpoint_path"])),
            reward=reward,
            finite_env_type=env_config["parallel_mode"],
            concurrency=env_config["concurrency"],
        )







# === borrow from train_onpolicy.py ===


def main(config: dict, run_training: bool, run_backtest: bool) -> None:

    # === Meta-DQN setup ===
    meta_conf = config.get("meta_trigger", {})
    meta_enabled = meta_conf.get("enabled", False)

    macro_features_path = "./data3/macro_features.pkl"
    macro_df = pd.read_pickle(macro_features_path)
    macro_df.index.set_names(["instrument", "datetime"], inplace=True)

    macro_df["signal_strength"] = macro_df["abs_q50"] / macro_df["signal_thresh"]
    macro_df["spread_quality"] = 1 - (macro_df["spread"] / macro_df["spread_thresh"])

    macro_df["tier_confidence"] = np.clip(((macro_df["abs_q50"] / macro_df["signal_thresh"]) + (1 - (macro_df["spread"] / macro_df["spread_thresh"]))) / 2, 0.0, 1.0)

    macro_df["side"] = np.where(
        (macro_df["q90"] - macro_df["q50"]) > (macro_df["spread_thresh"] / 2),
        1,  # Buy
        np.where(
            (macro_df["q10"] - macro_df["q50"]) < (-macro_df["spread_thresh"] / 2),
            0,  # Sell
            -1  # Hold or uncertain
        )
    )

    meta_policy = None
    if meta_enabled:
        input_dim = len(macro_df.columns)
        checkpoint_path = meta_conf.get("checkpoint_path") # ./checkpoints/meta_dqn.pt
        meta_policy = MetaDQNPolicy(input_dim=input_dim, checkpoint_path=checkpoint_path)

    # === Extract feature vector for a given order ===
    def extract_meta_features(order):    
        dt = pd.Timestamp(order["date"])
        inst = order["instrument"]
        try:
            row = macro_df.loc[(inst, dt)]            
            return row.to_numpy(dtype=np.float32)
        except KeyError:
            return None

    # === Simulate order loading ===
    orders = pd.read_pickle(config["data"]["source"]["order_dir"] + f"/all/{stock}.pkl.target")  # replace if needed

    orders = orders.reset_index()
    selected_orders = []
    meta_buffer = ExperienceBuffer(capacity=50000)

    for idx, order in orders.iterrows():    
        features = extract_meta_features(order)    
        if features is None:
            continue
        feature_dict = dict(zip(macro_df.columns, features))

        tier_score = feature_dict.get("tier_confidence", 0.0)
        force_execute = tier_score > 0.95  # optional override

        rolling_spread_thresh = feature_dict.get("spread_thresh", 0.01) 
        q10 = feature_dict.get("q10", 0.0)
        q50 = feature_dict.get("q50", 0.0)
        q90 = feature_dict.get("q90", 0.0)
        side = feature_dict.get("side", 0)

        decision = meta_policy.decide(feature_dict)

        order.direction = OrderDir.BUY if feature_dict.get("side") == 1 else OrderDir.SELL
        order.order_type = 1 if feature_dict.get("side") == 1 else 0

        print(f"decision: {decision}, force_execute: {force_execute}, q10: {q10}, q50: {q50}, q90: {q90}, direction: {order.direction}")
        
        if decision or force_execute:
            print("selected: ", order)
            selected_orders.append(order)
            # Log transition (mock reward until PPO finishes)
            reward = 0.0
            meta_buffer.add(features, action=1, reward=reward, next_state=None, done=True, direction=order.direction)

    # === Optional: Save buffer for later Meta-DQN training ===
    with open("./data3/meta_buffer.pkl", "wb") as f:
        import pickle
        pickle.dump(meta_buffer.buffer, f)

    train_meta_dqn_model(
        buffer_path="./data3/meta_buffer.pkl",
        checkpoint_out="./checkpoints/meta_dqn.pt"
    )

    # === Wrap into PPO training ===
    print(f"[Meta-DQN] Selected {len(selected_orders)} / {len(orders)} orders for training.")
        
    selected_orders_df = pd.DataFrame(selected_orders)
    selected_orders_df.set_index(["date", "instrument"], inplace=True)  # or ["datetime", "instrument"] if needed

    order_train = selected_orders_df[selected_orders_df.index.get_level_values(0) <= pd.Timestamp("2023-12-31")]
    order_test = selected_orders_df[selected_orders_df.index.get_level_values(0) > pd.Timestamp("2024-01-01")]
    order_valid = order_test[order_test.index.get_level_values(0) <= pd.Timestamp("2024-10-01")]
    order_test = order_test[order_test.index.get_level_values(0) > pd.Timestamp("2024-10-01")]
    
    for order, tag in zip((order_train, order_valid, order_test, selected_orders_df), ("train", "valid", "test", "all")):
        path = OUTPUT_PATH / tag
        os.makedirs(path, exist_ok=True)
        if len(order) > 0:
            order.to_pickle(path / f"{stock}.pkl.target")
            #order.to_csv(path / f"{stock}.csv")
    
    ##
    ## COPIED FROM train_onpolicy.py
    ##

    if not run_training and not run_backtest:
        warnings.warn("Skip the entire job since training and backtest are both skipped.")
        return

    if "seed" in config["runtime"]:
        seed_everything(config["runtime"]["seed"])

    for extra_module_path in config["env"].get("extra_module_paths", []):
        sys.path.append(extra_module_path)

    state_interpreter: StateInterpreter = init_instance_by_config(config["state_interpreter"])
    action_interpreter: ActionInterpreter = init_instance_by_config(config["action_interpreter"])
    reward: Reward = init_instance_by_config(config["reward"])

    additional_policy_kwargs = {
        "obs_space": state_interpreter.observation_space,
        "action_space": action_interpreter.action_space,
    }

    # Create torch network
    if "network" in config:
        if "kwargs" not in config["network"]:
            config["network"]["kwargs"] = {}
        config["network"]["kwargs"].update({"obs_space": state_interpreter.observation_space})
        additional_policy_kwargs["network"] = init_instance_by_config(config["network"])

    # Create policy
    if "kwargs" not in config["policy"]:
        config["policy"]["kwargs"] = {}
    config["policy"]["kwargs"].update(additional_policy_kwargs)
    policy: BasePolicy = init_instance_by_config(config["policy"])

    use_cuda = config["runtime"].get("use_cuda", False)
    if use_cuda:
        policy.cuda()

    train_and_test(
        env_config=config["env"],
        simulator_config=config["simulator"],
        data_config=config["data"],
        trainer_config=config["trainer"],
        action_interpreter=action_interpreter,
        state_interpreter=state_interpreter,
        policy=policy,
        reward=reward,
        run_training=run_training,
        run_backtest=run_backtest,
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # === Load config ===
    with open("./rl_order_execution/exp_configs/train_ppo.yml", "r") as f:
        config = yaml.safe_load(f)

    main(config, run_training=True, run_backtest=False)







