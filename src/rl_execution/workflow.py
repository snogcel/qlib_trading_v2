# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path

import qlib
import pandas as pd
from qlib.config import C, REG_CN
from qlib.workflow import R
from qlib.workflow.cli import workflow

if __name__ == "__main__":
    # region initialize qlib
    # please register your account to access data as instructed here:
    # https://github.com/microsoft/qlib#best-practice-for-qlib-data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    # endregion

    # region create toy orders
    # create some toy orders if the file doesn't exist
    order_file_path = Path("data/orders/toy.pkl")
    if not order_file_path.exists():
        print("Order file not found. Creating toy orders...")
        order_file_path.parent.mkdir(parents=True, exist_ok=True)
        orders = []
        for i in range(10):
            orders.append(
                {
                    "stock_id": "SH600519",
                    "amount": 1000.0,
                    "direction": "buy",
                    "start_time": pd.Timestamp("2020-01-01 09:30:00"),
                    "end_time": pd.Timestamp("2020-01-01 14:55:00"),
                }
            )
        pd.DataFrame(orders).to_pickle(order_file_path)
    # endregion

    # region backtest with TWAP
    with R.uri_context("workflow_TWAP.pkl"):
        # R.start can be called multiple times
        R.start(experiment_name="workflow_TWAP")
        # record some variables
        # R.log_params can only be called once
        R.log_params(
            dataset={
                "class": "PickleStyledOrderExecution",
                "module_path": "qlib.rl.order_execution.data",
                "kwargs": {
                    "order_file": order_file_path,
                    "instrument": "SH600519",
                    "start_time": "2019-01-01",
                    "end_time": "2020-12-31",
                    "freq": "5min",
                },
            }
        )
        R.log_params(
            strategy={
                "class": "TWAP",
                "module_path": "qlib.rl.order_execution.strategy",
                "kwargs": {
                    "order": None,
                    "split_interval": 30,
                },
            }
        )
        R.log_params(
            executor={
                "class": "NestedExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "5min",
                    "verbose": True,
                },
            }
        )
        R.log_params(
            exchange={
                "class": "Exchange",
                "module_path": "qlib.backtest.exchange",
                "kwargs": {
                    "freq": "5min",
                    "codes": "SH600519",
                    "deal_price": "close",
                },
            }
        )
        R.log_params(
            backtest={
                "class": "MultiOrderExecution",
                "module_path": "qlib.rl.order_execution.backtest",
                "kwargs": {
                    "orders": None,
                    "strategy": None,
                    "executor": None,
                    "exchange": None,
                    "benchmark": 0.0,
                },
            }
        )
        # R.end can be called multiple times
        R.end()
    # endregion

    # region backtest with PPO
    # get the latest checkpoint of PPO from remote
    # NOTE: You can get the same result by running `examples/train_opds.yml`
    # and using the latest checkpoint.
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from examples.rl_order_execution.exp_configs.util import download_checkpoint

    checkpoint_path = download_checkpoint("PPO")

    with R.uri_context("workflow_PPO.pkl"):
        R.start(experiment_name="workflow_PPO")
        R.log_params(
            dataset={
                "class": "PickleStyledOrderExecution",
                "module_path": "qlib.rl.order_execution.data",
                "kwargs": {
                    "order_file": order_file_path,
                    "instrument": "SH600519",
                    "start_time": "2019-01-01",
                    "end_time": "2020-12-31",
                    "freq": "5min",
                },
            }
        )
        R.log_params(
            strategy={
                "class": "PPO",
                "module_path": "qlib.rl.order_execution.policy",
                "kwargs": {
                    "order": None,
                    "weight_file": checkpoint_path,
                },
            }
        )
        R.log_params(
            executor={
                "class": "NestedExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "5min",
                    "verbose": True,
                },
            }
        )
        R.log_params(
            exchange={
                "class": "Exchange",
                "module_path": "qlib.backtest.exchange",
                "kwargs": {
                    "freq": "5min",
                    "codes": "SH6f00519",
                    "deal_price": "close",
                },
            }
        )
        R.log_params(
            backtest={
                "class": "MultiOrderExecution",
                "module_path": "qlib.rl.order_execution.backtest",
                "kwargs": {
                    "orders": None,
                    "strategy": None,
                    "executor": None,
                    "exchange": None,
                    "benchmark": 0.0,
                },
            }
        )
        R.end()
    # endregion

    # backtest all strategies
    workflow("workflow_TWAP.pkl", "MultiOrderExecution")
    workflow("workflow_PPO.pkl", "MultiOrderExecution")
