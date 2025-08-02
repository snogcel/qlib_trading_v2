# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import cast, List

import numpy as np
import pandas as pd

from qlib.backtest import CommonInfrastructure, Decision
from qlib.backtest.decision import Order
from qlib.rl.data.pickle_styled import PickleStyledData
from qlib.rl.simulator import Simulator
from qlib.utils import init_instance_by_config
from qlib_custom.custom_exchange import FeatureAwareExchange


class OrderExecutionEnv(Simulator):
    """Single-asset order execution environment.

    Parameters
    ----------
    order
        The order to be executed.
    trade_exchange
        Exchange instance.
    """

    def __init__(
        self,
        order: Order,
        trade_exchange: CommonInfrastructure,
    ):
        super().__init__()

        self.order = order
        self.trade_exchange = trade_exchange

        self.action_space = None
        self.observation_space = None

    def reset(self) -> None:
        self.trade_exchange.reset()

    def step(self, action: np.ndarray) -> None:
        """Execute one step.

        Parameters
        ----------
        action
            A numpy array with shape (1,). It indicates the number of shares to be traded.
            E.g., array([0.5]) means to trade 50% of shares.
        """
        assert action.ndim == 1

        order = self.order.copy()
        order.amount = action.item() * self.order.amount

        self.trade_exchange.step(order)

    def get_obs(self) -> pd.DataFrame:
        return self.trade_exchange.get_obs()

    def get_reward(self) -> float:
        return self.trade_exchange.get_reward()

    def done(self) -> bool:
        return self.trade_exchange.done()


class SAOEEnv(Simulator):
    """Single-asset order execution environment.

    This environment is used in training.

    Parameters
    ----------
    order
        The order to be executed.
    trade_exchange
        Exchange instance.
    """

    def __init__(
        self,
        order: Order,
        trade_exchange: CommonInfrastructure,
    ):
        super().__init__()

        self.order = order
        self.trade_exchange = trade_exchange

        self.action_space = None
        # TODO: find a better way to get the shape of observation
        self.observation_space = None

    def reset(self) -> None:
        self.trade_exchange.reset()

    def step(self, action: np.ndarray) -> None:
        """Execute one step.

        Parameters
        ----------
        action
            A numpy array with shape (1,). It indicates the number of shares to be traded.
            E.g., array([0.5]) means to trade 50% of shares.
        """
        assert action.ndim == 1

        order = self.order.copy()
        order.amount = action.item() * self.order.amount

        self.trade_exchange.step(order)

    def get_obs(self) -> pd.DataFrame:
        return self.trade_exchange.get_obs()

    def get_reward(self) -> float:
        return self.trade_exchange.get_reward()

    def done(self) -> bool:
        return self.trade_exchange.done()


class PickleStyledOrderExecution(PickleStyledData):
    """A wrapper to make pickle-styled data work with order execution envs."""

    def __init__(
        self,
        order_file: str | Path,
        instrument: str,
        start_time: str | pd.Timestamp,
        end_time: str | pd.Timestamp,
        **kwargs,
    ):
        super().__init__(instrument, start_time, end_time, **kwargs)

        self.order_file = order_file

    def _get_decision_from_order(self, order: pd.Series) -> Decision:
        return Decision(
            Order(
                stock_id=order.stock_id,
                amount=order.amount,
                direction=Order.BUY if order.direction == "buy" else Order.SELL,
                start_time=order.start_time,
                end_time=order.end_time,
            ),
            # TODO: find a better way to get the length of all data
            # The length of all data is used to create trade_calendar,
            # while trade_calendar is used to create exchange.
            # Exchange is not used in this decision, so we can pass a placeholder here.
            # A better way is to decouple the creation of exchange from decision.
            trade_calendar=pd.date_range(start=order.start_time, end=order.end_time, freq="1d"),
        )

    def __getitem__(self, key: int) -> Decision:
        order = self.orders.iloc[key]
        return self._get_decision_from_order(order)

    def __len__(self) -> int:
        return len(self.orders)

    def setup(self) -> None:
        super().setup()

        self.orders = pd.read_pickle(self.order_file)


class MultiOrderExecution(Simulator):
    def __init__(self, orders: List[Order], executor: object, exchange: object, benchmark: float = 0.0):
        super().__init__()

        self.orders = orders
        self.executor = executor
        self.exchange = exchange
        self.benchmark = benchmark

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def get_obs(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_reward(self) -> float:
        raise NotImplementedError

    def done(self) -> bool:
        raise NotImplementedError

    def execute(self) -> float:
        """Execute all orders and return the average PA."""
        pa_all = []
        for order in self.orders:
            self.executor.reset(order)
            self.exchange.reset(order)

            while not self.exchange.done():
                action = self.executor.get_action(self.exchange.get_obs())
                self.exchange.step(action)

            pa_all.append(self.exchange.get_pa())

        return cast(float, np.mean(pa_all))


def create_env(config: dict) -> OrderExecutionEnv:
    """Create an order execution environment."""
    exchange_config = config.get("exchange", {})
    exchange_config["class"] = "FeatureAwareExchange"
    exchange_config["module_path"] = "qlib_custom.custom_exchange"
    return init_instance_by_config(config, {"module_path": "qlib.rl.order_execution.sim", "class": "SAOE"})