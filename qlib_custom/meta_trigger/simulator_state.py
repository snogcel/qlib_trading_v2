# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass

import typing
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from qlib.backtest import Order
from qlib.rl.order_execution.state import SAOEMetrics
from qlib.typehint import TypedDict

if typing.TYPE_CHECKING:
    from qlib.rl.data.base import BaseIntradayBacktestData

@dataclass
class SAOEExtendedState:
    """Data structure holding a state for SAOE simulator."""

    order: Order
    """The order we are dealing with."""
    cur_time: pd.Timestamp
    """Current time, e.g., 9:30."""
    cur_step: int
    """Current step, e.g., 0."""
    position: float
    """Current remaining volume to execute."""
    history_exec: pd.DataFrame
    """See :attr:`SingleAssetOrderExecution.history_exec`."""
    history_steps: pd.DataFrame
    """See :attr:`SingleAssetOrderExecution.history_steps`."""

    metrics: Optional[SAOEMetrics]
    """Daily metric, only available when the trading is in "done" state."""

    backtest_data: BaseIntradayBacktestData
    """Backtest data is included in the state.
    Actually, only the time index of this data is needed, at this moment.
    I include the full data so that algorithms (e.g., VWAP) that relies on the raw data can be implemented.
    Interpreter can use this as they wish, but they should be careful not to leak future data.
    """

    ticks_per_step: int
    """How many ticks for each step."""
    ticks_index: pd.DatetimeIndex
    """Trading ticks in all day, NOT sliced by order (defined in data). e.g., [9:30, 9:31, ..., 14:59]."""
    ticks_for_order: pd.DatetimeIndex
    """Trading ticks sliced by order, e.g., [9:45, 9:46, ..., 14:44]."""

    current_bin: Optional[int] = None
    """Used for order_execution purposes"""
