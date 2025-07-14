from __future__ import annotations

import math
from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd
from gym import spaces

from qlib.constant import EPS
from qlib.rl.data.base import ProcessedDataProvider
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution.state import SAOEState
from qlib.typehint import TypedDict

from qlib.utils import init_instance_by_config

class CustomActionInterpreter(ActionInterpreter[SAOEState, int, float]):
    """Convert a discrete policy action to a continuous action, then multiplied by ``order.amount``.

    Parameters
    ----------
    values
        It can be a list of length $L$: $[a_1, a_2, \\ldots, a_L]$.
        Then when policy givens decision $x$, $a_x$ times order amount is the output.
        It can also be an integer $n$, in which case the list of length $n+1$ is auto-generated,
        i.e., $[0, 1/n, 2/n, \\ldots, n/n]$.
    max_step
        Total number of steps (an upper-bound estimation). For example, 390min / 30min-per-step = 13 steps.
    """

    def __init__(self, values: int | List[float], max_step: Optional[int] = None) -> None:
        super().__init__()

        if isinstance(values, int):
            values = [i / values for i in range(0, values + 1)]
        self.action_values = values
        self.max_step = max_step

    @property
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_values))

    def interpret(self, state: SAOEState, action: int) -> float:
        assert 0 <= action < len(self.action_values)

        if self.max_step is not None and state.cur_step >= self.max_step - 1:            
            return state.position
        else:            
            print("state.action_values: ", self.action_values[action])            
            
            return min(state.position, state.order.amount * self.action_values[action])
