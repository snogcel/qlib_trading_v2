from typing import Any, List, Optional, cast

import numpy as np
import pandas as pd

from qlib.rl.order_execution.interpreter import FullHistoryStateInterpreter, FullHistoryObs, canonicalize
from qlib.rl.data.base import ProcessedDataProvider
from qlib_custom.meta_trigger.simulator_state import SAOEExtendedState
from qlib.utils import init_instance_by_config

class VWAPAwareInterpreter(FullHistoryStateInterpreter):
    def __init__(self,
        max_step: int,
        data_ticks: int,
        data_dim: int,
        processed_data_provider: dict | ProcessedDataProvider,
    ) -> None:
        super().__init__()

        self.max_step = max_step
        self.data_ticks = data_ticks
        self.data_dim = data_dim
        self.processed_data_provider: ProcessedDataProvider = init_instance_by_config(
            processed_data_provider,
            accept_types=ProcessedDataProvider,
        )

    def interpret(self, state: SAOEExtendedState) -> FullHistoryObs:
        processed = self.processed_data_provider.get_data(
            stock_id=state.order.stock_id,
            date=pd.Timestamp(state.order.start_time.date()),
            feature_dim=self.data_dim,
            time_index=state.ticks_index,
        )

        position_history = np.full(self.max_step + 1, 0.0, dtype=np.float32)
        position_history[0] = state.order.amount
        position_history[1 : len(state.history_steps) + 1] = state.history_steps["position"].to_numpy()



        # The min, slice here are to make sure that indices fit into the range,
        # even after the final step of the simulator (in the done step),
        # to make network in policy happy.
        return cast(
            FullHistoryObs,
            canonicalize(
                {
                    "data_processed": np.array(self._mask_future_info(processed.today, state.cur_time)),
                    "data_processed_prev": np.array(processed.yesterday),
                    "acquiring": _to_int32(state.order.direction == state.order.BUY),
                    "cur_tick": _to_int32(min(int(np.sum(state.ticks_index < state.cur_time)), self.data_ticks - 1)),
                    "cur_step": _to_int32(min(state.cur_step, self.max_step - 1)),
                    "num_step": _to_int32(self.max_step),
                    "target": _to_float32(state.order.amount),
                    "position": _to_float32(state.position),
                    "position_history": _to_float32(position_history[: self.max_step]),
                },
            ),
        )    

def _to_int32(val):
    return np.array(int(val), dtype=np.int32)

def _to_float32(val):
    return np.array(val, dtype=np.float32)
