from __future__ import annotations

from typing import cast

import numpy as np

from qlib.backtest.decision import OrderDir
from qlib.rl.order_execution.state import SAOEMetrics
from qlib_custom.meta_trigger.simulator_state import SAOEExtendedState
from qlib.rl.reward import Reward

class HybridExecutionReward(Reward[SAOEExtendedState]):
    """
    Combines PA, VWAP alignment, pressure awareness, and execution dispersion into a unified reward signal.
    """

    def __init__(
        self,
        weight_pa=1.0,
        weight_vwap=0.5,
        weight_pressure=0.3,
        weight_penalty=1.0,
        vwap_feature="$vwap_3h",
        pressure_feature="$vwap_pressure"
    ) -> None:
        self.weight_pa = weight_pa
        self.weight_vwap = weight_vwap
        self.weight_pressure = weight_pressure
        self.weight_penalty = weight_penalty
        self.vwap_feature = vwap_feature
        self.pressure_feature = pressure_feature        

    def reward(self, simulator_state: SAOEExtendedState) -> float:
        order = simulator_state.order
        whole_order = order.amount
        assert whole_order > 0

        last_step = simulator_state.history_steps.reset_index().iloc[-1].to_dict()        
        pa = last_step["pa"] * last_step["amount"] / whole_order
        
        # VWAP deviation
        cur_dt = last_step["datetime"]        
        features = simulator_state.backtest_data.df

        trade_price = last_step["trade_price"]
        vwap_price = features.loc[cur_dt, self.vwap_feature]
        vwap_penalty = -abs(trade_price - vwap_price)

        # Pressure signal
        pressure = features.loc[cur_dt, self.pressure_feature]
        pressure_penalty = -abs(pressure)

        # Execution clumping penalty (quadratic dispersion)
        breakdown = simulator_state.history_exec.loc[last_step["datetime"] :]
        weights = (breakdown["amount"] / whole_order) ** 2
        dispersion_penalty = -weights.sum()

        # Combine reward components
        total_reward = (
            self.weight_pa * pa
            + self.weight_vwap * vwap_penalty
            + self.weight_pressure * pressure_penalty
            + self.weight_penalty * dispersion_penalty
        )

        # Safety check
        assert not np.isnan(total_reward), f"NaN reward for simulator state: {simulator_state}"

        # Logging for diagnostics
        bin_idx = getattr(simulator_state, "current_bin", None)

        if bin_idx is not None:
            self.log(f"action_bin/reward_pa_bin_{bin_idx}", pa)
            self.log(f"action_bin/reward_avg_bin_{bin_idx}", total_reward)

        self.log("reward/pa", pa)
        self.log("reward/vwap_penalty", vwap_penalty)
        self.log("reward/pressure_penalty", pressure_penalty)
        self.log("reward/dispersion_penalty", dispersion_penalty)
        self.log("reward/avg", total_reward)

        return total_reward




# class VWAPAnchoredReward(Reward[SAOEState]):
#     """Encourage higher PAs, but penalize stacking all the amounts within a very short time.
#     Formally, for each time step, the reward is :math:`(PA_t * vol_t / target - vol_t^2 * penalty)`.

#     Parameters
#     ----------
#     penalty
#         The penalty for large volume in a short time.
#     scale
#         The weight used to scale up or down the reward.
#     """

#     def __init__(self, penalty: float = 100.0, scale: float = 1.0) -> None:
#         self.penalty = penalty
#         self.scale = scale

#     def reward(self, simulator_state: SAOEState) -> float:
#         # Access today's dynamic macro features
#         today_feats = simulator_state.simulator.processed.today

#         # Optional: also grab yesterday’s for deltas
#         yesterday_feats = simulator_state.simulator.processed.yesterday

#         # Example: use VWAP pressure
#         vwap_pressure = today_feats.get("vwap_pressure", 0.0)
#         tier_conf = today_feats.get("tier_confidence", 0.0)
#         deal_amount = simulator_state.order.amount


#         # existing PPO Penalty logic
#         whole_order = simulator_state.order.amount
#         assert whole_order > 0
#         last_step = cast(SAOEMetrics, simulator_state.history_steps.reset_index().iloc[-1].to_dict())

#         pa = last_step["pa"] * last_step["amount"] / whole_order

#         # Inspect the "break-down" of the latest step: trading amount at every tick
#         last_step_breakdown = simulator_state.history_exec.loc[last_step["datetime"] :]
#         penalty = -self.penalty * ((last_step_breakdown["amount"] / whole_order) ** 2).sum()


#         # Sample reward logic       
#         penalty_vwap = abs(vwap_pressure) * deal_amount
#         pa_vwap = tier_conf * 1.2  # boost confidence-based advantage



#         reward = pa + penalty

#         # Throw error in case of NaN
#         assert not (np.isnan(reward) or np.isinf(reward)), f"Invalid reward for simulator state: {simulator_state}"

#         self.log("reward/pa", pa)
#         self.log("reward/penalty", penalty)
#         return reward * self.scale