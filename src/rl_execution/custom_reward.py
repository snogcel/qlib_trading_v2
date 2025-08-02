from __future__ import annotations

from typing import cast, Optional
import numpy as np
import pandas as pd

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
        pressure_feature="$vwap_pressure",
        confidence_feature="tier_confidence"
    ) -> None:
        self.weight_pa = weight_pa
        self.weight_vwap = weight_vwap
        self.weight_pressure = weight_pressure
        self.weight_penalty = weight_penalty
        self.vwap_feature = vwap_feature
        self.pressure_feature = pressure_feature
        self.tier_confidence = confidence_feature

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

        # Confidence Tiering
        tier_confidence = features.loc[cur_dt, self.tier_confidence]

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
        self.log("reward/tier_confidence", tier_confidence)
        self.log("reward/avg", total_reward)

        return total_reward



## V2

class HybridExecutionRewardV2(Reward[SAOEExtendedState]):
    def __init__(
        self,
        weight_execution_quality: float = 1.0,
        weight_confidence: float = 0.5,
        weight_dispersion: float = 0.3,
        max_step: Optional[int] = None,
    ):
        self.weight_execution_quality = weight_execution_quality
        self.weight_confidence = weight_confidence
        self.weight_dispersion = weight_dispersion
        self.max_step = max_step        

    def compute_vwap(self, history_exec):
        if history_exec["deal_amount"].sum() == 0.0:
            return np.average(history_exec["market_price"])
        return np.average(
            history_exec["market_price"],
            weights=history_exec["deal_amount"]
        )

    def compute_dispersion(self, history_exec):
        print(history_exec)
        raise SystemExit()
        bin_counts = history_exec["bin_index"].value_counts(normalize=True)
        dispersion = -np.sum(bin_counts * np.log(bin_counts + 1e-6))
        return dispersion
    
    def compute_ffr_dispersion(self, history_exec):
        ffr_values = history_exec["ffr"].values
        ffr_norm = ffr_values / (ffr_values.sum() + 1e-6)
        dispersion = -np.sum(ffr_norm * np.log(ffr_norm + 1e-6))
        return dispersion

    def reward(self, simulator_state: SAOEExtendedState) -> float:
        if simulator_state.cur_step == self.max_step - 1 or simulator_state.position < 1e-6:
            vwap_price = self.compute_vwap(simulator_state.history_exec)
            twap_price = simulator_state.backtest_data.get_deal_price().mean()

            if simulator_state.order.direction == OrderDir.SELL:
                ratio = vwap_price / twap_price if twap_price != 0 else 1.0
            else:
                ratio = twap_price / vwap_price if vwap_price != 0 else 1.0

            execution_quality = np.clip((ratio - 1.0) * 10, -1.0, 1.0)
            confidence = simulator_state.order.tier_confidence
            dispersion = self.compute_ffr_dispersion(simulator_state.history_exec)

            dispersion_effect = self.weight_dispersion * dispersion
            total_reward = (
                self.weight_execution_quality * execution_quality +
                self.weight_confidence * confidence +
                (-dispersion_effect if execution_quality < 0 else +dispersion_effect)
            )

            self.log("reward/execution_quality", execution_quality)
            self.log("reward/tier_confidence", confidence)            
            self.log("reward/dispersion_penalty", (-dispersion_effect if execution_quality < 0 else +dispersion_effect))            
            self.log("reward/total", total_reward)
            
            print(f"execution_quality: {execution_quality}, tier_confidence: {confidence}, dispersion_penalty: {(-dispersion_effect if execution_quality < 0 else +dispersion_effect)}, total_reward: {total_reward}")
            print(simulator_state.history_exec)

            return total_reward
        else:
            return 0.0




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