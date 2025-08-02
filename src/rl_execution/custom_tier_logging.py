import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TierLoggingCallback(BaseCallback):
    def __init__(self, env, verbose=0, log_interval=100):
        super().__init__(verbose)
        self.env = env
        self.log_interval = log_interval
        self.counter = 0

    def _on_step(self) -> bool:
        self.counter += 1
        if self.counter % self.log_interval == 0:
            weights = self.env.calibrator.weights if hasattr(self.env, 'calibrator') else self.env.tier_weights
            for tier, weight in weights.items():
                self.logger.record(f"tier_weights/{tier}", float(weight), exclude="stdout")
            
            if hasattr(self.env, 'trade_log') and self.env.trade_log:                
                latest_step = self.env.trade_log[-1]
                tier = latest_step["tier"]
                self.logger.record(f"tier_total_pnl/{tier}", latest_step["reward"], exclude="stdout")               
                self.logger.record(f"tier_raw_pnl/{tier}", latest_step["raw_pnl"], exclude="stdout")
                self.logger.record(f"meta/vol_estimate", latest_step["vol_estimate"], exclude="stdout")
                self.logger.record(f"meta/vol_scaled", latest_step["vol_scaled"], exclude="stdout")
                self.logger.record(f"meta/raw_pnl", latest_step["raw_pnl"], exclude="stdout")
                self.logger.record(f"meta/fee", latest_step["fee"], exclude="stdout")
                self.logger.record(f"meta/slippage", latest_step["slippage"], exclude="stdout")
                self.logger.record(f"meta/delta_position", latest_step["delta_position"], exclude="stdout")

            if hasattr(self.model, 'ent_coef'):                
                self.logger.record("meta/ent_coef", float(self.model.ent_coef), exclude="stdout")
            
            if hasattr(self.env, 'reward_log') and self.env.reward_log:
                recent_reward = self.env.reward_log[-1]
                self.logger.record("meta/reward_latest", recent_reward, exclude="stdout")

            if hasattr(self.env, 'position'):
                self.logger.record("meta/position", self.env.position, exclude="stdout")

        return True
