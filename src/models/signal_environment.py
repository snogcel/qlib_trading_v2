import gym
import numpy as np
import pandas as pd

from collections import deque
import numpy as np
import pandas as pd

class TierReliabilityTracker:
    def __init__(self, window=90):
        self.window = window
        self.rewards = {tier: deque(maxlen=window) for tier in ["A", "B", "C", "D"]}
        self.raw_pnl = {tier: deque(maxlen=window) for tier in ["A", "B", "C", "D"]}
        self.weights = {tier: 1.0 for tier in ["A", "B", "C", "D"]}

    def log_trade(self, tier, reward, raw_pnl):
        if tier in self.rewards:
            self.rewards[tier].append(reward)
        if tier in self.raw_pnl:            
            self.raw_pnl[tier].append(raw_pnl)

    def update_weights(self, scale_range=(0.5, 1.5)):
        new_weights = {}
        for tier, rewards in self.rewards.items():
            if len(rewards) < 10:  # not enough data yet
                new_weights[tier] = 1.0
                continue
            r = np.array(rewards)
            mean = np.mean(r)
            std = np.std(r) + 1e-6
            sharpe = mean / std
            # Map Sharpe to bounded [min, max] scale via sigmoid-like transform
            weight = np.clip(0.5 + 0.5 * np.tanh(sharpe), *scale_range)
            new_weights[tier] = weight
        self.weights = new_weights
        return self.weights

class SignalEnv(gym.Env):
    def __init__(self, df, window_size=1, position_limit=1.0, tier_weights=None, reward_type="tier_weighted", eval_mode=False):
        super().__init__()
        self.calibrator = TierReliabilityTracker(window=90)
        self.writer = None  # will be set from outside
        self.log_step = 0   # track episode progress

        self.df = df.copy()
        self.window_size = window_size
        self.position_limit = position_limit
        self.pointer = 0
        self.position = 0.0  # starting flat

        self.fee_rate = 0.001
        self.slippage_rate = 0.0005

        self.reward_type = reward_type
        self.eval_mode = eval_mode

        self.tier_log = []
        self.reward_log = []
        self.position_log = []

        # Allow external override, else use default
        self.tier_weights = tier_weights or {
            "A": 1.0,
            "B": 1.0,
            "C": 1.0,
            "D": 1.0
        }
  
        # Continuous action: [-1, 1] represents full short to full long
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))

        print(self.df)
        #self.df.to_csv("df_checknulls.csv")

        # features + tier one-hot
        state_dim = 14 + 4 # SignalEnv.get_state
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))

    def reset(self):        
        assert len(self.df) > self.window_size, "DataFrame too short for chosen window"
        self.pointer = self.window_size
        self.tier_log = []
        self.reward_log = []
        self.position_log = []
        self.trade_log = []  # Full trade trace
        self.position = 0.0  # Reset held position
        return self._get_state()
    
    def get_recent_vol_scaled(self, window=10):
        values = self.df["vol_scaled"].iloc[max(0, self.pointer - window):self.pointer]
        return float(np.clip(np.mean(values), 0.0, 1.0))

    def step(self, action):
        # Clip action to valid range
        action = np.clip(action[0], -self.position_limit, self.position_limit)

        # Return for the current step
        next_return = self.df.iloc[self.pointer]["truth"]
        vol_estimate = self.df.iloc[self.pointer].get("vol_rank", 1.0)  # optional if using normalized rewards
        vol_scaled = self.df.iloc[self.pointer].get("vol_scaled", 1.0)  # provided to PPO

        # Tier (use previous step index for signal context)
        idx = max(0, self.pointer - 1)
        tier = self.df["signal_tier"].iloc[idx]
        tier_weight = self.tier_weights.get(tier, 1.0) # TODO -- this is broken / non-weighted, signal_tier is being passed as a float not character (3.0 instead of "A")

        # optional: neutralize weighting in test mode
        if self.eval_mode:
            tier_weight = 1.0
 
        # Trading cost: delta in position from previous to now
        delta_position = action - self.position
        fee = self.fee_rate * abs(delta_position)
        slippage = self.slippage_rate * (delta_position ** 2)  # Optional: quadratic penalty

        # Reward based on PREVIOUS position (i.e. held through this return), scaled
        raw_pnl = self.position * next_return

        if self.reward_type == "risk_normalized":
            reward = (raw_pnl / vol_estimate) * tier_weight            
        else:
            reward = raw_pnl * tier_weight

        # Apply trading costs (in all cases unless eval_mode)
        if not self.eval_mode:
            reward -= fee + slippage

        # Log to calibrator
        self.calibrator.log_trade(tier, reward, raw_pnl)

        # Update tier weights dynamically
        if not self.eval_mode:
            self.tier_weights = self.calibrator.update_weights()        

        # output to TensorBoard
        if self.writer and hasattr(self.writer, "record"):
            for tier, weight in self.tier_weights.items():
                self.writer.record(f"tier_weights/{tier}", weight, exclude="stdout")
                self.writer.record(f"raw_pnl/{tier}", raw_pnl, exclude="stdout")

            # self.writer.record(f"meta/vol_estimate", vol_estimate, exclude="stdout")
            self.log_step += 1

        # Logging
        self.reward_log.append(reward)
        self.position_log.append(self.position)
        self.tier_log.append(tier)
        self.trade_log.append({
            "tier": tier,
            "reward": reward,
            "raw_pnl": raw_pnl,
            "vol_estimate": vol_estimate,
            "vol_scaled": vol_scaled,
            "fee": fee,
            "slippage": slippage,
            "delta_position": delta_position,
            "position_before": self.position,
            "position_after": action,
            "date": self.df.iloc[idx].name
        })

        # Transition: update position and pointer
        self.position = action
        self.pointer += 1
        done = self.pointer >= len(self.df)

        if done:
            return self._get_terminal_state(), reward, done, {}

        next_state = self._get_state()
        info = {
            "date": self.df.iloc[idx].name,
            "raw_reward": raw_pnl,
            "tier_weight": tier_weight,
            "fee": fee,
            "slippage": slippage,
            "tier": tier,
            "position": self.position
        }

        return next_state, reward, done, info

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def _get_terminal_state(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_state(self):
        if self.pointer >= len(self.df):
            raise IndexError(f"Pointer {self.pointer} out of bounds for df with length {len(self.df)}")

        row = self.df.iloc[self.pointer]
        spread = row["q90"] - row["q10"]

        tier = row["signal_tier"]
        tier_onehot = [int(tier == t) for t in [3.0, 2.0, 1.0, 0.0]]        
        
        # Guard against early pointer values
        if self.pointer < 8:
            recent_vol = 0.0
        else:
            recent_vol = self.df["truth"].iloc[self.pointer - 7:self.pointer].std()
            recent_vol = 0.0 if pd.isna(recent_vol) else recent_vol        

        # state = np.array([
        #     row["q10"],
        #     row["q90"],
        #     row["q50"],
        #     spread,
        #     row["fg_index"],
        #     row["btc_dom"],
        #     row["momentum_5d"],
        #     row["momentum_10d"],
        #     self.position,  # <— position persistence added here
        #     row["vol_scaled"],  # 🌶️ Regime-aware observation
        #     row["signal_tier"]
        # ])

        state = np.array([
            row["q10"],
            row["q50"],
            row["q90"],
            
            row["$fg_index"],
            row["$btc_dom"],
            row["vol_scaled"],
            row["$momentum_5"],
            row["$momentum_10"],

            self.position,  # <— position persistence added here

            row["signal_score"],
            row["spread_score"],
            row["tier_confidence"],

            row["side"],           # persistence

            row.get("prob_up", 0), # optional
        ] + tier_onehot) 

        print(state)
        raise SystemExit()       

        return state
