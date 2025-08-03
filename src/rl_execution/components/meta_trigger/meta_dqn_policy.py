from tianshou.policy import BasePolicy
from .meta_dqn_model import MetaQNetwork
import torch
import os

class MetaDQNPolicy(BasePolicy):
    def __init__(self, input_dim, checkpoint_path=None, threshold=0.5):
        super().__init__()

        self.model = MetaQNetwork(input_dim)

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model.eval()
        else:
            print(f"[MetaDQNPolicy] Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")

        self.threshold = threshold  # optional confidence gating

    # def decide(self, feature_vector: dict) -> bool:        
    #     state = torch.FloatTensor([list(feature_vector.values())])
    #     with torch.no_grad():
    #         q_values = self.model(state)            
    #         action = torch.argmax(q_values, dim=1).item()            
    #         print(f"tier: {feature_vector['tier_confidence']}, q50: {feature_vector['q50']}, q_values: {q_values}")
    #     return action == 1  # 1 = trigger execution

    def decide(self, feature_vector: dict) -> bool:
        state = torch.FloatTensor([list(feature_vector.values())])
        with torch.no_grad():
            q_values = self.model(state)
            gap = q_values[0, 1] - q_values[0, 0]
        print(f"tier: {feature_vector['tier_confidence']}, q50: {feature_vector['q50']}, q_values: {q_values}, gap: {gap}")
        return gap > 0.3 and q_values[0, 1] > 0.0

    def forward(self, batch, state=None, **kwargs):
        raise NotImplementedError("MetaDQNPolicy does not support Tianshou rollout yet.")

    def learn(self, batch, **kwargs):
        raise NotImplementedError("MetaDQNPolicy is a gating layer, not a trainer.")