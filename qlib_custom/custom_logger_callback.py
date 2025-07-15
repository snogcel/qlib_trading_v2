from qlib.rl.trainer.callbacks import Callback
from .custom_train import CustomTrainer
from .custom_training_vessel import CustomTrainingVessel
from .meta_trigger.meta_dqn_policy import MetaDQNPolicy

from pathlib import Path
import torch

class MetaDQNCheckpointManager(Callback):
    def __init__(self, meta_policy: MetaDQNPolicy, checkpoint_dir: Path, frequency: int = 1):
        self.meta_policy = meta_policy
        self.checkpoint_dir = checkpoint_dir
        self.frequency = frequency

    def on_iter_end(self, trainer, vessel):
        #trainer = kwargs.get("trainer", None)
        for key, value in trainer.metrics.items():
            print(f"{key}: {value}")

        if trainer and trainer.current_iter % self.frequency == 0:
            save_path = self.checkpoint_dir / f"meta_dqn_iter_{trainer.current_iter}.pt"
            torch.save(self.meta_policy.model.state_dict(), save_path)
            print(f"[Meta-DQN Checkpoint] Saved at: {save_path}")
