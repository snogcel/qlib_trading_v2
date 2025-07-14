import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir="./runs", name="default"):
        path = os.path.join(log_dir, name)
        self.writer = SummaryWriter(path)
        self.step = 0

    def log_scalar(self, tag, value, step=None):
        self.writer.add_scalar(tag, value, step if step is not None else self.step)

    def log_scalars(self, metrics: dict, prefix="", step=None):
        for key, value in metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag, values, step=None):
        self.writer.add_histogram(tag, values, step if step is not None else self.step)

    def set_step(self, step: int):
        self.step = step

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
