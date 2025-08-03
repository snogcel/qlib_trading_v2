# experience_buffer.py

import random
import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity=50000):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done, direction):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done, direction))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return map(np.array, [state, action, reward, next_state, done])

    def __len__(self):
        return len(self.buffer)