from .experience_buffer import ExperienceBuffer
from .meta_dqn_model import MetaQNetwork
import torch
import torch.optim as optim

class MetaDQNTrainer:
    def __init__(self, input_dim, buffer):
        self.net = MetaQNetwork(input_dim)
        self.target = MetaQNetwork(input_dim)
        self.target.load_state_dict(self.net.state_dict())
        self.buffer = buffer
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()
        self.gamma = 0.99

    def train(self, batch_size=64):
        if len(self.buffer.buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_val = self.net(state).gather(1, action).squeeze()
        with torch.no_grad():
            q_next = self.target(next_state).max(1)[0]
        target = reward + self.gamma * q_next * (1 - done)

        loss = self.loss_fn(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def update_target(self):
        self.target.load_state_dict(self.net.state_dict())