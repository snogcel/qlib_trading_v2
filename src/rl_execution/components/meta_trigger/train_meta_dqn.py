import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from .meta_dqn_model import MetaQNetwork


def train_meta_dqn_model(
    buffer_path: str,
    checkpoint_out: str,
    input_dim_override: int = None,
    num_epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:

    # === Load experience buffer ===
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)

    states = torch.tensor(np.array([item[0] for item in buffer]), dtype=torch.float32)
    actions = torch.tensor([item[1] for item in buffer], dtype=torch.int64)
    rewards = torch.tensor([item[2] for item in buffer], dtype=torch.float32)

    print(states)
    print(actions)
    print(rewards)

    input_dim = input_dim_override or states.shape[1]
    dataset = TensorDataset(states, actions, rewards)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MetaQNetwork(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for state, action, reward in loader:
            q_values = model(state)
            target_q = q_values.clone()
            for i in range(len(action)):
                target_q[i, action[i]] = reward[i]

            loss = loss_fn(q_values, target_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Meta-DQN Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), checkpoint_out)
    print(f"Meta-DQN model saved to: {checkpoint_out}")
