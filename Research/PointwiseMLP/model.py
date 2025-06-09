import torch
import torch.nn as nn
from config import Config


class Message(nn.Module):
    """Message network: 2 → 36 → 8 → 3 → 2."""
    def __init__(self):
        super().__init__()
        m1, m2, m3 = Config.MESSAGE_HIDDEN_SIZES
        self.fc1 = nn.Linear(Config.MESSAGE_INPUT_DIM, m1)
        self.fc2 = nn.Linear(m1, m2)
        self.fc3 = nn.Linear(m2, m3)
        self.fc4 = nn.Linear(m3, Config.MESSAGE_INPUT_DIM)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)


class Update(nn.Module):
    """Update network: 4 → 36 → 8 → 3 → 1."""
    def __init__(self):
        super().__init__()
        u1, u2, u3 = Config.UPDATE_HIDDEN_SIZES
        self.fc1 = nn.Linear(Config.UPDATE_INPUT_DIM, u1)
        self.fc2 = nn.Linear(u1, u2)
        self.fc3 = nn.Linear(u2, u3)
        self.fc4 = nn.Linear(u3, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)


class ImplicitGNN(nn.Module):
    """
    One‐step implicit GNN:
      - split h into neighbor features (8 dims) and center (u + phi)
      - message‐pass over 4 neighbors → sum
      - concat with center u and phi_u → update network
    """
    def __init__(self):
        super().__init__()
        self.message = Message()
        self.update = Update()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [batch, 10] = [4×(x,phi), u, phi_u]
        # neighbors = first 8 dims → reshape into (batch,4,2)
        h_other = h[:, :8].view(-1, 4, Config.MESSAGE_INPUT_DIM)
        # center u = column index 8
        h_center = h[:, [8]]
        # phi_u = column index 9
        phi_u = h[:, [9]]

        # message‐pass and sum
        m = self.message(h_other)           # (batch,4,2)
        m_sum = m.sum(dim=1)                # (batch,2)

        # concat [m_sum, u, phi_u] → (batch,4)
        u_in = torch.cat([m_sum, h_center, phi_u], dim=1)
        return self.update(u_in)
