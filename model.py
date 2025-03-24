import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=8):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def get_action(self, state):
        action_probs = self(torch.tensor(state))
        action_dist = torch.distributions.Categorical(action_probs)
        return action_dist.sample().item()
