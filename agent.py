from abc import ABC, abstractmethod
from model import PolicyNetwork
import torch
from state import ACTION_SIZE, StateManager


class Agent(ABC):
    @abstractmethod
    def get_action(self):
        ...


class HumanAgent(Agent):
    def __init__(self):
        self.state_manager = StateManager()

    def get_action(self, obs):
        state, info = self.state_manager.get_state(obs)
        print(state, info)
        action = int(input())
        self.state_manager.update_action(action)
        return action


class PolicyGradientAgent(Agent):
    def __init__(self, path):
        self.state_manager = StateManager()
        self.policy = PolicyNetwork(self.state_manager.state_size, ACTION_SIZE)
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy.eval()

    @torch.no_grad()
    def get_action(self, obs):
        state, _ = self.state_manager.get_state(obs)
        action = self.policy.get_action(state)
        self.state_manager.update_action(action)
        return action
