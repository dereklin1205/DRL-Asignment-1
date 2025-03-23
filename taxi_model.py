import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, namedtuple
from enum import IntEnum
from typing import Tuple, Dict, Optional, List

# Define actions similar to the original
class Action(IntEnum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5

# More intuitive station type classification
class StationType(IntEnum):
    UNKNOWN = 0
    EMPTY = 1    # Changed from NONE to EMPTY for clarity
    PICKUP = 2   # Changed from PASSENGER to PICKUP
    DROPOFF = 3  # Changed from DESTINATION to DROPOFF

    def to_vector(self):
        vec = torch.zeros(len(self.__class__))
        vec[self.value] = 1
        return vec

# Feature processing similar to StateManager but with different architecture
class EnvironmentProcessor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.has_passenger = False
        self.station_types = [StationType.UNKNOWN] * 4
        self.pickup_position = None
        self.last_action = None
        self.visited_states = defaultdict(int)
        self.steps = 0
    
    def process_observation(self, obs) -> Tuple[List[float], Dict]:
        # Extract observation components
        self.taxi_position = (obs[0], obs[1])
        self.stations = [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7]),
            (obs[8], obs[9]),
        ]
        # Obstacles information
        obstacles = (obs[10], obs[11], obs[12], obs[13])
        near_pickup = obs[14] 
        near_dropoff = obs[15]
        
        # Update station information when discovering new information
        if self.pickup_position is None and near_pickup and self.taxi_position in self.stations:
            self.pickup_position = self.taxi_position
            station_idx = self.stations.index(self.taxi_position)
            self.station_types[station_idx] = StationType.PICKUP
        
        if self.taxi_position in self.stations:
            idx = self.stations.index(self.taxi_position)
            if near_dropoff:
                self.station_types[idx] = StationType.DROPOFF
            elif self.station_types[idx] == StationType.UNKNOWN:
                self.station_types[idx] = StationType.EMPTY
        
        # Infer unknown stations based on what we know
        known_count = sum(t != StationType.UNKNOWN for t in self.station_types)
        
        if known_count == 3:
            # If we know 3 stations, we can infer the 4th
            idx = self.station_types.index(StationType.UNKNOWN)
            # Calculate what the unknown station must be based on what we know
            total_value = sum(t.value for t in self.station_types if t != StationType.UNKNOWN)
            # The sum of all station types should be 6 (0+1+2+3)
            missing_value = 6 - total_value
            self.station_types[idx] = StationType(missing_value)
            if self.station_types[idx] == StationType.PICKUP:
                self.pickup_position = self.stations[idx]
        
        # Can we pickup or dropoff?
        can_pickup = int(not self.has_passenger and self.taxi_position == self.pickup_position)
        can_dropoff = int(self.has_passenger and near_dropoff and self.taxi_position in self.stations)
        
        # Update passenger position if we've picked them up
        if self.has_passenger:
            self.pickup_position = self.taxi_position
        
        # Determine target based on current state
        if not self.has_passenger:
            if self.pickup_position is not None:
                target = self.pickup_position
            else:
                # Find closest unknown station
                target = min(
                    (s for s, t in zip(self.stations, self.station_types) if t == StationType.UNKNOWN),
                    key=lambda pos: abs(pos[0] - self.taxi_position[0]) + abs(pos[1] - self.taxi_position[1]),
                    default=self.stations[0]  # Fallback if no unknown stations
                )
        else:
            if StationType.DROPOFF in self.station_types:
                dropoff_idx = self.station_types.index(StationType.DROPOFF)
                target = self.stations[dropoff_idx]
            else:
                # If we don't know the dropoff yet, explore unknown stations
                target = min(
                    (s for s, t in zip(self.stations, self.station_types) if t == StationType.UNKNOWN),
                    key=lambda pos: abs(pos[0] - self.taxi_position[0]) + abs(pos[1] - self.taxi_position[1]),
                    default=self.stations[0]
                )
        
        # Track visited states for exploration
        state_key = (*self.taxi_position, self.has_passenger)
        self.visited_states[state_key] += 1
        
        # Increment step counter
        self.steps += 1
        
        # Create feature vector for policy network
        features = [
            *obstacles,                     # Obstacles in four directions
            can_pickup,                     # Can we pick up now?
            can_dropoff,                    # Can we drop off now?
            target[0] - self.taxi_position[0],  # Relative x distance to target
            target[1] - self.taxi_position[1],  # Relative y distance to target
        ]
        
        # Additional info for debugging or reward shaping
        info = {
            "taxi_position": self.taxi_position,
            "has_passenger": self.has_passenger,
            "station_types": self.station_types,
            "pickup_position": self.pickup_position,
            "target": target,
            "can_pickup": can_pickup,
            "can_dropoff": can_dropoff,
            "steps": self.steps,
            "visited_count": self.visited_states[state_key]
        }
        
        return features, info
    
    def update_with_action(self, action: int):
        """Update internal state based on action taken"""
        self.last_action = Action(action)
        
        if action == Action.PICKUP and self.taxi_position == self.pickup_position:
            self.has_passenger = True
        elif action == Action.DROPOFF:
            self.has_passenger = False
    
    @property
    def feature_size(self):
        # Calculate feature size based on the observation processing
        return 8  # 4 obstacles + can_pickup + can_dropoff + 2 relative coordinates

# Actor network architecture (different from the original)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(ActorNetwork, self).__init__()
        # Three-layer network instead of two
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights with a different method
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # Using softmax to get action probabilities
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state, deterministic=False):
        """
        Select an action based on the current state
        Args:
            state: Current state features
            deterministic: If True, select the most probable action
        """
        with torch.no_grad():
            action_probs = self(state)
            
            if deterministic:
                # During testing/deployment we can choose the most likely action
                return torch.argmax(action_probs).item()
            else:
                # During training we sample from the distribution
                dist = torch.distributions.Categorical(action_probs)
                return dist.sample().item()

# This function shapes rewards to guide learning
def shape_reward(reward, prev_info, new_info):
    """Apply reward shaping to guide learning"""
    if reward == 50 - 0.1:  # Successful delivery
        reward = 50
    elif reward == -10.1:   # Incorrect pickup/dropoff
        reward = -30
    elif reward == -5.1:    # Hit obstacle
        reward = -20
    elif reward == -0.1:    # Regular movement
        reward = -0.1
    
    # Bonus for picking up passenger
    if not prev_info["has_passenger"] and new_info["has_passenger"]:
        reward += 15
    
    # Penalty for dropping passenger at wrong location
    if prev_info["has_passenger"] and not new_info["has_passenger"] and not prev_info["can_dropoff"]:
        reward -= 15
    
    # Bonus for discovering new station types
    unknown_before = sum(1 for t in prev_info["station_types"] if t == StationType.UNKNOWN)
    unknown_after = sum(1 for t in new_info["station_types"] if t == StationType.UNKNOWN)
    if unknown_before > unknown_after:
        reward += 10
    
    # Small bonus for getting closer to target
    prev_dist = manhattan_distance(prev_info["taxi_position"], prev_info["target"])
    new_dist = manhattan_distance(new_info["taxi_position"], new_info["target"])
    if prev_info["target"] == new_info["target"]:  # Only if target hasn't changed
        reward += 0.2 * (prev_dist - new_dist)
    
    return reward

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])