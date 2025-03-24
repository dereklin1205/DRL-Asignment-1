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

# Improved Feature processing to avoid oscillation
class EnvironmentProcessor:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.has_passenger = False
        self.station_types = [StationType.UNKNOWN] * 4
        self.pickup_position = None
        self.dropoff_position = None
        self.last_action = None
        self.visited_states = defaultdict(int)
        self.steps = 0
        self.last_positions = []  # Track recent positions to detect oscillation
        self.max_history = 10     # Number of positions to remember
    
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
        if near_pickup and self.taxi_position in self.stations:
            station_idx = self.stations.index(self.taxi_position)
            if self.pickup_position is None or self.pickup_position != self.taxi_position:
                self.pickup_position = self.taxi_position
                self.station_types[station_idx] = StationType.PICKUP
        
        if near_dropoff and self.taxi_position in self.stations:
            station_idx = self.stations.index(self.taxi_position)
            if self.dropoff_position is None or self.dropoff_position != self.taxi_position:
                self.dropoff_position = self.taxi_position
                self.station_types[station_idx] = StationType.DROPOFF
        
        # If at a station but not pickup or dropoff, mark as empty
        if self.taxi_position in self.stations:
            idx = self.stations.index(self.taxi_position)
            if not near_pickup and not near_dropoff and self.station_types[idx] == StationType.UNKNOWN:
                self.station_types[idx] = StationType.EMPTY
        
        # Infer unknown stations based on what we know
        known_count = sum(t != StationType.UNKNOWN for t in self.station_types)
        
        if known_count == 3:
            # If we know 3 stations, we can infer the 4th
            idx = self.station_types.index(StationType.UNKNOWN)
            # Calculate what the unknown station must be based on what we know
            pickup_count = sum(1 for t in self.station_types if t == StationType.PICKUP)
            dropoff_count = sum(1 for t in self.station_types if t == StationType.DROPOFF)
            empty_count = sum(1 for t in self.station_types if t == StationType.EMPTY)
            
            # Determine the missing station type
            if pickup_count == 0:
                self.station_types[idx] = StationType.PICKUP
                self.pickup_position = self.stations[idx]
            elif dropoff_count == 0:
                self.station_types[idx] = StationType.DROPOFF
                self.dropoff_position = self.stations[idx]
            else:
                self.station_types[idx] = StationType.EMPTY
        
        # Determine if we can pickup or dropoff
        can_pickup = 0
        can_dropoff = 0
        
        if not self.has_passenger and self.taxi_position == self.pickup_position:
            can_pickup = 1
            
        if self.has_passenger and self.taxi_position == self.dropoff_position:
            can_dropoff = 1
        
        # Update position history to detect oscillation
        self.last_positions.append((self.taxi_position, self.has_passenger))
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)
        
        # Check for oscillation pattern
        is_oscillating = self._detect_oscillation()
        
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
            if self.dropoff_position is not None:
                target = self.dropoff_position
            else:
                # If we don't know the dropoff yet, explore stations we haven't visited
                unvisited_stations = [s for s, t in zip(self.stations, self.station_types) 
                                      if t == StationType.UNKNOWN or t == StationType.DROPOFF]
                
                if unvisited_stations:
                    target = min(
                        unvisited_stations,
                        key=lambda pos: abs(pos[0] - self.taxi_position[0]) + abs(pos[1] - self.taxi_position[1])
                    )
                else:
                    # If all stations are known, prioritize the dropoff
                    dropoff_stations = [s for s, t in zip(self.stations, self.station_types) 
                                        if t == StationType.DROPOFF]
                    if dropoff_stations:
                        target = dropoff_stations[0]
                    else:
                        target = self.stations[0]  # Fallback
        
        # Track visited states for exploration
        state_key = (*self.taxi_position, self.has_passenger)
        self.visited_states[state_key] += 1
        
        # Increment step counter
        self.steps += 1
        
        # Create better feature vector for policy network
        features = [
            *obstacles,                     # Obstacles in four directions
            can_pickup,                     # Can we pick up now?
            can_dropoff,                    # Can we drop off now?
            target[0] - self.taxi_position[0],  # Relative x distance to target
            target[1] - self.taxi_position[1],  # Relative y distance to target
            int(self.has_passenger),        # Whether we have a passenger or not
            int(is_oscillating),            # Flag for oscillation detection
            self.visited_states[state_key]  # How many times we've been in this state
        ]
        
        # Additional info for debugging or reward shaping
        info = {
            "taxi_position": self.taxi_position,
            "has_passenger": self.has_passenger,
            "station_types": self.station_types,
            "pickup_position": self.pickup_position,
            "dropoff_position": self.dropoff_position,
            "target": target,
            "can_pickup": can_pickup,
            "can_dropoff": can_dropoff,
            "steps": self.steps,
            "visited_count": self.visited_states[state_key],
            "is_oscillating": is_oscillating,
            "obstacle_north": obstacles[0],
            "obstacle_south": obstacles[1],
            "obstacle_east": obstacles[2],
            "obstacle_west": obstacles[3],
            "last_action": self.last_action
        }
        
        return features, info
    
    def _detect_oscillation(self):
        """Detect if the agent is oscillating between the same positions"""
        if len(self.last_positions) < 6:  # Need sufficient history
            return False
            
        # Check for a repeating pattern
        for pattern_length in range(2, 4):  # Check for patterns of length 2 and 3
            if len(self.last_positions) >= pattern_length * 2:
                # Get the most recent positions for the pattern
                recent = self.last_positions[-pattern_length:]
                previous = self.last_positions[-(pattern_length*2):-pattern_length]
                
                if recent == previous:
                    return True
                    
        return False
    
    def update_with_action(self, action: int):
        """Update internal state based on action taken"""
        self.last_action = Action(action)
        
        if action == Action.PICKUP and self.taxi_position == self.pickup_position:
            self.has_passenger = True
        elif action == Action.DROPOFF and self.taxi_position == self.dropoff_position:
            self.has_passenger = False
    
    @property
    def feature_size(self):
        # Updated feature size based on the observation processing
        return 11  # 4 obstacles + can_pickup + can_dropoff + 2 relative coordinates + has_passenger + is_oscillating + visited_count

# Actor network architecture (different from the original)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
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
                # During testing/deployment we choose the most likely action
                return torch.argmax(action_probs).item()
            else:
                # During training we sample from the distribution
                dist = torch.distributions.Categorical(action_probs)
                return dist.sample().item()

# Improved reward shaping function to discourage oscillation
def shape_reward(reward, prev_info, new_info):
    """Apply reward shaping to guide learning"""
    # Base reward from the environment
    if reward == 50 - 0.1:  # Successful delivery
        reward = 50
    elif reward == -10.1:   # Incorrect pickup/dropoff
        reward = -30
    elif reward == -5.1:    # Hit obstacle
        reward = -20
    else:  # Regular movement
        reward = -0.1
    
    # Bonus for picking up passenger correctly
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
    
    # Small bonus for getting closer to target when appropriate
    prev_dist = manhattan_distance(prev_info["taxi_position"], prev_info["target"])
    new_dist = manhattan_distance(new_info["taxi_position"], new_info["target"])
    if prev_info["target"] == new_info["target"]:  # Only if target hasn't changed
        reward += 0.5 * (prev_dist - new_dist)
    
    # Severe penalty for oscillating behavior
    if new_info.get("is_oscillating", False):
        reward -= 5.0  # Strong penalty for oscillation
    
    # Penalty for revisiting states too often (encourages exploration)
    if new_info["visited_count"] > 3:
        reward -= 0.5 * min(new_info["visited_count"], 10)  # Cap the penalty
        
    # Larger penalty for invalid pickup/dropoff attempts
    if prev_info.get("last_action") == Action.PICKUP and not prev_info.get("can_pickup", False):
        reward -= 10
    if prev_info.get("last_action") == Action.DROPOFF and not prev_info.get("can_dropoff", False):
        reward -= 10
    
    return reward

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])