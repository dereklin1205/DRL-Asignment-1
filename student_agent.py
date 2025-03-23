from taxi_model import EnvironmentProcessor, ActorNetwork
import torch
import os
import random
from collections import deque, Counter

class TaxiAgent:
    def __init__(self, model_path='dqn_agent.pth'):
        """
        Initialize the Taxi Agent with a pre-trained model
        
        Args:
            model_path: Path to the saved model weights
        """
        self.env_processor = EnvironmentProcessor()
        
        # Load model architecture
        self.model = ActorNetwork(
            input_dim=self.env_processor.feature_size,
            output_dim=6,  # 6 possible actions
            hidden_dim=64   # Increased from original 8 to 64
        )
        
        # Load pre-trained weights if they exist
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Support different model formats (normal dict or checkpoint dict)
            if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                self.model.load_state_dict(checkpoint['q_network'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()  # Set to evaluation mode
        else:
            print(f"Warning: Model file '{model_path}' not found. Using untrained model.")
        
        # For detecting and preventing oscillations
        self.state_history = deque(maxlen=10)
        self.action_history = deque(maxlen=10)
        self.oscillation_count = 0
        self.break_pattern = False
    
    def get_action(self, obs):
        """
        Select an action based on the current observation
        
        Args:
            obs: The current environment observation
            
        Returns:
            int: The selected action (0-5)
        """
        # Process the observation to get state features
        state, info = self.env_processor.process_observation(obs)
        
        # Record current state for oscillation detection
        current_state = (info["taxi_position"], info["has_passenger"])
        self.state_history.append(current_state)
        
        # Check for oscillation
        is_oscillating = self._detect_oscillation()
        if is_oscillating:
            self.oscillation_count += 1
            self.break_pattern = True
        else:
            self.oscillation_count = max(0, self.oscillation_count - 1)
            if self.oscillation_count == 0:
                self.break_pattern = False
        
        # Use the model to get action probabilities
        with torch.no_grad():
            action_probs = self.model(state).cpu().numpy()
        
        # Special case handling
        if info["can_dropoff"]:
            # Always drop off when possible
            action = 5  # DROPOFF
        elif info["can_pickup"] and not info["has_passenger"]:
            # Always pick up when possible and we don't have a passenger
            action = 4  # PICKUP
        elif self.break_pattern:
            # Breaking oscillation pattern
            
            # If oscillating severely, take a more drastic approach
            if self.oscillation_count >= 3:
                # Get sorted actions by probability
                sorted_actions = action_probs.argsort()[::-1]
                
                # Get counts of recent actions to avoid repeating them
                action_counts = Counter(self.action_history)
                
                # Choose the least used action among the top 3 highest value actions
                candidate_actions = sorted_actions[:3]
                action = min(candidate_actions, key=lambda a: action_counts.get(a, 0))
                
                # If still in a severe oscillation, sometimes take a completely random action
                if self.oscillation_count > 5 and random.random() < 0.3:
                    movement_actions = [0, 1, 2, 3]  # Only consider movement actions
                    action = random.choice(movement_actions)
            else:
                # Just take the second-best action to break mild oscillation
                sorted_actions = action_probs.argsort()[::-1]
                action = sorted_actions[1]  # Second-best action
        else:
            # Normal operation - use the model's best action
            action = action_probs.argmax()
        
        # Validate actions - don't try pickup/dropoff inappropriately
        if action == 4 and not info["can_pickup"]:  # Trying PICKUP without passenger nearby
            # Choose a movement action toward the target instead
            action = self._choose_movement_action(info)
        elif action == 5 and not info["can_dropoff"]:  # Trying DROPOFF without being at destination
            # Choose a movement action toward the target instead
            action = self._choose_movement_action(info)
        
        # Remember this action for oscillation detection
        self.action_history.append(action)
        
        # Update the processor with the selected action
        self.env_processor.update_with_action(action)
        
        return action
    
    def _detect_oscillation(self):
        """Detect if the agent is stuck in an oscillation pattern"""
        if len(self.state_history) < 6:
            return False
        
        # Check for a pattern of length 2
        last_two = list(self.state_history)[-2:]
        prev_two = list(self.state_history)[-4:-2]
        if last_two == prev_two:
            return True
            
        # Check for a pattern of length 3
        if len(self.state_history) >= 6:
            last_three = list(self.state_history)[-3:]
            prev_three = list(self.state_history)[-6:-3]
            if last_three == prev_three:
                return True
        
        return False
    
    def _choose_movement_action(self, info):
        """Choose a movement action toward the target"""
        taxi_pos = info["taxi_position"]
        target = info["target"]
        
        # Calculate direction to target
        dx = target[0] - taxi_pos[0]
        dy = target[1] - taxi_pos[1]
        
        # Check if we can move in the preferred direction (no obstacles)
        obstacles = [info.get("obstacle_north", 0), info.get("obstacle_south", 0), 
                    info.get("obstacle_east", 0), info.get("obstacle_west", 0)]
        
        # Prioritize larger distance axis first
        if abs(dx) > abs(dy):
            if dx > 0 and obstacles[0] == 0:  # Can move south
                return 0  # SOUTH
            elif dx < 0 and obstacles[1] == 0:  # Can move north
                return 1  # NORTH
            elif dy > 0 and obstacles[2] == 0:  # Can move east
                return 2  # EAST
            elif dy < 0 and obstacles[3] == 0:  # Can move west
                return 3  # WEST
        else:
            if dy > 0 and obstacles[2] == 0:  # Can move east
                return 2  # EAST
            elif dy < 0 and obstacles[3] == 0:  # Can move west
                return 3  # WEST
            elif dx > 0 and obstacles[0] == 0:  # Can move south
                return 0  # SOUTH
            elif dx < 0 and obstacles[1] == 0:  # Can move north
                return 1  # NORTH
        
        # If all preferred directions are blocked, choose a random valid direction
        valid_actions = [i for i, blocked in enumerate(obstacles) if blocked == 0 and i < 4]
        if valid_actions:
            return random.choice(valid_actions)
        else:
            # All directions blocked (unlikely), just pick a random movement
            return random.randint(0, 3)

# Create agent instance
agent = TaxiAgent('dqn_agent.pth')

# Implement the required get_action function for the environment
def get_action(obs):
    """
    Interface function called by the environment
    
    Args:
        obs: Current observation from the environment
        
    Returns:
        int: Selected action (0-5)
    """
    return agent.get_action(obs)