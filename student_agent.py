from taxi_model import EnvironmentProcessor, ActorNetwork
import torch
import os
import random
import numpy as np
from collections import deque, Counter

class TaxiAgent:
    def __init__(self, model_path='dqn_agent.pth'):
        """
        Initialize the Taxi Agent with a pre-trained model
        
        Args:
            model_path: Path to the saved model weights
        """
        self.env_processor = EnvironmentProcessor()
        
        # Check if we're using the old or new environment processor
        feature_size = 8  # Original feature size
        
        # Load model architecture with the correct input size
        self.model = ActorNetwork(
            input_dim=feature_size,
            output_dim=6,  # 6 possible actions
            hidden_dim=64   # Increased from original 8 to 64
        )
        
        # Load pre-trained weights if they exist
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # Support different model formats (normal dict or checkpoint dict)
                if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                    self.model.load_state_dict(checkpoint['q_network'])
                else:
                    self.model.load_state_dict(checkpoint)
                    
                self.model.eval()  # Set to evaluation mode
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using untrained model instead.")
        else:
            print(f"Warning: Model file '{model_path}' not found. Using untrained model.")
        
        # For detecting and preventing oscillations
        self.position_history = deque(maxlen=20)  # Track more positions
        self.action_history = deque(maxlen=20)  # Track more actions
        self.oscillation_count = 0
        self.severe_oscillation_count = 0  # Count for severe oscillations
        self.axis_diversity_count = 0  # Track how long we've been moving in same axis
        self.last_axis = None  # Track which axis we're moving along (0=NS, 1=EW)
        self.force_axis_change_after = 4  # Force axis change after this many steps on same axis
        
        # Set up a forced exploration sequence for escaping loops
        self.escape_mode = False
        self.escape_sequence = []
        self.escape_index = 0
        
        # Track direction frequencies to ensure diversity
        self.axis_counts = {0: 0, 1: 0}  # 0=NS axis, 1=EW axis
        
        # Track success/fail over time
        self.success_history = []
        self.steps_since_last_success = 0
    
    def get_action(self, obs):
        """
        Select an action based on the current observation
        
        Args:
            obs: The current environment observation
            
        Returns:
            int: The selected action (0-5)
        """
        # Process the observation to get state features
        state_full, info = self.env_processor.process_observation(obs)
        
        # Take only the first 8 features to match the model's input size
        state = state_full[:8]
        
        # Record current position for oscillation detection
        current_position = (info["taxi_position"], info["has_passenger"])
        self.position_history.append(current_position)
        
        # Check for oscillation
        is_oscillating = self._detect_oscillation()
        
        # Update oscillation count
        if is_oscillating:
            self.oscillation_count += 1
            # Enter escape mode if oscillation is persistent
            if self.oscillation_count >= 3 and not self.escape_mode:
                self._enter_escape_mode(info)
                
            # Check for severe oscillation (may need to force exit)
            if self.oscillation_count > 10:
                self.severe_oscillation_count += 1
                # If severely stuck, consider exit code 1 to signal problem
                if self.severe_oscillation_count > 5:
                    print("\n*** SEVERE OSCILLATION DETECTED ***")
                    print(f"Position history (last {len(self.position_history)} steps):")
                    for i, (pos, has_pass) in enumerate(self.position_history):
                        print(f"Step {i+1}: Position {pos} {'with' if has_pass else 'without'} passenger")
                    print("\nAction history:")
                    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
                    for i, action in enumerate(self.action_history):
                        print(f"Step {i+1}: {action_names[action]}")
                    
                    # If completely unable to solve, exit with code 1
                    if self.severe_oscillation_count > 15:
                        print("Agent unable to solve environment due to severe oscillation. Exiting.")
                        # Uncomment the next line to actually exit
                        # exit(1)
        else:
            self.oscillation_count = max(0, self.oscillation_count - 1)
            self.severe_oscillation_count = 0
            # Exit escape mode if oscillation has stopped
            if self.oscillation_count == 0:
                self.escape_mode = False
        
        # Use the model to get action probabilities
        with torch.no_grad():
            action_probs = self.model(state).cpu().numpy()
        
        # Special case handling
        if info["can_dropoff"]:
            # Always drop off when possible
            action = 5  # DROPOFF
            self.success_history.append(True)
            self.steps_since_last_success = 0
        elif info["can_pickup"] and not info["has_passenger"]:
            # Always pick up when possible and we don't have a passenger
            action = 4  # PICKUP
        elif self.escape_mode:
            # Use the pre-planned escape sequence
            action = self.escape_sequence[self.escape_index]
            self.escape_index = (self.escape_index + 1) % len(self.escape_sequence)
            
            # If we've completed the escape sequence, exit escape mode
            if self.escape_index == 0:
                self.escape_mode = False
        else:
            # Normal operation with advanced anti-oscillation
            
            # Sort actions by probability
            sorted_actions = action_probs.argsort()[::-1]
            
            # Check if we need to force axis diversity
            current_axis = self._get_axis_from_last_actions()
            
            if is_oscillating or self.axis_diversity_count >= self.force_axis_change_after:
                # Force a change of movement axis
                if current_axis == 0:  # If moving North-South
                    # Force East-West movement
                    candidate_actions = [a for a in sorted_actions if a in [2, 3]]  # East or West
                    if candidate_actions:
                        action = candidate_actions[0]
                    else:
                        action = random.choice([2, 3])  # Fallback to random E/W
                elif current_axis == 1:  # If moving East-West
                    # Force North-South movement
                    candidate_actions = [a for a in sorted_actions if a in [0, 1]]  # South or North
                    if candidate_actions:
                        action = candidate_actions[0]
                    else:
                        action = random.choice([0, 1])  # Fallback to random N/S
                else:
                    # If axis is unknown, try second best action
                    if self.oscillation_count > 5:
                        # For severe oscillation, pick a completely random movement
                        action = random.choice([0, 1, 2, 3])
                    else:
                        # Try second best action
                        action = sorted_actions[1] if len(sorted_actions) > 1 else sorted_actions[0]
                
                # Reset axis diversity counter
                self.axis_diversity_count = 0
            else:
                # Use the best action from the model
                action = sorted_actions[0]
                
                # Update axis diversity counter if continuing on same axis
                new_axis = self._get_action_axis(action)
                if new_axis is not None and new_axis == current_axis:
                    self.axis_diversity_count += 1
                else:
                    self.axis_diversity_count = 0
        
        # Validate actions - don't try pickup/dropoff inappropriately
        if action == 4 and not info["can_pickup"]:  # Trying PICKUP without passenger nearby
            # Choose a movement action toward the target instead
            action = self._choose_movement_action(info, avoid_axis=current_axis)
        elif action == 5 and not info["can_dropoff"]:  # Trying DROPOFF without being at destination
            # Choose a movement action toward the target instead
            action = self._choose_movement_action(info, avoid_axis=current_axis)
        
        # Remember this action for oscillation detection
        self.action_history.append(action)
        
        # Update axis counts for statistical tracking
        action_axis = self._get_action_axis(action)
        if action_axis is not None:
            self.axis_counts[action_axis] += 1
        
        # Update the processor with the selected action
        self.env_processor.update_with_action(action)
        
        # Increment step counter since last success
        self.steps_since_last_success += 1
        
        # Check if we're not making progress toward goal
        if self.steps_since_last_success > 1000:
            print("Warning: Agent hasn't succeeded in 1000 steps. May be stuck.")
            self.steps_since_last_success = 0
        
        return action
    
    def _get_action_axis(self, action):
        """Determine which axis an action moves along"""
        if action in [0, 1]:  # South, North
            return 0  # North-South axis
        elif action in [2, 3]:  # East, West
            return 1  # East-West axis
        return None  # Not a movement action
    
    def _get_axis_from_last_actions(self):
        """Determine which axis we've been moving along recently"""
        # Look at last few actions
        movement_actions = [a for a in self.action_history if a in [0, 1, 2, 3]]
        if not movement_actions:
            return None
            
        # Count actions by axis
        ns_count = sum(1 for a in movement_actions[-4:] if a in [0, 1])
        ew_count = sum(1 for a in movement_actions[-4:] if a in [2, 3])
        
        if ns_count > ew_count:
            return 0  # North-South dominant
        elif ew_count > ns_count:
            return 1  # East-West dominant
        return None  # No clear dominance
    
    def _enter_escape_mode(self, info):
        """Generate an escape sequence to break out of oscillation"""
        self.escape_mode = True
        self.escape_index = 0
        
        # Determine current movement axis
        current_axis = self._get_axis_from_last_actions()
        
        # Create a sequence that will definitely break the pattern
        if current_axis == 0:  # If stuck in North-South
            # Make an East-West-North-South pattern with randomness
            self.escape_sequence = [2, 3, 1, 0]  # E, W, N, S
        elif current_axis == 1:  # If stuck in East-West
            # Make a North-South-East-West pattern with randomness
            self.escape_sequence = [0, 1, 3, 2]  # S, N, W, E
        else:
            # Mix of all directions
            self.escape_sequence = [0, 2, 1, 3]  # S, E, N, W
        
        # Add randomness for severe oscillation
        if self.oscillation_count > 5:
            # Completely randomize the escape sequence
            random.shuffle(self.escape_sequence)
            # Make it longer for more exploration
            self.escape_sequence = self.escape_sequence * 2
            
    def _detect_oscillation(self):
        """Detect if the agent is stuck in an oscillation pattern"""
        if len(self.position_history) < 6:
            return False
        
        # Check for a pattern of length 2
        last_two = list(self.position_history)[-2:]
        prev_two = list(self.position_history)[-4:-2]
        if last_two == prev_two:
            return True
            
        # Check for a pattern of length 3
        if len(self.position_history) >= 6:
            last_three = list(self.position_history)[-3:]
            prev_three = list(self.position_history)[-6:-3]
            if last_three == prev_three:
                return True
                
        # Check for East-West or North-South dominance
        movement_actions = [a for a in self.action_history if a in [0, 1, 2, 3]]
        if len(movement_actions) >= 8:
            recent_actions = movement_actions[-8:]
            
            # Count directional actions
            ns_count = sum(1 for a in recent_actions if a in [0, 1])
            ew_count = sum(1 for a in recent_actions if a in [2, 3])
            
            # If heavily biased toward one axis (>75%), consider it oscillating
            if ns_count >= 6 or ew_count >= 6:
                return True
        
        return False
    
    def _choose_movement_action(self, info, avoid_axis=None):
        """Choose a movement action toward the target with axis diversity"""
        taxi_pos = info["taxi_position"]
        target = info["target"]
        
        # Calculate direction to target
        dx = target[0] - taxi_pos[0]
        dy = target[1] - taxi_pos[1]
        
        # Check if we can move in the preferred direction (no obstacles)
        obstacles = [info.get("obstacle_north", 0), info.get("obstacle_south", 0), 
                    info.get("obstacle_east", 0), info.get("obstacle_west", 0)]
        
        # Get valid actions (not blocked by obstacles)
        valid_actions = [i for i, blocked in enumerate(obstacles) if blocked == 0 and i < 4]
        if not valid_actions:
            return random.randint(0, 3)  # All blocked, pick random
            
        # If we need to avoid an axis, filter actions
        if avoid_axis is not None:
            if avoid_axis == 0:  # Avoid North-South
                filtered_actions = [a for a in valid_actions if a in [2, 3]]
                if filtered_actions:
                    valid_actions = filtered_actions
            elif avoid_axis == 1:  # Avoid East-West
                filtered_actions = [a for a in valid_actions if a in [0, 1]]
                if filtered_actions:
                    valid_actions = filtered_actions
        
        # Choose based on direction to target
        preferred_actions = []
        
        if dx > 0 and 0 in valid_actions:  # Need to go south
            preferred_actions.append(0)
        if dx < 0 and 1 in valid_actions:  # Need to go north
            preferred_actions.append(1)
        if dy > 0 and 2 in valid_actions:  # Need to go east
            preferred_actions.append(2)
        if dy < 0 and 3 in valid_actions:  # Need to go west
            preferred_actions.append(3)
            
        # If we have preferred actions, choose one
        if preferred_actions:
            return random.choice(preferred_actions)
        else:
            # No preferred action, choose a random valid one
            return random.choice(valid_actions)

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