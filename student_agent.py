from taxi_model import EnvironmentProcessor, ActorNetwork
import torch
import os

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
    
    def get_action(self, obs):
        """
        Select an action based on the current observation
        
        Args:
            obs: The current environment observation
            
        Returns:
            int: The selected action (0-5)
        """
        # Process the observation to get state features
        state, _ = self.env_processor.process_observation(obs)
        
        # Use the model to get the best action
        with torch.no_grad():
            action = self.model.get_action(state, deterministic=True)
        
        # Update the processor with the selected action
        self.env_processor.update_with_action(action)
        
        return action

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