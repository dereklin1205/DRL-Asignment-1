# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque 


# ======================== DRQN ========================

# ---------------- State Parsing ----------------

def parse_state(obs):
    """
    Parses the raw state tuple into a structured tuple.
    """
    padded_stations = [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ]
    (
        taxi_row,
        taxi_col,
        padded_stations[0][0],
        padded_stations[0][1],
        padded_stations[1][0],
        padded_stations[1][1],
        padded_stations[2][0],
        padded_stations[2][1],
        padded_stations[3][0],
        padded_stations[3][1],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ) = obs

    lst = []
    for station in padded_stations:
        station_row, station_col = station
        taxi_distance = abs(taxi_row - station_row) + abs(taxi_col - station_col)
        lst.append(taxi_distance)

    return (
        taxi_row,
        taxi_col,
        padded_stations[0][0],
        padded_stations[0][1],
        padded_stations[1][0],
        padded_stations[1][1],
        padded_stations[2][0],
        padded_stations[2][1],
        padded_stations[3][0],
        padded_stations[3][1],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    )

# ---------------- DRQN Network ----------------

class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_hidden_size, action_size):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, action_size)
    
    def forward(self, x, hidden_state=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        x = torch.relu(self.fc1(x))
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
            hidden_state = (h0, c0)
        x, hidden_state = self.lstm(x, hidden_state)
        q_values = self.fc2(x)  # shape: (batch, seq_len, action_size)
        return q_values, hidden_state

class DRQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lstm_hidden_size=64,
                 gamma=0.99, learning_rate=0.001, tau=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995, chunk_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.chunk_size = chunk_size  # for truncated BPTT
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
        self.policy_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def reset_hidden_state(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.policy_net.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.policy_net.lstm.hidden_size).to(self.device)
        return (h0, c0)

    def act(self, state, hidden_state):
        """
        Choose an action using an epsilon-greedy policy.
        state: tuple or numpy array of shape (state_size,)
        hidden_state: current hidden state tuple for the LSTM.
        Returns: action (int) and updated hidden state.
        """
        # Expand dimensions: (1, 1, state_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
            # Update hidden state with a forward pass
            with torch.no_grad():
                _, hidden_state = self.policy_net(state_tensor, hidden_state)
            return action, hidden_state
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values, hidden_state = self.policy_net(state_tensor, hidden_state)
            self.policy_net.train()
            action = torch.argmax(q_values, dim=2).item()
            return action, hidden_state

    def learn_from_episode(self, episode):
        """
        Learns from a single episode using truncated BPTT.
        Each element in 'episode' is a tuple: (state, action, reward, next_state, done)
        """
        seq_len = len(episode)
        if seq_len < 1:
            return 0
        
        # Prepare arrays
        states = np.array([transition[0] for transition in episode], dtype=np.float32)
        actions = np.array([transition[1] for transition in episode], dtype=np.int64)
        rewards = np.array([transition[2] for transition in episode], dtype=np.float32)
        next_states = np.array([transition[3] for transition in episode], dtype=np.float32)
        dones = np.array([transition[4] for transition in episode], dtype=np.float32)
        
        loss_total = 0.0
        num_chunks = 0
        
        # Initialize hidden state for the sequence
        hidden_state = self.reset_hidden_state()
        
        # Process episode in chunks
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_states = torch.FloatTensor(states[start:end]).unsqueeze(0).to(self.device)      # (1, chunk_len, state_size)
            chunk_actions = torch.LongTensor(actions[start:end]).unsqueeze(0).to(self.device)       # (1, chunk_len)
            chunk_rewards = torch.FloatTensor(rewards[start:end]).unsqueeze(0).to(self.device)      # (1, chunk_len)
            chunk_next_states = torch.FloatTensor(next_states[start:end]).unsqueeze(0).to(self.device)
            chunk_dones = torch.FloatTensor(dones[start:end]).unsqueeze(0).to(self.device)
            
            # Forward pass through policy network for current chunk
            q_values, hidden_state = self.policy_net(chunk_states, hidden_state)
            # Select Q-values corresponding to actions taken
            q_values = q_values.gather(2, chunk_actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute target Q-values using target network.
            # For target network we use a fresh hidden state to avoid backprop through time
            with torch.no_grad():
                target_hidden = self.reset_hidden_state()
                q_next, _ = self.target_net(chunk_next_states, target_hidden)
                max_q_next = q_next.max(dim=2)[0]
                targets = chunk_rewards + self.gamma * max_q_next * (1 - chunk_dones)
            
            loss = nn.MSELoss()(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.optimizer.step()
            
            loss_total += loss.item()
            num_chunks += 1
            
            # Detach hidden state to truncate backpropagation
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
        
        self.soft_update()
        return loss_total / num_chunks if num_chunks > 0 else 0
    
    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'policy_model_state_dict': self.policy_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# ======================== MAIN FUNCTION ========================

def get_action(obs):
    STATE_SIZE = 20
    ACTION_SIZE = 6 

    if not hasattr(get_action, "agent"):
        get_action.agent = DRQNAgent(STATE_SIZE, ACTION_SIZE)
        get_action.agent.load("drqn_final.pt")
        get_action.hidden_state = get_action.agent.reset_hidden_state()

    # Parse and convert observation
    parsed_obs = parse_state(obs)
    state = np.array(parsed_obs, dtype=np.float32)
    
    if state.shape[0] != STATE_SIZE:
        raise ValueError(f"Expected state size {STATE_SIZE}, got {state.shape[0]}")
    
    # Get action with epsilon = 0 (no exploration during evaluation)
    epsilon_backup = get_action.agent.epsilon
    get_action.agent.epsilon = 0  # Turn off exploration during evaluation
    action, get_action.hidden_state = get_action.agent.act(state, get_action.hidden_state)
    get_action.agent.epsilon = epsilon_backup  # Restore epsilon
    
    return action