import numpy as np
import random
from tqdm import *
import torch
import torch.nn as nn
import torch.optim as optim
from simple_custom_taxi_env import SimpleTaxiEnv

# ======================== PRIORITIZED EXPERIENCE REPLAY ========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** -beta
        weights /= weights.max()
        
        batch = list(zip(*samples))  # batch = [states, actions, rewards, next_states, dones]
        return batch, indices, torch.FloatTensor(weights).to("cuda" if torch.cuda.is_available() else "cpu")

    def update_priorities(self, batch_indices, td_errors):
        for idx, error in zip(batch_indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5
    
    def __len__(self):
        return len(self.buffer)


# ======================== GET STATE (Parse 20D) ========================
def get_state(obs):
    """
    Convert the environment's raw observation (16D) into a custom 20D representation.
    Adjust as needed to ensure consistent shapes.
    """
    stations = [[0, 0], [0, 0], [0, 0], [0, 0]]
    (
        taxi_row,
        taxi_col,
        stations[0][0], stations[0][1],
        stations[1][0], stations[1][1],
        stations[2][0], stations[2][1],
        stations[3][0], stations[3][1],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ) = obs

    # Calculate manhattan distances from taxi to each station
    station_dis = []
    for station in stations:
        station_dis.append(abs(station[0] - taxi_row) + abs(station[1] - taxi_col))

    # Return a 20D tuple
    return (
        taxi_row,
        taxi_col,
        stations[0][0], stations[0][1],
        station_dis[0],
        stations[1][0], stations[1][1],
        station_dis[1],
        stations[2][0], stations[2][1],
        station_dis[2],
        stations[3][0], stations[3][1],
        station_dis[3],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    )


# ======================== PURE DRQN (No Dueling) ========================
class DRQN(nn.Module):
    """
    Expects an input of shape (batch_size, input_size).
    We'll do:
      1) fc1 => (batch_size, hidden_size)
      2) unsqueeze => (batch_size, 1, hidden_size)
      3) LSTM => (batch_size, 1, lstm_hidden_size)
      4) final fc2 => (batch_size, 1, action_size)
    """
    def __init__(self, input_size, hidden_size, lstm_hidden_size, action_size):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, action_size)

        self.hidden_size = lstm_hidden_size
    
    def forward(self, x, hidden_state=None):
        # x shape: (batch_size, input_size)
        x = torch.relu(self.fc1(x))     # => (batch_size, hidden_size)
        x = x.unsqueeze(1)             # => (batch_size, 1, hidden_size)

        if hidden_state is None:
            batch_size = x.size(0)
            device = x.device
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            hidden_state = (h0, c0)

        x, hidden_state = self.lstm(x, hidden_state)
        # => x shape: (batch_size, 1, lstm_hidden_size)

        x = self.fc2(x)  # => (batch_size, 1, action_size)

        return x, hidden_state


# ======================== DRQN AGENT ========================
class DRQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lstm_hidden_size=64,
                 gamma=0.99, learning_rate=0.001, tau=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, buffer_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()

    def reset_hidden_state(self):
        """
        We'll store the hidden state as (h, c), each shaped (1, batch_size=1, hidden_size).
        """
        h0 = torch.zeros(1, 1, self.policy_net.hidden_size).to(self.device)
        c0 = torch.zeros(1, 1, self.policy_net.hidden_size).to(self.device)
        return (h0, c0)

    def act(self, state, hidden_state):
        """
        state shape: (state_size,) => (1, state_size)
        => forward => (1, 1, action_size)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  
        
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            # do a forward pass to update hidden_state
            with torch.no_grad():
                _, hidden_state = self.policy_net(state_t, hidden_state)
        else:
            with torch.no_grad():
                q_values, hidden_state = self.policy_net(state_t, hidden_state)
                # q_values => (1, 1, action_size)
                action = q_values.argmax(dim=2).item()
        return action, hidden_state

    def learn(self, batch_size=64, beta=0.4):
        if len(self.memory) < batch_size:
            return
        
        batch, indices, weights = self.memory.sample(batch_size, beta)
        # batch => [states, actions, rewards, next_states, dones]
        states, actions, rewards, next_states, dones = batch
        
        states_t = torch.FloatTensor(states).to(self.device)      
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        q_values, _ = self.policy_net(states_t)              # => (batch_size, 1, action_size)
        next_q_values, _ = self.target_net(next_states_t)    # => (batch_size, 1, action_size)

        # max along actions => shape (batch_size, 1) => then squeeze => (batch_size,)
        max_next_q = next_q_values.max(dim=2)[0].squeeze(1)
        target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        actions_t = torch.LongTensor(actions).to(self.device)
        predicted_q = q_values.squeeze(1).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        td_error = target_q - predicted_q
        loss = (weights * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ======================== GET ACTION ========================
def get_action(obs):
    """
    Called at test time by the environment.
    We'll parse the obs into 20D, load the model if needed,
    run the forward pass with epsilon=0, etc.
    """
    STATE_SIZE = 20
    ACTION_SIZE = 6
    
    if not hasattr(get_action, "agent"):
        get_action.agent = DRQNAgent(STATE_SIZE, ACTION_SIZE)
        get_action.agent.load("drqn_final.pt")
        get_action.hidden_state = get_action.agent.reset_hidden_state()
        get_action.agent.epsilon = 0.0  # Turn off exploration at test

    # Convert env obs (16D) to 20D
    state_20d = get_state(obs)
    state_t = np.array(state_20d, dtype=np.float32)

    action, get_action.hidden_state = get_action.agent.act(state_t, get_action.hidden_state)
    return action


# ======================== TRAINING LOOP ========================
def train(env, agent, num_episodes=500):
    for episode in trange(num_episodes):
        raw_obs, _ = env.reset()
        obs = get_state(raw_obs)  # convert 16D to 20D
        hidden_state = agent.reset_hidden_state()
        done = False

        total_reward = 0.0
        rewards = np.array([])
        while not done:
            action, hidden_state = agent.act(obs, hidden_state)
            raw_next_obs, reward, done, _, _ = env.step(action)
            next_obs = get_state(raw_next_obs)

            agent.memory.add(obs, action, reward, next_obs, done)
            agent.learn()

            obs = next_obs
            total_reward += reward
            rewards = np.append(rewards, reward)
        if (episode + 1) % 50 == 0:
            print(f"[Episode {episode+1}/{num_episodes}] Reward={rewards[-100:].mean:.2f}, Epsilon={agent.epsilon:.3f}")

    agent.save("drqn_final.pt")


# ======================== MAIN ========================
if __name__ == "__main__":
    env = SimpleTaxiEnv()
    state_size = 20  # after parsing
    action_size = 6
    agent = DRQNAgent(state_size, action_size)
    train(env, agent, num_episodes=500)
