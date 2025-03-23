import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from simple_custom_taxi_env import SimpleTaxiEnv

# ========================= STATE PARSER =========================
def parse_state(obs,passenger_on):
    """
    Convert the environment's 16D observation into a 16D representation using (dx, dy).
    For each station:
      dx_i = station_i_row - taxi_row
      dy_i = station_i_col - taxi_col
    """
    (
        taxi_row,
        taxi_col,
        st0_row, st0_col,
        st1_row, st1_col,
        st2_row, st2_col,
        st3_row, st3_col,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ) = obs
    
    dx0 = st0_row - taxi_row
    dy0 = st0_col - taxi_col
    dx1 = st1_row - taxi_row
    dy1 = st1_col - taxi_col
    dx2 = st2_row - taxi_row
    dy2 = st2_col - taxi_col
    dx3 = st3_row - taxi_row
    dy3 = st3_col - taxi_col

    state_tuple = (
        dx0, dy0,
        dx1, dy1,
        dx2, dy2,
        dx3, dy3,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
        passenger_on
    )
    return np.array(state_tuple, dtype=np.float32)


# ========================= REPLAY BUFFER =========================
class ReplayBuffer:
    """
    A simple replay buffer storing (state, action, reward, next_state, done).
    Uses a deque with fixed capacity.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


# ========================= DQN NETWORK =========================
class DQNNetwork(nn.Module):
    """
    A simple feedforward network with two hidden layers.
    Takes as input the state (16D) and outputs Q-values for each action (6).
    """
    def __init__(self, state_dim=16, action_dim=6, hidden_dim=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# ========================= DQN AGENT =========================
class DQNAgent:
    """
    A DQN Agent with:
    - Q-network (policy_net)
    - Target network (target_net)
    - Replay buffer
    - Epsilon-greedy policy
    - Train loop with target net soft or hard updates
    """
    def __init__(
        self,
        state_dim=15,
        action_dim=6,
        hidden_dim=64,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        tau=0.01,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau  # for soft update or can do periodic hard updates
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def act(self, state):
        """
        Epsilon-greedy action selection. state shape: (16,)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            action = q_values.argmax(dim=1).item()
            return action

    def step(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample from replay buffer and update Q-network.
        Decouple epsilon decay from here, so we do once per episode in training loop.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Get current Q
        q_values = self.policy_net(states_t)
        current_q = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next Q from target net
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        # Loss = MSE
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update (tau) or Hard update
        self.soft_update()

    def soft_update(self):
        """
        Soft update target network:
        target = tau*policy + (1-tau)*target
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1.0 - self.tau)*target_param.data)

    def save(self, filename="dqn_model.pt"):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict()
        }, filename)

    def load(self, filename="dqn_model.pt"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])


# ========================= GET ACTION FOR TESTING =========================
def get_action(obs):
    """
    This function is called by the environment to get an action from our
    loaded DQN. We parse the obs (16D), then do a forward pass with epsilon=0.
    """
    # We'll store the agent as a static attribute so we only load once.
    if not hasattr(get_action, "agent"):
        get_action.agent = DQNAgent()
        get_action.agent.load("dqn_model.pt")
        get_action.agent.epsilon = 0.0  # No exploration at test time
    get_action.prev_state = None
    get_action.prev_action = None
    if get_action.prev_state is not None and get_action.prev_action is not None:
        prev_action = get_action.prev_action
        if prev_action == 4 and obs[13] == 1 and ((state[0] == 0 and state[1] == 0)or (state[2] == 0 and state[3] == 0)or (state[4] == 0 and state[5] == 0)or (state[6] == 0 and state[7] == 0)):
            passenger_on = 1
        elif prev_action == 5:
            passenger_on = 0
    # Convert raw 16D obs to final (16,) state
    state = parse_state(obs)
    action = get_action.agent.act(state)
    return action


# ========================= TRAIN FUNCTION =========================
def train(env, agent, num_episodes=500):
    """
    Train the DQN over multiple episodes. Decay epsilon once per episode.
    """
    all_rewards = []
    passenger_on = 0
    for ep in range(num_episodes):
        raw_obs, _ = env.reset()
        state = parse_state(raw_obs, passenger_on)
        done = False
        total_reward = 0.0
        passenger_on = 0
        while not done:
            action = agent.act(state)
            raw_next_obs, reward, done, _ = env.step(action)
            next_state = parse_state(raw_next_obs,passenger_on)
            if reward >=0 and action == 4 :
                passenger_on = 1
            if action == 5:
                passenger_on = 0
            agent.step(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        # Decay epsilon once per episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        all_rewards.append(total_reward)
        if (ep + 1) % 50 == 0:
            avg_50 = np.mean(all_rewards[-50:])
            print(f"Episode {ep+1}/{num_episodes} | Epsilon={agent.epsilon:.3f} | AvgReward(Last50)={avg_50:.2f}")

    agent.save("dqn_model.pt")
    return all_rewards


# ========================= MAIN =========================
if __name__ == "__main__":
    env = SimpleTaxiEnv(5,5000)
    agent = DQNAgent(
        state_dim=15,   # (taxi_row, taxi_col, dx0,dy0, ..., passenger_look, destination_look)
        action_dim=6,
        hidden_dim=64,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        tau=0.01,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999
    )
    train(env, agent, num_episodes=20000)
