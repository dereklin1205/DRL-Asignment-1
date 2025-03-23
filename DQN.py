import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from simple_custom_taxi_env_ import SimpleTaxiEnv

# ------------------ Constants ------------------
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5

STATE_SIZE = 6  # We now want 21 features in parse_state
ACTION_SIZE = 6

# ================== parse_state with 21 dims ================== #
def parse_state(obs, passenger_on):
    """
    Example parse that returns 21D state:
     1) 20 dims from environment (taxi_row, station dist, obstacles, etc.)
     2) plus 1 extra dim: passenger_on
    """
    (
        taxi_row,
        taxi_col,
        st0_row,
        st0_col,
        st1_row,
        st1_col,
        st2_row,
        st2_col,
        st3_row,
        st3_col,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
    ) = obs

    # Example distances:
    dist0 = abs(taxi_row - st0_row) + abs(taxi_col - st0_col)
    dist1 = abs(taxi_row - st1_row) + abs(taxi_col - st1_col)
    dist2 = abs(taxi_row - st2_row) + abs(taxi_col - st2_col)
    dist3 = abs(taxi_row - st3_row) + abs(taxi_col - st3_col)

    # Here we just do a simple 20D
    twenty_dims = (
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
    )  # 20 dims

    # Add passenger_on => total 21D
    twenty_one_dims = twenty_dims
    return twenty_one_dims


# ================== DRQN Network ================== #
class DRQN(nn.Module):
    """
    Simple DRQN:
      fc1 -> LSTM -> fc2
    Expects input shape (batch, seq_len, input_size).
    """
    def __init__(self, input_size, hidden_size=64, lstm_hidden_size=64, action_size=6):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.fc2 = nn.Linear(lstm_hidden_size, action_size)
        self.hidden_size = lstm_hidden_size

    def forward(self, x, hidden_state=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        x = torch.relu(self.fc1(x))

        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=x.device)
            hidden_state = (h0, c0)

        x, hidden_state = self.lstm(x, hidden_state)  # => (batch, seq_len, lstm_hidden_size)
        q_values = self.fc2(x)                       # => (batch, seq_len, action_size)
        return q_values, hidden_state


# ================== DRQN Agent ================== #
class DRQNAgent:
    def __init__(
        self,
        state_size=21,
        action_size=6,
        hidden_size=64,
        lstm_hidden_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        chunk_size=32
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.chunk_size = chunk_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net = DRQN(state_size, hidden_size, lstm_hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def reset_hidden_state(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.policy_net.hidden_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.policy_net.hidden_size).to(self.device)
        return (h0, c0)

    def act(self, state, hidden_state):
        """
        state => shape (1, state_size)
        => we expand to (1, 1, state_size) for LSTM
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1,1,21)
        if random.random() < self.epsilon:
            with torch.no_grad():
                _, hidden_state = self.policy_net(state_t, hidden_state)
            action = random.randrange(self.action_size)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values, hidden_state = self.policy_net(state_t, hidden_state)
            self.policy_net.train()
            # q_values => (1,1,6)
            action = q_values[0, -1].argmax().item()
        return action, hidden_state

    def learn_from_episode(self, episode):
        """
        episode => list of (state(21,), action, reward, next_state(21,), done)
        We do truncated BPTT with chunk_size
        """
        seq_len = len(episode)
        if seq_len == 0:
            return 0.0

        states = np.array([e[0] for e in episode], dtype=np.float32)
        actions = np.array([e[1] for e in episode], dtype=np.int64)
        rewards = np.array([e[2] for e in episode], dtype=np.float32)
        next_states = np.array([e[3] for e in episode], dtype=np.float32)
        dones = np.array([e[4] for e in episode], dtype=np.float32)

        hidden_state = self.reset_hidden_state()
        loss_total = 0.0
        num_chunks = 0

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)

            chunk_s = torch.FloatTensor(states[start:end]).unsqueeze(0).to(self.device)
            chunk_a = torch.LongTensor(actions[start:end]).unsqueeze(0).to(self.device)
            chunk_r = torch.FloatTensor(rewards[start:end]).unsqueeze(0).to(self.device)
            chunk_ns = torch.FloatTensor(next_states[start:end]).unsqueeze(0).to(self.device)
            chunk_d = torch.FloatTensor(dones[start:end]).unsqueeze(0).to(self.device)

            q_values, hidden_state = self.policy_net(chunk_s, hidden_state)
            q_values = q_values.gather(2, chunk_a.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                target_hid = self.reset_hidden_state()
                q_next, _ = self.target_net(chunk_ns, target_hid)
                max_q_next = q_next.max(dim=2)[0]
                targets = chunk_r + self.gamma * max_q_next * (1 - chunk_d)

            loss = nn.MSELoss()(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            loss_total += loss.item()
            num_chunks += 1

            # detach hidden state
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        self.soft_update()
        return loss_total / max(num_chunks, 1)

    def soft_update(self):
        """
        target = tau*policy + (1-tau)*target
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    # ================== PARTIAL LOAD ================== #
    def partial_load(self, filename):
        """
        Load only layers that match. If the old checkpoint was for a 7-dim input,
        'fc1.weight' shape => [64,7], but now we have [64,21].
        We'll skip that mismatch and load anything else that fits.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        old_policy_dict = checkpoint['policy_model_state_dict']
        new_policy_dict = self.policy_net.state_dict()

        loaded_dict = {}
        for k, v in old_policy_dict.items():
            if k in new_policy_dict and v.shape == new_policy_dict[k].shape:
                loaded_dict[k] = v
            else:
                print(f"Skipping layer {k}, shape mismatch: {v.shape} != {new_policy_dict[k].shape}")

        new_policy_dict.update(loaded_dict)
        self.policy_net.load_state_dict(new_policy_dict)

        # Do the same for target_net
        old_target_dict = checkpoint['target_model_state_dict']
        new_target_dict = self.target_net.state_dict()

        loaded_target = {}
        for k, v in old_target_dict.items():
            if k in new_target_dict and v.shape == new_target_dict[k].shape:
                loaded_target[k] = v
            else:
                print(f"Skipping layer {k}, shape mismatch: {v.shape} != {new_target_dict[k].shape}")

        new_target_dict.update(loaded_target)
        self.target_net.load_state_dict(new_target_dict)

        # same for optimizer, if you'd like:
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print("Skipping optimizer load due to mismatch:", e)

        self.epsilon = checkpoint.get('epsilon', 1.0)  # load old epsilon if present
        print("Partial load from:", filename)

    def save(self, filename="drqn_final.pt"):
        torch.save({
            'policy_model_state_dict': self.policy_net.state_dict(),
            'target_model_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

# ================== Training Loop ================== #
def train_drqn(env, agent, num_episodes=500):
    all_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        passenger_on = 0  # Keep track of whether passenger is on board
        done = False
        total_reward = 0.0

        # parse obs => 21D
        parsed_obs = parse_state(obs, passenger_on)
        state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)

        hidden_state = agent.reset_hidden_state()
        episode_experience = []

        while not done:
            action, hidden_state_next = agent.act(state, hidden_state)
            next_obs, reward, done, info = env.step(action)

            # For example: if pickup is successful => passenger_on=1, etc.
            if action == ACTION_PICKUP and reward >= 0:
                passenger_on = 1
            elif action == ACTION_DROPOFF and reward >= 0:
                passenger_on = 0

            next_parsed = parse_state(next_obs, passenger_on)
            next_state = np.array(next_parsed, dtype=np.float32).reshape(1, -1)

            episode_experience.append((state.squeeze(0), action, reward, next_state.squeeze(0), float(done)))

            state = next_state
            hidden_state = hidden_state_next
            total_reward += reward

        loss_val = agent.learn_from_episode(episode_experience)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)  # decay once per episode
        all_rewards.append(total_reward)

        if (ep+1) % 50 == 0:
            avg_50 = np.mean(all_rewards[-50:])
            print(f"[Episode {ep+1}/{num_episodes}] AvgReward(Last50)={avg_50:.2f}, Eps={agent.epsilon:.3f}")

    agent.save("drqn_final.pt")
    return all_rewards

# ================== get_action for Testing ================== #
def get_action(obs):
    """
    Called by the environment at test time. We'll keep a static agent with partial load if desired.
    """
    if not hasattr(get_action, "agent"):
        get_action.agent = DRQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
        # PARTIAL LOAD if you want to load an older checkpoint with mismatch
        # get_action.agent.partial_load("old_drqn_final.pt")  # Use partial_load
        # OR you can do a direct load if you have a new checkpoint that matches 21 dims
        get_action.agent.load("drqn_final.pt")

        get_action.hidden_state = get_action.agent.reset_hidden_state()
        get_action.passenger_on = 0
        get_action.last_obs = None
        get_action.last_action = None

    # Optionally track the last action to see if pickup => passenger_on=1
    # We'll skip that for brevity in this snippet.
    # Just parse with passenger_on=0 always, or maintain get_action.passenger_on if you want.

    # parse -> 21D
    parsed_obs = parse_state(obs, get_action.passenger_on)
    state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)

    # No exploration in test
    old_epsilon = get_action.agent.epsilon
    get_action.agent.epsilon = 0.0
    action, new_hidden = get_action.agent.act(state, get_action.hidden_state)
    get_action.agent.epsilon = old_epsilon

    get_action.hidden_state = new_hidden
    return action

# ================== MAIN ================== #
if __name__ == "__main__":
    env = SimpleTaxiEnv(5, 5000)

    agent = DRQNAgent(
        state_size=6,
        action_size=6,
        hidden_size=64,
        lstm_hidden_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,
        chunk_size=32
    )
    rewards_log = train_drqn(env, agent, num_episodes=4000)

