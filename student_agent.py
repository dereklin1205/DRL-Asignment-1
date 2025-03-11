import numpy as np
import random
import pickle
from collections import defaultdict
from simple_custom_taxi_env import SimpleTaxiEnv

# ======================== STATE PARSING ========================
def parse_state(obs):
    """
    Convert the environment's raw 16D observation into a 16D representation:
      - (taxi_row, taxi_col)
      - dx, dy for each of the 4 stations
      - obstacle_north, obstacle_south, obstacle_east, obstacle_west
      - passenger_look, destination_look

    Raw obs layout (16D):
    (
        taxi_row, taxi_col,
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
    )
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

    # Differences in x(row) and y(col) for each station
    dx0 = st0_row - taxi_row
    dy0 = st0_col - taxi_col
    dx1 = st1_row - taxi_row
    dy1 = st1_col - taxi_col
    dx2 = st2_row - taxi_row
    dy2 = st2_col - taxi_col
    dx3 = st3_row - taxi_row
    dy3 = st3_col - taxi_col

    # Final 16D tuple
    parsed = (
        dx0, dy0,            # 4
        dx1, dy1,            # 6
        dx2, dy2,            # 8
        dx3, dy3,            # 10
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,       # 14
        passenger_look,
        destination_look     # 16
    )
    return parsed

# ======================== Q-LEARNING AGENT ========================
class QLearningAgent:
    """
    A tabular Q-learning agent that uses a dictionary-based Q-table.
    The keys are (state_tuple), and the values are Q-values for each action (0..5).
    """
    def __init__(self,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 action_size=6):
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount
        self.epsilon = epsilon      # exploration prob
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        
        # Dictionary for Q-table: Q[state][action] -> float
        # We'll store each state's action-values in a length-6 np.array
        self.Q = defaultdict(lambda: np.zeros(self.action_size))

    def act(self, state):
        """
        Epsilon-greedy action selection based on Q-table.
        state is a tuple (16D).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q[state])  # best known action

    def learn(self, state, action, reward, next_state, done):
        """
        Classic Q-learning update:
        Q(s,a) <- Q(s,a) + alpha [r + gamma * max Q(s',.) - Q(s,a)]
        """
        old_value = self.Q[state][action]
        best_next = 0.0 if done else np.max(self.Q[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * best_next - old_value)
        self.Q[state][action] = new_value

    def update_epsilon(self):
        """
        Decay epsilon once per episode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="q_table.pkl"):
        """
        Save Q-table to disk via pickle
        """
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)  # convert defaultdict to dict

    def load(self, filename="q_table.pkl"):
        """
        Load Q-table from disk
        """
        with open(filename, "rb") as f:
            loaded_dict = pickle.load(f)
        # Re-wrap as defaultdict
        self.Q = defaultdict(lambda: np.zeros(self.action_size), loaded_dict)

# ======================== TRAINING FUNCTION ========================
def train(env, agent, num_episodes=1000):
    """
    Train the Q-table over multiple episodes.
    We'll do epsilon decay once at the end of each episode.
    """
    all_rewards = []
    for episode in range(num_episodes):
        raw_obs, _ = env.reset()
        state = parse_state(raw_obs)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            raw_next_obs, reward, done,  _ = env.step(action)
            next_state = parse_state(raw_next_obs)

            # Q-learning update
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        # Decay epsilon once per episode
        agent.update_epsilon()

        all_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_100 = np.mean(all_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Epsilon={agent.epsilon:.3f}, AvgReward(Last100)={avg_100:.2f}")

    # Save final Q-table
    agent.save("q_table.pkl")
    return all_rewards

# ======================== GET ACTION FOR TESTING ========================
def get_action(obs):
    """
    This function is called by the environment to get an action from
    our loaded Q-table. We parse the obs and do a greedy action selection.
    """
    # Lazy-load QLearningAgent
    if not hasattr(get_action, "agent"):
        get_action.agent = QLearningAgent(epsilon=0.0)  # no exploration in test
        get_action.agent.load("q_table.pkl")

    # Convert raw obs -> 16D state
    state = parse_state(obs)
    # Greedy action
    Q_values = get_action.agent.Q[state]
    action = np.argmax(Q_values)
    return action

# ======================== MAIN ========================
if __name__ == "__main__":
    env = SimpleTaxiEnv(5,5000)

    # Initialize Q agent (only difference is we parse (dx,dy) for stations)
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999,
        action_size=6
    )

    # Train for N episodes
    rewards = train(env, agent, num_episodes=10000)
    print("Training complete! Final epsilon=", agent.epsilon)
