import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 假設你的 Taxi 環境自訂檔
from simple_custom_taxi_env_ import SimpleTaxiEnv

# ------------------ 常量：動作編號 ------------------
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5

# =============== 1. 修正 parse_state：確實回傳 21 維 =============== #
def parse_state(obs, passenger_on):
    """
    - 假設原本 obs 有 16 維 (或你可能已有程式把它算到 20 維)。
    - 這裡示範最小化：目前顯示 obs 當中只使用了 obstacle_north, etc. => 6 維。
    - 我們要把 taxi_row, taxi_col, station distance... 也都算進來，如你原先程式需求。
    - 然後在最後面加上 passenger_on => 21 維。
    """
    # 示範：這裡直接把 obs 拆開
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

    # 例如，我們想算每個站台與 taxi 的距離 (Manhattan)
    # 這裡只舉一個例子 st0，實際你要 4 個站台 => 再加更多
    st0_dist = ((taxi_row - st0_row) ,abs(taxi_col - st0_col))
    st1_dist = ((taxi_row - st1_row) ,abs(taxi_col - st1_col))
    st2_dist = ((taxi_row - st2_row) ,abs(taxi_col - st2_col))
    st3_dist = ((taxi_row - st3_row) ,abs(taxi_col - st3_col))

    # 這裡先示範把上面資訊組成 20 維 => (你的需求可自己加/改)
    # 假設我們把 taxi_row, taxi_col, obstacle_north.., passenger_look, station distance... 合成:
    twenty_dims = (
        st0_dist[0],
        st0_dist[1],
        st1_dist[0],
        st1_dist[1],
        st2_dist[0],
        st2_dist[1],
        st3_dist[0],
        st3_dist[1],
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look
        # 這裡目前是 12 維
        # 你可以再把 st0_row, st0_col, st1_row, st1_col, etc. 都加進來
        # 直到你想要的 20 維
    )

    # 把 passenger_on 加到 tuple 結尾 => 21 維
    twenty_one_dims = twenty_dims + (passenger_on,)

    return twenty_one_dims


# =============== 2. DRQN 架構 (大致同你的程式) =============== #
class DRQN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_hidden_size, action_size):
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

        x, hidden_state = self.lstm(x, hidden_state)   # (batch, seq_len, lstm_hidden_size)
        q_values = self.fc2(x)                         # (batch, seq_len, action_size)
        return q_values, hidden_state


class DRQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lstm_hidden_size=64,
                 gamma=0.99, learning_rate=0.001, tau=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 chunk_size=32):
        # 確保 state_size=21
        self.state_size = state_size    # e.g. 21
        self.action_size = action_size  # e.g. 6
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
        h0 = torch.zeros(1, batch_size, self.policy_net.hidden_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.policy_net.hidden_size).to(self.device)
        return (h0, c0)

    def act(self, state, hidden_state):
        """
        state: shape (1, state_size) or (batch=1, state_size)
        hidden_state: (h, c)
        回傳 (action, new_hidden_state)
        """
        # 變成 (1, 1, state_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < self.epsilon:
            with torch.no_grad():
                # still forward pass so hidden state updates
                _, hidden_state = self.policy_net(state_tensor, hidden_state)
            action = random.randrange(self.action_size)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q_values, hidden_state = self.policy_net(state_tensor, hidden_state)
            self.policy_net.train()
            # q_values shape: (1, 1, action_size)
            action = torch.argmax(q_values[0, -1]).item()
        return action, hidden_state

    def learn_from_episode(self, episode):
        """
        episode: list of (state(21,), action, reward, next_state(21,), done)
        """
        seq_len = len(episode)
        if seq_len == 0:
            return 0

        states = np.array([e[0] for e in episode], dtype=np.float32)
        actions = np.array([e[1] for e in episode], dtype=np.int64)
        rewards = np.array([e[2] for e in episode], dtype=np.float32)
        next_states = np.array([e[3] for e in episode], dtype=np.float32)
        dones = np.array([e[4] for e in episode], dtype=np.float32)

        loss_total = 0.0
        num_chunks = 0
        hidden_state = self.reset_hidden_state(batch_size=1)

        # Truncated BPTT
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)

            chunk_states = torch.FloatTensor(states[start:end]).unsqueeze(0).to(self.device)
            chunk_actions = torch.LongTensor(actions[start:end]).unsqueeze(0).to(self.device)
            chunk_rewards = torch.FloatTensor(rewards[start:end]).unsqueeze(0).to(self.device)
            chunk_next_states = torch.FloatTensor(next_states[start:end]).unsqueeze(0).to(self.device)
            chunk_dones = torch.FloatTensor(dones[start:end]).unsqueeze(0).to(self.device)

            q_values, hidden_state = self.policy_net(chunk_states, hidden_state)
            # q_values shape: (1, chunk_len, action_size)
            # gather => (1, chunk_len)
            q_values = q_values.gather(2, chunk_actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                target_hidden = self.reset_hidden_state(batch_size=1)
                q_next, _ = self.target_net(chunk_next_states, target_hidden)
                max_q_next = q_next.max(dim=2)[0]  # (1, chunk_len)
                targets = chunk_rewards + self.gamma * max_q_next * (1 - chunk_dones)

            loss = nn.MSELoss()(q_values, targets)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
            self.optimizer.step()

            loss_total += loss.item()
            num_chunks += 1

            # detach hidden state
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        self.soft_update()
        return loss_total / max(num_chunks, 1)

    def soft_update(self):
        """微小更新 target_net"""
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
        print("DRQN loaded from:", filename)


# =============== 3. 訓練：維護 passenger_on =============== #
def train_drqn(env, agent, num_episodes=1000):
    all_rewards = []
    for ep in range(num_episodes):
        passenger_on = 0
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # obs + passenger_on => 21 維 state
        parsed_obs = parse_state(obs, passenger_on)
        state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)
        # print(obs)
        # print(state)
        hidden_state = agent.reset_hidden_state(batch_size=1)
        episode_experience = []

        while not done:
            action, hidden_state_next = agent.act(state, hidden_state)
            next_obs, reward, done, info = env.step(action)

            # 如果 pick up 成功 => passenger_on=1; drop off 成功 => passenger_on=0
            if action == ACTION_PICKUP and reward >= 0:
                passenger_on = 1
            elif action == ACTION_DROPOFF:
                passenger_on = 0

            next_parsed_obs = parse_state(next_obs, passenger_on)
            next_state = np.array(next_parsed_obs, dtype=np.float32).reshape(1, -1)

            episode_experience.append((
                state.squeeze(0),  # shape (21,)
                action,
                reward,
                next_state.squeeze(0),  # shape (21,)
                float(done)
            ))

            state = next_state
            hidden_state = hidden_state_next
            total_reward += reward

        # 每個 episode 結束後，用 learn_from_episode
        loss_val = agent.learn_from_episode(episode_experience)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)  # 1 episode => decay once
        all_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            avg_reward_100 = np.mean(all_rewards[-100:])
            print(f"Ep {ep+1}/{num_episodes}, AvgR={avg_reward_100:.2f}, Eps={agent.epsilon:.4f}")

    agent.save("drqn_final.pt")
    return all_rewards


# =============== 4. 推理: get_action(obs) =============== #
def get_action(obs):
    STATE_SIZE = 15
    ACTION_SIZE = 6

    if not hasattr(get_action, "agent"):
        get_action.agent = DRQNAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE
        )
        get_action.agent.load("drqn_final.pt")
        get_action.hidden_state = get_action.agent.reset_hidden_state(batch_size=1)

        get_action.passenger_on = 0
        get_action.last_obs = None
        get_action.last_action = None

    # 判斷上一步的動作 & obs => 更新 passenger_on
    if get_action.last_obs is not None and get_action.last_action is not None:
        prev_obs = get_action.last_obs
        prev_action = get_action.last_action
        if prev_action == ACTION_PICKUP and ((prev_obs[0] == 0 and prev_obs[1] )== 0 or (prev_obs[2] == 0 and prev_obs[3]) == 0 or (prev_obs[4] == 0 and prev_obs[5]) == 0 or( prev_obs[6] == 0 and prev_obs[7] == 0)):
            get_action.passenger_on = 1
        elif prev_action == ACTION_DROPOFF:
            get_action.passenger_on = 0

    state = parse_state(obs, get_action.passenger_on)
    state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)

    # 評估時 epsilon=0
    old_epsilon = get_action.agent.epsilon
    get_action.agent.epsilon = 0.0
    action, new_hidden = get_action.agent.act(state, get_action.hidden_state)
    get_action.agent.epsilon = old_epsilon

    get_action.hidden_state = new_hidden
    get_action.last_obs = state
    get_action.last_action = action

    return action


# =============== 5. main: 訓練 + 測試 (範例) =============== #
if __name__ == "__main__":
    # 環境: SimpleTaxiEnv(5, 5000) => 視你的實作情況
    env = SimpleTaxiEnv(5, 5000)
    agent = DRQNAgent(
        state_size=15,
        action_size=6,
        hidden_size=64,
        lstm_hidden_size=64,
        gamma=0.99,
        learning_rate=0.001,
        tau=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999,
        chunk_size=32
    )

    # 訓練
    rewards_log = train_drqn(env, agent, num_episodes=10000)
    print("Training Finished! Model saved to drqn_final.pt")

    # 簡單測試
    test_episodes = 3
    for t_ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        passenger_on_test = 0
        hidden_state = agent.reset_hidden_state(batch_size=1)
        total_r = 0

        while not done:
            parsed_obs = parse_state(obs, passenger_on_test)
            state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            action, hidden_state = agent.act(state, hidden_state)
            agent.epsilon = old_epsilon

            next_obs, r, done, info = env.step(action)
            if action == ACTION_PICKUP and r >= 0:
                passenger_on_test = 1
            elif action == ACTION_DROPOFF and r >= 0:
                passenger_on_test = 0

            obs = next_obs
            total_r += r

        print(f"[Test Episode {t_ep+1}] Total Reward = {total_r}")
