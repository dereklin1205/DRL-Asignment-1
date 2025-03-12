
import numpy as np
import pickle
import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 假設你的 Taxi 環境自訂檔
from simple_custom_taxi_env import SimpleTaxiEnv

# ------------------ 常量：動作編號 ------------------
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5

# =============== 1. 改良 parse_state：多加 passenger_on 維度 =============== #
def parse_state(obs, passenger_on):
    """
    假設原本 obs 維度 => 16 or 20
    舉例：你在舊程式 parse_state(obs) 中計算出 20 維 (包含距離等資訊)。
    這裡只示範概念：最後多加 passenger_on => 回傳 21 維。
    """
    # ----------------------
    # 以下只是示範如何將 16 維 raw obs 轉為 20 維
    # (或你已經寫好的 parse_state(obs) 也行)
    # ----------------------
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
    for station_row, station_col in padded_stations:
        taxi_distance = abs(taxi_row - station_row) + abs(taxi_col - station_col)
        lst.append(taxi_distance)

    # 原本 20 維 (示範)
    twenty_dims = (
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look   # 第 19 維 (0-based)
    )

    # 多加一維 passenger_on => 最後維度變 21
    twenty_one_dims = twenty_dims + (passenger_on,)

    return twenty_one_dims


# =============== 2. DRQN 架構 (幾乎跟你原先程式相同) =============== #
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
        self.state_size = state_size    # 21
        self.action_size = action_size  # 6
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
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # => (1, 1, state_size)
        if random.random() < self.epsilon:
            # exploration
            with torch.no_grad():
                _, hidden_state = self.policy_net(state_tensor, hidden_state)
            action = random.randrange(self.action_size)
            return action, hidden_state
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

        # Truncated BPTT by chunk
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)

            chunk_states = torch.FloatTensor(states[start:end]).unsqueeze(0).to(self.device)      # (1, chunk_len, state_size)
            chunk_actions = torch.LongTensor(actions[start:end]).unsqueeze(0).to(self.device)      # (1, chunk_len)
            chunk_rewards = torch.FloatTensor(rewards[start:end]).unsqueeze(0).to(self.device)     # (1, chunk_len)
            chunk_next_states = torch.FloatTensor(next_states[start:end]).unsqueeze(0).to(self.device)
            chunk_dones = torch.FloatTensor(dones[start:end]).unsqueeze(0).to(self.device)

            q_values, hidden_state = self.policy_net(chunk_states, hidden_state)
            # 取出對應 actions 的 Q-value: (1, chunk_len, 1)
            q_values = q_values.gather(2, chunk_actions.unsqueeze(-1)).squeeze(-1)

            # target net 處理 next_states
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

            # detach hidden state for truncated BPTT
            hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

        self.soft_update()
        return loss_total / max(num_chunks, 1)

    def soft_update(self):
        """ 微小更新 target_net """
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
    """
    示範如何在收集資料時，用 reward>=0 判斷 PICKUP(4)/DROPOFF(5) 是否成功
    來更新 passenger_on。
    若你的環境設計不一樣(例如成功pickup會給0、錯誤pickup給-10)，可自行微調。
    """
    all_rewards = []
    for ep in range(num_episodes):
        passenger_on = 0
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        # 把 obs + passenger_on 轉成 21 維 state
        parsed_obs = parse_state(obs, passenger_on)
        state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)

        # 重置 LSTM hidden state
        hidden_state = agent.reset_hidden_state(batch_size=1)

        # 收集一整回合的經驗
        episode_experience = []

        while not done:
            action, hidden_state_next = agent.act(state, hidden_state)
            next_obs, reward, done, info = env.step(action)

            # 判斷 PICKUP / DROPOFF 是否成功
            if action == ACTION_PICKUP:
                if reward >= 0:
                    passenger_on = 1
            elif action == ACTION_DROPOFF:
                if reward >= 0:
                    passenger_on = 0

            next_parsed_obs = parse_state(next_obs, passenger_on)
            next_state = np.array(next_parsed_obs, dtype=np.float32).reshape(1, -1)

            # 記錄 (s, a, r, s', done)
            episode_experience.append((
                state.squeeze(0),        # shape (21,)
                action,
                reward,
                next_state.squeeze(0),   # shape (21,)
                float(done)
            ))

            state = next_state
            hidden_state = hidden_state_next
            total_reward += reward

        loss_val = agent.learn_from_episode(episode_experience)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        all_rewards.append(total_reward)

        if (ep+1) % 50 == 0:
            avg_reward_100 = np.mean(all_rewards[-100:])
            print(f"Episode {ep+1}/{num_episodes}, AvgReward={avg_reward_100:.2f}, Eps={agent.epsilon:.3f}")

    agent.save("drqn_final.pt")
    return all_rewards


# =============== 4. 推理: get_action(obs) ===============
def get_action(obs):
    """
    這個函式在測試/評估時被呼叫，只會得到 obs，沒有 reward。
    因此，我們用「上一個動作 + 上一次 obs 的 passenger_look」來猜測 PICKUP / DROPOFF 是否成功，
    然後維持 passenger_on。
    你也可以改用別的方法，只要與訓練時規則一致即可。
    """
    # 假設訓練時 parse_state 是 21 維 => 這裡也要保持一樣
    STATE_SIZE = 21
    ACTION_SIZE = 6

    if not hasattr(get_action, "agent"):
        # 第一次呼叫 => 初始化 Agent & hidden_state
        get_action.agent = DRQNAgent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE
        )
        get_action.agent.load("drqn_final.pt")
        get_action.hidden_state = get_action.agent.reset_hidden_state(batch_size=1)

        # 用來判斷 PICKUP / DROPOFF
        get_action.passenger_on = 0
        get_action.last_obs = None
        get_action.last_action = None

    # 如果上一步是 PICKUP 且上一步 obs[14] == 1 => 成功載客
    # 如果上一步是 DROPOFF => passenger_on=0 (簡化假設)
    if get_action.last_obs is not None and get_action.last_action is not None:
        prev_obs = get_action.last_obs
        prev_action = get_action.last_action

        # 這裡取上一步 passenger_look = prev_obs[14]
        # (若你 parse_state 之前就改寫 obs，請對應改)
        # 但要注意：當我們在 get_action 中取得 obs 是環境此刻的 obs，不是 parse_state(...) 之後的
        # 所以可直接用: if prev_action==4 and prev_obs[14] == 1 => get_action.passenger_on=1
        # 這只是簡化作法，你也可更嚴謹地判斷「是否在正確位置 pickup」。
        if prev_action == ACTION_PICKUP and prev_obs[14] == 1:
            get_action.passenger_on = 1
        elif prev_action == ACTION_DROPOFF:
            get_action.passenger_on = 0

    # 轉成 21 維 state
    parsed_obs = parse_state(obs, get_action.passenger_on)
    state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)

    # 測試時關掉探索 => epsilon=0
    epsilon_backup = get_action.agent.epsilon
    get_action.agent.epsilon = 0.0

    action, new_hidden = get_action.agent.act(state, get_action.hidden_state)

    get_action.agent.epsilon = epsilon_backup
    get_action.hidden_state = new_hidden
    get_action.last_obs = obs
    get_action.last_action = action

    return action


# =============== 5. (可選) 主程式示範訓練 + 測試 ===============
if __name__ == "__main__":
    # 初始化環境
    env = SimpleTaxiEnv(5, 5000)

    # 建立 DRQN Agent (注意: state_size=21)
    agent = DRQNAgent(
        state_size=7,
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

    # 訓練
    rewards_log = train_drqn(env, agent, num_episodes=4000)
    print("Training Finished! Saved model as drqn_final.pt")

    # (選擇性) 測試一下
    #   這裡只是簡單跑個 few episodes 查看結果
    test_episodes = 3
    for t_ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        passenger_on_test = 0
        hidden_state = agent.reset_hidden_state(batch_size=1)
        total_r = 0

        while not done:
            # 轉成 21維
            parsed_obs = parse_state(obs, passenger_on_test)
            state = np.array(parsed_obs, dtype=np.float32).reshape(1, -1)
            # 用 epsilon=0
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            action, hidden_state = agent.act(state, hidden_state)
            agent.epsilon = old_epsilon

            next_obs, r, done, info = env.step(action)

            # 用 reward 判斷 PICKUP / DROPOFF
            if action == ACTION_PICKUP and r >= 0:
                passenger_on_test = 1
            elif action == ACTION_DROPOFF and r >= 0:
                passenger_on_test = 0

            obs = next_obs
            total_r += r
        print(f"[Test Episode {t_ep+1}] Total Reward = {total_r}")
