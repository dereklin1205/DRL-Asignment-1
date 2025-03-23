# student_agent.py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from tqdm import trange

# 載入你自訂的環境
from simple_custom_taxi_env import SimpleTaxiEnv

# -----------------
# ACTION 常數
# -----------------
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5

# =============== [1] parse_state：保留階段控制 (stage_0, stage_1) =============== #
def parse_state(obs, passenger_on, stage_0, stage_1):
    """
    把原始 obs (16 維) + passenger_on + stage_0 + stage_1 整合起來，並回傳一個可餵給 NN 的向量。
    這段程式基本上沿用你的版本，但要特別把 (dx, dy) + 其他 bool, int 攤平成 1D。
    """
    (taxi_row,
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
     destination_look) = obs

    dx0 = st0_row - taxi_row
    dy0 = st0_col - taxi_col
    dx1 = st1_row - taxi_row
    dy1 = st1_col - taxi_col
    dx2 = st2_row - taxi_row
    dy2 = st2_col - taxi_col
    dx3 = st3_row - taxi_row
    dy3 = st3_col - taxi_col

    # 決定當前目標 goal = (dx, dy)
    if not passenger_on:
        # 還沒載到客 => stage_0
        if stage_0 == 0:
            goal = (dx0, dy0)
        elif stage_0 == 1:
            goal = (dx1, dy1)
        elif stage_0 == 2:
            goal = (dx2, dy2)
        else:
            goal = (dx3, dy3)
    else:
        # 載到客 => stage_1
        if stage_1 == 0:
            goal = (dx0, dy0)
        elif stage_1 == 1:
            goal = (dx1, dy1)
        elif stage_1 == 2:
            goal = (dx2, dy2)
        else:
            goal = (dx3, dy3)

    # 檢查是否抵達某站點 => 更新 stage_0 / stage_1
    if passenger_on == 0:  # 還沒載到客
        if stage_0 == 0 and dx0 == 0 and dy0 == 0:
            stage_0 += 1
        elif stage_0 == 1 and dx1 == 0 and dy1 == 0:
            stage_0 += 1
        elif stage_0 == 2 and dx2 == 0 and dy2 == 0:
            stage_0 += 1
        elif stage_0 == 3 and dx3 == 0 and dy3 == 0:
            stage_0 = 0
    else:
        # 已載客
        if stage_1 == 0 and dx0 == 0 and dy0 == 0:
            stage_1 += 1
        elif stage_1 == 1 and dx1 == 0 and dy1 == 0:
            stage_1 += 1
        elif stage_1 == 2 and dx2 == 0 and dy2 == 0:
            stage_1 += 1
        elif stage_1 == 3 and dx3 == 0 and dy3 == 0:
            stage_1 = 0

    # 組成 tuple/state
    # 這裡做成 9 維: (dx, dy) + 4 obstacle bool + passenger_look bool + destination_look bool + passenger_on
    dx, dy = goal
    obs_n = int(obstacle_north)
    obs_s = int(obstacle_south)
    obs_e = int(obstacle_east)
    obs_w = int(obstacle_west)
    pass_lk = int(passenger_look)
    dest_lk = int(destination_look)
    pass_on = int(passenger_on)

    # shape=(9,)
    parsed_state = np.array([
        dx, dy,
        obs_n, obs_s, obs_e, obs_w,
        pass_lk, dest_lk,
        pass_on
    ], dtype=np.float32)

    return parsed_state, stage_0, stage_1

# =============== [2] Policy Network + REINFORCE Agent =============== #
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits  # (batch, action_dim)

class PolicyGradientAgent:
    def __init__(self,
                 state_dim=9,     # 9 維輸入
                 action_dim=6,    # 6 種動作
                 lr=1e-3,
                 gamma=0.99):
        self.gamma = gamma
        self.net = PolicyNetwork(state_dim, action_dim, hidden_size=64)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)


        self.loss_fn = nn.CrossEntropyLoss()

        # 用來存一個 episode 的 (log_prob, reward)
        self.log_probs = []
        self.rewards   = []

    def _to_tensor(self, state_np):
        """
        將 numpy array 轉為 torch tensor。
        state_np shape=(9,) => (1,9)
        """
        return torch.from_numpy(state_np).unsqueeze(0)

    def select_action(self, state_np):
        """
        給 state (np array)，用 policy 網路輸出 logits，
        建立 Categorical 分布後做抽樣 -> action。
        回傳 (action, log_prob)。
        """
        state_tensor = self._to_tensor(state_np)  # shape=(1,9)
        logits = self.net(state_tensor)           # shape=(1,6)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                    # shape=(1,)
        log_prob = dist.log_prob(action)          # shape=(1,)
        return action.item(), log_prob

    def store_outcome(self, log_prob, reward):
        """把此 timestep 的資料存進 list，等 episode 結束再做一次性更新。"""
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def finish_episode(self):
        """
        REINFORCE 更新：
        Gt = r_t + gamma*r_{t+1} + ...
        loss = - Σ[ log_prob_t * Gt ]。
        """
        self.optimizer.zero_grad()

        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma*G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 如果想做 baseline，可 returns -= returns.mean()，這裡先不做。
        loss = 0
        for log_p, R in zip(self.log_probs, returns):
            loss += -log_p * R

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空暫存
        self.log_probs = []
        self.rewards   = []
        

# =============== [3] REINFORCE 訓練流程 =============== #
def train(env, agent, num_episodes=1000):
    """
    和 Q-learning 不同之處在於：
      1) 整個 episode 收集 (log_prob, reward)
      2) episode 結束再做 agent.finish_episode()
    """
    all_rewards = []
    for episode in trange(num_episodes):
        # 初始化
        raw_obs, _ = env.reset()
        passenger_on = 0
        stage_0 = 0
        stage_1 = 0
        state_np, stage_0, stage_1 = parse_state(raw_obs, passenger_on, stage_0, stage_1)
        done = False
        total_reward = 0.0
        prev_dis = (0,0)

        while not done:
            # 用 policy 抽樣動作
            action, log_prob = agent.select_action(state_np)

            # 與環境互動
            next_obs, reward, done, info = env.step(action)
            
            # 根據動作判斷是否 PICKUP / DROPOFF 成功，更新 passenger_on
            # （範例：僅僅示意，實際可依 reward > 0 或檢查 obs 來判斷）
            if action == ACTION_PICKUP:
                # 若剛好到乘客位置 => 可能 passenger_on=1
                if state_np[0] == 0 and state_np[1] == 0 and state_np[5] == 1:
                    passenger_on = 1
            elif action == ACTION_DROPOFF:
                if passenger_on == 1:
                    passenger_on = 0
            
            # 存下 (log_prob, reward)
            state_np, stage_0, stage_1 = parse_state(next_obs, passenger_on, stage_0, stage_1)

            if abs(prev_dis[0]) + abs(prev_dis[1]) > abs(state_np[0]) + abs(state_np[1]):
                reward +=3
            else:
                reward -=3 
            prev_dis = (state_np[0], state_np[1])
            # 更新 state_np
            agent.store_outcome(log_prob, reward)
            total_reward += reward
        # 一個 episode 結束 => 做一次性更新
        agent.finish_episode()
        all_rewards.append(total_reward)

        # 每隔 100 回合顯示
        if (episode+1) % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes} | AvgReward(100ep)={avg_reward:.2f}")

    return all_rewards

# =============== [4] 測試時：get_action(obs) =============== #
def get_action(obs):
    """
    作業系統在測試時，只會呼叫 get_action(obs) 取得動作。
    因此我們需要在這裡把「PolicyGradientAgent + PyTorch Model」載入 (或快取)，
    然後把 obs 轉成 state，再用網路 forward。
    
    注意：因為在測試時，我們只拿到單步 obs，而不知道前一步動作/回饋，
          所以若你需要 passenger_on、stage_0, stage_1 等狀態切換，
          得在這裡維護靜態變數 (或 global 變數)。
    """

    # 第一次呼叫 => 初始化 agent, 載入已訓練好的 model
    if not hasattr(get_action, "agent"):
        # 建 Agent
        get_action.agent = PolicyGradientAgent(state_dim=9, action_dim=6)
        # 載入權重 (如果你在 train 完把權重存成 "policy_net.pt")
        # 若沒有就註解掉
        try:
            get_action.agent.net.load_state_dict(torch.load("policy_net.pt"))
            print("Loaded policy_net.pt!")
        except:
            print("Warning: policy_net.pt not found. Using untrained weights.")

        # 同時，也要在測試階段維護 passenger_on, stage_0, stage_1
        get_action.passenger_on = 0
        get_action.stage_0 = 0
        get_action.stage_1 = 0

    # 讀取靜態屬性
    passenger_on = get_action.passenger_on
    stage_0 = get_action.stage_0
    stage_1 = get_action.stage_1

    # parse state
    state_np, stage_0, stage_1 = parse_state(obs, passenger_on, stage_0, stage_1)

    # 若需要依「上一個動作」去判斷 pickup/droppoff 的成功，需要紀錄 last_action, last_obs, etc.
    # 這裡僅示意：如果你確定 obs[5]==1 & 目標在 (0,0) => passenger_on=1 等等，也行。
    # -----------------------------------------------------------------------

    # 用訓練好的 policy 做前向傳播
    state_tensor = torch.from_numpy(state_np).unsqueeze(0)  # shape=(1,9)
    logits = get_action.agent.net(state_tensor)
    # 測試時可用貪心 argmax，也可以照樣抽樣
    action = torch.argmax(logits, dim=1).item()

    # 更新靜態變數
    get_action.passenger_on = passenger_on
    get_action.stage_0 = stage_0
    get_action.stage_1 = stage_1

    return action

# =============== [5] main：訓練 & 儲存權重 =============== #
if __name__ == "__main__":
    # 你可以在這裡做完整的訓練流程
    env = SimpleTaxiEnv(5,5000)

    agent = PolicyGradientAgent(
        state_dim=9,   # 取決於 parse_state 的輸出維度
        action_dim=6,
        lr=1e-3,
        gamma=0.99
    )

    print("開始訓練 Policy Gradient (REINFORCE) ...")
    train(env, agent, num_episodes=1000)

    # 訓練結束後，存檔
    torch.save(agent.net.state_dict(), "policy_net.pt")
    print("訓練完成！已將權重存為 policy_net.pt。")
