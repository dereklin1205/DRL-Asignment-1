import numpy as np
import random
import pickle
from collections import defaultdict
from simple_custom_taxi_env import SimpleTaxiEnv
from tqdm import *

# 方便對照
ACTION_SOUTH   = 0
ACTION_NORTH   = 1
ACTION_EAST    = 2
ACTION_WEST    = 3
ACTION_PICKUP  = 4
ACTION_DROPOFF = 5

# =============== [1] STATE PARSING：加入 passenger_on 維度 =============== #
def parse_state(obs, passenger_on):
    """
    原本 obs(16D):
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

    我們將最後多加一個 passenger_on (0/1)，讓 Q-Table 區分「是否已載客」.
    這樣最終 state 會是 17 維(以 tuple 表示)。
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

    # 組合成 tuple
    parsed_state = (
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
        passenger_on  # <--- 關鍵：新增一維
    )
    return parsed_state

# =============== [2] Q-Learning Agent =============== #
class QLearningAgent:
    def __init__(self,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 action_size=6):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size

        # Q-table: dict[ state_tuple ] = np.array of length action_size
        self.Q = defaultdict(lambda: np.zeros(self.action_size))

    def act(self, state):
        """Epsilon-greedy 選擇行動."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update."""
        old_value = self.Q[state][action]
        best_next = 0.0 if done else np.max(self.Q[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * best_next - old_value) *( 1 - done)
        self.Q[state][action] = new_value

    def update_epsilon(self):
        """每個 episode 結束後，更新 epsilon (探索率)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="q_table.pkl"):
        """將 Q-table 存起來."""
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, filename="q_table.pkl"):
        """從檔案載入 Q-table."""
        with open(filename, "rb") as f:
            loaded_dict = pickle.load(f)
        self.Q = loaded_dict
        print("Q-table loaded from", filename)

# =============== [3] Training function =============== #
def train(env, agent, num_episodes=1000):
    """
    這裡示範用「檢查 reward 是否 >= 0」來判斷 PICKUP/DROPOFF 是否成功。
    - 一開始 passenger_on=0 (未載客)
    - 若 action=ACTION_PICKUP (4)，且 reward >= 0 => passenger_on=1
    - 若 action=ACTION_DROPOFF (5)，且 reward >= 0 => passenger_on=0
    其餘時間 passenger_on 保持原值。
    """
    all_rewards = []

    for episode in range(num_episodes):
        # =========== Episode 初始化 =========== #
        passenger_on = 0
        raw_obs, _ = env.reset()
        state = parse_state(raw_obs, passenger_on)
        total_reward = 0.0
        done = False

        while not done:
            # 選 action
            action = agent.act(state)

            # 與環境互動
            raw_next_obs, reward, done, info = env.step(action)

            # =======================
            # 用 reward 判斷 PickUp/DropOff 是否成功
            # =======================
            if action == ACTION_PICKUP:
                # 假設成功 PICKUP => reward >= 0
                if reward >= 0:
                    passenger_on = 1

            elif action == ACTION_DROPOFF:
                # 假設成功 DROPOFF => reward >= 0
                if reward >= 0:
                    passenger_on = 0

            # 計算 next_state
            next_state = parse_state(raw_next_obs, passenger_on)

            # Q-learning update
            agent.learn(state, action, reward, next_state, done)

            # 移動到下個 state
            state = next_state
            raw_obs = raw_next_obs
            total_reward += reward

        # episode 結束，更新 epsilon
        agent.update_epsilon()

        all_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            avg_100 = np.mean(all_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"Epsilon={agent.epsilon:.3f}, "
                  f"AvgReward(Last100)={avg_100:.2f}")

    # 存下 Q-table
    agent.save("q_table.pkl")
    return all_rewards

# =============== [4] 測試時使用 get_action =============== #
def get_action(obs):
    """
    測試時呼叫 get_action(obs) 取得行動。
    注意：在這裡我們無法直接用 reward 判斷 PICKUP/DROPOFF 是否成功，
    因為環境測試時通常不會把 reward 傳進來，只會給下一個 obs。
    所以維持原本透過「上一個動作+觀測值」去猜測是否成功 PICKUP 或 DROPOFF。
    """
    # 用靜態屬性儲存 agent、passenger_on、last_obs、last_action
    if not hasattr(get_action, "agent"):
        # 第一次呼叫 => 初始化
        get_action.agent = QLearningAgent(epsilon=0.0)  # no exploration in test
        get_action.agent.load("q_table.pkl")

        get_action.passenger_on = 0  # 尚未載客
        get_action.last_obs = None
        get_action.last_action = None

    # 先判斷上一個 action 是 PICKUP 或 DROPOFF，
    # 並嘗試依環境觀測 obs[13] 判斷是否成功
    # （或者你可以另外寫邏輯，只要該位置確實是目的地就 passenger_on=0，等等）
    if get_action.last_obs is not None and get_action.last_action is not None:
        # 上一次觀測
        prev_obs = get_action.last_obs
        prev_action = get_action.last_action
        
        if prev_action == ACTION_PICKUP:
            # 如果上一刻的 obs[13]==1 (旁邊有乘客) => 表示成功 PICKUP
            if prev_obs[13] == 1 :
                get_action.passenger_on = 1
        elif prev_action == ACTION_DROPOFF:
            # 如果原本載客 => 嘗試放下 => 令 passenger_on=0
            # 這邊簡單處理，可再根據是否在正確地點做判定
            if get_action.passenger_on == 1:
                get_action.passenger_on = 0

    # 現在用 obs + passenger_on 組合當前 state
    state = parse_state(obs, get_action.passenger_on)

    Q_values = get_action.agent.Q.get(state, None)
    if Q_values is None:
        # 若這個狀態不在 Q-table => 隨機
        action = random.randint(0, 5)
    else:
        # 貪心選擇最大 Q-value 的動作
        action = np.argmax(Q_values)

    # 記住這一刻
    get_action.last_obs = obs
    get_action.last_action = action

    return action

# =============== [5] MAIN 訓練/測試範例 (可自行調整) =============== #
if __name__ == "__main__":
    ##222
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99999,
        action_size=6
    )
    env = SimpleTaxiEnv(10, 5000)
    rewards = train(env, agent, num_episodes=100000)
    agent.epsilon = 1.0
        
    print("Training complete! Final epsilon=", agent.epsilon)
    # 開始訓練
    

    # 你也可以在這裡測試看看 agent 的表現
    # obs, _ = env.reset()
    # passenger_on = 0
    # done = False
    # total_r = 0
    # while not done:
    #     state = parse_state(obs, passenger_on)
    #     act_ = agent.act(state)
    #     next_obs, r, done, info = env.step(act_)
    #
    #     # 用 reward 判斷 PICKUP/DROPOFF
    #     if act_ == ACTION_PICKUP and r >= 0:
    #         passenger_on = 1
    #     elif act_ == ACTION_DROPOFF and r >= 0:
    #         passenger_on = 0
    #
    #     total_r += r
    #     obs = next_obs
    #     env.render()  # 看看走向
    #
    # print("Test run total reward:", total_r)
