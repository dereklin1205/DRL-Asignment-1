import gym
import numpy as np
from gym import spaces
import random

class TrainingTaxiEnv(gym.Env):
    """
    A training environment that randomizes the grid (5 <= n <= 10),
    station locations, obstacles, passenger, and destination—while
    keeping the same observation format as 'simple_custom_taxi_env.py'.
    """

    def __init__(self, n=5, max_fuel=5000, obstacle_prob=0.1):
        """
        :param n: Grid dimension (n x n), must be >= 5
        :param max_fuel: Fuel limit
        :param obstacle_prob: Probability of any given cell being an obstacle
        """
        super().__init__()
        self.n = max(5, n)               # ensure n >= 5
        self.max_fuel = max_fuel
        self.obstacle_prob = obstacle_prob

        # Action space: 0=South,1=North,2=East,3=West,4=Pickup,5=Dropoff
        self.action_space = spaces.Discrete(6)

        # We replicate the same “obs” tuple shape from simple_custom_taxi_env:
        # (taxi_row, taxi_col,
        #  Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
        #  obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        #  passenger_look, destination_look)
        # Each entry is an integer or boolean (converted to int).
        # For safety, define a large enough observation space; you can refine exact bounds if you like.
        obs_high = np.array([
            self.n-1, self.n-1,  # taxi_row, taxi_col
            self.n-1, self.n-1,  # Rrow, Rcol
            self.n-1, self.n-1,  # Grow, Gcol
            self.n-1, self.n-1,  # Yrow, Ycol
            self.n-1, self.n-1,  # Brow, Bcol
            1, 1, 1, 1,          # obstacle_north/south/east/west
            1, 1                 # passenger_look, destination_look
        ], dtype=np.int32)
        obs_low = np.zeros_like(obs_high, dtype=np.int32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)

        # Initialize placeholders
        self.taxi_row = 0
        self.taxi_col = 0
        self.fuel = self.max_fuel
        # We will define stations as R, G, Y, B
        self.stations = []
        self.passenger_loc = None
        self.destination_loc = None
        self.passenger_in_taxi = False

        # 2D array to store obstacles (True=obstacle, False=free)
        self.obstacle_map = np.zeros((self.n, self.n), dtype=bool)

        # For gym compatibility:
        self.reward_range = (-float('inf'), float('inf'))


    def reset(self, seed=None, options=None, random_stations= False):
        super().reset(seed=seed)
        # reset random state
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.fuel = self.max_fuel
        # Randomly place stations
        if random_stations:
            self.stations = []
            for _ in range(4):
                self.stations.append((random.randint(0, self.n-1), random.randint(0, self.n-1))
            )
        else:
            self.stations = []
            self.stations.append((0,0))
            self.stations.append((0, self.n-1))     
            self.stations.append((self.n-1, 0))
            self.stations.append((self.n-1, self.n-1))
            
        # Randomly place obstacles
        self.obstacle_map = np.zeros((self.n, self.n), dtype=bool)
        for r in range(self.n):
            for c in range(self.n):
                # do not place obstacle in corners if you like
                # or do random
                if random.random() < self.obstacle_prob:
                    self.obstacle_map[r, c] = True
                    while (r, c) in self.stations:
                        # ensure no obstacle on station
                        r = random.randint(0, self.n-1)
                        c = random.randint(0, self.n-1)
        
        
        # We will also ensure those corners are not obstacles:
        for (rr, cc) in self.stations:
            self.obstacle_map[rr, cc] = False

        # Make sure the agent spawns in a non-obstacle cell
        while True:
            rr = random.randint(0, self.n-1)
            cc = random.randint(0, self.n-1)
            if not self.obstacle_map[rr, cc]:
                self.taxi_row, self.taxi_col = rr, cc
                break

        # Randomly choose passenger station and destination station
        # passenger station index
        pass_idx = random.randint(0, 3)
        # destination must be different
        dest_idx = random.randint(0, 3)
        while dest_idx == pass_idx:
            dest_idx = random.randint(0, 3)

        self.passenger_loc = self.stations[pass_idx]
        self.destination_loc = self.stations[dest_idx]
        self.passenger_in_taxi = False

        return self._get_obs(), {}


    def step(self, action):
        """
        Action meaning:
        0: Move South
        1: Move North
        2: Move East
        3: Move West
        4: Pick Up
        5: Drop Off
        """
        reward = 0.0
        done = False

        # Decrement fuel for any action
        self.fuel -= 1
        if self.fuel < 0:
            # out of fuel
            return self._get_obs(), -10, True, False, {}

        old_row, old_col = self.taxi_row, self.taxi_col

        # Handle movement
        if action == 0:  # South
            new_row = self.taxi_row + 1
            new_col = self.taxi_col
        elif action == 1:  # North
            new_row = self.taxi_row - 1
            new_col = self.taxi_col
        elif action == 2:  # East
            new_row = self.taxi_row
            new_col = self.taxi_col + 1
        elif action == 3:  # West
            new_row = self.taxi_row
            new_col = self.taxi_col - 1
        else:
            new_row = self.taxi_row
            new_col = self.taxi_col
            
        # Check if there's an obstacle in the new cell or move out of boundary. If so, revert and penalize
        if action in [0,1,2,3]:
            if new_row < 0 or new_row >= self.n or new_col < 0 or new_col >= self.n or self.obstacle_map[new_row, new_col]:
                # obstacle => can't move
                reward += -5
                # remain in old position
                new_row, new_col = old_row, old_col
            else:
                # normal movement cost
                reward += -0.1

        # update taxi location
        self.taxi_row, self.taxi_col = new_row, new_col

        # Handle pickup
        if action == 4:
            # if taxi is at passenger location and passenger not already in taxi
            if not self.passenger_in_taxi and (self.taxi_row, self.taxi_col) == self.passenger_loc:
                self.passenger_in_taxi = True
            else:
                # incorrect pickup
                reward += -10

        # Handle dropoff
        if action == 5:
            # if passenger in taxi and we're at destination
            if self.passenger_in_taxi and (self.taxi_row, self.taxi_col) == self.destination_loc:
                # success
                reward += 50
                done = True
            else:
                # incorrect dropoff
                reward += -10

        # If we are done because of success or out of steps:
        if self.fuel <= 0 and not done:
            # ran out of fuel => end with penalty
            reward += -10
            done = True

        return self._get_obs(), reward, done, False, {}


    def _get_obs(self):
        """
        Must match the format from simple_custom_taxi_env.py:
        (taxi_row, taxi_col,
         Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
         obstacle_north, obstacle_south, obstacle_east, obstacle_west,
         passenger_look, destination_look)
        """
        # Just extract from self.stations
        Rrow, Rcol = self.stations[0]
        Grow, Gcol = self.stations[1]
        Yrow, Ycol = self.stations[2]
        Brow, Bcol = self.stations[3]

        # Obstacle booleans for the tile directly adjacent
        obstacle_north = int(
            self.taxi_row == 0 or self.obstacle_map[self.taxi_row - 1, self.taxi_col]
        )
        obstacle_south = int(
            self.taxi_row == self.n-1 or self.obstacle_map[self.taxi_row + 1, self.taxi_col]
        )
        obstacle_east = int(
            self.taxi_col == self.n-1 or self.obstacle_map[self.taxi_row, self.taxi_col + 1]
        )
        obstacle_west = int(
            self.taxi_col == 0 or self.obstacle_map[self.taxi_row, self.taxi_col - 1]
        )

        # passenger_look => is passenger in one of the adjacent squares or same square
        # but passenger_look is from original code: it was checking if passenger loc is exactly north/south/east/west or same
        # We'll do something similar.
        passenger_look = 0
        if not self.passenger_in_taxi:
            p_r, p_c = self.passenger_loc
            if abs(p_r - self.taxi_row) + abs(p_c - self.taxi_col) == 0 or \
               abs(p_r - self.taxi_row) + abs(p_c - self.taxi_col) == 1:
                passenger_look = 1

        # destination_look => same idea but for the destination
        destination_look = 0
        # even if passenger not in taxi, we can still "sense" the destination near us
        d_r, d_c = self.destination_loc
        if abs(d_r - self.taxi_row) + abs(d_c - self.taxi_col) == 0 or \
           abs(d_r - self.taxi_row) + abs(d_c - self.taxi_col) == 1:
            destination_look = 1

        obs = (
            self.taxi_row, self.taxi_col,
            Rrow, Rcol, Grow, Gcol, Yrow, Ycol, Brow, Bcol,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west,
            passenger_look, destination_look
        )
        return np.array(obs, dtype=np.int32)


    def render(self, mode='human'):
        """Very minimal text-based rendering. Just for debug."""
        for r in range(self.n):
            row_str = ''
            for c in range(self.n):
                if self.taxi_row == r and self.taxi_col == c:
                    row_str += '🚖 '
                elif (r, c) == self.passenger_loc and not self.passenger_in_taxi:
                    row_str += 'P  '
                elif (r, c) == self.destination_loc:
                    row_str += 'D  '
                elif self.obstacle_map[r, c]:
                    row_str += 'X  '
                elif (r, c) in self.stations:
                    # figure out which station
                    idx = self.stations.index((r,c))
                    if idx == 0: row_str += 'R  '
                    if idx == 1: row_str += 'G  '
                    if idx == 2: row_str += 'Y  '
                    if idx == 3: row_str += 'B  '
                else:
                    row_str += '.  '
            print(row_str)

        print(f"Fuel: {self.fuel} / {self.max_fuel}")
        print(f"Passenger in taxi? {self.passenger_in_taxi}\n")


class EvaluationTaxiEnv(TrainingTaxiEnv):
    """
    An evaluation environment that might have a similar or identical
    setup to TrainingTaxiEnv but potentially with different random seeds
    or parameters. In practice, you could load a *fixed* environment
    or randomize it differently. The main requirement:
    the observation must remain the same shape and meaning.
    """

    def __init__(self, n=5, max_fuel=5000, obstacle_prob=0.15):
        # maybe different obstacle probability or some other variation
        super().__init__(n=n, max_fuel=max_fuel, obstacle_prob=obstacle_prob)

    # Optionally override reset or step if you want slightly different logic
    # for evaluation, or keep it the same.
    # def reset(self, seed=None, options=None):
    #     return super().reset(seed=seed, options=options)

    # def step(self, action):
    #     return super().step(action)


def demo():
    """
    Quick demonstration of how you might run these environments.
    """
    # Create a training env with random obstacles, etc.
    train_env = TrainingTaxiEnv(n=6, obstacle_prob=0.2)
    obs, _ = train_env.reset()
    done = False
    total_reward = 0
    while not done:
        # For debugging, pick random action
        action = train_env.action_space.sample()
        obs, reward, done, _, _ = train_env.step(action)
        total_reward += reward

    print(f"[Training Env] Episode ended with total reward = {total_reward}\n")

    # Create an evaluation env (maybe bigger or same)
    eval_env = EvaluationTaxiEnv(n=7, obstacle_prob=0.25)
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        # Again, random action
        action = eval_env.action_space.sample()
        obs, reward, done, _, _ = eval_env.step(action)
        total_reward += reward

    print(f"[Evaluation Env] Episode ended with total reward = {total_reward}")


if __name__ == "__main__":
    demo()