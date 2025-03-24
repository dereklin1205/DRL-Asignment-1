import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from collections import defaultdict, deque

# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=50, difficulty="easy"):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.stations = [] 
        self.passenger_loc = None
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None
        self.passenger_ever_picked_up = False
        self.flag  = False
        self.difficulty = difficulty
    def is_valid(self):
        return self.is_reachable(self.taxi_pos, self.passenger_loc) and \
            self.is_reachable(self.passenger_loc, self.destination) and self.is_reachable(self.taxi_pos, self.stations[0]) and self.is_reachable(self.taxi_pos, self.stations[1]) and self.is_reachable(self.taxi_pos, self.stations[2]) and self.is_reachable(self.taxi_pos, self.stations[3])

    def is_reachable(self, p, q):
        """
        Check if there is a path from p to q using BFS
        
        Args:
            p: (row, col) tuple for start position
            q: (row, col) tuple for end position
            
        Returns:
            bool: True if path exists, False otherwise
        """
        if p == q:
            return True
        
        # Directions: up, right, down, left
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # BFS for path finding
        queue = [p]
        visited = set()
        
        while queue:
            x, y = queue.pop(0)
            
            if (x, y) == q:
                return True
            
            if (x, y) in visited:
                continue
                
            visited.add((x, y))
            
            # Try all four directions
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and 
                    0 <= ny < self.grid_size and 
                    (nx, ny) not in self.obstacles):
                    queue.append((nx, ny))
        
        # If we've exhausted all possibilities without finding the end
        return False

    def reset(self, difficulty="easy"):
        self.difficulty = difficulty 
        self.grid_size = random.choice([5, 6, 7, 8,9,10])
        self.passenger_ever_picked_up = False
        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]
        while True:
            self.stations = []
            for _ in range(4):
                x, y = random.choice(available_positions)
                self.stations.append((x, y))
                for dx, dy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                    if (x + dx, y + dy) in available_positions:
                        available_positions.remove((x + dx, y + dy))

            # self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]

            self.current_fuel = self.fuel_limit
            self.passenger_picked_up = False

            available_positions = [
                (x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                if (x, y) not in self.stations
            ]

            self.taxi_pos = random.choice(available_positions)

            self.passenger_loc = random.choice(self.stations)

            self.destination = random.choice([s for s in self.stations
                                            if s != self.passenger_loc])

            available_positions = [
                (x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                if (x, y) not in self.stations and (x, y) != self.taxi_pos
            ]
            self.n_obstacle = int((self.grid_size**2*np.random.uniform(0.2,0.3)))
            if self.difficulty == "easy":
                self.n_obstacle = int((self.grid_size**2*np.random.uniform(0,0.1)))
            else:
                self.n_obstacle = int((self.grid_size**2*np.random.uniform(0.2,0.3)))

            self.obstacles = set(random.sample(available_positions, self.n_obstacle))

            if self.is_valid():
                break

        return self.get_state(), {}

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        
        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -=20
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc and self.passenger_picked_up == False:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    if not self.passenger_ever_picked_up:
                        reward += 50
                    self.passenger_ever_picked_up = True  
                else:
                    reward = -100
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 200
                        return self.get_state(), reward -0.1, True, {"success": True,"pick_up_passenger": True}
                    reward -=1000
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 1000
        reward -= 1  

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward -1, True, {}

        if self.passenger_ever_picked_up == True and self.flag == False:
            self.flag = True
            return self.get_state(), reward, False, {"pick_up_passenger": True}
        

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination
        
        obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west  = int(taxi_col == 0 or (taxi_row , taxi_col-1) in self.obstacles)

        passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
       
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''
        
        
        for station in self.stations:
            y,x = station
            grid[y][x] = 'S'
        for obstacle in self.obstacles:
            y,x = obstacle
            grid[y][x] = 'X'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    # print
    print(f"Final Score: {agent_score}")