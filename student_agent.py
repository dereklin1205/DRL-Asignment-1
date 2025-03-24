import numpy as np
import random
import pickle
from collections import defaultdict
from simple_custom_taxi_env import SimpleTaxiEnv
from tqdm import *

# Action constants
ACTION_SOUTH = 0
ACTION_NORTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_PICKUP = 4
ACTION_DROPOFF = 5

# Global variables
visited = []
unvisited = []
destionation_station = []
passenger_station = []
passenger_place = None

def find_nearest_station(taxi_row, taxi_col, stations):
    min_distance = 1000
    nearest_station = []
    for station in stations:
        distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
        if distance < min_distance:
            min_distance = distance
    # May have multiple nearest stations, add all of them
    for station in stations:
        distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
        if distance == min_distance:
            nearest_station.append(station)
    return nearest_station

def parse_state(obs, passenger_on, stage_0, stage_1, visited, unvisited, destionation_station, passenger_station, passenger_place):
    """Parse the state from observation"""
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
    
    goal = ()
    dx0 = st0_row - taxi_row
    dy0 = st0_col - taxi_col
    dx1 = st1_row - taxi_row
    dy1 = st1_col - taxi_col
    dx2 = st2_row - taxi_row
    dy2 = st2_col - taxi_col
    dx3 = st3_row - taxi_row
    dy3 = st3_col - taxi_col
    
    if len(visited) == 4:
        visited = []
        unvisited = [(st0_row, st0_col), (st1_row, st1_col), (st2_row, st2_col), (st3_row, st3_col)]
    
    # Update passenger and destination stations
    if passenger_look:
        if passenger_station and len(passenger_station)>1:
            passenger_station = find_nearest_station(taxi_row, taxi_col, [(st0_row, st0_col), (st1_row, st1_col), (st2_row, st2_col), (st3_row, st3_col)])
        else:
            if not passenger_station:
                passenger_station = find_nearest_station(taxi_row, taxi_col, [(st0_row, st0_col), (st1_row, st1_col), (st2_row, st2_col), (st3_row, st3_col)])
    if destination_look:
        if destionation_station and len(destionation_station)>1:
            destionation_station = find_nearest_station(taxi_row, taxi_col, [(st0_row, st0_col), (st1_row, st1_col), (st2_row, st2_col), (st3_row, st3_col)])
        else:
            if not destionation_station:
                destionation_station = find_nearest_station(taxi_row, taxi_col, [(st0_row, st0_col), (st1_row, st1_col), (st2_row, st2_col), (st3_row, st3_col)])   
    
    # Update visited and unvisited stations
    if dx0 == 0 and dy0 == 0:
        if (st0_row, st0_col) not in visited:
            visited.append((st0_row, st0_col))
        if (st0_row, st0_col) in unvisited:
            unvisited.remove((st0_row, st0_col))
    if dx1 == 0 and dy1 == 0:
        if (st1_row, st1_col) not in visited:
            visited.append((st1_row, st1_col))
        if (st1_row, st1_col) in unvisited:
            unvisited.remove((st1_row, st1_col))
    if dx2 == 0 and dy2 == 0:
        if (st2_row, st2_col) not in visited:
            visited.append((st2_row, st2_col))
        if (st2_row, st2_col) in unvisited:
            unvisited.remove((st2_row, st2_col))
    if dx3 == 0 and dy3 == 0:
        if (st3_row, st3_col) not in visited:
            visited.append((st3_row, st3_col))
        if (st3_row, st3_col) in unvisited:
            unvisited.remove((st3_row, st3_col))
    
    # Determine goal based on passenger status
    if passenger_on == 0:
        if passenger_station:
            if len(passenger_station) >= 1:
                goal = (passenger_station[0][0] - taxi_row, passenger_station[0][1] - taxi_col)
        else:
            if unvisited:
                goal = (unvisited[0][0] - taxi_row, unvisited[0][1] - taxi_col)
            else:
                goal = (visited[0][0] - taxi_row, visited[0][1] - taxi_col) 
    else:
        if destionation_station:
            if len(destionation_station) >= 1:
                goal = (destionation_station[0][0] - taxi_row, destionation_station[0][1] - taxi_col)
        else:
            if unvisited:
                goal = (unvisited[0][0] - taxi_row, unvisited[0][1] - taxi_col)
            else:
                goal = (visited[0][0] - taxi_row, visited[0][1] - taxi_col) 
    
    # Update destination stations that are not the destination
    if dx0 == 0 and dy0 == 0 and not destination_look:
        if (st0_row, st0_col) in destionation_station:
            destionation_station.remove((st0_row, st0_col))
    if dx1 == 0 and dy1 == 0 and not destination_look:
        if (st1_row, st1_col) in destionation_station:
            destionation_station.remove((st1_row, st1_col))
    if dx2 == 0 and dy2 == 0 and not destination_look:
        if (st2_row, st2_col) in destionation_station:
            destionation_station.remove((st2_row, st2_col))
    if dx3 == 0 and dy3 == 0 and not destination_look:
        if (st3_row, st3_col) in destionation_station:
            destionation_station.remove((st3_row, st3_col))
    
    # Update passenger stations that are not the passenger location
    if dx0 == 0 and dy0 == 0 and not passenger_look:
        if (st0_row, st0_col) in passenger_station:
            passenger_station.remove((st0_row, st0_col))
    if dx1 == 0 and dy1 == 0 and not passenger_look:
        if (st1_row, st1_col) in passenger_station:
            passenger_station.remove((st1_row, st1_col))
    if dx2 == 0 and dy2 == 0 and not passenger_look:
        if (st2_row, st2_col) in passenger_station:
            passenger_station.remove((st2_row, st2_col))
    if dx3 == 0 and dy3 == 0 and not passenger_look:
        if (st3_row, st3_col) in passenger_station:
            passenger_station.remove((st3_row, st3_col))
    
    # Use known passenger place if available
    if not passenger_on and passenger_place is not None:
        goal = (passenger_place[0] - taxi_row, passenger_place[1] - taxi_col)
    
    parsed_state = (
        goal,
        obstacle_north,
        obstacle_south,
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
        passenger_on
    )
    
    return parsed_state, visited, unvisited, destionation_station, passenger_station, passenger_place

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
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q[state])

    def learn(self, state, action, reward, next_state, done):
        """Q-learning update"""
        old_value = self.Q[state][action]
        best_next = 0.0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - old_value
        self.Q[state][action] += self.alpha * td_error

    def update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename="q_table.pkl"):
        """Save Q-table to file"""
        with open(filename, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, filename="q_table.pkl"):
        """Load Q-table from file"""
        with open(filename, "rb") as f:
            loaded_dict = pickle.load(f)
        self.Q = loaded_dict
        print("Q-table loaded from", filename)

def print_last_episode_info(episode_info):
    """Print detailed information about the last episode"""
    print("\n==== Last Episode Details ====")
    print(f"Episode Length: {len(episode_info['actions'])} steps")
    print(f"Total Reward: {episode_info['total_reward']:.2f}")
    print(f"Passenger Picked Up: {'Yes' if episode_info['passenger_picked_up'] else 'No'}")
    print(f"Destination Reached: {'Yes' if episode_info['destination_reached'] else 'No'}")
    
    # Print debug info
    if 'debug_info' in episode_info:
        debug = episode_info['debug_info']
        print("\nDebug Information:")
        print(f"Wall Collisions: {debug['wall_hits']}")
        print(f"Invalid Pickups: {debug['invalid_pickups']}")
        print(f"Invalid Dropoffs: {debug['invalid_dropoffs']}")
        
        if debug['first_pickup_step'] is not None:
            print(f"First Pickup At: Step {debug['first_pickup_step']}")
        else:
            print("First Pickup At: Never")
            
        if debug['successful_dropoff_step'] is not None:
            print(f"Successful Dropoff At: Step {debug['successful_dropoff_step']}")
        else:
            print("Successful Dropoff At: Never")
    
    print("\nAction Sequence:")
    action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
    for i, action in enumerate(episode_info['actions']):
        action_str = action_names[action]
        position = episode_info['positions'][i] if i < len(episode_info['positions']) else "N/A"
        reward = episode_info['rewards'][i] if i < len(episode_info['rewards']) else "N/A"
        passenger = "With passenger" if episode_info['passenger_statuses'][i] else "No passenger"
        
        print(f"Step {i+1}: {action_str} at {position} → Reward: {reward}, {passenger}")
    
    print("\nFinal Status:")
    if episode_info['destination_reached']:
        print("✅ SUCCESS: Passenger successfully delivered!")
    elif episode_info['passenger_picked_up']:
        print("❌ FAILED: Passenger picked up but not delivered")
    else:
        print("❌ FAILED: Passenger not picked up")
    print("============================")

def train(env, agent, num_episodes=1000):
    """
    Training function with additional debugging metrics
    """
    all_rewards = []
    all_steps = []
    
    # For tracking success rates
    batch_size = 100
    batch_pickups = 0   # Successful pickups in current batch
    batch_deliveries = 0  # Successful deliveries in current batch
    
    # For advanced debugging
    batch_wall_hits = 0
    batch_invalid_pickups = 0
    batch_invalid_dropoffs = 0
    batch_pickup_steps = []  # Steps to first pickup in each episode
    batch_dropoff_steps = []  # Steps to successful dropoff in each episode
    
    # For tracking across all batches
    batch_rewards = []  # Average reward per batch
    batch_steps = []    # Average steps per batch
    batch_pickup_rates = []  # Pickup rate per batch
    batch_delivery_rates = []  # Delivery rate per batch
    
    # Advanced debugging metrics across batches
    batch_avg_wall_hits = []
    batch_avg_invalid_pickups = []
    batch_avg_invalid_dropoffs = []
    batch_avg_pickup_time = []
    batch_avg_dropoff_time = []
    
    # For tracking current batch
    current_batch_rewards = []
    current_batch_steps = []
    
    # For tracking last episode in each batch
    last_episode_info = None
    
    global visited
    global unvisited
    global destionation_station
    global passenger_station
    global passenger_place
    
    for episode in range(num_episodes):
        # Episode initialization
        passenger_on = 0
        stage_0 = 0
        stage_1 = 0
        raw_obs, _ = env.reset()
        unvisited = [(raw_obs[2], raw_obs[3]), (raw_obs[4], raw_obs[5]), (raw_obs[6], raw_obs[7]), (raw_obs[8], raw_obs[9])]
        visited = []
        destionation_station = []
        passenger_station = []
        passenger_place = None
        
        state, visited, unvisited, destionation_station, passenger_station, passenger_place = parse_state(
            raw_obs, passenger_on, stage_0, stage_1, visited, unvisited, destionation_station, passenger_station, None
        )
        
        total_reward = 0.0
        done = False
        steps = 0
        
        # Debug tracking for this episode
        episode_wall_hits = 0
        episode_invalid_pickups = 0
        episode_invalid_dropoffs = 0
        first_pickup_step = None
        successful_dropoff_step = None
        
        # For tracking detailed information about this episode
        episode_tracking = {
            'actions': [],
            'positions': [],
            'rewards': [],
            'passenger_statuses': [],
            'total_reward': 0,
            'passenger_picked_up': False,
            'destination_reached': False,
            'debug_info': {
                'wall_hits': 0,
                'invalid_pickups': 0,
                'invalid_dropoffs': 0,
                'first_pickup_step': None,
                'successful_dropoff_step': None
            }
        }
        
        action_all = []
        queue = []
        
        # For tracking current episode
        episode_pickup = False
        episode_delivery = False
        
        while not done:
            # Select action
            action = agent.act(state)
            action_all.append(action)
            
            # Track this step
            episode_tracking['actions'].append(action)
            episode_tracking['positions'].append((raw_obs[0], raw_obs[1]))
            episode_tracking['passenger_statuses'].append(passenger_on)
            
            # Handle pickup/dropoff logic
            if action == ACTION_PICKUP and passenger_on == 0 and state[5] == True and state[0] == (0, 0):
                passenger_on = 1
                episode_tracking['passenger_picked_up'] = True
                episode_pickup = True
                
                # Record first successful pickup
                if first_pickup_step is None:
                    first_pickup_step = steps
            
            if action == ACTION_DROPOFF and passenger_on:
                passenger_on = 0
                passenger_place = (raw_obs[0], raw_obs[1])
            
            # Take action in environment
            raw_next_obs, reward, done, info = env.step(action)
            episode_tracking['rewards'].append(reward)
            
            # Track debugging information based on rewards and actions
            # Wall hits (movement action with negative reward)
            if action in [ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST] and reward <= -5:
                episode_wall_hits += 1
                batch_wall_hits += 1
                
            # Invalid pickups
            if action == ACTION_PICKUP and reward <= -5:
                episode_invalid_pickups += 1
                batch_invalid_pickups += 1
                
            # Invalid dropoffs
            if action == ACTION_DROPOFF and reward <= -5:
                episode_invalid_dropoffs += 1
                batch_invalid_dropoffs += 1
                
            # Successful dropoff (completion)
            if done and action == ACTION_DROPOFF and reward > 0:
                successful_dropoff_step = steps
                episode_delivery = True
            
            # Additional logic for loop prevention
            if action in queue:
                reward -= 3
            queue.append(action)
            if len(queue) == 4:
                queue.pop(0)
            
            # Calculate next state
            next_state, next_visited, next_unvisited, next_destionation_station, next_passenger_station, next_passenger_place = parse_state(
                raw_next_obs, passenger_on, stage_0, stage_1, visited, unvisited, destionation_station, passenger_station, passenger_place
            )
            
            # Reward shaping based on movement toward goal
            if (abs(next_state[0][0]) + abs(next_state[0][1])) < (abs(state[0][0]) + abs(state[0][1])):
                reward += 2
            else:
                reward -= 2
            
            # Track successful pickups
            if info.get("pick_up_passenger", False):
                episode_pickup = True
                
                # Record first successful pickup if not already recorded
                if first_pickup_step is None:
                    first_pickup_step = steps
            
            # Q-learning update
            agent.learn(state, action, reward, next_state, done)
            
            # Update current state
            state = next_state
            raw_obs = raw_next_obs
            visited = next_visited
            unvisited = next_unvisited
            destionation_station = next_destionation_station
            passenger_station = next_passenger_station
            passenger_place = next_passenger_place
            
            total_reward += reward
            steps += 1
        
        # Update batch counters
        if episode_pickup:
            batch_pickups += 1
            batch_pickup_steps.append(first_pickup_step if first_pickup_step is not None else steps)
        
        if episode_delivery:
            batch_deliveries += 1
            batch_dropoff_steps.append(successful_dropoff_step)
        
        # Store total reward for this episode
        episode_tracking['total_reward'] = total_reward
        all_rewards.append(total_reward)
        all_steps.append(steps)
        
        # Store for current batch
        current_batch_rewards.append(total_reward)
        current_batch_steps.append(steps)
        
        # Update debugging info
        episode_tracking['debug_info'] = {
            'wall_hits': episode_wall_hits,
            'invalid_pickups': episode_invalid_pickups,
            'invalid_dropoffs': episode_invalid_dropoffs,
            'first_pickup_step': first_pickup_step,
            'successful_dropoff_step': successful_dropoff_step
        }
        
        # Update exploration rate
        agent.update_epsilon()
        
        # If this is the last episode in a batch, store its info and print statistics
        if (episode + 1) % batch_size == 0:
            # Store the last episode info
            last_episode_info = episode_tracking
            
            # Calculate batch statistics
            current_batch_avg_reward = np.mean(current_batch_rewards)
            current_batch_avg_steps = np.mean(current_batch_steps)
            current_batch_pickup_rate = (batch_pickups / batch_size) * 100
            current_batch_delivery_rate = (batch_deliveries / batch_size) * 100
            
            # Calculate advanced debugging metrics
            current_batch_avg_wall_hits = batch_wall_hits / batch_size
            current_batch_avg_invalid_pickups = batch_invalid_pickups / batch_size
            current_batch_avg_invalid_dropoffs = batch_invalid_dropoffs / batch_size
            
            # Calculate average pickup and dropoff times (only for successful episodes)
            current_batch_avg_pickup_time = np.mean(batch_pickup_steps) if batch_pickup_steps else float('inf')
            current_batch_avg_dropoff_time = np.mean(batch_dropoff_steps) if batch_dropoff_steps else float('inf')
            
            # Store batch statistics
            batch_rewards.append(current_batch_avg_reward)
            batch_steps.append(current_batch_avg_steps)
            batch_pickup_rates.append(current_batch_pickup_rate)
            batch_delivery_rates.append(current_batch_delivery_rate)
            
            # Store advanced debugging metrics
            batch_avg_wall_hits.append(current_batch_avg_wall_hits)
            batch_avg_invalid_pickups.append(current_batch_avg_invalid_pickups)
            batch_avg_invalid_dropoffs.append(current_batch_avg_invalid_dropoffs)
            batch_avg_pickup_time.append(current_batch_avg_pickup_time)
            batch_avg_dropoff_time.append(current_batch_avg_dropoff_time)
            
            # Get batch number
            batch_num = (episode + 1) // batch_size
            
            # Print current batch summary
            print(f"\n=== Batch {batch_num} (Episodes {episode+1-batch_size+1}-{episode+1}) ===")
            print(f"Epsilon: {agent.epsilon:.4f}")
            
            # Print comparison table
            print("\n=== Batch Comparison ===")
            print("Metric                | Previous Batch     | Current Batch")
            print("----------------------|-------------------|------------------")
            
            # Previous batch info (if available)
            if batch_num > 1:
                prev_reward = batch_rewards[-2]
                prev_steps = batch_steps[-2]
                prev_pickup = batch_pickup_rates[-2]
                prev_delivery = batch_delivery_rates[-2]
                
                prev_wall_hits = batch_avg_wall_hits[-2]
                prev_invalid_pickups = batch_avg_invalid_pickups[-2]
                prev_invalid_dropoffs = batch_avg_invalid_dropoffs[-2]
                prev_pickup_time = batch_avg_pickup_time[-2]
                prev_dropoff_time = batch_avg_dropoff_time[-2]
                
                print(f"Avg Reward            | {prev_reward:10.2f}       | {current_batch_avg_reward:10.2f}")
                print(f"Avg Steps             | {prev_steps:10.2f}       | {current_batch_avg_steps:10.2f}")
                print(f"Pickup Rate           | {prev_pickup:10.2f}%      | {current_batch_pickup_rate:10.2f}%")
                print(f"Delivery Rate         | {prev_delivery:10.2f}%      | {current_batch_delivery_rate:10.2f}%")
                
                print("\n=== Advanced Debugging Metrics ===")
                print(f"Avg Wall Hits         | {prev_wall_hits:10.2f}       | {current_batch_avg_wall_hits:10.2f}")
                print(f"Avg Invalid Pickups   | {prev_invalid_pickups:10.2f}       | {current_batch_avg_invalid_pickups:10.2f}")
                print(f"Avg Invalid Dropoffs  | {prev_invalid_dropoffs:10.2f}       | {current_batch_avg_invalid_dropoffs:10.2f}")
                
                # Handle infinite values for better display
                prev_pickup_str = f"{prev_pickup_time:10.2f}" if prev_pickup_time != float('inf') else "    N/A    "
                curr_pickup_str = f"{current_batch_avg_pickup_time:10.2f}" if current_batch_avg_pickup_time != float('inf') else "    N/A    "
                
                prev_dropoff_str = f"{prev_dropoff_time:10.2f}" if prev_dropoff_time != float('inf') else "    N/A    "
                curr_dropoff_str = f"{current_batch_avg_dropoff_time:10.2f}" if current_batch_avg_dropoff_time != float('inf') else "    N/A    "
                
                print(f"Avg Steps to Pickup   | {prev_pickup_str}       | {curr_pickup_str}")
                print(f"Avg Steps to Dropoff  | {prev_dropoff_str}       | {curr_dropoff_str}")
            else:
                # No previous batch
                print(f"Avg Reward            | N/A              | {current_batch_avg_reward:10.2f}")
                print(f"Avg Steps             | N/A              | {current_batch_avg_steps:10.2f}")
                print(f"Pickup Rate           | N/A              | {current_batch_pickup_rate:10.2f}%")
                print(f"Delivery Rate         | N/A              | {current_batch_delivery_rate:10.2f}%")
                
                print("\n=== Advanced Debugging Metrics ===")
                print(f"Avg Wall Hits         | N/A              | {current_batch_avg_wall_hits:10.2f}")
                print(f"Avg Invalid Pickups   | N/A              | {current_batch_avg_invalid_pickups:10.2f}")
                print(f"Avg Invalid Dropoffs  | N/A              | {current_batch_avg_invalid_dropoffs:10.2f}")
                
                # Handle infinite values for better display
                curr_pickup_str = f"{current_batch_avg_pickup_time:10.2f}" if current_batch_avg_pickup_time != float('inf') else "    N/A    "
                curr_dropoff_str = f"{current_batch_avg_dropoff_time:10.2f}" if current_batch_avg_dropoff_time != float('inf') else "    N/A    "
                
                print(f"Avg Steps to Pickup   | N/A              | {curr_pickup_str}")
                print(f"Avg Steps to Dropoff  | N/A              | {curr_dropoff_str}")
            
            # Print detailed information about the last episode
            # print_last_episode_info(last_episode_info)
            
            # Reset batch counters and trackers
            batch_pickups = 0
            batch_deliveries = 0
            batch_wall_hits = 0
            batch_invalid_pickups = 0
            batch_invalid_dropoffs = 0
            batch_pickup_steps = []
            batch_dropoff_steps = []
            current_batch_rewards = []
            current_batch_steps = []
            
            # Save model periodically
            if (episode + 1) % (batch_size * 10) == 0:
                agent.save(f"q_table_{episode+1}.pkl")
    
    # Save final model
    agent.save("q_table.pkl")
    
    # Return training history
    return {
        "rewards": all_rewards,
        "steps": all_steps,
        "batch_rewards": batch_rewards,
        "batch_steps": batch_steps,
        "batch_pickup_rates": batch_pickup_rates,
        "batch_delivery_rates": batch_delivery_rates,
        "batch_avg_wall_hits": batch_avg_wall_hits,
        "batch_avg_invalid_pickups": batch_avg_invalid_pickups,
        "batch_avg_invalid_dropoffs": batch_avg_invalid_dropoffs,
        "batch_avg_pickup_time": batch_avg_pickup_time,
        "batch_avg_dropoff_time": batch_avg_dropoff_time
    }

def get_action(obs):
    """Function called during testing to get the next action"""
    # Initialize on first call
    if not hasattr(get_action, "agent"):
        get_action.agent = QLearningAgent(epsilon=0.0)  # no exploration in test
        get_action.agent.load("q_table.pkl")
        get_action.stage_0 = 0
        get_action.stage_1 = 0
        get_action.passenger_on = 0  # not carrying passenger
        get_action.last_state = None
        get_action.last_action = None
        get_action.visited = []
        get_action.unvisited = []
        get_action.destionation_station = []
        get_action.passenger_station = []
        get_action.passenger_place = None
        get_action.queue = []
    
    # Update based on previous action
    if get_action.last_state is not None and get_action.last_action is not None:
        prev_state = get_action.last_state
        prev_action = get_action.last_action
        
        if prev_action == ACTION_PICKUP:
            if prev_state[5] == 1 and get_action.passenger_on == 0 and prev_state[0] == (0, 0):
                get_action.passenger_on = 1
        elif prev_action == ACTION_DROPOFF:
            if get_action.passenger_on == 1:
                get_action.passenger_on = 0
                get_action.passenger_place = (obs[0], obs[1])
    
    # Parse current state
    state, get_action.visited, get_action.unvisited, get_action.destionation_station, get_action.passenger_station, get_action.passenger_place = parse_state(
        obs, get_action.passenger_on, get_action.stage_0, get_action.stage_1, 
        get_action.visited, get_action.unvisited, get_action.destionation_station, 
        get_action.passenger_station, get_action.passenger_place
    )
    if len(get_action.queue)<11:
        get_action.queue.append(action)
    elif len(get_action.queue)==10:
        get_action.queue.pop(0)
    
    # Get action from Q-table
    Q_values = get_action.agent.Q.get(state, None)
    if Q_values is None:
        action = random.randint(0, 5)
    else:
        action = np.argmax(Q_values)
        count = 0
        for action_q in get_action.queue:
            if action == action_q:
                count +=1
        if len(get_action.queue) == 10:
            if count > 5:
                ## pick second large Q value action
                action = np.argsort(Q_values)[-2]
    
    # Remember current state and action
    get_action.last_state = state
    get_action.last_action = action
    
    return action

if __name__ == "__main__":
    # Create environment and agent
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99995,
        action_size=6
    )
    
    env = SimpleTaxiEnv(10, 5000)
    
    # Extend SimpleTaxiEnv to add detailed tracking
    original_step = env.step
    
    def step_with_tracking(action):
        next_obs, reward, done, info = original_step(action)
        if not isinstance(info, dict):
            info = {}
            
        # Track successful pickup
        if action == ACTION_PICKUP and reward >= 0:
            info["pick_up_passenger"] = True
        else:
            info["pick_up_passenger"] = False
            
        # Track wall hits
        if action in [ACTION_NORTH, ACTION_SOUTH, ACTION_EAST, ACTION_WEST] and reward <= -5:
            info["wall_hit"] = True
        else:
            info["wall_hit"] = False
            
        # Track invalid pickups
        if action == ACTION_PICKUP and reward <= -5:
            info["invalid_pickup"] = True
        else:
            info["invalid_pickup"] = False
            
        # Track invalid dropoffs
        if action == ACTION_DROPOFF and reward <= -5:
            info["invalid_dropoff"] = True
        else:
            info["invalid_dropoff"] = False
            
        return next_obs, reward, done, info
    
    env.step = step_with_tracking
    
    # Train agent
    print("Starting training...")
    try:
        history = train(env, agent, num_episodes=30000)
        
        # Second round of training with reset epsilon
        agent.epsilon = 1.0
        print("\nStarting second round of training...")
        history = train(env, agent, num_episodes=30000)
        
        print("Training complete! Final epsilon =", agent.epsilon)
        
        # Try to create a plot of the training history
        try:
            import matplotlib.pyplot as plt
            
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            
            # Plot 1: Success Rates
            axes[0, 0].plot(history['batch_pickup_rates'], label='Pickup Rate')
            axes[0, 0].plot(history['batch_delivery_rates'], label='Delivery Rate')
            axes[0, 0].set_title('Success Rates per Batch')
            axes[0, 0].set_xlabel('Batch')
            axes[0, 0].set_ylabel('Success Rate (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: Rewards and Steps
            ax1 = axes[0, 1]
            ax2 = ax1.twinx()
            
            l1, = ax1.plot(history['batch_rewards'], 'b-', label='Avg Reward')
            l2, = ax2.plot(history['batch_steps'], 'r-', label='Avg Steps')
            
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('Average Reward', color='b')
            ax2.set_ylabel('Average Steps', color='r')
            
            ax1.legend(handles=[l1, l2], loc='upper left')
            ax1.grid(True)
            ax1.set_title('Reward and Steps per Batch')
            
            # Plot 3: Wall Hits
            axes[1, 0].plot(history['batch_avg_wall_hits'])
            axes[1, 0].set_title('Average Wall Hits per Batch')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Average Wall Hits')
            axes[1, 0].grid(True)
            
            # Plot 4: Invalid Pickups and Dropoffs
            axes[1, 1].plot(history['batch_avg_invalid_pickups'], label='Invalid Pickups')
            axes[1, 1].plot(history['batch_avg_invalid_dropoffs'], label='Invalid Dropoffs')
            axes[1, 1].set_title('Invalid Actions per Batch')
            axes[1, 1].set_xlabel('Batch')
            axes[1, 1].set_ylabel('Average Invalid Actions')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # Plot 5: Pickup and Dropoff Times
            # Filter out inf values
            pickup_times = [t if t != float('inf') else None for t in history['batch_avg_pickup_time']]
            dropoff_times = [t if t != float('inf') else None for t in history['batch_avg_dropoff_time']]
            
            # Replace None with NaN for plotting
            pickup_times = np.array([float('nan') if t is None else t for t in pickup_times])
            dropoff_times = np.array([float('nan') if t is None else t for t in dropoff_times])
            
            axes[2, 0].plot(pickup_times, label='Time to Pickup')
            axes[2, 0].plot(dropoff_times, label='Time to Dropoff')
            axes[2, 0].set_title('Average Time to Complete Tasks')
            axes[2, 0].set_xlabel('Batch')
            axes[2, 0].set_ylabel('Average Steps')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            # Plot 6: Efficiency (Dropoff/Pickup Ratio)
            delivery_pickup_ratio = []
            for pickup_rate, delivery_rate in zip(history['batch_pickup_rates'], history['batch_delivery_rates']):
                ratio = (delivery_rate / pickup_rate) * 100 if pickup_rate > 0 else 0
                delivery_pickup_ratio.append(ratio)
            
            axes[2, 1].plot(delivery_pickup_ratio)
            axes[2, 1].set_title('Delivery to Pickup Ratio per Batch')
            axes[2, 1].set_xlabel('Batch')
            axes[2, 1].set_ylabel('Ratio (%)')
            axes[2, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            print("\nTraining history plot saved to 'training_history.png'")
        except Exception as e:
            print(f"Could not create plot: {e}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        agent.save("q_table_interrupted.pkl")
        print("Model saved to 'q_table_interrupted.pkl'")
    # Get