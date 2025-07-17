import numpy as np
from collections import defaultdict
import random

# Constants
GRID_SIZE = 9  # 3x3 grid
LAND_REWARD = 100
PENALTY = -100
INSPECT_COST = -1
OBSERVATION_ACCURACY = 0.8
OBSERVATION_NOISE = 0.1
MAX_STEPS = 50

class GridState:
    def __init__(self, cell_states):
        self.cell_states = cell_states  # List of 'L', 'O', 'E'
    
    def __hash__(self):
        return hash(tuple(self.cell_states))
    
    def __eq__(self, other):
        return self.cell_states == other.cell_states

class Action:
    def __init__(self, name, cell_index=None):
        self.name = name
        self.cell_index = cell_index
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name

class Observation:
    def __init__(self, cell_index, observed_state):
        self.cell_index = cell_index
        self.observed_state = observed_state  # 'L', 'O', 'E'

class POMCPNode:
    def __init__(self):
        self.children = {}  # Action -> child nodes
        self.visits = 0
        self.value = 0

class POMCPSolver:
    def __init__(self, max_depth=20, num_sims=1000, exploration_const=50):
        self.max_depth = max_depth
        self.num_sims = num_sims
        self.exploration_const = exploration_const
        self.root = POMCPNode()
    
    def uct_select_action(self, node, possible_actions):
        if node.visits == 0:
            return random.choice(possible_actions)
        
        def uct_value(child_node, action):
            exploitation = child_node.value / child_node.visits
            exploration = np.sqrt(np.log(node.visits) / child_node.visits)
            return exploitation + self.exploration_const * exploration
        
        return max(possible_actions,
                 key=lambda a: uct_value(node.children.get(a, POMCPNode()), a))

class GridPOMDP:
    def __init__(self):
        self.true_state = self._create_random_state()
        self.belief_particles = self._initialize_belief(1000)
    
    def _create_random_state(self):
        cell_states = ['E'] * GRID_SIZE
        landing_pos = random.randint(0, GRID_SIZE - 1)
        cell_states[landing_pos] = 'L'
        obstacle_positions = random.sample(
            [i for i in range(GRID_SIZE) if i != landing_pos], 3)
        for pos in obstacle_positions:
            cell_states[pos] = 'O'
        return GridState(cell_states)
    
    def _initialize_belief(self, num_particles):
        particles = []
        for _ in range(num_particles):
            particles.append(self._create_random_state())
        return particles
    
    def get_possible_actions(self):
        actions = []
        for i in range(GRID_SIZE):
            actions.append(Action(f"inspect-{i}", i))
            actions.append(Action(f"land-{i}", i))
        return actions
    
    def transition(self, state, action):
        return state  # World is static
    
    def observation_prob(self, obs, next_state, action):
        if not action.name.startswith("inspect"):
            return 0.0 if obs is not None else 1.0
        
        true_state = next_state.cell_states[action.cell_index]
        if true_state == obs.observed_state:
            return OBSERVATION_ACCURACY
        return OBSERVATION_NOISE
    
    def sample_observation(self, state, action):
        if not action.name.startswith("inspect"):
            return None
        
        true_state = state.cell_states[action.cell_index]
        r = random.random()
        if r < OBSERVATION_ACCURACY:
            obs_state = true_state
        elif r < OBSERVATION_ACCURACY + OBSERVATION_NOISE:
            obs_state = 'L' if true_state != 'L' else 'E'
        else:
            obs_state = 'O' if true_state != 'O' else 'E'
        return Observation(action.cell_index, obs_state)
    
    def reward(self, state, action, next_state):
        if action.name.startswith("inspect"):
            return INSPECT_COST
        elif action.name.startswith("land"):
            if state.cell_states[action.cell_index] == 'L':
                return LAND_REWARD
            else:
                return PENALTY
        return 0
    
    def update_belief(self, action, observation):
        if not action.name.startswith("inspect"):
            return
        
        weights = []
        for state in self.belief_particles:
            true_state = state.cell_states[action.cell_index]
            if true_state == observation.observed_state:
                weights.append(OBSERVATION_ACCURACY)
            else:
                weights.append(OBSERVATION_NOISE)
        
        if sum(weights) == 0:
            weights = [1/len(weights)] * len(weights)
        else:
            weights = [w/sum(weights) for w in weights]
        
        new_particles = random.choices(
            self.belief_particles, weights=weights, k=len(self.belief_particles))
        self.belief_particles = new_particles
    
    def sample_belief_state(self):
        return random.choice(self.belief_particles)
    
    def get_landing_probabilities(self):
        probs = [0] * GRID_SIZE
        for state in self.belief_particles:
            for i in range(GRID_SIZE):
                if state.cell_states[i] == 'L':
                    probs[i] += 1
        return [p/len(self.belief_particles) for p in probs]

def run_simulation():
    pomdp = GridPOMDP()
    solver = POMCPSolver()
    
    total_reward = 0
    terminated = False
    
    for step in range(MAX_STEPS):
        print(f"\nStep {step + 1}")
        
        # Select action using POMCP
        action = None
        best_value = -float('inf')
        possible_actions = pomdp.get_possible_actions()
        
        for _ in range(solver.num_sims):
            # Sample a state from belief
            state = pomdp.sample_belief_state()
            
            # Run simulation from this state
            value = simulate(pomdp, state, solver.max_depth)
            
            # Update best action
            if value > best_value:
                best_value = value
                action = random.choice(possible_actions)  # Simplified selection
        
        if action is None:
            action = random.choice(possible_actions)
        
        print(f"Action: {action.name}")
        
        # Execute action
        next_state = pomdp.transition(pomdp.true_state, action)
        observation = pomdp.sample_observation(pomdp.true_state, action)
        
        if observation:
            print(f"Observation: Cell {observation.cell_index} appears {observation.observed_state}")
        
        reward = pomdp.reward(pomdp.true_state, action, next_state)
        total_reward += reward
        print(f"Reward: {reward}")
        print(f"Total reward: {total_reward}")
        
        # Update belief
        if observation:
            pomdp.update_belief(action, observation)
        
        # Check termination
        if action.name.startswith("land"):
            print("Landing attempted - episode terminated")
            terminated = True
        
        # Update true state (though agent doesn't know it)
        pomdp.true_state = next_state
        
        # Print belief statistics
        landing_probs = pomdp.get_landing_probabilities()
        print("Estimated landing probabilities:")
        for i in range(GRID_SIZE):
            print(f"Cell {i}: {landing_probs[i]:.2f}")
        
        if terminated:
            break
    
    print("\nSimulation complete")
    print(f"Total reward: {total_reward}")
    print("True grid state:")
    for i in range(0, 9, 3):
        print(pomdp.true_state.cell_states[i:i+3])

def simulate(pomdp, state, depth):
    if depth == 0:
        return 0
    
    # Simplified rollout policy - random actions
    action = random.choice(pomdp.get_possible_actions())
    next_state = pomdp.transition(state, action)
    observation = pomdp.sample_observation(state, action)
    reward = pomdp.reward(state, action, next_state)
    
    if action.name.startswith("land"):
        return reward
    else:
        return reward + 0.95 * simulate(pomdp, next_state, depth - 1)

if __name__ == "__main__":
    run_simulation()