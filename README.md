# codetech-task-6-reinforcement

1. Environment Definition
The environment is a simple grid world where the agent can move up, down, left, or right. The goal is to reach a specific position while avoiding obstacles.

python
Copy code
class GridWorld:
    def __init__(self, size, start, goal, obstacles):
        self.size = size  # Size of the grid (width, height)
        self.start = start  # Starting position of the agent
        self.goal = goal  # Goal position
        self.obstacles = obstacles  # Set of obstacle positions
        self.reset()
    
    def reset(self):
        self.agent_pos = list(self.start)  # Reset the agent's position to the start
        return tuple(self.agent_pos)  # Return the initial state
    
    def step(self, action):
        if action == 'up' and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 'down' and self.agent_pos[1] < self.size[1] - 1:
            self.agent_pos[1] += 1
        elif action == 'left' and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 'right' and self.agent_pos[0] < self.size[0] - 1:
            self.agent_pos[0] += 1
        
        if self.agent_pos == list(self.goal):
            return tuple(self.agent_pos), 1, True  # Reached the goal
        elif tuple(self.agent_pos) in self.obstacles:
            return tuple(self.agent_pos), -1, True  # Hit an obstacle
        else:
            return tuple(self.agent_pos), -0.1, False  # Normal move

# Define actions
actions = ['up', 'down', 'left', 'right']
__init__: Initializes the grid world with its size, start position, goal position, and obstacles.
reset: Resets the agent's position to the start position and returns the initial state.
step: Takes an action and updates the agent's position. It returns the new state, reward, and whether the episode is done.
2. Q-learning Agent
The agent learns to navigate the grid world using the Q-learning algorithm.

python
Copy code
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        self.env = env
        self.q_table = {}  # Initialize the Q-table
        self.learning_rate = learning_rate  # Learning rate
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.exploration_rate = exploration_rate  # Exploration rate for epsilon-greedy strategy
        self.exploration_decay = exploration_decay  # Decay rate for the exploration rate

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)  # Return the Q-value for the state-action pair, default to 0.0

    def choose_action(self, state):
        if random.random() < self.exploration_rate:
            return random.choice(actions)  # Explore: choose a random action
        q_values = [self.get_q_value(state, action) for action in actions]
        max_q = max(q_values)
        return actions[q_values.index(max_q)]  # Exploit: choose the action with the highest Q-value

    def learn(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in actions])
        new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * next_max_q)
        self.q_table[(state, action)] = new_q  # Update the Q-value for the state-action pair

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            self.exploration_rate *= self.exploration_decay  # Decay the exploration rate after each episode
__init__: Initializes the agent with the environment, learning rate, discount factor, exploration rate, and exploration decay.
get_q_value: Retrieves the Q-value for a given state-action pair from the Q-table. If the pair is not in the table, it returns 0.0.
choose_action: Chooses an action based on the epsilon-greedy strategy. With probability exploration_rate, it explores by choosing a random action. Otherwise, it exploits by choosing the action with the highest Q-value.
learn: Updates the Q-value for a state-action pair using the Q-learning update rule.
train: Trains the agent over a specified number of episodes. In each episode, the agent interacts with the environment, updates the Q-values, and decays the exploration rate.
3. Training and Testing
The main block of the code trains the agent and tests its learned policy.
![Uploading WhatsApp Image 2024-08-06 at 9.12.12 PM.jpeg…]()

python
Copy code
# Training the agent
env = GridWorld(size=(5, 5), start=(0, 0), goal=(4, 4), obstacles={(1, 1), (2, 2), (3, 3)})
agent = QLearningAgent(env)
agent.train(episodes=1000)

# Test the trained agent
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}")
Environment Setup: A GridWorld environment is created with a size of 5x5, a start position at (0, 0), a goal at (4, 4), and obstacles at positions (1, 1), (2, 2), and (3, 3).
Agent Training: The QLearningAgent is created with the environment, and the agent is trained for 1000 episodes.
Agent Testing: After training, the agent's policy is tested by resetting the environment and taking actions according to the learned Q-values. 

![Uploading WhatsApp Image 2024-08-06 at 9.12.12 PM.jpeg…]()
