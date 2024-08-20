import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import random
from collections import deque

from QuoridorEnv import QuoridorEnv
from agents import ShortestPathAgent

class DQNModel:
    def __init__(self, input_shape, num_actions):
        self.model = self.build_dqn(input_shape, num_actions)

    def build_dqn(self, input_shape, num_actions):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_actions, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, target_f):
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class DQNAgent:
    def __init__(self, state_shape, num_actions):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.model = DQNModel(state_shape, num_actions)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, invalid_actions):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state)
        print(invalid_actions)
        q_values[invalid_actions] = -np.inf  # Mask invalid actions
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load(name)

    def save(self, name):
        self.model.save(name)
        
        
# Constants for player identifiers
DQN_MODEL = 1
OPPONENT_MODEL = -1

# Initialize environment and agents
env = QuoridorEnv()
state_shape = env.state_shape
num_actions = env.action_space
dqn_agent = DQNAgent(state_shape, num_actions)
opponent_agent = ShortestPathAgent(OPPONENT_MODEL)  # Assume this is already implemented
episodes = 1000

# Train the DQN agent
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    for time in range(500):
        env.render()
        print("-" * 20)

        # DQN Agent's turn
        action = dqn_agent.act(state, invalid_actions=env.invalid_actions(DQN_MODEL))
        next_state, reward, done, _ = env.step(action, DQN_MODEL)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, *state_shape])
        dqn_agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {dqn_agent.epsilon:.2}")
            break

        # Opponent's turn
        # print(env.invalid_actions(OPPONENT_MODEL))
        opponent_action = opponent_agent.act(state, invalid_actions=env.invalid_actions(OPPONENT_MODEL))
        next_state, opponent_reward, done, _ = env.step(opponent_action, OPPONENT_MODEL)
        next_state = np.reshape(next_state, [1, *state_shape])

        if done:
            reward = -10  # Penalize DQN agent if opponent wins
            dqn_agent.remember(state, action, reward, next_state, done)
            print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {dqn_agent.epsilon:.2}")
            break

        state = next_state

        # Train the DQN agent if enough experience has been gathered
        if len(dqn_agent.memory) > dqn_agent.batch_size:
            dqn_agent.replay()

# Run a single game with the trained DQN agent against the opponent
state = env.reset()
state = np.reshape(state, [1, *state_shape])
done = False
while not done:
    env.render()
    print("-" * 20)

    # DQN Agent's turn
    action = dqn_agent.act(state, invalid_actions=env.invalid_actions(DQN_MODEL))
    next_state, reward, done, _ = env.step(action, DQN_MODEL)
    state = np.reshape(next_state, [1, *state_shape])

    if done:
        break

    # Opponent's turn
    opponent_action = opponent_agent.act(state, invalid_actions=env.invalid_actions(OPPONENT_MODEL))
    next_state, reward, done, _ = env.step(opponent_action, OPPONENT_MODEL)
    state = np.reshape(next_state, [1, *state_shape])
