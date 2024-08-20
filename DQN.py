import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import random
from collections import deque

import QuoridorEnv

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

    def act(self, state, invalid_actions=[]):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state)
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

# Initialize environment and agent
env = QuoridorEnv()
state_shape = env.state_shape
num_actions = env.action_space
agent = DQNAgent(state_shape, num_actions)
episodes = 1000

# Train the agent
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, *state_shape])
    for time in range(500):
        env.render()
        action = agent.act(state,invalid_actions=env.invalid_actions())
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, *state_shape])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
            break
        if len(agent.memory) > agent.batch_size:
            agent.replay()

# Run a single game with the trained agent
state = env.reset()
state = np.reshape(state, [1, *state_shape])
done = False
while not done:
    env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = np.reshape(next_state, [1, *state_shape])
