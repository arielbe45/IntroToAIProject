import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from run.player import AbstractQuoridorPlayer
from game.move import Move
from game.game_state import GameState


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(QNetwork, self).__init__()
        # Define a simple feedforward neural network
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class DeepQLearningPlayer(AbstractQuoridorPlayer):
    def __init__(self, state_size: int, action_size: int, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1, gamma: float = 0.99, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = []  # Replay buffer
        self.batch_size = 64
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_network(next_state))

            current_q = self.q_network(state)[action]
            loss = self.loss_fn(current_q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_next_move(self, state: GameState) -> Move:
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random legal move
            legal_moves = state.get_legal_moves()
            return random.choice(legal_moves)

        # Exploitation: choose the best move based on Q-values
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        legal_moves = state.get_legal_moves()

        # Select the legal move with the highest Q-value
        best_move = max(legal_moves, key=lambda move: q_values[self._move_to_index(move)].item())
        return best_move

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.replay()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _state_to_tensor(self, state: GameState) -> torch.Tensor:
        # Convert game state to a PyTorch tensor (for simplicity, assuming it's a 1D vector)
        return torch.tensor(state.to_vector(), dtype=torch.float32)

    def _move_to_index(self, move: Move) -> int:
        # Convert a move into an index for the Q-network output (this assumes a fixed set of possible moves)
        return move.to_index()
