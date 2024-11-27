import json
import os
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from game.move import ALL_MOVES
from run.player import *


def print_gradients(model):
    return
    print("Parameters:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(
                f"\t{name}:  \tmean={param.data.mean().item():.6f},\tmax={param.data.max().item():.6f},\tmin={param.data.min().item():.6f},\tnorm={param.data.norm(2).item():.6f}")
        else:
            print(f"\t{name}: No gradient (likely frozen or unused)")

    print("Gradients for each parameter:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(
                f"\t{name}:  \tmean={param.grad.mean().item():.6f},\tmax={param.grad.max().item():.6f},\tmin={param.grad.min().item():.6f},\tnorm={param.grad.norm(2).item():.6f}")
        else:
            print(f"\t{name}: No gradient (likely frozen or unused)")


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
    def __init__(self, restrict, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.2, gamma: float = 0.996, learning_rate: float = 0.001):
        self.restrict = restrict
        self.state_size = len(GameState().to_vector())
        print(f"State vector shape: {GameState().to_vector().shape}")
        self.action_size = len(ALL_MOVES)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory = []  # Replay buffer
        self.batch_size = 64
        print(self.state_size, self.action_size, "!")
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        if os.path.exists("data/dqn_weights.pth"):
            self.load_model("data/dqn_weights.pth")

    def save_model(self, filename="q_network.pth"):
        """
        Saves the Q-network to a file.
        """
        torch.save(self.q_network.state_dict(), filename)
        # print(f"Model saved to {filename}")

    def load_model(self, filename="q_network.pth"):
        """
        Loads the Q-network from a file.
        """
        self.q_network.load_state_dict(torch.load(filename))
        self.q_network.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filename}")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def replay(self):
        # Ensure enough experiences are in memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors for batch processing
        states = torch.stack(states)  # Stack states into a batch tensor
        actions = torch.tensor(actions, dtype=torch.int64)  # Action indices
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)  # 1.0 for terminal states, 0.0 otherwise

        # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # Batch of Q-values for next states
            next_q_values = -next_q_values
            max_next_q_values = next_q_values.max(dim=1)[0]  # Max Q-value for each next state
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute current Q-values using the main Q-network
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the loss between the target and current Q-values
        loss = self.loss_fn(current_q_values, targets)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        print_gradients(self.q_network)
        self.optimizer.step()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def get_next_move(self, state: GameState) -> Move:
        legal_moves = state.get_legal_moves(restrict=self.restrict, check_bfs=True)
        if not legal_moves:
            return random.choice(state.get_legal_moves(restrict=False, check_bfs=True))

        if np.random.rand() < self.epsilon:
            # Exploration: choose a random legal move
            return random.choice(legal_moves)

        # Exploitation: choose the best move based on Q-values
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

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
        return ALL_MOVES.index(move)


def train_agent_single_episode(dqn_player: DeepQLearningPlayer, reward_heuristic, opponent: AbstractQuoridorPlayer):
    state = GameState()
    while not state.is_game_over():
        if state.p1_turn:
            move = dqn_player.get_next_move(state)
            old_state = dqn_player._state_to_tensor(state)
            state.apply_move(move=move)
            new_state = dqn_player._state_to_tensor(state)
            reward = reward_heuristic(state, player=1)

            # Remember this experience
            dqn_player.remember(
                state=old_state,
                action=dqn_player._move_to_index(move),
                reward=reward,
                next_state=new_state,
                done=state.is_game_over()
            )
        else:
            # move = opponent.get_next_move(state)
            # state.apply_move(move=move)

            move = opponent.get_next_move(state)
            old_state = dqn_player._state_to_tensor(state)
            state.apply_move(move=move)
            new_state = dqn_player._state_to_tensor(state)
            reward = reward_heuristic(state, player=2)

            # Remember this experience
            dqn_player.remember(
                state=old_state,
                action=dqn_player._move_to_index(move),
                reward=reward,
                next_state=new_state,
                done=state.is_game_over()
            )


def train_agent(dqn_player: DeepQLearningPlayer, opponent_model,
                reward_heuristic: typing.Callable[[GameState, int], float],
                total_episodes=10_000_000, update_target_every=100, log_interval=60, verbose=False):  # log_interval in seconds
    print("training")
    opponent = opponent_model

    # Tracking variables
    start_time = time.time()
    last_log_time = start_time

    # Load existing loss data if it exists
    if os.path.exists("data/loss_data.json"):
        with open("data/loss_data.json", "r") as f:
            data = json.load(f)
            losses = data.get("losses", [])
            batch_indices = data.get("batches", [])
    else:
        losses = []
        batch_indices = []

    initial_batch = batch_indices[-1] + 1 if batch_indices else 0
    current_batch_index = initial_batch

    for episode in tqdm(range(total_episodes)):
        train_agent_single_episode(dqn_player=dqn_player, reward_heuristic=reward_heuristic, opponent=opponent)

        # Replay experiences to train the network
        loss = dqn_player.replay()

        # Update target network periodically
        if episode % update_target_every == 0:
            dqn_player.update_target_network()

        # print(f"Batch {episode + 1}/{total_episodes} complete. Epsilon: {dqn_player.epsilon:.2f}")
        dqn_player.save_model("data/dqn_weights.pth")
        # print("Model saved.")

        if loss is None:
            continue

        losses.append(loss.item())
        batch_indices.append(current_batch_index)
        current_batch_index += 1

        # Visualization and logging every log_interval
        current_time = time.time()
        if current_time - last_log_time >= log_interval:
            last_log_time = current_time

            # Save loss data to file
            with open("data/loss_data.json", "w") as f:
                json.dump({"losses": losses, "batches": batch_indices}, f)

            if verbose:
                # Visualize Q-values
                visualize_minimax_game_q_values(dqn_player=dqn_player, reward_heuristic=reward_heuristic)
                # Plot loss vs episode
                plot_loss_graph()
    print("Training complete.")


def plot_loss_graph():
    if os.path.exists("data/loss_data.json"):
        with open("data/loss_data.json", "r") as f:
            data = json.load(f)
            losses = data.get("losses", [])
            batch_indices = data.get("batches", [])

    plt.figure(figsize=(8, 6))
    plt.plot(batch_indices, losses, label="Loss")
    plt.xlabel("Batch Iterations")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss vs Batch Iterations")
    plt.legend()
    plt.savefig("images/loss_vs_batch.png")
    plt.show()


def visualize_minimax_game_q_values(dqn_player: DeepQLearningPlayer, reward_heuristic):
    states = []
    game_state = GameState()
    while not game_state.is_game_over():
        states.append(game_state.copy())
        game_state.apply_move(
            move=MinimaxPlayer(heuristic_evaluation=reward_heuristic, depth=3).get_next_move(state=game_state))
    chosen_states = random.choices(states, k=10)

    visualize_q_values(dqn_player=dqn_player, sample_states=chosen_states,
                       reward_heuristic=dqn_normalized_distance_to_end_heuristic)


def visualize_q_values(dqn_player, reward_heuristic, sample_states, gamma=0.996):
    """
    Function to visualize Q-values for sample states as a bar plot,
    with heuristic values of the resulting states after each action.
    Includes the target Q-values (desired Q-values).

    Args:
        dqn_player: The trained DQN player.
        reward_heuristic: A function that computes the heuristic for a given state and player.
        sample_states: A list of sample game states to visualize.
        gamma: Discount factor for future rewards, default is 0.99.
    """
    print("Visualizing Q-values")

    for i, state in enumerate(sample_states):
        tensor_state = dqn_player._state_to_tensor(state)
        with torch.no_grad():
            q_values = dqn_player.q_network(tensor_state.unsqueeze(0)).squeeze().numpy()  # Q-values for current state

        # Calculate heuristic values for the resulting states after each action
        heuristic_values = []
        target_q_values = []

        actions = ALL_MOVES
        for move in actions:
            if not state.is_move_legal(move=move, check_bfs=False):
                heuristic_values.append(0)
                target_q_values.append(0)
                continue

            next_state = state.copy()
            next_state.apply_move(move, check_legal=False)
            heuristic_value = reward_heuristic(next_state, player=state.current_player)  # Compute heuristic for resulting state
            heuristic_values.append(heuristic_value)

            # Compute the target Q-value (Double DQN target)
            next_state_tensor = dqn_player._state_to_tensor(next_state)
            with torch.no_grad():
                next_q_values = dqn_player.q_network(next_state_tensor.unsqueeze(0)).squeeze()
                next_q_values = -next_q_values
                # Double DQN: Use the current Q-network to select the next action, and use the target network for the Q-value
                next_action = torch.argmax(next_q_values).item()
                target_q_value = heuristic_value + gamma * next_q_values[next_action]
                target_q_values.append(target_q_value.item())

        # Ensure actions and Q-values align in dimension
        if len(q_values) != len(actions):
            print(f"Warning: Mismatch in actions ({len(actions)}) and Q-values ({len(q_values)}).")
            continue

        # Create a bar plot of Q-values, target Q-values, and heuristic values
        x_labels = [f"Action {j}" for j in range(len(actions))]
        x = np.arange(len(actions))  # Indices for the bars
        bar_width = 0.3

        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width, q_values, bar_width, label="Q-Values", color='blue', alpha=0.7)
        plt.bar(x, target_q_values, bar_width, label="Target Q-Values", color='green', alpha=0.7)
        plt.bar(x + bar_width, heuristic_values, bar_width, label="Heuristic", color='orange', alpha=0.7)
        plt.title(f"Q-Values, Target Q-Values and Heuristics for Sample State {i + 1}")
        plt.xlabel("Actions")
        plt.ylabel("Values")
        plt.xticks(x, x_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save and show the plot
        plot_filename = f"q_values_target_q_values_and_heuristics_state_{i + 1}.png"
        plt.savefig(plot_filename)
        plt.show()
        print(f"Q-values, target Q-values, and heuristic visualization saved as '{plot_filename}'.")
