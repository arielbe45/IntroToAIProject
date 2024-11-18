import abc
import math
import random

from game.bfs import bfs_shortest_paths
from game.game_state import GameState, Move, BOARD_SIZE


class AbstractQuoridorPlayer(abc.ABC):
    @abc.abstractmethod
    def get_next_move(self, state: GameState) -> Move:
        pass


class RandomQuoridorPlayer(AbstractQuoridorPlayer):
    def get_next_move(self, state: GameState) -> Move:
        return random.choice(state.get_legal_moves())


class MinimaxPlayer(AbstractQuoridorPlayer):
    def __init__(self, heuristic_evaluation, depth: int = 3):
        self.depth = depth
        self.heuristic_evaluation = heuristic_evaluation

    def get_next_move(self, state: GameState) -> Move:
        """Decides the best move using the minimax algorithm with alpha-beta pruning."""
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Iterate over all legal moves
        for move in state.get_legal_moves():
            new_state = state.get_new_state(move)
            move_value = self.alphabeta(new_state, self.depth - 1, alpha, beta, False)

            if move_value > best_value:
                best_value = move_value
                best_move = move

            # Update alpha
            alpha = max(alpha, best_value)

        return best_move

    def alphabeta(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> int:
        """Alpha-beta pruning algorithm with depth limit."""
        # Base case: return the heuristic value if depth is 0 or the game is over
        if depth == 0 or state.is_game_over():
            return self.heuristic_evaluation(state)

        if maximizing_player:
            max_eval = -math.inf
            for move in state.get_legal_moves():
                new_state = state.get_new_state(move)
                eval_value = self.alphabeta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = math.inf
            for move in state.get_legal_moves():
                new_state = state.get_new_state(move)
                eval_value = self.alphabeta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def heuristic_evaluation(self, state: GameState) -> int:
        """Evaluate the game state for the minimax algorithm."""
        # This should be defined according to the rules of the game,
        # and will typically evaluate the game state favorably for the player
        # e.g., the distance to the goal or the number of walls remaining
        raise NotImplementedError("Heuristic evaluation must be defined for specific game.")


def distance_to_end_heuristic(state: GameState) -> int:
    bfs_result = bfs_shortest_paths(start_node=tuple(state.player1_pos),
                                    get_free_neighbors=state.get_free_neighbor_tiles)
    winning_tiles = [(x, BOARD_SIZE - 1) for x in range(BOARD_SIZE)]
    player1_distance = min([len(path) for tile, path in bfs_result.items() if tile in winning_tiles])

    bfs_result = bfs_shortest_paths(start_node=tuple(state.player2_pos),
                                    get_free_neighbors=state.get_free_neighbor_tiles)
    winning_tiles = [(x, 0) for x in range(BOARD_SIZE)]
    player2_distance = min([len(path) for tile, path in bfs_result.items() if tile in winning_tiles])

    if player2_distance == 0:
        return math.inf
    if player1_distance == 0:
        return -math.inf
    return player1_distance - player2_distance