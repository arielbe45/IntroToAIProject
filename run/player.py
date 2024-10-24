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
        """Decides the best move using the minimax algorithm."""
        best_move = None
        best_value = -math.inf

        for move in state.get_legal_moves():
            new_state = state.get_new_state(move)
            move_value = self.minimax(new_state, self.depth, False)

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

    def minimax(self, state: GameState, depth: int, maximizing_player: bool) -> int:
        """Minimax algorithm with depth limit."""
        if depth == 0 or state.is_game_over():
            return self.heuristic_evaluation(state)

        if maximizing_player:
            max_eval = -math.inf
            for move in state.get_legal_moves():
                new_state = state.get_new_state(move)
                eval_value = self.minimax(new_state, depth - 1, False)
                max_eval = max(max_eval, eval_value)
            return max_eval
        else:
            min_eval = math.inf
            for move in state.get_legal_moves():
                new_state = state.get_new_state(move)
                eval_value = self.minimax(new_state, depth - 1, True)
                min_eval = min(min_eval, eval_value)
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