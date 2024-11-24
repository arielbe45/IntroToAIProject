import abc
import math
import random

from game.bfs import bfs_shortest_paths
from game.game_state import GameState, Move, BOARD_SIZE, player1_dist_to_goal, player2_dist_to_goal


def min2(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return min(x, y)


def max2(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)


class AbstractQuoridorPlayer(abc.ABC):
    @abc.abstractmethod
    def get_next_move(self, state: GameState) -> Move:
        pass


class RandomQuoridorPlayer(AbstractQuoridorPlayer):
    def get_next_move(self, state: GameState) -> Move:
        return random.choice(state.get_legal_moves())


class MinimaxPlayer(AbstractQuoridorPlayer):
    def __init__(self, heuristic_evaluation, depth: int = 3, restrict: bool = False, check_bfs: bool = False):
        self.depth = depth
        self.heuristic_evaluation = heuristic_evaluation
        self.restrict = restrict
        self.check_bfs = check_bfs

    def get_next_move(self, state: GameState) -> Move:
        """Decides the best move using the minimax algorithm with alpha-beta pruning."""
        best_move = None
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Iterate over all legal moves
        for move in state.get_legal_moves(check_bfs=self.check_bfs, restrict=self.restrict):
            new_state = state.get_new_state(move, check_legal=False)
            move_value = self.alphabeta(new_state, self.depth - 1, alpha, beta, False, main_player=state.current_player)

            if max2(move_value, best_value) != best_value:
                best_value = move_value
                best_move = move

            # Update alpha
            alpha = max2(alpha, best_value)

        if state.is_move_legal(move=best_move, check_bfs=True):
            return best_move
        return max(state.get_legal_moves(restrict=False, check_bfs=True),
                   key=lambda move: self.heuristic_evaluation(state=state.get_new_state(move=move, check_legal=False),
                                                              player=state.current_player))

    def alphabeta(self, state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool,
                  main_player: int) -> int:
        """Alpha-beta pruning algorithm with depth limit."""
        # Base case: return the heuristic value if depth is 0 or the game is over
        if depth == 0 or state.is_game_over():
            # avoid illegal states
            if not state.check_state_legal():
                return None
            return self.heuristic_evaluation(state, main_player)

        if maximizing_player:
            max_eval = -math.inf
            for move in state.get_legal_moves(restrict=self.restrict, check_bfs=self.check_bfs):
                new_state = state.get_new_state(move, check_legal=False)
                eval_value = self.alphabeta(new_state, depth - 1, alpha, beta, False,
                                            main_player=main_player)
                max_eval = max2(max_eval, eval_value)
                alpha = max2(alpha, eval_value)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = math.inf
            for move in state.get_legal_moves(restrict=self.restrict, check_bfs=self.check_bfs):
                new_state = state.get_new_state(move, check_legal=False)
                eval_value = self.alphabeta(new_state, depth - 1, alpha, beta, True,
                                            main_player=main_player)
                min_eval = min2(min_eval, eval_value)
                beta = min2(beta, eval_value)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def heuristic_evaluation(self, state: GameState, player: int) -> float:
        """Evaluate the game state for the minimax algorithm."""
        raise NotImplementedError("Heuristic evaluation must be defined")


# heuristic 1 - prioritizes states where you are closer and the opponent is further
def distance_to_end_heuristic(state: GameState, player) -> float:
    if state.p1_wins():
        return float('inf') if player == 1 else -float('inf')
    elif state.p2_wins():
        return float('inf') if player == 2 else -float('inf')
    elif state.tie():
        return 0

    player1_distance = player1_dist_to_goal(state)
    player2_distance = player2_dist_to_goal(state)

    if player == 1:
        return player2_distance - player1_distance
    return player1_distance - player2_distance


def normalized_distance_to_end_heuristic(state: GameState, player) -> float:
    if state.p1_wins():
        return 1 if player == 1 else 0
    elif state.p2_wins():
        return 1 if player == 2 else 0
    elif state.tie():
        return 0.5

    player1_distance = player1_dist_to_goal(state)
    player2_distance = player2_dist_to_goal(state)
    c = 2

    if player == 1:
        dist = -1 / (player2_distance + c) + 1 / (player1_distance + c)
    else:
        dist = 1 / (player1_distance + c) - 1 / (player2_distance + c)

    if not math.isfinite(dist):
        return 0.5
    return (dist + 1) / 2


def dqn_normalized_distance_to_end_heuristic(state: GameState, player):
    heuristic01 = normalized_distance_to_end_heuristic(state=state, player=player)
    return heuristic01 * 2 - 1


# heuristic 2 - Prioritizes states where you are closer to the winning sqaures
def winning_heuristic(state: GameState, player) -> int:
    player2_distance = player2_dist_to_goal(state)
    if player == 1:
        return player2_distance
    return -player2_distance


# heuristic 3 - Prioritizes states where walls make it harder for the opponent to progress.
def blocking_heuristic(state: GameState, player) -> int:
    player1_distance = player1_dist_to_goal(state)
    if player == 1:
        return -player1_distance
    return player1_distance


# heuristic 4 - Encourages the player to stay closer to the center of the board, improving flexibility.
def central_position_heuristic(state: GameState, player) -> int:
    center = BOARD_SIZE // 2
    player1_distance_to_center = abs(state.player1_pos[0] - center) + abs(state.player1_pos[1] - center)
    player2_distance_to_center = abs(state.player2_pos[0] - center) + abs(state.player2_pos[1] - center)

    # Favor positions closer to the center
    if player == 2:
        return player1_distance_to_center - player2_distance_to_center
    return player2_distance_to_center - player1_distance_to_center


# heuristic 5 - Counts the number of viable paths to the goal for both players.
# not good heuristic
def escape_route_heuristic(state: GameState, player) -> int:
    player1_routes = len(bfs_shortest_paths(start_node=tuple(state.player1_pos),
                                            get_free_neighbors=state.get_free_neighbor_tiles))
    player2_routes = len(bfs_shortest_paths(start_node=tuple(state.player2_pos),
                                            get_free_neighbors=state.get_free_neighbor_tiles))

    # Reward states where player 1 has more paths available
    if player == 2:
        return player2_routes - player1_routes
    return player1_routes - player2_routes


# heuristic 6 - Encourages the player to stay closer to the opponent for potential blocking opportunities.

def proximity_heuristic(state: GameState, player) -> int:
    distance_between_players = abs(state.player1_pos[0] - state.player2_pos[0]) + abs(
        state.player1_pos[1] - state.player2_pos[1])
    return -distance_between_players  # Closer proximity is better


# heuristic 7 - Penalizes players for being near the edges of the board, which limits movement options.
def edge_avoidance_heuristic(state: GameState, player) -> int:
    player1_edge_distance = min(state.player1_pos[0], BOARD_SIZE - 1 - state.player1_pos[0],
                                state.player1_pos[1], BOARD_SIZE - 1 - state.player1_pos[1])
    player2_edge_distance = min(state.player2_pos[0], BOARD_SIZE - 1 - state.player2_pos[0],
                                state.player2_pos[1], BOARD_SIZE - 1 - state.player2_pos[1])

    # Favor states where player 1 is farther from edges
    if player == 2:
        return player2_edge_distance - player1_edge_distance
    return player1_edge_distance - player2_edge_distance


# heuristic 8 - pure distance to the winning tiles
def manhattan_dist(state: GameState, player) -> int:
    dist1 = BOARD_SIZE - 1 - state.player1_pos[1]
    dist2 = state.player2_pos[1]
    if player == 1:
        return dist2 - dist1
    return dist1 - dist2


# heuristic 9 - combine these heuristics into a weighted formula for evaluation:
def combined_heuristic(state: GameState, player) -> float:
    return (
            distance_to_end_heuristic(state, player) +
            2 * winning_heuristic(state, player) +
            2 * blocking_heuristic(state, player) +
            0.5 * central_position_heuristic(state, player) +
            0.5 * escape_route_heuristic(state, player) +
            proximity_heuristic(state, player) +
            edge_avoidance_heuristic(state, player)
    )
