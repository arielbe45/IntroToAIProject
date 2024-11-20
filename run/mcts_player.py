import math
import random
from typing import Optional

from run.player import AbstractQuoridorPlayer
from game.game_state import GameState
from game.move import Move


class Node:
    def __init__(self, state: GameState, parent: Optional['Node'] = None, move: Optional[Move] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_param: float = 1.414) -> 'Node':
        # Uses the UCB1 formula to choose the best child based on exploration-exploitation
        return max(self.children, key=lambda child: child.wins / child.visits + exploration_param * math.sqrt(
            math.log(self.visits) / child.visits))

    def expand(self):
        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            next_state = self.state.get_new_state(move)
            child_node = Node(state=next_state, parent=self, move=move)
            self.children.append(child_node)

    def update(self, result: float):
        self.visits += 1
        self.wins += result


class MCTSPlayer(AbstractQuoridorPlayer):
    def __init__(self, heurisric, player, depth, restrict_walls, num_simulations: int = 1000,
                 exploration_param: float = 1.414):
        self.heuristic = heurisric
        self.player = player
        self.depth = depth
        self.restrict_walls = restrict_walls
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param

    def get_next_move(self, state: GameState) -> Move:
        root = Node(state=state)

        for _ in range(self.num_simulations):
            node = self._select(root)
            result = self._simulate(node.state)
            self._backpropagate(node, result)

        # Select the move with the most visits as the final choice
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def _select(self, node: Node) -> Node:
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                node = node.best_child(self.exploration_param)
        return node

    def _expand(self, node: Node) -> Node:
        # Expand a new child from unexplored moves
        untried_moves = [move for move in node.state.get_legal_moves(self.restrict_walls) if
                         move not in [child.move for child in node.children]]
        move = random.choice(untried_moves)
        next_state = node.state.get_new_state(move)
        child_node = Node(state=next_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    # def _expand(self, node: Node) -> Node:
    #     # Get legal moves and rank them based on heuristic
    #     untried_moves = [move for move in node.state.get_legal_moves(self.restrict_walls) if
    #                      move not in [child.move for child in node.children]]
    #     if not untried_moves:
    #         return None
    #
    #     # Use heuristic to sort moves and pick the best one to expand
    #     sorted_moves = sorted(untried_moves, key=lambda move: self.heuristic(node.state.get_new_state(move)),
    #                           reverse=True)
    #     move = sorted_moves[0]  # Choose the move with the highest heuristic score
    #     next_state = node.state.get_new_state(move)
    #     child_node = Node(state=next_state, parent=node, move=move)
    #     node.children.append(child_node)
    #     return child_node

    def _simulate(self, state: GameState) -> float:
        current_state = state
        depth = 0
        while not current_state.is_game_over():
            # If depth exceeds a certain limit, evaluate using heuristic
            if depth >= self.depth:
                return self.heuristic(current_state, player=self.player)

            # Choose a move based on heuristics rather than randomly
            legal_moves = current_state.get_legal_moves(self.restrict_walls)
            move = max(legal_moves,
                       key=lambda move: self.heuristic(current_state.get_new_state(move), player=self.player), )
            current_state = current_state.get_new_state(move)
            depth += 1

        # At game end, return binary win/loss
        return 1.0 if current_state.is_winner() else 0.0

    def _backpropagate(self, node: Node, result: float):
        while node is not None:
            node.update(result)
            result = 1 - result  # Alternate win/loss for the opponent's turn
            node = node.parent
