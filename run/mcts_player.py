import logging
import math
import random
from typing import Optional

from game.game_state import GameState
from game.move import Move, ALL_MOVES
from run.player import AbstractQuoridorPlayer


class Node:
    def __init__(self, state: GameState, parent: Optional['Node'] = None, move: Optional[Move] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_legal_moves = self.state.get_legal_moves(restrict=True, check_bfs=False)

    def is_fully_expanded(self) -> bool:
        return not self.untried_legal_moves

    def best_child(self, exploration_param: float = 1.414) -> 'Node':
        # Uses the UCB1 formula to choose the best child based on exploration-exploitation
        return max(self.children, key=lambda child: child.wins / child.visits + exploration_param * math.sqrt(
            math.log(self.visits) / child.visits))

    def update(self, result: float):
        self.visits += 1
        self.wins += result


class MCTSPlayer(AbstractQuoridorPlayer):
    def __init__(self, heurisric, depth, restrict_walls, num_simulations: int = 10000,
                 exploration_param: float = 1.414, rollout_exploration_param: float = 1,
                 selection_exploration_param: float = 0.1):
        self.heuristic = heurisric
        self.depth = depth
        self.restrict_walls = restrict_walls
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param
        self.rollout_exploration_param = rollout_exploration_param
        self.selection_exploration_param = selection_exploration_param

    def get_next_move(self, state: GameState) -> Move:
        root = Node(state=state)
        self.leaves = 0
        self.heuristic.count = 0

        for _ in range(self.num_simulations):
            node = self._select(root)
            result = self._simulate(node.state)
            self._backpropagate(node, result)

        logging.debug('visits\twins\tmove')
        for i, child in enumerate(sorted(root.children, key=lambda child: child.visits, reverse=True)):
            logging.debug(f'{child.visits}\t{round(child.wins)}\t{child.move}')
            if i < 3:
                for grandchild in sorted(child.children, key=lambda grandchild: grandchild.visits, reverse=True):
                    logging.debug(f'\t{grandchild.visits}\t{round(grandchild.wins)}\t{grandchild.move}')
        logging.debug(f'leaves: {self.leaves}')
        logging.debug(f'heuristic: {self.heuristic.count}')

        # Select the move with the most visits as the final choice
        best_child = max(root.children, key=lambda child: child.visits)
        if state.is_move_legal(move=best_child.move, check_bfs=True):
            return best_child.move

        # return a random move if all restricted moves are illegal
        return random.choice(state.get_legal_moves(check_bfs=True, restrict=False))

    def _select(self, node: Node) -> Node:
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                node = self._expand(node)
                if node.parent.parent is None:
                    heuristic = self.heuristic(node.state.get_new_state(node.move, check_legal=False),
                                               player=node.state.current_player)
                    logging.debug(f'move selected: {node.state.current_player}\t{float(heuristic):.04}\t{node.move}')
                return node
            else:
                node = node.best_child(self.exploration_param)
        return node

    def _expand(self, node: Node) -> Node:
        # Expand a new child from unexplored moves
        if not self.restrict_walls and random.random() < self.selection_exploration_param:
            # allow unrestricted moves
            child_moves = [child.move for child in node.children]
            possible_moves = [move for move in ALL_MOVES if move not in child_moves]
            random.shuffle(possible_moves)
            for move in possible_moves:
                if node.state.is_move_legal(move=move, check_bfs=False):
                    break
            else:
                raise Exception
        else:
            # explore new moves
            # return a random restricted legal move
            move = random.choice(node.untried_legal_moves)

        # remove chosen move from untried moves list
        if move in node.untried_legal_moves:
            node.untried_legal_moves.remove(move)

        next_state = node.state.get_new_state(move, check_legal=False)
        child_node = Node(state=next_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def _simulate(self, state: GameState) -> float:
        current_state = state
        depth = 0
        while not current_state.is_game_over():
            # If depth exceeds a certain limit, evaluate using heuristic
            if depth >= self.depth:
                self.leaves += 1
                other_player = 1 if state.current_player == 2 else 2
                return self.heuristic(current_state, player=other_player)

            legal_moves = state.get_legal_moves(restrict=self.restrict_walls, check_bfs=False)

            if random.random() < self.rollout_exploration_param:
                # choose a random move
                move = random.choice(legal_moves)
            else:
                # Choose a move based on heuristics rather than randomly, minimize next player heuristic
                move = max(legal_moves, key=lambda move: self.heuristic(state.get_new_state(move, check_legal=False),
                                                                        player=state.current_player))

            current_state = current_state.get_new_state(move, check_legal=False)
            depth += 1

        # At game end, return binary win/loss
        return 1.0 if current_state.is_winner() else 0.0

    def _backpropagate(self, node: Node, result: float):
        while node is not None:
            node.update(result)
            result = 1 - result  # Alternate win/loss for the opponent's turn
            node = node.parent
