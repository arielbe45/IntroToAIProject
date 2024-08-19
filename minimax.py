from copy import deepcopy

from board import Board
from cell import Cell
from player import Player
from wall import Wall

HITOCHI_RATIO = 0.15


class Minimax:
    def __init__(self, board, depth, weights=None):
        self.board = board
        self.depth = depth
        self.weights = weights
        self.evaluation_cache = {}  # Add this line

    def score_board(self, board, player, old_board=None, move=None):
        opponent = board.get_opponent(player)
        # print player and opponent colors
        player_shortest_path = board.find_shortest_path_to_goal(player)
        opponent_shortest_path = board.find_shortest_path_to_goal(opponent)

        if player_shortest_path == 0:
            return 100000

        delta = player_shortest_path - opponent_shortest_path

        num_of_left_walls = player.get_num_of_walls()
        num_of_opponent_walls = opponent.get_num_of_walls()
        wall_advantage = num_of_left_walls - num_of_opponent_walls
        if old_board is not None:
            # find the old_player_loc
            if player.get_color() == 1:
                old_player = old_board.get_player(1)
                old_opponent = old_board.get_player(-1)
            else:
                old_player = old_board.get_player(-1)
                old_opponent = old_board.get_player(1)

            old_player_shortest_path = old_board.find_shortest_path_to_goal(old_player)
            old_opponent_shortest_path = old_board.find_shortest_path_to_goal(old_opponent)
            old_delta = (old_player_shortest_path - old_opponent_shortest_path)
            num_of_left_walls = old_player.get_num_of_walls()
            num_of_opponent_walls = old_opponent.get_num_of_walls()

            if old_opponent_shortest_path == 1 and opponent_shortest_path > 1:
                return 100000
            if opponent_shortest_path <= 1:
                return -100000

        total_walls = num_of_left_walls + num_of_opponent_walls
        early_game = total_walls > 15
        mid_game = 10 <= total_walls <= 15
        late_game = total_walls < 10

        if self.weights is None:
            if early_game:
                WALL_ADVANTAGE = 3
                DELTA = 6
                OLD_DELTA = 1
            elif mid_game:
                WALL_ADVANTAGE = 2
                DELTA = 8
                OLD_DELTA = 5
            elif late_game:
                WALL_ADVANTAGE = 1
                DELTA = 12
                OLD_DELTA = 5
        else:
            if early_game:
                WALL_ADVANTAGE = self.weights['early']['WALL_ADVANTAGE']
                DELTA = self.weights['early']['DELTA']
                OLD_DELTA = self.weights['early']['OLD_DELTA']
            elif mid_game:
                WALL_ADVANTAGE = self.weights['mid']['WALL_ADVANTAGE']
                DELTA = self.weights['mid']['DELTA']
                OLD_DELTA = self.weights['mid']['OLD_DELTA']
            elif late_game:
                WALL_ADVANTAGE = self.weights['late']['WALL_ADVANTAGE']
                DELTA = self.weights['late']['DELTA']
                OLD_DELTA = self.weights['late']['OLD_DELTA']

        # Factor in old board for deltas
        if old_board is not None:
            old_player_shortest_path = old_board.find_shortest_path_to_goal(player)
            old_opponent_shortest_path = old_board.find_shortest_path_to_goal(opponent)
            old_delta = old_player_shortest_path - old_opponent_shortest_path

            if old_opponent_shortest_path == 1 and opponent_shortest_path > 1:
                return 100000
            if opponent_shortest_path <= 1:
                return -100000

            return WALL_ADVANTAGE * wall_advantage - OLD_DELTA * old_delta - DELTA * delta

        return WALL_ADVANTAGE * wall_advantage - DELTA * delta

    def best_board(self, board, player):
        possible_moves = board.get_all_possible_moves(player)
        # create a sorted list of possible moves sorted by evaluation
        grade = []
        for move in possible_moves:
            new_player = player.copy()
            new_board = board.copy()

            if move[0] == 'move':
                new_board.execute_move(new_player, move[1], move[2])

            elif move[0] == 'wall':
                if move[2] == 1 and move[3] == 0 and move[1] == 'horizontal':
                    print('here')
                wall = Wall(move[1], (move[2], move[3]), 3)
                new_board.execute_wall(wall)
                new_player.use_wall()
            score = self.score_board(new_board, new_player, old_board=board, move=move[0])
            if score == 100000:
                return [(score, move)]

            grade.append((score, move))
        grade.sort(key=lambda x: x[0], reverse=True)
        if len(grade) < 1 + 1 / HITOCHI_RATIO:
            return grade
        top_moves = grade[:int(len(grade) * HITOCHI_RATIO)]
        return top_moves

    def minimax(self, board, depth, alpha, beta, maximizing_player, player):
        if depth == 0 or board.is_game_over():
            return self.score_board(board, player), None
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            sorted_moves = self.best_board(board, player)
            for eval_score, move in sorted_moves:
                new_board = board.copy()
                new_player = player.copy()
                if move[0] == 'move':
                    new_board.execute_move(new_player, move[1], move[2])
                elif move[0] == 'wall':
                    wall = Wall(move[1], (move[2], move[3]), 3)
                    new_board.execute_wall(wall)
                    new_player.use_wall()
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False, new_player)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            sorted_moves = self.best_board(board, player)
            for eval_score, move in sorted_moves:
                new_board = board.copy()
                new_player = player.copy()
                if move[0] == 'move':
                    new_board.execute_move(new_player, move[1], move[2])
                elif move[0] == 'wall':
                    wall = Wall(move[1], (move[2], move[3]), 3)
                    new_board.execute_wall(wall)
                    new_player.use_wall()
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True, new_player)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def run(self, player):
        for depth in range(1, self.depth + 1):
            eval, best_move = self.minimax(self.board, depth, float('-inf'), float('inf'), True, player)
            if eval == 10000:  # If the evaluation function returns a winning score
                return best_move
        return best_move
