import itertools
import random
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from game import Game
from minimax import Minimax
from move import Move
from placeWall import PlaceWall
from wall import Wall
from player import Player
from cell import Cell
from board import Board  # Ensure to import the Board class
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='parameter_search_results.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')


class ParameterSearch:
    def __init__(self, early_game_ranges, mid_game_ranges, late_game_ranges, num_games=1):
        self.early_game_ranges = early_game_ranges
        self.mid_game_ranges = mid_game_ranges
        self.late_game_ranges = late_game_ranges
        self.num_games = num_games

        self.param_combinations = list(itertools.product(
            early_game_ranges['WALL_ADVANTAGE'], early_game_ranges['DELTA'], early_game_ranges['OLD_DELTA'],
            mid_game_ranges['WALL_ADVANTAGE'], mid_game_ranges['DELTA'], mid_game_ranges['OLD_DELTA'],
            late_game_ranges['WALL_ADVANTAGE'], late_game_ranges['DELTA'], late_game_ranges['OLD_DELTA']
        ))

        self.best_performance = float('-inf')
        self.best_parameters = None

    @staticmethod
    def evaluate_parameters(param_comb_white, param_comb_black, num_games):
        early_weights_white = {'WALL_ADVANTAGE': param_comb_white[0], 'DELTA': param_comb_white[1], 'OLD_DELTA': param_comb_white[2]}
        mid_weights_white = {'WALL_ADVANTAGE': param_comb_white[3], 'DELTA': param_comb_white[4], 'OLD_DELTA': param_comb_white[5]}
        late_weights_white = {'WALL_ADVANTAGE': param_comb_white[6], 'DELTA': param_comb_white[7], 'OLD_DELTA': param_comb_white[8]}
        weights_white = {'early': early_weights_white, 'mid': mid_weights_white, 'late': late_weights_white}

        early_weights_black = {'WALL_ADVANTAGE': param_comb_black[0], 'DELTA': param_comb_black[1], 'OLD_DELTA': param_comb_black[2]}
        mid_weights_black = {'WALL_ADVANTAGE': param_comb_black[3], 'DELTA': param_comb_black[4], 'OLD_DELTA': param_comb_black[5]}
        late_weights_black = {'WALL_ADVANTAGE': param_comb_black[6], 'DELTA': param_comb_black[7], 'OLD_DELTA': param_comb_black[8]}
        weights_black = {'early': early_weights_black, 'mid': mid_weights_black, 'late': late_weights_black}

        white_wins = 0
        black_wins = 0

        game_number = 0
        for _ in tqdm(range(num_games), desc="Games", leave=False):
            game = Game()
            game.minimax_white = Minimax(game.board, 5, weights=weights_white)
            game.minimax_black = Minimax(game.board, 5, weights=weights_black)

            print(f"Starting game {game_number + 1} with white weights: {weights_white} and black weights: {weights_black}")
            logging.info(f"Starting game {game_number + 1} with white weights: {weights_white} and black weights: {weights_black}")

            start_time = time.time()
            while game.is_game_over() == 0:
                current_player = game.players[game.current_player_turn]
                minimax = game.minimax_white if current_player.get_color() == 1 else game.minimax_black
                best_move = minimax.run(current_player)
                if best_move[0] == 'move':
                    game.execute_move(Move(current_player, game.board,
                                           game.board.get_cell(current_player.get_position()[0],
                                                               current_player.get_position()[1]),
                                           game.board.get_cell(best_move[1], best_move[2])))
                elif best_move[0] == 'wall':
                    wall = Wall(best_move[1], (best_move[2], best_move[3]), 3)
                    if not PlaceWall(game.board).place_wall(wall, game.white_player, game.black_player):
                        logging.warning("AI made an invalid wall placement, something went wrong.")
                    else:
                        game.execute_wall(wall)
                game.switch_player()
            end_time = time.time()
            game_duration = end_time - start_time

            game_number += 1
            winner = game.get_board_winner()
            print(f"Game {game_number} finished. Winner: {winner}. Duration: {game_duration:.2f} seconds")
            logging.info(f"Game {game_number} finished. Winner: {winner}. Duration: {game_duration:.2f} seconds")

            if winner == 1:
                white_wins += 1
            elif winner == -1:
                black_wins += 1

        performance = white_wins - black_wins
        return (performance, weights_white, weights_black)

    def run_parameter_search(self, max_workers=None):
        start_time = time.time()
        param_comb_pairs = list(itertools.product(self.param_combinations, repeat=2))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ParameterSearch.evaluate_parameters, param_comb_pair[0], param_comb_pair[1], self.num_games)
                       for param_comb_pair in param_comb_pairs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parameter Sets"):
                try:
                    performance, weights_white, weights_black = future.result()
                    logging.info(f"Weights (White): {weights_white}, Weights (Black): {weights_black}, Performance: {performance}")
                    print(f"Weights (White): {weights_white}, Weights (Black): {weights_black}, Performance: {performance}")

                    if performance > self.best_performance:
                        self.best_performance = performance
                        self.best_parameters = (weights_white, weights_black)

                    # Save intermediate results
                    with open('best_parameters.txt', 'w') as f:
                        f.write(f"Best performance: {self.best_performance}\n")
                        f.write(f"Best parameters (White): {self.best_parameters[0]}\n")
                        f.write(f"Best parameters (Black): {self.best_parameters[1]}\n")

                except Exception as e:
                    logging.error(f"Error retrieving future result: {e}")
                    print(f"Error retrieving future result: {e}")
        end_time = time.time()
        total_duration = end_time - start_time

        logging.info(f"Total parameter search duration: {total_duration:.2f} seconds")
        print(f"Total parameter search duration: {total_duration:.2f} seconds")
        logging.info(f"Best performance: {self.best_performance}")
        logging.info(f"Best parameters (White): {self.best_parameters[0]}")
        logging.info(f"Best parameters (Black): {self.best_parameters[1]}")
        print(f"Best performance: {self.best_performance}")
        print(f"Best parameters (White): {self.best_parameters[0]}")
        print(f"Best parameters (Black): {self.best_parameters[1]}")


if __name__ == "__main__":
    # Define ranges for parameters for different game phases
    early_game_ranges = {
        'WALL_ADVANTAGE': [0.5, 1, 1.5, 2, 3],
        'DELTA': [5, 8, 10, 12, 15],
        'OLD_DELTA': [1, 2, 4, 5, 6]

    }

    mid_game_ranges = {
        'WALL_ADVANTAGE': [0.5, 1, 1.5, 2, 3],
        'DELTA': [5, 8, 10, 12, 15],
        'OLD_DELTA': [1, 2, 4, 5, 6]

    }

    late_game_ranges = {
        'WALL_ADVANTAGE': [0.5, 1, 1.5, 2, 3],
        'DELTA': [5, 8, 10, 12, 15],
        'OLD_DELTA': [1, 2, 4, 5, 6]
    }

    # Create an instance of ParameterSearch
    parameter_search = ParameterSearch(early_game_ranges, mid_game_ranges, late_game_ranges, num_games=1)

    # Run the parameter search with a specified number of workers
    parameter_search.run_parameter_search(max_workers=4)  # Use 4 cores
