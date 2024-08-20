from board import Board
from minimax import Minimax
from player import Player
from move import Move
from placeWall import PlaceWall
from wall import Wall
from cell import Cell

Y_players = 8
x_White = 0
x_Black = 16

class Game:
    def __init__(self):
        """
        Initializes the Game class.
        """
        self.board = Board()
        self.current_player_turn = 1  # 1 for white (human), -1 for black (AI)
        self.white_player = Player((x_White, Y_players), 1)
        self.black_player = Player((x_Black, Y_players), -1)
        self.board.set_cell(x_White, Y_players, Cell(x_White, Y_players, self.white_player))
        self.board.set_cell(x_Black, Y_players, Cell(x_Black, Y_players, self.black_player))
        self.board.add_player(self.white_player)
        self.board.add_player(self.black_player)
        self.players = {1: self.white_player, -1: self.black_player}
        self.minimax = Minimax(self.board, 5)  # Initialize Minimax with depth 3


    def switch_player(self):
        """
        Switches the turn to the other player.
        """
        self.current_player_turn *= -1

    def is_game_over(self):
        """
        Checks if the game is over.

        :return: The winner of the game, or 0 if the game is not over.
        """
        if self.white_player.get_x() == 16:
            self.board.winner = 1
            return 1
        if self.black_player.get_x() == 0:
            self.board.winner = -1
            return -1
        return 0

    def execute_move(self, move):
        """
        Executes a move on the board.
        :param move: The move to execute.
        """
        self.board.set_cell(move.from_square.get_x(), move.from_square.get_y(),
                            Cell(move.from_square.get_x(), move.from_square.get_y(),player=None, empty=True))
        self.board.set_cell(move.to_square.get_x(), move.to_square.get_y(),
                            Cell(move.to_square.get_x(), move.to_square.get_y(), player=move.player, empty=False))
        move.player.set_position((move.to_square.get_x(), move.to_square.get_y()))

    def execute_wall(self, wall):
        """
        Executes a wall placement on the board.

        :param wall: The wall to place.
        """
        self.board.execute_wall(wall)
        current_player = self.players[self.current_player_turn]
        current_player.use_wall()

    def display_board(self):
        """
        Displays the current state of the board.
        """
        for i in range(17):
            for j in range(17):
                cell = self.board.get_cell(i, j)
                if cell.is_empty():
                    print(".", end=" ")
                elif cell.is_player() and cell.get_player() == 1:
                    print("1", end=" ")
                elif cell.is_player() and cell.get_player() == -1:
                    print("2", end=" ")
                else:
                    print("W", end=" ")
            print()

    def play(self):
        """
        The main game loop. Alternates turns between players until the game is over.
        """

        # Initialize Minimax for both players with different weights
        white_weights = {
            'early': {'WALL_ADVANTAGE': 2, 'DELTA': 5, 'OLD_DELTA': 2},
            'mid': {'WALL_ADVANTAGE': 1.5, 'DELTA': 7, 'OLD_DELTA': 4},
            'late': {'WALL_ADVANTAGE': 0.5, 'DELTA': 10, 'OLD_DELTA': 6}
        }
        black_weights = {
            'early': {'WALL_ADVANTAGE': 0.5, 'DELTA': 5, 'OLD_DELTA': 1},
            'mid': {'WALL_ADVANTAGE': 0.5, 'DELTA': 5, 'OLD_DELTA': 1},
            'late': {'WALL_ADVANTAGE': 0.5, 'DELTA': 5, 'OLD_DELTA': 5}
        }

        counter = 0

        while self.is_game_over() == 0:
            counter = counter + 1
            self.display_board()
            current_player = self.players[self.current_player_turn]
            print(f"Player {self.current_player_turn}'s turn")
            if current_player.get_color() == 1:
                self.minimax = Minimax(self.board, 3, white_weights)  # Initialize Minimax with depth 3
            else:
                self.minimax = Minimax(self.board, 3, black_weights)

            best_move = self.minimax.run(current_player)
            if counter == 4:
                wall = Wall("horizontal", (5, 8), 3)
                self.execute_wall(wall)
            elif best_move[0] == 'move':
                self.execute_move(Move(current_player, self.board,
                                       self.board.get_cell(current_player.get_position()[0],
                                                           current_player.get_position()[1]),
                                       self.board.get_cell(best_move[1], best_move[2])))
                print(f"AI moved to ({best_move[1]}, {best_move[2]})")
            elif best_move[0] == 'wall':
                wall = Wall(best_move[1], (best_move[2], best_move[3]), 3)
                if not PlaceWall(self.board).place_wall(wall, self.white_player, self.black_player):
                    print("AI made an invalid wall placement, something went wrong.")
                else:
                    print(f"AI placed a wall at ({best_move[2]}, {best_move[3]})")
                    self.execute_wall(wall)
            self.switch_player()

        print(f"Game over! Player {self.board.winner} wins!")
        self.display_board()

    def set_white_playe(self, position):
        self.white_player.set_position(position)
        self.board.set_cell(position[0], position[1], Cell(position[0], position[1], self.white_player))

    def get_board_winner(self):
        return self.board.winner


if __name__ == "__main__":
    game = Game()
    game.play()
