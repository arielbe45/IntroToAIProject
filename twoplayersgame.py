from board import Board
from minimax import Minimax
from player import Player
from move import Move
from placeWall import PlaceWall
from wall import Wall
from cell import Cell

Y_White = 8
Y_Black = 8
x_White = 0
x_Black = 16

class Game:
    def __init__(self):
        """
        Initializes the Game class.
        """
        self.board = Board()
        self.current_player_turn = 1  # 1 for white (human), -1 for black (AI)
        self.white_player = Player((x_White, Y_White), 1)
        self.black_player = Player((x_Black, Y_Black), -1)
        self.board.set_cell(x_White, Y_White, Cell(x_White, Y_White, self.white_player))
        self.board.set_cell(x_Black, Y_Black, Cell(x_Black, Y_Black, self.black_player))
        self.board.add_player(self.white_player)
        self.board.add_player(self.black_player)
        self.players = {1: self.white_player, -1: self.black_player}
        self.minimax = Minimax(self.board, 3)  # Initialize Minimax with depth 3


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
        #self.board.get_cell(move.to_square.get_x(),move.to_square.get_y()).set_empty(False)
        #self.board.get_cell(move.to_square.get_x(),move.to_square.get_y()).set_empty(True)
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
                elif cell.is_player():
                    print("P", end=" ")
                else:
                    print("W", end=" ")
            print()

    def play(self):
        """
        The main game loop. Alternates turns between players until the game is over.
        """
        while self.is_game_over() == 0:
            self.display_board()
            current_player = self.players[self.current_player_turn]
            print(f"Player {self.current_player_turn}'s turn")

            if self.current_player_turn == -1:  # Human player's turn
                move_type = input("Enter 'm' to move or 'w' to place a wall: ")
                while move_type != 'm' and move_type != 'w':
                    print("Invalid input, try again.")
                    move_type = input("Enter 'm' to move or 'w' to place a wall: ")
                if move_type == 'm':
                    flag = False
                    while not flag:
                        try:
                            x = int(input("Enter new x position: "))
                            y = int(input("Enter new y position: "))
                        except ValueError:
                            print("Invalid input, try again.")
                            continue
                        if x < 0 or x > 16 or y < 0 or y > 16:
                            print("Invalid move, try again.")
                        else:
                            flag = True
                    if not Move(current_player, self.board,
                                self.board.get_cell(current_player.get_position()[0],
                                                    current_player.get_position()[1]),
                                self.board.get_cell(x, y)).is_valid_move():
                        print("Invalid move, try again.")
                    else:
                        self.execute_move(Move(current_player, self.board,
                                               self.board.get_cell(current_player.get_position()[0],
                                                                   current_player.get_position()[1]),
                                               self.board.get_cell(x, y)))
                        print("Move successful!")
                        self.switch_player()
                elif move_type == 'w':
                    if current_player.get_num_of_walls() == 0:
                        print("Player has no walls left, try again.")
                        continue
                    flag = False
                    while not flag:
                        try:
                            orientation = input("Enter wall orientation - or | : ")
                            start_x = int(input("Enter start x position: "))
                            start_y = int(input("Enter start y position: "))
                        except ValueError:
                            print("Invalid input, try again.")
                            continue
                        if start_x < 0 or start_x > 16 or start_y < 0 or start_y > 16 or orientation not in ['-', '|']:
                            print("Invalid wall placement, try again.")
                        else:
                            flag = True
                            if orientation == '-':
                                orientation = "horizontal"
                            else:
                                orientation = "vertical"

                    wall = Wall(orientation, (start_x, start_y), 3)

                    if not PlaceWall(self.board).place_wall(wall, self.white_player, self.black_player):
                        print("Invalid wall placement, try again.")
                    else:
                        print("Wall placed successfully!")
                        self.execute_wall(wall)
                        self.switch_player()

            else:  # Human player's turn
                move_type = input("Enter 'm' to move or 'w' to place a wall: ")
                while move_type != 'm' and move_type != 'w':
                    print("Invalid input, try again.")
                    move_type = input("Enter 'm' to move or 'w' to place a wall: ")
                if move_type == 'm':
                    flag = False
                    while not flag:
                        try:
                            x = int(input("Enter new x position: "))
                            y = int(input("Enter new y position: "))
                        except ValueError:
                            print("Invalid input, try again.")
                            continue
                        if x < 0 or x > 16 or y < 0 or y > 16:
                            print("Invalid move, try again.")
                        else:
                            flag = True
                    if not Move(current_player, self.board,
                                self.board.get_cell(current_player.get_position()[0],
                                                    current_player.get_position()[1]),
                                self.board.get_cell(x, y)).is_valid_move():
                        print("Invalid move, try again.")
                    else:
                        self.execute_move(Move(current_player, self.board,
                                               self.board.get_cell(current_player.get_position()[0],
                                                                   current_player.get_position()[1]),
                                               self.board.get_cell(x, y)))
                        print("Move successful!")
                        self.switch_player()
                elif move_type == 'w':
                    if current_player.get_num_of_walls() == 0:
                        print("Player has no walls left, try again.")
                        continue
                    flag = False
                    while not flag:
                        try:
                            orientation = input("Enter wall orientation - or | : ")
                            start_x = int(input("Enter start x position: "))
                            start_y = int(input("Enter start y position: "))
                        except ValueError:
                            print("Invalid input, try again.")
                            continue
                        if start_x < 0 or start_x > 16 or start_y < 0 or start_y > 16 or orientation not in ['-', '|']:
                            print("Invalid wall placement, try again.")
                        else:
                            flag = True
                            if orientation == '-':
                                orientation = "horizontal"
                            else:
                                orientation = "vertical"

                    wall = Wall(orientation, (start_x, start_y), 3)

                    if not PlaceWall(self.board).place_wall(wall, self.white_player, self.black_player):
                        print("Invalid wall placement, try again.")
                    else:
                        print("Wall placed successfully!")
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
