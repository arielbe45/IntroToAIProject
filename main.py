from board import Board
from player import Player
from wall import Wall
from minimax import Minimax
from move import Move
from placeWall import PlaceWall

def apply_move(board, move, player):
    if move[0] == 'move':
        print(f"Applying move: Player {player.get_color()} to ({move[1]}, {move[2]})")
        board.execute_move(player, move[1], move[2])
    elif move[0] == 'wall':
        print(f"Placing wall: {move[1]} at ({move[2]}, {move[3]})")
        wall = Wall(move[1], (move[2], move[3]), 3)
        place_wall = PlaceWall(board)
        place_wall.place_wall(wall, player, board.get_opponent(player))
        player.use_wall()

def main():
    # Initialize the board and players
    board = Board()
    player1 = Player((0, 8), 1)   # White player starts at (0, 8)
    player2 = Player((16, 8), -1) # Black player starts at (16, 8)
    board.add_player(player1)
    board.add_player(player2)
    minimax1 = Minimax(board, 3)  # Minimax depth set to 3 for player 1
    minimax2 = Minimax(board, 3)  # Minimax depth set to 3 for player 2

    # Play the first 10 moves for each player
    for move_num in range(10):
        # Player 1's turn
        print(f"Move {move_num * 2 + 1}: Player 1's turn")
        best_move = minimax1.run(player1)
        print(f"Best move for Player 1: {best_move}")
        apply_move(board, best_move, player1)
        board.display()
        print()

        if board.is_game_over():
            print(f"Player 1 wins after {move_num * 2 + 1} moves!")
            break

        # Player 2's turn
        print(f"Move {move_num * 2 + 2}: Player 2's turn")
        best_move = minimax2.run(player2)
        print(f"Best move for Player 2: {best_move}")
        apply_move(board, best_move, player2)
        board.display()
        print()

        if board.is_game_over():
            print(f"Player 2 wins after {move_num * 2 + 2} moves!")
            break

if __name__ == "__main__":
    main()
