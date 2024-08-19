import unittest
from board import Board
from player import Player
from move import Move
from placeWall import PlaceWall
from wall import Wall
from minimax import Minimax
from game import Game

width = 16
class TestGame(unittest.TestCase):

    def setUp(self):
        self.board = Board()
        self.white_player = Player((0, width/2), 1)
        self.black_player = Player((width, width/2), -1)
        self.board.add_player(self.white_player)
        self.board.add_player(self.black_player)
        self.minimax = Minimax(self.board, depth=3)
        self.game = Game()

    def test_initial_setup(self):
        self.assertEqual(self.white_player.get_position(), (0, width/2))
        self.assertEqual(self.black_player.get_position(), (width, width/2))
        self.assertTrue(self.board.get_cell(0, width/2).is_player())
        self.assertTrue(self.board.get_cell(width, width/2).is_player())

    def test_player_move(self):
        move = Move(self.white_player, self.board, self.board.get_cell(0, 8), self.board.get_cell(0, 10))
        self.assertTrue(move.is_valid_move())
        self.game.execute_move(move)
        self.assertEqual(self.white_player.get_position(), (0, 10))
        self.assertTrue(self.board.get_cell(0, 10).is_player())
        self.assertFalse(self.board.get_cell(0, 8).is_player())

    def test_wall_placement(self):
        wall = Wall('horizontal', (7, 8), 3)
        place_wall = PlaceWall(self.board)
        self.assertTrue(place_wall.place_wall(wall, self.white_player, self.black_player))
        self.assertTrue(self.board.get_cell(7, 8).is_wall())
        self.assertTrue(self.board.get_cell(7, 9).is_wall())
        self.assertTrue(self.board.get_cell(7, 10).is_wall())

    def test_minimax_move(self):
        best_board = self.minimax.run(self.black_player)
        self.assertIsNotNone(best_board)

    def test_game_over(self):
        self.game.set_white_playe((16, 8))
        self.assertTrue(self.game.is_game_over()==1)
        self.assertEqual(self.game.get_board_winner(), 1)

    def test_board_clone(self):
        clone_board = self.board.clone()
        self.assertEqual(clone_board.get_cell(0, 8).is_player(), self.board.get_cell(0, 8).is_player())
        self.assertEqual(clone_board.get_cell(16, 8).is_player(), self.board.get_cell(16, 8).is_player())

    def test_find_shortest_path(self):
        board = Board()
        self.assertTrue(board.find_shortest_path(board.get_cell(0, 0), board.get_cell(16, 16))>0)

    def test_invalid_move(self):
        invalid_move = Move(self.white_player, self.board, self.board.get_cell(0, 8), self.board.get_cell(16, 8))
        self.assertFalse(invalid_move.is_valid_move())

    def test_invalid_wall_placement(self):
        wall = Wall('horizontal', (0, 8), 3)
        place_wall = PlaceWall(self.board)
        self.assertFalse(place_wall.place_wall(wall, self.white_player, self.black_player))

    def setUp(self):
        self.board = Board()
        self.white_player = Player((0, 8), 1)
        self.black_player = Player((16, 8), -1)
        self.board.add_player(self.white_player)
        self.board.add_player(self.black_player)
        self.minimax = Minimax(self.board, depth=3)
        self.game = Game()

    def test_wall_placement(self):
        wall = Wall('horizontal', (7, 8), 3)
        place_wall = PlaceWall(self.board)
        self.assertTrue(place_wall.place_wall(wall, self.white_player, self.black_player))
        self.assertTrue(self.board.get_cell(7, 8).is_wall())
        self.assertTrue(self.board.get_cell(7, 9).is_wall())
        self.assertTrue(self.board.get_cell(7, 10).is_wall())

        wall = Wall('vertical', (8, 7), 3)
        self.assertTrue(place_wall.place_wall(wall, self.white_player, self.black_player))
        self.assertTrue(self.board.get_cell(8, 7).is_wall())
        self.assertTrue(self.board.get_cell(9, 7).is_wall())
        self.assertTrue(self.board.get_cell(10, 7).is_wall())


if __name__ == '__main__':
    unittest.main()
