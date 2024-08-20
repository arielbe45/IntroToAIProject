
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from board import Board
from game import Game
from player import Player
from wall import Wall
from move import Move
from cell import Cell
from placeWall import PlaceWall
from minimax import Minimax

class QuoridorEnv(gym.Env):
    def __init__(self):
        super(QuoridorEnv, self).__init__()
        # self.action_space = spaces.Discrete(128)  # Simplified action space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(9, 9, 2), dtype=np.float32)  # 9x9 board with two planes (players)
        self.game = Game()

    def all_possible_moves(player):
        Y_White = 8
        Y_Black = 8
        x_White = 8
        x_Black = 4
        board = Board()
        white_player = Player((x_White, Y_White), 1)
        black_player = Player((x_Black, Y_Black), -1)
        board.set_cell(x_White, Y_White, Cell(x_White, Y_White, white_player))
        board.set_cell(x_Black, Y_Black, Cell(x_Black, Y_Black, black_player))
        board.add_player(white_player)
        board.add_player(black_player)
        return board.get_all_possible_moves(player)

    def reset(self):
        self.game.board = Board()
        return self.game.board

    def step(self, action, action_type):
        reward = 0

        # Apply action to board (this part is simplified)
        # Update player position or place a wall based on the action
        if action_type == 0: # move
            self.game.board.execute_move(action)
        elif action_type == 1: # place wall
            self.game.board.execute_wall(action)

        if self.game.board.is_game_over():
            reward = self.game.board.winner  # Reward for winning
        return self.board, reward, self.done, {}

    def invalid_actions(self, player):
        legal_moves = self.game.board.get_all_possible_moves(player)
        # Return a list of invalid actions based on the current board state
        return self.all_possible_moves(player) - legal_moves

    def render(self, mode='human'):
        self.game.display_board()