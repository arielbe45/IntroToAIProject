
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

class QuoridorEnv():
    def __init__(self):
        super(QuoridorEnv, self).__init__()
        # self.action_space = spaces.Discrete(128)  # Simplified action space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(9, 9, 2), dtype=np.float32)  # 9x9 board with two planes (players)
        self.game = Game()
        self.int_to_action = {i: value for i, value in enumerate(self.all_possible_moves(1))}
        print(self.int_to_action)
        
    def all_possible_moves(self,player): # create players in the middle of an empty board and get all possible moves
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
        if player == 1:
            all_moves = board.get_all_possible_moves(white_player)
            for move in all_moves:
                if move[0] == "move":
                    move[1] = move[1] - x_White
                    move[2] = move[2] - Y_White
        if player == -1:
            all_moves = board.get_all_possible_moves(black_player)
            for move in all_moves:
                if move[0] == "move":
                    move[1] = move[1] - x_Black
                    move[2] = move[2] - Y_Black
        return all_moves

    def reset(self):
        self.game = Game()
        return self.game

    def step(self, action, action_type):
        reward = 0

        # Apply action to board (this part is simplified)
        # Update player position or place a wall based on the action
        if action_type == 0: # move
            self.game.board.execute_move(self.int_to_action[action])
        elif action_type == 1: # place wall
            self.game.board.execute_wall(self.int_to_action[action])

        if self.game.board.is_game_over():
            reward = self.game.board.winner  # Reward for winning
        return self.board, reward, self.done, {}

    def invalid_actions(self, player):
        legal_moves = self.game.board.get_all_possible_moves(player)
        if player == 1:
            for move in legal_moves:
                if move[0] == "move":
                    move[1] = move[1] - self.game.white_player.get_x()
                    move[2] = move[2] - self.game.white_player.get_y()
        if player == -1:
            for move in legal_moves:
                if move[0] == "move":
                    move[1] = move[1] - self.game.black_player.get_x()
                    move[2] = move[2] - self.game.black_player.get_y()
        # Return a list of invalid actions based on the current board state
        return {k: v for k, v in self.int_to_action.items() if v not in legal_moves}.keys()

    def render(self, mode='human'):
        self.game.display_board()
        
        
env = QuoridorEnv()
