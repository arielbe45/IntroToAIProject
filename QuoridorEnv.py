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
        # super(QuoridorEnv, self).__init__()
        # self.action_space = spaces.Discrete(128)  # Simplified action space
        # self.observation_space = spaces.Box(low=0, high=1, shape=(9, 9, 2), dtype=np.float32)  # 9x9 board with two planes (players)
        self.game = Game()
        self.int_to_action = {i: value for i, value in enumerate(self.all_possible_moves())}
        self.action_space = len(self.int_to_action)
        self.state_shape = (17,17)
        # print(self.int_to_action)

    def all_possible_moves(self): # create players in the middle of an empty board and get all possible moves
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
        all_moves = board.get_all_possible_moves_without_optimize(white_player)
        normelized_all_moves = []
        for move in all_moves:
            if move[0] == "move":
                normelized_all_moves.insert(0, ("move", move[1] - x_White, move[2] - Y_White))
            else:
                normelized_all_moves.append(move)
        print("normlized", normelized_all_moves)
        return normelized_all_moves

    def reset(self):
        self.game = Game()
        return self.get_state()

    def step(self, action, player):
        reward = 0

        # Apply action to board (this part is simplified)
        # Update player position or place a wall based on the action
        player_to_play = self.game.white_player
        if player == -1:
            player_to_play = self.game.black_player
        if action <= 3: # move
            print(player_to_play.get_x())
            print(self.int_to_action[action])
            self.game.board.execute_move(player_to_play, player_to_play.get_x() + self.int_to_action[action][1], player_to_play.get_y() + self.int_to_action[action][2])
        else: # place wall
            self.game.board.execute_wall(Wall(self.int_to_action[action][1], (self.int_to_action[action][2], self.int_to_action[action][3]), 3))

        if self.game.board.is_game_over():
            reward = self.game.board.winner  # Reward for winning
        return self.get_state(), reward, self.game.board.is_game_over(), {}

    def invalid_actions(self, player):
        normelized_legal_moves = []
        if player == 1:
            legal_moves = self.game.board.get_all_possible_moves_without_optimize(self.game.white_player)
            for move in legal_moves:
                if move[0] == "move":
                    normelized_legal_moves.append(("move", move[1] - self.game.white_player.get_x(),
                                                    move[2] - self.game.white_player.get_y()))
                else:
                    normelized_legal_moves.append(move)
        elif player == -1:
            legal_moves = self.game.board.get_all_possible_moves_without_optimize(self.game.black_player)
            for move in legal_moves:
                if move[0] == "move":
                    normelized_legal_moves.append(("move", move[1] - self.game.black_player.get_x(),
                                                    move[2] - self.game.black_player.get_y()))
                else:
                    normelized_legal_moves.append(move)
        else:
            return []
        # Return a list of invalid actions based on the current board state
        # print("normlized", normelized_legal_moves)
        return list({k: v for k, v in self.int_to_action.items() if v not in normelized_legal_moves}.keys())

    def render(self, mode='human'):
        self.game.board.display()
    
    def get_state(self):
        return self.game.board.encode()

# env = QuoridorEnv()
# state_shape = env.observation_space.shape
# num_actions = env.action_space.n
# print(state_shape)
# print(num_actions)
# print(env.action_space)
# print("invalid", env.invalid_actions(1))
# env.render()
# print()
# env.step(3, 1)
# env.render()
# env.step(0, -1)
# env.render()
# print("put wall")
# env.step(13, 1)
# print(env.invalid_actions(1))
# for action in env.invalid_actions(1):
#     print(env.int_to_action[action])
# print("\n")
# env.render()
# print(env.game.board.encode())