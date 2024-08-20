import random

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, invalid_actions=[]):
        options=self.action_space[:]
        options=[x for x in options if x not in invalid_actions]
        return random.choice(options)


import numpy as np
class ShortestPathAgent:
    ROWS = 9
    COLS = 9
    GOAL_ROW_P1 = ROWS - 1  # Goal row for Player 1
    GOAL_ROW_P2 = 0         # Goal row for Player 2

    def __init__(self, player_id):
        self.player_id = player_id  # Player 1 (0) or Player 2 (1)

    def find_player_position(self, board):
        board=board[0]
        for r, row in enumerate(board):
            for c, cell in enumerate(row):
                if cell == self.player_id:
                    return (r, c)
        raise ValueError(f"Player {self.player_id} not found on board.")


    def act(self, state, invalid_actions):
        player_position =self.find_player_position(state)
        # Determine goal row based on the player ID
        goal_row = self.GOAL_ROW_P1 if self.player_id == 0 else self.GOAL_ROW_P2

        # Try to move directly towards the goal row
        if player_position[0] < goal_row and self.can_move_down(invalid_actions):
            return self.move_down()
        elif player_position[0] > goal_row and self.can_move_up(invalid_actions):
            return self.move_up()

        # If blocked, try to move horizontally to find another path
        if player_position[1] < self.COLS - 1 and self.can_move_right(invalid_actions):
            return self.move_right()
        elif player_position[1] > 0 and self.can_move_left(invalid_actions):
            return self.move_left()

        # If all else fails, move backwards or stay
        if player_position[0] < goal_row and self.can_move_up(invalid_actions):
            return self.move_up()
        elif player_position[0] > goal_row and self.can_move_down(invalid_actions):
            return self.move_down()
        
        #should never appen
        assert False, "No valid moves available"
        return self.move_stay()

    def can_move_up(self, invalid_actions):
        return  self.move_up() not in invalid_actions

    def can_move_down(self, invalid_actions):
        return self.move_down() not in invalid_actions

    def can_move_left(self, invalid_actions):
        return self.move_left() not in invalid_actions

    def can_move_right(self, invalid_actions):
        return self.move_right() not in invalid_actions

    def move_up(self):
        return 0  # Replace with actual action ID for moving up

    def move_down(self):
        return 2  # Replace with actual action ID for moving down

    def move_left(self):
        return 1  # Replace with actual action ID for moving left

    def move_right(self):
        return 3  # Replace with actual action ID for moving right

    def move_stay(self):
        return 4  # Replace with actual action ID for staying in place
