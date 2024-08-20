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

    def act(self, state):
        board = state[0, :, :, self.player_id]
        player_position = np.argwhere(board == 1)[0]  # Find the player's position

        # Determine goal row based on the player ID
        goal_row = self.GOAL_ROW_P1 if self.player_id == 0 else self.GOAL_ROW_P2

        # Try to move directly towards the goal row
        if player_position[0] < goal_row and self.can_move_down(player_position):
            return self.move_down()
        elif player_position[0] > goal_row and self.can_move_up(player_position):
            return self.move_up()

        # If blocked, try to move horizontally to find another path
        if player_position[1] < self.COLS - 1 and self.can_move_right(player_position):
            return self.move_right()
        elif player_position[1] > 0 and self.can_move_left(player_position):
            return self.move_left()

        # If all else fails, move backwards or stay
        if player_position[0] < goal_row and self.can_move_up(player_position):
            return self.move_up()
        elif player_position[0] > goal_row and self.can_move_down(player_position):
            return self.move_down()
        
        #should never appen
        assert False, "No valid moves available"
        return self.move_stay()

    def can_move_up(self, position):
        return position[0] > 0 and not self.wall_above(position)

    def can_move_down(self, position):
        return position[0] < self.ROWS - 1 and not self.wall_below(position)

    def can_move_left(self, position):
        return position[1] > 0 and not self.wall_left(position)

    def can_move_right(self, position):
        return position[1] < self.COLS - 1 and not self.wall_right(position)

    def move_up(self):
        return 0  # Replace with actual action ID for moving up

    def move_down(self):
        return 1  # Replace with actual action ID for moving down

    def move_left(self):
        return 2  # Replace with actual action ID for moving left

    def move_right(self):
        return 3  # Replace with actual action ID for moving right

    def move_stay(self):
        return 4  # Replace with actual action ID for staying in place
