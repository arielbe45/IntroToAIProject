import typing

import numpy as np

from game.bfs import bfs_distance_to_goal
from game.move import (WallPlacement, BOARD_SIZE, NUMBER_OF_WALLS, Move, Movement, WallOrientation, ALL_MOVES,
                       apply_movement, RESTRICT_WALLS_PLAYER_RADIUS, MAX_NUMBER_OF_TURNS)


def player1_dist_to_goal(state: 'GameState'):
    return bfs_distance_to_goal(start_node=tuple(state.player1_pos), get_free_neighbors=state.get_free_neighbor_tiles,
                                check_goal=lambda pos: pos[1] == BOARD_SIZE - 1)


def player2_dist_to_goal(state: 'GameState'):
    return bfs_distance_to_goal(start_node=tuple(state.player2_pos), get_free_neighbors=state.get_free_neighbor_tiles,
                                check_goal=lambda pos: pos[1] == 0)


class GameState:
    def __init__(self):
        super().__init__()
        self.walls: list[WallPlacement] = []
        # In range 0 to BOARD_SIZE - 1 inclusive, top left is (0,0)
        self.player1_pos: list[int] = [int(BOARD_SIZE / 2), 0]
        self.player2_pos: list[int] = [int(BOARD_SIZE / 2), BOARD_SIZE - 1]
        self.p1_walls_remaining = NUMBER_OF_WALLS
        self.p2_walls_remaining = NUMBER_OF_WALLS
        self.time = 0
        self.p1_turn = True

    def get_new_state(self, move: Move, check_legal) -> 'GameState':
        state = self.copy()
        state.apply_move(move=move, check_legal=check_legal)
        return state

    @property
    def current_player(self):
        return 1 if self.p1_turn else 2

    def is_winner(self) -> bool:
        return self.p1_wins() and not self.p1_turn or self.p2_wins() and self.p1_turn

    def copy(self):
        state = GameState()
        state.walls = self.walls.copy()
        state.player1_pos = self.player1_pos[:]
        state.player2_pos = self.player2_pos[:]
        state.player1_walls_remaining = self.p1_walls_remaining
        state.player2_walls_remaining = self.p2_walls_remaining
        state.p1_turn = self.p1_turn
        state.time = self.time
        return state

    def get_current_player_pos(self) -> list[int]:
        return self.player1_pos if self.p1_turn else self.player2_pos

    def get_free_neighbor_tiles(self, pos: typing.Tuple[int, int]) -> list[typing.Tuple[int, int]]:
        free_tiles = []
        for move in Movement:
            new_pos = apply_movement(movement=move, pos=pos)

            # Check bounds
            if not (0 <= new_pos[0] < BOARD_SIZE and 0 <= new_pos[1] < BOARD_SIZE):
                continue  # Out of bounds

            # Check if the move collides with walls
            if self.check_wall_collision(pos=pos, move=move):
                continue

            free_tiles.append(tuple(new_pos))
        return free_tiles

    def set_current_player_pos(self, pos: list[int]) -> None:
        self.get_current_player_pos()[:] = pos

    def check_wall_collision(self, pos: typing.Tuple[int, int], move: Movement) -> bool:
        for wall in self.walls:
            if move == Movement.MOVE_UP and wall.orientation == WallOrientation.HORIZONTAL:
                # Horizontal wall blocks upward movement at both (col, col + 1)
                if pos[1] == wall.center_y + 1 and pos[0] in (
                        wall.center_x, wall.center_x + 1):
                    return True  # Wall above blocks upward movement

            elif move == Movement.MOVE_DOWN and wall.orientation == WallOrientation.HORIZONTAL:
                # Horizontal wall blocks downward movement at both (col, col + 1)
                if pos[1] == wall.center_y and pos[0] in (
                        wall.center_x, wall.center_x + 1):
                    return True  # Wall below blocks downward movement

            elif move == Movement.MOVE_LEFT and wall.orientation == WallOrientation.VERTICAL:
                # Vertical wall blocks leftward movement at both (row, row + 1)
                if pos[0] == wall.center_x + 1 and pos[1] in (
                        wall.center_y, wall.center_y + 1):
                    return True  # Wall to the left blocks left movement

            elif move == Movement.MOVE_RIGHT and wall.orientation == WallOrientation.VERTICAL:
                # Vertical wall blocks rightward movement at both (row, row + 1)
                if pos[0] == wall.center_x and pos[1] in (
                        wall.center_y, wall.center_y + 1):
                    return True  # Wall to the right blocks right movement
        return False

    def is_move_legal(self, move: Move, check_bfs: bool = True) -> bool:
        # Check if it's a wall placement
        if isinstance(move, WallPlacement):
            # Check if the current player has walls left to place
            if (self.p1_turn and self.p1_walls_remaining <= 0) or (not self.p1_turn and self.p2_walls_remaining <= 0):
                return False  # No walls remaining for this player
            # Ensure wall is within bounds
            if not (0 <= move.center_x < BOARD_SIZE - 1 and 0 <= move.center_y < BOARD_SIZE - 1):
                return False  # Wall out of bounds

            # Check for overlapping or adjacent walls
            for wall in self.walls:
                if move.center_x == wall.center_x and move.center_y == wall.center_y:
                    return False  # Overlapping wall

                # Check adjacent walls with the same orientation
                if move.orientation == WallOrientation.HORIZONTAL == wall.orientation and \
                        move.center_y == wall.center_y and abs(move.center_x - wall.center_x) == 1:
                    return False  # Adjacent horizontal wall
                elif move.orientation == WallOrientation.VERTICAL == wall.orientation and \
                        move.center_x == wall.center_x and abs(move.center_y - wall.center_y) == 1:
                    return False  # Adjacent vertical wall

        # Check if it's a player movement
        elif isinstance(move, Movement):
            new_pos = apply_movement(move, self.get_current_player_pos())

            # Check bounds
            if not (0 <= new_pos[0] < BOARD_SIZE and 0 <= new_pos[1] < BOARD_SIZE):
                return False  # Out of bounds

            # Check if the move collides with walls
            if self.check_wall_collision(pos=self.get_current_player_pos(), move=move):
                return False

            # Check if the new position is already occupied by the other player
            other_player_pos = self.player2_pos if self.p1_turn else self.player1_pos
            if new_pos == other_player_pos:
                return False  # Collision with the other player
        else:
            return False  # Unknown move type

        if not check_bfs:
            return True

        new_state = self.get_new_state(move=move, check_legal=False)
        return new_state.check_state_legal()

    def check_state_legal(self):
        return player1_dist_to_goal(self) < float('inf') and player2_dist_to_goal(self) < float('inf')

    def get_legal_moves(self, restrict=False, check_bfs: bool = True) -> list[Move]:
        if not restrict:
            return [move for move in ALL_MOVES if self.is_move_legal(move=move, check_bfs=check_bfs)]
        # allows walls that are only near players
        res = []
        for move in ALL_MOVES:
            if isinstance(move, WallPlacement):
                near1 = ((self.player1_pos[0] - RESTRICT_WALLS_PLAYER_RADIUS <= move.center_x <= self.player1_pos[
                    0] + RESTRICT_WALLS_PLAYER_RADIUS - 1)
                         and (self.player1_pos[1] - RESTRICT_WALLS_PLAYER_RADIUS <= move.center_y <= self.player1_pos[
                            1] + RESTRICT_WALLS_PLAYER_RADIUS))
                near2 = ((self.player2_pos[0] - RESTRICT_WALLS_PLAYER_RADIUS <= move.center_x <= self.player2_pos[
                    0] + RESTRICT_WALLS_PLAYER_RADIUS - 1)
                         and (self.player2_pos[1] - RESTRICT_WALLS_PLAYER_RADIUS <= move.center_y <= self.player2_pos[
                            1] + RESTRICT_WALLS_PLAYER_RADIUS))
                if (near1 and not self.p1_turn or near2 and self.p1_turn) and self.is_move_legal(move=move,
                                                                                                 check_bfs=check_bfs):
                    res.append(move)
            elif self.is_move_legal(move=move, check_bfs=check_bfs):
                res.append(move)
        return res

    def apply_move(self, move: Move, check_legal: bool = True) -> None:
        if check_legal and not self.is_move_legal(move):
            raise ValueError("Illegal move")

        # Apply a wall placement
        if isinstance(move, WallPlacement):
            self.walls.append(move)
            if self.p1_turn:
                self.p1_walls_remaining -= 1
            else:
                self.p2_walls_remaining -= 1
        # Apply player movement
        elif isinstance(move, Movement):
            self.set_current_player_pos(apply_movement(move, self.get_current_player_pos()))

        # Switch the turn
        self.p1_turn = not self.p1_turn
        self.time += 1

    def is_game_over(self) -> bool:
        return self.p1_wins() or self.p2_wins() or self.tie()

    def p1_wins(self) -> bool:
        return self.player1_pos[1] == BOARD_SIZE - 1 or (not self.p1_turn and not self.can_move())

    def p2_wins(self) -> bool:
        return self.player2_pos[1] == 0 or (self.p1_turn and not self.can_move())

    def tie(self):
        return self.time >= MAX_NUMBER_OF_TURNS

    def can_move(self) -> bool:
        for move in Movement:
            if self.is_move_legal(move=move, check_bfs=False):
                return True
        return False

    def to_vector(self):
        """
        Converts the game state into a flattened 1D array representation.

        Returns:
            np.ndarray: A 1D array representing the game state.
        """
        board = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # Plane 0: First player position
        board[0, self.player1_pos[0], self.player1_pos[1]] = 1.0

        # Plane 1: Second player position
        board[0, self.player2_pos[0], self.player2_pos[1]] = 1.0

        # Plane 2: Wall positions
        for wall in self.walls:
            if wall.orientation == WallOrientation.HORIZONTAL:
                board[2, wall.center_x, wall.center_y] = 1.0
            elif wall.orientation == WallOrientation.VERTICAL:
                board[3, wall.center_x, wall.center_y] = 1.0

        # Flatten the board and append walls
        flat_board = board.flatten()
        walls = np.array([self.p1_turn, self.p1_walls_remaining, self.p2_walls_remaining], dtype=np.float32)
        return np.concatenate((flat_board, walls))
