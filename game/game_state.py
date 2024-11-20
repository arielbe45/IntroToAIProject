import abc
import typing
import numpy as np

from game.bfs import bfs_shortest_paths
from game.move import (WallPlacement, BOARD_SIZE, Move, Movement, WallOrientation, ALL_MOVES, apply_movement,
                       AROUND_PLAYER)


class AbstractGameState(abc.ABC):
    def __init__(self):
        self.p1_turn: bool = True

    def get_new_state(self, move: Move, check_legal: bool = True) -> 'AbstractGameState':
        state = self.copy()
        state.apply_move(move=move, check_legal=check_legal)
        return state

    @abc.abstractmethod
    def copy(self) -> 'AbstractGameState':
        pass

    @abc.abstractmethod
    def is_move_legal(self, move: Move) -> bool:
        pass

    @abc.abstractmethod
    def p1_wins(self) -> bool:
        pass

    @abc.abstractmethod
    def p2_wins(self) -> bool:
        pass

    def is_winner(self) -> bool:
        return self.p1_wins() and not self.p1_turn or self.p2_wins() and self.p1_turn

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abc.abstractmethod
    def get_legal_moves(self) -> list[Move]:
        pass

    @abc.abstractmethod
    def apply_move(self, move: Move, check_legal: bool = True) -> None:
        pass


class GameState(AbstractGameState):
    def __init__(self):
        super().__init__()
        self.walls: list[WallPlacement] = []
        # In range 0 to BOARD_SIZE - 1 inclusive, top left is (0,0)
        self.player1_pos: list[int] = [int(BOARD_SIZE / 2), 0]
        self.player2_pos: list[int] = [int(BOARD_SIZE / 2), BOARD_SIZE - 1]

    def copy(self):
        state = GameState()
        state.walls = self.walls.copy()
        state.player1_pos = self.player1_pos[:]
        state.player2_pos = self.player2_pos[:]
        state.p1_turn = self.p1_turn
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

            # REMOVED: fixed bug
            # Check if the new position is already occupied by a player
            # if new_pos == self.player1_pos or new_pos == self.player2_pos:
            #     continue  # Tile is occupied

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

    def is_move_legal(self, move: Move) -> bool:
        # Check if it's a wall placement
        if isinstance(move, WallPlacement):
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

        # Check if move does not prevent players to reach other side
        new_state = self.get_new_state(move=move, check_legal=False)
        bfs_result = bfs_shortest_paths(start_node=tuple(self.player1_pos),
                                        get_free_neighbors=new_state.get_free_neighbor_tiles)
        winning_tiles = [(x, BOARD_SIZE - 1) for x in range(BOARD_SIZE)]
        if not any(tile in bfs_result for tile in winning_tiles):
            return False

        bfs_result = bfs_shortest_paths(start_node=tuple(self.player2_pos),
                                        get_free_neighbors=new_state.get_free_neighbor_tiles)
        winning_tiles = [(x, 0) for x in range(BOARD_SIZE)]
        if not any(tile in bfs_result for tile in winning_tiles):
            return False

        return True

    def get_legal_moves(self, restrict=False) -> list[Move]:
        legal_moves = [move for move in ALL_MOVES if self.is_move_legal(move=move)]
        if not restrict:
            return legal_moves
        # allows walls that are only near players
        res = []
        for move in legal_moves:
            if isinstance(move, WallPlacement):
                near1 = ((self.player1_pos[0] - AROUND_PLAYER <= move.center_x <= self.player1_pos[
                    0] + AROUND_PLAYER - 1)
                         and (self.player1_pos[1] - AROUND_PLAYER <= move.center_y <= self.player1_pos[
                            1] + AROUND_PLAYER))
                near2 = ((self.player2_pos[0] - AROUND_PLAYER <= move.center_x <= self.player2_pos[
                    0] + AROUND_PLAYER - 1)
                         and (self.player2_pos[1] - AROUND_PLAYER <= move.center_y <= self.player2_pos[
                            1] + AROUND_PLAYER))
                if near1 or near2:
                    res.append(move)
            else:
                res.append(move)
        return res

    def apply_move(self, move: Move, check_legal: bool = True) -> None:
        if check_legal and not self.is_move_legal(move):
            raise ValueError("Illegal move")

        # Apply a wall placement
        if isinstance(move, WallPlacement):
            self.walls.append(move)

        # Apply player movement
        elif isinstance(move, Movement):
            self.set_current_player_pos(apply_movement(move, self.get_current_player_pos()))

        # Switch the turn
        self.p1_turn = not self.p1_turn

    def is_game_over(self) -> bool:
        return self.p1_wins() or self.p2_wins()

    def p1_wins(self) -> bool:
        return self.player1_pos[1] == BOARD_SIZE - 1 or (not self.p1_turn and not self.has_legal_moves())

    def p2_wins(self) -> bool:
        return self.player2_pos[1] == 0 or (self.p1_turn and not self.has_legal_moves())

    def has_legal_moves(self) -> bool:
        for move in ALL_MOVES:
            if self.is_move_legal(move=move):
                return True
        return False

    def to_vector(self):
        """
        Converts the game state into a flattened 1D array representation.

        Returns:
            np.ndarray: A 1D array representing the game state.
        """
        board = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # Plane 0: Player 1 position
        board[0, self.player1_pos[0], self.player1_pos[1]] = 1.0

        # Plane 1: Player 2 position
        board[1, self.player2_pos[0], self.player2_pos[1]] = 1.0

        # Plane 2: Wall positions
        for wall in self.walls:
            if wall.orientation == WallOrientation.HORIZONTAL:
                board[2, wall.center_x, wall.center_y] = 1.0
            elif wall.orientation == WallOrientation.VERTICAL:
                board[2, wall.center_x + 1, wall.center_y] = 1.0

        # Flatten the board and append turn indicator
        flat_board = board.flatten()
        turn_indicator = np.array([1.0 if self.p1_turn else 0.0], dtype=np.float32)

        return np.concatenate((flat_board, turn_indicator))
