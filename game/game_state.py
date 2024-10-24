import abc
import typing

from game.move import WallPlacement, BOARD_SIZE, Move, Movement, WallOrientation, ALL_MOVES, apply_movement
from game.bfs import bfs_shortest_paths


class AbstractGameState(abc.ABC):
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
    def is_game_over(self) -> bool:
        pass

    @abc.abstractmethod
    def get_legal_moves(self) -> list[Move]:
        pass

    @abc.abstractmethod
    def apply_move(self, move: Move) -> None:
        pass


class GameState(AbstractGameState):
    def __init__(self):
        self.walls: list[WallPlacement] = []
        # In range 0 to BOARD_SIZE - 1 inclusive, top left is (0,0)
        self.player1_pos: list[int] = [int(BOARD_SIZE / 2), 0]
        self.player2_pos: list[int] = [int(BOARD_SIZE / 2), BOARD_SIZE - 1]
        self.p1_turn: bool = True

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

            # Check if the new position is already occupied by a player
            if new_pos == self.player1_pos or new_pos == self.player2_pos:
                continue  # Tile is occupied

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
        bfs_result = bfs_shortest_paths(start_node=tuple(self.player1_pos), get_free_neighbors=new_state.get_free_neighbor_tiles)
        winning_tiles = [(x, BOARD_SIZE - 1) for x in range(BOARD_SIZE)]
        if not any(tile in bfs_result for tile in winning_tiles):
            return False

        bfs_result = bfs_shortest_paths(start_node=tuple(self.player2_pos), get_free_neighbors=new_state.get_free_neighbor_tiles)
        winning_tiles = [(x, 0) for x in range(BOARD_SIZE)]
        if not any(tile in bfs_result for tile in winning_tiles):
            return False

        return True

    def get_legal_moves(self) -> list[Move]:
        return [move for move in ALL_MOVES if self.is_move_legal(move=move)]

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
        # Player 1 wins by reaching the rightmost column
        if self.player1_pos[1] == BOARD_SIZE - 1:
            return True

        # Player 2 wins by reaching the leftmost column
        if self.player2_pos[1] == 0:
            return True

        return False
