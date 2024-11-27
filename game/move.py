import enum
from dataclasses import dataclass
from typing import Tuple

BOARD_SIZE = 9
NUMBER_OF_WALLS = 20
MAX_NUMBER_OF_TURNS = 50


class WallOrientation(enum.Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class Move:
    pass


@dataclass
class WallPlacement(Move):
    orientation: WallOrientation
    center_x: int  # Range 0 to BOARD_SIZE - 2 inclusive, top left is (0,0)
    center_y: int


class Movement(Move, enum.Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3


def apply_movement(movement: Move, pos: list[int]):
    pos = list(pos).copy()
    if movement == Movement.MOVE_UP:
        pos[1] -= 1
    elif movement == Movement.MOVE_DOWN:
        pos[1] += 1
    elif movement == Movement.MOVE_LEFT:
        pos[0] -= 1
    elif movement == Movement.MOVE_RIGHT:
        pos[0] += 1
    return pos


ALL_MOVES = [Movement.MOVE_UP, Movement.MOVE_DOWN, Movement.MOVE_LEFT, Movement.MOVE_RIGHT] + \
            [WallPlacement(orientation=orientation, center_x=center_x, center_y=center_y)
             for orientation in WallOrientation for center_x in range(BOARD_SIZE - 1) for center_y in
             range(BOARD_SIZE - 1)]

AROUND_PLAYER = 1
