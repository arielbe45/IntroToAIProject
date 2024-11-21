import time
from typing import Tuple, Optional

import pygame

from game.game_state import GameState, Move, BOARD_SIZE, WallOrientation, WallPlacement, Movement
from game.move import apply_movement

TILE_SIZE = 60  # Size of each square tile in pixels
WALL_THICKNESS = 10  # Thickness of the walls
PLAYER_RADIOS = TILE_SIZE // 3  # Radios of each circle representing a player
SCREEN_SIZE = TILE_SIZE * BOARD_SIZE
SCREEN_HEIGHT = TILE_SIZE * BOARD_SIZE + 50  # Extra space for text


def get_tile_closest_to_coords(coords) -> Tuple[int, int, int]:
    """
    Returns the coordinates of the tile closest to the given pixel on screen and the distance to the tile squared
    :param coords: A tuple of (x, y)
    :return: (tile_x, tile_y, distance squared) where tile_x and tile_y are the coordinates of the closed tile
    """
    tile_x, tile_y = round(coords[0] / TILE_SIZE - 0.5), round(coords[1] / TILE_SIZE - 0.5)
    # result is in range 0 - BOARD_SIZE - 1
    tile_x, tile_y = min(max(tile_x, 0), BOARD_SIZE - 1), min(max(tile_y, 0), BOARD_SIZE - 1)
    tile_coords = ((tile_x + 0.5) * TILE_SIZE, (tile_y + 0.5) * TILE_SIZE)
    distance = (coords[0] - tile_coords[0]) ** 2 + (coords[1] - tile_coords[1]) ** 2
    return tile_x, tile_y, distance


def get_wall_closest_to_coords(coords) -> Tuple[int, int, int]:
    """
    Returns the coordinates of the wall closest to the given pixel on screen and the distance to the wall squared
    :param coords: A tuple of (x, y)
    :return: (wall_x, wall_y, distance squared) where wall_x and wall_y are the coordinates of the closed wall
    """
    wall_x, wall_y = round(coords[0] / TILE_SIZE) - 1, round(coords[1] / TILE_SIZE) - 1
    # result is in range 0 - BOARD_SIZE - 2
    wall_x, wall_y = min(max(wall_x, 0), BOARD_SIZE - 2), min(max(wall_y, 0), BOARD_SIZE - 2)
    wall_coords = ((wall_x + 1) * TILE_SIZE, (wall_y + 1) * TILE_SIZE)
    distance = (coords[0] - wall_coords[0]) ** 2 + (coords[1] - wall_coords[1]) ** 2
    return wall_x, wall_y, distance


class Graphics:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_HEIGHT))
        pygame.display.set_caption('Quoridor Game')

        self.font = pygame.font.Font(None, 36)  # Default font for wall count
        self.colors = {
            "background": (255, 255, 255),  # White background
            "grid": (0, 0, 0),  # Black grid
            "player1": (255, 0, 0),  # Red for player 1
            "player2": (0, 0, 255),  # Blue for player 2
            "wall": (0, 0, 0),  # Black walls
            "hover": (120, 120, 120)  # Gray hover walls
        }
        self.new_wall_orientation = WallOrientation.VERTICAL

    def draw_grid(self):
        # Fill the screen with the background color
        self.screen.fill(self.colors["background"])

        # Draw grid
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(
                    self.screen,
                    self.colors["grid"],
                    pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                    1  # Grid thickness
                )

    def draw_wall(self, wall: WallPlacement, color):
        if wall.orientation == WallOrientation.HORIZONTAL:
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(wall.center_x * TILE_SIZE, (wall.center_y + 1) * TILE_SIZE - WALL_THICKNESS // 2,
                            TILE_SIZE * 2, WALL_THICKNESS)
            )
        elif wall.orientation == WallOrientation.VERTICAL:
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect((wall.center_x + 1) * TILE_SIZE - WALL_THICKNESS // 2, wall.center_y * TILE_SIZE,
                            WALL_THICKNESS, TILE_SIZE * 2)
            )

    def draw_player(self, player_pos: list[int], color: Tuple[int, int, int]):
        pygame.draw.circle(
            self.screen,
            color,
            (int(player_pos[0] * TILE_SIZE + TILE_SIZE / 2), int(player_pos[1] * TILE_SIZE + TILE_SIZE / 2)),
            PLAYER_RADIOS
        )

    def draw_wall_counts(self, state: GameState):
        """Displays the remaining wall counts for both players."""
        p1_text = self.font.render(f"P1 Walls: {state.p1_walls_remaining}", True, self.colors["player1"])
        p2_text = self.font.render(f"P2 Walls: {state.p2_walls_remaining}", True, self.colors["player2"])

        self.screen.blit(p1_text, (10, SCREEN_SIZE + 10))  # Bottom-left of the screen
        self.screen.blit(p2_text, (SCREEN_SIZE - 200, SCREEN_SIZE + 10))  # Bottom-right of the screen

    def display_board(self, state: GameState) -> None:
        self.draw_grid()

        # Draw walls
        for wall in state.walls:
            self.draw_wall(wall, color=self.colors["wall"])

        # Draw players
        self.draw_player(player_pos=state.player1_pos, color=self.colors["player1"])
        self.draw_player(player_pos=state.player2_pos, color=self.colors["player2"])
        self.draw_wall_counts(state)
        pygame.display.flip()

    def get_chosen_move(self, mouse_pos: tuple[int, int], state: GameState) -> Optional[Move]:
        tile_x, tile_y, tile_distance = get_tile_closest_to_coords(coords=mouse_pos)
        wall_x, wall_y, wall_distance = get_wall_closest_to_coords(coords=mouse_pos)
        if tile_distance < wall_distance:
            current_pos = state.player1_pos if state.p1_turn else state.player2_pos
            if tile_x == current_pos[0] and tile_y == current_pos[1] + 1:
                move = Movement.MOVE_DOWN
            elif tile_x == current_pos[0] and tile_y == current_pos[1] - 1:
                move = Movement.MOVE_UP
            elif tile_x == current_pos[0] + 1 and tile_y == current_pos[1]:
                move = Movement.MOVE_RIGHT
            elif tile_x == current_pos[0] - 1 and tile_y == current_pos[1]:
                move = Movement.MOVE_LEFT
            else:
                return None
        else:
            move = WallPlacement(center_x=wall_x, center_y=wall_y, orientation=self.new_wall_orientation)
        if state.is_move_legal(move=move):
            return move
        return None

    def display_chosen_move(self, mouse_pos: tuple[int, int], state: GameState) -> None:
        move = self.get_chosen_move(mouse_pos=mouse_pos, state=state)
        if move is None:
            return
        if isinstance(move, Movement):
            new_pos = apply_movement(movement=move, pos=state.get_current_player_pos())
            self.draw_player(player_pos=new_pos, color=self.colors['hover'])
        elif isinstance(move, WallPlacement):
            self.draw_wall(wall=move, color=self.colors['hover'])

    def wait_for_move(self, state: GameState) -> Move:
        while True:
            for event in pygame.event.get():
                pygame.time.delay(10)

                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = pygame.mouse.get_pos()
                    self.display_board(state=state)
                    self.display_chosen_move(mouse_pos=mouse_pos, state=state)
                    pygame.display.flip()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == pygame.BUTTON_RIGHT:
                        # Flip wall orientation on right mouse button click
                        if self.new_wall_orientation is WallOrientation.VERTICAL:
                            self.new_wall_orientation = WallOrientation.HORIZONTAL
                        else:
                            self.new_wall_orientation = WallOrientation.VERTICAL
                        mouse_pos = pygame.mouse.get_pos()
                        self.display_board(state=state)
                        self.display_chosen_move(mouse_pos=mouse_pos, state=state)
                        pygame.display.flip()
                    elif event.button == pygame.BUTTON_LEFT:
                        # Get the mouse position when left mouse button is clicked
                        mouse_pos = pygame.mouse.get_pos()
                        move = self.get_chosen_move(mouse_pos=mouse_pos, state=state)
                        if move is not None:
                            return move
