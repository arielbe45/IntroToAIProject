from game.game_state import GameState
from game.move import Move
from run.graphics import Graphics
from run.player import AbstractQuoridorPlayer


class HumanPlayer(AbstractQuoridorPlayer):
    def __init__(self, graphics: Graphics):
        self.graphics = graphics

    def get_next_move(self, state: GameState) -> Move:
        self.graphics.display_board(state=state)
        move = self.graphics.wait_for_move(state=state)
        return move
