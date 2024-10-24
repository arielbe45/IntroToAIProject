from game.game_state import GameState
from run.graphics import Graphics
from run.human_player import HumanPlayer
from run.player import AbstractQuoridorPlayer, RandomQuoridorPlayer, MinimaxPlayer, distance_to_end_heuristic


def game(p1: AbstractQuoridorPlayer, p2: AbstractQuoridorPlayer):
    game_state = GameState()
    while not game_state.is_game_over():
        current_player = p1 if game_state.p1_turn else p2
        move = current_player.get_next_move(state=game_state)
        game_state.apply_move(move=move)


def player_vs_player():
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), HumanPlayer(graphics)
    game(p1=p1, p2=p2)


def player_vs_ai(ai_player: AbstractQuoridorPlayer):
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), ai_player
    game(p1=p1, p2=p2)


if __name__ == '__main__':
    # player_vs_player()
    player_vs_ai(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=0))
    # player_vs_ai(RandomQuoridorPlayer())
