from game import game_state
from game.game_state import GameState
from run.graphics import Graphics
from run.human_player import HumanPlayer
from run.player import AbstractQuoridorPlayer, RandomQuoridorPlayer, MinimaxPlayer
from run.player import *
from run.rl import DeepQLearningPlayer, train_agent
from run.mcts_player import MCTSPlayer
from tqdm import tqdm


def game(p1: AbstractQuoridorPlayer, p2: AbstractQuoridorPlayer):
    # returns True if p1 won
    game_state = GameState()
    while not game_state.is_game_over():
        current_player = p1 if game_state.p1_turn else p2
        move = current_player.get_next_move(state=game_state)
        game_state.apply_move(move=move)
    return game_state.p1_wins()


def player_vs_player():
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), HumanPlayer(graphics)
    game(p1=p1, p2=p2)


def player_vs_ai(ai_player: AbstractQuoridorPlayer):
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), ai_player
    game(p1=p1, p2=p2)


def competition(p1, p2):
    GAMES = 10
    cnt1 = 0
    cnt2 = 0
    for i in tqdm(range(GAMES)):
        does_p1_won = game(p1=p1, p2=p2)
        if does_p1_won:
            cnt1 += 1
        else:
            cnt2 += 1
    print("################")
    print(f"{GAMES} Games")
    print(p1, "\t", p2)
    print(cnt1, "\t", cnt2)
    print("################")


if __name__ == '__main__':
    player_vs_ai(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, player=2, depth=1))
    # player_vs_ai(
    #     MCTSPlayer(heurisric=distance_to_end_heuristic, player=2, depth=1, restrict_walls=True, num_simulations=10))
    # train_agent(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, player=2, depth=0))
    # player_vs_ai(DeepQLearningPlayer(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=0)))
    # competition(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=1),
    #             MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=1))
    # player_vs_ai(MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=1))
    # player_vs_ai(RandomQuoridorPlayer())
    # dqn_player = DeepQLearningPlayer()
    # dqn_player.load_model("trained_q_network.pth")
