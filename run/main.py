import time

from tqdm import tqdm
from multiprocessing import Pool

from run.graphics import Graphics
from run.human_player import HumanPlayer
from run.mcts_player import MCTSPlayer
from run.player import *
from run.rl import DeepQLearningPlayer, train_agent, visualize_minimax_game_q_values, plot_loss_graph


def game(p1: AbstractQuoridorPlayer, p2: AbstractQuoridorPlayer, graphics: Graphics = None):
    # returns 1 if p1 won, 0 if lost and 0.5 if tie
    game_state = GameState()
    while not game_state.is_game_over():
        current_player = p1 if game_state.p1_turn else p2
        move = current_player.get_next_move(state=game_state)
        game_state.apply_move(move=move)
    if graphics is not None:
        graphics.display_board(game_state)
        time.sleep(1)
    if game_state.p1_wins():
        return 1
    elif game_state.p2_wins():
        return 0
    return 0.5


def player_vs_player():
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), HumanPlayer(graphics)
    game(p1=p1, p2=p2, graphics=graphics)


def player_vs_ai(ai_player: AbstractQuoridorPlayer):
    graphics = Graphics()
    p1, p2 = HumanPlayer(graphics), ai_player
    game(p1=p1, p2=p2, graphics=graphics)


def display_ai_vs_ai(p1: AbstractQuoridorPlayer, p2: AbstractQuoridorPlayer):
    graphics = Graphics()
    game_state = GameState()
    graphics.display_board(state=game_state)
    while not game_state.is_game_over():
        current_player = p1 if game_state.p1_turn else p2
        move = current_player.get_next_move(state=game_state)
        game_state.apply_move(move=move)
        graphics.display_board(state=game_state)
        time.sleep(1)
    graphics.display_board(state=game_state)
    time.sleep(1)
    return game_state.p1_wins()


def competition(p1, p2, games=100, title: str = '?', verbose=False):
    cnt1 = 0
    cnt2 = 0
    for i in tqdm(range(1, games + 1)):
        if random.random() < 0.5:
            p1_score = game(p1=p1, p2=p2)
            p2_score = 1 - p1_score
        else:
            p2_score = game(p1=p2, p2=p1)
            p1_score = 1 - p2_score
        cnt1 += p1_score
        cnt2 += p2_score

        if verbose:
            print("################")
            print(f"{i} Games: {title}")
            print(f"player1 wins:\t{cnt1}/{i},{cnt1 / i * 100:1}%")
            print(f"player2 wins:\t{cnt2}/{i},{cnt2 / i * 100:1}%")
            print("################")

    print("################")
    print(f"{games} Games: {title}")
    print(f"player1 wins:\t{cnt1}/{i},{cnt1 / i * 100:1}%")
    print(f"player2 wins:\t{cnt2}/{i},{cnt2 / i * 100:1}%")
    print("################")


def heuristic(*args, **kwargs):
    heuristic.count += 1
    return normalized_distance_to_end_heuristic(*args, **kwargs)


heuristic_mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=True,
                            num_simulations=10000, rollout_exploration_param=0.8)
# unrestricted_mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=False, selection_exploration_param=0.2,
#                                num_simulations=15000, rollout_exploration_param=1)
# unrestricted_mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=False, selection_exploration_param=0.2,
#                                num_simulations=5000, rollout_exploration_param=1)
# unrestricted_mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=False, selection_exploration_param=0.3,
#                                num_simulations=20000, rollout_exploration_param=1)
# unrestricted_mcts = MCTSPlayer(heurisric=heuristic, depth=5, restrict_walls=False, selection_exploration_param=0.3,
#                                num_simulations=10000, rollout_exploration_param=1)
unrestricted_mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=False, selection_exploration_param=0.2,
                               num_simulations=20000, rollout_exploration_param=1)
mcts = MCTSPlayer(heurisric=heuristic, depth=10, restrict_walls=True,
                  num_simulations=10000, rollout_exploration_param=1)
deep_mcts = MCTSPlayer(heurisric=heuristic, depth=20, restrict_walls=True,
                       num_simulations=20000, rollout_exploration_param=1)

restricted_minimax = MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=5, restrict=True,
                                   check_bfs=True)
minimax3 = MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=3, restrict=False,
                         check_bfs=True)
minimax2 = MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=2, restrict=False,
                         check_bfs=True)
minimax1 = MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=1, restrict=False,
                         check_bfs=True)

heuristics = {
    # 'distance_to_end_heuristic': distance_to_end_heuristic,
    # 'wall_difference_heuristic': wall_difference_heuristic,
    'winning_heuristic': winning_heuristic,
    'blocking_heuristic': blocking_heuristic,
    # 'central_position_heuristic': central_position_heuristic,
    # 'escape_route_heuristic': escape_route_heuristic,
    # 'proximity_heuristic': proximity_heuristic,
    # 'edge_avoidance_heuristic': edge_avoidance_heuristic,
    # 'manhattan_dist': manhattan_dist,
    # 'combined_heuristic': combined_heuristic,
    # 'normalized_distance_to_end_heuristic': normalized_distance_to_end_heuristic
}


# Function to wrap competition to work with Pool
def run_competition(args):
    p1, p2, title = args
    competition(p1, p2, title=title)


def minimax_competition():
    tasks = []  # List to store all tasks to be run in parallel

    for i in range(1, 4):
        p1 = MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=i, restrict=False)
        for name, heuristic in heuristics.items():
            p2 = MinimaxPlayer(heuristic_evaluation=heuristic, depth=i, restrict=False)
            title = f'distance_to_end_heuristic vs {name} (depth {i})'
            # Append the arguments for each competition to the task list
            tasks.append((p1, p2, title))

    # Create a pool of 10 workers and distribute tasks among them
    with Pool(processes=10) as pool:
        pool.map(run_competition, tasks)  # Map tasks to the worker processes



def run_minimax_time_check(args):
    p1, p2, title = args
    games = 1
    times = []

    for i in tqdm(range(1, games + 1)):
        if random.random() < 0.5:
            p1, p2 = p2, p1

        game_state = GameState()
        while not game_state.is_game_over():
            current_player = p1 if game_state.p1_turn else p2

            t0 = time.time()
            move = current_player.get_next_move(state=game_state)
            times.append(time.time() - t0)

            game_state.apply_move(move=move)

    print("################")
    print(f"{games} Games: {title}, avg time: {sum(times) / len(times)}")
    print("################")


def minimax_time_check():
    tasks = []  # List to store all tasks to be run in parallel

    for i in range(3, 6):
        for name, heuristic in heuristics.items():
            p1 = p2 = MinimaxPlayer(heuristic_evaluation=heuristic, depth=i, restrict=False)
            title = f'distance_to_end_heuristic vs {name} (depth {i})'
            # Append the arguments for each competition to the task list
            run_minimax_time_check([p1, p2, title])


def competitions():
    games = 100
    # competition(mcts, minimax1, games, 'mcts vs minimax1')
    # competition(deep_mcts, minimax1, games, 'deep mcts vs minimax1')
    # competition(heuristic_mcts, minimax1, games, 'heuristic mcts vs minimax1')
    competition(unrestricted_mcts, minimax1, games, 'unrestricted mcts vs minimax1', verbose=True)

    competition(mcts, minimax2, games, 'mcts vs minimax2')
    competition(deep_mcts, minimax2, games, 'deep mcts vs minimax2')
    competition(heuristic_mcts, minimax2, games, 'heuristic mcts vs minimax2')
    competition(unrestricted_mcts, minimax2, games, 'unrestricted mcts vs minimax2')

    competition(mcts, minimax3, games, 'mcts vs minimax3')
    competition(deep_mcts, minimax3, games, 'deep mcts vs minimax3')
    competition(heuristic_mcts, minimax3, games, 'heuristic mcts vs minimax3')
    competition(unrestricted_mcts, minimax3, games, 'unrestricted mcts vs minimax3')


def rl_training():
    dqn_player = DeepQLearningPlayer(restrict=False)
    # train_agent(dqn_player=dqn_player, reward_heuristic=dqn_normalized_distance_to_end_heuristic,
    #             opponent_model=minimax2, verbose=False)
    plot_loss_graph()
    player_vs_ai(dqn_player)
    # display_ai_vs_ai(dqn_player, minimax2)

    # graphics = Graphics()
    # p1, p2 = HumanPlayer(graphics), dqn_player
    # game(p1=p2, p2=p1, graphics=graphics)

    visualize_minimax_game_q_values(dqn_player=dqn_player, reward_heuristic=dqn_normalized_distance_to_end_heuristic)
    # train_agent(dqn_player=dqn_player, opponent_model=minimax1, total_episodes=100000000,
    #             reward_heuristic=dqn_normalized_distance_to_end_heuristic)


def main():
    # competitions()
    minimax_time_check()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    # main()
    # competition(p1=MinimaxPlayer(heuristic_evaluation=distance_to_end_heuristic, depth=3, restrict=False),
    #             p2=MinimaxPlayer(heuristic_evaluation=manhattan_dist, depth=3, restrict=False), games=100, verbose=True)
    # run_minimax_time_check()
    # competitions()
    # player_vs_ai(minimax2)
    # minimax_competition()
    rl_training()
