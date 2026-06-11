import random

import torch

from .env import Card_Env
from .models import DQN
from .policies import policy_legal_move


def simulate_game_random(policy, verbose=False, from_move=0):
    moves_played = 0
    cumulative_reward = 0

    active_player = from_move % 4
    if verbose:
        print(f"Starting new game as player {active_player} from turn {from_move}.")

    env = Card_Env()

    while env.game.current_player != active_player:
        move = env.game.sample_legal_move()
        env.game.play_card(move)

    for _ in range(13):
        move = policy(env.game)
        if not (move in env.game.get_legal_moves()):
            if verbose:
                print(f"Tried to play illegal move {move}")
            return moves_played, cumulative_reward

        _, reward, terminated = env.step(move)

        moves_played += 1
        cumulative_reward += reward

        if terminated:
            return moves_played, cumulative_reward

    return moves_played, cumulative_reward


def simulate(policy, num_games, verbose=False):
    durations = []
    rewards = []
    simul_dist = [0 for _ in range(14)]

    for _ in range(num_games):
        moves_played, cumulative_reward = simulate_game_random(policy, verbose, from_move=random.randint(0, 3))
        durations.append(moves_played)
        rewards.append(cumulative_reward)
        simul_dist[moves_played] += 1

    return {
        "durations": durations,
        "rewards": rewards,
        "average_duration": sum(durations) / num_games,
        "average_reward": sum(rewards) / num_games,
        "duration_distribution": simul_dist,
    }


def policy_agent(net, game):
    with torch.no_grad():
        return policy_legal_move(net, game.get_network_input()).item()


def load_dqn(weights, n_input=3016, n_output=52, device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    trained_network = DQN(n_input, n_output).to(device)
    if device == torch.device("cpu"):
        trained_network.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
    else:
        trained_network.load_state_dict(torch.load(weights))
    return trained_network


def simulate_with_network(weights, num_games, verbose=False, device=None):
    trained_network = load_dqn(weights, device=device)
    trained_policy = lambda game: policy_agent(trained_network, game)
    return simulate(trained_policy, num_games, verbose)
