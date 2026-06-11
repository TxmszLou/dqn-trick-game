import argparse
import json
import random

import torch

from .env import Card_Env
from .models import DQN
from .policies import greedy_policy, policy_legal_move, random_agent


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


def raw_policy_agent(net, game):
    with torch.no_grad():
        return net(game.get_network_input().to(next(net.parameters()).device)).argmax().item()


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
    return simulate_network(trained_network, num_games, verbose=verbose)


def simulate_network(net, num_games, verbose=False, legal_only=True):
    if legal_only:
        trained_policy = lambda game: policy_agent(net, game)
    else:
        trained_policy = lambda game: raw_policy_agent(net, game)
    return simulate(trained_policy, num_games, verbose)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate policies in the trick-taking environment.")
    parser.add_argument("--policy", choices=["random", "greedy", "network"], default="network")
    parser.add_argument("--weights", help="Path to DQN weights when --policy network is used.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--raw-network", action="store_true", help="Do not mask network actions to legal moves.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.policy == "random":
        policy = random_agent
    elif args.policy == "greedy":
        policy = greedy_policy
    else:
        if not args.weights:
            raise SystemExit("--weights is required when --policy network is used")
        net = load_dqn(args.weights)
        result = simulate_network(net, args.games, verbose=args.verbose, legal_only=not args.raw_network)
        print(json.dumps(result))
        return

    result = simulate(policy, args.games, verbose=args.verbose)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
