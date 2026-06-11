import argparse
import json
import random
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import torch
import torch.optim as optim

from .dqn import optimize_model, select_action, soft_update_target
from .env import Card_Env
from .evaluate import simulate_network
from .models import DQN
from .policies import net_policy
from .replay import StratifiedReplayMemory


@dataclass
class DQNTrainingConfig:
    episodes: int = 10_000
    legal_only: bool = False
    opponent_weights: str | None = None
    output: str = "weights/dqn-final.pth"
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    eval_every: int = 1_000
    eval_games: int = 100
    batch_size: int = 100
    memory_size: int = 10_000
    lr: float = 1e-4
    tau: float = 0.005
    gamma: float = 0.5
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: int = 5_000
    seed: int | None = None


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(config, n_input, n_actions, device):
    if config.opponent_weights is None:
        return Card_Env()

    foreign_net = DQN(n_input, n_actions).to(device)
    if device == torch.device("cpu"):
        foreign_net.load_state_dict(torch.load(config.opponent_weights, map_location=torch.device("cpu")))
    else:
        foreign_net.load_state_dict(torch.load(config.opponent_weights))
    foreign_net.eval()
    return Card_Env(foreign_policy=lambda game: net_policy(foreign_net, game))


def train_dqn(config):
    seed_everything(config.seed)
    device = get_device()

    probe_env = Card_Env()
    n_input = len(probe_env.game.get_network_input())
    n_actions = 52

    policy_net = DQN(n_input, n_actions).to(device)
    target_net = DQN(n_input, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=config.lr, amsgrad=True)
    memory = StratifiedReplayMemory(num_buckets=13, capacity=config.memory_size)

    env = build_env(config, n_input, n_actions, device)
    steps_done = 0
    rewards_window = []
    episode_durations = []
    episode_rewards = []
    evaluations = []

    for i_episode in range(1, config.episodes + 1):
        env.reset()

        player_ind = random.randint(0, 3)
        while env.game.current_player != player_ind:
            move = env.game.sample_legal_move()
            env.game.play_card(move)

        current_reward = 0
        state = env.game.get_network_input().to(dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action, steps_done, _ = select_action(
                env.game,
                policy_net,
                steps_done,
                device,
                eps_start=config.eps_start,
                eps_end=config.eps_end,
                eps_decay=config.eps_decay,
                legal_only=config.legal_only,
            )

            observation, reward_value, terminated = env.step(action.item())
            rewards_window.append(reward_value)
            reward = torch.tensor([reward_value], device=device)
            current_reward += reward_value

            if not terminated:
                next_state = observation.to(dtype=torch.float32, device=device).unsqueeze(0)
            else:
                next_state = None

            memory[t].push(state, action, next_state, reward)
            state = next_state

            optimize_model(
                memory,
                policy_net,
                target_net,
                optimizer,
                device,
                batch_size=config.batch_size,
                gamma=config.gamma,
            )
            soft_update_target(policy_net, target_net, tau=config.tau)

            if terminated:
                episode_durations.append(t + 1)
                episode_rewards.append(current_reward)
                break

        if config.checkpoint_dir and config.checkpoint_every and i_episode % config.checkpoint_every == 0:
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy_net.state_dict(), checkpoint_dir / f"dqn-{i_episode}.pth")

        if config.eval_every and i_episode % config.eval_every == 0:
            avg_reward = sum(rewards_window) / len(rewards_window) if rewards_window else 0.0
            eval_result = simulate_network(policy_net, config.eval_games, legal_only=True)
            evaluations.append(
                {
                    "episode": i_episode,
                    "average_reward_per_move": avg_reward,
                    "evaluation": eval_result,
                    "memory_bank": [len(mem) for mem in memory.values()],
                }
            )
            print(json.dumps(evaluations[-1]))
            rewards_window = []

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_net.state_dict(), output)

    return {
        "config": asdict(config),
        "output": str(output),
        "episode_durations": episode_durations,
        "episode_rewards": episode_rewards,
        "evaluations": evaluations,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a DQN policy for the trick-taking environment.")
    parser.add_argument("--episodes", type=int, default=DQNTrainingConfig.episodes)
    parser.add_argument("--legal-only", action="store_true")
    parser.add_argument("--opponent-weights")
    parser.add_argument("--output", default=DQNTrainingConfig.output)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=DQNTrainingConfig.eval_every)
    parser.add_argument("--eval-games", type=int, default=DQNTrainingConfig.eval_games)
    parser.add_argument("--batch-size", type=int, default=DQNTrainingConfig.batch_size)
    parser.add_argument("--memory-size", type=int, default=DQNTrainingConfig.memory_size)
    parser.add_argument("--lr", type=float, default=DQNTrainingConfig.lr)
    parser.add_argument("--tau", type=float, default=DQNTrainingConfig.tau)
    parser.add_argument("--gamma", type=float, default=DQNTrainingConfig.gamma)
    parser.add_argument("--eps-start", type=float, default=DQNTrainingConfig.eps_start)
    parser.add_argument("--eps-end", type=float, default=DQNTrainingConfig.eps_end)
    parser.add_argument("--eps-decay", type=int, default=DQNTrainingConfig.eps_decay)
    parser.add_argument("--seed", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = DQNTrainingConfig(**vars(args))
    result = train_dqn(config)
    print(json.dumps({"output": result["output"], "episodes": config.episodes}))


if __name__ == "__main__":
    main()
