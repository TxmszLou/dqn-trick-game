import torch
import torch.optim as optim

from trick_game.dqn import optimize_model
from trick_game.models import DQN
from trick_game.replay import ReplayMemory
from trick_game.train_dqn import parse_args as parse_train_args
from trick_game.evaluate import parse_args as parse_eval_args


def test_train_cli_parser_accepts_legal_training_args():
    args = parse_train_args(
        [
            "--episodes",
            "5",
            "--legal-only",
            "--output",
            "weights/test.pth",
            "--eval-every",
            "0",
        ]
    )

    assert args.episodes == 5
    assert args.legal_only is True
    assert args.output == "weights/test.pth"
    assert args.eval_every == 0


def test_evaluate_cli_parser_accepts_network_args():
    args = parse_eval_args(["--policy", "network", "--weights", "weights/ev_q_function_output.pth", "--games", "3"])

    assert args.policy == "network"
    assert args.weights == "weights/ev_q_function_output.pth"
    assert args.games == 3


def test_optimize_model_handles_all_terminal_transition_batch():
    device = torch.device("cpu")
    policy_net = DQN(3016, 52)
    target_net = DQN(3016, 52)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4)
    memory = {0: ReplayMemory(10)}

    for _ in range(2):
        state = torch.zeros(1, 3016)
        action = torch.tensor([[0]], dtype=torch.long)
        reward = torch.tensor([1.0])
        memory[0].push(state, action, None, reward)

    loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=2, gamma=0.5)

    assert loss is not None
