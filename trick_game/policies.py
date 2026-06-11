import torch

from .cards import move_to_card
from .encoding import get_legal_moves


def random_agent(game):
    moves = game.get_legal_moves()
    if len(moves) == 0:
        return
    i = torch.randint(len(moves), (1,))
    chosen_move = moves[i]
    return chosen_move


def greedy_policy(game):
    if len(game.hands[game.current_player]) == 0:
        return None

    highest_values = game.get_highest_value_card()

    if game.current_suit == None or highest_values[game.current_suit] == -1:
        if highest_values[3] != -1:
            suit = 3
        else:
            suit = highest_values.argmax()
        return suit * game.num_cards + highest_values[suit]

    return game.current_suit * game.num_cards + highest_values[game.current_suit]


def policy_legal_move(net, input):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    legal_mask = get_legal_moves(input).to(device)
    with torch.no_grad():
        x = (net(input.to(device))) * legal_mask
        x[x == 0] = -float("inf")
        return x.max(0).indices.view(1, 1)


def net_policy(net, game):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return policy_legal_move(net, game.get_network_input().to(device))


__all__ = ["greedy_policy", "move_to_card", "net_policy", "policy_legal_move", "random_agent"]
