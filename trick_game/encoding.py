import torch


def unpack(input):
    hand = input[0:52]
    state_winner = torch.unflatten(input[52:52 * 56 + 52], 0, (52, 56))
    state = state_winner[:, 0:52]
    winner_tracker = state_winner[:, 52:]
    return hand, state, winner_tracker


def get_last_turn(input):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    _, state, _ = unpack(input)

    turn_num = 0
    while turn_num < 52 and (not torch.equal(state[turn_num, :].to(device), torch.zeros(52).to(device))):
        turn_num += 1

    return turn_num - 1


def get_current_suit(input):
    _, state, _ = unpack(input)
    last_turn = get_last_turn(input)

    if (last_turn + 1) % 4 == 0:
        return None

    assert 0 <= last_turn and last_turn <= 51
    card = state[last_turn - (last_turn % 4)]
    by_suit = torch.unflatten(card, 0, (4, 13)).count_nonzero(1)

    return torch.argmax(by_suit)


def get_legal_moves(input):
    hand, state, winner_tracker = unpack(input)

    current_suit = get_current_suit(input)
    if current_suit == None:
        card_idx = hand.nonzero().flatten()
    else:
        by_suit = torch.unflatten(hand, 0, (4, 13))
        if by_suit[current_suit].sum() == 0:
            card_idx = hand.nonzero().flatten()
        else:
            new_moves = torch.zeros((4, 13))
            new_moves[current_suit] = by_suit[current_suit]
            card_idx = new_moves.flatten().nonzero().flatten()

    moves = torch.zeros(52)
    moves[card_idx] = 1
    return moves
