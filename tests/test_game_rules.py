import torch

from trick_game.game import Card_Game


def _game_with_hand(cards, current_suit=None):
    game = Card_Game()
    game.hands = torch.zeros(4, 52, dtype=torch.int64)
    game.hands[0, cards] = 1
    game.current_player = torch.tensor(0, dtype=int)
    game.current_suit = current_suit
    return game


def _trick(cards):
    return torch.stack([torch.nn.functional.one_hot(torch.tensor(card), 52) for card in cards])


def test_deal_has_expected_shape_and_unique_card_ownership():
    torch.manual_seed(0)
    game = Card_Game()

    assert game.hands.shape == (4, 52)
    assert game.hands.sum().item() == 52
    assert torch.all(game.hands.sum(dim=0) == 1)
    assert torch.all(game.hands.sum(dim=1) == 13)
    assert game.get_network_input().shape == (3016,)


def test_legal_moves_when_leading_are_all_cards_in_hand():
    game = _game_with_hand([0, 5, 13, 39], current_suit=None)

    assert set(game.get_legal_moves().tolist()) == {0, 5, 13, 39}


def test_legal_moves_follow_current_suit_when_available():
    game = _game_with_hand([0, 5, 13, 39], current_suit=torch.tensor(0))

    assert set(game.get_legal_moves().tolist()) == {0, 5}


def test_legal_moves_allow_any_card_when_void_in_current_suit():
    game = _game_with_hand([13, 20, 39], current_suit=torch.tensor(0))

    assert set(game.get_legal_moves().tolist()) == {13, 20, 39}


def test_trick_winner_uses_lead_suit_when_no_spades_are_played():
    game = Card_Game()

    assert game.get_trick_winner(_trick([0, 5, 13, 14])).item() == 1


def test_trick_winner_prefers_highest_spade():
    game = Card_Game()

    assert game.get_trick_winner(_trick([0, 5, 39, 51])).item() == 3
