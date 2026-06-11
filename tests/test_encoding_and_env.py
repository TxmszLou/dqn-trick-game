import torch

from trick_game.encoding import get_legal_moves
from trick_game.env import Card_Env
from trick_game.game import Card_Game


def test_encoded_legal_mask_matches_game_legal_moves_at_start():
    torch.manual_seed(1)
    game = Card_Game()

    legal_from_game = set(game.get_legal_moves().tolist())
    legal_from_encoding = set(get_legal_moves(game.get_network_input()).nonzero().flatten().tolist())

    assert legal_from_encoding == legal_from_game


def test_encoded_legal_mask_matches_game_legal_moves_after_lead_card():
    game = Card_Game()
    game.hands = torch.zeros(4, 52, dtype=torch.int64)
    game.hands[0, [0, 5, 13]] = 1
    game.hands[1, [2, 14, 40]] = 1
    game.hands[2, [3, 15, 41]] = 1
    game.hands[3, [4, 16, 42]] = 1
    game.current_player = torch.tensor(0, dtype=int)

    game.play_card(0)

    legal_from_game = set(game.get_legal_moves().tolist())
    legal_from_encoding = set(get_legal_moves(game.get_network_input()).nonzero().flatten().tolist())

    assert game.current_player.item() == 1
    assert legal_from_encoding == legal_from_game == {2}


def test_env_step_returns_rl_tuple_for_legal_move():
    torch.manual_seed(2)
    env = Card_Env()
    move = env.game.sample_legal_move().item()

    observation, reward, terminated = env.step(move)

    assert observation is None or observation.shape == (3016,)
    assert reward in (0, 1)
    assert isinstance(terminated, bool)
