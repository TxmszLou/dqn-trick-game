from pathlib import Path

import torch

from trick_game.models import DQN
from utils.card_engine import Card_Env, Card_Game, get_legal_moves, greedy_policy, policy_legal_move, random_agent


def test_legacy_utils_card_engine_exports_still_import():
    assert Card_Game is not None
    assert Card_Env is not None
    assert callable(random_agent)
    assert callable(greedy_policy)
    assert callable(get_legal_moves)
    assert callable(policy_legal_move)


def test_dqn_forward_shape_and_layer_names_match_saved_weights():
    model = DQN(3016, 52)
    output = model(torch.zeros(3016))

    assert output.shape == (52,)
    assert {"layer1.weight", "layer2.weight", "layer3.weight"}.issubset(model.state_dict())


def test_existing_dqn_weight_file_loads_into_extracted_model():
    weights = Path("weights/ev_q_function_output.pth")
    assert weights.exists()

    model = DQN(3016, 52)
    state_dict = torch.load(weights, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    assert model(torch.zeros(3016)).shape == (52,)
