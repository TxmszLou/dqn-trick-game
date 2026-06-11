import torch

from .game import Card_Game
from .policies import random_agent


class Card_Env:
    def __init__(
        self,
        num_players=torch.tensor(4),
        num_cards=torch.tensor(13),
        trump=torch.tensor(3),
        foreign_policy=random_agent,
    ):
        self.game = Card_Game(num_players, num_cards, trump)
        self.foreign_policy = foreign_policy

    def reset(self):
        self.game.reset()
        return self.game.get_network_input()

    def get_state(self):
        return self.game.get_network_input()

    def step(self, deck_index):
        current_player = torch.clone(self.game.current_player)
        current_tricks_won = torch.clone(self.game.tricks_won[current_player])

        if not self.game.is_move_legal(deck_index):
            return None, 0, True

        self.game.play_card(deck_index)

        while True:
            if self.game.current_player == current_player:
                break

            if len(self.game.get_legal_moves()) == 0:
                return None, 0, True

            move = self.foreign_policy(self.game).item()

            if not self.game.is_move_legal(move):
                move = self.game.sample_legal_move()
            self.game.play_card(move)

        reward = 1 if self.game.tricks_won[current_player] > current_tricks_won else 0

        return self.game.get_network_input(), reward, len(self.game.hands[current_player].nonzero()) == 0
