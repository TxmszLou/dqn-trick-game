import torch


class Card_Game:
    def __init__(self, num_players=torch.tensor(4), num_cards=torch.tensor(13), trump=torch.tensor(3)):
        self.num_cards = num_cards
        self.num_players = num_players
        self.trump = trump
        self.current_player = torch.tensor(0, dtype=int)
        self.turn_counter = torch.tensor(0, dtype=int)
        self.current_suit = None
        deck = torch.nn.functional.one_hot(torch.randperm(self.num_players * self.num_cards))
        self.hands = torch.unflatten(deck, 0, (self.num_players, self.num_cards)).sum(1)
        self.state = torch.zeros(52, 52, dtype=int)
        self.tricks_won = torch.zeros(self.num_players, dtype=int)
        self.winners_tracker = torch.zeros(self.num_cards * self.num_players, self.num_players, dtype=int)

    def reset(self):
        self.__init__(self.num_players, self.num_cards, self.trump)

    def play_card(self, deck_index):
        deck_index = int(deck_index)
        assert self.hands[self.current_player][deck_index] != 0
        card = torch.nn.functional.one_hot(
            torch.tensor(deck_index, dtype=int),
            self.num_cards * self.num_players,
        ).flatten()

        if self.current_suit == None:
            self.current_suit = torch.argmax(self.get_suits(card))
        if self.current_suit != torch.argmax(self.get_suits(card)):
            assert self.is_void(self.current_suit)

        self.hands[self.current_player][deck_index] = 0
        self.state[self.turn_counter] = card

        if (self.turn_counter + 1) % self.num_players == 0:
            trick = self.state[self.turn_counter + 1 - self.num_players:self.turn_counter]
            winning_card = self.get_trick_winner(trick)
            trick_winner = (self.current_player + 1 + winning_card) % self.num_players
            self.tricks_won[trick_winner] += 1

            self.winners_tracker[
                self.turn_counter + 1 - self.num_players:self.turn_counter + 1,
                trick_winner,
            ] = torch.tensor([1, 1, 1, 1])

            self.current_player = trick_winner
            self.current_suit = None
        elif self.current_player == self.num_players - 1:
            self.current_player = torch.tensor(0, dtype=int)
        else:
            self.current_player += 1
        self.turn_counter += 1

    def get_suits(self, sum_of_cards):
        by_suit = torch.unflatten(
            sum_of_cards,
            0,
            (4, (self.num_players * self.num_cards / 4).int()),
        )
        return by_suit.count_nonzero(1)

    def get_value(self, card):
        torch.argmax(card) % self.num_cards

    def get_trick_winner(self, trick):
        trick_suit = self.get_suits(trick[0])
        trick_suit[3] = 13
        card_vals = torch.matmul(trick.unflatten(1, (4, 13)).argmax(2), trick_suit.reshape(4, 1))
        return torch.argmax(card_vals)

    def is_void(self, suit):
        cards_by_suit = self.get_suits(self.hands[self.current_player])
        if cards_by_suit[suit] == 0:
            return True
        return False

    def get_network_input(self):
        flat_state = torch.concat((self.state, self.winners_tracker), dim=1).flatten()
        unseen_cards = torch.concat((self.hands[:self.current_player], self.hands[self.current_player + 1:])).sum(0)
        return torch.concat((self.hands[self.current_player], flat_state, unseen_cards))

    def is_move_legal(self, deck_index):
        legal_moves = self.get_legal_moves()
        return deck_index in legal_moves

    def get_legal_moves(self):
        moves = self.hands[self.current_player]
        if self.current_suit == None:
            return moves.nonzero().flatten()

        by_suit = torch.unflatten(moves, 0, (4, (self.num_players * self.num_cards / 4).int()))
        if by_suit[self.current_suit].sum() == 0:
            return moves.nonzero().flatten()
        new_moves = torch.zeros((4, (self.num_players * self.num_cards / 4).int()))
        new_moves[self.current_suit] = by_suit[self.current_suit]
        return new_moves.flatten().nonzero().flatten()

    def get_highest_value_card(self):
        result = torch.ones(4, dtype=torch.int) * (-1)
        moves = self.hands[self.current_player]
        by_suit = torch.unflatten(moves, 0, (4, self.num_players * self.num_cards // 4))

        for i in range(4):
            cards_in_suit = (by_suit[i] == 1).nonzero()
            if len(cards_in_suit) != 0:
                result[i] = cards_in_suit.max()

        return result

    def sample_legal_move(self):
        moves = self.get_legal_moves()
        i = torch.randint(len(moves), (1,))
        chosen_move = moves[i]
        return chosen_move
