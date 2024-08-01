import torch

'''
Attributes
self.num_players: integer number of players
self.num_cards: integer number of cards in each hand
self.trump: integer trump suit (default 3)
self.current_turn: two integers, the first is the current player and the second is the turn number
self.current_player: integer index of the player whose turn it currently is
self.turn_counter: counts the number of turns which have occurred, defaultly 0 to 51
self.current_suit: a character denoting the suit of the current trick in play, or None if there is no current trick
self.hands: array of integer lists, one for each hand (default 4 x 52) where zeros denote voids
self.state: integer torch tensor, where the (i, j) entry denotes the card played by the i-th player on the j-th turn
self.tricks_won: torch integer tensor, where the i-th entry is the number of tricks won by the i-th player
self.winners_tracker: integer array (defaultly 52 x 4), tracks which player won which card


Methods:
self.play_card(index): plays the card of corresponding index from the current player's hand
self.get_suits(cards): returns the suit distribution of the cards
self.get_value(card): returns the relative value of the card
self.get_trick_winner(trick): returns the index of the card which won the trick
self.is_void(suit): returns True if the current player has no cards in the given suit
self.get_network_input(): returns one-dimensional array to feed into the network
self.get_legal_moves(): returns an array of indices (defaultly 0 to 51) of cards which may be played
'''
class Card_Game:

    def __init__(self, num_players=torch.tensor(4), num_cards=torch.tensor(13), trump=torch.tensor(3)):
        self.num_cards = num_cards
        self.num_players = num_players
        self.trump = trump
        self.current_player = torch.tensor(0, dtype=int)
        self.turn_counter = torch.tensor(0, dtype=int)
        self.current_suit = None
        # self.hands = []
        deck = torch.nn.functional.one_hot(torch.randperm(self.num_players * self.num_cards))
        # for player in range(self.num_players):
        #     self.hands.append(deck[num_cards * player: num_cards * (player+1)])
        self.hands = torch.unflatten(deck, 0, (self.num_players, self.num_cards)).sum(1)
        # self.state = torch.zeros(self.num_players, self.num_cards)
        self.state = torch.zeros(52, 52, dtype=int)
        self.tricks_won = torch.zeros(self.num_players, dtype=int)
        self.winners_tracker = torch.zeros(self.num_cards * self.num_players, self.num_players, dtype=int)

    '''
    Description
    Reset and start a new game
    '''
    def reset(self):
        self.__init__(self.num_players, self.num_cards, self.trump)

    '''
    Input
    index: the integer index of the card in the deck, defaultly from 0 to 51

    Description
    Replaces the card played with 0, changes the game state to reflect the played card, and changes the current turn
    '''
    def play_card(self, deck_index):
        # card = self.hands[self.current_player][index]
        assert self.hands[self.current_player][deck_index] != 0
        card = torch.nn.functional.one_hot(torch.tensor(deck_index, dtype=int), self.num_cards * self.num_players).flatten()

        if self.current_suit == None:
            self.current_suit = torch.argmax(self.get_suits(card))
        if self.current_suit != torch.argmax(self.get_suits(card)):
            assert self.is_void(self.current_suit)

        self.hands[self.current_player][deck_index] = 0
        self.state[self.turn_counter] = card


        if (self.turn_counter + 1) % self.num_players == 0:
            # print('Trick done')
            trick = self.state[self.turn_counter+1-self.num_players:self.turn_counter]
            winning_card = self.get_trick_winner(trick)
            trick_winner = (self.current_player + 1 + winning_card) % self.num_players
            self.tricks_won[trick_winner] += 1

            self.winners_tracker[self.turn_counter+1-self.num_players:self.turn_counter+1, trick_winner] = torch.tensor([1,1,1,1])

            self.current_player = trick_winner
            self.current_suit = None
        elif self.current_player == self.num_players-1:
            self.current_player = torch.tensor(0, dtype=int)
        else:
            self.current_player += 1
        self.turn_counter += 1



    '''
    Input
    sum_of_cards: 1-d tensor of zeros and ones

    returns: the count of cards in the sum, separated by suit
    '''
    def get_suits(self, sum_of_cards):
        # suits = ['C', 'D', 'H', 'S']
        by_suit = torch.unflatten(sum_of_cards, 0, (4, (self.num_players * self.num_cards / 4).int()))
        return by_suit.count_nonzero(1)


    '''
    Input
    card: 1d integer array with exactly one non-zero entry

    returns: the relative value of the specified card
    '''
    def get_value(self, card):
        torch.argmax(card) % self.num_cards


    '''
    Input
    trick: torch tensor of cards in the trick (generally 4 x 52)

    returns: the index of the card which won the trick
    '''
    def get_trick_winner(self, trick):
        trick_suit = self.get_suits(trick[0])
        trick_suit[3] = 13
        card_vals = torch.matmul(trick.unflatten(1, (4, 13)).argmax(2), trick_suit.reshape(4, 1))
        return torch.argmax(card_vals)

    '''
    Input
    suit: integer from 0 to 3

    returns: True if the current player is void of suit, else False
    '''
    def is_void(self, suit):
        cards_by_suit = self.get_suits(self.hands[self.current_player])
        if cards_by_suit[suit] == 0:
            return True
        return False


    '''
    returns: a one-dimensional array to feed into the neural network, defaultly of length 3016
    '''
    def get_network_input(self):
        flat_state = torch.concat((self.state, self.winners_tracker), dim=1).flatten()
        unseen_cards = torch.concat((self.hands[:self.current_player], self.hands[self.current_player+1:])).sum(0)
        return torch.concat((self.hands[self.current_player], flat_state, unseen_cards))
    
    '''
    returns: True if the deck_index is a legal card to play for the current player
    '''
    def is_move_legal(self, deck_index):
        legal_moves = self.get_legal_moves()
        return deck_index in legal_moves

    '''
    returns: a list of indices, corresponding to cards in the deck which may be legally played at this time
    '''
    def get_legal_moves(self):
        moves = self.hands[self.current_player]
        if self.current_suit == None:
            return moves.nonzero().flatten()

        by_suit = torch.unflatten(moves, 0, (4, (self.num_players * self.num_cards / 4).int()))
        if by_suit[self.current_suit].sum()==0:
            return moves.nonzero().flatten()
        new_moves = torch.zeros((4, (self.num_players * self.num_cards / 4).int()))
        new_moves[self.current_suit] = by_suit[self.current_suit]
        return new_moves.flatten().nonzero().flatten()

    '''
    returns a random legal move from the current player
    '''
    def sample_legal_move(self):
        moves = self.get_legal_moves()
        i = torch.randint(len(moves), (1,))
        chosen_move = moves[i]
        return chosen_move


# a random agent that choose a random legal move and plays it
# returns the chosen card to play : 0-51
def random_agent(game):
    moves = game.get_legal_moves()
    if len(moves) == 0:
        return
    i = torch.randint(len(moves), (1,))
    chosen_move = moves[i]
    # game.play_card(chosen_move)
    return chosen_move

# a card playing environment that maintains a game environment and is responsible for
# using some agent to get to the next state from the current state, and computes the reward
class Card_Env:
    def __init__(self, num_players=torch.tensor(4), num_cards=torch.tensor(13), trump=torch.tensor(3), foreign_policy=random_agent):
        self.game = Card_Game(num_players, num_cards, trump)
        # foreign_policy : game -> deck_index
        self.foreign_policy = foreign_policy
    
    def reset(self):
        self.game.reset()
        return self.game.get_network_input()
    
    def get_state(self):
        return self.game.get_network_input()

    # return observation, reward, terminated
    def step(self, deck_index):
        # step to the next state using the given foreign policy to play three turns in the game
        # TODO: if try to step through an illegal move, the game ends immediately
        #       may want to modify in the future

        current_player = self.game.current_player
        current_tricks_won = self.game.tricks_won[current_player]

        # first play the current move
        if not self.game.is_move_legal(deck_index):
            print('player plays an illegal move')
            return None, -10, True
        
        self.game.play_card(deck_index)

        # let the next three players play using the foreign_policy
        # for i in range(3):
        while self.game.current_player != current_player:
            move = self.foreign_policy(self.game)
            if not move:
                return None, 0, True    # TODO: Not sure about this, what to do if the game is over
            if not self.game.is_move_legal(move):
                return None, 0, True
            self.game.play_card(move)

        # reward is 1 if the player won this trick, 0 otherwise
        reward = 1 if self.game.tricks_won[current_player] > current_tricks_won else 0
        
        return self.game.get_network_input(), reward, False