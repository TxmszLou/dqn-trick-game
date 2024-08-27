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
        #print('moves are', moves)
        if self.current_suit == None:
            #print('none is the suit. will return', moves.nonzero().flatten())
            return moves.nonzero().flatten()

        by_suit = torch.unflatten(moves, 0, (4, (self.num_players * self.num_cards / 4).int()))
        if by_suit[self.current_suit].sum()==0:
            #print('no card in suit. will return', moves.nonzero().flatten())
            return moves.nonzero().flatten()
        new_moves = torch.zeros((4, (self.num_players * self.num_cards / 4).int()))
        new_moves[self.current_suit] = by_suit[self.current_suit]
        #print('have cards in suit. will return', new_moves.flatten().nonzero().flatten())
        return new_moves.flatten().nonzero().flatten()
    
    '''
    returns a torch tensor of shape(4) each entry is the card with highest value [-1, 12] in that suit
    -1 denotes there is no card with that suit
    '''
    def get_highest_value_card(self):
        result = torch.ones(4, dtype=torch.int) * (-1)
        moves = self.hands[self.current_player]
        by_suit = torch.unflatten(moves, 0, (4, self.num_players * self.num_cards // 4))

        for i in range(4):
            cards_in_suit = (by_suit[i] == 1).nonzero()
            if len(cards_in_suit) != 0:
                result[i] = cards_in_suit.max()

        return result


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
# the argument fp is not used, but is necessary to interchange random_agent and policy_agent in Card_Env
def random_agent(game):
    moves = game.get_legal_moves()
    if len(moves) == 0:
        return
    i = torch.randint(len(moves), (1,))
    chosen_move = moves[i]
    # game.play_card(chosen_move)
    return chosen_move

'''
print the card [0-52] in human readable format: suit-card
'''
def move_to_card(move):
    assert 0 <= move and move <= 52

    suits = ['C', 'D', 'H', 'S']

    return f'{suits[move // 13]} {move % 13}'



'''
One turn greedy policy:
Choose the card with highest chance of wining this trick.
If current suit is none or no card with current suit,
- if there is a Spade, play a Spade with highest value
- othrewise play the card with highest value among all suits.
If there are cards with current suit, play the one with highest value.

Returns None if no legal moves
'''
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

# def policy_agent(game, fp):
#     with torch.no_grad():
#         q_values = fp(game.get_network_input().to(torch.float32).to(evice))
#     move = torch.argmax(q_values).item()
#     if move in game.get_legal_moves():
#         return move
#     return game.sample_legal_move()


def unpack(input):
    hand = input[0:52]

    state_winner = torch.unflatten(input[52:52*56 + 52], 0, (52, 56))

    state = state_winner[:, 0:52]
    winner_tracker = state_winner[:, 52:]

    return hand, state, winner_tracker

'''
returns: the number of last played turn at this point [-1, 51]
returns -1 if no card has been played yet
'''
def get_last_turn(input):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    _, state, _ = unpack(input)

    turn_num = 0

    while turn_num < 52 and (not torch.equal(state[turn_num, :].to(device), torch.zeros(52).to(device))):
        turn_num += 1
    
    return turn_num - 1


'''
return the current suit of the state 0-3 corresponding to suits ['C', 'D', 'H', 'S']
returns None if no leading suit
'''
def get_current_suit(input):
    _, state, _ = unpack(input)
    last_turn = get_last_turn(input)

    if (last_turn + 1) % 4 == 0:
        # Trick done, suit should be none
        return None

    assert 0 <= last_turn  and last_turn <= 51
    card = state[last_turn - (last_turn % 4)]
    by_suit = torch.unflatten(card, 0, (4, 13)).count_nonzero(1)
    
    return torch.argmax(by_suit)


'''
state: the current state
(1x52)    +  (56x52)       +       (1x52): the current state
^hand       ^who plays each card  ^cards not seen yet
                    + cards played
returns one-hot coded tensor of legal moves from the given state
'''
def get_legal_moves(input):
    hand, state, winner_tracker = unpack(input)

    current_suit = get_current_suit(input)
    if current_suit == None:
        card_idx = hand.nonzero().flatten()
    else:
        by_suit = torch.unflatten(hand, 0, (4, 13))
        if by_suit[current_suit].sum() == 0:
            # no card in suit
            card_idx = hand.nonzero().flatten()
        else:
            new_moves = torch.zeros((4, 13))
            new_moves[current_suit] = by_suit[current_suit]
            card_idx = new_moves.flatten().nonzero().flatten()

    moves = torch.zeros(52)
    moves[card_idx] = 1
    return moves

def policy_legal_move(net, input):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    legal_mask = get_legal_moves(input).to(device)
    with torch.no_grad():
        x = (net(input.to(device))) * legal_mask
        x[x == 0] = -float('inf')
        return x.max(0).indices.view(1,1)

def net_policy(net, game):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    return policy_legal_move(net, game.get_network_input().to(device))
    

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

        current_player = torch.clone(self.game.current_player)
        current_tricks_won = torch.clone(self.game.tricks_won[current_player])

        # first play the current move
        if not self.game.is_move_legal(deck_index):
            # should not happen
            return None, 0, True

        
        self.game.play_card(deck_index)

        # let the next three players play using the foreign_policy
        # for i in range(3):
        while True:
            if self.game.current_player == current_player:
                break

            # if self.foreign_policy != random_agent:
            #     legal_mask = get_legal_moves(input).to(device)
            #     with torch.no_grad():
            #         x = (fp(input.to(device))) * legal_mask
            #         x[x == 0] = -float('inf')
            #         move = x.max(0).indices.view(1,1)
            # else:
            #     move = self.foreign_policy(self.game, fp)
            
            if len(self.game.get_legal_moves()) == 0:
                return None, 0, True    # TODO: Not sure about this, what to do if the game is over

            move = self.foreign_policy(self.game).item()

            if not self.game.is_move_legal(move):
                move = self.game.sample_legal_move()
            self.game.play_card(move)

        # reward is 1 if the player won this trick, 0 otherwise
        reward = 1 if self.game.tricks_won[current_player] > current_tricks_won else 0
        # reward = 1
        
        return self.game.get_network_input(), reward, len(self.game.hands[current_player].nonzero()) == 0