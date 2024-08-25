import torch
from card_engine import Card_Game

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
    
    suit = highest_values[game.current_suit]
    return suit * game.num_cards + highest_values[suit]