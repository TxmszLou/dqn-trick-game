SUITS = ["C", "D", "H", "S"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NUM_SUITS = 4
CARDS_PER_SUIT = 13
NUM_CARDS = NUM_SUITS * CARDS_PER_SUIT


def card_suit(card_index):
    return int(card_index) // CARDS_PER_SUIT


def card_rank(card_index):
    return int(card_index) % CARDS_PER_SUIT


def move_to_card(move):
    assert 0 <= move and move <= 52
    return f"{SUITS[int(move) // CARDS_PER_SUIT]} {int(move) % CARDS_PER_SUIT}"
