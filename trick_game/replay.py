import random
from collections import deque, namedtuple


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class StratifiedReplayMemory(dict):
    def __init__(self, num_buckets=13, capacity=10000):
        super().__init__((turn, ReplayMemory(capacity)) for turn in range(num_buckets))

    def sample_ready(self, batch_size):
        transitions = []
        for mem in self.values():
            if len(mem) >= batch_size:
                transitions += mem.sample(batch_size)
        return transitions
