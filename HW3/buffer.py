import numpy as np
import collections
import random
import torch


class OffP_Buffer:
    # reference: https://github.com/Lazydok/RL-Pytorch-cartpole/blob/master/2_double_dqn.py
    # reference: https://github.com/Kaixhin/ACER/blob/master/memory.py
    # def __init__(self, capacity):
    #     self.capacity = capacity
    #     self.memory = []
    #
    # def push(self, transition):
    #     self.memory.append(transition)
    #     if len(self.memory) > self.capacity:
    #         del self.memory[0]
    #
    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)
    #
    # def __len__(self):
    #     return len(self.memory)
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch):
        states, actions, rewards, dones, next_states = zip(*random.sample(self.buffer, batch))
        # print(states.shape())
        # print(actions.shape())
        # print(rewards.shape())
        # print(dones.shape())
        # print(next_states.shape())
        return torch.tensor(np.array(states), dtype=torch.float), \
               torch.tensor(np.array(actions), dtype=torch.long), \
               torch.tensor(np.array(rewards), dtype=torch.float), \
               torch.tensor(np.array(dones), dtype=torch.float), \
               torch.tensor(np.array(next_states), dtype=torch.float)

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)
