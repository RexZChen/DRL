import gym
import pybullet_envs
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import get_nn
import torch.optim as optim
from torch.nn import MSELoss


class DQNagent_d(nn.Module):

    def __init__(self, obv_dim, act_dim, replay_buffer, batch_size, gamma, lr):
        super(DQNagent_d, self).__init__()
        self.obv_dim = obv_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.q_net = get_nn(obv_dim=self.obv_dim, act_dim=self.act_dim)
        self.qt_net = get_nn(obv_dim=self.obv_dim, act_dim=self.act_dim)

        self.gamma = gamma
        self.lr = lr

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_func = MSELoss()

    def give_action(self, ep, env, state):
        if np.random.random() < ep:
            action = env.action_space.sample()

        else:
            state_vec = state.reshape(1, -1)  # (obv_dim,) -> (1, obv_dim)
            q_value = self.q_net(state_vec)
            _, action_vec = torch.max(q_value, dim=1)
            action = action_vec.item()

        return action, action

    def train(self, *args):
        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)  # sample()

        q_value = self.q_net(state)  # act_dim
        next_q_value = self.qt_net(next_state).max(1)[0]  # act_dim

        q_a_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)
        # print(q_a_value)

        target_q = reward + self.gamma * next_q_value * (1 - done)
        # print(target_q)
        loss = self.loss_func(q_a_value, target_q.detach())
        # print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), reward

    def update_qt_net(self):
        self.qt_net.load_state_dict(self.q_net.state_dict())