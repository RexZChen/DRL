import gym
import pybullet_envs
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import get_nn, EnvNet
import torch.optim as optim
from torch.nn import MSELoss


class DynaQagent_d(nn.Module):

    def __init__(self, obv_dim, act_dim, replay_buffer, batch_size, gamma, q_lr, env_lr):
        super(DynaQagent_d, self).__init__()
        self.obv_dim = obv_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        self.gamma = gamma
        self.q_lr = q_lr
        self.env_lr = env_lr

        self.q_net = get_nn(self.obv_dim, self.act_dim)
        self.qt_net = get_nn(self.obv_dim, self.act_dim)
        self.env_net = EnvNet(self.obv_dim, 1)

        self.q_opim = optim.Adam(self.q_net.parameters(), lr=self.q_lr)
        self.env_optim = optim.Adam(self.env_net.parameters(), lr=self.env_lr)

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
        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)
        next_state = torch.tensor(next_state, dtype=torch.float)

        q_value = self.q_net(state)  # act_dim
        next_q_value = self.qt_net(next_state).max(1)[0]  # act_dim

        q_a_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)
        # print(q_a_value)

        target_q = reward + self.gamma * next_q_value * (1 - done)
        # print(target_q)
        loss = self.loss_func(q_a_value, target_q.detach())
        # print(loss)

        self.q_opim.zero_grad()
        loss.backward()
        self.q_opim.step()

        return loss.item(), reward

    def update_env(self):
        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)
        # print(torch.unsqueeze(action, 1).shape)
        # print(state.shape)
        next_state_predicted, reward_predicted, done_predicted = self.env_net(state, torch.unsqueeze(action, 1))

        state_net_loss = self.loss_func(next_state_predicted, next_state)
        reward_net_loss = self.loss_func(reward_predicted, reward.squeeze())
        done_net_loss = self.loss_func(done_predicted, done.squeeze())

        self.env_optim.zero_grad()
        state_net_loss.backward()
        reward_net_loss.backward()
        done_net_loss.backward()
        self.env_optim.step()

        return state_net_loss, reward_net_loss, done_net_loss

    def exploration(self):
        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)
        random_act = torch.tensor(np.random.randint(self.act_dim, size=(self.batch_size, 1)))

        next_state_predicted, reward_predicted, done_predicted = self.env_net(state, random_act)

        q_value = self.q_net(state)
        next_q_value = self.qt_net(next_state_predicted).max(1)[0]

        q_a_value = q_value.gather(1, random_act).squeeze(1)
        target_q = torch.squeeze(reward_predicted) + self.gamma * next_q_value * (1 - torch.squeeze(done_predicted))

        loss = self.loss_func(q_a_value, target_q.detach())

        self.q_opim.zero_grad()
        loss.backward()
        self.q_opim.step()

        return loss

    def update_qt_net(self):
        self.qt_net.load_state_dict(self.q_net.state_dict())
