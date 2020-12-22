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
import random


def get_act_num(action_dim, bin_num=10):
    """continuous to discrete"""
    act_out_num = np.power(bin_num, action_dim)
    return act_out_num  # (act_out_num)


def quantization(pred_act, pred_act_high, action_low, action_high, action_dim, bin_num=10):
    """
    这个是预测的num给convert回continuous action
    pred_act: scalar, like: 90
    pred_act_high: 能预测到的最高discrete num
    action_low: array-like, like: [-1, -1, -1] (这个数字可能每一维不一样)
    action_high: array_like, like: [1, 1, 1]
    """
    # 转成list of 每一位上的数: 90 => [9, 0] (长度是3是因为这个case我们最高action可以取到999(0 - 10^3))
    act_bin_list = []
    # print(pred_act)
    while int(pred_act / bin_num) >= 1:
        act_bin_list.append(pred_act % bin_num)
        pred_act = pred_act // bin_num
    act_bin_list.reverse()

    # 要把不够位数的补全: [9, 0] => [0, 9, 0] (长度是3是因为这个case我们最高action可以取到999(0 - 10^3))
    for _ in range(action_dim - len(act_bin_list)):
        act_bin_list.insert(0, 0)
    # print(act_bin_list)
    act_out = np.zeros(action_dim, dtype=np.float)

    # 还原成continuous action space => 每一位对应一维
    # 先construct对应map: range若是0-1, bin是10的话 => [0-0.1, 0.1-0.2, ..., 0.9-1.0]
    for i in range(action_dim):
        act_low = action_low[i]  # -1
        act_high = action_high[i]  # 1
        diff = act_high - act_low  # 2
        add_term = diff / bin_num  # 2 / 10 = 0.2
        current_low = act_low
        act_tuple_list = []
        for j in range(bin_num):  # [..., (0.0, 0.2), (0.2, 0.4), ..., (0.8, 1.0)]
            current_high = current_low + add_term
            act_tuple_list.append((current_low, current_high))
            current_low = current_high
        index = act_bin_list[i]
        # print(index)
        # print(act_bin_list)
        pred_high, pred_low = act_tuple_list[index]  # (0.2, 0.4)
        act_out[i] = np.random.uniform(pred_low, pred_high)

    return act_out


class DynaQagent_c(nn.Module):

    def __init__(self, obv_dim, act_dim, replay_buffer, batch_size, gamma, q_lr, env_lr, env, act_high, act_low):
        super(DynaQagent_c, self).__init__()
        self.obv_dim = obv_dim
        self.act_dim = act_dim
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.act_high = act_high
        self.act_low = act_low
        self.env = env

        self.gamma = gamma
        self.q_lr = q_lr
        self.env_lr = env_lr

        self.q_net = get_nn(obv_dim=self.obv_dim, act_dim=get_act_num(self.act_dim))
        self.qt_net = get_nn(obv_dim=self.obv_dim, act_dim=get_act_num(self.act_dim))

        self.env_net = EnvNet(self.obv_dim, act_dim=1)
        # self.env_net = EnvNet(self.obv_dim, act_dim=get_act_num(self.act_dim))

        self.q_opim = optim.Adam(self.q_net.parameters(), lr=self.q_lr)
        self.env_optim = optim.Adam(self.env_net.parameters(), lr=self.env_lr)

        self.loss_func = MSELoss()

    def give_action(self, ep, env, state):
        if np.random.random() < ep:
            q_value = random.randint(0, 10**self.act_dim-1)
            action = quantization(
                pred_act=q_value,
                pred_act_high=self.act_high,
                action_low=self.env.action_space.low,
                action_high=self.env.action_space.high,
                action_dim=self.act_dim,
            )

        else:
            state_vec = state.reshape(1, -1)  # (obv_dim,) -> (1, obv_dim)
            q_value = self.q_net(state_vec)
            q_value = torch.argmax(q_value).item()

            action = quantization(
                pred_act=q_value,
                pred_act_high=self.act_high,
                action_low=self.env.action_space.low,
                action_high=self.env.action_space.high,
                action_dim=self.act_dim,
            )
        # print(action.shape, q_value)
        return q_value, action

    def train(self, *args):
        state, action, reward, done, next_state = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.int)
        next_state = torch.tensor(next_state, dtype=torch.float)

        q_value = self.q_net(state)  # act_dim
        # print("q: {}".format(q_value))
        next_q_value = self.qt_net(next_state).max(1)[0]  # act_dim
        # print("action: {}".format(action))

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