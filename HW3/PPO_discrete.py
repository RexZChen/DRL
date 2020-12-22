import torch
import torch.nn as nn
from utils import get_nn
from torch.distributions.categorical import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, obv_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # input state => action
        self.mlp1 = nn.Linear(obv_dim, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.mlp3 = nn.Linear(32, action_dim)
        # self.std = nn.Parameter(-0.5 * torch.ones((action_dim,)), requires_grad=True)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, obv):
        x = self.mlp1(obv)
        x = self.sig(x)
        x = self.mlp2(x)
        x = self.sig(x)
        x = self.mlp3(x)
        # x = self.relu(x)
        dist = Categorical(logits=x)
        action = dist.sample()
        # action = self.tanh(action)
        return action

    def getDist(self, obv):
        x = self.mlp1(obv)
        x = self.sig(x)
        x = self.mlp2(x)
        x = self.sig(x)
        x = self.mlp3(x)
        # x = self.relu(x)
        dist = Categorical(logits=x)
        return dist


class ValueNetwork(nn.Module):
    def __init__(self, obv_dim):
        super(ValueNetwork, self).__init__()
        # input state => action
        self.mlp1 = nn.Linear(obv_dim, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.mlp3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, obv):
        x = self.mlp1(obv)
        x = self.sig(x)
        x = self.mlp2(x)
        x = self.sig(x)
        x = self.mlp3(x)
        return x


class PPOAgent_d:
    def __init__(self, obv_dim, act_dim):
        self.policy_net = PolicyNetwork(obv_dim, act_dim)
        self.value_net = ValueNetwork(obv_dim)
