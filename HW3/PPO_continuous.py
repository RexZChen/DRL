import torch
import torch.nn as nn
from utils import get_nn
from torch.distributions.normal import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, obv_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # input state => action
        self.mlp1 = nn.Linear(obv_dim, 64)
        self.mlp2 = nn.Linear(64, 32)
        self.mlp3 = nn.Linear(32, action_dim)
        self.std = nn.Parameter(-0.5 * torch.ones((action_dim,)), requires_grad=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, obv):
        x = self.mlp1(obv)
        x = self.sig(x)
        x = self.mlp2(x)
        x = self.sig(x)
        mean = self.mlp3(x)
        std = torch.exp(self.std)
        dist = Normal(mean, std)
        action = dist.sample()
        action = self.tanh(action)
        return action

    def getSTD(self):
        return torch.exp(self.std)

    def getParam(self):
        return self.mlp3

    def getDist(self, obv):
        x = self.mlp1(obv)
        x = self.sig(x)
        x = self.mlp2(x)
        x = self.sig(x)
        mean = self.mlp3(x)
        std = torch.exp(self.std)
        dist = Normal(mean, std)
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


class PPOAgent_c:
    def __init__(self, obv_dim, act_dim):
        self.policy_net = PolicyNetwork(obv_dim, act_dim)
        self.value_net = ValueNetwork(obv_dim)
