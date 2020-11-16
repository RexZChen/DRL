import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from collections import OrderedDict


class PolicyAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        """
        :param obs_dim: 26
        :param act_dim: 6

        Two network have shared layers
        """
        super(PolicyAgent, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_layers = 3
        self.hidden_size = 64

        self.activation = nn.Sigmoid()

        self._construct_polciy_net()
        self._construct_value_net()
        self.log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(act_dim, dtype=np.float32)), requires_grad=True)

    def _construct_polciy_net(self):
        layer_info = [self.obs_dim] + [self.hidden_size] + [self.hidden_size] + [self.act_dim]
        for i in range(1, len(layer_info)):
            self.add_module("Policy_net_layer_{}".format(i), nn.Linear(layer_info[i - 1], layer_info[i]))

    def _construct_value_net(self):
        layer_info = [self.obs_dim] + [self.hidden_size] + [self.hidden_size] + [1]
        for i in range(1, len(layer_info)):
            self.add_module("Value_net_layer_{}".format(i), nn.Linear(layer_info[i - 1], layer_info[i]))

    def _value_foward(self, obs, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = obs

        for i in range(1, self.num_layers):
            output = F.linear(output, params["Value_net_layer_{}.weight".format(i)], params["Value_net_layer_{}.bias".format(i)])

        output = F.linear(output, params["Value_net_layer_{}.weight".format(self.num_layers)], params["Value_net_layer_{}.bias".format(self.num_layers)])

        return output

    def _policy_forward(self, obs, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = obs

        for i in range(1, self.num_layers):
            output = F.linear(output, params["Policy_net_layer_{}.weight".format(i)], params["Policy_net_layer_{}.bias".format(i)])

        output = F.linear(output, params["Policy_net_layer_{}.weight".format(self.num_layers)], params["Policy_net_layer_{}.bias".format(self.num_layers)])

        return output

    def getVal(self, obs, params=None):
        v = self._value_foward(obs, params)
        return torch.squeeze(v, dim=-1)

    def getDist(self, obs, act, params=None):
        mean = self._policy_forward(obs, params)
        if params is None:
            dist = Normal(mean, torch.exp(self.log_std))
        else:
            dist = Normal(mean, torch.exp(params['log_std']))
        log_prob = dist.log_prob(act).sum(axis=-1)
        return dist, log_prob

    def getAct(self, obs, params=None):
        with torch.no_grad():
            val = self.getVal(obs, params)
            mean = self._policy_forward(obs, params)
            if params is None:
                dist = Normal(mean, torch.exp(self.log_std))
            else:
                dist = Normal(mean, torch.exp(params['log_std']))

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        return action.numpy(), val.numpy(), log_prob.numpy()

    def update_params(self, grads, step_size=0.5):
        """
        :param grads:
        :param step_size: unsure about whether I should fix step_size...
        :return:
        """
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params
