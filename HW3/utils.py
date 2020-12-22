import torch
import torch.nn as nn


def get_nn(obv_dim, act_dim):
    model = nn.Sequential(
        nn.Linear(obv_dim, 64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
        nn.Sigmoid(),
        nn.Linear(32, act_dim),
    )

    return model


class EnvNet(nn.Module):

    def __init__(self, obv_dim, act_dim):
        super(EnvNet, self).__init__()

        # take obv and act as the whole input?
        self.state_net = get_nn(obv_dim + act_dim, obv_dim)
        self.reward_net = get_nn(obv_dim + act_dim, 1)
        self.done_net = get_nn(obv_dim + act_dim, 1)

    def forward(self, obv, act):
        # print(obv.shape)
        # print(act.shape)
        input_ = torch.cat((obv, act.float()), dim=1)
        # print(input_.shape)

        next_state = self.state_net(input_)
        reward = self.reward_net(input_)
        done = self.done_net(input_)

        return next_state, reward, done
