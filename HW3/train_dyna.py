import gym
import pybullet_envs
import torch
import numpy as np
import torch.nn as nn
import argparse
from DynaQ_discrete import DynaQagent_d
from DynaQ_continuous import DynaQagent_c
from buffer import OffP_Buffer

parser = argparse.ArgumentParser()

parser.add_argument("--env", default='CartPole-v1', type=str)
# C: MountainCarContinuous-v0, AntBulletEnv-v0
# D: CartPole-v1, MountainCar-v0
parser.add_argument("--epoch", default=500000, type=int)
parser.add_argument("--step", default=5000, type=int)
parser.add_argument("--trajs", default=2000, type=int)
parser.add_argument("--bs", default=500, type=int)  # buffer size
parser.add_argument("--traj_limit", default=False, type=bool)  # there are some envs that will never give a Done

args = parser.parse_args()

max_len = 500
ep_greedy = 0.1
exploration_steps = 10

env = gym.make(args.env)
discrete = isinstance(env.action_space, gym.spaces.Discrete)

obv_dim = env.observation_space.shape[0]

if discrete:
    act_dim = env.action_space.n
    # obv_dim, act_dim, replay_buffer, batch_size, gamma, q_lr, env_lr
    agent = DynaQagent_d(
        obv_dim=obv_dim,
        act_dim=act_dim,
        replay_buffer=OffP_Buffer(args.bs),
        batch_size=64,
        gamma=0.99,
        q_lr=1e-3,
        env_lr=1e-3
    )
else:
    act_dim = env.action_space.shape[0]
    # obv_dim, act_dim, replay_buffer, batch_size, gamma, q_lr, env_lr, env, act_high, act_low
    agent = DynaQagent_c(
        obv_dim=obv_dim,
        act_dim=act_dim,
        replay_buffer=OffP_Buffer(capacity=args.bs),
        batch_size=64,
        gamma=0.99,
        q_lr=1e-3,
        env_lr=1e-3,
        env=env,
        act_high=env.action_space.high[0],
        act_low=env.action_space.low[0]
    )

freezing_steps = 500  # fill the buffer before training
update_freq = 20  # inner loop, update target for each {} steps
R = []
S = []
t = 0

for i in range(args.trajs):
    obv = env.reset()
    done = False
    rewards_list = []
    step = 0
    while not done and (step < max_len if args.traj_limit else True):

        if discrete:
            # ep, env, state
            q_val, act = agent.give_action(
                ep=ep_greedy,
                env=env,
                state=torch.tensor(obv, dtype=torch.float32)
            )
        else:
            # ep, env, state
            q_val, act = agent.give_action(
                ep=ep_greedy,
                env=env,
                state=torch.tensor(obv, dtype=torch.float32)
            )

        next_obv, reward, done, _ = env.step(act)
        rewards_list.append(reward)
        agent.replay_buffer.push((obv, q_val, reward, done, next_obv))
        obv = next_obv
        step += 1
        t += 1

        if agent.replay_buffer.size() < freezing_steps:
            continue

        loss, rewards = agent.train()
        loss_env = agent.update_env()
        if i % update_freq == 0:
            agent.update_qt_net()

        print("traj steps: {}, return: {}".format(len(rewards_list), sum(rewards_list)))
        for _ in range(exploration_steps):
            agent.exploration()

    R.append(sum(rewards_list))
    S.append(len(rewards_list))