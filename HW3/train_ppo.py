import torch
import torch.optim as optim
import pybullet_envs
import gym
import argparse
from PPO_continuous import PPOAgent_c
from PPO_discrete import PPOAgent_d


class OnPolicyBuffer:
    def __init__(self):
        self.buffer = []

    def evict(self):
        self.buffer = []

    def put(self, sample):
        self.buffer.append(sample)


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MountainCarContinuous-v0")
# C: MountainCarContinuous-v0, AntBulletEnv-v0
# D: CartPole-v1, MountainCar-v0
parser.add_argument("--epoch", default=500, type=int)

args = parser.parse_args()

env = gym.make(args.env)
discrete = isinstance(env.action_space, gym.spaces.Discrete)
obv_dim = env.observation_space.shape[0]

if discrete:
    act_dim = env.action_space.n
    agent = PPOAgent_d(obv_dim=obv_dim, act_dim=act_dim)
else:
    act_dim = env.action_space.shape[0]
    agent = PPOAgent_c(obv_dim=obv_dim, act_dim=act_dim)

buffer = OnPolicyBuffer()
num_of_trajectories = 20
gamma = 0.99
learning_rate_alpha = 1e-5
learning_rate_beta = 1e-3
lambda_ = torch.tensor([0.95])
epsilon = 0.2
c1 = torch.tensor([0.5])

optimizer_pi = optim.Adam(agent.policy_net.parameters(), lr=learning_rate_alpha)
optimizer_val = optim.Adam(agent.value_net.parameters(), lr=learning_rate_beta)

for epoch in range(args.epoch):
    #  Collect Trajectories
    running_loss_pi = 0
    # running_loss_val = 0
    policy_loss = 0
    # value_loss = 0
    R = 0
    # G = []
    T = 0
    for n in range(num_of_trajectories):
        observation = torch.tensor(env.reset(), dtype=torch.float)
        action = agent.policy_net(observation)
        if discrete:
            observation_prime, reward, done, _ = env.step(action.item())
        else:
            observation_prime, reward, done, _ = env.step(action)
        log_prob_old = torch.sum(agent.policy_net.getDist(observation).log_prob(action))
        buffer.put([observation, action, reward, observation_prime, log_prob_old])

        while not done:
            observation = observation_prime
            action = agent.policy_net(torch.tensor(observation, dtype=torch.float))
            if discrete:
                observation_prime, reward, done, _ = env.step(action.item())
            else:
                observation_prime, reward, done, _ = env.step(action)
            log_prob_old = torch.sum(
                agent.policy_net.getDist(torch.tensor(observation, dtype=torch.float)).log_prob(torch.tensor(action)))
            buffer.put([observation, action, reward, observation_prime, log_prob_old])

        # Estimate Returns
        estimated_A = []
        estimated_r = []
        for t in range(len(buffer.buffer)):
            temp_A = 0
            temp_return = 0
            for t_prime in range(t, len(buffer.buffer)):
                r_t_prime = buffer.buffer[t_prime][2]
                s_t_prime = buffer.buffer[t_prime][0]
                s_t_prime_next = buffer.buffer[t_prime][3]
                temp_A = torch.tensor((lambda_ * gamma) ** (t_prime - t) * (
                        r_t_prime + gamma * agent.value_net(torch.tensor(s_t_prime_next, dtype=torch.float)) - agent.value_net(
                    torch.tensor(s_t_prime, dtype=torch.float))))
                temp_return = torch.tensor(gamma ** (t_prime - t) * r_t_prime, dtype=torch.float)

            estimated_A.append(temp_A)
            estimated_r.append(temp_return)

        estimator = 0
        accumulated_reward = 0
        time_steps = 0
        # temp_value_loss = 0
        for t in range(len(buffer.buffer)):
            s_t = torch.tensor(buffer.buffer[t][0])
            a_t = torch.tensor(buffer.buffer[t][1])  # (8,)
            r_t = torch.tensor(buffer.buffer[t][2])
            log_prob_old = torch.tensor(buffer.buffer[t][4])
            log_prob_new = torch.sum(agent.policy_net.getDist(torch.tensor(s_t, dtype=torch.float)).log_prob(a_t), dtype=torch.float)
            log_prob_ratio = log_prob_new / log_prob_old
            term1 = log_prob_ratio * estimated_A[t]
            term2 = torch.clamp(log_prob_ratio, 1 - epsilon, 1 + epsilon) * estimated_A[t]
            temp_estimator = torch.min(term1, term2) - c1 * ((estimated_r[t] - agent.value_net(torch.tensor(s_t, dtype=torch.float))) ** 2)
            accumulated_reward += r_t
            estimator = estimator + temp_estimator
            # temp_value_loss = temp_value_loss + (estimated_r[t] - agent.value_net(s_t)) ** 2

        time_steps += len(buffer.buffer)
        buffer.evict()
        # print("Reward from {}th traj is {}".format(n+1, accumulated_reward))
        R += accumulated_reward
        T += time_steps
        # print(type(log_term))
        policy_loss = estimator

    policy_loss = torch.tensor(- policy_loss / num_of_trajectories, requires_grad=True)
    R = torch.tensor(R / num_of_trajectories)
    print("Avg Reward: {}, among {} steps".format(R, T))
    # Improve Policy
    # file = r'ppo.txt'
    # with open(file, 'a+') as f:
    #     f.write(str([R.detach().numpy(), T]) + '\n')
    optimizer_pi.zero_grad()
    optimizer_val.zero_grad()
    policy_loss.backward()
    optimizer_pi.step()
    optimizer_val.step()

    # Log Statistics
    running_loss_pi += policy_loss.item()
    print('[%d\] loss: %.3f' % (epoch + 1, running_loss_pi))
    # print('Current Covariance: {}'.format(agent.policy_net.getSTD()))
