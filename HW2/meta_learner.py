from torch.optim import Adam
import torch
import numpy as np
from replay import Buffer


class MetaLearnerAgent:
    def __init__(self, policy_agent, _gamma, _lambda, clip_ratio, c1, meta_lr):
        """
        :param policy_agent: policy agent
        :param _gamma: ppo_param
        :param _lambda: ppo_param
        :param clip_ratio: ppo_param
        :param c1: ppo_param

        NOTICE: parameters in network are OrderedDicts!
        """
        self.policy_agent = policy_agent
        self._gamma = _gamma
        self._lambda = _lambda
        self.clip_ratio = clip_ratio
        self.c1 = c1
        self.meta_lr = meta_lr
        self.meta_optimizer = Adam(self.policy_agent.parameters(), self.meta_lr)

    def adapt(self, alpha, buffer, params):
        observations, actions, rewards, raw_rewards, _, _ = buffer.get_data()
        q_values = self.get_decaying_rewards(rewards, self._gamma)
        # Adv = q_val - val_net output
        advantage_values = self.get_ADV(observations, q_values, params)

        # loss_pi
        pi, logprob_a = self.policy_agent.getDist(torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), params)
        loss_pi = -(logprob_a * torch.tensor(advantage_values, dtype=torch.float32)).mean()

        # loss_val
        targets_val = torch.tensor(q_values, dtype=torch.float32)
        predicted_val = self.policy_agent.getVal(torch.tensor(observations, dtype=torch.float32), params)
        loss_val = ((predicted_val - targets_val) ** 2).mean()  # MSE
        grads = torch.autograd.grad(loss_pi + loss_val, [i[1] for i in params.items()])  # It is OrderedDict!!

        # update params
        updated_params = self.policy_agent.update_params(grads, alpha)

        return updated_params

    def step(self, adapted_params_list, meta_paths_list):
        # meta optimization

        losses = []

        for adapted_params, meta_paths in zip(adapted_params_list, meta_paths_list):
            loss = self.outer_loss(meta_paths, adapted_params)
            losses.append(loss)

        self.meta_optimizer.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        self.meta_optimizer.step()

        return loss.item()

    def outer_loss(self, meta_paths, adapted_params):

        buffer = Buffer()
        buffer.assign_info(meta_paths)
        observations, actions, rewards, raw_rewards, logprob_as, _ = buffer.get_data()

        q_values = self.get_decaying_rewards(rewards, self._gamma)
        gae = self.GAE(rewards, observations, adapted_params, self._lambda)

        pi, logprob_a = self.policy_agent.getDist(torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), adapted_params)
        ratio = torch.exp(logprob_a - torch.tensor(logprob_as, dtype=torch.float32))
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * torch.tensor(gae, dtype=torch.float32)

        loss_pi = -(torch.min(ratio * torch.tensor(gae, dtype=torch.float32), clip_adv)).mean()
        targets_val = torch.tensor(q_values , dtype=torch.float32)
        predicted_val = self.policy_agent.getVal(torch.tensor(observations, dtype=torch.float32), adapted_params)
        loss_val = ((predicted_val - targets_val) ** 2).mean()

        return loss_pi + self.c1 * loss_val

    def get_decaying_rewards(self, x, discount):
        # TOGO!
        estimated_r = []
        for t in range(len(x)):
            temp_return = 0
            for t_prime in range(t, len(x)):
                r_t_prime = x[t_prime]
                temp_return = torch.tensor(discount ** (t_prime - t) * r_t_prime)
            estimated_r.append(temp_return)

        return np.array(estimated_r)

    def get_ADV(self, observations, q_values, params):
        """
        adv = q - v
        """
        val = self.policy_agent.getVal(torch.tensor(observations, dtype=torch.float32), params)
        val = val.detach().numpy() * np.std(q_values) + np.mean(q_values)
        adv_n = q_values - val
        return adv_n

    def GAE(self, rewards, observations, params, _lambda):

        values = self.policy_agent.getVal(torch.tensor(observations, dtype=torch.float32), params).detach().numpy()

        advs = []
        adv = 0.0
        next_value = 0.0

        for r, v in zip(rewards[::-1], values[::-1]):
            td_error = r + next_value * self._gamma - v
            adv = td_error + adv * self._gamma * _lambda
            next_value = v
            advs.insert(0, adv)

        advs = torch.tensor(advs)

        return np.array(advs, dtype=np.float32)
