import numpy as np


def preprocessing(paths):
    """
    :param paths: [obs, act, rewards, logp_a, done]
    :return:
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    rewards = np.concatenate([path["reward"] for path in paths])
    raw_rewards = [path["reward"] for path in paths]
    concatenated_logp_as = np.concatenate([path["logp_a"] for path in paths])
    done = np.concatenate([path["done"] for path in paths])
    return observations, actions, rewards, raw_rewards, concatenated_logp_as, done


class Buffer:
    def __init__(self):
        self.paths = []
        self.observation = None
        self.actions = None
        self.rewards = None
        self.raw_rewards = []
        self.logprob_as = None
        self.done = None

    def assign_info(self, paths):
        self.paths += paths

        observations, actions, rewards, raw_rewards, logp_as, done = preprocessing(paths)

        # assign observations
        if self.observation is None:
            self.observation = observations
        else:
            self.observation = np.concatenate([self.observation, observations])

        # assign actions
        if self.actions is None:
            self.actions = actions
        else:
            self.actions = np.concatenate([self.actions, actions])

        # assign rewards
        if self.rewards is None:
            self.rewards = rewards
        else:
            self.rewards = np.concatenate([self.rewards, rewards])

        # assign raw_rewards
        if isinstance(raw_rewards, list):
            self.raw_rewards += raw_rewards
        else:
            self.raw_rewards.append(raw_rewards)

        # assign logprob over a
        if self.logprob_as is None:
            self.logprob_as = logp_as
        else:
            self.logprob_as = np.concatenate([self.logprob_as, logp_as])

        # assign done info
        if self.done is None:
            self.done = done
        else:
            self.done = np.concatenate([self.done, done])

    def evict(self):
        self.paths = []
        self.observation = None
        self.actions = None
        self.rewards = None
        self.raw_rewards = []
        self.logprob_as = None
        self.done = None

    def get_data(self):
        return self.observation, self.actions, self.rewards, self.raw_rewards, self.logprob_as, self.done