import numpy as np
import torch


def collect_trajs(env, policy_agent, params, max_path_length, limit_path_length=False):
    observation = env.reset()
    observations, actions, rewards, logp_as, terminals = [], [], [], [], []
    steps = 0
    while True:
        observations.append(observation.copy())
        action, _, logp_a = policy_agent.getAct(torch.as_tensor(observation, dtype=torch.float32), params)
        logp_as.append(logp_a)
        actions.append(action)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards.append(reward)
        end = (steps == max_path_length or done) if limit_path_length else done
        terminals.append(end)
        if end:
            break
    return Path(observations, actions, rewards, logp_as, terminals)


def collect_n_trajs(env, policy_agent, params, num_trajectory, max_path_length, limit_path_length=False):
    paths = []
    for _ in range(num_trajectory):
        paths.append(collect_trajs(env, policy_agent, params, max_path_length, limit_path_length))
    return paths


def Path(obs, acs, rewards, logp_as, terminals):
    return {"observation": np.array(obs, dtype=np.float32),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "logp_a": np.array(logp_as, dtype=np.float32),
            "done": np.array(terminals, dtype=np.float32)}


def compute_R(meta_paths):
    return [path["reward"].sum() for path in meta_paths]