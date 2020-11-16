from maml_env import HalfCheetahDirecBulletEnv
import random
import torch
import numpy as np
from utils import collect_n_trajs, compute_R
from policy_agent import PolicyAgent
from meta_learner import MetaLearnerAgent
from collections import OrderedDict, defaultdict
from replay import Buffer
import os


class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)


def main(args):
    # I tried to put some random seeds here, but it did not improve the results much.
    # So I deleted this part.

    tasks = Tasks(("Forward", True), ("Backward", False))

    env = HalfCheetahDirecBulletEnv()

    # Observation and action sizes
    obs_dim = 26
    act_dim = 6

    policy_agent = PolicyAgent(obs_dim, act_dim)

    meta_learner = MetaLearnerAgent(policy_agent, args.discount, args.lam, args.clip_ratio, args.c1, args.beta)

    policy_list = []  # [(policy, average return)]
    average_return_list = []

    env.close()

    # Outer loop
    for meta_iter in range(args.meta_iteration):

        adapted_params_list = []
        meta_paths_list = []
        task_return_dict = defaultdict(list)  # dic of list
        R = []

        for task_config in tasks.sample_tasks(args.meta_batch_size):
            # Inner loop
            task_name, env_args = task_config[0], task_config[1:]
            env = HalfCheetahDirecBulletEnv(*env_args)

            adapted_params = OrderedDict(policy_agent.named_parameters())

            # Adaptation
            for i in range(args.num_of_steps):
                # Collect trajs
                buffer = Buffer()
                paths = collect_n_trajs(env, policy_agent, adapted_params, args.num_trajectory, args.horizon, True)
                # cut at horizon if necessary

                buffer.assign_info(paths)
                alpha = args.alpha
                adapted_params = meta_learner.adapt(alpha, buffer, adapted_params)

            adapted_params_list.append(adapted_params)
            # Run adapted policy
            # collect trajs with new params

            meta_paths = collect_n_trajs(env, policy_agent, adapted_params, args.num_trajectory, args.horizon, True)
            meta_paths_list.append(meta_paths)

            total_r = compute_R(meta_paths)
            print("returns of all steps: ", total_r)
            task_return_dict[task_name] += total_r
            R += total_r
            env.close()

        # Meta Optimization
        loss = meta_learner.step(adapted_params_list, meta_paths_list)

        # log statistics
        print("{} MetaLoss at {}th iteration".format(loss, meta_iter))
        print("Avg Returns on validation: ", np.mean(R))
        average_return_list.append(np.mean(R))

        # returns on different tasks
        for task_name, return_list in task_return_dict.items():
            print("Return/{}".format(task_name), np.mean(return_list))

        # saving policies...
        dir_name = "./policy_{}/meta_{}".format(args.num_of_steps, meta_iter)
        os.makedirs(dir_name)
        torch.save(meta_learner.policy_agent, dir_name + "/model.pt")
        policy_list.append((dir_name, np.mean(R)))

    # save the average return list
    return_dir_name = "./avg_return"
    os.makedirs(return_dir_name, exist_ok=True)
    np.savetxt(return_dir_name + "/{}_adapt.csv".format(args.num_of_steps), np.array(average_return_list))

    # Oracle Hypothesis Selection for the best one for testing!
    dir_name, highest_return = sorted(policy_list, key=lambda x: x[1], reverse=True)[0]
    best_policy_agent = torch.load(dir_name + "/model.pt")

    test_meta_learner = MetaLearnerAgent(best_policy_agent, args.discount, args.lam, args.clip_ratio, args.c1,
                                         args.beta)

    test_task_return_dict = defaultdict(list)
    test_R = []

    #################################################  test #################################################
    # In test, basically it is doing the same, so pasting...

    for task_config in tasks.sample_tasks(args.meta_batch_size):
        # Inner loop
        task_name, env_args = task_config[0], task_config[1:]
        env = HalfCheetahDirecBulletEnv(*env_args)

        adapted_params = OrderedDict(test_meta_learner.policy_agent.named_parameters())

        # Adaptation
        for i in range(args.num_of_steps):
            # Collect trajectories
            buffer = Buffer()
            paths = collect_n_trajs(env, test_meta_learner.policy_agent, adapted_params, args.num_trajectory,
                                    args.horizon, True)
            buffer.assign_info(paths)
            alpha = args.alpha
            adapted_params = test_meta_learner.adapt(alpha, buffer, adapted_params)
        # Run adapted policy
        # collect trajs with new params

        meta_paths = collect_n_trajs(env, test_meta_learner.policy_agent, adapted_params, args.num_trajectory,
                                     args.horizon, True)

        test_R = compute_R(meta_paths)
        print("test returns: ", test_R)
        test_task_return_dict[task_name] += test_R
        test_R += test_R
        env.close()

    print("Avg Return on test: ", np.mean(test_R))
    for task_name, return_list in test_task_return_dict.items():
        print("Test Return/{}".format(task_name), np.mean(return_list))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=500, type=int)
    parser.add_argument("--meta_batch_size", default=40, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_of_steps", default=1, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)  # 1e-5, 1e-3, 0.1 tried
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument("--num_trajectory", "-K", default=20, type=int)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--lam', type=float, default=0.9)
    parser.add_argument('--c1', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1e-4)  # lr for meta-learning, 1e-4, 0.1, 0.01 tried

    args = parser.parse_args()

    main(args)
