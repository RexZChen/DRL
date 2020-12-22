Environments: Python = 3.8.x, Pytorch = 1.4.0

How to run:

            DQN: python train_dqn.py --env[ENV] --epochs --trajs[num of trajectories] --bs[buffer size]
            PPO: python train_ppo.py --env[ENV] --epochs
            DynaQ: python train_dyna.py --env[ENV] --epochs --trajs[num of trajectories] --bs[buffer size]
    (All parameters have already been set to some value by default, you can directly run those files without any args.)

Notice:
    [1] Higher torch versions might raise errors while training, because some operations will be identified as in-place
    ones and as a result, the training will fail.
    [2] Although I recevied a lot of help from friends and classmates, I finished this assignment independently.
    [3] My version of PPO needs to collect the whole trajs and compute the returns, thus may need more time(~20 mins) to
    get the results of first epoch.
    [4] Log files are not included. See figures under /figures.
    [5] Some hyper parameters are lost due to too frequent experiments.
    [6] See write-up in /write-up.
    [7] Some comments are written in Chinese, hope it won't bother.



Happy end of the semester and happy holidays! :)