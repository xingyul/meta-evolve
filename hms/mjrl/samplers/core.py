import logging
import numpy as np
import os
from mjrl.utils import tensor_utils
import gym
import psutil
import torch
import multiprocessing
import time as timer
logging.disable(logging.CRITICAL)



# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        queue = None,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environments (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :return:
    """

    # get the correct env behavior
    if isinstance(env, gym.wrappers.time_limit.TimeLimit):
        pass
    elif isinstance(env, dict):
        env = gym.make(**env)
    elif isinstance(env, list):
        env_list = []
        for e in env:
            if isinstance(e, gym.wrappers.time_limit.TimeLimit):
                env_list.append(e)
            elif isinstance(e, dict):
                env_list.append(gym.make(**e))
        env = env_list

    if isinstance(env, list):
        horizon = min(horizon, env[0].env.horizon)
    else:
        try:
            horizon = min(horizon, env.env.horizon)
        except:
            horizon = min(horizon, env.horizon)

    paths = []
    for ep in range(num_traj):
        if isinstance(env, list):
            e = env[ep]
        else:
            e = env

        # seeding
        if isinstance(base_seed, list):
            seed = base_seed[ep]
            # e.env.seed(seed)
            # np.random.seed(seed)
        else:
            seed = base_seed + ep
            # e.env.seed(seed)
            # np.random.seed(seed)

        o = e.reset(seed=seed)

        observations=[]
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []

        done = False
        t = 0

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']
            try:
                next_o, r, done, env_info = e.step(a)
            except:
                observations = []
                actions=[]
                rewards=[]
                agent_infos = []
                env_infos = []
                done = False
                seed = seed + num_traj
                o = e.env.reset(seed=seed)
                t = 0

            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            seed=seed,
            terminated=done
        )
        paths.append(path)

    if isinstance(env, list):
        for e in env:
            del(e)
    del(env)

    if queue is not None:
        queue.put(paths)
    else:
        return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        kill_children=False,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed,
                          queue=None)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    original_policy_device = policy.device
    if original_policy_device.type != 'cpu':
        policy.device = torch.device('cpu')
        policy.model.to(policy.device)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    paths = [None] * num_traj
    queue = multiprocessing.Queue()

    for i in range(1, num_cpu):
        if isinstance(env, list):
            env_batch = env[i*paths_per_cpu:(i+1)*paths_per_cpu]
        else:
            env_batch = env
        if isinstance(base_seed, list):
            seed = base_seed[i*paths_per_cpu:(i+1)*paths_per_cpu]
        else:
            seed = base_seed + i*paths_per_cpu

        worker_args = (paths_per_cpu, env_batch, policy, eval_mode, horizon,
                       seed, queue)
        worker = multiprocessing.Process(target=do_rollout, args=worker_args)
        worker.start()

    if isinstance(env, list):
        env_batch = env[0:paths_per_cpu]
    else:
        env_batch = env
    if isinstance(base_seed, list):
        seed = base_seed[0:paths_per_cpu]
    else:
        seed = base_seed
    input_dict = dict(num_traj=paths_per_cpu, env=env_batch, policy=policy,
                      eval_mode=eval_mode, horizon=horizon,
                      base_seed=seed, queue=None)
    worker_path_batch = do_rollout(**input_dict)

    idx = 0
    for path in worker_path_batch:
        paths[idx] = path
        idx += 1
    while True:
        worker_path_batch = queue.get()
        for path in worker_path_batch:
            paths[idx] = path
            idx += 1
            if idx == num_traj:
                break
        if idx == num_traj:
            break

    if original_policy_device.type != 'cpu':
        policy.device = original_policy_device
        policy.model.to(policy.device)

    ##### kill child processes to prevent memory leakage
    if kill_children:
        own_pid = os.getpid()
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        child_pids = [child.pid for child in children]
        pids = [pid for pid in child_pids if pid != own_pid]
        command = 'kill -9 {}'.format(' '.join([str(pid) for pid in pids]))
        os.popen(command)

    return paths


