

import copy
import gym
import numpy as np
from tabulate import tabulate
import time as timer
import collections

import mjrl.samplers.core as trajectory_sampler
from mjrl.utils.logger import DataLog
import mjrl.utils.process_samples as process_samples


def interp_eval(interp_vector, phys_param, agent, args, \
        seed=0, log_string=print, **env_kwargs):
    env_kwargs = {'id': args.generalized_env, \
            'interp_param_vector': interp_vector, 'phys_param_vector_all': phys_param, \
            'ycb_scene_dir': args.ycb_scene_dir, 'simulation_mode': True, 'dense_reward': False, \
            'finger_option': 'ff', 'lf_shrink': 0.7, 'virtual_confine': True, 'horizon': args.horizon}
    env = gym.make(**env_kwargs)
    success = 0

    input_dict = dict(num_traj=args.num_eval, env=env_kwargs, policy=agent.policy, \
            horizon=args.horizon, base_seed=seed, num_cpu=args.num_eval, eval_mode=True)
    paths = trajectory_sampler.sample_paths(**input_dict)
    success_rate_eval = env.env.evaluate_success(paths) / 100.
    log_string('num eval: {}'.format(args.num_eval))

    paths = sorted(paths, key=lambda d: d['seed'])
    path_rewards = np.array([sum(p["rewards"]) for p in paths])

    log_string("...........................................................")
    log_string("Eval Success Rate : {} ".format(success_rate_eval * 100.) + '     Eval Reward: {}'.format(np.mean(path_rewards)))
    log_string("...........................................................")
    return success_rate_eval, path_rewards


def interp_forward(target_interp_vector, phys_param, agent, args, \
        interp_sample_range=0.01, \
        expl_success_rate_thres=0.5, \
        eval_success_rate_thres=0.8, \
        sample_direction=None, \
        eval_freq=5, num_eval=20, interp_range_shrink=0.99, \
        max_iter=np.inf, \
        replay_buffer_size=768, \
        log_string=print, \
        **env_kwargs):
    env = gym.make(args.generalized_env, \
            interp_param_vector=target_interp_vector, \
            dense_reward=False, **env_kwargs)

    sample_direction = sample_direction / np.linalg.norm(sample_direction)

    current_interp_range = interp_sample_range
    num_iter = 0
    eval_iter = False
    # replay_buffer = collections.deque()
    replay_buffer = []
    while True:
        if num_iter >= max_iter:
            return None, None

        if eval_iter or ((num_iter % eval_freq) == 0):
            if len(replay_buffer) > 0:
                success_rate_eval, path_rewards = interp_eval( \
                        target_interp_vector, phys_param, \
                        agent, args, seed=agent.seed+2**31+eval_iter, \
                        log_string=log_string, **env_kwargs)
                # success_rate_eval, path_rewards = interp_eval(target_interp_vector, agent, args, \
                #         num_eval=num_eval, seed=agent.seed+2**31, log_string=log_string, **env_kwargs)

                if success_rate_eval >= eval_success_rate_thres:
                    mean_path_reward = np.mean(path_rewards)
                    return agent, mean_path_reward
                eval_iter = False

        log_string("Sample of {} Iteration: {} ".format(args.num_traj, num_iter))
        log_string('current_interp_range: {}'.format(current_interp_range))

        env_kwargs_init = {'id': args.generalized_env, \
                'phys_param_vector_all': phys_param, \
                'ycb_scene_dir': args.ycb_scene_dir, 'simulation_mode': True, 'dense_reward': False, \
                'finger_option': 'ff', 'lf_shrink': 0.7, 'virtual_confine': True, 'horizon': args.horizon}
        # env_kwargs_init = {'id': args.generalized_env, 'dense_reward': False}
        env_kwargs_init.update(env_kwargs)

        env_kwargs_list = []
        for i in range(args.num_traj):
            sampled_interp_vector = target_interp_vector + \
                    (-current_interp_range) * i/float(args.num_traj-1) * sample_direction

            env_kwargs_tmp = copy.deepcopy(env_kwargs_init)
            env_kwargs_tmp['interp_param_vector'] = sampled_interp_vector
            env_kwargs_list.append(env_kwargs_tmp)

        ##### train from sampled paths
        ts = timer.time()

        input_dict = dict(num_traj=args.num_traj, env=env_kwargs_list, \
                policy=agent.policy, horizon=1e6,
                base_seed=agent.seed, num_cpu=args.num_cpu)
        paths = trajectory_sampler.sample_paths(**input_dict)
        paths = sorted(paths, key=lambda d: d['seed'])
        max_reward_returns = np.max([sum(p["rewards"]) for p in paths])
        agent.seed = agent.seed + args.num_traj if agent.seed is not None else agent.seed

        sample_success_rate = []
        for i in range(len(paths)):
            # replay_buffer.append([paths[i], env_kwargs_list[i]])
            success = env.evaluate_success([paths[i]]) > 50.
            sample_success_rate.append(success)
        replay_buffer = paths

        sample_success_rate = np.mean(sample_success_rate)
        log_string('sample_success_rate: {}'.format(sample_success_rate))

        if agent.save_logs:
            agent.logger.log_kv('time_sampling', timer.time() - ts)

        ts = timer.time()
        # num_train_iter = min(len(replay_buffer) // args.num_traj, 5)
        num_train_iter = 1
        for _ in range(num_train_iter):
            ###### sample paths for training
            ### success/positive
            # sampled_idx = np.random.choice(len(replay_buffer), args.num_traj, replace=False)
            # sampled_paths = [replay_buffer[idx][0] for idx in sampled_idx]
            # paths = sampled_paths
            ###### sample paths for training

            log_string("Train Iteration")
            # compute returns
            process_samples.compute_returns(paths, args.gamma)
            # compute advantages
            process_samples.compute_advantages(paths, agent.baseline, args.gamma, args.gae)
            # train from paths
            eval_statistics = []
            eval_statistics = agent.train_from_paths(paths)

            # fit baseline
            if agent.save_logs:
                error_before, error_after = agent.baseline.fit(paths, return_errors=True)
                agent.logger.log_kv('VF_error_before', error_before)
                agent.logger.log_kv('VF_error_after', error_after)
            else:
                agent.baseline.fit(paths)
        ##### train from sampled paths

        # log number of samples
        if agent.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            agent.logger.log_kv('time_train', timer.time()-ts)

        train_log = agent.logger.get_current_log()

        print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1, train_log.items()))
        log_string(tabulate(print_data))

        # success_rate = float(train_log['success_rate']) / 100.
        if sample_success_rate >= expl_success_rate_thres:
            eval_iter = True

        ##### update replay buffer
        # if len(replay_buffer) > replay_buffer_size:
        #     for _ in range(len(replay_buffer) - replay_buffer_size):
        #         replay_buffer.popleft()
        # log_string('replay_buffer size: {}'.format(len(replay_buffer)))
        # log_string('')
        ##### update replay buffer

        num_iter += 1
        # if sample_success_rate > (1./args.num_traj + 1e-8):
        if sample_success_rate > 1e-8:
            current_interp_range *= interp_range_shrink



