"""
This is a job script for running policy gradient algorithms on gym tasks.
Separate job scripts are provided to run few other algorithms
- For DAPG see here: https://github.com/aravindr93/hand_dapg/tree/master/dapg/examples
- For model-based NPG see here: https://github.com/aravindr93/mjrl/tree/master/mjrl/algos/model_accel
"""

from mjrl.algos.npg_cg import NPG

import os
import json
import glob
import gym
import importlib
import numpy as np
from datetime import datetime
import copy
import torch
import pickle
import random
import json
import argparse
import ast

import utils.rl_env_expansion as rl_env_expansion

import hand_manipulation_suite
# import interp_utils
import interp_utils_replay as interp_utils


# ===============================================================================
# Get command line arguments
# ===============================================================================

parser = argparse.ArgumentParser(description='Natural policy gradient from mjrl on mujoco environments')

parser.add_argument('--command_file', type=str, default='command_train_evolve_tree.sh', help='command file name')
parser.add_argument('--exp', type=str, default='steiner', help='exp name')
parser.add_argument('--metric', type=str, default='l1', help='metric to use')
parser.add_argument('--generalized_env', type=str, default='generalized-hammer-v2', help='env name')
parser.add_argument('--gpu', type=str, default='-1', help='gpu id')
parser.add_argument('--algorithm', type=str, default='NPG', help='algo name')
parser.add_argument('--obs_expansion', type=int, default=7, help='observation dim expansion')
parser.add_argument('--act_expansion', type=int, default=7, help='activation dim expansion')
parser.add_argument('--ycb_scene_dir', type=str, default='', help='ycb scene directory')
parser.add_argument('--init_log_std', type=float, default=-1.5, help='initial expanded log std')
parser.add_argument('--interp_vector_dir', type=str, default='None', help='directory for initial interp vector')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--num_eval', type=int, default=10, help='number of eval epochs')
parser.add_argument('--interp_progression', type=float, default=0.01, help='interp progression every step')
parser.add_argument('--interp_sample_range', type=float, default=0.02, help='range of random')
parser.add_argument('--initial_dir_random', type=int, default=1, help='initial direction random or not')
parser.add_argument('--eval_success_rate_thres', type=float, default=0.8, help='evaluation success rate threshold')
parser.add_argument('--expl_success_rate_thres', type=float, default=0.5, help='exploration success rate threshold')
parser.add_argument('--env_sample_method', type=str, default='radius', help='env sample method')
parser.add_argument('--sample_mode', type=str, default='trajectories', help='sample mode')
parser.add_argument('--attract_lambda', type=float, default=1., help='attract lambda factor')
parser.add_argument('--local_r_shaping', type=float, default=0., help='local reward shaping factor')
parser.add_argument('--seed_len', type=int, default=1, help='seed length')
parser.add_argument('--horizon', type=int, default=200, help='scenario horizon')
parser.add_argument('--num_traj', type=int, default=16, help='num of trajectories to sample')
parser.add_argument('--num_cpu', type=int, default=1, help='num of cpu')
parser.add_argument('--num_jac', type=int, default=120, help='num of jacobian vectors')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--step_size', type=float, default=0.0001, help='training step size')
parser.add_argument('--gamma', type=float, default=0.995, help='gamma')
parser.add_argument('--gae', type=float, default=0.97, help='gae')
parser.add_argument('--log_dir', type=str, required=True, help='location to store results')
args = parser.parse_args()



def one_zero_confine_forward(goal_interp, current_interp_param, \
        forward_vec, args):
    correction = goal_interp - current_interp_param
    correction = correction / np.linalg.norm(correction)
    ##### ensure it stay in [0, 1]
    forward_vec = forward_vec / np.linalg.norm(forward_vec) * args.interp_progression
    while True:
        ##### clip between 0 and 1
        forward_vec_clipped = np.clip(forward_vec, \
                np.zeros(len(current_interp_param)) - current_interp_param, \
                np.ones(len(current_interp_param)) - current_interp_param)
        clip_diff = np.max(np.linalg.norm(forward_vec_clipped - forward_vec, axis=-1))
        forward_vec = forward_vec_clipped
        forward_vec = forward_vec / np.linalg.norm(forward_vec) * args.interp_progression
        if clip_diff < 1e-6:
            ##### make sure the forward is in right direction
            if np.dot(forward_vec, correction) > 1e-3:
                break
            else:
                forward_vec = forward_vec + args.attract_lambda * correction
    return forward_vec


if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

os.system('cp {} {}'.format(args.command_file, args.log_dir))
os.system('cp {} {}'.format('interp_utils*.py', args.log_dir))
os.system('cp {} {}'.format('make_generalized_envs*.py', args.log_dir))
os.system('cp {} {}'.format('hand_manipulation_suite/generalized_*.py', args.log_dir))
os.system('cp {} {}'.format('hand_manipulation_suite/*interp_params_dict*.json', args.log_dir))
os.system('cp {} {}'.format('hand_manipulation_suite/*phys_param*.json', args.log_dir))
os.system('cp {} {}'.format('test_env_unified.py', args.log_dir))
os.system('cp {} {}'.format(__file__, args.log_dir))





# ===============================================================================
# Train Loop
# ===============================================================================


if (args.gpu != 'None') and (args.gpu != '-1'):
    gpu_device = torch.device('cuda:{}'.format(args.gpu))
else:
    gpu_device = torch.device('cpu')
cpu_device = torch.device('cpu')


interp_param_filename = os.path.join('hand_manipulation_suite', \
    'interp_params_dict_{}.json'.format(args.metric))
with open(interp_param_filename, 'r') as f:
    interp_params_dict = json.load(f)


phys_param_vector_all_filename = os.path.join('hand_manipulation_suite', 'phys_param_vector_all.json')
with open(phys_param_vector_all_filename, 'r') as f:
    phys_param_vector_all = json.load(f)
phys_param_vector_all = np.array(phys_param_vector_all)

env_kwargs = {'phys_param_vector_all': phys_param_vector_all, \
        'ycb_scene_dir': args.ycb_scene_dir, 'simulation_mode': True, \
        'finger_option': 'ff', 'lf_shrink': 0.7, 'virtual_confine': True, \
        'horizon': args.horizon}


interp_params = [np.array(interp_params_dict[g]) for g in interp_params_dict.keys()]
interp_space_dim = len(interp_params[0])


tree = importlib.import_module('tree_{}'.format(args.exp))

for start_robot_name, end_robot_name in tree.tree:
    log_dir = os.path.join(args.log_dir, \
            'log__{}__{}'.format(start_robot_name, end_robot_name))
    os.makedirs(os.path.join(log_dir, 'iterations'), exist_ok=True)

    LOG_FOUT = open(os.path.join(log_dir, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args)+'\n')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)
    log_string('pid: %s'%(str(os.getpid())))
    log_string('INTERP_PARAMS: {}'.format(interp_params_dict))

    if start_robot_name == 'original':
        interp_vector_dir = args.interp_vector_dir
    else:
        interp_vector_dir = glob.glob(os.path.join(args.log_dir, \
                'log_*_{}'.format(start_robot_name), 'iterations'))[0]
    interp_param_vector = np.zeros([interp_space_dim]).astype('float32')

    round_filenames = glob.glob(os.path.join(interp_vector_dir, 'round_*.txt'))
    round_filenames.sort()
    round_filename = round_filenames[-1]
    round_id = os.path.basename(round_filename).split('.txt')[0].split('round_')[1]
    policy_filename = os.path.join(interp_vector_dir, 'policy_{}.pickle'.format(round_id))
    baseline_filename = os.path.join(interp_vector_dir, 'baseline_{}.pickle'.format(round_id))
    interp_vector_file = round_filename
    with open(interp_vector_file, 'r') as f:
        interp_param_vector = ast.literal_eval(f.read())
    interp_param_vector = np.array(interp_param_vector)

    policy = pickle.load(open(policy_filename, 'rb'))
    baseline = pickle.load(open(baseline_filename, 'rb'))

    if start_robot_name == 'original':
        if (args.obs_expansion > 0) or (args.act_expansion > 0):
            policy = rl_env_expansion.policy_expansion(policy, \
                    num_obs_expansion=args.obs_expansion, num_act_expansion=args.act_expansion, \
                    init_log_std=args.init_log_std)
        if args.obs_expansion > 0:
            baseline = rl_env_expansion.baseline_expansion(baseline, \
                    num_obs_expansion=args.obs_expansion)

    policy.to(gpu_device, leaf=False)
    baseline.to(gpu_device)
    policy.seed(args.seed)

    if start_robot_name == 'original':
        round_id = 1
    else:
        round_id = int(round_id) + 1


    env = gym.make(args.generalized_env, \
            interp_param_vector=interp_param_vector, \
            dense_reward=False, **env_kwargs)

    agent = NPG(env, policy, baseline, \
            normalized_step_size=args.step_size,
            seed=args.seed, save_logs=True, device=gpu_device, **dict())


    np.random.seed(args.seed)

    prev_eval_reward = None
    interp_path = None

    goal_interp_param = np.array(interp_params_dict[end_robot_name])

    if args.initial_dir_random:
        initial_direction = np.random.uniform(0, 1, [interp_space_dim])
    else:
        initial_direction = goal_interp_param - interp_param_vector
    initial_direction = initial_direction / np.linalg.norm(initial_direction)


    iteration = 0
    forward_vec = initial_direction * args.interp_progression
    final_round = False
    while True:
        log_string("")
        log_string("...........................................................")
        log_string(str(datetime.now()))

        iteration += 1

        ##### progress interp_param_vector with forward_vec
        if np.linalg.norm(interp_param_vector - goal_interp_param) < args.interp_progression:
            sample_vec = goal_interp_param - interp_param_vector
            next_interp_param_vector = goal_interp_param
        else:
            forward_vec = one_zero_confine_forward(goal_interp_param, \
                    interp_param_vector, forward_vec, args)

            ##### try out the current forward and see if it's still good
            log_string('trying out next forward evals ...'.format(args.num_cpu))
            success_rate_eval, eval_rewards = interp_utils.interp_eval( \
                    interp_param_vector + forward_vec, phys_param_vector_all, \
                    agent, args, seed=agent.seed+2**31+iteration, \
                    log_string=log_string, **env_kwargs)

            if success_rate_eval < args.eval_success_rate_thres:
                if args.attract_lambda > 1e5:
                    correction = goal_interp_param - interp_param_vector
                    forward_vec = correction
                    sample_direction = correction
                    forward_vec = forward_vec / np.linalg.norm(forward_vec) * args.interp_progression
                else:
                    jacobian = interp_utils.interp_jacobian_plus( \
                            interp_param_vector, agent=agent, args=args, num_eval=args.num_jac, \
                            delta=0.02, \
                            seed=args.seed+args.num_traj*iteration, \
                            base_eval_reward=prev_eval_reward, log_string=log_string)
                    jacobian = jacobian / np.linalg.norm(jacobian)
                    log_string('jacobian {}'.format(jacobian.tolist()))

                    forward_vec = one_zero_confine_forward(goal_interp_param, \
                            interp_param_vector, jacobian, args)
            else:
                prev_eval_reward = np.mean(eval_rewards)

            next_interp_param_vector = interp_param_vector + forward_vec
            sample_vec = forward_vec

        next_interp_param_vector = np.clip(next_interp_param_vector, 0., 1.)

        log_string('after forward ...')
        log_string("Current Interp: {} ".format(next_interp_param_vector.tolist()))
        log_string("Current Goal: {} ".format(goal_interp_param.tolist()))
        log_string("Current Interp distance to goal: {} ".format(np.linalg.norm(next_interp_param_vector - goal_interp_param)))
        log_string("Current Interp distance to goal normalized: {} ".format(np.linalg.norm(next_interp_param_vector - goal_interp_param) / np.sqrt(interp_space_dim) ))

        ##### set param for training
        if np.linalg.norm(next_interp_param_vector - goal_interp_param) < 1e-8:
            final_round = True

        # if final_round:
        if final_round and (not end_robot_name.startswith('steiner_')) and (not end_robot_name.startswith('geom_med')):
            interp_sample_range = 0.
            expl_success_rate_thres = 1.1
            eval_success_rate_thres = 0.8
        else:
            interp_sample_range = np.linalg.norm(sample_vec)
            expl_success_rate_thres = args.expl_success_rate_thres
            eval_success_rate_thres = args.eval_success_rate_thres

        if success_rate_eval < eval_success_rate_thres:
            ##### finetune to the next interp vector
            agent_return, mean_eval_reward = interp_utils.interp_forward( \
                    target_interp_vector=next_interp_param_vector.astype('float32'), \
                    phys_param=phys_param_vector_all, \
                    agent=agent, args=args, \
                    interp_sample_range=interp_sample_range, \
                    expl_success_rate_thres=expl_success_rate_thres, \
                    eval_success_rate_thres=eval_success_rate_thres, \
                    sample_direction=sample_vec, eval_freq=5, num_eval=args.num_eval, \
                    log_string=log_string, \
                    replay_buffer_size=args.num_traj, \
                    **env_kwargs)
                    # agent=agent, args=args, seed=seed,\
            if agent_return is None:
                continue
            else:
                ##### finetune to the next interp vector
                agent = agent_return
                prev_eval_reward = mean_eval_reward
        interp_param_vector = next_interp_param_vector

        ##### save checkpoint
        policy_file = 'policy_{}.pickle'.format(str(round_id).zfill(4))
        baseline_file = 'baseline_{}.pickle'.format(str(round_id).zfill(4))
        policy_file = os.path.join(log_dir, 'iterations', policy_file)
        baseline_file = os.path.join(log_dir, 'iterations', baseline_file)
        pickle.dump(agent.policy, open(policy_file, 'wb'))
        pickle.dump(agent.baseline, open(baseline_file, 'wb'))
        round_id_file = os.path.join(log_dir, 'iterations', 'round_{}.txt'.format(str(round_id).zfill(4)))
        with open(round_id_file, 'w') as f:
            f.write('{}'.format(interp_param_vector.tolist()))
        ##### save checkpoint

        round_id += 1

        if final_round:
            log_string("...........................................................")
            log_string("End of training of {}-{}".format(start_robot_name, end_robot_name))
            break

log_string('Finished')


