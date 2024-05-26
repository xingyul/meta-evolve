
import numpy as np
import pickle
import shutil
import json
import glob
import os
import argparse
import cv2
import sys
import ast
import torch
import pyquaternion
import copy
import time
import gym

import hand_manipulation_suite



parser = argparse.ArgumentParser(description='Natural policy gradient from mjrl on mujoco environments')

parser.add_argument('--env_name', required=True, type=str, default='generalized-hammer-v0', help='name of the task')
parser.add_argument('--exp_dir', type=str, required=True, default='log_generalized-hammer-v0_steiner__l1__seed_1025', help='experiment root directory')
parser.add_argument('--path_name', type=str, required=True, \
        choices=['log_original_steiner_1', 'log_steiner_1_four_finger', 'log_steiner_1_steiner_2', 'log_steiner_2_three_finger', 'log_steiner_2_two_finger'], \
        default='log_steiner_2_two_finger', help='name of the path in the evolution tree')
parser.add_argument('--round_idx', type=int, required=True, default=383, help='the step index of the policy inside the path to show')
args = parser.parse_args()


env_name = args.env_name


log_dir = os.path.join(args.exp_dir, args.path_name)
round_idx = str(args.round_idx).zfill(4)



cam_distance = 1.
elevation = -25
lookat = [0, -0.2, 0.2]
azimuth = 225


round_idx_file = os.path.join(log_dir, 'iterations', 'round_' + round_idx + '.txt')

with open(round_idx_file, 'r') as f:
    interp_param_vector = f.read()
    interp_param_vector = ast.literal_eval(interp_param_vector)
interp_param_vector = np.array(interp_param_vector)


frame_skip = 1
horizon = 1000 // frame_skip

env_kwargs = {'id': env_name, \
    'interp_param_vector': interp_param_vector, \
    'dense_reward': False, }
env = gym.make(**env_kwargs)

env.reset(seed=0)
env.frame_skip = frame_skip

xml_string = env.env.sim.model.get_xml()
with open('tmp.xml', 'w') as f:
    f.write(xml_string)

policy = pickle.load(open(os.path.join(log_dir, 'iterations', 'policy_{}.pickle'.format(round_idx)), 'rb'))
policy.to(torch.device('cpu'))

env.horizon = horizon
env.env.horizon = horizon


for _ in range(30):
    seed = int(time.time() * 100) % (2**20)
    obs = env.reset(seed=seed)
    score = 0.0
    d = False
    goal_achieved = False
    t = 0
    while t < env.env.horizon:

        action = policy.get_action(obs)[1]['evaluation']

        obs, r, d, g = env.step(action)
        if g['goal_achieved']:
            goal_achieved = True
        score = score + r

        env.render()

        t += 1

    print("Episode score = {}, goal achieved: {}".format(score, goal_achieved))

