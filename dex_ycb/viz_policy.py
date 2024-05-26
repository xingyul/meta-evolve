
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

import mujoco_py


parser = argparse.ArgumentParser(description='Natural policy gradient from mjrl on mujoco environments')

parser.add_argument('--exp_dir', type=str, default='log_generalized-ycb-v2-unified__008_pudding_box_20200918_114346__steiner__l1__seed_1005', help='experiment root directory')
parser.add_argument('--path_name', type=str, required=True, \
        choices=['log__original__steiner_1', 'log__steiner_1__steiner_2', 'log__steiner_2__iiwa', 'log__steiner_2__kinova', 'log__steiner_1__jaco'], \
        default='log__steiner_1__jaco', help='name of the path in the evolution tree')
parser.add_argument('--round_idx', type=int, required=True, default=218, help='the step index of the policy inside the path to show')
args = parser.parse_args()


env_name = 'generalized-ycb-v2-unified'


log_dir = os.path.join(args.exp_dir, args.path_name)
round_idx = str(args.round_idx).zfill(4)


ycb_scene_name = '008_pudding_box_20200918_114346'
ignore_useless_obj = []
obj_armature = 0

cam_distance = 1.
elevation = -25
lookat = [0, -0.2, 0.2]
azimuth = 225


current_dir = os.path.dirname(os.path.abspath(__file__))
ycb_scene_dir = os.path.join(current_dir, ycb_scene_name)
current_curri_t = 0

round_idx_file = os.path.join(log_dir, 'iterations', 'round_' + round_idx + '.txt')

with open(round_idx_file, 'r') as f:
    interp_param_vector = f.read()
    interp_param_vector = ast.literal_eval(interp_param_vector)
interp_param_vector = np.array(interp_param_vector)

with open('hand_manipulation_suite/phys_param_vector_all.json', 'r') as f:
    phys_param_vector_all = json.load(f)

frame_skip = 1
horizon = 1000 // frame_skip

env_kwargs = {'phys_param_vector_all': phys_param_vector_all, \
    'ycb_scene_dir': ycb_scene_dir, 'simulation_mode': True, \
    'finger_option': 'ff', 'lf_shrink': 0.7, 'virtual_confine': True, \
    'horizon': horizon}
env = gym.make(env_name, \
        interp_param_vector=interp_param_vector, \
        dense_reward=False, **env_kwargs)

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

