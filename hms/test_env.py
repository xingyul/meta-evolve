

from collections import OrderedDict
import torch
import glob
import gym
import numpy as np
import copy
import os
import cv2
import json
import pickle
import pyquaternion
import time
import matplotlib.pyplot as plt
import jsbeautifier

import hand_manipulation_suite
import utils.rl_env_expansion as rl_env_expansion

opts = jsbeautifier.default_options()
opts.indent_size = 4

target_robot = 'original'
# target_robot = 'two_finger'
# target_robot = 'three_finger'
# target_robot = 'four_finger'


preview = 0
preview = 1

interp_param = 1.
generalized_env = 'generalized-relocate-v0'
policy_file = 'log_relocate-v0/iterations/policy_3999.pickle'
generalized_env = 'generalized-hammer-v0'
policy_file = 'log_hammer-v0/iterations/policy_2000.pickle'
# generalized_env = 'generalized-door-v0'
# policy_file = 'log_door-v0/iterations/policy_3999.pickle'

policy = pickle.load(open(policy_file, 'rb'))
policy.to(torch.device('cpu'))



interp_param_dicts = {}


interp_param_dict = OrderedDict()
interp_param_dict['finger_len_th_proximal'] = 1
interp_param_dict['finger_len_th_middle'] = 1
interp_param_dict['finger_len_th_distal'] = 1
interp_param_dict['finger_len_mf_proximal'] = 1
interp_param_dict['finger_len_mf_middle'] = 1
interp_param_dict['finger_len_mf_distal'] = 1
interp_param_dict['finger_len_rf_proximal'] = 1
interp_param_dict['finger_len_rf_middle'] = 1
interp_param_dict['finger_len_rf_distal'] = 1
interp_param_dict['finger_len_lf_proximal'] = 1
interp_param_dict['finger_len_lf_middle'] = 1
interp_param_dict['finger_len_lf_distal'] = 1
interp_param_dict['finger_width_th_proximal'] = 1
interp_param_dict['finger_width_th_middle'] = 1
interp_param_dict['finger_width_th_distal'] = 1
interp_param_dict['finger_width_ff_proximal'] = 1
interp_param_dict['finger_width_ff_middle'] = 1
interp_param_dict['finger_width_ff_distal'] = 1
interp_param_dict['finger_width_mf_proximal'] = 1
interp_param_dict['finger_width_mf_middle'] = 1
interp_param_dict['finger_width_mf_distal'] = 1
interp_param_dict['finger_width_rf_proximal'] = 1
interp_param_dict['finger_width_rf_middle'] = 1
interp_param_dict['finger_width_rf_distal'] = 1
interp_param_dict['finger_width_lf_proximal'] = 1
interp_param_dict['finger_width_lf_middle'] = 1
interp_param_dict['finger_width_lf_distal'] = 1
interp_param_dict['joint_range_th_4'] = 1
interp_param_dict['joint_range_th_3'] = 1
interp_param_dict['joint_range_th_2'] = 1
interp_param_dict['joint_range_th_1'] = 1
interp_param_dict['joint_range_th_0'] = 1
interp_param_dict['joint_range_ff_3'] = 1
interp_param_dict['joint_range_ff_2'] = 1
interp_param_dict['joint_range_ff_1'] = 1
interp_param_dict['joint_range_ff_0'] = 1
interp_param_dict['joint_range_mf_3'] = 1
interp_param_dict['joint_range_mf_2'] = 1
interp_param_dict['joint_range_mf_1'] = 1
interp_param_dict['joint_range_mf_0'] = 1
interp_param_dict['joint_range_rf_3'] = 1
interp_param_dict['joint_range_rf_2'] = 1
interp_param_dict['joint_range_rf_1'] = 1
interp_param_dict['joint_range_rf_0'] = 1
interp_param_dict['joint_range_lf_4'] = 1
interp_param_dict['joint_range_lf_3'] = 1
interp_param_dict['joint_range_lf_2'] = 1
interp_param_dict['joint_range_lf_1'] = 1
interp_param_dict['joint_range_lf_0'] = 1
interp_param_dict['knuckle_angle_th'] = 1
interp_param_dict['knuckle_pos_th'] = 1
interp_param_dict['knuckle_pos_ff_x'] = 1
interp_param_dict['knuckle_pos_ff_z'] = 1
interp_param_dict['knuckle_angle_ff'] = 1
interp_param_dict['knuckle_pos_mf'] = 1
interp_param_dict['knuckle_pos_rf'] = 1
interp_param_dict['knuckle_pos_lf'] = 1
interp_param_dict['knuckle_angle_lf'] = 1
interp_param_dict['palm'] = 1


# original
interp_param_dict_copy = interp_param_dict.copy()
for k in interp_param_dict_copy.keys():
    interp_param_dict_copy[k] = 0
interp_param_dicts['original'] = interp_param_dict_copy

# two-finger robot
interp_param_dict_copy = interp_param_dict.copy()
interp_param_dict_copy['knuckle_angle_ff'] = 0
interp_param_dict_copy['knuckle_angle_lf'] = 0
interp_param_dicts['two_finger'] = interp_param_dict_copy

# three-finger robot
interp_param_dict_copy = interp_param_dict.copy()
interp_param_dict_copy['knuckle_pos_ff_x'] = 0
interp_param_dict_copy['finger_len_lf_proximal'] = 0.
interp_param_dict_copy['finger_len_lf_middle'] = 0.
interp_param_dict_copy['finger_len_lf_distal'] = 0.
interp_param_dict_copy['finger_width_lf_proximal'] = 1 - 0.007 / 0.01
interp_param_dict_copy['finger_width_lf_middle'] = 1 - 0.007 / 0.00805
interp_param_dict_copy['finger_width_lf_distal'] = 1 - 0.007 / 0.00705
interp_param_dicts['three_finger'] = interp_param_dict_copy

# four-finger robot
interp_param_dict_copy = interp_param_dict.copy()
interp_param_dict_copy['knuckle_pos_ff_x'] = 0
interp_param_dict_copy['finger_len_lf_proximal'] = 0.
interp_param_dict_copy['finger_len_lf_middle'] = 0.
interp_param_dict_copy['finger_len_lf_distal'] = 0.
interp_param_dict_copy['finger_width_lf_proximal'] = 1 - 0.007 / 0.01
interp_param_dict_copy['finger_width_lf_middle'] = 1 - 0.007 / 0.00805
interp_param_dict_copy['finger_width_lf_distal'] = 1 - 0.007 / 0.00705
interp_param_dict_copy['finger_len_mf_proximal'] = 0.
interp_param_dict_copy['finger_len_mf_middle'] = 0.
interp_param_dict_copy['finger_len_mf_distal'] = 0.
interp_param_dict_copy['finger_width_mf_proximal'] = 1 - 0.007 / 0.01
interp_param_dict_copy['finger_width_mf_middle'] = 1 - 0.007 / 0.00805
interp_param_dict_copy['finger_width_mf_distal'] = 1 - 0.007 / 0.00705
interp_param_dicts['four_finger'] = interp_param_dict_copy



interp_param_dict_full = {}
for env_name in ['relocate', 'hammer', 'door']:
    interp_param_save_dict = {}
    for target_r in interp_param_dicts:
        interp_param_dict = interp_param_dicts[target_r]
        interp_param_vector = []
        for k in list(interp_param_dict.keys()):
            interp_param_vector.append(interp_param_dict[k])
        interp_param_save_dict[target_r] = interp_param_vector
    interp_param_dict_full[env_name] = interp_param_save_dict

    dump_filename = os.path.join('hand_manipulation_suite', 'interp_params_dict.json')
    with open(dump_filename, 'w') as f:
        f.write(jsbeautifier.beautify(json.dumps(interp_param_save_dict), opts))


interp_param_vector = interp_param_dict_full[generalized_env.split('-')[1]][target_robot]


env = gym.make(generalized_env, \
        interp_param_vector=interp_param_vector, \
        dense_reward=False)


xml_string = env.model.get_xml()
with open('tmp.xml', 'w') as f:
    f.write(xml_string)


'''
env.reset()
while True:
    env.sim.data.qpos[env.sim.model.get_joint_qpos_addr('ARTz')] += 0.001
    env.sim.forward()
    env.render()
'''


for _ in range(5):
    o = env.reset()
    d = False
    t = 0
    score = 0.0
    goal = 0
    # while True:
    for i in range(200):
        a = policy.get_action(o)[1]['evaluation']
        # a[0] = 1
        o, r, d, g = env.step(a)
        if preview:
            env.render()
        if g['goal_achieved']:
            goal = goal + 1
        score = score + r
        t = t + 1
    print("Episode score = {}, goal achieved: {}".format(score, goal))

