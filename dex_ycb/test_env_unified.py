

from collections import OrderedDict
import torch
import glob
import gym
import numpy as np
import copy
import os
import json
import pickle
import pyquaternion
import time

import hand_manipulation_suite
import utils.rl_env_expansion as rl_env_expansion


azimuth = -45
distance = 1.1
lookat = [0, -0., 0.2]
elevation = -45


preview = 0
preview = 1

generalized_env = 'generalized-ycb-v2-unified'

policy_file = 'log_expert_008_pudding_box_20200918_114346/iterations/policy_0000.pickle'
ycb_scene_dir='/home/xyl/Projects/meta-evolve/dex_ycb/008_pudding_box_20200918_114346'

num_obs_expansion = 1
num_act_expansion = 7

policy = pickle.load(open(policy_file, 'rb'))
policy.to(torch.device('cpu'))

policy = rl_env_expansion.policy_expansion(policy, \
        num_obs_expansion=num_obs_expansion, \
        num_act_expansion=num_act_expansion)

##### jaco params
phys_param_dict_jaco = OrderedDict()
phys_param_dict_jaco['finger_len_th_proximal'] = [1, 1] # 0.044/0.038]
phys_param_dict_jaco['finger_len_th_middle'] = [1, 0.035/0.032]
phys_param_dict_jaco['finger_len_th_distal'] = [1, 0.01]
phys_param_dict_jaco['finger_len_ff_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_jaco['finger_len_ff_middle'] = [1, 0.035/0.025]
phys_param_dict_jaco['finger_len_ff_distal'] = [1, 0.01]
phys_param_dict_jaco['finger_len_mf_proximal'] = [1, 0.01] # 0.044/0.045]
phys_param_dict_jaco['finger_len_mf_middle'] = [1, 0.01]
phys_param_dict_jaco['finger_len_mf_distal'] = [1, 0.01]
phys_param_dict_jaco['finger_len_rf_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_jaco['finger_len_rf_middle'] = [1, 1]
phys_param_dict_jaco['finger_len_rf_distal'] = [1, 0.01]
phys_param_dict_jaco['finger_len_lf_proximal'] = [1, 0.01] # 0.044/0.045]
phys_param_dict_jaco['finger_len_lf_middle'] = [1, 0.01]
phys_param_dict_jaco['finger_len_lf_distal'] = [1, 0.01]
phys_param_dict_jaco['joint_range_th_4_upper'] = [1.047, 1e-5]
phys_param_dict_jaco['joint_range_th_4_lower'] = [-1.047, -1e-5]
phys_param_dict_jaco['joint_range_th_3_upper'] = [1.309, 1e-5]
phys_param_dict_jaco['joint_range_th_3_lower'] = [0, -1e-5]
phys_param_dict_jaco['joint_range_th_3_position_upper'] = [1e-5, np.pi/4*3]
phys_param_dict_jaco['joint_range_th_3_position_lower'] = [0, np.pi/6+np.pi/24]
phys_param_dict_jaco['joint_range_th_2_upper'] = [0.262, 0]
phys_param_dict_jaco['joint_range_th_2_lower'] = [-0.262, -np.pi/2]
phys_param_dict_jaco['joint_range_th_2_position_upper'] = [1e-5, 1e-5]
phys_param_dict_jaco['joint_range_th_2_position_lower'] = [0, 0]
phys_param_dict_jaco['actuation_th_2'] = [1, 1e-5]
phys_param_dict_jaco['joint_range_th_1_upper'] = [0.524, 1e-5]
phys_param_dict_jaco['joint_range_th_1_lower'] = [-0.524, -1e-5]
phys_param_dict_jaco['joint_range_th_0_upper'] = [0, 1e-5]
phys_param_dict_jaco['joint_range_th_0_lower'] = [-1.571, -1e-5]
phys_param_dict_jaco['joint_range_ff_3_upper'] = [0.436, 1e-5]
phys_param_dict_jaco['joint_range_ff_3_lower'] = [-0.436, -1e-5]
phys_param_dict_jaco['joint_range_ff_2_upper'] = [1.571, 1e-5]
phys_param_dict_jaco['joint_range_ff_2_lower'] = [0, -1e-5]
phys_param_dict_jaco['joint_range_ff_2_position_upper'] = [1e-5, np.pi/6*5-np.pi/24]
phys_param_dict_jaco['joint_range_ff_2_position_lower'] = [0, np.pi/4]
phys_param_dict_jaco['joint_range_ff_1_upper'] = [1.571, np.pi/2]
phys_param_dict_jaco['joint_range_ff_1_lower'] = [0, 0]
phys_param_dict_jaco['joint_range_ff_1_position_upper'] = [1e-5, 1e-5]
phys_param_dict_jaco['joint_range_ff_1_position_lower'] = [0, 0]
phys_param_dict_jaco['actuation_ff_1'] = [1, 1e-5]
phys_param_dict_jaco['joint_range_ff_0_upper'] = [1.571, 1e-5]
phys_param_dict_jaco['joint_range_ff_0_lower'] = [0, -1e-5]
phys_param_dict_jaco['joint_range_rf_3_upper'] = [0.436, 1e-5]
phys_param_dict_jaco['joint_range_rf_3_lower'] = [-0.436, -1e-5]
phys_param_dict_jaco['joint_range_rf_2_upper'] = [1.571, 1e-5]
phys_param_dict_jaco['joint_range_rf_2_lower'] = [0, -1e-5]
phys_param_dict_jaco['joint_range_rf_2_position_upper'] = [1e-5, np.pi/6*5]
phys_param_dict_jaco['joint_range_rf_2_position_lower'] = [0, np.pi/4]
phys_param_dict_jaco['joint_range_rf_1_upper'] = [1.571, 1.571]
phys_param_dict_jaco['joint_range_rf_1_lower'] = [0, -0.125]
phys_param_dict_jaco['actuation_rf_1'] = [1, 1e-5]
phys_param_dict_jaco['joint_range_rf_0_upper'] = [1.571, 1e-5]
phys_param_dict_jaco['joint_range_rf_0_lower'] = [0, -1e-5]
phys_param_dict_jaco['shape_th_jaco'] = [0, 1]
phys_param_dict_jaco['shape_th_kinova'] = [0, 0]
phys_param_dict_jaco['shape_th_iiwa'] = [0, 0]
phys_param_dict_jaco['shape_ff_jaco'] = [0, 1]
phys_param_dict_jaco['shape_ff_kinova'] = [0, 0]
phys_param_dict_jaco['shape_ff_iiwa'] = [0, 0]
phys_param_dict_jaco['shape_mf'] = [0, 1]
phys_param_dict_jaco['shape_rf'] = [0, 0.7]
phys_param_dict_jaco['shape_rf_jaco'] = [0, 1]
phys_param_dict_jaco['shape_lf'] = [0, 1]
phys_param_dict_jaco['knuckle_pos_th_x'] = [0.034, 0.00841201]
phys_param_dict_jaco['knuckle_pos_th_y'] = [-0.009, -0.114668]
phys_param_dict_jaco['knuckle_pos_th_z'] = [0.029, -0.03023553]
phys_param_dict_jaco['knuckle_pos_ff_x'] = [0.033, 0.01697403]
phys_param_dict_jaco['knuckle_pos_ff_y'] = [0, -0.114668]
phys_param_dict_jaco['knuckle_pos_ff_z'] = [0.095, 0.03066202]
phys_param_dict_jaco['knuckle_pos_mf_x'] = [0.011, 0.0]
phys_param_dict_jaco['knuckle_pos_mf_y'] = [0, 0]
phys_param_dict_jaco['knuckle_pos_mf_z'] = [0.099, 0.027073]
phys_param_dict_jaco['knuckle_pos_rf_x'] = [-0.011, -0.02679905]
phys_param_dict_jaco['knuckle_pos_rf_y'] = [0, -0.114668]
phys_param_dict_jaco['knuckle_pos_rf_z'] = [0.095, 0.02258513]
# phys_param_dict_jaco['knuckle_pos_th_q0'] = [0.923956, 0.95922726]
# phys_param_dict_jaco['knuckle_pos_th_q1'] = [0, 0.262085]
# phys_param_dict_jaco['knuckle_pos_th_q2'] = [0.382499, 0.10213274]
# phys_param_dict_jaco['knuckle_pos_th_q3'] = [0, 0.02762996]
phys_param_dict_jaco['knuckle_pos_th_euler_x'] = [0, 0.5433716158535712]
phys_param_dict_jaco['knuckle_pos_th_euler_y'] = [0.7849990259705161, 0] # 0.18246501911627286
phys_param_dict_jaco['knuckle_pos_th_euler_z'] = [0, 0.10855332395269136]
# phys_param_dict_jaco['knuckle_pos_ff_q0'] = [1, 0.96181018]
# phys_param_dict_jaco['knuckle_pos_ff_q1'] = [0, -0.25771638]
# phys_param_dict_jaco['knuckle_pos_ff_q2'] = [0, 0.08907205]
# phys_param_dict_jaco['knuckle_pos_ff_q3'] = [0, 0.0238668]
phys_param_dict_jaco['knuckle_pos_ff_euler_x'] = [0, -0.5235990101711093]
phys_param_dict_jaco['knuckle_pos_ff_euler_y'] = [0, 0]
phys_param_dict_jaco['knuckle_pos_ff_euler_z'] = [0, 0]
# phys_param_dict_jaco['knuckle_pos_rf_q0'] = [1, 0.96181018]
# phys_param_dict_jaco['knuckle_pos_rf_q1'] = [0, -0.25771638]
# phys_param_dict_jaco['knuckle_pos_rf_q2'] = [0, -0.08907205]
# phys_param_dict_jaco['knuckle_pos_rf_q3'] = [0, -0.0238668]
phys_param_dict_jaco['knuckle_pos_rf_euler_x'] = [0, -0.5235990101711093]
phys_param_dict_jaco['knuckle_pos_rf_euler_y'] = [0, -0.36938147514]
phys_param_dict_jaco['knuckle_pos_rf_euler_z'] = [0, 0]
phys_param_dict_jaco['palm_jaco'] = [0, 1]
phys_param_dict_jaco['palm_kinova'] = [0, 0]
phys_param_dict_jaco['palm_iiwa'] = [0, 0]
phys_param_dict_jaco['wrist'] = [0, 1]
phys_param_dict_jaco['arm_length'] = [0, 1]
phys_param_dict_jaco['arm_range'] = [0, 1]
phys_param_dict_jaco['robot_arm_body_1'] = [0.15675, 0.15675]
phys_param_dict_jaco['robot_arm_body_2'] = [0.11875, 0.11875]
phys_param_dict_jaco['robot_arm_body_3'] = [0.205, 0.205]
phys_param_dict_jaco['robot_arm_body_4'] = [0.205, 0.205]
phys_param_dict_jaco['robot_arm_body_5'] = [0.2073, 0.2073]
phys_param_dict_jaco['robot_arm_body_5_y'] = [-0.0114, -0.0114]
phys_param_dict_jaco['robot_arm_body_6'] = [0.10375, 0.10375]
phys_param_dict_jaco['robot_arm_body_7'] = [0.10375, 0.10375]
phys_param_dict_jaco['robot_arm_body_hand'] = [-0.2, 0]
phys_param_dict_jaco['robot_arm_shape_jaco'] = [0, 1]
phys_param_dict_jaco['robot_arm_shape_kinova'] = [0, 0]
phys_param_dict_jaco['robot_arm_shape_iiwa'] = [0, 0]
phys_param_dict_jaco['tendon_stiffness_th'] = [0, 0]
phys_param_dict_jaco['tendon_stiffness_ff'] = [0, 0]
phys_param_vector_jaco = []
for k in list(phys_param_dict_jaco.keys()):
    phys_param_vector_jaco.append(phys_param_dict_jaco[k])



##### kinova params
phys_param_dict_kinova = OrderedDict()
phys_param_dict_kinova['finger_len_th_proximal'] = [1, 0.05712057422/0.038] # 0.044/0.038]
phys_param_dict_kinova['finger_len_th_middle'] = [1, 0.035/0.032]
phys_param_dict_kinova['finger_len_th_distal'] = [1, 0.01]
phys_param_dict_kinova['finger_len_ff_proximal'] = [1, 0.05712057422/0.045] # 0.044/0.045]
phys_param_dict_kinova['finger_len_ff_middle'] = [1, 0.035/0.025]
phys_param_dict_kinova['finger_len_ff_distal'] = [1, 0.01]
phys_param_dict_kinova['finger_len_mf_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_kinova['finger_len_mf_middle'] = [1, 1]
phys_param_dict_kinova['finger_len_mf_distal'] = [1, 0.01]
phys_param_dict_kinova['finger_len_rf_proximal'] = [1, 0.01] # 0.044/0.045]
phys_param_dict_kinova['finger_len_rf_middle'] = [1, 0.01]
phys_param_dict_kinova['finger_len_rf_distal'] = [1, 0.01]
phys_param_dict_kinova['finger_len_lf_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_kinova['finger_len_lf_middle'] = [1, 1]
phys_param_dict_kinova['finger_len_lf_distal'] = [1, 0.01]
phys_param_dict_kinova['joint_range_th_4_upper'] = [1.047, 1e-5]
phys_param_dict_kinova['joint_range_th_4_lower'] = [-1.047, -1e-5]
phys_param_dict_kinova['joint_range_th_3_upper'] = [1.309, 1e-5]
phys_param_dict_kinova['joint_range_th_3_lower'] = [0, -1e-5]
phys_param_dict_kinova['joint_range_th_3_position_upper'] = [1e-5, np.pi/4*3]
phys_param_dict_kinova['joint_range_th_3_position_lower'] = [0, np.pi/2-np.pi/16]
phys_param_dict_kinova['joint_range_th_2_upper'] = [0.262, 1e-5]
phys_param_dict_kinova['joint_range_th_2_lower'] = [-0.262, -1e-5]
phys_param_dict_kinova['joint_range_th_2_position_upper'] = [1e-5, np.pi/16]
phys_param_dict_kinova['joint_range_th_2_position_lower'] = [0, -np.pi/4]
phys_param_dict_kinova['actuation_th_2'] = [1, 1]
phys_param_dict_kinova['joint_range_th_1_upper'] = [0.524, 1e-5]
phys_param_dict_kinova['joint_range_th_1_lower'] = [-0.524, -1e-5]
phys_param_dict_kinova['joint_range_th_0_upper'] = [0, 1e-5]
phys_param_dict_kinova['joint_range_th_0_lower'] = [-1.571, -1e-5]
phys_param_dict_kinova['joint_range_ff_3_upper'] = [0.436, 1e-5]
phys_param_dict_kinova['joint_range_ff_3_lower'] = [-0.436, -1e-5]
phys_param_dict_kinova['joint_range_ff_2_upper'] = [1.571, 1e-5]
phys_param_dict_kinova['joint_range_ff_2_lower'] = [0, -1e-5]
phys_param_dict_kinova['joint_range_ff_2_position_upper'] = [1e-5, np.pi/2+np.pi/16]
phys_param_dict_kinova['joint_range_ff_2_position_lower'] = [0, np.pi/4]
phys_param_dict_kinova['joint_range_ff_1_upper'] = [1.571, 1e-5]
phys_param_dict_kinova['joint_range_ff_1_lower'] = [0, -1e-5]
phys_param_dict_kinova['joint_range_ff_1_position_upper'] = [1e-5, np.pi/4]
phys_param_dict_kinova['joint_range_ff_1_position_lower'] = [0, -np.pi/16]
phys_param_dict_kinova['actuation_ff_1'] = [1, 1]
phys_param_dict_kinova['joint_range_ff_0_upper'] = [1.571, 1e-5]
phys_param_dict_kinova['joint_range_ff_0_lower'] = [0, -1e-5]
phys_param_dict_kinova['joint_range_rf_3_upper'] = [0.436, 0.436]
phys_param_dict_kinova['joint_range_rf_3_lower'] = [-0.436, -0.436]
phys_param_dict_kinova['joint_range_rf_2_upper'] = [1.571, 1e-5]
phys_param_dict_kinova['joint_range_rf_2_lower'] = [0, -1e-5]
phys_param_dict_kinova['joint_range_rf_2_position_upper'] = [1e-5, np.pi/3]
phys_param_dict_kinova['joint_range_rf_2_position_lower'] = [0, np.pi/8]
phys_param_dict_kinova['joint_range_rf_1_upper'] = [1.571, 1.571]
phys_param_dict_kinova['joint_range_rf_1_lower'] = [0, -0.125]
phys_param_dict_kinova['actuation_rf_1'] = [1, 1e-5]
phys_param_dict_kinova['joint_range_rf_0_upper'] = [1.571, 1.571]
phys_param_dict_kinova['joint_range_rf_0_lower'] = [0, 0]
phys_param_dict_kinova['shape_th_jaco'] = [0, 0]
phys_param_dict_kinova['shape_th_kinova'] = [0, 1]
phys_param_dict_kinova['shape_th_iiwa'] = [0, 0]
phys_param_dict_kinova['shape_ff_jaco'] = [0, 0]
phys_param_dict_kinova['shape_ff_kinova'] = [0, 1]
phys_param_dict_kinova['shape_ff_iiwa'] = [0, 0]
phys_param_dict_kinova['shape_mf'] = [0, 1]
phys_param_dict_kinova['shape_rf'] = [0, 1]
phys_param_dict_kinova['shape_rf_jaco'] = [0, 0]
phys_param_dict_kinova['shape_lf'] = [0, 1]
phys_param_dict_kinova['knuckle_pos_th_x'] = [0.034, 0]
phys_param_dict_kinova['knuckle_pos_th_y'] = [-0.009, -0.06142]
phys_param_dict_kinova['knuckle_pos_th_z'] = [0.029, -0.0127]
phys_param_dict_kinova['knuckle_pos_ff_x'] = [0.033, 0]
phys_param_dict_kinova['knuckle_pos_ff_y'] = [0, -0.06142]
phys_param_dict_kinova['knuckle_pos_ff_z'] = [0.095, 0.0127]
phys_param_dict_kinova['knuckle_pos_mf_x'] = [0.011, 0.0]
phys_param_dict_kinova['knuckle_pos_mf_y'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_mf_z'] = [0.099, 0.027073]
phys_param_dict_kinova['knuckle_pos_rf_x'] = [-0.011, -0.022256]
phys_param_dict_kinova['knuckle_pos_rf_y'] = [0, -0.06142]
phys_param_dict_kinova['knuckle_pos_rf_z'] = [0.095, 0.027073]
# phys_param_dict_kinova['knuckle_pos_th_q0'] = [0.923956, 1]
# phys_param_dict_kinova['knuckle_pos_th_q1'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_th_q2'] = [0.382499, 0]
# phys_param_dict_kinova['knuckle_pos_th_q3'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_th_euler_x'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_th_euler_y'] = [0.7849990259705161, 0]
phys_param_dict_kinova['knuckle_pos_th_euler_z'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_ff_q0'] = [1, 1]
# phys_param_dict_kinova['knuckle_pos_ff_q1'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_ff_q2'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_ff_q3'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_ff_euler_x'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_ff_euler_y'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_ff_euler_z'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_rf_q0'] = [1, 1]
# phys_param_dict_kinova['knuckle_pos_rf_q1'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_rf_q2'] = [0, 0]
# phys_param_dict_kinova['knuckle_pos_rf_q3'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_rf_euler_x'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_rf_euler_y'] = [0, 0]
phys_param_dict_kinova['knuckle_pos_rf_euler_z'] = [0, 0]
phys_param_dict_kinova['palm_jaco'] = [0, 0]
phys_param_dict_kinova['palm_kinova'] = [0, 1]
phys_param_dict_kinova['palm_iiwa'] = [0, 0]
phys_param_dict_kinova['wrist'] = [0, 1]
phys_param_dict_kinova['arm_length'] = [0, 1]
phys_param_dict_kinova['arm_range'] = [0, 1]
phys_param_dict_kinova['robot_arm_body_1'] = [0.15675, 0.15643]
phys_param_dict_kinova['robot_arm_body_2'] = [0.11875, 0.12838]
phys_param_dict_kinova['robot_arm_body_3'] = [0.205, 0.21038]
phys_param_dict_kinova['robot_arm_body_4'] = [0.205, 0.21038]
phys_param_dict_kinova['robot_arm_body_5'] = [0.2073, 0.20843]
phys_param_dict_kinova['robot_arm_body_5_y'] = [-0.0114, -0.006375]
phys_param_dict_kinova['robot_arm_body_6'] = [0.10375, 0.10593]
phys_param_dict_kinova['robot_arm_body_7'] = [0.10375, 0.10593]
phys_param_dict_kinova['robot_arm_body_hand'] = [-0.2, -0.065]
phys_param_dict_kinova['robot_arm_shape_jaco'] = [0, 0]
phys_param_dict_kinova['robot_arm_shape_kinova'] = [0, 1]
phys_param_dict_kinova['robot_arm_shape_iiwa'] = [0, 0]
phys_param_dict_kinova['tendon_stiffness_th'] = [0, 1000]
phys_param_dict_kinova['tendon_stiffness_ff'] = [0, 1000]
phys_param_vector_kinova = []
for k in list(phys_param_dict_kinova.keys()):
    phys_param_vector_kinova.append(phys_param_dict_kinova[k])


##### iiwa params
phys_param_dict_iiwa = OrderedDict()
phys_param_dict_iiwa['finger_len_th_proximal'] = [1, 0.1/0.038] # 0.044/0.038]
phys_param_dict_iiwa['finger_len_th_middle'] = [1, 0.035/0.032]
phys_param_dict_iiwa['finger_len_th_distal'] = [1, 0.01]
phys_param_dict_iiwa['finger_len_ff_proximal'] = [1, 0.1/0.045] # 0.044/0.045]
phys_param_dict_iiwa['finger_len_ff_middle'] = [1, 0.035/0.025]
phys_param_dict_iiwa['finger_len_ff_distal'] = [1, 0.01]
phys_param_dict_iiwa['finger_len_mf_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_iiwa['finger_len_mf_middle'] = [1, 1]
phys_param_dict_iiwa['finger_len_mf_distal'] = [1, 0.01]
phys_param_dict_iiwa['finger_len_rf_proximal'] = [1, 0.01] # 0.044/0.045]
phys_param_dict_iiwa['finger_len_rf_middle'] = [1, 0.01]
phys_param_dict_iiwa['finger_len_rf_distal'] = [1, 0.01]
phys_param_dict_iiwa['finger_len_lf_proximal'] = [1, 1] # 0.044/0.045]
phys_param_dict_iiwa['finger_len_lf_middle'] = [1, 1]
phys_param_dict_iiwa['finger_len_lf_distal'] = [1, 0.01]
phys_param_dict_iiwa['joint_range_th_4_upper'] = [1.047, 1e-5]
phys_param_dict_iiwa['joint_range_th_4_lower'] = [-1.047, -1e-5]
phys_param_dict_iiwa['joint_range_th_3_upper'] = [1.309, 1e-5]
phys_param_dict_iiwa['joint_range_th_3_lower'] = [0, -1e-5]
phys_param_dict_iiwa['joint_range_th_3_position_upper'] = [1e-5, np.pi/4*3]
phys_param_dict_iiwa['joint_range_th_3_position_lower'] = [0, np.pi/2-np.pi/32]
phys_param_dict_iiwa['joint_range_th_2_upper'] = [0.262, 1e-5]
phys_param_dict_iiwa['joint_range_th_2_lower'] = [-0.262, -1e-5]
phys_param_dict_iiwa['joint_range_th_2_position_upper'] = [1e-5, np.pi/32]
phys_param_dict_iiwa['joint_range_th_2_position_lower'] = [0, -np.pi/4]
phys_param_dict_iiwa['actuation_th_2'] = [1, 1]
phys_param_dict_iiwa['joint_range_th_1_upper'] = [0.524, 1e-5]
phys_param_dict_iiwa['joint_range_th_1_lower'] = [-0.524, -1e-5]
phys_param_dict_iiwa['joint_range_th_0_upper'] = [0, 1e-5]
phys_param_dict_iiwa['joint_range_th_0_lower'] = [-1.571, -1e-5]
phys_param_dict_iiwa['joint_range_ff_3_upper'] = [0.436, 1e-5]
phys_param_dict_iiwa['joint_range_ff_3_lower'] = [-0.436, -1e-5]
phys_param_dict_iiwa['joint_range_ff_2_upper'] = [1.571, 1e-5]
phys_param_dict_iiwa['joint_range_ff_2_lower'] = [0, -1e-5]
phys_param_dict_iiwa['joint_range_ff_2_position_upper'] = [1e-5, np.pi/2+np.pi/32]
phys_param_dict_iiwa['joint_range_ff_2_position_lower'] = [0, np.pi/4]
phys_param_dict_iiwa['joint_range_ff_1_upper'] = [1.571, 1e-5]
phys_param_dict_iiwa['joint_range_ff_1_lower'] = [0, -1e-5]
phys_param_dict_iiwa['joint_range_ff_1_position_upper'] = [1e-5, np.pi/4]
phys_param_dict_iiwa['joint_range_ff_1_position_lower'] = [0, -np.pi/32]
phys_param_dict_iiwa['actuation_ff_1'] = [1, 1]
phys_param_dict_iiwa['joint_range_ff_0_upper'] = [1.571, 1e-5]
phys_param_dict_iiwa['joint_range_ff_0_lower'] = [0, -1e-5]
phys_param_dict_iiwa['joint_range_rf_3_upper'] = [0.436, 0.436]
phys_param_dict_iiwa['joint_range_rf_3_lower'] = [-0.436, -0.436]
phys_param_dict_iiwa['joint_range_rf_2_upper'] = [1.571, 1e-5]
phys_param_dict_iiwa['joint_range_rf_2_lower'] = [0, -1e-5]
phys_param_dict_iiwa['joint_range_rf_2_position_upper'] = [1e-5, np.pi/3]
phys_param_dict_iiwa['joint_range_rf_2_position_lower'] = [0, np.pi/8]
phys_param_dict_iiwa['joint_range_rf_1_upper'] = [1.571, np.pi/2]
phys_param_dict_iiwa['joint_range_rf_1_lower'] = [0, -0.125]
phys_param_dict_iiwa['actuation_rf_1'] = [1, 1e-5]
phys_param_dict_iiwa['joint_range_rf_0_upper'] = [1.571, 1.571]
phys_param_dict_iiwa['joint_range_rf_0_lower'] = [0, 0]
phys_param_dict_iiwa['shape_th_jaco'] = [0, 0]
phys_param_dict_iiwa['shape_th_kinova'] = [0, 0]
phys_param_dict_iiwa['shape_th_iiwa'] = [0, 1]
phys_param_dict_iiwa['shape_ff_jaco'] = [0, 0]
phys_param_dict_iiwa['shape_ff_kinova'] = [0, 0]
phys_param_dict_iiwa['shape_ff_iiwa'] = [0, 1]
phys_param_dict_iiwa['shape_mf'] = [0, 1]
phys_param_dict_iiwa['shape_rf'] = [0, 1]
phys_param_dict_iiwa['shape_rf_jaco'] = [0, 0]
phys_param_dict_iiwa['shape_lf'] = [0, 1]
phys_param_dict_iiwa['knuckle_pos_th_x'] = [0.034, 0]
phys_param_dict_iiwa['knuckle_pos_th_y'] = [-0.009, -0.06142]
phys_param_dict_iiwa['knuckle_pos_th_z'] = [0.029, -0.0127]
phys_param_dict_iiwa['knuckle_pos_ff_x'] = [0.033, 0]
phys_param_dict_iiwa['knuckle_pos_ff_y'] = [0, -0.06142]
phys_param_dict_iiwa['knuckle_pos_ff_z'] = [0.095, 0.0127]
phys_param_dict_iiwa['knuckle_pos_mf_x'] = [0.011, 0.0]
phys_param_dict_iiwa['knuckle_pos_mf_y'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_mf_z'] = [0.099, 0.027073]
phys_param_dict_iiwa['knuckle_pos_rf_x'] = [-0.011, -0.022256]
phys_param_dict_iiwa['knuckle_pos_rf_y'] = [0, -0.06142]
phys_param_dict_iiwa['knuckle_pos_rf_z'] = [0.095, 0.027073]
# phys_param_dict_iiwa['knuckle_pos_th_q0'] = [0.923956, 1]
# phys_param_dict_iiwa['knuckle_pos_th_q1'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_th_q2'] = [0.382499, 0]
# phys_param_dict_iiwa['knuckle_pos_th_q3'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_th_euler_x'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_th_euler_y'] = [0.7849990259705161, 0]
phys_param_dict_iiwa['knuckle_pos_th_euler_z'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_ff_q0'] = [1, 1]
# phys_param_dict_iiwa['knuckle_pos_ff_q1'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_ff_q2'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_ff_q3'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_ff_euler_x'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_ff_euler_y'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_ff_euler_z'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_rf_q0'] = [1, 1]
# phys_param_dict_iiwa['knuckle_pos_rf_q1'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_rf_q2'] = [0, 0]
# phys_param_dict_iiwa['knuckle_pos_rf_q3'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_rf_euler_x'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_rf_euler_y'] = [0, 0]
phys_param_dict_iiwa['knuckle_pos_rf_euler_z'] = [0, 0]
phys_param_dict_iiwa['palm_jaco'] = [0, 0]
phys_param_dict_iiwa['palm_kinova'] = [0, 0]
phys_param_dict_iiwa['palm_iiwa'] = [0, 1]
phys_param_dict_iiwa['wrist'] = [0, 1]
phys_param_dict_iiwa['arm_length'] = [0, 1]
phys_param_dict_iiwa['arm_range'] = [0, 1]
phys_param_dict_iiwa['robot_arm_body_1'] = [0.15675, 0.15]
phys_param_dict_iiwa['robot_arm_body_2'] = [0.11875, 0.185]
phys_param_dict_iiwa['robot_arm_body_3'] = [0.205, 0.21]
phys_param_dict_iiwa['robot_arm_body_4'] = [0.205, 0.19]
phys_param_dict_iiwa['robot_arm_body_5'] = [0.2073, 0.21]
phys_param_dict_iiwa['robot_arm_body_5_y'] = [-0.0114, 0]
phys_param_dict_iiwa['robot_arm_body_6'] = [0.10375, 0.19]
phys_param_dict_iiwa['robot_arm_body_7'] = [0.10375, 0.081]
phys_param_dict_iiwa['robot_arm_body_hand'] = [-0.2, -0.044]
phys_param_dict_iiwa['robot_arm_shape_jaco'] = [0, 0]
phys_param_dict_iiwa['robot_arm_shape_kinova'] = [0, 0]
phys_param_dict_iiwa['robot_arm_shape_iiwa'] = [0, 1]
phys_param_dict_iiwa['tendon_stiffness_th'] = [0, 1000]
phys_param_dict_iiwa['tendon_stiffness_ff'] = [0, 1000]
phys_param_vector_iiwa = []
for k in list(phys_param_dict_iiwa.keys()):
    phys_param_vector_iiwa.append(phys_param_dict_iiwa[k])




phys_param_vector_jaco = np.array(phys_param_vector_jaco)
phys_param_vector_kinova = np.array(phys_param_vector_kinova)
phys_param_vector_iiwa = np.array(phys_param_vector_iiwa)

phys_param_vector_all = np.stack([ \
        phys_param_vector_jaco, \
        phys_param_vector_kinova, \
        phys_param_vector_iiwa, ], axis=0)
phys_param_vector_all_min = np.min(np.min(phys_param_vector_all, axis=-1), axis=0)
phys_param_vector_all_max = np.max(np.max(phys_param_vector_all, axis=-1), axis=0)
phys_param_vector_all = np.stack([
    phys_param_vector_all_min, phys_param_vector_all_max], axis=-1)

# phys_param_vector_all[:, 0] -= 1e-3
phys_param_vector_all[:, 1] += 1e-5


interp_param_vector_dict = {}
interp_param_vector_dict['jaco'] = \
        (phys_param_vector_jaco[:, 1] - phys_param_vector_all[:, 0])/\
        (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0])
interp_param_vector_dict['kinova'] = \
        (phys_param_vector_kinova[:, 1] - phys_param_vector_all[:, 0])/\
        (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0])
interp_param_vector_dict['iiwa'] = \
        (phys_param_vector_iiwa[:, 1] - phys_param_vector_all[:, 0])/\
        (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0])
interp_param_vector_dict['original'] = \
        (phys_param_vector_kinova[:, 0] - phys_param_vector_all[:, 0])/\
        (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0])

with open('hand_manipulation_suite/interp_params_dict.json', 'w') as f:
    for k in interp_param_vector_dict:
        interp_param_vector_dict[k] = interp_param_vector_dict[k].tolist()
    json.dump(interp_param_vector_dict, f, indent=4)
with open('hand_manipulation_suite/phys_param_vector_all.json', 'w') as f:
    json.dump(phys_param_vector_all.tolist(), f, indent=4)


with open('hand_manipulation_suite/interp_params_dict_l1.json', 'r') as f:
    tree_interp_param_dict = json.load(f)

interp_param_vector = tree_interp_param_dict['steiner_1']
# interp_param_vector = tree_interp_param_dict['steiner_2']
# interp_param_vector = tree_interp_param_dict['original']
# interp_param_vector = tree_interp_param_dict['iiwa']
# interp_param_vector = tree_interp_param_dict['kinova']
# interp_param_vector = tree_interp_param_dict['jaco']


for i in range(105):
    # interp_param_vector = np.zeros([105])
    # interp_param_vector[i] = 1.

    env = gym.make(generalized_env, \
            ycb_scene_dir=ycb_scene_dir, \
            dense_reward=False, \
            # robot='Kinova3', \
            interp_param_vector=interp_param_vector, \
            simulation_mode=True, \
            virtual_confine=True, \
            finger_option='ff', \
            lf_shrink=0.7, \
            phys_param_vector_all=phys_param_vector_all, \
            frame_skip=50, \
            )
    o = env.reset()

    xml_string = env.env.sim.model.get_xml()
    with open('tmp.xml', 'w') as f:
        f.write(xml_string)

    for ep in range(300):
        o = env.reset()
        score = 0
        goal_achieved = False
        for _ in range(200):
            a = policy.get_action(o)[1]['evaluation']
            a[-1] = 1
            o, r, d, g = env.step(a)
            score += r
            if g['goal_achieved']:
                goal_achieved = True

            env.render()
        print('goal_achieved: {}, score: {}'.format(goal_achieved, score))


