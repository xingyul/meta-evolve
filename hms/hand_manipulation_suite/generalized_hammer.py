import numpy as np
from gym import utils
from mujoco_py import MjViewer
import xml.etree.ElementTree as ET
import json
import io
import os

import hand_manipulation_suite.mujoco_env as mujoco_env
import make_generalized_envs
import utils.mjcf_utils as mjcf_utils
import utils.quatmath as quatmath


class GeneralizedHammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, \
            model_path='/tmp/tmp.xml', \
            interp_param_vector=[0.]*61, \
            dense_reward=False, \
            horizon=200):
        self.dense_reward = dense_reward
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1
        self.old_qpos_ids = [-1]
        self.slide_qpos_ids = [-1]
        self.current_slide_action = 0
        self.horizon = horizon
        self.slide_dir = np.array([1,-1,-1,-1])
        self.illegal_contact = False

        ##### get the interpolated bodies
        generalized_env = 'hammer-v0-shrink'
        hand_env = make_generalized_envs.generalized_envs[generalized_env] \
                (interp_param_vector=interp_param_vector, dense_reward=dense_reward, \
                return_xml_only=True)

        ##### change geom to box
        joint = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='joint', attribs={'name': 'nail_dir'})
        joint.attrib['frictionloss'] = '0.5'
        joint.attrib['range'] = '-0.01 0.095'

        ##### change hammer mass
        scale = 1 / 5.
        object_body = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': 'Object'})
        inertial = mjcf_utils.find_elements(object_body, tags='inertial')
        inertial.attrib['mass'] = str(float(inertial.attrib['mass']) * scale)
        inertial.attrib['diaginertia'] = ' '.join([str(d_) for d_ in np.array([float(d) for d in inertial.attrib['diaginertia'].split()]) * scale])

        ##### dump to xml file for later read, for interp_env
        with io.StringIO() as string:
            string.write(ET.tostring(hand_env.getroot(), encoding='unicode'))
            xml_string = string.getvalue()

        mujoco_env.MujocoEnv.__init__(self, model_path, 5, model_xml=xml_string)
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = self.sim.model.site_name2id('S_target')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.tool_sid = self.sim.model.site_name2id('tool')
        self.goal_sid = self.sim.model.site_name2id('nail_goal')
        self.act_mid = np.mean(self.sim.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.sim.model.actuator_ctrlrange[:,1]-self.sim.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.sim.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.sim.model.actuator_ctrlrange[:,0])

        ##### set up joint ids for adroit hand
        self.old_qpos_ids = []
        self.slide_qpos_ids = []
        for i, name in enumerate(self.model.joint_names):
            if not 'slide_' in name:
                self.old_qpos_ids.append(self.sim.model.get_joint_qpos_addr(name))
            else:
                self.slide_qpos_ids.append(self.sim.model.get_joint_qpos_addr(name))

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        hand_action = a
        try:
            a = self.act_mid + hand_action * self.act_rng
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()

        ##### deal with illegal contact
        self.illegal_contact = False
        for con in self.sim.data.contact:
            contact_geom_1 = self.sim.model.geom_id2name(con.geom1)
            contact_geom_2 = self.sim.model.geom_id2name(con.geom2)
            other_contact = None
            if 'neck' == contact_geom_1:
                other_contact = contact_geom_2
            if 'neck' == contact_geom_2:
                other_contact = contact_geom_1
            if other_contact is not None:
                # not to touch hammer head or hammer neck with hand
                if other_contact.startswith('C_'):
                    self.illegal_contact = True

        reward = 0
        if self.dense_reward:
            # get to hammer
            reward = - 0.1 * np.linalg.norm(palm_pos - obj_pos)
            # take hammer head to nail
            reward -= np.linalg.norm((tool_pos - target_pos))
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())

        # make nail go inside
        reward -= 10 * np.linalg.norm(target_pos - goal_pos)
        # bonus for hammering the nail
        if (np.linalg.norm(target_pos - goal_pos) < 0.020):
            reward += 25
        if (np.linalg.norm(target_pos - goal_pos) < 0.010):
            reward += 75
        # bonus for lifting up the hammer
        if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
            reward += 2

        if np.linalg.norm(target_pos - goal_pos) < 0.010:
            # make sure not to drop hammer after completion
            reward -= 90 * (obj_pos[-1] < 0.05)

        # make sure not to touch the nail with hand
        reward -= 120 * self.illegal_contact

        goal_achieved = False
        if (np.linalg.norm(target_pos - goal_pos) < 0.010) and (obj_pos[-1] > 0.05):
            goal_achieved = True

        # touch the nail with hand -> fail
        if self.illegal_contact:
            goal_achieved = False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        qp_obs = qp[self.old_qpos_ids]
        qp_obs = qp_obs[:-6]

        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)

        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_rot = quatmath.quat2euler(self.data.body_xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = 0.0
        return np.concatenate([qp_obs, qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset(self, seed=None):
        self.sim.reset()
        ob = self.reset_model(seed=seed)
        return ob

    def reset_model(self, seed=None):
        np.random.seed(seed)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        for i, rng in enumerate(self.sim.model.jnt_range):
            ##### if both are larger than 0, or both are smaller than 0
            if rng[0] * rng[1] > 1e-8:
                smaller_limit = rng[0] if abs(rng[0]) < abs(rng[1]) else rng[1]
                avg_limit = np.mean(rng)
                self.sim.data.qpos[i] = smaller_limit
            joint_name = self.sim.model.joint_id2name(i)
            if 'finger_slide' in joint_name:
                self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(joint_name)] = \
                        self.sim.model.jnt_range[i, int(self.slide_dir[self.sim.model.actuator_name2id('act_' + joint_name)] < 0)]

        self.current_slide_action = self.slide_dir * -1

        target_bid = self.sim.model.body_name2id('nail_board')
        self.sim.model.body_pos[target_bid,2] = np.random.uniform(low=0.1, high=0.15)
        self.sim.forward()

        ##### set illegal contact back to False
        self.illegal_contact = False

        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.sim.model.body_pos[self.sim.model.body_name2id('nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.sim.model.body_pos[self.sim.model.body_name2id('nail_board')] = board_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 45
        self.viewer.cam.distance = 2.0
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 80:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
