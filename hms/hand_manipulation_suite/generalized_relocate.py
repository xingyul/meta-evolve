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


class GeneralizedRelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, \
            model_path='/tmp/tmp.xml', \
            interp_param_vector=[0.]*61, \
            dense_reward=False, \
            horizon=200):
        self.dense_reward = dense_reward
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.old_qpos_ids = [-1]
        self.slide_qpos_ids = [-1]
        self.current_slide_action = 0
        self.horizon = horizon
        self.slide_dir = np.array([1,-1,-1,-1])

        ##### get the interpolated bodies
        generalized_env = 'relocate-v0-shrink'
        hand_env = make_generalized_envs.generalized_envs[generalized_env] \
                (interp_param_vector=interp_param_vector, dense_reward=dense_reward, \
                return_xml_only=True)

        ##### change geom to box
        geom = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='geom', attribs={'name': 'sphere'})
        geom.attrib['name'] = 'object'
        geom.attrib['type'] = 'capsule'
        geom.attrib['size'] = '0.03 0.02'
        geom.attrib['quat'] = '{} 0 {} 0'.format(np.sqrt(0.5), np.sqrt(0.5))

        ##### change object mass
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

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
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
        try:
            a = self.act_mid + a * self.act_rng
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = 0
        if self.dense_reward:
            reward += -0.1*np.linalg.norm(palm_pos - obj_pos) # take hand to object
            if obj_pos[2] > 0.04: # if object off the table
                reward += -0.5*np.linalg.norm(palm_pos - target_pos) # make hand go to target
                reward += -0.5*np.linalg.norm(obj_pos - target_pos) # make object go to target
        if obj_pos[2] > 0.04: # if object off the table
            reward += 1.0 # bonus for lifting the object

        if np.linalg.norm(obj_pos - target_pos) < 0.1:
            reward += 10.0 # bonus for object close to target
        if np.linalg.norm(obj_pos - target_pos) < 0.05:
            reward += 20.0 # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos-target_pos) < 0.1 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        qp_obs = qp[:-6]

        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp_obs, palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])

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
                # self.sim.data.qpos[i] = smaller_limit
                self.sim.data.qpos[i] = avg_limit
            joint_name = self.sim.model.joint_id2name(i)
            if 'finger_slide' in joint_name:
                self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(joint_name)] = \
                        self.sim.model.jnt_range[i, int(self.slide_dir[self.sim.model.actuator_name2id('act_' + joint_name)] < 0)]

        self.current_slide_action = self.slide_dir * -1

        self.model.body_pos[self.obj_bid,0] = np.random.uniform(low=-0.1, high=0.1)
        self.model.body_pos[self.obj_bid,1] = np.random.uniform(low=-0.1, high=0.1)
        self.model.site_pos[self.target_obj_sid, 0] = np.random.uniform(low=-0.1, high=0.1)
        self.model.site_pos[self.target_obj_sid,1] = np.random.uniform(low=-0.1, high=0.1)
        self.model.site_pos[self.target_obj_sid,2] = np.random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = -45

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) >= 40:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
