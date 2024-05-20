import numpy as np
import os
import pickle
import glob
from gym import utils
import json
import cv2
from mujoco_py import MjViewer
import copy
import pyquaternion
import io
import meshio

import xml.etree.ElementTree as ET
import make_generalized_envs
import hand_manipulation_suite.mujoco_env as mujoco_env
from utils.quatmath import *
import robot_arm_interp

import dex_ycb_utils
from models.base import MujocoXML
from environments.manipulation.hand_env import HandEnv
import utils.mjcf_utils as mjcf_utils
import utils.transform_utils as transform_utils
from robots import ROBOT_CLASS_MAPPING
import utils.body_pose_restriction as body_pose_restriction


def ycb_obj_xml(ycb_scene_dir, ycb_ids, grasp_ycb_id, pose_Rs, pose_ts, \
        texture_files=None, global_offset=np.array([0,0,0]), \
        manual_offset_grasp=None, initial_place=False, textured=False, \
        ignore_useless_obj=[], obj_armature=0.005):
    meshes = []
    textures = []
    materials = []
    bodies = []

    average_pose_ts = np.mean(np.stack(pose_ts, axis=0), axis=0)
    average_pose_ts[-1] = 0.

    rgbas = []
    texture_rgbas = []
    for idx, ycb_id in enumerate(ycb_ids):
        texture_file = texture_files[idx]
        avg_rgb_file = os.path.join(ycb_scene_dir, 'texture_average_{}.npz'.format(ycb_id))
        if not os.path.exists(avg_rgb_file):
            print('reading', texture_file)
            texture = cv2.imread(texture_file)[:, :, ::-1]
            rgb = texture[texture.sum(axis=-1) > 0] / 255.
            avg_rgb = np.mean(rgb, axis=0)
            np.savez_compressed(avg_rgb_file, avg_rgb=avg_rgb)
        avg_rgb = np.load(avg_rgb_file)['avg_rgb']
        rgba = np.concatenate([avg_rgb, np.array([1])])
        texture_rgbas.append(rgba)
        if textured:
            rgba = np.concatenate([avg_rgb, np.array([0])])
        else:
            rgba = np.concatenate([avg_rgb, np.array([0.2])])
        rgbas.append(rgba)

    for idx, ycb_id in enumerate(ycb_ids):
        if not initial_place:
            if ycb_id != grasp_ycb_id:
                continue
        if textured:
            msh_file = glob.glob(os.path.join(ycb_scene_dir, str(ycb_id) + '.stl'))[0]
            mesh = mjcf_utils.new_element(tag='mesh', \
                    name='ycb_mesh_vis_'+str(ycb_id), \
                    file=os.path.abspath(msh_file), scale=[1,1,1])
            meshes.append(mesh)
        stls = glob.glob(os.path.join(ycb_scene_dir, str(ycb_id) + '_vhacd*.stl'))
        stls.sort()
        for i, stl in enumerate(stls):
            mesh = mjcf_utils.new_element(tag='mesh', \
                    name='ycb_mesh_'+str(ycb_id)+'_{}'.format(i), \
                    file=os.path.abspath(stl), scale=[1,1,1])
            meshes.append(mesh)
        ##### read pose
        pose_R = pose_Rs[idx]
        pose_t = pose_ts[idx]
        pose_R = pose_R + 0.5 * np.dot((np.eye(3) - np.dot(pose_R, pose_R.T)), pose_R)
        pose_q = np.array(list(pyquaternion.Quaternion(matrix=pose_R)))
        average_pose_ts = np.mean(np.stack(pose_ts, axis=0), axis=0)
        average_pose_ts[-1] = 0.
        ##### body and joints
        if manual_offset_grasp is None: # grasp object init and other objs
            manual_offset = np.zeros([3])
            manual_offset[:2] = manual_offset[:2] - average_pose_ts[:2]
            manual_offset += global_offset

            obj_filename = os.path.join(ycb_scene_dir, str(ycb_id) + '.stl')
            mesh = meshio.read(obj_filename)
            vertices = np.dot(mesh.points[:, :3], pose_R.T) + pose_t + manual_offset
            z_min = np.min(vertices[:, -1])
            if initial_place:
                manual_offset[2] = manual_offset[2] - z_min
        else: # target obj, provided with offset
            manual_offset = manual_offset_grasp

        pos = copy.deepcopy(pose_t)
        pos = pos + manual_offset
        if ycb_id == grasp_ycb_id: # this is the offset that is returned and used
            manual_offset_return = manual_offset

        body = mjcf_utils.new_body(name='ycb_body_'+str(ycb_id), pos=pos, \
                quat=pose_q)
        ##### body and joints
        add_geom = False
        if initial_place:
            if ycb_id == grasp_ycb_id:
                add_geom = True
            elif ycb_id not in ignore_useless_obj:
                add_geom = True
        else:
            if ycb_id == grasp_ycb_id:
                add_geom = True
        if add_geom:
            for i, stl in enumerate(stls):
                geom = mjcf_utils.new_element(tag='geom', \
                        name='ycb_geom_'+str(ycb_id)+'_{}'.format(i), \
                        pos=[0,0,0], mesh='ycb_mesh_'+str(ycb_id)+'_{}'.format(i), \
                        type='mesh', solref=[0.002, 1], solimp=[0.998, 0.998, 0.001], \
                        density=1000, friction=[1, 0.5, 0.01], group=1, \
                        condim=4, rgba=rgbas[idx], margin=0)
                body.append(geom)
            if textured:
                geom = mjcf_utils.new_element(tag='geom', \
                        name='ycb_geom_vis_'+str(ycb_id), pos=[0,0,0], \
                        mesh='ycb_mesh_vis_'+str(ycb_id), type='mesh', \
                        group=1, conaffinity=0, contype=0, rgba=texture_rgbas[idx], mass=1e-9)
                        # group=0, conaffinity=0, contype=0, material='ycb_material_'+str(ycb_id))
                body.append(geom)
        ##### only grasp object has joints
        if initial_place:
            if ycb_id == grasp_ycb_id:
                inertial = mjcf_utils.new_element(tag='inertial', name=None, \
                        diaginertia=[1e-4,1e-4,1e-4], mass=0.02, pos=[0,0,0])
                joint_x = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Tx', pos=[0,0,0], \
                        armature=obj_armature, \
                        axis=[1,0,0], type='slide', limited='false', damping=0.001)
                joint_y = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Ty', pos=[0,0,0], \
                        armature=obj_armature, \
                        axis=[0,1,0], type='slide', limited='false', damping=0.001)
                joint_z = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Tz', pos=[0,0,0], \
                        armature=obj_armature, \
                        axis=[0,0,1], type='slide', limited='false', damping=0.001)
                joint_Rx = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Rx', pos=[0,0,0], \
                        armature=obj_armature, axis=[1,0,0])
                joint_Ry = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Ry', pos=[0,0,0], \
                        armature=obj_armature, axis=[0,1,0])
                joint_Rz = mjcf_utils.new_joint( \
                        name='ycb_joint_'+str(ycb_id)+'_Rz', pos=[0,0,0], \
                        armature=obj_armature, axis=[0,0,1])
                body.append(inertial)
                body.append(joint_x)
                body.append(joint_y)
                body.append(joint_z)
                body.append(joint_Rx)
                body.append(joint_Ry)
                body.append(joint_Rz)
        bodies.append(body)
    return meshes, bodies, textures, materials, manual_offset_return


class GeneralizedYCBEnvV2Unified(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
            ycb_scene_dir='/home/xyl/Projects/meta-evolve/dex_ycb/006_mustard_bottle_20200709_143211',
            robot='Jaco',
            interp_param_vector=[0.]*42,
            phys_param_vector_all=[0.]*100,
            dense_reward=False,
            simulation_mode=True,
            virtual_confine=True,
            virtual_confine_penalty=0.,
            horizon=200,
            gripper_name='jaco',
            show_ycb_affordance=False,
            ycb_textured=False,
            demo_path=None,
            ignore_useless_obj=[],
            obj_armature=0.0,
            local_r_shaping=0.,
            show_arm_geom=True,
            frame_skip=50,
            **kwargs):
        self.dense_reward = dense_reward
        self.robot = robot
        self.frame_skip = frame_skip
        self.simulation_mode = simulation_mode
        self.virtual_confine = virtual_confine
        self.virtual_confine_penalty = virtual_confine_penalty
        self.ycb_scene_dir = ycb_scene_dir
        self.gripper_name = gripper_name
        self.show_ycb_affordance = show_ycb_affordance
        self.ycb_textured = ycb_textured
        self.ignore_useless_obj = ignore_useless_obj
        self.obj_armature = obj_armature
        self.horizon = horizon
        self.show_arm_geom = show_arm_geom
        self.starting_t = 0

        self.target_obj_bid = -1
        self.S_grasp_sid = -1
        self.finger_joint_names = ['table_top']
        self.finger_tip_site_names = ['table_top']
        self.obj_bid = -1
        self.hand_actuator_ids = [0,1]
        self.old_hand_joint_ids = [0,1]
        self.non_virtual_old_joint_ids = [0,1]
        self.finger_slide_joint_id = -1
        self.gripper_body_id = -1
        self.robot_joint_ids = [0,1]
        self.obj_init_pos = [-1]
        self.current_gripper_action = 0.
        self.interp_param_vector = np.array(interp_param_vector)
        self.phys_param_vector_all = np.array(phys_param_vector_all)
        self.local_r_shaping = local_r_shaping

        self.phys_param_vector = (self.phys_param_vector_all[:, 1] - self.phys_param_vector_all[:, 0]) * self.interp_param_vector + self.phys_param_vector_all[:, 0]
        arm_interp_param = self.phys_param_vector[-16] ##### subject to change
        ##### read json configs

        robot_config = {
            "initialization_noise": "default",
            "mount_type": None,
            "controller_config": {
                "hard_reset": False,
                "impedance_mode": "fixed",
                "position_limits": None,
                "input_min": -1,
                "kp": 150,
                "control_delta": True,
                "kp_limits": [0, 300],
                "uncouple_pos_ori": True,
                "output_min": [-0.2, -0.2, -0.2, -1, -1, -1],
                "damping_ratio": 1,
                "ramp_ratio": 0.2,
                "orientation_limits": None,
                "damping_ratio_limits": [0, 10],
                "interpolation": None,
                "input_max": 1,
                "output_max": [0.2, 0.2, 0.2, 1, 1, 1],
                "type": "OSC_POSE"
            },
            "control_freq": 20,
            "gripper_type": None,
        }

        # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
        #         self.robot + '.json'), 'r') as f:
        #     robot_config = json.load(f)
        # robot_config['gripper_type'] = None
        options = {'controller_configs': robot_config['controller_config'], \
                'robots': [self.robot], 'gripper_types': None}

        ##### import robosuite env as a starter
        robosuite_env = HandEnv(
            **options,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            horizon=self.horizon,
            control_freq=20,
        )
        original_robosuite_robot_pos = np.array([float(i) for i in robosuite_env.robots[0].robot_model._elements["root_body"].attrib['pos'].split()]) + robosuite_env.robots[0].robot_model.bottom_offset

        ##### change robot location
        rotation = np.pi / 2
        rot = np.array((0, 0, rotation))
        target_dist = 0.5
        target_dist_pos = np.array([-0.15, target_dist, 0])
        init_pos = np.array([0.1603, -0.95, -0.112])
        pos = init_pos * (1 - arm_interp_param) + \
                (init_pos + target_dist_pos) * arm_interp_param
        robosuite_env.robots[0].robot_model.set_base_xpos(pos)
        robosuite_env.robots[0].robot_model.set_base_ori(rot)

        ##### calculate the updates of robot base due to my set
        updated_robosuite_robot_pos = np.array([float(i) for i in robosuite_env.robots[0].robot_model._elements["root_body"].attrib['pos'].split()]) + robosuite_env.robots[0].robot_model.bottom_offset
        updated_robosuite_robot_mat = transform_utils.euler2mat(rot)
        original_robosuite_robot_pos = np.dot(updated_robosuite_robot_mat, \
                original_robosuite_robot_pos)

        ##### change the ctrlrange of motors
        motors = [e for e in robosuite_env.model.root.iter() if (e.tag =='motor')]
        for motor in motors:
            ctrlrange = [float(c) for c in motor.attrib['ctrlrange'].split()]
            ctrlrange = np.array(ctrlrange) * 10
            motor.attrib['ctrlrange'] = ' '.join([str(c) \
                    for c in ctrlrange.tolist()])

        ##### get the interpolated bodies
        generalized_env = 'relocate-v0-unified'
        hand_env = make_generalized_envs.generalized_envs[generalized_env] \
                (interp_param_vector=interp_param_vector, \
                phys_param_vector_all=self.phys_param_vector_all, \
                dense_reward=dense_reward, \
                return_xml_only=True, **kwargs)

        ##### dump to xml file for later read, for interp_env
        with io.StringIO() as string:
            string.write(ET.tostring(hand_env.getroot(), encoding='unicode'))
            xml_string = string.getvalue()
        hand_env = MujocoXML(xml_string=xml_string, folder=os.path.dirname(__file__))

        ##### remove mount
        mount_base = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'mount0_base'})
        mount_parent = mjcf_utils.find_parent(robosuite_env.model.root, mount_base)
        mount_parent.remove(mount_base)

        ##### add collision-free setting
        contact = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='contact')
        robot_base = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'robot0_base'})
        robot_bodies = [elem for elem in robot_base.iter() if (elem.tag == 'body')]
        forearm = mjcf_utils.find_elements(hand_env.root, tags='body', attribs={'name': 'forearm'})
        forearm_bodies = [elem for elem in forearm.iter() if (elem.tag == 'body')]
        for forearm_body in forearm_bodies:
            for robot_body in robot_bodies:
                new_exclude = mjcf_utils.new_element(tag='exclude', name=None, \
                        body1=forearm_body.attrib['name'], \
                        body2=robot_body.attrib['name'])
                contact.append(new_exclude)

        ##### adjust robosuite item height by -0.8
        worldbody = [elem for elem in robosuite_env.model.root.iter() \
                if (elem.tag == 'worldbody')][0]
        for child in worldbody:
            if (child.tag == 'body') or \
               (child.tag == 'camera') or \
               (child.tag == 'geom'):
                pos = child.attrib['pos'].split()
                child.attrib['pos'] = ' '.join(pos[:2] + \
                        [str(float(pos[-1]) - 0.8)])

        ##### adjust robot base height by 0.015: SafeAI Lab setup
        body = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'robot0_base'})
        body_pos = body.attrib['pos'].split()
        body.attrib['pos'] = ' '.join([str(p) for p in body_pos])


        robosuite_env.model.merge(hand_env)

        ##### attach forearm to robot eef
        forearm = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'forearm'})
        gripper_body = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'gripper0_null_gripper'})
        gripper_inertial = mjcf_utils.find_elements(gripper_body, tags='inertial')
        gripper_inertial.attrib['diaginertia'] = '1e-8 1e-8 1e-8'
        new_forearm = copy.deepcopy(forearm)
        new_forearm.attrib['pos'] = '0 0 0'
        new_forearm.attrib['quat'] = '0.5 -0.5 0.5 -0.5'
        ##### attach forearm to robot eef
        gripper_body.append(new_forearm)
        forearm_parent = mjcf_utils.find_parent(robosuite_env.model.root, forearm)
        forearm_parent.remove(forearm)

        ##### compute the pose of eef defined in the model file (when it is born)
        robosuite_env.sim.data.qpos[:] = 0.
        robosuite_env.sim.forward()
        body_born_xpos = robosuite_env.sim.data.site_xpos[robosuite_env.sim.model.site_name2id('gripper0_ft_frame')]
        update_offset = updated_robosuite_robot_pos - \
                original_robosuite_robot_pos
        body_born_xpos = np.dot(updated_robosuite_robot_mat, \
                body_born_xpos) + update_offset - np.array([0.,0.,0.8])
        body_born_xmat = robosuite_env.sim.data.site_xmat[robosuite_env.sim.model.site_name2id('gripper0_ft_frame')]
        body_born_xmat = np.reshape(body_born_xmat, [3, 3])
        body_born_xmat = np.dot(updated_robosuite_robot_mat, body_born_xmat)
        init_constraint_pos = np.array([0, -0.7, 0.4])
        target_constraint_pos = init_constraint_pos + np.array([0,target_dist,0])
        self.constraint_pos = init_constraint_pos * (1 - arm_interp_param) + \
                target_constraint_pos * arm_interp_param
        constraint_rot = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        constraint_rot = pyquaternion.Quaternion(matrix=constraint_rot)
        self.constraint_rot = np.array([constraint_rot[0], \
                constraint_rot[1], constraint_rot[2], constraint_rot[3] ])
        ##### get the virtual mount, attacher and equality
        target_pos_range = 0.65
        target_quat_range = np.array([1e-8, 1e-8, np.pi/3])
        self.virtual_pos_range = 1e-8 * (1 - arm_interp_param) + \
                target_pos_range * arm_interp_param
        self.virtual_quat_range = 1e-8 * (1 - arm_interp_param) + \
                target_quat_range * arm_interp_param
        virtual_mount, equality = body_pose_restriction.body_pose_restriction( \
                body_name='gripper0_null_gripper', pos=body_born_xpos, \
                born_rot=body_born_xmat, constr_rot=self.constraint_rot, \
                pos_range=self.virtual_pos_range, \
                quat_range=self.virtual_quat_range)
        if self.virtual_confine:
            worldbody.append(virtual_mount)
            robosuite_env.model.root.append(equality)

        ##### get the position joints tendon
        tendon_stiffness_th = float(self.phys_param_vector[-2]) # subject to change
        tendon_stiffness_ff = float(self.phys_param_vector[-1]) # subject to change

        tendon = mjcf_utils.new_element(tag='tendon', name=None)
        springlength = np.pi/2

        damping = tendon_stiffness_ff / 3.
        tendon_fixed_1 = mjcf_utils.new_element(tag='fixed', \
                name='tendon_1', range=[-1, 1], \
                stiffness=tendon_stiffness_ff, \
                damping=damping, \
                springlength=springlength)
        joint_1 = mjcf_utils.new_element(tag="joint", name=None, \
                joint='finger_position_joint_ff_2', coef=1)
        joint_2 = mjcf_utils.new_element(tag="joint", name=None, \
                joint='finger_position_joint_ff_1', coef=1.)
        tendon_fixed_1.append(joint_1); tendon_fixed_1.append(joint_2);
        tendon.append(tendon_fixed_1)

        damping = tendon_stiffness_th / 3.
        tendon_fixed_2 = mjcf_utils.new_element(tag='fixed', \
                name='tendon_2', range=[-1, 1], \
                stiffness=tendon_stiffness_th, \
                damping=damping, \
                springlength=springlength)
        joint_1 = mjcf_utils.new_element(tag="joint", name=None, \
                joint='finger_position_joint_th_3', coef=1)
        joint_2 = mjcf_utils.new_element(tag="joint", name=None, \
                joint='finger_position_joint_th_2', coef=1.)
        tendon_fixed_2.append(joint_1); tendon_fixed_2.append(joint_2);
        tendon.append(tendon_fixed_2)

        robosuite_env.model.root.append(tendon)
        ##### get the position joints tendon

        ##### get visible mode
        '''
        worldbody = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='worldbody')
        if self.simulation_mode:
            geoms = [e for e in worldbody.iter() if (e.tag =='geom')]
            for geom in geoms:
                if 'type' in geom.attrib:
                    if geom.attrib['type'] == 'mesh':
                        geom_parent = mjcf_utils.find_parent( \
                                robosuite_env.model.root, geom)
                        geom_parent.remove(geom)
        '''

        ##### prevent contact penetrating and harden joint limit
        geoms = [e for e in robosuite_env.model.root.iter() if (e.tag =='geom')]
        geoms = [g for g in geoms if 'name' in g.attrib]
        for geom in geoms:
            geom.attrib['margin'] = '0'
            geom.attrib['solimp'] = '0.999 0.9999 0.001 0.01 6'
            geom.attrib['solref'] = '2e-3 1'
        joints = [e for e in robosuite_env.model.root.iter() if (e.tag =='joint')]
        joints = [j for j in joints if 'name' in j.attrib]
        for joint in joints:
            if 'limited' in joint.attrib:
                if joint.attrib['limited'] == 'true':
                    joint.attrib['margin'] = '0'
                    joint.attrib['solimplimit'] = '0.999 0.9999 0.001 0.01 6'
                    joint.attrib['solreflimit'] = '2e-2 1'

        ##### remove object and target
        object_body = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='body', attribs={'name': 'Object'})
        worldbody.remove(object_body)
        object_target = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='site', attribs={'name': 'target'})
        worldbody.remove(object_target)

        ##### change table size
        table_width = 1
        table_length = 1
        table_col = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_collision'})
        table_col.attrib['size'] = '{} {} 0.025'.format(table_width, table_length)
        table_col.attrib['friction'] = '0.5 0.1 0.01'
        table_col.attrib['priority'] = '1'
        table_vis = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_visual'})
        table_vis.attrib['size'] = '{} {} 0.025'.format(table_width, table_length)
        # legs
        table_leg_vis = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_leg1_visual'})
        table_leg_vis.attrib['pos'] = '{} 0.3 -0.3875'.format(table_width-0.1)
        table_leg_vis = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_leg2_visual'})
        table_leg_vis.attrib['pos'] = '-{} 0.3 -0.3875'.format(table_width-0.1)
        table_leg_vis = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_leg3_visual'})
        table_leg_vis.attrib['pos'] = '-{} -0.3 -0.3875'.format(table_width-0.1)
        table_leg_vis = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='geom', attribs={'name': 'table_leg4_visual'})
        table_leg_vis.attrib['pos'] = '{} -0.3 -0.3875'.format(table_width-0.1)

        ##### compiler and option: switch to inertialfromgeom
        compiler = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='compiler')
        compiler.attrib['inertiafromgeom'] = 'true'
        compiler.attrib['eulerseq'] = 'XYZ'
        option = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='option')
        option.attrib['timestep'] = '0.0002'

        asset = mjcf_utils.find_elements(robosuite_env.model.root, tags='asset')
        worldbody = mjcf_utils.find_elements(robosuite_env.model.root, tags='worldbody')
        ##### load ycb objects
        ycb_scene_xml = os.path.join(self.ycb_scene_dir, 'scene.xml')
        if os.path.exists(ycb_scene_xml):
            with open(ycb_scene_xml, 'r') as f:
                xml_string = f.read()
            ycb_scene_tree = ET.ElementTree(ET.fromstring(xml_string))
            ycb_worldbody = mjcf_utils.find_elements(ycb_scene_tree.getroot(), tags='worldbody')
            for ycb_body in ycb_worldbody:
                worldbody.append(ycb_body)
                if ycb_body.attrib['name'].endswith('_target'):
                    grasp_ycb_id = int(ycb_body.attrib['name'].split('_')[-2])
            ycb_asset = mjcf_utils.find_elements(ycb_scene_tree.getroot(), tags='asset')
            if ycb_asset is not None:
                for ya in ycb_asset:
                    asset.append(ya)
        else:
            global_offset_file = os.path.join(self.ycb_scene_dir, 'global_offset.txt')
            if os.path.exists(global_offset_file):
                with open(global_offset_file, 'r') as f:
                    global_offset = f.read().split(',')
                self.global_offset = np.array([float(o) for o in global_offset])
            else:
                self.global_offset = np.array([0., 0., 0.])
            self.ycb_label_files = glob.glob(os.path.join(self.ycb_scene_dir, '*.pkl'))
            self.ycb_label_files.sort()
            self.manual_offset = None
            # for label_idx, label_file in enumerate(self.ycb_label_files[[0, -1]]):
            for label_idx in [0, len(self.ycb_label_files)-1]:
                label_file = self.ycb_label_files[label_idx]
                with open(label_file, 'rb') as f:
                    label = pickle.load(f)
                meta = label['meta']
                ycb_ids = meta['ycb_ids']
                grasp_id = int(meta['ycb_grasp_ind'])
                grasp_ycb_id = meta['ycb_ids'][grasp_id]
                ycb_ids = [grasp_ycb_id]
                grasp_id = 0
                pose_Rs = label['pose_R']
                pose_ts = label['pose_t']
                obj_file = [label['obj_file'][ycb_id] for ycb_id in ycb_ids]
                texture_files = [os.path.join(os.path.dirname(f), 'texture_map.png') \
                        for f in obj_file]
                ##### initial object placement
                meshes, bodies, textures, materials, self.manual_offset = \
                        ycb_obj_xml(self.ycb_scene_dir, \
                        ycb_ids, grasp_ycb_id, pose_Rs, pose_ts, texture_files, \
                        self.global_offset, self.manual_offset, \
                        initial_place=(label_idx==0), textured=self.ycb_textured, \
                        ignore_useless_obj=self.ignore_useless_obj, \
                        obj_armature=self.obj_armature)
                if label_idx == 0:
                    for body in bodies:
                        if body.attrib['name'] == 'ycb_body_{}'.format(grasp_ycb_id):
                            self.initial_place_pos = np.array([float(p) \
                                    for p in body.attrib['pos'].split()])
                ##### include new mesh assets
                if label_idx == 0:
                    for mesh in meshes:
                        asset.append(mesh)
                    for texture in textures:
                        asset.append(texture)
                    for material in materials:
                        asset.append(material)
                ##### goal object placement
                # if label_idx == (len(label_files)-1):
                if label_idx != 0:
                    bodies = [body for body in bodies \
                            if body.attrib['name'] == 'ycb_body_{}'.format(grasp_ycb_id)]
                    bodies[0].attrib['name'] = bodies[0].attrib['name'] + '_target'
                    geoms = [e for e in bodies[0].iter() if (e.tag == 'geom')]
                    for g in geoms:
                        g.attrib['name'] = g.attrib['name'] + '_target'
                        g.attrib['contype'] = '0'
                        g.attrib['conaffinity'] = '0'
                        g.attrib['rgba'] = '0 1 0 0.125'
                for body in bodies:
                    worldbody.append(body)

            ##### load dex trajectory sites
            self.dexycb_joint_site_names_list = []
            self.dexycb_id_list = []
            for label_idx, label_file in enumerate(self.ycb_label_files):
                with open(label_file, 'rb') as f:
                    dexycb_anno = pickle.load(f)
                dexycb_joints = dexycb_anno['joint_3d_cano']
                dexycb_joints_dist = np.mean(np.linalg.norm(dexycb_joints, axis=-1))
                # only keep those close enough to the object
                if (dexycb_joints_dist > 0.15):
                    continue
                dexycb_id = int(os.path.basename(label_file).split('.pkl')[0])
                self.dexycb_id_list.append(dexycb_id)
                # read grasp obj R and t
                meta = dexycb_anno['meta']
                ycb_ids = meta['ycb_ids']
                grasp_id = int(meta['ycb_grasp_ind'])
                grasp_ycb_id = meta['ycb_ids'][grasp_id]
                index = ycb_ids.index(grasp_ycb_id)
                obj_pose_R = dexycb_anno['pose_R']
                obj_pose_t = dexycb_anno['pose_t']
                grasp_obj_pose_R = obj_pose_R[index]
                grasp_obj_pose_t = obj_pose_t[index]
                obj_pose_t = dexycb_anno['pose_t']
                dexycb_joints = np.dot(dexycb_joints, grasp_obj_pose_R.T) + \
                        grasp_obj_pose_t + self.manual_offset
                dexycb_joint_site_names = []
                for i, dexycb_joint_name in enumerate(dex_ycb_utils._MANO_JOINTS):
                    if dexycb_joint_name not in dex_ycb_utils._ADROIT_TO_MANO_MAP.values():
                        continue
                    site_name = 'ycb_joint_site_{}_{}'.format(str(dexycb_id).zfill(3), dexycb_joint_name)
                    site_rgba = [i/(dexycb_joints.shape[0]-1),0,0, 0.2 if self.show_ycb_affordance else 0.]
                    dexycb_joint_site_names.append(site_name)
                    site = mjcf_utils.new_element(tag='site', \
                            pos=dexycb_joints[i], group=1, \
                            name=site_name, size=0.01, rgba=site_rgba)
                    # worldbody.append(site)
                self.dexycb_joint_site_names_list.append(dexycb_joint_site_names)

            ##### read all pose_t
            self.pose_R_dict = {}
            self.pose_t_dict = {}
            for label_idx in range(len(self.ycb_label_files)):
                label_file = self.ycb_label_files[label_idx]
                with open(label_file, 'rb') as f:
                    label = pickle.load(f)
                meta = label['meta']
                ycb_ids = meta['ycb_ids']
                grasp_id = int(meta['ycb_grasp_ind'])
                grasp_ycb_id = meta['ycb_ids'][grasp_id]
                pose_Rs = label['pose_R']
                pose_ts = label['pose_t']
                pose_t = pose_ts[grasp_id]
                pose_R = pose_Rs[grasp_id]

                dexycb_id = int(os.path.basename(label_file).split('.pkl')[0])
                self.pose_R_dict[dexycb_id] = pose_R
                self.pose_t_dict[dexycb_id] = pose_t
            ##### read all ik-ed dex joints
            self.dex_label_files = glob.glob(os.path.join(self.ycb_scene_dir, '*_ik.json'))
            self.dex_label_files.sort()
            self.dex_joints = {}
            for dex_label_file in self.dex_label_files:
                with open(dex_label_file, 'r') as f:
                    target_joint = json.load(f)
                idx = int(os.path.basename(dex_label_file).split('_ik')[0])
                self.dex_joints[idx] = np.array(target_joint)

        ##### get original general actuator params and change it
        actuator = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='actuator')
        actuators = [e for e in actuator.iter() if 'name' in e.attrib]
        human_actuators = [a for a in actuators if a.attrib['name'].startswith('A_')]
        self.original_ctrlrange = np.array([[float(s) \
                for s in a.attrib['ctrlrange'].split()] for a in human_actuators])
        self.original_gainprm = [float(a.attrib['gainprm'].split()[0]) \
                if 'gainprm' in a.attrib else 1. for a in human_actuators]
        self.original_biasprm = [float(a.attrib['biasprm'].split()[1]) \
                for a in human_actuators]
        self.original_forcelimit = np.array(self.original_gainprm) * \
                (self.original_ctrlrange[:,1] - self.original_ctrlrange[:,0])
        self.act_joint_map = {}
        self.joint_act_map = {}
        for act in actuators:
            self.act_joint_map[act.attrib['name']] = act.attrib['joint']
            self.joint_act_map[act.attrib['joint']] = act.attrib['name']

        ##### add robot arm geom
        robosuite_env = robot_arm_interp.robot_arm_interp(robosuite_env, \
                interp_param_vector[-14:], phys_param_vector_all[-14:], \
                include_geoms=self.show_arm_geom) # subject to change
        ##### add robot arm geom

        ##### add eef site for sim-to-real
        gripper_eef = mjcf_utils.find_elements(robosuite_env.model.root, \
                tags='site', attribs={'name': 'gripper0_ft_frame'})
        gripper_eef.attrib['pos'] = '0 0 0.13'
        gripper_eef.attrib['group'] = '1'
        ##### add eef site for sim-to-real

        ##### dump to xml file for later read
        xml_string = robosuite_env.model.get_xml()

        ##### load robot model, only for its controller
        robot = ROBOT_CLASS_MAPPING[self.robot](robot_type=self.robot, \
                idn=0, **robot_config)
        robot.load_model()

        ##### set initial robot qpos, correspond to "change robot location" part
        # self.robot_init_qpos = np.array([3.161703395268201, -0.7958170109060476, -0.7801621424185103, -2.329391025481599, 1.265246447699409, -0.5549965690904118, 1.3406693826971114])
        self.robot_init_qpos = np.array([3.511225, -0.31007293, \
                -1.1095288, -2.14046, 0.3587526, -0.89182976, 2.190176670])

        self.robot_joint_indexes = robosuite_env.robots[0]._ref_joint_indexes

        # with open('/home/xyl/Projects/meta-evolve-real/dex_ycb/debug.xml', 'r') as f:
        #     xml_string = f.read()

        ##### construct our own env by reading xml file
        mujoco_env.MujocoEnv.__init__(self, model_path=None, model_xml=xml_string, frame_skip=self.frame_skip)
        utils.EzPickle.__init__(self)
        del robosuite_env

        self.obj_init_pos = {}
        ycb_bids = [self.sim.model.body_name2id(b) for b in self.sim.model.body_names if 'ycb_body' in b]
        for ycb_bid in ycb_bids:
            self.obj_init_pos[ycb_bid] = self.sim.model.body_pos[ycb_bid].copy()

        self.reset()

        ##### set up controller
        robot.reset_sim(self.sim)
        robot.setup_references()
        robot._load_controller()
        self.controller = robot.controller
        self.robot_torque_limits = robot.torque_limits

        ##### set up actuator ids for adroit hand and gripper
        self.hand_actuator_ids = []
        self.gripper_actuator_ids = []
        self.hand_act2jointids = []
        for i, name in enumerate(self.model.actuator_names):
            if name.startswith('A_'):
                self.hand_actuator_ids.append(i)
            if name.startswith('act_'):
                self.gripper_actuator_ids.append(i)
        self.hand_actuator_ids = np.array(self.hand_actuator_ids)
        self.gripper_actuator_ids = np.array(self.gripper_actuator_ids)
        self.act_joint_id_map = {self.sim.model.actuator_name2id(a): \
                self.sim.model.joint_name2id(self.act_joint_map[a]) for a in self.act_joint_map}
        self.joint_act_id_map = {self.sim.model.joint_name2id(a): \
                self.sim.model.actuator_name2id(self.joint_act_map[a]) for a in self.joint_act_map}
        self.hand_act_joint_ids = np.array([self.act_joint_id_map[a] \
                for a in self.hand_actuator_ids])
        self.all_act_joint_ids = np.array([self.act_joint_id_map[a] \
                for a in range(len(self.sim.model.actuator_names))])
        self.dexycb_joint_site_id_list = []
        # for site_names in self.dexycb_joint_site_names_list:
        #     self.dexycb_joint_site_id_list.append([self.sim.model.site_name2id(n)
        #             for n in site_names])

        self.target_obj_bid = self.sim.model.body_name2id('ycb_body_{}_target'.format(grasp_ycb_id))
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('ycb_body_{}'.format(grasp_ycb_id))
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        self.act_mid = np.copy(np.mean(self.model.actuator_ctrlrange, axis=1))[self.hand_actuator_ids]
        self.act_rng = np.copy(0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]))[self.hand_actuator_ids]

        ##### object initial height
        # self.obj_init_pos = self.data.body_xpos[self.obj_bid].copy().ravel()

        ##### set up joint ids for adroit hand
        self.old_hand_joint_ids = []
        for i, name in enumerate(self.model.joint_names):
            if (not 'robot' in name) and \
               (not 'gripper' in name) and \
               (not 'OBJ' in name) and \
               (not 'virtual_' in name) and \
               (not 'position_' in name) and \
               (not 'ycb' in name):
                self.old_hand_joint_ids.append(i)

        ##### set up joint id for non-virtual-mount joints
        self.non_virtual_old_joint_ids = []
        for i, name in enumerate(self.model.joint_names):
            if ('virtual_' not in name) and ('position_' not in name):
                self.non_virtual_old_joint_ids.append(i)

        ##### set up joint id for robot joints
        self.robot_joint_ids = []
        for i, name in enumerate(self.model.joint_names):
            if name.startswith('robot'):
                self.robot_joint_ids.append(i)

        ##### set up joint id for the three position finger joints
        self.gripper_body_id = \
                self.sim.model.body_name2id('gripper0_null_gripper')
        ##### set up gripper site init quat
        gripper_mat = self.sim.data.get_site_xmat('gripper0_ft_frame')
        self.current_goal_gripper_quat = np.array(list(pyquaternion.Quaternion(matrix=gripper_mat)))
        self.current_goal_gripper_pos = self.sim.data.get_site_xpos('gripper0_ft_frame')

        ##### set up virtual joint ids
        if self.virtual_confine:
            self.virtual_joint_x_id = self.sim.model.joint_name2id('virtual_joint_x')
            self.virtual_joint_y_id = self.sim.model.joint_name2id('virtual_joint_y')
            self.virtual_joint_z_id = self.sim.model.joint_name2id('virtual_joint_z')
            self.virtual_joint_ids = np.array([
                self.virtual_joint_x_id,
                self.virtual_joint_y_id,
                self.virtual_joint_z_id ])
            self.vj_range = self.sim.model.jnt_range[self.virtual_joint_ids]

        ##### dexycb
        self.finger_joint_names = [j for j in self.sim.model.joint_names if j in list(dex_ycb_utils._ADROIT_TO_MANO_MAP.keys())]
        self.finger_joint_names.sort()
        self.finger_tip_site_names = [j for j in self.sim.model.site_names if j in list(dex_ycb_utils._ADROIT_TO_MANO_MAP.keys())]
        self.finger_tip_site_names.sort()

        ##### adjust transparency of geoms and sites
        if not self.simulation_mode:
            self.mj_viewer_setup()
            self.viewer.vopt.geomgroup[0] = 0
        self.sim.model.site_rgba[:, 3] = 0
        if self.virtual_confine:
            self.sim.model.site_rgba[self.sim.model.site_name2id('virtual_mount_site'), 3] = 0.25

        self.illegal_contact = False

    def step(self, a):

        self.sim.forward()

        num_hand_joints = len(self.old_hand_joint_ids)
        hand_action = a[:num_hand_joints]
        # operational 6DoF control action
        robot_action = a[num_hand_joints:(6+num_hand_joints)]
        gripper_action = a[-1]
        try:
            hand_action = self.act_mid + hand_action * self.act_rng # mean & scale
            self.sim.data.ctrl[self.hand_actuator_ids] = hand_action

            '''
            hand_action = np.clip(hand_action, -1, 1)
            hand_action = hand_action * self.original_forcelimit

            self.sim.data.ctrl[:] = 0.
            self.sim.data.xfrc_applied[:] = 0.
            self.sim.data.qfrc_applied[:] = 0.
            self.sim.data.qfrc_applied[self.old_hand_joint_ids] = hand_action
            '''

            self.controller.set_goal(robot_action)
            torques = self.controller.run_controller()

            low, high = self.robot_torque_limits
            torques = np.clip(torques, low, high)
            self.sim.data.ctrl[self.robot_joint_indexes] = torques

            speed = 0.8
            speed = 10
            self.current_gripper_action = np.clip(self.current_gripper_action + \
                    np.array([-1.0,1.0,1.0,1.0,-1.0]) * speed * \
                    np.sign(gripper_action), -1.0, 1.0)
            ##### rescale normalized gripper action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange[self.gripper_actuator_ids]
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_gripper_action = bias + weight * self.current_gripper_action
            self.sim.data.ctrl[self.gripper_actuator_ids] = applied_gripper_action

            for _ in range(self.frame_skip):
                self.sim.step()
        except:
            # only for the initialization phase
            ob = self.get_obs()
            return ob, 0., False, dict(goal_achieved=False)

        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        obj_mat = self.data.body_xmat[self.obj_bid].reshape([-1, 3])
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()

        ##### deal with illegal contact
        self.illegal_contact = False
        for con in self.sim.data.contact:
            contact_geom_1 = self.sim.model.geom_id2name(con.geom1)
            contact_geom_2 = self.sim.model.geom_id2name(con.geom2)
            if (contact_geom_1 == 'floor') and (contact_geom_2 == 'floor'):
                break
            if contact_geom_1.endswith('_jaco') or contact_geom_1.endswith('_kinova'):
                if contact_geom_2 == 'table_collision':
                    self.illegal_contact = True
                    break
            if contact_geom_2.endswith('_jaco') or contact_geom_2.endswith('_kinova'):
                if contact_geom_1 == 'table_collision':
                    self.illegal_contact = True
                    break

        target_pos = self.data.body_xpos[self.target_obj_bid].ravel()
        target_mat = self.data.body_xmat[self.target_obj_bid].reshape([-1, 3])

        reward = 0
        if self.dense_reward:
            # get to hammer
            # reward -= 0.1 * adroit_to_dexycb_dist
            reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
            if obj_pos[2] > (self.obj_init_pos[-1] + 0.004): # if object off the table
                reward += 1.0 # bonus for lifting the object
                reward += -0.5*np.linalg.norm(obj_pos-target_pos) # make object go to target

        pos_diff = np.linalg.norm(obj_pos - target_pos)
        mat_diff = np.dot(obj_mat, target_mat.T)
        quat_diff = pyquaternion.Quaternion(matrix=mat_diff)
        angle_diff = np.abs(quat_diff.radians)

        close_reward = 0.
        if pos_diff < 0.1:
            close_reward += 5.0 * (0.2-pos_diff) / 0.1
        if pos_diff < 0.05:
            close_reward += 10.0 * (0.1-pos_diff) / 0.05
            if angle_diff < 0.2:
                close_reward += 10.0 # bonus for object close to target
            if angle_diff < 0.1:
                close_reward += 20.0 # bonus for object "very" close to target
            if angle_diff < 0.05:
                close_reward += 30.0 # bonus for object "very" close to target

        palm_obj_dist = np.linalg.norm(palm_pos-obj_pos)
        if palm_obj_dist < 0.2:
            reward += close_reward

        # if adroit_to_dexycb_dist < 0.05:
        #     reward += close_reward

        # make sure not to let one object touch another object
        reward -= 1. * self.illegal_contact

        # if reward > 0:
        #     reward = reward * 0.01 / adroit_to_dexycb_dist

        # local reward shaping during interpolation
        reward = reward * np.exp(self.local_r_shaping)

        goal_achieved = False
        # if (pos_diff < 0.05) and (angle_diff < (np.pi/4)):
        if (pos_diff < 0.05):
            goal_achieved = True

        if self.illegal_contact:
            goal_achieved = False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def adroit_joints(self):
        adroit_joints = []
        for joint_name in self.finger_joint_names:
            joint_anchor = self.sim.data.get_joint_xanchor(joint_name)
            adroit_joints.append(joint_anchor)
        for site_name in self.finger_tip_site_names:
            finger_site = self.sim.data.site_xpos[self.sim.model.site_name2id(site_name)]
            adroit_joints.append(finger_site)
        adroit_joints = np.stack(adroit_joints, 0)
        return adroit_joints

    def affordance_sites(self, idx=0):
        affordance_sites = []
        for joint_name in self.finger_joint_names:
            dexycb_site = self.sim.data.site_xpos[self.sim.model.site_name2id( \
                    'ycb_joint_site_{}_'.format(str(idx).zfill(3)) + \
                    dex_ycb_utils._ADROIT_TO_MANO_MAP[joint_name])]
            affordance_sites.append(dexycb_site)
        for site_name in self.finger_tip_site_names:
            dexycb_site = self.sim.data.site_xpos[self.sim.model.site_name2id( \
                    'ycb_joint_site_{}_'.format(str(idx).zfill(3)) + \
                    dex_ycb_utils._ADROIT_TO_MANO_MAP[site_name])]
            affordance_sites.append(dexycb_site)
        affordance_sites = np.stack(affordance_sites, 0)
        return affordance_sites

    def get_obs(self):
        # qpos for hand and obj (original)
        qp = self.data.qpos.ravel()
        qp_obs_idx = self.old_hand_joint_ids
        qp_obs = qp[qp_obs_idx]

        # for obj
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        obj_mat  = self.data.body_xmat[self.obj_bid].reshape([-1, 3])
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        palm_mat = self.data.site_xmat[self.S_grasp_sid].reshape([-1, 3])
        # xpos for target
        target_pos = self.data.body_xpos[self.target_obj_bid].ravel()
        target_mat = self.data.body_xmat[self.target_obj_bid].reshape([-1, 3])
        # quat diffs
        obj_palm_diff_mat = np.dot(palm_mat.T, obj_mat)
        obj_palm_diff_quat = list(pyquaternion.Quaternion(matrix=obj_palm_diff_mat))
        obj_palm_diff_rot = quat2euler(obj_palm_diff_quat).ravel()
        obj_target_diff_mat = np.dot(target_mat.T, obj_mat)
        obj_target_diff_quat = list(pyquaternion.Quaternion(matrix=obj_target_diff_mat))
        obj_target_diff_rot = quat2euler(obj_target_diff_quat).ravel()
        palm_target_diff_mat = np.dot(target_mat.T, palm_mat)
        palm_target_diff_quat = list(pyquaternion.Quaternion(matrix=palm_target_diff_mat))
        palm_target_diff_rot = quat2euler(palm_target_diff_quat).ravel()
        # xpos for gripper
        gripper_pos = self.data.body_xpos[self.gripper_body_id]
        gripper_xquat = self.data.body_xquat[self.gripper_body_id]
        gripper_rot = quat2euler(gripper_xquat.ravel()).ravel()

        # qpos for slide finger (expanded)
        finger_slide_qp = np.array([self.data.qpos[self.finger_slide_joint_id]])

        ##### it's a bug, but only a compromise
        target_body_pos = self.data.body_xpos[self.target_obj_bid]
        target_body_xquat = self.data.body_xquat[self.target_obj_bid]
        target_body_rot = quat2euler(target_body_xquat.ravel()).ravel()

        return np.concatenate([qp_obs, \
                palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos, \
                obj_palm_diff_rot, obj_target_diff_rot, palm_target_diff_rot, \
                finger_slide_qp])

    def interped_init_joints(self, starting_t):
        ##### interpolate weights
        if starting_t < self.dexycb_id_list[0]:
            starting_t_floor = 0
            starting_t_ceil = self.dexycb_id_list[0]
        elif starting_t < self.dexycb_id_list[-1]:
            for i in range(len(self.dexycb_id_list) - 1):
                if (starting_t >= self.dexycb_id_list[i]) and \
                   (starting_t < self.dexycb_id_list[i+1]):
                    starting_t_floor = self.dexycb_id_list[i]
                    starting_t_ceil = self.dexycb_id_list[i+1]
                    break
        else:
            starting_t_floor = self.dexycb_id_list[-1]
            starting_t_ceil = self.dexycb_id_list[-1]

        if starting_t_ceil == starting_t_floor: # integer
            weight_floor = 0.
            weight_ceil = 1.
        else:
            weight_floor = (starting_t_ceil - starting_t) / (starting_t_ceil - starting_t_floor)
            weight_ceil = (starting_t - starting_t_floor) / (starting_t_ceil - starting_t_floor)

        ##### interpolate dex_joints
        if starting_t_floor == 0:
            dex_joints_floor = self.sim.data.qpos[self.old_hand_joint_ids]
        else:
            dex_joints_floor = self.dex_joints[starting_t_floor]
        dex_joints_ceil = self.dex_joints[starting_t_ceil]
        old_hand_joints = \
            dex_joints_floor * weight_floor + dex_joints_ceil * weight_ceil

        ##### interpolate obj pose_t
        if starting_t_floor == 0:
            pose_t_floor = self.pose_t_dict[starting_t_ceil] + self.manual_offset
        else:
            pose_t_floor = self.pose_t_dict[starting_t_floor] + self.manual_offset
        pose_t_ceil = self.pose_t_dict[starting_t_ceil] + self.manual_offset
        pose_t = pose_t_floor * weight_floor + pose_t_ceil * weight_ceil

        ##### interpolate obj pose_R
        if starting_t_floor == 0:
            pose_R_floor = self.pose_R_dict[starting_t_ceil]
        else:
            pose_R_floor = self.pose_R_dict[starting_t_floor]
        pose_R_ceil = self.pose_R_dict[starting_t_ceil]
        ceil_floor_diff = np.dot(pose_R_ceil, pose_R_floor.T)
        ceil_floor_diff = ceil_floor_diff + 0.5 * np.dot((np.eye(3) - np.dot(ceil_floor_diff, ceil_floor_diff.T)), ceil_floor_diff)
        ceil_floor_diff_quat = pyquaternion.Quaternion(matrix=ceil_floor_diff)
        if abs(1-ceil_floor_diff_quat[0]) < 1e-8:
            interp_quat = pyquaternion.Quaternion(axis=[1,0,0], radians=0.)
        else:
            interp_quat = pyquaternion.Quaternion( \
                    axis=ceil_floor_diff_quat.axis,
                    radians=ceil_floor_diff_quat.radians * weight_ceil)
        pose_R = np.dot(interp_quat.rotation_matrix, pose_R_floor)
        pose_R = pose_R + 0.5 * np.dot((np.eye(3) - np.dot(pose_R, pose_R.T)), pose_R)

        pose_quat = list(pyquaternion.Quaternion(matrix=pose_R))
        pose_rot = quat2euler(pose_quat).ravel()
        return old_hand_joints, pose_t, pose_rot

    def reset(self, seed=None):
        np.random.seed(seed)
        self.sim.reset()
        self.sim.forward()

        for i, rng in enumerate(self.sim.model.jnt_range):
            ##### if both are larger than 0, or both are smaller than 0
            if rng[0] * rng[1] > 1e-8:
                smaller_limit = rng[0] if abs(rng[0]) < abs(rng[1]) else rng[1]
                self.sim.data.qpos[i] = smaller_limit

            ##### position joints
            joint_name = self.sim.model.joint_names[i]
            if joint_name in ['finger_position_joint_ff_2', 'joint_range_rf_2_position']:
                smaller_limit = rng[0] if abs(rng[0]) < abs(rng[1]) else rng[1]
                self.sim.data.qpos[i] = smaller_limit
            if joint_name in ['finger_position_joint_th_2',  'finger_position_joint_ff_1', 'finger_position_joint_th_3']:
                larger_limit = rng[0] if abs(rng[0]) > abs(rng[1]) else rng[1]
                self.sim.data.qpos[i] = larger_limit

        ##### change virtual mount body position
        if self.virtual_confine:
            virtual_mount_id = self.sim.model.body_name2id('virtual_mount')
            self.sim.model.body_pos[virtual_mount_id] = self.constraint_pos
            self.sim.model.body_quat[virtual_mount_id] = np.array([1,0,0,0])

        ycb_bids = [self.sim.model.body_name2id(b) for b in self.sim.model.body_names if 'ycb_body' in b]
        xy_noise = np.random.uniform(-0.01, 0.01, [2])
        # xy_noise = np.array([0.01, -0.01])
        for ycb_bid in ycb_bids:
            self.sim.model.body_pos[ycb_bid][:2] = self.obj_init_pos[ycb_bid][:2] + xy_noise

        self.sim.data.qpos[self.robot_joint_indexes] = self.robot_init_qpos
        self.sim.data.qvel[:] = 0.

        np.random.seed(seed)

        self.sim.forward()

        ##### set illegal contact back to False
        self.illegal_contact = False
        self.current_gripper_action = 0.

        return self.get_obs()

    def set_starting_t(self, t):
        self.starting_t = t

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        return dict(qpos=qpos, qvel=qvel)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.viewer.cam.distance = 0.8
        self.sim.forward()

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) >= 30:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
