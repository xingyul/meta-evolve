
import gym
import os
import numpy as np
import pyquaternion
import io
import xml.dom.minidom
import xml.etree.ElementTree as ET
import copy

try:
    import mjcf_utils
except:
    import utils.mjcf_utils as mjcf_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generalized_relocate_unified( \
        interp_param_vector=[0.]*12, \
        return_xml_only=False, \
        finger_option=['ff', 'rf'], \
        phys_param_vector_all=[0.]*12, \
        **env_kwargs):
    return generalized_hand_env_unified( \
        interp_param_vector=interp_param_vector, \
        return_xml_only=return_xml_only, \
        base_env_name='relocate-v0', \
        finger_option=finger_option, \
        phys_param_vector_all=phys_param_vector_all, \
        **env_kwargs)


def generalized_hand_env_unified( \
        interp_param_vector=[0.]*12, \
        return_xml_only=False, \
        base_env_name='hammer-v0', \
        finger_option=['ff', 'rf'], \
        lf_shrink=1., \
        phys_param_vector_all=[0.]*12, \
        **env_kwargs):
    print_debug = False

    interp_param_vector = np.array([float(np.clip(i, 0., 1.)) for i in interp_param_vector])
    phys_param_vector_all = np.array(phys_param_vector_all)

    phys_param_vector = (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0]) * interp_param_vector + phys_param_vector_all[:, 0]

    ##### debug
    # interp_param_vector = [0.] * len(interp_param_vector)
    #####

    ##### vector to dict expansion
    phys_param_dict = {}
    interp_param_dict = {'cut_length': {}, 'joint_range': {}, 'merge': {}}
    finger_parts = ['proximal', 'middle', 'distal']
    finger_names = ['th', 'ff', 'mf', 'rf', 'lf']

    pointer = 0

    phys_param_dict['finger_len_th_proximal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_th_middle'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_th_distal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_ff_proximal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_ff_middle'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_ff_distal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_mf_proximal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_mf_middle'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_mf_distal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_rf_proximal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_rf_middle'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_rf_distal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_lf_proximal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_lf_middle'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['finger_len_lf_distal'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_4_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_4_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_3_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_3_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_3_position_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_3_position_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_2_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_2_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_2_position_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_2_position_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['actuation_th_2'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_1_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_1_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_0_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_th_0_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_3_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_3_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_2_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_2_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_2_position_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_2_position_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_1_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_1_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_1_position_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_1_position_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['actuation_ff_1'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_0_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_ff_0_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_3_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_3_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_2_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_2_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_2_position_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_2_position_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_1_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_1_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['actuation_rf_1'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_0_upper'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['joint_range_rf_0_lower'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_th_jaco'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_th_kinova'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_th_iiwa'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_ff_jaco'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_ff_kinova'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_ff_iiwa'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_mf'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_rf'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_rf_jaco'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['shape_lf'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_mf_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_mf_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_mf_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_euler_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_euler_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_th_euler_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_euler_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_euler_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_ff_euler_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_euler_x'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_euler_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['knuckle_pos_rf_euler_z'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['palm_jaco'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['palm_kinova'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['palm_iiwa'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['wrist'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['arm_length'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['arm_range'] = phys_param_vector[pointer]; pointer += 1

    joint_ids = ['3', '2', '1', '0']

    ##### read original env
    env = gym.make(base_env_name)
    env.reset()
    xml_string = env.env.model.get_xml()

    ##### change to absolute paths
    hand_env = ET.ElementTree(ET.fromstring(xml_string))
    compilers = [elem for elem in hand_env.iter() if (elem.tag == 'compiler') ]
    for compiler in compilers:
        if 'meshdir' in compiler.attrib:
            compiler.attrib['meshdir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/meshes')
        if 'texturedir' in compiler.attrib:
            compiler.attrib['texturedir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/textures')

    ##### switch to contact models
    default = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='default', attribs={'class': 'DC_Hand'})
    default_children = default.getchildren()
    for child in default_children:
        if child.tag == 'geom':
            child.attrib['group'] = '1'
            child.attrib['material'] = 'MatViz'
    bodies = [elem for elem in hand_env.iter() if (elem.tag == 'body') ]
    for body in bodies:
        body_children = body.getchildren()
        idx = 0
        while idx < len(body_children):
            child = body_children[idx]
            if 'geom' == child.tag:
                if 'name' in child.attrib:
                    if child.attrib['name'].startswith('V_'):
                        body.remove(child)
                    if child.attrib['name'].startswith('C_'):
                        if 'rgba' in child.attrib:
                            del child.attrib['rgba']
            idx += 1

    ##### remove tendon and sensor
    for elem in hand_env.getroot():
        if elem.tag in ['tendon', 'sensor']:
            tendon_parent = mjcf_utils.find_parent(hand_env.getroot(), elem)
            tendon_parent.remove(elem)
    ##### remove `user` in all general
    actuators = [elem for elem in hand_env.iter() if (elem.tag == 'general') ]
    for elem in actuators:
        if elem.tag == 'general':
            if 'user' in elem.attrib:
                del elem.attrib['user']

    ##### add armature to ARR joints
    joints = [e for e in hand_env.iter() if (e.tag =='joint')]
    joints = [j for j in joints if 'name' in j.attrib]
    joints = [j for j in joints if j.attrib['name'].startswith('ARR')]
    for joint in joints:
        joint.attrib['armature'] = '0.1'

    ##### shrink lf finger
    lfproximal = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='body', attribs={'name': 'lfproximal'})
    elems = [elem for elem in lfproximal.iter()]
    for elem in elems:
        if 'pos' in elem.attrib:
            old_pos = [float(i) for i in elem.attrib['pos'].split()]
            old_pos[-1] = old_pos[-1] * lf_shrink
            elem.attrib['pos'] = ' '.join([str(p) for p in old_pos])
        if 'size' in elem.attrib:
            old_size = [float(i) for i in elem.attrib['size'].split()]
            if len(old_size) == 2:
                old_size[-1] = old_size[-1] * lf_shrink
                elem.attrib['size'] = ' '.join([str(p) for p in old_size])

    ##### interpolate step 1
    # shrink fingers
    bodies = [elem for elem in hand_env.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if any([body.attrib['name'].startswith(f) for f in finger_names])]
    for finger_part_idx, finger_part in enumerate(finger_parts):
        finger_bodies = [body for body in bodies if body.attrib['name'].endswith(finger_part)]
        for body in finger_bodies:
            finger_name = body.attrib['name'].split(finger_part)[0]
            body_children = body.getchildren()
            # interp_param = interp_param_dict['finger_len_{}_{}'.format(finger_name, finger_part)]
            for child in body_children:
                phys_param = phys_param_dict['finger_len_{}_{}'.format(finger_name, finger_part)]
                '''
                if finger_part == 'distal':
                    scale = 1e-7 * interp_param + 1. * (1 - interp_param)
                elif finger_part == 'proximal':
                    if (finger_name == 'ff') or (finger_name == 'rf'):
                        scale = 0.044/0.045 * interp_param + 1. * (1 - interp_param)
                    elif finger_name == 'th':
                        scale = 0.044/0.038 * interp_param + 1. * (1 - interp_param)
                else: # middle
                    scale = 0.5 * interp_param + 1. * (1 - interp_param)
                '''
                scale = phys_param
                if 'pos' in child.attrib:
                    pos = [float(i) for i in child.attrib['pos'].split()]
                    pos = ' '.join([str(i) for i in (np.array(pos) * scale).tolist()])
                    child.attrib['pos'] = pos
                if 'size' in child.attrib:
                    size_split = child.attrib['size'].split()
                    size_split[1] = str(float(size_split[1]) * scale)
                    child.attrib['size'] = ' '.join(size_split)

    ##### add position servo for the three joints
    actuator = mjcf_utils.find_elements(hand_env.getroot(), tags='actuator')
    force_range = [-200, 200]
    damping = 5.
    # force_range = [-20, 20]
    ##### add joint for THJ3
    joint_range_upper = phys_param_dict['joint_range_th_3_position_upper']
    joint_range_lower = phys_param_dict['joint_range_th_3_position_lower']
    joint = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='joint', attribs={'name': 'THJ3'})
    joint_parent = mjcf_utils.find_parent(hand_env.getroot(), joint)
    # joint_range = np.array([np.pi/6+np.pi/24, np.pi/4*3]) * (1e-8 + interp_param)
    joint_range = [joint_range_lower, joint_range_upper]
    position_joint = mjcf_utils.new_joint(name='finger_position_joint_th_3', \
            type='hinge', pos=[0,0,0], range=joint_range, \
            axis=[1,0,0], solimplimit=[0.9998, 0.9999, 0.001, 0.01, 6], \
            solreflimit=[1e-8,1], damping=damping, limited='true', \
            armature=1, frictionloss=0, margin=0)
    joint_parent.insert(1, position_joint)
    new_position = mjcf_utils.new_element(tag='position', \
            joint='finger_position_joint_th_3', \
            name='act_finger_position_joint_th_3', \
            ctrllimited='true', ctrlrange=joint_range, \
            forcelimited='true', forcerange=force_range, kp=20,)
    actuator.append(new_position)
    ##### add joint for FFJ2
    joint_range_upper = phys_param_dict['joint_range_ff_2_position_upper']
    joint_range_lower = phys_param_dict['joint_range_ff_2_position_lower']
    joint = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='joint', attribs={'name': 'FFJ2'})
    joint_parent = mjcf_utils.find_parent(hand_env.getroot(), joint)
    # joint_range = np.array([np.pi/4, np.pi/6*5-np.pi/24]) * (1e-8 + interp_param)
    joint_range = [joint_range_lower, joint_range_upper]
    position_joint = mjcf_utils.new_joint(name='finger_position_joint_ff_2', \
            type='hinge', pos=[0,0,0], range=joint_range, \
            axis=[1,0,0], solimplimit=[0.9998, 0.9999, 0.001, 0.01, 6], \
            solreflimit=[1e-8,1], damping=damping, limited='true', \
            armature=1, frictionloss=0, margin=0)
    joint_parent.insert(1, position_joint)
    new_position = mjcf_utils.new_element(tag='position', \
            joint='finger_position_joint_ff_2', \
            name='act_finger_position_joint_ff_2', \
            ctrllimited='true', ctrlrange=joint_range, \
            forcelimited='true', forcerange=force_range, kp=20,)
    actuator.append(new_position)
    ##### add joint for RFJ2
    joint_range_upper = phys_param_dict['joint_range_rf_2_position_upper']
    joint_range_lower = phys_param_dict['joint_range_rf_2_position_lower']
    joint = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='joint', attribs={'name': 'RFJ2'})
    joint_parent = mjcf_utils.find_parent(hand_env.getroot(), joint)
    # joint_range = np.array([np.pi/4, np.pi/6*5]) * (1e-8 + interp_param)
    joint_range = [joint_range_lower, joint_range_upper]
    position_joint = mjcf_utils.new_joint(name='finger_position_joint_rf_2', \
            type='hinge', pos=[0,0,0], range=joint_range, \
            axis=[1,0,0], solimplimit=[0.9998, 0.9999, 0.001, 0.01, 6], \
            solreflimit=[1e-8,1], damping=damping, limited='true', \
            armature=1, frictionloss=0, margin=0)
    joint_parent.insert(1, position_joint)
    new_position = mjcf_utils.new_element(tag='position', \
            joint='finger_position_joint_rf_2', \
            name='act_finger_position_joint_rf_2', \
            ctrllimited='true', ctrlrange=joint_range, \
            forcelimited='true', forcerange=force_range, kp=20,)
    actuator.append(new_position)
    ##### add secondary joint for THJ2
    joint_range_upper = phys_param_dict['joint_range_th_2_position_upper']
    joint_range_lower = phys_param_dict['joint_range_th_2_position_lower']
    joint = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='joint', attribs={'name': 'THJ2'})
    joint_parent = mjcf_utils.find_parent(hand_env.getroot(), joint)
    # joint_range = np.array([np.pi/4, np.pi/6*5]) * (1e-8 + interp_param)
    joint_range = [joint_range_lower, joint_range_upper]
    position_joint = mjcf_utils.new_joint(name='finger_position_joint_th_2', \
            type='hinge', pos=[0,0,0], range=joint_range, \
            axis=[1,0,0], solimplimit=[0.9998, 0.9999, 0.001, 0.01, 6], \
            solreflimit=[1e-8,1], damping=damping, limited='true', \
            armature=1, frictionloss=0, margin=0)
    joint_parent.insert(1, position_joint)
    new_position = mjcf_utils.new_element(tag='position', \
            joint='finger_position_joint_th_2', \
            name='act_finger_position_joint_th_2', \
            ctrllimited='true', ctrlrange=joint_range, \
            forcelimited='true', forcerange=force_range, kp=20,)
    actuator.append(new_position)
    ##### add secondary joint for FFJ1
    joint_range_upper = phys_param_dict['joint_range_ff_1_position_upper']
    joint_range_lower = phys_param_dict['joint_range_ff_1_position_lower']
    joint = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='joint', attribs={'name': 'FFJ1'})
    joint_parent = mjcf_utils.find_parent(hand_env.getroot(), joint)
    # joint_range = np.array([np.pi/4, np.pi/6*5]) * (1e-8 + interp_param)
    joint_range = [joint_range_lower, joint_range_upper]
    position_joint = mjcf_utils.new_joint(name='finger_position_joint_ff_1', \
            type='hinge', pos=[0,0,0], range=joint_range, \
            axis=[1,0,0], solimplimit=[0.9998, 0.9999, 0.001, 0.01, 6], \
            solreflimit=[1e-8,1], damping=damping, limited='true', \
            armature=1, frictionloss=0, margin=0)
    joint_parent.insert(1, position_joint)
    new_position = mjcf_utils.new_element(tag='position', \
            joint='finger_position_joint_ff_1', \
            name='act_finger_position_joint_ff_1', \
            ctrllimited='true', ctrlrange=joint_range, \
            forcelimited='true', forcerange=force_range, kp=20,)
    actuator.append(new_position)

    # gradually fix joint
    joint_names = ['FFJ0', 'FFJ1', 'FFJ2', 'FFJ3', 'RFJ0', 'RFJ1', 'RFJ2', 'RFJ3', 'THJ0', 'THJ1', 'THJ2', 'THJ3', 'THJ4']
    target_joint_ranges = {
            'FFJ1': [-0.125, np.pi/2], \
            'RFJ1': [-0.125, np.pi/2], \
            'THJ2': [-np.pi/2, 0.125], \
            }
    for joint_name in joint_names:
        joint = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='joint', attribs={'name': joint_name})
        finger_name = joint.attrib['name'].split('J')[0].lower()
        joint_id = joint.attrib['name'].split('J')[1]
        # interp_param = interp_param_dict['joint_range_{}_{}'.format(finger_name, joint_id)]
        range_upper = phys_param_dict['joint_range_{}_{}_upper'.format(finger_name, joint_id)]
        range_lower = phys_param_dict['joint_range_{}_{}_lower'.format(finger_name, joint_id)]
        joint_range = [range_lower, range_upper]
        # joint_range_split = joint.attrib['range'].split()
        if joint.attrib['name'] in target_joint_ranges:
            target_range = np.array(target_joint_ranges[joint.attrib['name']])
        else:
            target_range = np.array([-1e-8, 1e-8])
        joint.attrib['range'] = ' '.join([str(r) for r in joint_range])

        '''
        joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(joint_range_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])
        '''

    # change joint range for joints
    actuators = [elem for elem in hand_env.iter() if (elem.tag == 'general') ]
    actuators = [actuator for actuator in actuators if 'name' in actuator.attrib]
    actuators = [actuator for actuator in actuators if actuator.attrib['name'].split('A_')[1] in joint_names]
    for actuator in actuators:
        joint_name = actuator.attrib['name'].split('A_')[1]
        joint = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='joint', attribs={'name': joint_name})
        joint_range = np.array([float(r) for r in joint.attrib['range'].split()])
        if joint_name in target_joint_ranges:
            finger_name = joint_name.split('J')[0].lower()
            joint_id = joint_name.split('J')[1]
            # interp_param = interp_param_dict['joint_range_{}_{}'.format(finger_name, joint_id)]
            phys_param = phys_param_dict['actuation_{}_{}'.format(finger_name, joint_id)]

            # scale = 1 * (1 - interp_param) + interp_param * 1e-8
            scale = phys_param
            actuator.attrib['gainprm'] = ' '.join([str(scale)] + ['0'] * 9)
            actuator.attrib['gaintype'] = 'fixed'
            actuator.attrib['biasprm'] = ' '.join(['0'] + [str(-scale)] + ['0'] * 8)

        control_range = joint_range
        actuator.attrib['ctrlrange'] = ' '.join([str(r) for r in control_range])

    # change all fingers' width
    geoms = [elem for elem in hand_env.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'type' in geom.attrib]
    geoms = [geom for geom in geoms if geom.attrib['type'] == 'capsule']
    geoms = [geom for geom in geoms if 'name' in geom.attrib]
    geoms = [geom for geom in geoms if any(['_{}'.format(f) in geom.attrib['name'] for f in finger_names])]
    for geom in geoms:
        geom_size_split = geom.attrib['size'].split()
        finger_name = geom.attrib['name'].split('_')[1][:2]
        finger_part = geom.attrib['name'].split('_')[1][2:]
        # interp_param = interp_param_dict['shape_{}'.format(finger_name)]
        if (finger_name == 'ff') or (finger_name == 'th'):
            phys_param_jaco = phys_param_dict['shape_{}_jaco'.format(finger_name)]
            phys_param_kinova = phys_param_dict['shape_{}_kinova'.format(finger_name)]
            phys_param_iiwa = phys_param_dict['shape_{}_iiwa'.format(finger_name)]
            phys_param = min(phys_param_jaco + phys_param_kinova + phys_param_iiwa, 1)
        else:
            phys_param = phys_param_dict['shape_{}'.format(finger_name)]
        size = float(geom_size_split[0])*(1 - phys_param) + 1e-5 * phys_param
        geom.attrib['size'] = ' '.join([str(size), geom_size_split[1]])

    # change palm's size and pos
    geoms = [elem for elem in hand_env.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'type' in geom.attrib]
    geoms = [geom for geom in geoms if 'size' in geom.attrib]
    geoms = [geom for geom in geoms if 'name' in geom.attrib]
    geoms = [geom for geom in geoms if ('metacarpal' in geom.attrib['name']) or \
            ('palm' in geom.attrib['name'])]
    # interp_param = interp_param_dict['palm']
    phys_param = min(phys_param_dict['palm_jaco'] + phys_param_dict['palm_kinova'] + phys_param_dict['palm_iiwa'], 1)
    for geom in geoms:
        geom_size_split = geom.attrib['size'].split()
        if '_palm0' in geom.attrib['name']:
            target_size = [1e-8, 1e-8, 1e-8]
            x_pos = 0.
        if '_palm1' in geom.attrib['name']:
            target_size = [1e-8, 1e-8, 1e-8]
            x_pos = 0.
        if '_lfmetacarpal' in geom.attrib['name']:
            target_size = [1e-8, 1e-8, 1e-8]
            x_pos = 0.017
        geom.attrib['size'] = ' '.join([
            str(float(geom_size_split[0])*(1-phys_param)+target_size[0]*phys_param),
            str(float(geom_size_split[1])*(1-phys_param)+target_size[1]*phys_param),
            str(float(geom_size_split[2])*(1-phys_param)+target_size[2]*phys_param),
            ])
        geom_pos_split = geom.attrib['pos'].split()
        geom.attrib['pos'] = ' '.join([\
            str(float(geom_pos_split[0])*(1-phys_param) + x_pos*phys_param), \
            geom_pos_split[1], geom_pos_split[2]])

    # change pos of four fingers
    bodies = [elem for elem in hand_env.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if 'pos' in body.attrib]
    body_names = ['thbase', 'ffknuckle', 'mfknuckle', 'rfknuckle']
    bodies = [body for body in bodies if \
            any([b in body.attrib['name'] for b in body_names])]
    # palm_interp_param = interp_param_dict['palm']
    palm_offset = -0.114668
    for body in bodies:
        name = body.attrib['name']

        body_pos_split = [float(p) for p in body.attrib['pos'].split()]
        body_pos_split = np.array(body_pos_split)

        finger_name = name[:2]
        param_x = phys_param_dict['knuckle_pos_{}_x'.format(finger_name)]
        param_y = phys_param_dict['knuckle_pos_{}_y'.format(finger_name)]
        param_z = phys_param_dict['knuckle_pos_{}_z'.format(finger_name)]
        body_pos = [param_x, param_y, param_z]

        # body_pos = body_pos_split * (1 - interp_param) + np.array(target_pos) * interp_param
        body.attrib['pos'] = ' '.join([str(p) for p in body_pos])

        if 'knuckle_pos_{}_euler_x'.format(finger_name) in phys_param_dict:
            ex = phys_param_dict['knuckle_pos_{}_euler_x'.format(finger_name)]
            ey = phys_param_dict['knuckle_pos_{}_euler_y'.format(finger_name)]
            ez = phys_param_dict['knuckle_pos_{}_euler_z'.format(finger_name)]

            if 'quat' in body.attrib:
                del body.attrib['quat']
            body.attrib['euler'] = ' '.join([str(ex), str(ey), str(ez)])
    ##### end of original interp

    ##### change (and interp) forearm
    geoms = [elem for elem in hand_env.getroot().iter() if (elem.tag == 'geom') ]
    for geom in geoms:
        if 'name' in geom.attrib:
            if 'forearm' in geom.attrib['name']:
                phys_param = phys_param_dict['arm_length']
                arm_length = 0.2 * (1 - phys_param) + \
                        1e-8 * phys_param
                geom.attrib['type'] = 'cylinder'
                geom.attrib['size'] = '0.03 {}'.format(arm_length)
                geom.attrib['pos'] = '0 0 {}'.format(arm_length)

    ##### interp wrist
    wrist_body = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='body', attribs={'name': 'wrist'})
    init_wrist_pos = np.array([0,0,0.396])
    interm_wrist_pos = np.array([0,0,0.45])
    target_wrist_pos = np.array([0,0,1e-8])

    arm_param = phys_param_dict['arm_length']
    wrist_param = phys_param_dict['wrist']
    wrist_pos = init_wrist_pos * (1 - wrist_param) + \
            interm_wrist_pos * wrist_param + \
            (target_wrist_pos - interm_wrist_pos) * arm_param
    wrist_body.attrib['pos'] = ' '.join([str(i) for i in wrist_pos])
    wrist_geom = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='geom', attribs={'name': 'C_wrist'})
    wrist_geom.attrib['type'] = 'cylinder'

    ##### interp wrist joints
    for joint_names, interp_name in zip([['WRJ0', 'WRJ1'], \
            ['ARRx', 'ARRy', 'ARRz', 'ARTx', 'ARTy', 'ARTz']], ['wrist', 'arm_range']):
        phys_param = phys_param_dict[interp_name]
        for joint_name in joint_names:
            joint = mjcf_utils.find_elements(hand_env.getroot(), \
                    tags='joint', attribs={'name': joint_name})
            if joint is None:
                continue
            joint_range = [float(j) for j in joint.attrib['range'].split()]
            joint_range = np.array(joint_range) * (1.+1e-8-phys_param)
            joint.attrib['range'] = ' '.join([str(r) for r in joint_range])
            joint.attrib['solimplimit'] = '0.999 0.9999 0.001 0.01 6'

    ##### adjust palm position
    palm_phys_param = min(phys_param_dict['palm_jaco'] + phys_param_dict['palm_kinova'] + phys_param_dict['palm_iiwa'], 1)
    palm = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='body', attribs={'name': 'palm'})
    initial_pos = np.array([float(j) for j in palm.attrib['pos'].split()])
    target_pos = np.array([0, 0.0, 0.0])
    pos = initial_pos * (1 - palm_phys_param) + \
            target_pos * palm_phys_param
    palm.attrib['pos'] = ' '.join([str(i) for i in pos.tolist()])

    ##### ensure no collision with wrist and forearm
    contact = mjcf_utils.find_elements(hand_env.getroot(), tags='contact')
    new_exclude = mjcf_utils.new_element(tag='exclude', name=None, \
            body1='palm', body2='forearm')
    contact.append(new_exclude)
    new_exclude = mjcf_utils.new_element(tag='exclude', name=None, \
            body1='palm', body2='wrist')
    contact.append(new_exclude)

    ##### add new geom for palm
    # jaco
    pos = np.array([0.0, 0., 0.0])
    quat = [-0.707107, 0.707107, 0, 0]
    quat_rot = pyquaternion.Quaternion(axis=[0, 1, 0], radians=-0.18246501911627286) # to make it facing straight
    quat = list(quat_rot * pyquaternion.Quaternion(quat))
    new_geom = mjcf_utils.new_element(tag='geom', name='gripper0_hand_visual_jaco', \
            mesh='gripper0_hand_3finger',
            type='mesh', margin=0, pos=pos, quat=quat, \
            rgba=[0.05, 0.05, 0.05, 1])
    new_geom.attrib['class'] = 'D_Vizual'
    palm.append(new_geom)
    new_geom = mjcf_utils.new_element(tag='geom', name='gripper0_hand_ring_visual_jaco', \
            mesh='gripper0_ring_small', \
            type='mesh', margin=0, pos=pos, quat=[0, 0, 0.707107, 0.707107],
            rgba=[0.88, 0.86, 0.86, 1])
            # type='mesh', margin=0, pos=pos, quat=[0, 0, 0.707107, 0.707107])
    new_geom.attrib['class'] = 'D_Vizual'
    palm.append(new_geom)
    # kinova
    new_geom = mjcf_utils.new_element(tag='geom', \
            name='robotiq_85_arg2f_base_link_collision_kinova', \
            mesh='robotiq_85_arg2f_base_link', type='mesh', margin=0, \
            pos=pos, quat=[0.707107, 0.707107, 0, 0], rgba=[0.1, 0.1, 0.1, 1], group=1)
    new_geom.attrib['class'] = 'DC_Hand'
    palm.append(new_geom)
    # iiwa
    new_geom = mjcf_utils.new_element(tag='geom', \
            name='robotiq_140_arg2f_base_link_collision_iiwa', \
            mesh='robotiq_140_arg2f_base_link', type='mesh', margin=0, \
            pos=pos, quat=[0.707107, 0.707107, 0, 0], rgba=[0.1, 0.1, 0.1, 1], group=1)
    new_geom.attrib['class'] = 'DC_Hand'
    palm.append(new_geom)
    ##### add new mesh geom for th, ff, rf
    quats = {'ff': [0, np.sqrt(0.5), 0, np.sqrt(0.5)], \
             'rf': [0, np.sqrt(0.5), 0, np.sqrt(0.5)], \
             'th': [np.sqrt(0.5), 0, -np.sqrt(0.5), 0], }
    # jaco
    for finger_name in ['ff', 'rf', 'th']:
        quat = quats[finger_name]

        proximal = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}proximal'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='gripper0_{}_proximal_collision_jaco'.format(finger_name), \
                mesh='gripper0_{}_proximal'.format(finger_name), \
                type='mesh', margin=0, quat=quat, \
                rgba=[0.88, 0.86, 0.86, 1])
        new_geom.attrib['class'] = 'DC_Hand'
        proximal.append(new_geom)

        middle = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}middle'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='gripper0_{}_distal_collision_jaco'.format(finger_name), \
                mesh='gripper0_{}_distal'.format(finger_name), \
                type='mesh', margin=0, quat=quat, \
                rgba=[0.88, 0.86, 0.86, 1])
        new_geom.attrib['class'] = 'DC_Hand'
        middle.append(new_geom)

    # kinova
    phys_param = phys_param_dict['shape_{}_kinova'.format(finger_name)]
    for finger_name in ['ff', 'th']:
        quat = quats[finger_name]

        proximal = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}proximal'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_inner_knuckle_collision_kinova'.format(finger_name), \
                mesh='robotiq_arg2f_85_inner_knuckle', \
                rgba=[0.1, 0.1, 0.1, 1], \
                # quat=[0.9238795325112867, 0.3826834323650898, 0, 0], \
                quat=[1, 0.36, 0, 0], \
                type='mesh', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        proximal.append(new_geom)

        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_outer_finger_collision_kinova'.format(finger_name), \
                mesh='robotiq_arg2f_85_outer_finger', \
                rgba=[0.1, 0.1, 0.1, 1], \
                pos=[0., 0.035, 0.0169187] if finger_name == 'ff' else \
                [0, -0.035, 0.0169187], \
                quat=[1, 0.38, 0, 0] if finger_name == 'ff' else [1, -0.25, 0, 0], \
                type='mesh', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        proximal.append(new_geom)

        scale = (1 - phys_param) * 0.01 + phys_param * 1
        middle = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}middle'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_inner_finger_collision_kinova'.format(finger_name), \
                size=[0.01*scale, 0.0035*scale, 0.02*scale], \
                pos=[0, -0.0035 if finger_name == 'ff' else 0.0035, 0.0175], \
                rgba=[0.1, 0.1, 0.1, 1], \
                type='box', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        middle.append(new_geom)
    # iiwa
    phys_param = phys_param_dict['shape_{}_iiwa'.format(finger_name)]
    for finger_name in ['ff', 'th']:
        quat = quats[finger_name]

        proximal = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}proximal'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_inner_knuckle_collision_iiwa'.format(finger_name), \
                mesh='robotiq_arg2f_140_inner_knuckle', \
                rgba=[0.1, 0.1, 0.1, 1], \
                quat=[0.7071067811865476, 0.7071067811865475, 0, 0], \
                type='mesh', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        proximal.append(new_geom)

        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_outer_finger_collision_iiwa'.format(finger_name), \
                mesh='robotiq_arg2f_140_outer_finger', \
                rgba=[0.1, 0.1, 0.1, 1], \
                pos=[0., 0.035, 0.0169187] if finger_name == 'ff' else \
                [0, -0.03, 0.019], \
                quat=[0, 0, 1, 1] if finger_name == 'ff' else [1, 1, 0, 0], \
                type='mesh', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        proximal.append(new_geom)

        scale = (1 - phys_param) * 0.01 + phys_param * 1
        middle = mjcf_utils.find_elements(hand_env.getroot(), \
                tags='body', attribs={'name': '{}middle'.format(finger_name)})
        new_geom = mjcf_utils.new_element(tag='geom', \
                name='{}_inner_finger_collision_iiwa'.format(finger_name), \
                size=[0.01*scale, 0.0035*scale, 0.02*scale], \
                pos=[0, -0.0035 if finger_name == 'ff' else 0.0035, 0.0175], \
                rgba=[0.1, 0.1, 0.1, 1], \
                type='box', margin=0)
        new_geom.attrib['class'] = 'DC_Hand'
        middle.append(new_geom)
    ##### add new mesh geom for th, ff, rf

    ##### resolve abs path
    compiler = mjcf_utils.find_elements(hand_env.getroot(), tags='compiler')
    asset = mjcf_utils.find_elements(hand_env.getroot(), tags='asset')
    meshdir = compiler.attrib['meshdir']
    texturedir = compiler.attrib['texturedir']
    for elem in asset:
        if 'file' in elem.attrib:
            elem.attrib['file'] = os.path.join( \
                    compiler.attrib['{}dir'.format(elem.tag)], \
                    elem.attrib['file'])

    ##### make sure hand meshes can be imported
    meshes = [elem for elem in hand_env.getroot().iter() if (elem.tag == 'mesh') ]
    for mesh in meshes:
        mesh.attrib['scale'] = '1 1 1'

    ##### prevent contact penetrating and harden joint limit
    geoms = [e for e in hand_env.getroot().iter() if (e.tag =='geom')]
    geoms = [g for g in geoms if 'name' in g.attrib]
    for geom in geoms:
        geom.attrib['margin'] = '0'
        geom.attrib['solimp'] = '0.999 0.9999 0.001 0.01 6'
        geom.attrib['solref'] = '2e-2 1'
    default = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='default', attribs={'class': 'main'})
    default_geom = [g for g in default if g.tag == 'geom'][0]
    default_geom.attrib['margin'] = '0'
    joints = [e for e in hand_env.getroot().iter() if (e.tag =='joint')]
    joints = [j for j in joints if 'name' in j.attrib]
    for joint in joints:
        joint.attrib['margin'] = '0'
        joint.attrib['solimplimit'] = '0.999 0.9999 0.001 0.01 6'
        joint.attrib['solreflimit'] = '2e-2 1'

    ##### interpolate color of the hands to robot gripper
    viz_material = mjcf_utils.find_elements(hand_env.getroot(), \
            tags='material', attribs={'name': 'MatViz'})
    phys_param = phys_param_dict['arm_length']
    init_rgba = [float(c) for c in viz_material.attrib['rgba'].split()]
    target_rgba = np.array([0.499, 0.499, 0.499, 1]) # rethink's rgba
    rgba = np.array(init_rgba) * (1 - phys_param) + \
            target_rgba * phys_param
    # viz_material.attrib['rgba'] = ' '.join([str(c) for c in rgba])

    ##### add assets
    # arm assets
    for robot_name in ['jaco', 'kinova3', 'iiwa']:
        asset = mjcf_utils.find_elements(hand_env.getroot(), tags='asset')
        with open('models/assets/robots/{}/robot.xml'.format(robot_name), 'r') as f:
            xml_string = f.read()
        jaco_tree = ET.ElementTree(ET.fromstring(xml_string))
        jaco_asset = mjcf_utils.find_elements(jaco_tree.getroot(), tags='asset')
        for elem in jaco_asset:
            if 'file' in elem.attrib:
                elem.attrib['file'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/assets/robots/{}'.format(robot_name), elem.attrib['file'])
            asset.append(elem)
    # jaco
    phys_param = phys_param_dict['palm_jaco']
    length_scale = (1 - phys_param) * 0.05 + phys_param * 1
    width_scale = (1 - phys_param) * 0.5 + phys_param * 1
    new_mesh = mjcf_utils.new_element(tag='mesh', name='gripper0_hand_3finger', \
            scale=[width_scale, width_scale, length_scale], \
            file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/jaco_three_finger_gripper/hand_3finger.stl')))
    asset.append(new_mesh)
    new_mesh = mjcf_utils.new_element(tag='mesh', name='gripper0_ring_small', \
            scale=[width_scale, width_scale, length_scale], \
            file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/jaco_three_finger_gripper/ring_small.stl')))
    asset.append(new_mesh)
    # kinova
    phys_param = phys_param_dict['palm_kinova']
    length_scale = (1 - phys_param) * 0.05 + phys_param * 1
    width_scale = (1 - phys_param) * 0.8 + phys_param * 1
    new_mesh = mjcf_utils.new_element(tag='mesh', \
            name='robotiq_85_arg2f_base_link', \
            scale=[width_scale, width_scale, length_scale], \
            file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_85_gripper/robotiq_arg2f_base_link.stl')))
    asset.append(new_mesh)
    # iiwa
    phys_param = phys_param_dict['palm_iiwa']
    length_scale = (1 - phys_param) * 0.05 + phys_param * 1
    width_scale = (1 - phys_param) * 0.8 + phys_param * 1
    new_mesh = mjcf_utils.new_element(tag='mesh', \
            name='robotiq_140_arg2f_base_link', \
            scale=[width_scale, width_scale, length_scale], \
            file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_140_gripper/robotiq_arg2f_base_link.stl')))
    asset.append(new_mesh)

    # jaco
    for finger_name in ['th', 'ff', 'rf']:
        phys_param = phys_param_dict['shape_{}_jaco'.format(finger_name)]
        width_scale = (1 - phys_param) * 0.02 + phys_param * 1
        length_scale = (1 - phys_param) * 0.1 + phys_param * 1

        new_mesh = mjcf_utils.new_element(tag='mesh', name='gripper0_{}_proximal'.format(finger_name), \
                scale=[length_scale, width_scale, width_scale],
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/jaco_three_finger_gripper/finger_proximal.stl')))
        asset.append(new_mesh)

        new_mesh = mjcf_utils.new_element(tag='mesh', name='gripper0_{}_distal'.format(finger_name), \
                scale=[length_scale, width_scale, width_scale],
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/jaco_three_finger_gripper/finger_distal.stl')))
        asset.append(new_mesh)

    # kinova
    for finger_name in ['th', 'ff']:
        phys_param = phys_param_dict['shape_{}_kinova'.format(finger_name)]
        original_scale = np.array([0.001, 0.001, 0.001])
        width_scale = (1 - phys_param) * 0.02 + phys_param * 1
        length_scale = (1 - phys_param) * 0.1 + phys_param * 1
        scale = np.array([length_scale, width_scale, width_scale]) * \
                original_scale

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_85_inner_knuckle', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_85_gripper/robotiq_arg2f_85_inner_knuckle.stl')))
        asset.append(new_mesh)

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_85_outer_finger', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_85_gripper/robotiq_arg2f_85_outer_finger.stl')))
        asset.append(new_mesh)

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_85_inner_finger', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_85_gripper/robotiq_arg2f_85_inner_finger.stl')))
        asset.append(new_mesh)

    # iiwa
    for finger_name in ['th', 'ff']:
        phys_param = phys_param_dict['shape_{}_iiwa'.format(finger_name)]
        original_scale = np.array([1, 1, 1])
        width_scale = (1 - phys_param) * 0.02 + phys_param * 1
        length_scale = (1 - phys_param) * 0.1 + phys_param * 1
        scale = np.array([length_scale, width_scale, width_scale]) * \
                original_scale

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_140_inner_knuckle', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_140_gripper/robotiq_arg2f_140_inner_knuckle.stl')))
        asset.append(new_mesh)

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_140_outer_finger', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_140_gripper/robotiq_arg2f_140_outer_finger.stl')))
        asset.append(new_mesh)

        new_mesh = mjcf_utils.new_element(tag='mesh', \
                name='robotiq_arg2f_140_inner_finger', \
                scale=scale,
                file=os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/assets/grippers/meshes/robotiq_140_gripper/robotiq_arg2f_140_inner_finger.stl')))
        asset.append(new_mesh)
    ##### add assets

    if return_xml_only: # for only using in robosuite envs
        ##### remove unused lights, cameras, bodies and geoms
        worldbody = mjcf_utils.find_elements(hand_env.getroot(), tags='worldbody')
        children = [elem for elem in worldbody]
        for elem in children:
            remove = False
            if elem.tag != 'body':
                remove = True
            elif elem.attrib['name'] in ['table', 'vive_tracker']:
                    remove = True
            if 'relocate' in base_env_name: # special for relocate env
                if elem.tag == 'site':
                    if 'name' in elem.attrib:
                        if elem.attrib['name'] == 'target':
                            remove = False
            if remove:
                worldbody.remove(elem)

        return hand_env

    with io.StringIO() as string:
        string.write(ET.tostring(hand_env.getroot(), encoding="unicode"))
        xml_string = string.getvalue()
        parsed_xml = xml.dom.minidom.parseString(xml_string)
        xml_string = parsed_xml.toprettyxml(newl="", )

    env = gym.make('generalized-{}'.format(base_env_name), \
            xml_string=xml_string, **env_kwargs)
    env.reset()

    return env



