
import gym
import os
import io
import numpy as np
import pyquaternion
import xml.etree.ElementTree as ET
import xml.dom.minidom
import copy

try:
    import mjcf_utils
except:
    import utils.mjcf_utils as mjcf_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generalized_hammer(interp_param_vector=0., \
        return_xml_only=False, \
        tmp_xml_dir='/tmp/', \
        **env_kwargs):
    return generalized_hand_env(interp_param_vector=interp_param_vector, \
        return_xml_only=return_xml_only, \
        base_env_name='hammer-v0', \
        tmp_xml_dir=tmp_xml_dir, \
        **env_kwargs)

def generalized_relocate(interp_param_vector=0., \
        return_xml_only=False, \
        tmp_xml_dir='/tmp/', \
        **env_kwargs):
    return generalized_hand_env(interp_param_vector=interp_param_vector, \
        return_xml_only=return_xml_only, \
        base_env_name='relocate-v0', \
        tmp_xml_dir=tmp_xml_dir,
        **env_kwargs)

def generalized_door(interp_param_vector=0., \
        return_xml_only=False, \
        tmp_xml_dir='/tmp/', \
        **env_kwargs):
    return generalized_hand_env(interp_param_vector=interp_param_vector, \
        return_xml_only=return_xml_only, \
        base_env_name='door-v0', \
        tmp_xml_dir=tmp_xml_dir, \
        **env_kwargs)


def generalized_hand_env(interp_param_vector=0., \
        return_xml_only=False, \
        base_env_name='hammer-v0', \
        tmp_xml_dir='/tmp/', \
        phase_change=0.5, \
        **env_kwargs):
    print_debug = False

    interp_param_vector = [float(np.clip(i, 0., 1.)) for i in interp_param_vector]

    ##### vector to dict expansion
    finger_parts = ['proximal', 'middle', 'distal']
    finger_names = ['th', 'ff', 'mf', 'rf', 'lf']
    interp_param_dict = {}
    pointer = 0
    interp_param_dict['finger_len_th_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_th_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_th_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_mf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_mf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_mf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_rf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_rf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_rf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_lf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_lf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_len_lf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_th_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_th_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_th_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_ff_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_ff_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_ff_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_mf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_mf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_mf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_rf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_rf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_rf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_lf_proximal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_lf_middle'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['finger_width_lf_distal'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_th_4'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_th_3'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_th_2'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_th_1'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_th_0'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_ff_3'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_ff_2'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_ff_1'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_ff_0'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_mf_3'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_mf_2'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_mf_1'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_mf_0'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_rf_3'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_rf_2'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_rf_1'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_rf_0'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_lf_4'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_lf_3'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_lf_2'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_lf_1'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['joint_range_lf_0'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_angle_th'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_pos_th'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_pos_ff'] = [interp_param_vector[pointer], 0, interp_param_vector[pointer+1]]; pointer +=2
    interp_param_dict['knuckle_angle_ff'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_pos_mf'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_pos_rf'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_pos_lf'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['knuckle_angle_lf'] = interp_param_vector[pointer]; pointer +=1
    interp_param_dict['palm'] = interp_param_vector[pointer]; pointer +=1


    env = gym.make(base_env_name) #, disable_env_checker=True)
    # env.reset()
    xml_string = env.env.model.get_xml()

    ##### change to absolute paths
    tree = ET.ElementTree(ET.fromstring(xml_string))
    compilers = [elem for elem in tree.iter() if (elem.tag == 'compiler') ]
    for compiler in compilers:
        if 'meshdir' in compiler.attrib:
            compiler.attrib['meshdir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/meshes')
        if 'texturedir' in compiler.attrib:
            compiler.attrib['texturedir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                    'dependencies/Adroit/resources/textures')

    ##### switch to contact models
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    for body in bodies:
        body_children = list(body)
        idx = 0
        while idx < len(body_children):
            child = body_children[idx]
            if 'geom' == child.tag:
                if 'name' in child.attrib:
                    if child.attrib['name'].startswith('V_'):
                        body.remove(child)
            idx += 1

    # change rgba
    forearm = mjcf_utils.find_elements(tree.getroot(), tags='body', attribs={'name': 'forearm'})
    geoms = [elem for elem in forearm.iter() if (elem.tag == 'geom') ]
    for geom in geoms:
        geom.attrib['rgba'] = '1 1 1 1'
    rgba_dict = {'th': [0.25,0.25,0.25,1], 'ff':[1,0,0,1], 'mf':[1,1,0,1], 'rf':[0,1,0,1], 'lf':[0,0,1,1]}
    for finger_name in rgba_dict:
        finger_knuckle = mjcf_utils.find_elements(tree.getroot(), \
                tags='body', attribs={'name': '{}proximal'.format(finger_name)})
        geoms = [elem for elem in finger_knuckle.iter() if (elem.tag == 'geom') ]
        for geom in geoms:
            geom.attrib['rgba'] = ' '.join([str(c) for c in rgba_dict[finger_name]])

    # change lfmetacarpal shape
    lfmetacarpal_geom = mjcf_utils.find_elements(tree.getroot(), \
            tags='geom', attribs={'name': 'C_lfmetacarpal'})
    lfmetacarpal_geom.attrib['size'] = '0.011 0.0111 0.033'
    lfmetacarpal_geom.attrib['pos'] = '-0.015 0 0.01'

    # shrink other three fingers
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    other_fingers = ['mf', 'rf', 'lf']
    bodies = [body for body in bodies if any([body.attrib['name'].startswith(f) for f in other_fingers + ['th']])]
    finger_parts = ['proximal', 'middle', 'distal']
    bodies = [body for body in bodies if any([body.attrib['name'].endswith(f) for f in finger_parts])]
    for body in bodies:
        body_children = list(body)
        finger_name = body.attrib['name'][:2]
        finger_part = body.attrib['name'][2:]
        interp_param = interp_param_dict['finger_len_{}_{}'.format(finger_name, finger_part)]
        target_ratio = 0.88 if 'th' == finger_name else 0.
        shrink_ratio = target_ratio * interp_param + 1. * (1 - interp_param)
        for child in body_children:
            if 'pos' in child.attrib:
                pos_split = child.attrib['pos'].split()
                pos_split[2] = str(float(pos_split[2])*shrink_ratio)
                child.attrib['pos'] = ' '.join(pos_split)
            if 'size' in child.attrib:
                size_split = child.attrib['size'].split()
                size_split[1] = str(max(float(size_split[1])*shrink_ratio, 1e-8))
                child.attrib['size'] = ' '.join(size_split)

    # fix lf4 joint
    joints = [elem for elem in tree.iter() if (elem.tag == 'joint') ]
    joints = [joint for joint in joints if 'name' in joint.attrib]

    interp_param = interp_param_dict['joint_range_lf_4']
    joint = [joint for joint in joints if joint.attrib['name'] == 'LFJ4'][0]
    joint_range_split = joint.attrib['range'].split()
    joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0]) * max(1 - interp_param, 1e-8)),
            str(float(joint_range_split[1]) * max(1 - interp_param, 1e-8))
            ])

    # gradually fix joint
    joint_names = ['FFJ0', 'FFJ1', 'FFJ3', 'THJ0', 'THJ1', 'THJ2', 'THJ4', 'MFJ3', 'MFJ1', 'MFJ0', \
            'RJF3', 'RFJ2', 'RFJ1', 'RFJ0', 'LFJ4', 'LFJ3', 'LFJ1', 'LFJ0']
    joints_to_fix = [joint for joint in joints if any([(j in joint.attrib['name']) for j in joint_names])]
    for joint in joints_to_fix:
        joint_range_split = joint.attrib['range'].split()
        finger_name = joint.attrib['name'].split('J')[0].lower()
        joint_id = joint.attrib['name'].split('J')[1]
        interp_param = interp_param_dict['joint_range_{}_{}'.format(finger_name, joint_id)]
        joint.attrib['range'] = ' '.join([
                str(float(joint_range_split[0]) * (max(1 - interp_param, 1e-8))),
                str(float(joint_range_split[1]) * (max(1 - interp_param, 1e-8)))
                ])

    # change joint range for FFJ2
    joint_names = ['FFJ2', 'THJ3', 'LFJ2', 'MFJ2']
    joints_to_change = [joint for joint in joints if \
            any([j in joint.attrib['name'] for j in joint_names])]
    target_range = np.array([np.pi/2-np.pi/4, np.pi/2+np.pi/4])
    for joint in joints_to_change:
        finger_name = joint.attrib['name'].split('J')[0].lower()
        joint_id = joint.attrib['name'].split('J')[1]
        interp_param = interp_param_dict['joint_range_{}_{}'.format(finger_name, joint_id)]

        joint_range_split = joint.attrib['range'].split()
        joint.attrib['range'] = ' '.join([
            str(float(joint_range_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(joint_range_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])
    actuators = [elem for elem in tree.iter() if (elem.tag == 'general') ]
    actuators = [actuator for actuator in actuators if 'name' in actuator.attrib]
    actuators_to_change = [actuator for actuator in actuators if \
            any([j in actuator.attrib['name'] for j in joint_names])]
    target_range = target_range + np.array([-0.01, 0.01])
    for actuator in actuators_to_change:
        finger_name = actuator.attrib['name'].split('A_')[1].split('J')[0].lower()
        joint_id = actuator.attrib['name'].split('A_')[1].split('J')[1]
        interp_param = interp_param_dict['joint_range_{}_{}'.format(finger_name, joint_id)]

        ctrlrange_split = actuator.attrib['ctrlrange'].split()
        actuator.attrib['ctrlrange'] = ' '.join([
            str(float(ctrlrange_split[0])*(1 - interp_param) + target_range[0]*interp_param),
            str(float(ctrlrange_split[1])*(1 - interp_param) + target_range[1]*interp_param)
            ])

    # change orientation for thbase
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    body = [body for body in bodies if body.attrib['name'] == 'thbase'][0]
    original_quat = pyquaternion.Quaternion([float(q) for q in body.attrib['quat'].split()])
    target_quat = pyquaternion.Quaternion([1, 0, 0, 0])
    interp_param = interp_param_dict['knuckle_angle_th']
    quat = pyquaternion.Quaternion.slerp(original_quat, target_quat, interp_param)
    body.attrib['quat'] = ' '.join([str(quat[0]), str(quat[1]), str(quat[2]), str(quat[3])])

    # change all fingers' width
    target_finger_width = 0.007
    geoms = [elem for elem in tree.iter() if (elem.tag == 'geom') ]
    geoms = [geom for geom in geoms if 'type' in geom.attrib]
    geoms = [geom for geom in geoms if geom.attrib['type'] == 'capsule']
    geoms = [geom for geom in geoms if 'name' in geom.attrib]
    geoms = [geom for geom in geoms if any(['_{}'.format(f) in geom.attrib['name'] for f in finger_names])]
    for geom in geoms:
        finger_name = geom.attrib['name'].split('_')[1][:2]
        finger_part = geom.attrib['name'].split('_')[1][2:]
        interp_param = interp_param_dict['finger_width_{}_{}'.format(finger_name, finger_part)]

        geom_size_split = geom.attrib['size'].split()
        if ('_th' in geom.attrib['name']) or ('_ff' in geom.attrib['name']):
            size = float(geom_size_split[0])*(1 - interp_param) + \
                    target_finger_width * interp_param
        elif any(['_{}'.format(f) in geom.attrib['name'] for f in other_fingers]):
            size = float(geom_size_split[0])*(1 - interp_param) + 1e-8*interp_param
        geom.attrib['size'] = ' '.join([str(size), geom_size_split[1]])

    # change pos of four fingers
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if 'pos' in body.attrib]
    body_names = ['thbase', 'ffknuckle', 'mfknuckle', 'rfknuckle', 'lfknuckle']
    bodies = [body for body in bodies if \
            any([b in body.attrib['name'] for b in body_names])]
    target_poses = {'thbase': np.array([0., -0.01, 0.0]), \
            'ffknuckle': np.array([0, -0.01, 0.08]), \
            'lfknuckle': np.array([-0.016,0.0,0.036]), \
            'mfknuckle': np.array([0.0,0.0,0.08]), \
            'rfknuckle': np.array([-0.011,-0.01,0.08]), \
            }
    for body in bodies:
        name = body.attrib['name']
        finger_name = name[:2]
        interp_param = np.array(interp_param_dict['knuckle_pos_{}'.format(finger_name)])

        target_pos = target_poses[name]

        body_pos_split = [float(p) for p in body.attrib['pos'].split()]
        body_pos_split = np.array(body_pos_split)

        body_pos = body_pos_split * (1 - interp_param) + target_pos * interp_param
        body.attrib['pos'] = ' '.join([str(p) for p in body_pos])

    ##### change angle of ff and lf knuckles
    bodies = [elem for elem in tree.iter() if (elem.tag == 'body') ]
    bodies = [body for body in bodies if 'pos' in body.attrib]
    body_names = ['ffknuckle', 'lfknuckle']
    bodies = [body for body in bodies if \
            any([b in body.attrib['name'] for b in body_names])]
    target_angles = {'ffknuckle': np.arctan2(0.033, 0.061), \
            'lfknuckle': -np.arctan2(0.033, 0.061)}
    for body in bodies:
        name = body.attrib['name']
        finger_name = name[:2]
        interp_param = interp_param_dict['knuckle_angle_{}'.format(finger_name)]
        target_angle = target_angles[name]
        angle = 0. * (1 - interp_param) + target_angle * interp_param

        quat = list(pyquaternion.Quaternion(axis=[0,1,0], radians=angle))
        body.attrib['quat'] = ' '.join([str(q) for q in quat])

    ##### palm shape
    target_expand_length = 0
    target_shrink_width = 0.02
    interp_param = interp_param_dict['palm']
    expand_length = 0. * (1 - interp_param) + target_expand_length / 2 * interp_param
    shrink_width = 0. * (1 - interp_param) + target_shrink_width / 2 * interp_param
    for geom_name in ['C_lfmetacarpal', 'C_palm0', 'C_palm1']:
        geom = mjcf_utils.find_elements(tree.getroot(), \
                tags='geom', attribs={'name': geom_name})
        size_x_change = -shrink_width
        pos_x_change = -shrink_width if geom_name == 'C_palm0' else shrink_width
        geom.attrib['size'] = ' '.join([ \
                str(float(geom.attrib['size'].split()[0]) + size_x_change), \
                geom.attrib['size'].split()[1], \
                str(float(geom.attrib['size'].split()[-1]) + expand_length)])
        geom.attrib['pos'] = ' '.join([ \
                str(float(geom.attrib['pos'].split()[0]) + pos_x_change), \
                geom.attrib['pos'].split()[1], \
                str(float(geom.attrib['pos'].split()[-1]) + expand_length)])

    ##### exclude contact
    contact = mjcf_utils.find_elements(tree.getroot(), tags='contact')
    for body1 in ['palm', 'lfmetacarpal']:
        for finger_name in ['ff', 'mf', 'rf', 'lf']:
            for finger_part in ['knuckle', 'proximal', 'middle', 'distal']:
                new_exclude = mjcf_utils.new_element(tag='exclude', name=None, \
                    body1=body1, body2='{}{}'.format(finger_name, finger_part))
                contact.append(new_exclude)

    ##### prevent contact penetrating and harden joint limit
    geoms = [e for e in tree.iter() if (e.tag =='geom')]
    for geom in geoms:
        geom.attrib['margin'] = '0'
        geom.attrib['solimp'] = '0.999 0.9999 0.001 0.01 6'
        geom.attrib['solref'] = '2e-2 1'
    default = mjcf_utils.find_elements(tree.getroot(), \
            tags='default', attribs={'class': 'main'})
    default_geom = [g for g in default if g.tag == 'geom'][0]
    default_geom.attrib['margin'] = '0'

    if return_xml_only:
        return tree

    with io.StringIO() as string:
        string.write(ET.tostring(tree.getroot(), encoding="unicode"))
        xml_string = string.getvalue()
        parsed_xml = xml.dom.minidom.parseString(xml_string)
        xml_string = parsed_xml.toprettyxml(newl="", )

    env = gym.make('generalized-{}'.format(base_env_name), \
            model_xml=xml_string, **env_kwargs)
    env.reset()

    return env


generalized_envs = {
        'door-v0-shrink': generalized_door,
        'hammer-v0-shrink': generalized_hammer,
        'relocate-v0-shrink': generalized_relocate,
        }


