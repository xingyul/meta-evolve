
import utils.mjcf_utils as mjcf_utils
import numpy as np

def robot_arm_interp(env, interp_param_vector, phys_param_vector_all, include_geoms=True):
    interp_param_vector = np.array([float(np.clip(i, 0., 1.)) for i in interp_param_vector])
    phys_param_vector_all = np.array(phys_param_vector_all)

    phys_param_vector = (phys_param_vector_all[:, 1] - phys_param_vector_all[:, 0]) * interp_param_vector + phys_param_vector_all[:, 0]

    phys_param_dict = {}
    pointer = 0

    phys_param_dict['robot_arm_body_1'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_2'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_3'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_4'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_5'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_5_y'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_6'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_7'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_body_hand'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_shape_jaco'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_shape_kinova'] = phys_param_vector[pointer]; pointer += 1
    phys_param_dict['robot_arm_shape_iiwa'] = phys_param_vector[pointer]; pointer += 1

    body_geom_dict_kinova_iiwa = {
            'robot0_j2s7s300_link_0': [ \
                ['base_link', {'quat': [1, 0, 0, 0], 'rgba': [1,1,1,1]}],\
                ['link_0', {'quat': [1, 0, 0, 0], 'rgba': [0.4, 0.4, 0.4, 1]}] \
                ], \
            'robot0_j2s7s300_link_1': [ \
                ['shoulder_link', {'quat': [1, 0, 0, 0], 'rgba': [1,1,1,1]}], \
                ['link_1', {'quat': [0, 0, 1, 0], 'rgba': [0.4, 0.4, 0.4, 1]}], \
                ], \
            'robot0_j2s7s300_link_2': [ \
                ['half_arm_1_link', {'quat': [0, 0, 0, 1], 'rgba': [1,1,1,1]}], \
                ['link_2', {'quat': [0, 0, 1, 0], 'rgba': [1, 0.423529, 0.0392157, 1], 'pos': [0, -0.0, 0] }], \
                ], \
            'robot0_j2s7s300_link_3': [ \
                ['half_arm_2_link', {'quat': [0, 0, 0, 1], 'rgba': [1,1,1,1]}], \
                ['link_3', {'quat': [0, 0, 1, 0], 'rgba': [0.4, 0.4, 0.4, 1], 'pos': [0, 0, 0.026]}], \
                ], \
            'robot0_j2s7s300_link_4': [ \
            ['forearm_link', {'quat': [1, 0, 0, 0], 'rgba': [1,1,1,1]}], \
                ['link_4', {'quat': [0, 1, 0, 0], 'rgba': [1, 0.423529, 0.0392157, 1]}], \
                ], \
            'robot0_j2s7s300_link_5': [ \
                ['spherical_wrist_1_link', {'quat': [0, 0, 0, 1], 'rgba': [1,1,1,1]}], \
                ['link_5', {'quat': [0, 0, 1, 0], 'rgba': [0.4, 0.4, 0.4, 1], 'pos': [0, 0, 0.026]}], \
                ], \
            'robot0_j2s7s300_link_6': [ \
                ['spherical_wrist_2_link', {'quat': [1, 0, 0, 0], 'rgba': [1,1,1,1]}], \
                ['link_6', {'quat': [0, 1, 0, 0], 'rgba': [1, 0.423529, 0.0392157, 1], 'pos': [0, 0, 0.0607]}], \
                ], \
            'robot0_j2s7s300_link_7': [ \
                ['bracelet_with_vision_link', {'quat': [1, 0, 0, 0], 'rgba': [1,1,1,1]}], \
                ['link_7', {'quat': [0, 0, 1, 0], 'rgba': [0.4, 0.4, 0.4, 1]}], \
                ], \
            }
    for body_name in body_geom_dict_kinova_iiwa:
        body = mjcf_utils.find_elements(env.model.root, \
                tags='body', attribs={'name': body_name})
        ##### handle existing jaco geoms
        existing_geoms = [elem for elem in body if elem.tag == 'geom']
        for geom in existing_geoms:
            geom.attrib['name'] = geom.attrib['name'] + '_jaco'
            geom.attrib['contype'] = '0'
            if not include_geoms:
                geom_parent = mjcf_utils.find_parent(env.model.root, geom)
                geom_parent.remove(geom)
        ##### handle existing jaco geoms
        ##### add kinova geoms
        if include_geoms:
            for idx, robot_name in enumerate(['kinova', 'iiwa']):
                mesh_name = body_geom_dict_kinova_iiwa[body_name][idx][0]
                mesh_attrib = body_geom_dict_kinova_iiwa[body_name][idx][1]
                geom = mjcf_utils.new_element(tag='geom', type="mesh", \
                        contype=0, conaffinity=0, group=1, \
                        name=body_name + '_{}'.format(robot_name), \
                        mesh=mesh_name, **mesh_attrib)
                body.append(geom)
        ##### add kinova geoms

    body_names = list(body_geom_dict_kinova_iiwa.keys())
    body_names.sort()
    for idx, body_name in enumerate(body_names[1:]):
        body = mjcf_utils.find_elements(env.model.root, \
                tags='body', attribs={'name': body_name})
        body_pos = np.array([float(p) for p in body.attrib['pos'].split()])
        body_pos_idx = np.argmax(np.abs(body_pos))

        phys_param = phys_param_dict['robot_arm_body_{}'.format(idx+1)]
        body_pos[body_pos_idx] = np.sign(body_pos[body_pos_idx]) * phys_param
        body.attrib['pos'] = ' '.join([str(p) for p in body_pos])
        if (idx+1) == 5:
            phys_param = phys_param_dict['robot_arm_body_{}_y'.format(idx+1)]
            body_pos[-1] = phys_param
            body.attrib['pos'] = ' '.join([str(p) for p in body_pos])

    ##### specially handle hand
    body = mjcf_utils.find_elements(env.model.root, \
            tags='body', attribs={'name': 'robot0_right_hand'})
    phys_param = phys_param_dict['robot_arm_body_hand']
    body_pos = np.array([0, 0, phys_param])
    body.attrib['pos'] = ' '.join([str(p) for p in body_pos])

    for robot_name in ['jaco', 'kinova', 'iiwa']:
        phys_param = phys_param_dict['robot_arm_shape_{}'.format(robot_name)]
        robot_mesh_names = []
        for idx, body_name in enumerate(body_names):
            body = mjcf_utils.find_elements(env.model.root, \
                    tags='body', attribs={'name': body_name})
            body_children = [elem for elem in body]
            for elem in body_children:
                if elem.tag == 'geom':
                    if '_' + robot_name in elem.attrib['name']:
                        if 'mesh' in elem.attrib:
                            mesh_name = elem.attrib['mesh']
                            robot_mesh_names.append(mesh_name)
        robot_mesh_names = list(set(robot_mesh_names))
        for mesh_name in robot_mesh_names:
            mesh = mjcf_utils.find_elements(env.model.root, \
                    tags='mesh', attribs={'name': mesh_name})
            if 'scale' in mesh.attrib:
                mesh_scale = np.array([float(s) for s in mesh.attrib['scale'].split()])
            else:
                mesh_scale = np.array([1., 1., 1.])
            minimum_scale = 0.025
            mesh_scale *= np.maximum(phys_param, minimum_scale)
            mesh.attrib['scale'] = ' '.join([str(s) for s in mesh_scale])

    return env
