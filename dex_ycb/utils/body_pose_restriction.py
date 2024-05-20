

import pyquaternion
import numpy as np
import xml.etree.ElementTree as ET

import utils.mjcf_utils as mjcf_utils


def body_pose_restriction(body_name, pos, born_rot, constr_rot, \
        pos_range, quat_range):
    born_mat = born_rot
    if len(np.shape(born_rot)) == 1:
        born_mat = pyquaternion.Quaternion(born_rot).rotation_matrix
    constr_mat = constr_rot
    if len(np.shape(constr_rot)) == 1:
        constr_mat = pyquaternion.Quaternion(constr_rot).rotation_matrix

    if len(np.array(pos_range).shape) == 0:
        pos_range = [pos_range] * 3
    if len(np.array(quat_range).shape) == 0:
        quat_range = [quat_range] * 3

    virtual_mount_mat = np.dot(born_mat, constr_mat.T)
    virtual_mount_quat = pyquaternion.Quaternion(matrix=virtual_mount_mat)
    virtual_mount_quat = np.array([virtual_mount_quat[0], \
            virtual_mount_quat[1], virtual_mount_quat[2], virtual_mount_quat[3]])

    virtual_mount = mjcf_utils.new_body(name='virtual_mount', \
            pos=pos, quat=virtual_mount_quat)
    virtual_joint_x = mjcf_utils.new_joint(name='virtual_joint_x', \
            axis=[1,0,0], pos=[0,0,0], range=[-pos_range[0], pos_range[0]], \
            damping=0, type='slide', limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_joint_y = mjcf_utils.new_joint(name='virtual_joint_y', \
            axis=[0,1,0], pos=[0,0,0], range=[-pos_range[1], pos_range[1]], \
            damping=0, type='slide', limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_joint_z = mjcf_utils.new_joint(name='virtual_joint_z', \
            axis=[0,0,1], pos=[0,0,0], range=[-pos_range[2], pos_range[2]], \
            damping=0, type='slide', limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_joint_Rx = mjcf_utils.new_joint(name='virtual_joint_Rx', \
            axis=[1,0,0], pos=[0,0,0], range=[-quat_range[0], quat_range[0]], \
            damping=0, limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_joint_Ry = mjcf_utils.new_joint(name='virtual_joint_Ry', \
            axis=[0,1,0], pos=[0,0,0], range=[-quat_range[1], quat_range[1]], \
            damping=0, limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_joint_Rz = mjcf_utils.new_joint(name='virtual_joint_Rz', \
            axis=[0,0,1], pos=[0,0,0], range=[-quat_range[2], quat_range[2]], \
            damping=0, limited=True, armature=1.0, \
            solimplimit=[0.999, 0.9999, 0.001, 0.01, 6], solreflimit=[2e-2,1])
    virtual_mount_site = mjcf_utils.new_site(name='virtual_mount_site', \
            pos=[0,0,0], rgba=[0.5,0.1,0.1,0.3], size=pos_range, type='box')

    virtual_mount.append(virtual_mount_site)

    # xmlstr = ET.tostring(virtual_mount, encoding='utf8', method='xml')
    # print(xmlstr)

    virtual_attacher = mjcf_utils.new_body(name='virtual_attacher', pos=[0,0,0])
    virtual_attacher_inertial = mjcf_utils.new_inertial( \
            pos=[0,0,0], mass=1e-6, diaginertia=[1e-8,1e-8,1e-8])
    virtual_attacher_site = mjcf_utils.new_site(name='virtual_attacher_site', \
            pos=[0,0,0], rgba=[0,0,1,0], size=[0.07,0.05,0.03], type='box')
    virtual_attacher.append(virtual_attacher_inertial)
    virtual_attacher.append(virtual_attacher_site)

    virtual_attacher.append(virtual_joint_x)
    virtual_attacher.append(virtual_joint_y)
    virtual_attacher.append(virtual_joint_z)
    virtual_attacher.append(virtual_joint_Rx)
    virtual_attacher.append(virtual_joint_Ry)
    virtual_attacher.append(virtual_joint_Rz)

    virtual_mount.append(virtual_attacher)

    equality = mjcf_utils.new_element(tag='equality', name=None)
    weld_constraints = mjcf_utils.new_element(tag='weld',  \
            name='weld_constraints', body1='virtual_attacher', body2=body_name, \
            solimp=[0.999, 0.9999, 0.001, 0.01, 6], solref=[2e-2,1])
    equality.append(weld_constraints)

    return virtual_mount, equality

