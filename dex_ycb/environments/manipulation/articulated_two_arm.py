from collections import OrderedDict
import numpy as np
import glob
import os
import copy
import pyquaternion

from environments.manipulation.two_arm_env import TwoArmEnv

from models.arenas import TableArena
from models.objects import PotWithHandlesObject
from models.tasks import ManipulationTask
from models.objects import MujocoXMLObject
from utils.placement_samplers import UniformRandomSampler
from utils.observables import Observable, sensor

import utils.transform_utils as T

import point_set_transform

class ArticulatedTwoArm(TwoArmEnv):
    """
    This class corresponds to the lifting task for two robot arms.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be either 2 single single-arm robots or 1 bimanual robot!

        env_configuration (str): Specifies how to position the robots within the environment. Can be either:

            :`'bimanual'`: Only applicable for bimanual robot setups. Sets up the (single) bimanual robot on the -x
                side of the table
            :`'single-arm-parallel'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots next to each other on the -x side of the table
            :`'single-arm-opposed'`: Only applicable for multi single arm setups. Sets up the (two) single armed
                robots opposed from each others on the opposite +/-y sides of the table.

        Note that "default" corresponds to either "bimanual" if a bimanual robot is used or "single-arm-opposed" if two
        single-arm robots are used.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        ValueError: [Invalid number of robots specified]
        ValueError: [Invalid env configuration]
        ValueError: [Invalid robots for specified env configuration]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        partnet_id='100144',
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        self.partnet_id = partnet_id

        self.read_mesh_pc()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def read_mesh_pc(self):
        self.points_canonical = {}
        for part_index in [0, 1]:
            xyz_name = os.path.join( \
                    'models/objects/partnet_mob_{}'.format(self.partnet_id), \
                    'mesh_part_{}.xyz'.format(part_index))
            with open(xyz_name, 'r') as f:
                data = f.read().split()
                pc = [float(p) for p in data]
            pc = np.reshape(pc, [-1, 3])
            self.points_canonical[part_index] = pc

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 3.0 is provided if the pot is lifted and is parallel within 30 deg to the table

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.5], per-arm component that is proportional to the distance between each arm and its
              respective pot handle, and exactly 0.5 when grasping the handle
              - Note that the agent only gets the lifting reward when flipping no more than 30 degrees.
            - Grasping: in {0, 0.25}, binary per-arm component awarded if the gripper is grasping its correct handle
            - Lifting: in [0, 1.5], proportional to the pot's height above the table, and capped at a certain threshold

        Note that the final reward is normalized and scaled by reward_scale / 3.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        # check if the pot is tilted more than 30 degrees
        mat = T.quat2mat(self._pot_quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        cos_z = np.dot(z_unit, z_rotated)
        cos_30 = np.cos(np.pi / 6)
        direction_coef = 1 if cos_z >= cos_30 else 0

        # check for goal completion: cube is higher than the table top above a margin
        if self._check_success():
            reward = 3.0 * direction_coef

        # use a shaping reward
        elif self.reward_shaping:
            # lifting reward
            pot_bottom_height = self.sim.data.site_xpos[self.articulated_obj_center_id][2] - self.articulated_obj.top_offset[2]
            table_height = self.sim.data.site_xpos[self.table_top_id][2]
            elevation = pot_bottom_height - table_height
            r_lift = min(max(elevation - 0.05, 0), 0.15)
            reward += 10. * direction_coef * r_lift

            _gripper0_to_handle0 = self._gripper0_to_handle0
            _gripper1_to_handle1 = self._gripper1_to_handle1

            # gh stands for gripper-handle
            # When grippers are far away, tell them to be closer

            # Get contacts
            (g0, g1) = (self.robots[0].gripper["right"], self.robots[0].gripper["left"]) if \
                self.env_configuration == "bimanual" else (self.robots[0].gripper, self.robots[1].gripper)

            _g0h_dist = np.linalg.norm(_gripper0_to_handle0)
            _g1h_dist = np.linalg.norm(_gripper1_to_handle1)

            # Grasping reward
            if self._check_grasp(gripper=g0, object_geoms=self.articulated_obj.handle0_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g0h_dist))

            # Grasping reward
            if self._check_grasp(gripper=g1, object_geoms=self.articulated_obj.handle1_geoms):
                reward += 0.25
            # Reaching reward
            reward += 0.5 * (1 - np.tanh(10.0 * _g1h_dist))

        if self.reward_scale is not None:
            reward *= self.reward_scale / 3.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose(s) accordingly
        if self.env_configuration == "bimanual":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
            self.robots[0].robot_model.set_base_xpos(xpos)
        else:
            if self.env_configuration == "single-arm-opposed":
                # Set up robots facing towards each other by rotating them from their default position
                for robot, rotation in zip(self.robots, (np.pi/2, -np.pi/2)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    rot = np.array((0, 0, rotation))
                    xpos = T.euler2mat(rot) @ np.array(xpos)
                    robot.robot_model.set_base_xpos(xpos)
                    robot.robot_model.set_base_ori(rot)
            elif self.env_configuration == "single-arm-cornered":
                robot = self.robots[1]
                rotation = np.pi / 2
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                rot = np.array((0, 0, rotation))
                # xpos = T.euler2mat(rot) @ np.array(xpos)
                offset = -0.4
                # xpos = np.array(xpos) + np.array((offset, 0, 0))
                xpos = np.array(xpos) + np.array((0, offset, 0))
                if len(self.partnet_id) == 5: # laptops
                    offset = -0.25
                    xpos = xpos + np.array([offset, 0, 0])
                robot.robot_model.set_base_xpos(xpos)
                # robot.robot_model.set_base_ori(rot)

                robot = self.robots[0]
                offset = 0.4
                xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                xpos = np.array(xpos) + np.array((0, offset, 0))
                if len(self.partnet_id) == 5: # laptops
                    offset = -0.25
                    xpos = xpos + np.array([offset, 0, 0])
                robot.robot_model.set_base_xpos(xpos)
            else:   # "single-arm-parallel" configuration setting
                # Set up robots parallel to each other but offset from the center
                for robot, offset in zip(self.robots, (-0.25, 0.25)):
                    xpos = robot.robot_model.base_xpos_offset["table"](self.table_full_size[0])
                    xpos = np.array(xpos) + np.array((0, offset, 0))
                    robot.robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.articulated_obj = MujocoXMLObject(name='articulated_obj', fname='models/objects/partnet_mob_{}/object_{}.xml'.format(self.partnet_id, self.partnet_id), joints=None)
        self.goal = MujocoXMLObject(name='art_goal', fname='models/objects/partnet_mob_{}/object_{}_goal_site.xml'.format(self.partnet_id, self.partnet_id), joints=None)

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.articulated_obj)
        else:
            if len(self.partnet_id) == 5: # laptop
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.articulated_obj,
                    x_range=[-0.32, -0.3],
                    y_range=[-0.1, 0.1],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    rotation=(0 + -np.pi/24, 0 + np.pi/24),
                    rotation_axis='z',
                )
            else:
                self.placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=self.articulated_obj,
                    x_range=[-0.1, 0.1],
                    y_range=[-0.1, 0.1],
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
                    rotation=(0 + -np.pi/12, 0 + np.pi/12),
                    rotation_axis='z',
                )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.articulated_obj, self.goal],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.articulated_obj_body_id = self.sim.model.body_name2id(self.articulated_obj.root_body)
        self.table_top_id = self.sim.model.site_name2id("table_top")
        self.articulated_obj_center_id = self.sim.model.site_name2id("{}_default_site".format(self.articulated_obj.name))

        self.joint_range = self.sim.model.jnt_range[self.sim.model.joint_name2id('{}_joint'.format(self.articulated_obj.name))]

        ##### object affordance
        self.obj_affordance_marker_sids = {}
        for link in ['base', 'link']:
            for finger in ['finger', 'thumb']:
                obj_affordance_marker_sids = []
                for j in range(50):
                    try:
                        obj_affordance_marker_sid = self.sim.model.site_name2id('{}_{}_{}_marker_{}'.format(self.articulated_obj.name, link, finger, j))
                        obj_affordance_marker_sids.append(obj_affordance_marker_sid)
                    except:
                        break
                self.obj_affordance_marker_sids['{}_{}'.format(link, finger)] = \
                        np.array(obj_affordance_marker_sids)
        ##### object affordance

        ##### object goal mesh
        self.goal_marker_sids = {}
        for part_index in [0, 1]:
            goal_marker_sids = []
            for j in range(80):
                goal_marker_sid = self.sim.model.site_name2id('{}_marker_{}_{}'.format(self.goal.name, part_index, j))
                goal_marker_sids.append(goal_marker_sid)
            self.goal_marker_sids[part_index] = np.array(goal_marker_sids)
        ##### object goal mesh

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            if self.env_configuration == "bimanual":
                pf0 = self.robots[0].robot_model.naming_prefix + "right_"
                pf1 = self.robots[0].robot_model.naming_prefix + "left_"
            else:
                pf0 = self.robots[0].robot_model.naming_prefix
                pf1 = self.robots[1].robot_model.naming_prefix
            modality = "object"

            # position and rotation of object

            @sensor(modality=modality)
            def articulated_obj_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.articulated_obj_body_id])

            @sensor(modality=modality)
            def articulated_obj_quat(obs_cache):
                return T.convert_quat(self.sim.data.body_xquat[self.articulated_obj_body_id], to="xyzw")

            @sensor(modality=modality)
            def obj_affordance(obs_cache):
                obj_affordance = {}
                for k in self.obj_affordance_marker_sids:
                    sids = self.obj_affordance_marker_sids[k]
                    if len(sids) == 0:
                        continue
                    obj_affordance[k] = np.array(self.sim.data.site_xpos[sids])
                return obj_affordance
                '''
                site_pos = self.sim.data.site_xpos[self.sim.model.site_name2id('art_default_site')]
                return site_pos
                '''

            @sensor(modality=modality)
            def goal(obs_cache):
                return np.array(self.sim.data.site_xpos[self.articulated_obj_body_id])

            @sensor(modality=modality)
            def gripper0_to_obj(obs_cache):
                return obs_cache["articulated_obj_pos"] - obs_cache["{}eef_pos".format(pf0)] if \
                    "articulated_obj_pos" in obs_cache and "{}eef_pos".format(pf0) in obs_cache else np.zeros(3)

            @sensor(modality=modality)
            def gripper1_to_obj(obs_cache):
                return obs_cache["articulated_obj_pos"] - obs_cache["{}eef_pos".format(pf1)] if \
                    "articulated_obj_pos" in obs_cache and "{}eef_pos".format(pf1) in obs_cache else np.zeros(3)

            sensors = [articulated_obj_pos, articulated_obj_quat, \
                    gripper0_to_obj, gripper1_to_obj, obj_affordance]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Reset their positions
            obj_pos, obj_quat, obj = list(object_placements.values())[0]
            base_name = '{}_main'.format(obj.name)
            obj_quat = pyquaternion.Quaternion(obj_quat) * pyquaternion.Quaternion(self.sim.model.body_quat[self.sim.model.body_name2id(base_name)])
            obj_quat = np.array(list(obj_quat))

            # set target locations
            x = np.random.uniform(*self.placement_initializer.x_range)
            y = np.random.uniform(*self.placement_initializer.y_range)
            z = np.random.uniform(self.table_offset[-1] + 0.1, self.table_offset[-1] + 0.2)
            rotation = pyquaternion.Quaternion(axis=np.random.uniform(-1, 1, [3]), \
                    radians=np.random.uniform(-np.pi/12, -np.pi/12))
            self.target_base_pos = np.array([x, y, z])
            target_base_quat = rotation * obj_quat.copy()
            self.target_base_quat = np.array(list(target_base_quat))
            self.target_joint_value = np.random.uniform(*self.joint_range)
            if len(self.partnet_id) == 5: # laptop
                self.target_joint_value = np.random.uniform(np.mean(self.joint_range), self.joint_range[1])

            # self.set_obj_state(obj.name, self.target_base_pos, self.target_base_quat, self.target_joint_value)
            self.target_base_pos, self.target_base_quat, self.target_joint_value = self.set_target_site(obj.name, self.target_base_pos, self.target_base_quat, self.target_joint_value)

            target_link_pos = self.sim.data.get_site_xpos('{}_link_geom'.format(obj.name))
            self.target_link_pos = target_link_pos.copy()
            target_link_mat = self.sim.data.get_site_xmat('{}_link_geom'.format(obj.name))
            target_link_quat = pyquaternion.Quaternion(matrix=target_link_mat)
            self.target_link_quat = np.array(list(target_link_quat)).copy()

            # Reset their positions
            self.sim.model.body_pos[self.sim.model.body_name2id(base_name)] = obj_pos
            self.sim.model.body_quat[self.sim.model.body_name2id(base_name)] = obj_quat
            self.sim.data.set_joint_qpos('{}_joint'.format(obj.name), np.random.uniform(*self.joint_range))
            if len(self.partnet_id) == 5: # laptop
                joint_value = np.random.uniform(np.mean(self.joint_range), self.joint_range[1])
                self.sim.data.set_joint_qpos('{}_joint'.format(obj.name), joint_value)

            self.sim.forward()

    def get_obj_state(self, obj_name):
        pos = self.sim.model.body_pos[self.sim.model.body_name2id('{}_main'.format(obj_name))].copy()
        quat = self.sim.model.body_quat[self.sim.model.body_name2id('{}_main'.format(obj_name))].copy()
        joint_value = self.sim.data.get_joint_qpos('{}_joint'.format(obj_name)).copy()
        return pos, quat, joint_value

    def set_obj_state(self, obj_name, pos, quat, joint_value):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        self.sim.model.body_pos[self.sim.model.body_name2id('{}_main'.format(obj_name))] = pos
        self.sim.model.body_quat[self.sim.model.body_name2id('{}_main'.format(obj_name))] = quat
        self.sim.data.set_joint_qpos('{}_joint'.format(obj_name), joint_value)
        self.sim.forward()

    def set_target_site(self, obj_name, pos, quat, joint_value):
        min_z = []

        self.set_obj_state(obj_name, pos, quat, joint_value)

        body_pos = self.sim.data.get_site_xpos('{}_base_geom'.format(obj_name))
        body_mat = self.sim.data.get_site_xmat('{}_base_geom'.format(obj_name))
        body_quat = pyquaternion.Quaternion(matrix=body_mat)
        link_pos = self.sim.data.get_site_xpos('{}_link_geom'.format(obj_name))
        link_mat = self.sim.data.get_site_xmat('{}_link_geom'.format(obj_name))
        link_quat = pyquaternion.Quaternion(matrix=link_mat)

        state_dict = {'obj_base_pos': body_pos, 'obj_base_quat': body_quat, \
                'obj_link_pos': link_pos, 'obj_link_quat': link_quat}
        mesh_points_world_list = {}
        for part_index in [0, 1]:
            mesh_points_world = point_set_transform.point_set_transform(\
                    self.points_canonical[part_index], state_dict, joint=str(part_index))
            min_z.append(mesh_points_world[:, -1].min())
            mesh_points_world_list[part_index] = mesh_points_world
        min_z = np.min(min_z)
        offset_z = min_z - (0.8 + 0.02)

        ##### ensure it's not below table surface
        if offset_z < 0:
            for part_index in [0, 1]:
                mesh_points_world_list[part_index] = mesh_points_world_list[part_index] + \
                        np.array([0,0,-offset_z])
            body_pos = self.sim.model.body_pos[self.sim.model.body_name2id('{}_main'.format(obj_name))].copy()
            body_pos += np.array([0, 0, -offset_z])
            self.sim.model.body_pos[self.sim.model.body_name2id('{}_main'.format(obj_name))] = body_pos

        for part_index in [0, 1]:
            original_num_sites = self.sim.model.site_pos[self.goal_marker_sids[part_index]].shape[0]
            self.sim.model.site_pos[self.goal_marker_sids[part_index]] = \
                    mesh_points_world_list[part_index][:original_num_sites]
        self.sim.forward()

        body_quat = np.array(list(body_quat))
        return body_pos.copy(), body_quat.copy(), copy.deepcopy(joint_value)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to each handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to each handle
        if vis_settings["grippers"]:
            handles = [self.articulated_obj.important_sites["handle{i}"] for i in range(2)]
            grippers = [self.robots[0].gripper[arm] for arm in self.robots[0].arms] if \
                self.env_configuration == "bimanual" else [robot.gripper for robot in self.robots]
            for gripper, handle in zip(grippers, handles):
                self._visualize_gripper_to_target(gripper=gripper, target=handle, target_type="site")

    def _check_success(self):
        """
        Check if pot is successfully lifted

        Returns:
            bool: True if pot is lifted
        """
        pot_bottom_height = self.sim.data.site_xpos[self.articulated_obj_center_id][2] - self.articulated_obj.top_offset[2]
        table_height = self.sim.data.site_xpos[self.table_top_id][2]

        # cube is higher than the table top above a margin
        return pot_bottom_height > table_height + 0.10

    @property
    def _handle0_xpos(self):
        """
        Grab the position of the left (blue) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle0_site_id]

    @property
    def _handle1_xpos(self):
        """
        Grab the position of the right (green) hammer handle.

        Returns:
            np.array: (x,y,z) position of handle
        """
        return self.sim.data.site_xpos[self.handle1_site_id]

    @property
    def _pot_quat(self):
        """
        Grab the orientation of the pot body.

        Returns:
            np.array: (x,y,z,w) quaternion of the pot body
        """
        return T.convert_quat(self.sim.data.body_xquat[self.articulated_obj_body_id], to="xyzw")

