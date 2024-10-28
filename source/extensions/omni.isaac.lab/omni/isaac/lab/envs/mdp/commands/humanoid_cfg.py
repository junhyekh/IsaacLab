from __future__ import annotations

import torch as th
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

from icecream import ic

@configclass
class LocalHandPoseCommandCfg(CommandTermCfg):
    """Configuration for humanoid pose command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_hand_body_name: str = MISSING
    """Name of the left hand body in the asset for which the commands are generated."""

    right_hand_body_name: str = MISSING
    """Name of the right hand body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        # Ranges for the commands in cylindrical coordinates
        r_range: tuple[float, float] = MISSING  # min, max [m]
        theta_range: tuple[float, float] = MISSING  # min, max [rad]
        z_range: tuple[float, float] = MISSING  # min, max [m]
    
    ranges: Ranges = MISSING

    # Configuration parameters for shifts and angle deltas
    hand_shift: float = 0.15  # Default shift along x-axis in meters
    delta_yaw: float = 30.0  # Default angle delta in degrees
    angle_noise: float = 0.001 # Added angle noise in degrees

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class LocalHandPoseCommand(CommandTerm):
    """Command generator for generating pose commands for a humanoid robot's hands."""

    cfg: LocalHandPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: LocalHandPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.left_hand_idx = self.robot.find_bodies(cfg.left_hand_body_name)[0][0]
        self.right_hand_idx = self.robot.find_bodies(cfg.right_hand_body_name)[0][0]

        # Target hand pose command in **cylindrical frame**
        self.pose_command_c_left = th.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_c_left[:, 3] = 1.0  
        self.pose_command_c_right = th.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_c_right[:, 3] = 1.0 

        # Target hand pose command in **world frame**
        self.pose_command_w_left = th.zeros_like(self.pose_command_c_left)
        self.pose_command_w_right = th.zeros_like(self.pose_command_c_right)

        # Cylindrical frame wrt **world frame**
        self.cylinder_pos_w = th.zeros(self.num_envs, 3, device=self.device)
        self.cylinder_quat_w = th.zeros(self.num_envs, 4, device=self.device)
        self.cylinder_quat_w[:, 0] = 1.0 

        # Metrics
        self.metrics["position_error_left"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error_left"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["position_error_right"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error_right"] = th.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HumanoidPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> th.Tensor:
        """The desired pose command for both hands in cylindrical frame. Shape is (num_envs, 12)."""
        self._update_command()

        # Compute pose delta between hand pose and target pose
        pos_delta_left, rot_delta_left = math_utils.subtract_frame_transforms(
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
            self.pose_command_w_left[:, :3],
            self.pose_command_w_left[:, 3:],
        )
        pos_delta_right, rot_delta_right = math_utils.subtract_frame_transforms(
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7],
            self.pose_command_w_right[:, :3],
            self.pose_command_w_right[:, 3:],
        )

        axa_delta_left = math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(rot_delta_left))
        axa_delta_right= math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(rot_delta_right))

        return th.cat((pos_delta_left, pos_delta_right, axa_delta_left, axa_delta_right), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self._update_command()

        # Compute the error for left hand
        pos_error_left, rot_error_left = math_utils.compute_pose_error(
            self.pose_command_w_left[:, :3],
            self.pose_command_w_left[:, 3:],
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
        )
        self.metrics["position_error_left"] = th.norm(pos_error_left, dim=-1)
        self.metrics["orientation_error_left"] = th.norm(rot_error_left, dim=-1)

        # Compute the error for right hand
        pos_error_right, rot_error_right = math_utils.compute_pose_error(
            self.pose_command_w_right[:, :3],
            self.pose_command_w_right[:, 3:],
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7],
        )
        self.metrics["position_error_right"] = th.norm(pos_error_right, dim=-1)
        self.metrics["orientation_error_right"] = th.norm(rot_error_right, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample new pose targets in attached cylindrical frame
        device = self.device
        num_envs = len(env_ids)

        r = th.empty(num_envs, device=device).uniform_(*self.cfg.ranges.r_range)
        theta = th.empty(num_envs, device=device).uniform_(*self.cfg.ranges.theta_range)
        z = th.empty(num_envs, device=device).uniform_(*self.cfg.ranges.z_range)

        target_pos_c = th.zeros((num_envs, 3), device=device)
        target_pos_c[..., 0] = r * th.cos(theta)
        target_pos_c[..., 1] = r * th.sin(theta)
        target_pos_c[..., 2] = z

        target_euler_c = th.zeros((num_envs, 3), device=device)
        target_euler_c[..., 2] = theta
        noise = th.empty((num_envs, 3), device=device).uniform_(
            -np.pi * self.cfg.angle_noise/180., 
            -np.pi * self.cfg.angle_noise/180.)
        target_euler_c = math_utils.wrap_to_pi(target_euler_c+noise)
        target_quat_c = math_utils.quat_from_euler_xyz(
            target_euler_c[..., 0],
            target_euler_c[..., 1],
            target_euler_c[..., 2],
        )

        # Shift vectors for left and right hands
        shift_vec = th.tensor(
            [0.0, self.cfg.hand_shift, 0.0], device=device).expand(num_envs, 3)

        # Target hand pos in cylindrical frame
        self.pose_command_c_left[env_ids, :3]= \
            target_pos_c + math_utils.quat_rotate(target_quat_c, shift_vec)
        self.pose_command_c_right[env_ids, :3]= \
            target_pos_c + math_utils.quat_rotate(target_quat_c, -shift_vec)

        # Compute delta_quat for left and right hands (±delta_yaw along yaw)
        delta_yaw_rad = th.tensor(th.pi * self.cfg.delta_yaw / 180, device=device)
        delta_quat_left = math_utils.quat_from_euler_xyz(
            th.zeros(num_envs, device=device),
            th.zeros(num_envs, device=device),
            -delta_yaw_rad,
        )
        delta_quat_right = math_utils.quat_from_euler_xyz(
            th.zeros(num_envs, device=device),
            th.zeros(num_envs, device=device),
            delta_yaw_rad,
        )

        # Target hand quat in cylindrical frame
        self.pose_command_c_left[env_ids, 3:7] = \
            math_utils.quat_mul(target_quat_c, delta_quat_left)
        self.pose_command_c_right[env_ids, 3:7] = \
            math_utils.quat_mul(target_quat_c, delta_quat_right)

    def _update_command(self):
        '''
        This is different from _resample_command method. Since the robot
        poses varies as it moves, we need to refresh the target pose
        considering the robot's movement.
        '''
        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w

        _, _, base_yaw = math_utils.euler_xyz_from_quat(base_quat_w)

        # Compute the pose of attached cylindrical frame
        cylinder_euler_w = th.zeros_like(base_pos_w)
        cylinder_euler_w[..., 2] = base_yaw

        self.cylinder_pos_w[..., :2] = base_pos_w[..., :2]
        self.cylinder_quat_w = math_utils.quat_from_euler_xyz(
            cylinder_euler_w[..., 0],
            cylinder_euler_w[..., 1],
            cylinder_euler_w[..., 2])

        self.pose_command_w_left[:, :3], self.pose_command_w_left[:, 3:] = \
            math_utils.combine_frame_transforms(
                self.cylinder_pos_w,
                self.cylinder_quat_w,
                self.pose_command_c_left[:, :3],
                self.pose_command_c_left[:, 3:])

        self.pose_command_w_right[:, :3], self.pose_command_w_right[:, 3:] = \
            math_utils.combine_frame_transforms(
                self.cylinder_pos_w,
                self.cylinder_quat_w,
                self.pose_command_c_right[:, :3],
                self.pose_command_c_right[:, 3:])


    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer_left"):
                self.goal_pose_visualizer_left = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                self.goal_pose_visualizer_right = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                self.current_pose_visualizer_left = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
                self.current_pose_visualizer_right = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            self.goal_pose_visualizer_left.set_visibility(True)
            self.goal_pose_visualizer_right.set_visibility(True)
            self.current_pose_visualizer_left.set_visibility(True)
            self.current_pose_visualizer_right.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer_left"):
                self.goal_pose_visualizer_left.set_visibility(False)
                self.goal_pose_visualizer_right.set_visibility(False)
                self.current_pose_visualizer_left.set_visibility(False)
                self.current_pose_visualizer_right.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return
        self._update_command()
        self.goal_pose_visualizer_left.visualize(
            self.pose_command_w_left[:, :3], self.pose_command_w_left[:, 3:]
        )
        self.goal_pose_visualizer_right.visualize(
            self.pose_command_w_right[:, :3], self.pose_command_w_right[:, 3:]
        )
        body_pose_w_left = self.robot.data.body_state_w[:, self.left_hand_idx]
        self.current_pose_visualizer_left.visualize(
            body_pose_w_left[:, :3], body_pose_w_left[:, 3:7]
        )
        body_pose_w_right = self.robot.data.body_state_w[:, self.right_hand_idx]
        self.current_pose_visualizer_right.visualize(
            body_pose_w_right[:, :3], body_pose_w_right[:, 3:7]
        )

@configclass
class GlobalHandPoseCommandCfg(CommandTermCfg):
    """Configuration for humanoid pose command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_hand_body_name: str = MISSING
    """Name of the left hand body in the asset for which the commands are generated."""

    right_hand_body_name: str = MISSING
    """Name of the right hand body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        # Ranges for the commands in cylindrical coordinates
        x_range : tuple[float, float] = MISSING  # min, max [m]
        y_range : tuple[float, float] = MISSING  # min, max [m]
        z_range : tuple[float, float] = MISSING  # min, max [m]
        yaw_range : tuple[float, float] = MISSING  # min, max [m]
    
    ranges: Ranges = MISSING

    # Configuration parameters for shifts and angle deltas
    hand_shift: float = 0.15  # Default shift along x-axis in meters
    delta_yaw: float = 30.0  # Default angle delta in degrees
    standing_dist: float = 0.4 # Distance from the standing point to the object
    success_threshold: float = 0.1 # Threshold for determining success

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    standing_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/standing_pose")

    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    standing_pose_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

class GlobalPoseCommand(CommandTerm):
    """Command generator for generating pose commands for a humanoid robot's hands."""

    cfg: GlobalHandPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg:GlobalHandPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.left_hand_idx = self.robot.find_bodies(cfg.left_hand_body_name)[0][0]
        self.right_hand_idx = self.robot.find_bodies(cfg.right_hand_body_name)[0][0]

        # Target hand pose command in **world frame**
        self.target_pose_w = th.zeros(self.num_envs, 7, device=self.device)
        self.standing_pose_w = th.zeros_like(self.target_pose_w)
        self.target_pose_w_left = th.zeros_like(self.target_pose_w)
        self.target_pose_w_right = th.zeros_like(self.target_pose_w)

        # Metrics
        self.metrics["position_error_left"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error_left"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["position_error_right"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error_right"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["position_error_base"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error_base"] = th.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = th.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "HumanoidPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> th.Tensor:
        """The desired pose command for both hands in cylindrical frame. Shape is (num_envs, 12)."""
        # self._update_command()

        # Compute pose delta between hand pose and target pose
        pos_delta_left, rot_delta_left = math_utils.subtract_frame_transforms(
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
            self.target_pose_w_left[:, :3],
            self.target_pose_w_left[:, 3:],
        )
        pos_delta_right, rot_delta_right = math_utils.subtract_frame_transforms(
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7],
            self.target_pose_w_right[:, :3],
            self.target_pose_w_right[:, 3:],
        )
        root_state_xy_w = th.zeros_like(self.robot.data.root_state_w[..., :3])
        root_state_xy_w[..., :2] = self.robot.data.root_state_w[..., :2]
        
        pos_delta_base, rot_delta_base = math_utils.subtract_frame_transforms(
            root_state_xy_w,
            math_utils.yaw_quat(self.robot.data.root_state_w[..., 3:7]),
            self.standing_pose_w[:, :3],
            math_utils.yaw_quat(self.standing_pose_w[:, 3:]),
        )

        axa_delta_left = math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(rot_delta_left))
        axa_delta_right= math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(rot_delta_right))
        axa_delta_base = math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(rot_delta_base))

        command = th.cat(
            (pos_delta_left, pos_delta_right, pos_delta_base,
             axa_delta_left, axa_delta_right, axa_delta_base), dim=-1)

        return command
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # self._update_command()

        # Compute the error for left hand
        pos_error_left, rot_error_left = math_utils.compute_pose_error(
            self.target_pose_w_left[:, :3],
            self.target_pose_w_left[:, 3:],
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
        )
        self.metrics["position_error_left"] = th.norm(pos_error_left, dim=-1)
        self.metrics["orientation_error_left"] = th.norm(rot_error_left, dim=-1)

        # Compute the error for right hand
        pos_error_right, rot_error_right = math_utils.compute_pose_error(
            self.target_pose_w_right[:, :3],
            self.target_pose_w_right[:, 3:],
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7],
        )
        self.metrics["position_error_right"] = th.norm(pos_error_right, dim=-1)
        self.metrics["orientation_error_right"] = th.norm(rot_error_right, dim=-1)

        root_state_xy_w = th.zeros_like(self.robot.data.root_state_w[..., :3])
        root_state_xy_w[..., :2] = self.robot.data.root_state_w[..., :2]
        pos_error_base, rot_error_base = math_utils.compute_pose_error(
            root_state_xy_w,
            math_utils.yaw_quat(self.robot.data.root_state_w[..., 3:7]),
            self.standing_pose_w[:, :3],
            math_utils.yaw_quat(self.standing_pose_w[:, 3:]),
        )
        self.metrics["position_error_base"] = th.norm(pos_error_right, dim=-1)
        self.metrics["orientation_error_base"] = th.norm(rot_error_right, dim=-1)
        successes = th.logical_and(
            self.metrics["position_error_left"] <= self.cfg.success_threshold,
            self.metrics["position_error_right"] <= self.cfg.success_threshold,
        )
        self.metrics["consecutive_success"] *= successes.float()
        self.metrics["consecutive_success"] += successes.float()
        
    def _resample_command(self, env_ids: Sequence[int]):
        # Sample new pose targets in attached cylindrical frame
        device = self.device
        num_envs = len(env_ids)

        base_pos_w = self.robot.data.root_pos_w

        # Sample target position around the agent's base position 
        self.target_pose_w[env_ids, :2] = base_pos_w[env_ids, :2]
        self.target_pose_w[env_ids, 0] += th.empty(
            num_envs, device=device).uniform_(*self.cfg.ranges.x_range)
        self.target_pose_w[env_ids, 1] += th.empty(
            num_envs, device=device).uniform_(*self.cfg.ranges.y_range)
        self.target_pose_w[env_ids, 2] = th.empty(
            num_envs, device=device).uniform_(*self.cfg.ranges.z_range)


        target_euler_w = th.zeros((num_envs, 3), device=device)
        target_euler_w[..., 2] = th.empty(
            num_envs, device=device).uniform_(*self.cfg.ranges.yaw_range)
        self.target_pose_w[env_ids, 3:7]= math_utils.quat_from_euler_xyz(
            target_euler_w[..., 0],
            target_euler_w[..., 1],
            target_euler_w[..., 2],
        )

        # Shift vectors for left and right hands
        shift_vec = th.tensor(
            [0.0, self.cfg.hand_shift, 0.0], device=device).expand(num_envs, 3)
        standing_vec = th.tensor(
            [-self.cfg.standing_dist, 0.0, 0.0], device=device).expand(num_envs, 3)

        # Target hand pos in cylindrical frame
        self.target_pose_w_left[env_ids, :3]= self.target_pose_w[env_ids, :3] \
            + math_utils.quat_rotate(self.target_pose_w[env_ids, 3:7], shift_vec)
        self.target_pose_w_right[env_ids, :3]= self.target_pose_w[env_ids, :3] \
            + math_utils.quat_rotate(self.target_pose_w[env_ids, 3:7], -shift_vec)

        self.standing_pose_w[env_ids, :2]= self.target_pose_w[env_ids, :2] \
            + math_utils.quat_rotate(self.target_pose_w[env_ids, 3:7], standing_vec)[..., :2]

        self.standing_pose_w[env_ids, 3:7]= self.target_pose_w[env_ids, 3:7] \

        # Compute delta_quat for left and right hands (±delta_yaw along yaw)
        delta_yaw_rad = th.tensor(th.pi * self.cfg.delta_yaw / 180, device=device)
        delta_quat_left = math_utils.quat_from_euler_xyz(
            th.zeros(num_envs, device=device),
            th.zeros(num_envs, device=device),
            -delta_yaw_rad,
        )
        delta_quat_right = math_utils.quat_from_euler_xyz(
            th.zeros(num_envs, device=device),
            th.zeros(num_envs, device=device),
            delta_yaw_rad,
        )

        # Target hand quat in cylindrical frame
        self.target_pose_w_left[env_ids, 3:7] = \
            math_utils.quat_mul(self.target_pose_w[env_ids, 3:7], delta_quat_left)
        self.target_pose_w_right[env_ids, 3:7] = \
            math_utils.quat_mul(self.target_pose_w[env_ids, 3:7], delta_quat_right)

        self.metrics["consecutive_success"][env_ids] = 0.

    def _update_command(self):
        pass


    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer_left"):
                self.goal_pose_visualizer_left = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                self.goal_pose_visualizer_right = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                self.current_pose_visualizer_left = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
                self.current_pose_visualizer_right = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
                self.standing_pose_visualizer = VisualizationMarkers(
                    self.cfg.standing_pose_visualizer_cfg
                )
            self.goal_pose_visualizer_left.set_visibility(True)
            self.goal_pose_visualizer_right.set_visibility(True)
            self.current_pose_visualizer_left.set_visibility(True)
            self.current_pose_visualizer_right.set_visibility(True)
            self.standing_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer_left"):
                self.goal_pose_visualizer_left.set_visibility(False)
                self.goal_pose_visualizer_right.set_visibility(False)
                self.current_pose_visualizer_left.set_visibility(False)
                self.current_pose_visualizer_right.set_visibility(False)
                self.standing_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer_left.visualize(
            self.target_pose_w_left[:, :3], self.target_pose_w_left[:, 3:]
        )
        self.goal_pose_visualizer_right.visualize(
            self.target_pose_w_right[:, :3], self.target_pose_w_right[:, 3:]
        )
        body_pose_w_left = self.robot.data.body_state_w[:, self.left_hand_idx]
        self.current_pose_visualizer_left.visualize(
            body_pose_w_left[:, :3], body_pose_w_left[:, 3:7]
        )
        body_pose_w_right = self.robot.data.body_state_w[:, self.right_hand_idx]
        self.current_pose_visualizer_right.visualize(
            body_pose_w_right[:, :3], body_pose_w_right[:, 3:7]
        )
        self.standing_pose_visualizer.visualize(
            self.standing_pose_w[:, :3], self.standing_pose_w[:, 3:7]
        )