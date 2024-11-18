from __future__ import annotations

import itertools
import torch as th
import numpy as np
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import (FRAME_MARKER_CFG,
                                           GREEN_ARROW_X_MARKER_CFG,
                                           RAY_CASTER_MARKER_CFG)
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.sim as sim_utils


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

from icecream import ic

def cart2sphere(cart):
    """
    Args:
        cart (Tensor): Cartesian coordinates with shape (..., 3), 
            where the last dimension represents [x, y, z].
    Returns:
        sphere (Tensor): Spherical coordinates with shape (..., 3), 
            where the last dimension represents [radius, azimuth, inclination].
    """
    sphere = th.zeros_like(cart)
    sphere[..., 0] = th.norm(cart, dim=-1)
    sphere[..., 1] = th.atan2(cart[..., 2], cart[..., 0])
    sphere[..., 2] = th.asin(cart[..., 1] / sphere[..., 0])
    return sphere

def sphere2cart(sphere):
    """
    Args:
        sphere (Tensor): Spherical coordinates with shape (..., 3), 
            where the last dimension represents [radius, azimuth, inclination].
    Returns:
        cart (Tensor): Cartesian coordinates with shape (..., 3), 
            where the last dimension represents [x, y, z].
    """
    cart = th.zeros_like(sphere)
    cart[..., 0] = sphere[..., 0] * th.cos(sphere[..., 2]) * th.cos(sphere[..., 1])
    cart[..., 1] = sphere[..., 0] * th.sin(sphere[..., 2])
    cart[..., 2] = sphere[..., 0] * th.cos(sphere[..., 2]) * th.sin(sphere[..., 1])
    return cart

def cart2cylinder(cart):
    """
    Args:
        cart (Tensor): Cartesian coordinates with shape (..., 3),
            where the last dimension represents [x, y, z].
    Returns:
        cylinder (Tensor): Cylindrical coordinates with shape (..., 3),
            where the last dimension represents [r, theta, z].
    """
    cylinder = th.zeros_like(cart)
    cylinder[..., 0] = th.norm(cart[..., :2], dim=-1)  # r = sqrt(x^2 + y^2)
    cylinder[..., 1] = th.atan2(cart[..., 1], cart[..., 0])  # theta = atan2(y, x)
    cylinder[..., 2] = cart[..., 2]  # z-coordinate remains the same
    return cylinder

def cylinder2cart(cylinder):
    """
    Args:
        cylinder (Tensor): Cylindrical coordinates with shape (..., 3),
            where the last dimension represents [r, theta, z].
        Returns:
            cart (Tensor): Cartesian coordinates with shape (..., 3),
            where the last dimension represents [x, y, z].
    """
    cart = th.zeros_like(cylinder)
    cart[..., 0] = cylinder[..., 0] * th.cos(cylinder[..., 1])  # x = r * cos(theta)
    cart[..., 1] = cylinder[..., 0] * th.sin(cylinder[..., 1])  # y = r * sin(theta)
    cart[..., 2] = cylinder[..., 2]  # z-coordinate remains the same
    return cart

def cart2euler(cart):
    euler = th.zeros_like(cart)
    
    # Pitch angle (φ)... angle between the vector and its projection on the xy-plane
    # Calculated as the angle between the vector and the z-axis
    euler[..., 1] = -th.atan2(cart[..., 2], th.sqrt(cart[..., 0]**2 + cart[..., 1]**2))
    
    # Yaw angle (θ)... angle between the projection of the vector on the xy-plane 
    # and the global x-axis
    euler[..., 2] = th.atan2(cart[..., 1], cart[..., 0])
    
    # Roll angle (ψ) is not defined in this setup and remains zero
    # euler[..., 0] remains 0 as roll is not applicable for aligning x-axis with 
    # the vector and y-axis in the xy-plane
    
    return euler

def safe_rotvec2quat(rotvec: th.Tensor,
                     form: Literal["xyzw", "wxyz"] = "wxyz"):
    angle = th.linalg.vector_norm(rotvec, dim=1)
    small_angle = (angle <= 1e-3)
    large_angle = ~small_angle
    scale = th.empty_like(angle)
    scale[small_angle] = (0.5 - angle[small_angle] ** 2 / 48 +
                        angle[small_angle] ** 4 / 3840)
    scale[large_angle] = (th.sin(angle[large_angle] / 2) /
                        angle[large_angle])

    xyz = scale[:, None] * rotvec
    w = th.cos(angle/2)[..., None]
    if form == 'wxyz':
        return th.cat([w, xyz], dim=-1)
    else:
        return th.cat([xyz, w], dim=-1)

def interpolate_pose(
        p1: th.Tensor, p2: th.Tensor, t: th.Tensor):
    """
    Interpolate between two poses p1 and p2.

    Parameters:
        p1: Tensor of shape [N,7], each row is [x, y, z, qx, qy, qz, qw]
        p2: Tensor of shape [N,7], same as p1
        t: Tensor of shape [N], with values between 0 and 1

    Returns:
        Interpolated pose of shape [N,7]
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)

    pos1 = p1[:, :3]
    pos2 = p2[:, :3]
    q1 = p1[:, 3:]
    q2 = p2[:, 3:]

    # Linear interpolation of position
    interp_pos = pos1 + (pos2 - pos1) * t

    # # Normalize quaternions
    # q1 = q1 / q1.norm(dim=1, keepdim=True)
    # q2 = q2 / q2.norm(dim=1, keepdim=True)

    # # Convert quaternions to axis-angle representation
    # aa1 = math_utils.axis_angle_from_quat(q1)
    # aa2 = math_utils.axis_angle_from_quat(q2)

    # # Interpolate axis-angles
    # interp_aa = aa1 + (aa2 - aa1) * t

    # Convert interpolated axis-angle back to quaternion
    interp_q = math_utils.slerp(q1, q2, t.squeeze())

    # Concatenate interpolated position and quaternion
    interp_pose = th.cat([interp_pos, interp_q], dim=1)
    return interp_pose


@configclass
class IKHandTrajCommandCfg(CommandTermCfg):
    """Configuration for humanoid pose command generator."""

    class_type: type = MISSING

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    left_hand_body_name: str = MISSING
    """Name of the left hand body in the asset for which the commands are generated."""

    right_hand_body_name: str = MISSING
    """Name of the right hand body in the asset for which the commands are generated."""

    left_foot_body_name: str = MISSING
    right_foot_body_name: str = MISSING
    torso_body_name: str = MISSING

    make_quat_unique: bool = False

    vis_left: bool = False
    vis_right: bool = True

    mode: Literal["cylinder", "cart"] = MISSING
    frame: Literal["torso", "z-inv", "foot"] = MISSING
    moving_time: float = MISSING
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        # Ranges for the commands in cylindrical coordinates
        r_range: tuple[float, float] = MISSING  # min, max [m]
        theta_range_left: tuple[float, float] = MISSING  # min, max [rad]
        theta_range_right: tuple[float, float] = MISSING  # min, max [rad]
        # Ranges for cartesian coord sampling
        y_left_range: tuple[float, float] = MISSING  # min, max [m]
        y_right_range: tuple[float, float] = MISSING  # min, max [m]
        x_range: tuple[float, float] = MISSING  # min, max [m]

        z_range: tuple[float, float] = MISSING  # min, max [m]
    
    ranges: Ranges = MISSING

    # Configuration parameters for shifts and angle deltas
    angle_noise: float = 0.001 # Added angle noise in degrees
    # angle_noise: float = 30. # Added angle noise in degrees
    spherical_z: float = 1.0 # Height of the spherical coordinate

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = \
        FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    vis_hand_bbox: bool = True
    # hand_bbox_file: str = '/home/user/dex-clutter/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/commands/right_hand.npy'
    hand_bbox = [[ 0.0000, -0.0214, -0.0439],
                [ 0.0000, -0.0214,  0.0439],
                [ 0.0000,  0.0728, -0.0439],
                [ 0.0000,  0.0728,  0.0439],
                [ 0.1038, -0.0214, -0.0439],
                [ 0.1038, -0.0214,  0.0439],
                [ 0.1038,  0.0728, -0.0439],
                [ 0.1038,  0.0728,  0.0439]]


    bbox_vis_ref_cfg = RAY_CASTER_MARKER_CFG.replace(
                                                    prim_path=f"/Visuals/Command/bbox")
    bbox_vis_ref_cfg.markers['hit'] .radius=0.01
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)



class IKHandTrajCommand(CommandTerm):
    """Command generator for generating pose commands for a humanoid robot's hands."""

    cfg: IKHandTrajCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: IKHandTrajCommandCfg, env: ManagerBasedEnv):
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

        self.left_foot_idx = self.robot.find_bodies(cfg.left_foot_body_name)[0][0]
        self.right_foot_idx = self.robot.find_bodies(cfg.right_foot_body_name)[0][0]

        self.torso_idx = self.robot.find_bodies(cfg.torso_body_name)[0][0]

        # Current command in the spherical coordinate
        self.curr_command_s_left = th.zeros(self.num_envs, 7, device=self.device)
        self.curr_command_s_right = th.zeros(self.num_envs, 7, device=self.device)

        # Next command in the spherical coordinate
        self.next_command_s_left = th.zeros(self.num_envs, 7, device=self.device)
        self.next_command_s_left[:, 3] = 1.0  
        self.next_command_s_right = th.zeros(self.num_envs, 7, device=self.device)
        self.next_command_s_right[:, 3] = 1.0 

        # Lerped hand pose command in **world frame**
        self.lerp_command_w_left = th.zeros_like(self.next_command_s_left)
        self.lerp_command_w_right = th.zeros_like(self.next_command_s_left)

        self.lerp_command_s_left = th.zeros_like(self.next_command_s_left)
        self.lerp_command_s_right = th.zeros_like(self.next_command_s_left)

        # Cylindrical frame wrt **world frame**
        self.cylinder_pos_w = th.zeros(self.num_envs, 3, device=self.device)
        self.cylinder_quat_w = th.zeros(self.num_envs, 4, device=self.device)

        # Hands pose wrt **cylindrical frame**
        self.pos_hand_s_left = th.zeros(self.num_envs, 3, device=self.device)
        self.pos_hand_s_right = th.zeros(self.num_envs, 3, device=self.device)
        self.quat_hand_s_left = th.zeros(self.num_envs, 4, device=self.device)
        self.quat_hand_s_right = th.zeros(self.num_envs, 4, device=self.device)


        if self.cfg.resampling_time_range[0] != self.cfg.resampling_time_range[1]:
            raise ValueError("Resampling time range should be unique in order to compute lerp")
        self.resampling_time = self.cfg.resampling_time_range[0]

        # hand_bboxes = th.as_tensor(np.load(cfg.hand_bbox_file),
        #                 dtype=th.float, device=self.device)
        hand_bboxes = th.as_tensor(cfg.hand_bbox,
                        dtype=th.float, device=self.device)
        self._hand_bboxes = hand_bboxes[None].repeat(self.num_envs, 1, 1)

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
        """Returns the pose delta in the **world frame**. Shape is (num_envs, 12)."""
        self._update_command()

        # Here, we should represent the pose delta in the base frame(or cylinder frame)
        self.pos_hand_s_left, self.quat_hand_s_left = math_utils.subtract_frame_transforms(
            self.cylinder_pos_w,
            self.cylinder_quat_w,
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7]
        )
        pos_delta_s_left, rot_delta_s_left = math_utils.compute_pose_error(
            self.pos_hand_s_left,
            self.quat_hand_s_left,
            self.lerp_command_s_left[:, :3],
            self.lerp_command_s_left[:, 3:],
        )
        self.pos_hand_s_right, self.quat_hand_s_right = math_utils.subtract_frame_transforms(
            self.cylinder_pos_w,
            self.cylinder_quat_w,
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7]
        )
        pos_delta_s_right, rot_delta_s_right = math_utils.compute_pose_error(
            self.pos_hand_s_right,
            self.quat_hand_s_right,
            self.lerp_command_s_right[:, :3],
            self.lerp_command_s_right[:, 3:],
        )


        axa_delta_s_left = math_utils.wrap_to_pi(rot_delta_s_left)
        axa_delta_s_right = math_utils.wrap_to_pi(rot_delta_s_right)

        return th.cat((pos_delta_s_right, axa_delta_s_right, pos_delta_s_left, axa_delta_s_left), dim=-1)
    
    def hand_pose_in_attached_frame(self):
        """
        Returns the computed hand pose wrt to the attached frame
        Should be called after calling command function
        """
        hand_axa_left_s = math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(self.quat_hand_s_left))
        hand_axa_right_s = math_utils.wrap_to_pi(
            math_utils.axis_angle_from_quat(self.quat_hand_s_right))

        hand_pose_b = th.cat(
            (self.pos_hand_s_right, self.pos_hand_s_left, hand_axa_left_s, hand_axa_right_s),
            dim=-1)
        
        return hand_pose_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # self._update_command()

        # Compute the error for left hand
        pos_error_left, rot_error_left = math_utils.compute_pose_error(
            self.lerp_command_w_left[:, :3],
            self.lerp_command_w_left[:, 3:],
            self.robot.data.body_state_w[:, self.left_hand_idx, :3],
            self.robot.data.body_state_w[:, self.left_hand_idx, 3:7],
        )
        self.metrics["position_error_left"] = th.norm(pos_error_left, dim=-1)
        self.metrics["orientation_error_left"] = th.norm(rot_error_left, dim=-1)

        # Compute the error for right hand
        pos_error_right, rot_error_right = math_utils.compute_pose_error(
            self.lerp_command_w_right[:, :3],
            self.lerp_command_w_right[:, 3:],
            self.robot.data.body_state_w[:, self.right_hand_idx, :3],
            self.robot.data.body_state_w[:, self.right_hand_idx, 3:7],
        )
        self.metrics["position_error_right"] = th.norm(pos_error_right, dim=-1)
        self.metrics["orientation_error_right"] = th.norm(rot_error_right, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        

        # Sample new pose targets in attached cylindrical frame
    
        device = self.device
        num_envs = len(env_ids)

        self.curr_command_s_left[env_ids] = self.next_command_s_left[env_ids].clone()
        self.curr_command_s_right[env_ids] = self.next_command_s_right[env_ids].clone()

        reset_envs = th.where(self.command_counter == 1)

        self._update_cylinder_frame()

        reset_envs = reset_envs[0]
        self.curr_command_s_left[reset_envs, 0] = 0.3
        self.curr_command_s_left[reset_envs, 1] = 0.3
        self.curr_command_s_left[reset_envs, 2] = 0.6
        self.curr_command_s_right[reset_envs, 0] = 0.3
        self.curr_command_s_right[reset_envs, 1] = -0.3
        self.curr_command_s_right[reset_envs, 2] = 0.6

        if self.cfg.mode == "cylinder":
            next_pos_s_left = th.zeros((num_envs, 3), device=device)
            next_pos_s_left[..., 0] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.y_left_range)
            next_pos_s_left[..., 1] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.x_range)
            next_pos_s_left[..., 2] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.z_range)


            self.next_command_s_left[env_ids, :3]= cylinder2cart(next_pos_s_left)
            next_euler_s_left = th.zeros((num_envs, 3), device=device)
            next_euler_s_left[..., 2] = next_pos_s_left[..., 1]

            noise_left = th.empty((num_envs, 3), device=device).uniform_(
                -np.pi * self.cfg.angle_noise/180., 
                np.pi * self.cfg.angle_noise/180.)
            next_euler_s_left = math_utils.wrap_to_pi(next_euler_s_left+noise_left)

            self.next_command_s_left[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
                next_euler_s_left[..., 0],
                next_euler_s_left[..., 1],
                next_euler_s_left[..., 2],
            )

            next_pos_s_right = th.zeros((num_envs, 3), device=device)
            next_pos_s_right[..., 0] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.r_range)
            next_pos_s_right[..., 1] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.theta_range_right)
            next_pos_s_right[..., 2] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.z_range)


            self.next_command_s_right[env_ids, :3]= cylinder2cart(next_pos_s_right)
            next_euler_s_right = th.zeros((num_envs, 3), device=device)
            next_euler_s_right[..., 2] = next_pos_s_right[..., 1]

            noise_right = th.empty((num_envs, 3), device=device).uniform_(
                -np.pi * self.cfg.angle_noise/180., 
                np.pi * self.cfg.angle_noise/180.)
            next_euler_s_right = math_utils.wrap_to_pi(next_euler_s_right+noise_right)

            self.next_command_s_right[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
                next_euler_s_right[..., 0],
                next_euler_s_right[..., 1],
                next_euler_s_right[..., 2],
            )
        elif self.cfg.mode == "cart":
            next_pos_s_left = th.zeros((num_envs, 3), device=device)
            next_pos_s_left[..., 0] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.x_range)
            next_pos_s_left[..., 1] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.y_left_range)
            next_pos_s_left[..., 2] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.z_range)

            self.next_command_s_left[env_ids, :3]= next_pos_s_left

            next_euler_s_left = th.zeros((num_envs, 3), device=device)

            noise_left = th.empty((num_envs, 3), device=device).uniform_(
                -np.pi * self.cfg.angle_noise/180., 
                np.pi * self.cfg.angle_noise/180.)
            next_euler_s_left = math_utils.wrap_to_pi(next_euler_s_left+noise_left)

            self.next_command_s_left[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
                next_euler_s_left[..., 0],
                next_euler_s_left[..., 1],
                next_euler_s_left[..., 2],
            )

            next_pos_s_right = th.zeros((num_envs, 3), device=device)
            next_pos_s_right[..., 0] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.x_range)
            next_pos_s_right[..., 1] = th.empty(num_envs, device=device).uniform_(
                *self.cfg.ranges.y_right_range)
            # next_pos_s_right[..., 2] = th.empty(num_envs, device=device).uniform_(
            #     *self.cfg.ranges.z_range)
            next_pos_s_right[..., 2] = next_pos_s_left[..., 2].clone()

            self.next_command_s_right[env_ids, :3]= next_pos_s_right

            next_euler_s_right = th.zeros((num_envs, 3), device=device)

            noise_right = th.empty((num_envs, 3), device=device).uniform_(
                -np.pi * self.cfg.angle_noise/180., 
                np.pi * self.cfg.angle_noise/180.)
            next_euler_s_right = math_utils.wrap_to_pi(next_euler_s_right+noise_right)

            self.next_command_s_right[env_ids, 3:7] = math_utils.quat_from_euler_xyz(
                next_euler_s_right[..., 0],
                next_euler_s_right[..., 1],
                next_euler_s_right[..., 2],
            )
    
    def _update_cylinder_frame(self):
        if self.cfg.frame == "torso":
            self.cylinder_pos_w = self.robot.data.body_state_w[:, self.torso_idx, :3]
            self.cylinder_quat_w = self.robot.data.body_state_w[:, self.torso_idx, 3:7]
        elif self.cfg.frame == "z-inv":
            self.cylinder_pos_w[..., :2] = self.robot.data.root_pos_w[..., :2]
            self.cylinder_quat_w = math_utils.yaw_quat(self.robot.data.root_quat_w)
        elif self.cfg.frame == "foot":
            self.cylinder_pos_w[..., :2] = 0.5 * (
                self.robot.data.body_state_w[:, self.left_foot_idx, :2] +
                self.robot.data.body_state_w[:, self.right_foot_idx, :2]
                )
            
            _, _, left_euler = math_utils.euler_xyz_from_quat(
                math_utils.yaw_quat(self.robot.data.body_state_w[:, self.left_foot_idx, 3:7])
            )
            _, _, right_euler = math_utils.euler_xyz_from_quat(
                # self.robot.data.body_state_w[:, self.right_foot_idx, 3:7]
                math_utils.yaw_quat(self.robot.data.body_state_w[:, self.right_foot_idx, 3:7])
            )
            # Compute average angle of two euler angles
            avg_cos = (th.cos(left_euler) + th.cos(right_euler)) / 2
            avg_sin = (th.sin(left_euler) + th.sin(right_euler)) / 2

            self.cylinder_quat_w = math_utils.quat_from_euler_xyz(
                th.zeros_like(left_euler),
                th.zeros_like(left_euler),
                th.atan2(avg_sin, avg_cos)
            )

    def _update_command(self):
        '''
        This is different from _resample_command method. Since the robot
        poses varies as it moves, we need to refresh the target pose
        considering the robot's movement.
        '''

        self._update_cylinder_frame()

        # interpolation = 1. - (self.time_left/self.resampling_time)
        interpolation = (1. - (self.time_left/self.cfg.moving_time)).clip(
            min=0., max=1.
        )

        self.lerp_command_s_left = interpolate_pose(
            self.curr_command_s_left,
            self.next_command_s_left,
            interpolation
        )
        self.lerp_command_w_left[:, :3], self.lerp_command_w_left[:, 3:] = \
            math_utils.combine_frame_transforms(
                self.cylinder_pos_w,
                self.cylinder_quat_w,
                self.lerp_command_s_left[:, :3],
                self.lerp_command_s_left[:, 3:])

        self.lerp_command_s_right = interpolate_pose(
            self.curr_command_s_right,
            self.next_command_s_right,
            interpolation
        )
        self.lerp_command_w_right[:, :3], self.lerp_command_w_right[:, 3:] = \
            math_utils.combine_frame_transforms(
                self.cylinder_pos_w,
                self.cylinder_quat_w,
                self.lerp_command_s_right[:, :3],
                self.lerp_command_s_right[:, 3:])

    def get_bbox_right(self):
        body_pose_w_right = self.robot.data.body_state_w[:, self.right_hand_idx]
        #right hand
        rhb = math_utils.transform_points(
            self._hand_bboxes,
            body_pose_w_right[..., :3],
            body_pose_w_right[..., 3:7]
        )
        rgb = math_utils.transform_points(
            self._hand_bboxes,
            self.lerp_command_w_right[..., :3],
            self.lerp_command_w_right[..., 3:7]
        )
        return rhb, rgb

    def get_bbox_left(self):
        body_pose_w_left = self.robot.data.body_state_w[:, self.left_hand_idx]
        #left hand
        rhb = math_utils.transform_points(
            self._hand_bboxes,
            body_pose_w_left[..., :3],
            body_pose_w_left[..., 3:7]
        )
        rgb = math_utils.transform_points(
            self._hand_bboxes,
            self.lerp_command_w_left[..., :3],
            self.lerp_command_w_left[..., 3:7]
        )
        return rhb, rgb

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer_right"):
                if self.cfg.vis_left:
                    self.goal_pose_visualizer_left = VisualizationMarkers(
                        self.cfg.goal_pose_visualizer_cfg
                    )
                    self.current_pose_visualizer_left = VisualizationMarkers(
                        self.cfg.current_pose_visualizer_cfg
                    )
                if self.cfg.vis_right:
                    self.goal_pose_visualizer_right = VisualizationMarkers(
                        self.cfg.goal_pose_visualizer_cfg
                    )
                    self.current_pose_visualizer_right = VisualizationMarkers(
                        self.cfg.current_pose_visualizer_cfg
                    )
            if self.cfg.vis_left:
                self.goal_pose_visualizer_left.set_visibility(True)
                self.current_pose_visualizer_left.set_visibility(True)
            if self.cfg.vis_right:
                self.goal_pose_visualizer_right.set_visibility(True)
                self.current_pose_visualizer_right.set_visibility(True)
            if self.cfg.vis_hand_bbox:
                if not hasattr(self, "bbox_visualizer"):
                    colors = list(itertools.product([0., 1.], repeat=3))
                    if hasattr(self, '_hand_bboxes'):
                        n_points = self._hand_bboxes.shape[1]
                    else:
                        n_points = len(self.cfg.hand_bbox)
                        pass
                        # bbox = np.load(self.cfg.hand_bbox_file)
                        # n_points = bbox.shape[0]
                    self.bbox_visualizer = []
                    for i in range(n_points):
                        cc = self.cfg.bbox_vis_ref_cfg.replace(
                            prim_path=f"/Visuals/Command/bbox_{i}"
                        )
                        cc.markers['hit'].visual_material.diffuse_color=colors[i]
                        self.bbox_visualizer.append(VisualizationMarkers(cc))
                for v in self.bbox_visualizer:
                    v.set_visibility(True)
                
        else:
            if hasattr(self, "goal_pose_visualizer_left"):
                # self.goal_pose_visualizer_left.set_visibility(False)
                self.goal_pose_visualizer_right.set_visibility(False)
                # self.current_pose_visualizer_left.set_visibility(False)
                self.current_pose_visualizer_right.set_visibility(False)
            if self.cfg.vis_hand_bbox:
                if hasattr(self, "bbox_visualizer"):
                    for v in self.bbox_visualizer:
                        v.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Check if robot is initialized
        if not self.robot.is_initialized:
            return
        
        if self.cfg.vis_left:
            self.goal_pose_visualizer_left.visualize(
                self.lerp_command_w_left[:, :3], self.lerp_command_w_left[:, 3:]
            )
            body_pose_w_left = self.robot.data.body_state_w[:, self.left_hand_idx]
            self.current_pose_visualizer_left.visualize(
                body_pose_w_left[:, :3], body_pose_w_left[:, 3:7]
            )
        if self.cfg.vis_right:
            self.goal_pose_visualizer_right.visualize(
                self.lerp_command_w_right[:, :3], self.lerp_command_w_right[:, 3:]
            )
            body_pose_w_right = self.robot.data.body_state_w[:, self.right_hand_idx]
            self.current_pose_visualizer_right.visualize(
                body_pose_w_right[:, :3], body_pose_w_right[:, 3:7]
            )
        if self.cfg.vis_hand_bbox:
            #right hand
            rhb, rgb = self.get_bbox_right()
            lhb, lgb = self.get_bbox_left()

            vis_bbox = []
            if self.cfg.vis_left:
                vis_bbox.append(lhb)
                vis_bbox.append(lgb)
            if self.cfg.vis_right:
                vis_bbox.append(rhb)
                vis_bbox.append(rgb)

            bb = th.cat(vis_bbox, dim=0)
            for idx, vis in enumerate(self.bbox_visualizer):
                vis.visualize(bb[..., idx, :])