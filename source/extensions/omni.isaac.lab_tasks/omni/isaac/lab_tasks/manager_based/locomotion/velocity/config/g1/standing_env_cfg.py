# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch as th
from dataclasses import MISSING
from typing import TYPE_CHECKING, List
from collections.abc import Sequence

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, ManagerBasedEnv
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CommandTermCfg, CommandTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

import domi.env.help.zmp as zmp
from domi.sim.contact_sensor_extra_cfg import ContactSensorExtraCfg
from domi.sim.contact_sensor_extra import ContactSensorExtra, ContactSensorExtraData

from icecream import ic



# Helper functions for calculating observation, rewards, and terminations
def projected_coms(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> th.Tensor:
    """Compute the projected coms wrt to the base frame"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.com_pos_w = zmp.compute_com(asset, env.device)
    asset.data.com_pos_b = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w,
        asset.data.com_pos_w- asset.data.root_pos_w
    )
    return asset.data.com_pos_b[..., :2]

def projected_zmps(
        env: ManagerBasedEnv, 
        asset_cfg: SceneEntityCfg,
        left_foot_sensor_cfg: SceneEntityCfg,
        right_foot_sensor_cfg: SceneEntityCfg
    ) -> th.Tensor:
    """Compute the projected coms wrt to the base frame"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    left_forces, left_points, left_masks = \
        zmp.process_contact_data(
            left_foot_sensor.data.c_force,
            left_foot_sensor.data.c_normal,
            left_foot_sensor.data.c_point,
            left_foot_sensor.data.c_idx,
            left_foot_sensor.data.c_num)

    right_forces, right_points, right_masks = \
        zmp.process_contact_data(
            right_foot_sensor.data.c_force,
            right_foot_sensor.data.c_normal,
            right_foot_sensor.data.c_point,
            right_foot_sensor.data.c_idx,
            right_foot_sensor.data.c_num)
            
    com_pos_w = zmp.compute_com(asset, env.device)
    asset.data.zmp_pos_w = zmp.compute_zmp(
        th.cat([left_forces, right_forces], dim=1),
        th.cat([left_points, right_points], dim=1),
        th.cat([left_masks, right_masks], dim=1),
        com_pos_w
    )
    asset.data.hull_points, asset.data.hull_idx = \
        zmp.compute_2d_convex_hull(
            th.cat([left_points, right_points], dim=1)[..., :2],
            th.cat([left_masks, right_masks], dim=1)
    )
    asset.data.zmp_pos_b = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w,
        asset.data.zmp_pos_w - asset.data.root_pos_w
    )
    return asset.data.zmp_pos_b[..., :2]


def zmp_supp_dist(
        env: ManagerBasedRLEnv, 
        sigma: float,
        asset_cfg: SceneEntityCfg,
    ) -> th.Tensor:
    # """
    # Computes the distance/margin between the support polygon and the zmp
    # """
    asset: Articulation = env.scene[asset_cfg.name]

    signed_zmp_dist = zmp.hull_point_signed_dist(
        asset.data.hull_points, 
        asset.data.hull_idx, 
        asset.data.zmp_pos_w[..., :2]
    )
    rew = (th.exp(sigma * -th.clip(signed_zmp_dist, max=0))-1)

    if "ZMP_margin" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["ZMP_margin"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["ZMP_margin"] += \
        -th.clip(signed_zmp_dist, max=0)

    # Code for COM, ZMP visualization
    if not hasattr(env, "com_markers"):
        com_markers_cfg= VisualizationMarkersCfg(
            prim_path="/Visuals/com",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            }
        )
        env.com_markers = VisualizationMarkers(com_markers_cfg)
        zmp_markers_cfg= VisualizationMarkersCfg(
            prim_path="/Visuals/zmp",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            }
        )
        env.zmp_markers = VisualizationMarkers(zmp_markers_cfg)
    proj_com = th.zeros_like(asset.data.com_pos_w)
    proj_com[..., :2] = asset.data.com_pos_w[..., :2]
    env.com_markers.visualize(proj_com)
    env.zmp_markers.visualize(asset.data.zmp_pos_w)

    return rew

def zmp_centroid_dist(
        env: ManagerBasedRLEnv, 
        sigma: float,
        asset_cfg: SceneEntityCfg,
        area_thresh: float = 1e-4
    ) -> th.Tensor:
    # """
    # Computes the distance/margin between the support polygon and the zmp
    # """
    asset: Articulation = env.scene[asset_cfg.name]

    centroid, area = zmp.compute_2d_hull_centroid(
        asset.data.hull_points, 
        asset.data.hull_idx
    )
    # Squared distance between the zmp and the support polygon centroid
    dist_l2 = th.sum(th.square(centroid-asset.data.zmp_pos_w[..., :2]), dim=1)
    norm_dist_l2 = dist_l2 / area

    if "ZMP_centroid_dist" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["ZMP_centroid_dist"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)
        env.reward_manager.episode_stat_sums["Supp_area"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["ZMP_centroid_dist"] += th.where(
        area > area_thresh, 
        dist_l2 ** 0.5, 
        th.zeros_like(area))
    env.reward_manager.episode_stat_sums["Supp_area"] += area

    rew = th.where(
        area > area_thresh, 
        th.exp(-sigma * norm_dist_l2), 
        th.zeros_like(area))
    
    return rew

def zmp_centroid_dist_v2(
        env: ManagerBasedRLEnv, 
        sigma: float,
        asset_cfg: SceneEntityCfg,
        area_thresh: float = 1e-4
    ) -> th.Tensor:
    """
    Computes the distance/margin between the support polygon and the zmp
    Rewards based on the direct distance
    """
    asset: Articulation = env.scene[asset_cfg.name]

    centroid, area = zmp.compute_2d_hull_centroid(
        asset.data.hull_points, 
        asset.data.hull_idx
    )
    # Squared distance between the zmp and the support polygon centroid
    dist_l2 = th.sum(th.square(centroid-asset.data.zmp_pos_w[..., :2]), dim=1)
    
    rew = th.where(
        area > area_thresh, 
        th.exp(-sigma * (dist_l2 ** 0.5)), 
        th.zeros_like(area))
    
    return rew

def com_supp_dist(
        env: ManagerBasedRLEnv, 
        sigma_1: float,
        sigma_2: float,
        asset_cfg: SceneEntityCfg,
    ) -> th.Tensor:
    # """
    # Computes the distance/margin between the support polygon and the com
    # """
    asset: Articulation = env.scene[asset_cfg.name]

    singed_com_dist = zmp.hull_point_signed_dist(
        asset.data.hull_points, 
        asset.data.hull_idx, 
        asset.data.com_pos_w[..., :2]
    )

    if "COM_margin" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["COM_margin"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)
        env.reward_manager.episode_stat_sums["COM_dist"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["COM_margin"] += \
        -th.clip(singed_com_dist, max=0)
    env.reward_manager.episode_stat_sums["COM_dist"] += \
        th.clip(singed_com_dist, min=0)

    rew = th.exp(sigma_1 * -th.clip(singed_com_dist, max=0)) - \
          th.exp(sigma_2 * th.clip(singed_com_dist, min=0))

    return rew

def energy(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> th.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    energy = th.clip(asset.data.joint_vel * asset.data.applied_torque, min=0)

    if "Energy" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Energy"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["Energy"] += th.sum(energy, dim=-1)

    return th.sum(energy, dim=-1)

def bad_ori(
    env: ManagerBasedRLEnv, 
    limit_euler_angle: List[float] = [0.5, 1.5], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> th.Tensor:
    """Terminate when the asset's orientation is out of predefined range

    Args:
        limit_euler_angle: euler angle threshold [roll, pitch]. Episode
            will be terminated if the abs of the root euler angle will
            exceed this threshold
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    euler = math_utils.wrap_to_pi(th.stack(
        math_utils.euler_xyz_from_quat(asset.data.root_quat_w), dim=-1))
    out_of_limit = th.logical_or(
        th.abs(euler[..., 0]) > limit_euler_angle[0], 
        th.abs(euler[..., 1]) > limit_euler_angle[1])
    return out_of_limit

def ang_momentum(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> th.Tensor:
    '''
    Computes the derivative of the angular momentum using finite difference.
    '''
    asset: Articulation = env.scene[asset_cfg.name]

    lin_mom, ang_mom = zmp.compute_lin_ang_momentum(
        asset, asset.data.com_pos_w)

    if "Ang_momentum_abs" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Ang_momentum_abs"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["Ang_momentum_abs"] += \
        th.sum(th.abs(ang_mom), dim= -1)

    # NOTE(ytcho): Hardcoded 
    ang_mom_l2 = th.sum(th.square(ang_mom), dim= -1).clip(min=0, max=1.0)
    rew = (th.exp(ang_mom_l2)-1) * (env.episode_length_buf > 30).float()

    return rew

def foot_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="left_ankle_roll_link"),
    right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="right_ankle_roll_link"),
) -> th.Tensor:
    """The position of the object in the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    left_foot_ids = asset.find_bodies(left_foot_cfg.body_names)[0][0]
    right_foot_ids = asset.find_bodies(right_foot_cfg.body_names)[0][0]
    foot_pos_left_b, foot_quat_left_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, left_foot_ids, :3],
        asset.data.body_state_w[:, left_foot_ids, 3:7],
    )
    foot_pos_right_b, foot_quat_right_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        asset.data.body_state_w[:, right_foot_ids, :3],
        asset.data.body_state_w[:, right_foot_ids, 3:7],
    )
    foot_axa_left_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(foot_quat_left_b))
    foot_axa_right_b = math_utils.wrap_to_pi(
        math_utils.axis_angle_from_quat(foot_quat_right_b))
    
    foot_pose_b = th.cat(
        (foot_pos_left_b, foot_pos_right_b, foot_axa_left_b, foot_axa_right_b),
        dim=-1)
    
    return foot_pose_b

def position_command_error(
        env: ManagerBasedRLEnv, 
        command_name: str) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command = env.command_manager.get_command(command_name)
    pos_error = command[..., :6].norm(dim=-1)

    if "Left_hand_target_dist" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Left_hand_target_dist"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)
        env.reward_manager.episode_stat_sums["Right_hand_target_dist"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["Left_hand_target_dist"] += \
        command[..., :3].norm(dim=-1)
    env.reward_manager.episode_stat_sums["Right_hand_target_dist"] += \
        command[..., 3:6].norm(dim=-1)

    return pos_error

def orientation_command_error(
        env: ManagerBasedRLEnv, 
        command_name: str) -> th.Tensor:
    # extract the asset (to enable type hinting)
    command = env.command_manager.get_command(command_name)
    ori_error = command[..., 6:12].norm(dim=-1)

    if "Left_hand_target_ori" not in env.reward_manager.episode_stat_sums.keys():
        env.reward_manager.episode_stat_sums["Left_hand_target_ori"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)
        env.reward_manager.episode_stat_sums["Right_hand_target_ori"] = \
            th.zeros(env.num_envs, dtype=th.float, device=env.device)

    env.reward_manager.episode_stat_sums["Left_hand_target_ori"] += \
        command[..., 6:9].norm(dim=-1)
    env.reward_manager.episode_stat_sums["Right_hand_target_ori"] += \
        command[..., 9:12].norm(dim=-1)

    return ori_error

# End of helper functions


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # contact sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        foot_pose = ObsTerm(func=foot_pose_in_robot_root_frame)
        projected_com = ObsTerm(
            func=projected_coms,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        projected_zmp = ObsTerm(
            func=projected_zmps,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "left_foot_sensor_cfg": SceneEntityCfg("contact_left_foot"),
                "right_foot_sensor_cfg": SceneEntityCfg("contact_right_foot"),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        hands_command= ObsTerm(func=mdp.generated_commands, params={"command_name": "hands_pose"})


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        # mode="reset",
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (1000.0, 1000.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval push
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class G1Rewards:
    """Reward terms for the MDP."""
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # -- task
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        weight=0,
    )
    angmom_penalty = RewTerm(
        func=ang_momentum,
        weight=0
        # weight=-0.5
    )
    energy = RewTerm(
        func=energy,
        # weight=-0.002
        # weight=-0.001
        weight=-0.0002
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-2.0e-8,
    )
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    zmp_supp_dist = RewTerm(
        func=zmp_supp_dist,
        # weight=0.3,
        # weight=0.1,
        weight=0.,
        params={
            "sigma": 10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    zmp_centroid_dist = RewTerm(
        func=zmp_centroid_dist,
        # weight=0.5,
        weight=0.,
        params={
            "sigma": 30,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    zmp_centroid_dist_v2 = RewTerm(
        func=zmp_centroid_dist_v2,
        # weight=0.5,
        weight=0.,
        params={
            "sigma": 30,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    zmp_com_dist = RewTerm(
        func=com_supp_dist,
        weight=0,
        params={
            "sigma_1": 10,
            "sigma_2": 4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Tracking rewards
    hand_pos_tracking = RewTerm(
        func=position_command_error,
        weight=-1,
        params={"command_name": "hands_pose"},
    )
    hand_ori_tracking = RewTerm(
        func=orientation_command_error,
        # weight=-0.2,
        weight=-0.5,
        params={"command_name": "hands_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                # body_names=[".*_hip_.*", "head_link"]), 
                body_names=["head_link"]), 
            "threshold": 1.0},
    )
    torso_height = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": 0.25})
    
    bad_ori = DoneTerm(
        func=bad_ori,
        params={"limit_euler_angle": [0.5, 1.5]})

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)



@configclass
class CommandsCfg:
    hands_pose = mdp.HumanoidPoseCommandCfg(
        class_type=mdp.HumanoidPoseCommand,
        asset_name="robot",
        resampling_time_range=(3.0, 3.0),
        left_hand_body_name="left_palm_link",
        right_hand_body_name="right_palm_link",
        debug_vis=True,
        ranges=mdp.HumanoidPoseCommandCfg.Ranges(
            # r_range=(0.4, 0.6),
            # r_range=(0.25, 0.55),
            r_range=(0.3, 0.55),
            theta_range=(-np.pi/4, np.pi/4),
            # z_range=(0.25, 1.0),
            z_range=(0.3, 1.2),
        ),
        hand_shift=0.15,
        delta_yaw=30.

    )



@configclass
class G1StandingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # commands: CommandsCfg = CommandsCfg()
    # MDP settings
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    rewards: G1Rewards = G1Rewards()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # post init of parent
        """Post initialization."""
        # general settings
        self.decimation = 4
        # self.episode_length_s = 10.
        self.episode_length_s = 15
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Scene
        self.scene.robot = G1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            
            )

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Contact sensors
        self.scene.contact_left_foot = ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096
        )
        self.scene.contact_right_foot = ContactSensorExtraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            max_contact_data_count=8 * 4096
        )

        

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


class G1StandingEnvCfg_PLAY(G1StandingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
