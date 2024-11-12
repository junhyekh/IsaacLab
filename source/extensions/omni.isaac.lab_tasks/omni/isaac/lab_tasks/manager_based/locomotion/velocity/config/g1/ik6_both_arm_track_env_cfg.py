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
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


import domi.env.help.zmp as zmp
from domi.sim.contact_sensor_extra_cfg import ContactSensorExtraCfg
from domi.sim.contact_sensor_extra import ContactSensorExtra, ContactSensorExtraData

from icecream import ic

from . import standing_env_cfg as stand_env
from . import arm_track_env_cfg as arm_track_env


    
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

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot",
                                           joint_names=[".*_hip_yaw_joint",
                                                        ".*_hip_roll_joint",
                                                        ".*_hip_pitch_joint",
                                                        ".*_knee_joint",
                                                        ".*_ankle_pitch_joint", 
                                                        ".*_ankle_roll_joint",
                                                        "left_shoulder_pitch_joint",
                                                        # "left_shoulder_roll_joint",
                                                        # "left_shoulder_yaw_joint",
                                                        # "left_elbow_joint",
                                                        # "left_wrist_.*",
                                                        "waist_.*",
                                                        "right_shoulder_pitch_joint",
                                                        ],
                                           scale=0.5,
                                           use_default_offset=True)
    right_arm = mdp.PassiveIKActionCfg(
            asset_name="robot",
            command_name='hands_pose',
            joint_names=[
                        # "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                        "right_wrist_.*",],

            body_name="right_hand_palm_link",
            control_arm="right",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                # ik_params={"lambda_val": 0.1},
                ik_params={"lambda_val": 0.05},
                # use_weighted_jacobian=True,
                use_weighted_jacobian=False,
                # use_max_clipping=False,
                use_max_clipping=True,
                max_delta_pos=0.5,
                weight_pos=[1.0, 1.0, 1.0, 0., 0., 0.],
                # weight_pos=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                weight_ori=[0., 0., 0., 1.0, 1.0, 1.0],
                # weight_ori=[0., 0., 0., 0., 0.1, 0.1, 0.1],
                # weight_ori=[0., 0., 0., 0., 0., 0., 0.],
                ),
            scale=1.0,
            compensate_gravity=True,
        )
    left_arm = mdp.PassiveIKActionCfg(
            asset_name="robot",
            command_name='hands_pose',
            joint_names=[
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "left_wrist_.*",],

            body_name="left_hand_palm_link",
            control_arm="left",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
                use_weighted_jacobian=False,
                use_max_clipping=True,
                max_delta_pos=0.5,
                weight_pos=[1.0, 1.0, 1.0, 0., 0., 0.],
                weight_ori=[0., 0., 0., 1.0, 1.0, 1.0],
                ),
            scale=1.0,
            compensate_gravity=True,
        )



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
        foot_pose = ObsTerm(func=stand_env.foot_pose_in_robot_root_frame)
        # hand_pose = ObsTerm(func=stand_env.hand_pose_in_robot_root_frame)
        projected_com = ObsTerm(
            func=stand_env.projected_coms,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        projected_zmp = ObsTerm(
            func=stand_env.projected_zmps,
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
        # momentum_change = ObsTerm(
        #     func=arm_track_env.relative_arm_mom_dev,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", 
        #             body_names=["right_shoulder_pitch_link",
        #                         "right_shoulder_roll_link",
        #                         "right_shoulder_yaw_link",
        #                         "right_elbow_link",
        #                         "right_wrist_.*",]),
        #     },
        #     scale=0.
        # )
        # momentum = ObsTerm(
        #     func=arm_track_env.relative_arm_mom,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", 
        #             body_names=["right_shoulder_pitch_link",
        #                         "right_shoulder_roll_link",
        #                         "right_shoulder_yaw_link",
        #                         "right_elbow_link",
        #                         "right_wrist_.*",]),
        #     },
        #     scale=0.
        # )
        right_arm_com = ObsTerm(
            func=arm_track_env.relative_arm_com,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    body_names=["right_shoulder_pitch_link",
                                "right_shoulder_roll_link",
                                "right_shoulder_yaw_link",
                                "right_elbow_link",
                                "right_wrist_.*",]),
            },
            scale=1.
        )
        left_arm_com = ObsTerm(
            func=arm_track_env.relative_arm_com,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    body_names=["left_shoulder_pitch_link",
                                "left_shoulder_roll_link",
                                "left_shoulder_yaw_link",
                                "left_elbow_link",
                                "left_wrist_.*",]),
            },
            scale=1.
        )



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
            "position_range": (0.9, 1.1),
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

    # elbow joint limit
    robot_joint_limits_elbow = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names="right_elbow_joint"),
            "lower_limit_distribution_params": (-1., -1.),
            "upper_limit_distribution_params": (1.5, 1.5),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_joint_limits_shoulder = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names="right_shoulder_yaw_joint"),
            "lower_limit_distribution_params": (-0.5, -0.5),
            "upper_limit_distribution_params": (2.6, 2.6),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


@configclass
class G1Rewards:
    """Reward terms for the MDP."""
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # -- task
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        # weight=-1.0e-5,
        weight=0.,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[
                            ".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                            ".*_ankle_pitch_joint", 
                            ".*_ankle_roll_joint",
                            "left_shoulder_pitch_joint",
                            # "left_shoulder_roll_joint",
                            # "left_shoulder_yaw_joint",
                            # "left_elbow_joint",
                            # "left_wrist_.*",
                            "waist_.*",
                            "right_shoulder_pitch_joint",
                            ])
        },
    )
    rel_torques_l2 = RewTerm(
        func=arm_track_env.rel_joint_torques_l2, 
        weight=-0.3,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                            ".*_ankle_pitch_joint", 
                            ".*_ankle_roll_joint",
                            "left_shoulder_pitch_joint",
                            # "left_shoulder_roll_joint",
                            # "left_shoulder_yaw_joint",
                            # "left_elbow_joint",
                            # "left_wrist_.*",
                            "waist_.*",
                            "right_shoulder_pitch_joint",
                            ])
        },
    )
    ankle_dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, 
        # weight=-5.0e-4,
        weight=0.,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[
                            ".*_ankle_roll_joint",
                            ".*_ankle_pitch_joint", 
                ])
        }
    )
    # angmom_penalty = RewTerm(
    #     func=stand_env.ang_momentum,
    #     weight=0
    #     # weight=-0.5
    # )
    energy = RewTerm(
        func=stand_env.energy,
        # weight=-0.002
        # weight=-0.001,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                            ".*_ankle_pitch_joint", 
                            ".*_ankle_roll_joint",
                            "left_shoulder_pitch_joint",
                            # "left_shoulder_roll_joint",
                            # "left_shoulder_yaw_joint",
                            # "left_elbow_joint",
                            # "left_wrist_.*",
                            "waist_.*",
                            "right_shoulder_pitch_joint",
                            ])
        },
        # weight=-0.0002
        weight=-0.
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, 
        weight=-2.0e-8,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                            ".*_ankle_pitch_joint", 
                            ".*_ankle_roll_joint",
                            "left_shoulder_pitch_joint",
                            # "left_shoulder_roll_joint",
                            # "left_shoulder_yaw_joint",
                            # "left_elbow_joint",
                            # "left_wrist_.*",
                            "waist_.*",
                            "right_shoulder_pitch_joint",
                            ])
        },
    )
    # dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-5)
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2, 
        weight=-5e-4,
        params={
            'asset_cfg':SceneEntityCfg(
                'robot',
                joint_names=[".*_hip_yaw_joint",
                            ".*_hip_roll_joint",
                            ".*_hip_pitch_joint",
                            ".*_knee_joint",
                            ".*_ankle_pitch_joint", 
                            ".*_ankle_roll_joint",
                            "left_shoulder_pitch_joint",
                            # "left_shoulder_roll_joint",
                            # "left_shoulder_yaw_joint",
                            # "left_elbow_joint",
                            # "left_wrist_.*",
                            "waist_.*",
                            "right_shoulder_pitch_joint",
                            ])
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)
    zmp_supp_dist = RewTerm(
        func=stand_env.zmp_supp_dist,
        weight=0.3,
        # weight=0.1,
        # weight=0.,
        params={
            "sigma": 10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    zmp_centroid_dist = RewTerm(
        func=stand_env.zmp_centroid_dist,
        # weight=0.5,
        weight=0.,
        params={
            "sigma": 30,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # zmp_centroid_dist_v2 = RewTerm(
    #     func=zmp_centroid_dist_v2,
    #     # weight=0.5,
    #     weight=0.,
    #     params={
    #         "sigma": 30,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    zmp_com_dist = RewTerm(
        func=stand_env.com_supp_dist,
        weight=0,
        params={
            "sigma_1": 10,
            "sigma_2": 4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Tracking rewards
    hand_pos_tracking = RewTerm(
        func=arm_track_env.position_command_error,
        # weight=-1,
        weight=0.,
        params={"command_name": "hands_pose"},
    )
    hand_ori_tracking = RewTerm(
        func=arm_track_env.orientation_command_error,
        # weight=-0.2,
        # weight=-0.5,
        weight=0.,
        params={"command_name": "hands_pose"},
    )
    # walk_to_target = RewTerm(
    #     func=stand_env.walk_to_target,
    #     # weight=1.0,
    #     # weight=0.8,
    #     weight=0.8,
    #     params={
    #         "command_name": "global_hand_goal",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_vel_x": 1.0,
    #         "target_vel_y": 0.,
    #         "close_threshold": 0.5,
    #         "c1": 0.4,
    #         "c2": 0.4,
    #         "c3": 0.2,
    #     },
    # )
    # maintaining_target = RewTerm(
    #     func=stand_env.maintaining_target,
    #     weight=0.8,
    #     params={
    #         "command_name": "global_hand_goal",
    #         "close_threshold": 0.5,
    #         "c_pos": 0.5,
    #         "c_ori": 0.25,
    #         "c_ori_base": 0.25,
    #         "c_pos_base": 0.25
    #     }
    # )
    # air_time = RewTerm(
    #     func=stand_env.air_time,
    #     weight=0.,
    #     params={
    #         "command_name": "global_hand_goal",
    #         "close_threshold": 0.5,
    #         "threshold": 0.2,
    #     }
    # )
    zmp_supp_dist_v2 = RewTerm(
        func=arm_track_env.zmp_supp_dist_v2,
        # weight=0.3,
        weight=0.0,
        # weight=0.,
        params={
            "sigma": 10,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # max_consecutive_success = RewTerm(
    #     func=stand_env.max_consecutive_success, 
    #     # weight=500.,
    #     # weight=1000.,
    #     # weight=2000.,
    #     weight=5000.,
    #     params={"num_success": 100, "command_name": "global_hand_goal"}
    # )
    torso_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.,
        # weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            # "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        }
    )
    pelvis_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.,
        # weight=-0.002,
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        }
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.5,
        # weight=0.,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    approaching_right = RewTerm(
        func=arm_track_env.approaching_pose,
        # weight=0.,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["right_shoulder_pitch_joint",
                            "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint",
                            "right_elbow_joint",
                            "right_wrist_.*",]),
            "command_name": "hands_pose",
            "penalize_joint_limit": True,
            "arm": "right",
            "sigma": 10.
        },
    )
    approaching_left = RewTerm(
        func=arm_track_env.approaching_pose,
        # weight=0.,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=["left_shoulder_pitch_joint",
                            "left_shoulder_roll_joint",
                            "left_shoulder_yaw_joint",
                            "left_elbow_joint",
                            "left_wrist_.*",]),
            "command_name": "hands_pose",
            "penalize_joint_limit": True,
            "arm": "left",
            "sigma": 10.
        },
    )

    maintain_target = RewTerm(
        func=arm_track_env.maintain_target,
        weight=0.,
        # weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "hands_pose"
        }
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
        func=stand_env.bad_ori,
        params={"limit_euler_angle": [0.7, 1.0]})
    # max_consecutive_success = DoneTerm(
    #     func=max_consecutive_success, 
    #     params={"num_success": 100, "command_name": "global_hand_goal"}
    # )
    jump = DoneTerm(
        func=arm_track_env.jump
    )

@configclass
class CommandsCfg:
    # hands_pose = mdp.LocalHandPoseCommandCfg(
    #     class_type=mdp.LocalHandPoseCommand,
    #     asset_name="robot",
    #     resampling_time_range=(3.0, 3.0),
    #     left_hand_body_name="left_palm_link",
    #     right_hand_body_name="right_palm_link",
    #     debug_vis=True,
    #     ranges=mdp.LocalHandPoseCommandCfg.Ranges(
    #         # r_range=(0.4, 0.6),
    #         # r_range=(0.25, 0.55),
    #         r_range=(0.3, 0.55),
    #         theta_range=(-np.pi/4, np.pi/4),
    #         # z_range=(0.25, 1.0),
    #         z_range=(0.3, 1.2),
    #     ),
    #     hand_shift=0.15,
    #     delta_yaw=30.

    # )
    hands_pose = mdp.IKHandTrajCommandCfg(
        class_type=mdp.IKHandTrajCommand,
        asset_name="robot",
        resampling_time_range=(3., 3.),
        # resampling_time_range=(6., 6.),
        moving_time=3.,
        left_hand_body_name="left_hand_palm_link",
        right_hand_body_name="right_hand_palm_link",
        left_foot_body_name="left_ankle_roll_link",
        right_foot_body_name="right_ankle_roll_link",
        torso_body_name="torso_link",
        debug_vis=True,
        # mode="cylinder",
        mode="cart",
        frame="foot",
        angle_noise=0.001,
        # frame="z-inv",
        ranges=mdp.IKHandTrajCommandCfg.Ranges(
            # r_range=(0.5, 0.6),
            # r_range=(0.2, 0.6),
            # r_range=(0.3, 0.6),
            r_range=(0.3, 0.65),
            theta_range_right=(-np.pi/3, 0.),
            theta_range_left=(0, np.pi/4),
            # z_range=(0.1, 0.5),
            # z_range=(0.4, 1.0),
            # z_range=(0.6, 1.0),
            z_range=(0.4, 1.1),
            # x_range=(0.15, 0.6),
            # x_range=(0.20, 0.55),
            x_range=(0.15, 0.5),
            # x_range=(0.05, 0.4),
            y_left_range=(0., 0.45),
            y_right_range=(-0.45, 0.),

        ),
    )

G1_29_FIXED_HAND_CFG =ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/extensions/omni.isaac.lab_assets/data/g1_29_non_convex/g1_hand.usd",
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
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_joint": 0.87,
            ".*_elbow_joint": 0.50,
            # "left_shoulder_roll_joint": 0.16,
            "left_shoulder_roll_joint": 0.35,
            # "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_pitch_joint": -0.20,
            # "right_shoulder_roll_joint": -0.16,
            "right_shoulder_roll_joint": -0.35,
            # "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_pitch_joint": -0.20,
            "waist_.*": 0.,
            ".*_wrist_.*": 0.
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "waist":  ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_.*"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "waist_.*": 150.0,
            },
            damping={
                "waist_.*": 5.0,
            },
            armature={
                "waist_.*": 0.01,
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
                ".*_elbow_joint",
                ".*_wrist_.*"
            ],
            effort_limit=100,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01
            },
        ),
    },
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
        self.episode_length_s = 20.
        # self.episode_length_s = 10.
        # self.episode_length_s = 3.
        # self.episode_length_s = 8.
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
        self.scene.robot = G1_29_FIXED_HAND_CFG.replace(
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
