
import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--disable-gravity",
                    action="store_true",
                    default=False)
parser.add_argument("--compensate-gravity",
                    action="store_true",
                    default=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch as th 
import numpy as np

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, Articulation, RigidObject
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import convert_quat
# import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


G1_Dual_Arm_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"source/extensions/omni.isaac.lab_assets/data/g1_dual_arm/g1.usd",
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
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*_wrist_.*": 0.
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.,
    actuators={
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
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot: G1_Dual_Arm_CFG = G1_Dual_Arm_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    right_arm = mdp.PassiveIKActionCfg(
            asset_name="robot",
            command_name='hands_pose',
            joint_names=[
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                        "right_wrist_.*",],

            # body_name="right_hand_palm_link",
            body_name="right_rubber_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                # ik_params={"lambda_val": 0.2},
                ik_params={"lambda_val": 0.1},
                # ik_params={"lambda_val": 0.01},
                use_max_clipping=False,
                max_delta_pos=0.5,
                # use_weighted_jacobian=True,
                use_weighted_jacobian=False,
                weight_pos=[1.0, 1.0, 1.0, 1.0, 0., 0., 0.],
                # weight_pos=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                weight_ori=[0., 0., 0., 0., 1.0, 1.0, 1.0],
                # weight_ori=[0., 0., 0., 0., 0., 0., 0.],
                ),
            scale=1.0,
            compensate_gravity=True,
        )

@configclass
class CommandsCfg:
    hands_pose = mdp.IKHandTrajCommandCfg(
        class_type=mdp.IKHandTrajCommand,
        asset_name="robot",
        resampling_time_range=(3., 3.),
        moving_time=3.,
        mode="cart",
        left_hand_body_name="left_rubber_hand",
        # right_hand_body_name="right_hand_palm_link",
        right_hand_body_name="right_rubber_hand",
        left_foot_body_name="left_rubber_hand",
        right_foot_body_name="left_rubber_hand",
        torso_body_name="torso_link",
        frame="torso",
        debug_vis=True,
        ranges=mdp.IKHandTrajCommandCfg.Ranges(
            # r_range=(0.5, 0.6),
            r_range=(0.3, 0.6),
            theta_range_right=(-np.pi/3, 0.),
            theta_range_left=(0, np.pi/4),
            z_range=(0.1, 0.5),
            x_range=(0.15, 0.5),
            y_left_range=(0., 0.45),
            y_right_range=(-0.45, 0.),
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""
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
class G1DualArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    # observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.
        # self.episode_length_s = 3.
        # self.episode_length_s = 8.
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True


def main():
    """Main function."""
    # setup base environment
    env_cfg = G1DualArmEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)


    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with th.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = th.zeros((env.num_envs, 0), device=env.device)

            # step env
            # obs, _ = env.step(action)
            env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
