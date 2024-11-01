
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

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import convert_quat
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab_assets.g1_hand import G1_Dual_Arm_CFG


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

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    robot = scene.articulations['robot']
    
    right_arm_bodies, body_names = robot.find_bodies("right_.*")
    right_arm_joints, joint_names = robot.find_joints("right_.*_joint")
    print(right_arm_bodies, body_names)
    print(robot.num_bodies)
    print(len(robot.find_bodies(".*")[0]))
    count = 0
    if args_cli.disable_gravity:
        robot.set_disable_gravity(body_ids=right_arm_bodies,
                                  env_ids=th.tensor([0,1]))
        # robot.set_disable_gravity(env_ids=th.tensor([0,1]))
        print(robot.root_physx_view.get_disable_gravities())
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 200 == 0:
            # reset counter
            count = 0
            default_joint_pos = robot.data.default_joint_pos
            default_joint_vel = robot.data.default_joint_vel
            # set into the physics simulation
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            #reset env 
                
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting object state...")
        # Apply random action
        # -- write data to sim
        robot.set_joint_position_target(robot.data.default_joint_pos)

        if args_cli.compensate_gravity:
            robot.set_joint_gravity_compensation(joint_ids=right_arm_joints)
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()