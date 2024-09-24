# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab_assets import CUSTOM_PANDA_CFG 
from omni.isaac.lab.utils.math import random_orientation
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
import omni.replicator.core as rep
from pathlib import Path
import numpy as np
import os

def define_sensor() -> Camera:
    """Defines the camera sensor to add to the scene."""
    # Setup camera sensor
    # In contrast to the ray-cast camera, we spawn the prim at these locations.
    # This means the camera sensor will be attached to these prims.
    camera_cfg = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        offset=CameraCfg.OffsetCfg(pos=(0.55, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="opengl"),
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create camera
    camera = Camera(cfg=camera_cfg)

    return camera

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

   

    # spawn a usd file of a table into the scene
    cfg = sim_utils.UsdFileCfg(usd_path="/input/env/usd/basket.usd",
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                               collision_props=sim_utils.CollisionPropertiesCfg())
    cfg.func("/World/Objects/Table", cfg, translation=(0.55, 0.0, 0.2))
    # panda_cfg = CUSTOM_PANDA_CFG.copy()
    # init_cfg = panda_cfg.init_state.copy()
    # init_cfg.pos = (0.0, 0.0, 0.4)
    # panda_cfg.init_state=init_cfg
    # panda_cfg.prim_path='/World/Robot/panda'
    # # panda_cfg.func("/World/Robot/panda", panda_cfg, translation=(0.0, 0.0, 0.4))
    # panda = Articulation(cfg=panda_cfg)

    robot = ArticulationCfg(
        prim_path="/World/Robot/panda",
        spawn=sim_utils.UsdFileCfg(
             usd_path="/input/robot/franka_description/robots/usd2/fr3_ability_hand.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": 1.157,
                "fr3_joint2": -1.066,
                "fr3_joint3": -0.155,
                "fr3_joint4": -2.239,
                "fr3_joint5": -1.841,
                "fr3_joint6": 1.003,
                "fr3_joint7": 0.469,
            },
            pos=(-0.555, 0.0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.62,
                stiffness=80.0,
                damping=4.0,
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=4.18,
                stiffness=80.0,
                damping=4.0,
            ),
            "thumb_joint1": ImplicitActuatorCfg(
                joint_names_expr=["thumb_q1"],
                effort_limit=1.2,
                velocity_limit=4,
                stiffness=80.0,
                damping=4.0,
            ),
            "thumb_joint2": ImplicitActuatorCfg(
                joint_names_expr=["thumb_q2"],
                effort_limit=6,
                velocity_limit=4,
                stiffness=80.0,
                damping=4.0,
            ),
            "finger_joints": ImplicitActuatorCfg(
                joint_names_expr=["middle_q1", "ring_q1", "index_q1", "pinky_q1"],
                effort_limit=6.0,
                velocity_limit=8,
                stiffness=80.0,
                damping=4.0,
            )
        },
    )

    objects = list(Path("source/extensions/omni.isaac.lab_assets/data/DGN").rglob("*.usd"))
    robot = Articulation(robot)
    selected = np.random.choice(objects, 50, False)

    target = selected[0]
    print(target)
    cfg = sim_utils.UsdFileCfg(usd_path=str(target),
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                                scale=(0.7, 0.7, 0.7),
                                semantic_tags =[("class", 'target')])
    q = random_orientation(1, "cpu")[0]
    t = np.array((0.55, 0.0, 0.9), dtype=np.float32)
    t[:2] += 0.05*np.random.randn(2)
    print(t, tuple(t.tolist()), q, q.shape)
    cfg.func("/World/Objects/target", cfg, translation=tuple(t.tolist()),   
                                            orientation=tuple(q.numpy().tolist())
                                            )
    for idx, ob in enumerate(selected[1:]):
        print(ob)
        cfg = sim_utils.UsdFileCfg(usd_path=str(ob),
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                                scale=(0.7, 0.7, 0.7),
                                semantic_tags =[("class", f'object_{idx}')])
        q = random_orientation(1, "cpu")[0]
        t = np.array((0.55, 0.0, 0.9+0.1*idx), dtype=np.float32)
        t[:2] += 0.05*np.random.randn(2)
        cfg.func(f"/World/Objects/object_{idx}", cfg, translation=tuple(t.tolist()),
                                                orientation=tuple(q.numpy().tolist()))
    scene_entities = {"panda": robot}

    return scene_entities

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device='cpu')
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        colorize_semantic_segmentation=True,
    )
    # Design scene by adding assets to it
    entities = design_scene()
    # camera = define_sensor()

    # cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
    # cfg.markers["hit"].radius = 0.002
    # pc_markers = VisualizationMarkers(cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()
        # camera.update(dt=sim.get_physics_dt())
        # # pointcloud = create_pointcloud_from_depth(
        # #     intrinsic_matrix=camera.data.intrinsic_matrices[0],
        # #     depth=camera.data.output[0]["distance_to_image_plane"],
        # #     position=camera.data.pos_w[0],
        # #     orientation=camera.data.quat_w_ros[0],
        # #     device=sim.device,
        # # )
        # single_cam_data = convert_dict_to_backend(camera.data.output[0], backend="numpy")

        # # Extract the other information
        # single_cam_info = camera.data.info[0]

        # Pack data back into replicator format to save them using its writer
        # rep_output = {"annotators": {}}
        # for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
        #     if info is not None:
        #         rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
        #     else:
        #         rep_output["annotators"][key] = {"render_product": {"data": data}}
        # # Save images
        # # Note: We need to provide On-time data for Replicator to save the images.
        # rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
        # rep_writer.write(rep_output)

        # # In the first few steps, things are still being instanced and Camera.data
        # # can be empty. If we attempt to visualize an empty pointcloud it will crash
        # # the sim, so we check that the pointcloud is not empty.
        # if pointcloud.size()[0] > 0:
        #     pc_markers.visualize(translations=pointcloud)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
