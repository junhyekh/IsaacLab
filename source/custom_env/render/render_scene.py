

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to spawn.")
parser.add_argument("--use_tile", action="store_true")
parser.add_argument("--source_file", type=str, default='0000')
parser.add_argument("--start_index", type=int)
parser.add_argument("--ep_id", type=int)
parser.add_argument("--render_object", action="store_true")
parser.add_argument("--render_scene", action="store_true")
parser.add_argument("--renderer", type=str, default='PathTracing')
parser.add_argument("--anti_aliasing", type=int, default=3)
parser.add_argument("--denoiser", type=bool, default=False)
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=256)
parser.add_argument("--max_bounces", type=int, default=64)
parser.add_argument("--max_volume_bounces", type=int, default=64)
parser.add_argument("--max_specular_transmission_bounces", type=int, default=64)

parser.add_argument("--r", type=float, default=0.)
parser.add_argument("--yaw", type=float, default=0.)
parser.add_argument("--x", type=float, default=0.)
parser.add_argument("--y", type=float, default=0.)
parser.add_argument("--z", type=float, default=0.)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

START_IDX = args_cli.start_index

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pickle
import torch
import numpy as np
import trimesh
from pathlib import Path
import os
from typing import Optional

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform, matrix_from_quat, euler_xyz_from_quat, quat_from_euler_xyz
from omni.isaac.lab.sensors.camera.utils import convert_orientation_convention
from omni.isaac.lab.sim.converters import UrdfConverterCfg, UrdfConverter
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.lab.sensors.camera import Camera, CameraCfg

from custom_env.scene.dgn_object_set import DGNObjectSet
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

import cv2
from custom_env.util.path import ensure_directory

def _list2str(d):
    return ' '.join(['{x:.03f}'.format(x=float(x)) for x in d])

def multi_box_link(name,
                   dims,
                   xyz: Optional[str] = None,
                   rpy: Optional[str] = None,
                   density: float = 0.0):
    assert(density == 0.0)

    if xyz is None:
        # FIXME(ycho): broadcast to
        # more than one batch dim
        xyz = ['0 0 0'] * len(dims)
    if rpy is None:
        rpy = ['0 0 0'] * len(dims)

    BOX_BODY_TEMPLATE: str = '''
        <visual>
            <origin xyz="{xyz}" rpy="{rpy}"/>
            <geometry>
                <box size="{s[0]} {s[1]} {s[2]}" />
            </geometry>
        </visual>
        <collision>
            <origin xyz="{xyz}" rpy="{rpy}"/>
            <geometry>
                <box size="{s[0]} {s[1]} {s[2]}" />
            </geometry>
        </collision>
    '''

    BOX_LINK_TEMPLATE: str = '''
        <link name="{name}">
        <inertial>
            <mass value="0"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="0" ixy="0" ixz="0"
            iyy="0" iyz="0" izz="0"/>
        </inertial>
        {bodies}
        </link>
    '''
    bodies = '\n'.join(
        [BOX_BODY_TEMPLATE.format(s=dim, xyz=x, rpy=r)
             for (dim, x, r) in zip(dims, xyz, rpy)
         ])
    return BOX_LINK_TEMPLATE.format(
        name=name,
        bodies=bodies
    )


@configclass
class RenderEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 8.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 23
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=1.3, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
             usd_path=f"/workspace/isaaclab/source/extensions/omni.isaac.lab_assets/data/Robots/custom/franka_panda_custom_v3.usd",
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
                "panda_joint1": 1.157,
                "panda_joint2": -1.066,
                "panda_joint3": -0.155,
                "panda_joint4": -2.239,
                "panda_joint5": -1.841,
                "panda_joint6": 1.003,
                "panda_joint7": 0.469,
            },
            pos=(-0.555, 0.0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=80.0,
                damping=4.0,
            )
        },
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0, 0, 0,),
                                                                        emissive_color=(0.,0.,0.),
                                                                        metallic=0.1,
                                                                        roughness=0.2)
    )
    
    use_tiled_camera: bool = False

class RednerEnv(DirectRLEnv):
    cfg: RenderEnvCfg

    def __init__(self, cfg: RenderEnvCfg,
                 render_mode: str | None = None, 
                 data = None,
                 **kwargs):
        self._data = data
        self.dgn = None
        # for k,v in self._data.items():
        #     if isinstance(v, np.ndarray):
        #         if k == 'name':
        #             continue
        #         self._data[k] = torch.as_tensor(v,
        #                                         device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, 
                                    self.cfg.num_actions,
                                    device=self.device)
        self._previous_actions = torch.zeros(self.num_envs,
                                             self.cfg.num_actions,
                                             device=self.device)
        self._joints_inits = torch.zeros(self.num_envs,
                                   7,
                                   device=self.device)
        
        # if "cuda" in self.device:
        
    def _create_boxes(self, box_dim, box_poses, prim_path,
                      env_origin, start):
        print(prim_path, env_origin, box_poses.shape, type(box_poses), 
              env_origin.shape, type(env_origin))
        # box_poses[:, 0] += env_origin[0]
        # box_poses[:, 1] += env_origin[1]
        # box_poses[:, 2] += env_origin[2]
        if start==2:
            print(box_dim)
            print(box_poses)
        for j in range(box_dim.shape[0]):
            vis_mat = (sim_utils.GlassMdlCfg(glass_color=(182/255.,215/255.,168/255.),
                                            frosting_roughness=0.,
                                            # thin_walled=True,
                                            # glass_ior=1.3
                                            ) if j>10 
                                            else sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)))
            cfg_cuboid = sim_utils.MeshCuboidCfg(
                            size=(box_dim[j,0], box_dim[j,1], box_dim[j,2]),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0),
                            #                                             opacity=opacity),
                            visual_material=vis_mat
                            )
            cfg_cuboid.func(f'{prim_path}/box_{j}',
                            cfg_cuboid,
                            translation=(box_poses[j,0],
                                        box_poses[j,1],
                                        box_poses[j,2]),
                            orientation=(
                                box_poses[j,6], 
                                box_poses[j,3],
                                box_poses[j,4],
                                box_poses[j,5]  
                            ))
        
    def _setup_scene(self):
        self.dgn = DGNObjectSet(DGNObjectSet.Config(data_path="/input/DGN/meta-v8"))

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        origins = self._terrain.env_origins.detach().clone().cpu().numpy()

        for i in range(0, self.scene.num_envs):
            prim_utils.create_prim(self.scene.env_prim_paths[i],
                                   "Xform", translation=origins[i])
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # clone, filter, and replicate
        # self.scene.clone_environments(copy_from_source=False)
        # manually set position of env 0 
        # origin0 = XFormPrimView(self.scene.env_prim_paths,
        #                         reset_xform_properties=False)
        # positions, orientations = origin0.get_world_poses()
        # positions[:] = self._terrain.env_origins[0]
        # origin0.set_world_poses(positions, orientations)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        print(origins, self.scene.env_origins)
        print(self.scene.env_prim_paths)
        # create cabinet
        for i in range(0, self.scene.num_envs):
            if args_cli.render_scene:
                self._create_boxes(self._data['box_dim'][i],
                                self._data['box_pose'][i],
                                    self.scene.env_prim_paths[i],
                                    origins[i], i)
            else:
                dest_path = F'/gate/boxes/{args_cli.source_file}/env_{START_IDX+i}/box.usd'
                if not Path(dest_path).exists():
                    urdf_file = F'/gate/boxes/{args_cli.source_file}/env_{START_IDX+i}.urdf'
                    xyzs = []
                    rpys = []
                    dims = []
                    for i_part, part in enumerate(self._data['box_dim'][i]):
                        xyz = _list2str(self._data['box_pose'][i][i_part][:3])
                        euler = euler_xyz_from_quat(
                            torch.as_tensor(self._data['box_pose'][i][i_part][3:7]).roll(1,-1)[None])
                        euler = [d.squeeze(0) for d in euler]
                        rpy = _list2str(euler)
                        xyzs.append(xyz)
                        rpys.append(rpy)
                        dims.append(list(part))
                    urdf_text: str = '''
                    <robot name="robot">
                    {box}
                    </robot>
                    '''.format(
                    box=multi_box_link(
                        F'base_link',
                        dims,
                        xyzs,
                        rpys,
                        density=0.0
                        )
                    )
                    urdf_file = F'/gate/boxes/{args_cli.source_file}/env_{START_IDX+i}/box.urdf'
                    os.makedirs(F'/gate/boxes/{args_cli.source_file}/env_{START_IDX+i}/',
                                exist_ok=True)
                    with open(str(urdf_file), 'w') as fp:
                        fp.write(urdf_text)

                    urdf_converter_cfg = UrdfConverterCfg(
                        asset_path=urdf_file,
                        usd_dir=os.path.dirname(dest_path),
                        usd_file_name=os.path.basename(dest_path),
                        fix_base=True,
                        convex_decompose_mesh=True,
                        merge_fixed_joints=True,
                        force_usd_conversion=True,
                        make_instanceable=False,
                    )
                    urdf_converter = UrdfConverter(urdf_converter_cfg)
                box_cfg = sim_utils.UsdFileCfg(usd_path=dest_path,
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0,),
                                                                        emissive_color=(0.,0.,0.),
                                                                        metallic=0.1,
                                                                        roughness=0.2))
                box_cfg.func(f"{self.scene.env_prim_paths[i]}/box", box_cfg, translation=(0, 0.0, 0.0))

        # spawn object
        if args_cli.render_object:
            for i in range(0, self.scene.num_envs):
                obj_key = self._data['name'][i]
                obj_usd = self.dgn.usd(obj_key)
                if not Path(obj_usd).exists():
                    urdf_file = self.dgn.urdf(obj_key)
                    urdf_converter_cfg = UrdfConverterCfg(
                            asset_path=urdf_file,
                            usd_dir=os.path.dirname(obj_usd),
                            usd_file_name=os.path.basename(obj_usd),
                            fix_base=False,
                            convex_decompose_mesh=True,
                            merge_fixed_joints=True,
                            force_usd_conversion=True,
                            make_instanceable=False,
                        )
                    urdf_converter = UrdfConverter(urdf_converter_cfg)
                scale = self._data['obj_scale'][i]
                obj_cfg = sim_utils.UsdFileCfg(usd_path=obj_usd,
                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                    # collision_props=sim_utils.CollisionPropertiesCfg(),
                                    scale=(scale, scale, scale),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(240/255., 110/255., 110/255.,),
                                                                                emissive_color=(0.05,0.0,0.0),
                                                                                metallic=0.0,
                                                                                roughness=0.2))
                pose = self._data['obj_pose'][i]
                goal = self._data['goal'][i]
                # print(pose, goal)
                obj_cfg.func(f"{self.scene.env_prim_paths[i]}/object", obj_cfg,
                            translation=(pose[0], pose[1], pose[2]),
                            orientation=(pose[6], pose[3], pose[4], pose[5]))
                
                goal_cfg = sim_utils.UsdFileCfg(usd_path=obj_usd,
                                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                                    scale=(scale, scale, scale),
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(112/255.,134/255.,192/255.),
                                                                                emissive_color=(0.0,0.0,0.05),
                                                                                # metallic=0.1,
                                                                                # roughness=0.2,
                                                                                # opacity=0.3
                                                                                ))
                                    # visual_material=sim_utils.GlassMdlCfg(glass_color=(112/255.,134/255.,192/255.),
                                    #                                       frosting_roughness=0.3))
                goal_cfg.func(f"{self.scene.env_prim_paths[i]}/goal", goal_cfg,
                            translation=(goal[0], goal[1], goal[2]),
                            orientation=(goal[6], goal[3], goal[4], goal[5]))

        if args_cli.render_scene:
            box_cfg = sim_utils.UsdFileCfg(
                               rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1,),
                                                                        emissive_color=(0.,0.,0.),
                                                                        roughness=0.5))
            floor_cfg = sim_utils.MeshCuboidCfg(
                            size=(200, 200, 0.01),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 0.01,),
                                                                        emissive_color=(0.,0.,0.),
                                                                        roughness=0.7))
            floor_cfg.func('/World/Floor',
                            floor_cfg,
                             translation=(0,0,0.005))
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1600.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        disk_cfg = sim_utils.DiskLightCfg(intensity=500.0, radius=1.0,
                                          color=(0.75, 0.75, 0.75))
        disk_cfg.func("/World/envs/env_.*/Light", disk_cfg,
                      translation = (-0.3,0.,1.),
                      orientation=(-0.952413, 0, 0.3048106, 0))
        if args_cli.render_scene:
            self._camera = self._define_sensor()


    def _reset_idx(self, env_ids: torch.Tensor | None):
        joint_pos = self._joints_inits[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
    def _get_observations(self):
        return {}
    
    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joints_inits)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs,
                           dtype=torch.float,
                           device=self.device)
    
    def _define_sensor(self) -> Camera:
        """Defines the camera sensor to add to the scene."""
        # Setup camera sensor
        # In contrast to the ray-cast camera, we spawn the prim at these locations.
        # This means the camera sensor will be attached to these prims.
        rot = quat_from_euler_xyz(torch.as_tensor([np.deg2rad(args_cli.r)]),
                                  torch.as_tensor([0.]),
                                  torch.as_tensor([np.deg2rad(args_cli.yaw)]))[0].to(torch.float)
        # rot = convert_orientation_convention(rot, origin="world", target='opengl').numpy()
        rot = rot.numpy()
        print(tuple([r for r in rot]))
        # camera_cfg = CameraCfg(
        #     prim_path="/World/envs/env_.*/Camera",
        #     update_period=0,
        #     height=960,
        #     width=1280,
        #     data_types=[
        #         "rgb",
        #     ],
        #     offset=CameraCfg.OffsetCfg(pos = (args_cli.x, args_cli.y, args_cli.z),
        #         # pos=(-1.17874, -0.64372, 1.96067),
        #                             #    rot=(0.7909,  0.3286, -0.1980, -0.4767),
        #                             rot = tuple([r for r in rot]),
        #                                convention="opengl"),
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, clipping_range=(0.1, 500)
        #     ),
        # )
        camera_cfg = CameraCfg(
            prim_path="/World/envs/Camera",
            update_period=0,
            height=1080,
            width=1920,
            data_types=[
                "rgb",
            ],
            offset=CameraCfg.OffsetCfg(pos = (args_cli.x, args_cli.y, args_cli.z),
                # pos=(-1.17874, -0.64372, 1.96067),
                                    #    rot=(0.7909,  0.3286, -0.1980, -0.4767),
                                    rot = tuple([r for r in rot]),
                                       convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=100.0,
                clipping_range=(0.1, 500),
            ),
        )
        # Create camera
        camera = Camera(cfg=camera_cfg)
        return camera

def main():
    cfg = RenderEnvCfg()
    cfg.scene.num_envs=args_cli.num_envs
    print(cfg.scene.num_envs)
    # with open(f"/input/traj/route-2024-08-31/route/{args_cli.source_file}.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    with open(f"/input/traj/{args_cli.source_file}.pkl", "rb") as fp:
        data = pickle.load(fp)
    global START_IDX
    if args_cli.start_index is None:
        START_IDX = np.nonzero(data['episode_id']==args_cli.ep_id)[0][0]
    start = START_IDX
    end = START_IDX + args_cli.num_envs
    print(start, end)
    for keys in ('joint_pos', 'box_dim',
                 'box_pose', 'name', 'obj_scale', 'obj_pose', 'goal'):
        data[keys] = data[keys][start:end]
        print(data[keys].shape)
    env = RednerEnv(cfg, data=data)

    joint_data = torch.as_tensor(data['joint_pos'],
                                 device=env.device)
    env._joints_inits = joint_data
    # env._setup_scene()
    env.reset()
    boxes = []
    target_idx = 2
    # for idx, B in enumerate(data['box_dim'][target_idx]):
    #     pose = np.eye(4)
    #     pose[:3, -1] = data['box_pose'][target_idx][idx][:3]
    #     pose[:3, :3] = matrix_from_quat(torch.as_tensor(data['box_pose'][target_idx][idx][3:7]
    #                                                     ).roll(1, -1)).numpy()
    #     print(B, pose)
    #     box = trimesh.creation.box(B,
    #                     transform=pose)
    #     boxes.append(box)

    # trimesh.Scene(boxes).show()
    # print(data['box_dim'][target_idx])
    # print(data['box_pose'][target_idx])
    # print('-'*30)
    vp_api = get_active_viewport()
    count = 0
    while simulation_app.is_running():
        count +=1
        env.step(torch.zeros(env.num_envs,
                             cfg.num_actions,
                             device=env.device))
        if count == 2 and args_cli.render_scene:
            env._camera.update(dt=env.sim.get_physics_dt())
            # capture_viewport_to_file(vp_api, f"/gate/screenshots/{count//300}.png")
            # print(env._camera.data.output['rgb'].shape)
            images = env._camera.data.output['rgb'].clone().detach().cpu().numpy()
            print(images.shape)
            with_object = 'with_object' if args_cli.render_object else 'no_object'
            # path = ensure_directory(f'/gate/images/render/{args_cli.source_file}/{with_object}')
            path = ensure_directory(f'/gate/images/render/')
            find=True
            for i in range(len(images)):
                idx = START_IDX+i
                im = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'/gate/images/render/{args_cli.source_file}_{with_object}_{idx}.png',
                            im)
            break
            # pass


if __name__ == "__main__":
    main()
    simulation_app.close()
