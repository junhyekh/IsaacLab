

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to spawn.")
parser.add_argument("--max_object", type=int,
                    default=7,
                    help="# of max object per each env")
parser.add_argument("--load_scene", type=str)
parser.add_argument("--save_scene", type=str)
parser.add_argument("--front_wall", action="store_true",
                    default=False, help="Need front wall?")
parser.add_argument("--load_robot", action="store_true",
                    default=False, help="Need robot?")
parser.add_argument("--render_scene", action="store_true",
                    default=False, help="Need rendering?")

# append AppLauncher cli args 
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

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
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import (sample_uniform,
                                       matrix_from_quat,
                                       convert_quat,
                                       euler_xyz_from_quat,
                                       quat_from_euler_xyz,
                                       random_orientation)
from omni.isaac.lab.sensors.camera.utils import convert_orientation_convention
from omni.isaac.lab.sim.converters import UrdfConverterCfg, UrdfConverter
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
import omni.physics.tensors.impl.api as physx


from custom_env.scene.dgn_object_set import DGNObjectSet
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

import cv2
from custom_env.util.path import ensure_directory
from custom_env.scene.util import _list2str, multi_box_link

@configclass
class SceneGeneraterCfg(DirectRLEnvCfg):
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
        gravity=(0,0,-3), # FIXME hardcoded gravity to reduce impulse during dropping
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

    def __post_init__(self):
        if args_cli.load_scene:
            # restore gravity 
            self.sim.gravity = (0,0,-9.81)

class SceneGenerater(DirectRLEnv):
    cfg: SceneGeneraterCfg

    def __init__(self, cfg: SceneGeneraterCfg,
                 render_mode: str | None = None, 
                 **kwargs):
        self._data = {}
        if args_cli.load_scene is not None:
            with open(args_cli.load_scene, "rb") as fp:
                data = pickle.load(fp)
            self._data = data

        self.dgn = None
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
                    
    def _create_bin(self, dest_path,
                    prim_path,
                    w=0.3, l=0.3, h=0.3,
                    t=0.01
                    ):
        box_dim = [
            [l,w,t], # base
            [t,w,h], # back
            [l,t,h], # left,
            [l,t,h], # right,
        ]
        box_pose = [
            [0,0,-t/2], # base
            [l/2+t/2,0,h/2], # back
            [0,w/2+t/2,h/2], # left,
            [0,-w/2-t/2,h/2], # right,
        ]
        if args_cli.load_scene:
            # add ceil
            box_dim.append([l,w,t])
            box_pose.append([0,0,h+t/2])
        if args_cli.front_wall:
            # add front wall
            box_dim.append([t,w,3*h/4])
            box_pose.append([-l/2-t/2,0,3*h/8])
        if True:
            # RigidBodyMaterialCfg
            for j in range(len(box_dim)):
                bd = box_dim[j]
                bp = box_pose[j]
                cfg_cuboid = sim_utils.MeshCuboidCfg(
                                size=bd,
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                collision_props=sim_utils.CollisionPropertiesCfg(),
                                physics_material=sim_utils.RigidBodyMaterialCfg(),
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                                # visual_material=vis_mat
                                )
                cfg_cuboid.func(f'{prim_path}/box_{j}',
                                cfg_cuboid,
                                translation=(bp[0],
                                            bp[1],
                                            bp[2]+0.2),
                                orientation=(1, 0, 0, 0))
        else:
            if not Path(dest_path).exists():
                urdf_file = F'/gate/boxes/container/env_{i}.urdf'
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
            box_cfg.func(f"{prim_path}/box", box_cfg, translation=(0, 0.0, 0.0))

    def _create_objects(self, prim_path):
        if not self._data:
            obj_keys = np.random.choice(list(self.dgn.keys()),
                                             (args_cli.max_object,),
                                             replace=False)
            radius = [self.dgn.radius(k) for k in obj_keys]
            # FIXME: hardcorded
            min_s = 0.045
            max_s = 0.06
            scales = np.random.uniform(min_s, max_s,
                                      size=(args_cli.max_object,))
            rel_scale = [s/r for (r,s) in zip(radius, scales)]
        else:
            obj_keys = list(self._data.keys())
            scales = [obj['scale'] for obj in self._data.values()]
            radius = [self.dgn.radius(k) for k in obj_keys]
            rel_scale = [s/r for (r,s) in zip(radius, scales)]
            # rel_scale = [obj['rel_scale'] for obj in self._data.values()]
            init_poses = np.stack([obj['pose'] for obj in self._data.values()],
                                  axis=0)
            self._init_poses = torch.as_tensor(init_poses,
                                               dtype=torch.float,
                                               device=self.device)
        self._obj_scales = scales
        self._obj_keys = obj_keys
        objs = []
        print(rel_scale, scales, radius)
        for ii, (k, s) in enumerate(zip(obj_keys, rel_scale)):
            obj_usd = self.dgn.usd(k)
            if not Path(obj_usd).exists():
                urdf_file = self.dgn.urdf(k)
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
            
            # obj_cfg = sim_utils.UsdFileCfg(usd_path=obj_usd,
            #                     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            #                     # collision_props=sim_utils.CollisionPropertiesCfg(),
            #                     physics_material=sim_utils.RigidBodyMaterialCfg(
            #                         static_friction=0.8,
            #                         dynamic_friction=0.6,
            #                         restitution=0.0
            #                     ),
            #                     scale=(s, s, s),
            #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(240/255., 110/255., 110/255.,)))
            # obj_cfg.func(f"{prim_path}/object_{ii}", obj_cfg)
            obj = RigidObject(
                cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object_{ii}",
                    spawn= sim_utils.UsdFileCfg(usd_path=obj_usd,
                                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    max_depenetration_velocity=1.0,
                                    max_contact_impulse=1.0,
                                ),
                                # collision_props=sim_utils.CollisionPropertiesCfg(),
                                # physics_material=sim_utils.RigidBodyMaterialCfg(
                                #     static_friction=0.8,
                                #     dynamic_friction=0.6,
                                #     restitution=0.0
                                # ),
                                scale=(s, s, s),
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(240/255., 110/255., 110/255.,))
                    )))
            objs.append(obj)
        self.__root_physx_view = None
        return objs

            
    def _setup_scene(self):
        if self.dgn is None:
            self.dgn = DGNObjectSet(DGNObjectSet.Config(data_path="/input/DGN/meta-v8"))

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        origins = self._terrain.env_origins.detach().clone().cpu().numpy()

        for i in range(0, self.scene.num_envs):
            prim_utils.create_prim(self.scene.env_prim_paths[i],
                                   "Xform", translation=origins[i])
        if args_cli.load_robot:
            self._robot = Articulation(self.cfg.robot)
            self.scene.articulations["robot"] = self._robot

        # create cabinet
        for i in range(0, self.scene.num_envs):
            dest_path = F'/gate/boxes/container/env_{i}/box.usd'
            self._create_bin(dest_path,
                             self.scene.env_prim_paths[i])

        # spawn object
        # for i in range(0, self.scene.num_envs):
        self.objs = self._create_objects(self.scene.env_prim_paths[i])
        # self.scene.rigid_objects["object"] = self.objs
        

        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
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
        if env_ids is None:
            env_ids = torch.arange(self.num_envs,
                                   device=self.device)
            
        if self.__root_physx_view is None:
            self.__physics_sim_view = physx.create_simulation_view("torch")
            self.__physics_sim_view.set_subspace_roots("/")
            objs_prim_path = self._get_object_views(env_id=0)
            self.__root_physx_view = self.__physics_sim_view.create_rigid_body_view(objs_prim_path)
        # save the config
        obj_pose = self.__root_physx_view.get_transforms().clone()
        velocity = self.__root_physx_view.get_velocities()
        if args_cli.save_scene is not None:
            # check whether objects are in box
            # FIXME hardcoded 
            self.extras['saved'] = False
            contained = torch.logical_and(obj_pose[env_ids, 0:2] <0.3,
                                          obj_pose[env_ids, 0:2] >-0.3).all()
            contained &= (obj_pose[env_ids, 2] >0.2).all()
            if contained:
                p = ensure_directory(args_cli.save_scene)
                obj_pose[..., 3:7] = convert_quat(obj_pose[..., 3:7], to="wxyz")
                for idx, obj_key in enumerate(self._obj_keys):
                    data = {}
                    data['scale'] = self._obj_scales[idx]
                    data['pose'] = obj_pose[idx].detach().cpu().numpy()
                    self._data[obj_key] = data

                with open(p/'config.pkl', 'wb') as fp:
                    pickle.dump(self._data, fp)
                self.extras['saved'] = True
        
        super()._reset_idx(env_ids)
        if args_cli.load_robot:
            joint_pos = self._joints_inits[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        if not self._data:
            qs = random_orientation(len(env_ids) * len(self.objs),
                                    self.device).view(len(self.objs),
                                                    len(env_ids),
                                                    -1)
            z = 0.2
            for idx, obj in enumerate(self.objs):
                object_default_state = obj.data.default_root_state.clone()[env_ids]
                #FIXME currently box shape is hardcoded
                s = self._obj_scales[idx]
                xy = sample_uniform(-0.15+s, 0.15-s, (len(env_ids), 2),
                                device=self.device)
                #FIXME currently hard code on margin (0.02)
                z = z + 0.02 + s

                # xy and z
                object_default_state[:, :2] = xy
                object_default_state[:, 2] = z

                # ori
                object_default_state[:, 3:7] = qs[idx]

                # zero vel
                object_default_state[:, 7:] = torch.zeros_like(obj.data.default_root_state[env_ids, 7:])
                obj.write_root_state_to_sim(object_default_state, env_ids)

                # to prevent penetration
                z += s
        else:
            for idx, obj in enumerate(self.objs):
                object_default_state = obj.data.default_root_state.clone()[env_ids]
                # xy and z
                object_default_state[:, :7] = self._init_poses[idx]

                # zero vel
                object_default_state[:, 7:] = torch.zeros_like(obj.data.default_root_state[env_ids, 7:])
                obj.write_root_state_to_sim(object_default_state, env_ids)
    def _get_object_views(self, env_id: int | None = None,
                          obj_id: int | None = None):
        env_rep = f'env_{env_id}' if env_id is not None else 'env_*'
        obj_rep = f'Object_{obj_id}' if obj_id is not None else 'Object_*'

        return f'/World/envs/{env_rep}/{obj_rep}/base_link'
    
    def _get_observations(self):
        return {}
    
    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        if args_cli.load_robot:
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
    cfg = SceneGeneraterCfg()
    cfg.scene.num_envs=args_cli.num_envs
    print(cfg.scene.num_envs)
    # with open(f"/input/traj/route-2024-08-31/route/{args_cli.source_file}.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    # with open(f"/input/traj/{args_cli.source_file}.pkl", "rb") as fp:
    #     data = pickle.load(fp)
    env = SceneGenerater(cfg)

    env.reset()
    vp_api = get_active_viewport()
    count = 0
    while simulation_app.is_running():
        count +=1
        _,_,_,timeout, extras = env.step(torch.zeros(env.num_envs,
                             cfg.num_actions,
                             device=env.device))
        if args_cli.save_scene and extras['saved']:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
