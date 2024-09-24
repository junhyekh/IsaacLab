

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to spawn.")
parser.add_argument("--use_tile", action="store_true")
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
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import sample_uniform, matrix_from_quat, euler_xyz_from_quat
from omni.isaac.lab.sim.converters import UrdfConverterCfg, UrdfConverter

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
    episode_length_s = 5.3333  # 500 timesteps
    decimation = 2
    num_actions = 9
    num_observations = 23
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 30,
        device="cpu",
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=1.5, replicate_physics=True)

    # robot
    # robot = ArticulationCfg(
    #     prim_path="/World/envs/env_.*/Robot",
    #     spawn=sim_utils.UsdFileCfg(
    #          usd_path="/input/robot/franka_description/robots/usd2/fr3_ability_hand.usd",
    #         activate_contact_sensors=False,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True,
    #             max_depenetration_velocity=5.0,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
    #         ),
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         joint_pos={
    #             "fr3_joint1": 1.157,
    #             "fr3_joint2": -1.066,
    #             "fr3_joint3": -0.155,
    #             "fr3_joint4": -2.239,
    #             "fr3_joint5": -1.841,
    #             "fr3_joint6": 1.003,
    #             "fr3_joint7": 0.469,
    #         },
    #         pos=(-0.3, 0.0, 0.0),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    #     actuators={
    #         "fr3_shoulder": ImplicitActuatorCfg(
    #             joint_names_expr=["fr3_joint[1-4]"],
    #             effort_limit=87.0,
    #             velocity_limit=2.62,
    #             stiffness=80.0,
    #             damping=4.0,
    #         ),
    #         "fr3_forearm": ImplicitActuatorCfg(
    #             joint_names_expr=["fr3_joint[5-7]"],
    #             effort_limit=12.0,
    #             velocity_limit=4.18,
    #             stiffness=80.0,
    #             damping=4.0,
    #         ),
    #         "thumb_joint1": ImplicitActuatorCfg(
    #             joint_names_expr=["thumb_q1"],
    #             effort_limit=1.2,
    #             velocity_limit=4,
    #             stiffness=1.0,
    #             damping=0.4,
    #         ),
    #         "thumb_joint2": ImplicitActuatorCfg(
    #             joint_names_expr=["thumb_q2"],
    #             effort_limit=6,
    #             velocity_limit=4,
    #             stiffness=1.0,
    #             damping=0.4,
    #         ),
    #         "finger_joints": ImplicitActuatorCfg(
    #             joint_names_expr=["middle_q1", "ring_q1", "index_q1", "pinky_q1"],
    #             effort_limit=6.0,
    #             velocity_limit=8,
    #             stiffness=1.0,
    #             damping=0.4,
    #         )
    #     },
    # )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
             usd_path="/input/robot/franka_description/robots/usd6/hand_flip.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # joint_pos={
            #     "fr3_joint1": 1.157,
            #     "fr3_joint2": -1.066,
            #     "fr3_joint3": -0.155,
            #     "fr3_joint4": -2.239,
            #     "fr3_joint5": -1.841,
            #     "fr3_joint6": 1.003,
            #     "fr3_joint7": 0.469,
            # },
            pos=(-0.3, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            # "fr3_shoulder": ImplicitActuatorCfg(
            #     joint_names_expr=["fr3_joint[1-4]"],
            #     effort_limit=87.0,
            #     velocity_limit=2.62,
            #     stiffness=80.0,
            #     damping=4.0,
            # ),
            # "fr3_forearm": ImplicitActuatorCfg(
            #     joint_names_expr=["fr3_joint[5-7]"],
            #     effort_limit=12.0,
            #     velocity_limit=4.18,
            #     stiffness=80.0,
            #     damping=4.0,
            # ),
            "thumb_joint1": ImplicitActuatorCfg(
                joint_names_expr=["thumb_q1"],
                effort_limit=1.2,
                velocity_limit=4,
                stiffness=1.0,
                damping=0.4,
            ),
            "thumb_joint2": ImplicitActuatorCfg(
                joint_names_expr=["thumb_q2"],
                effort_limit=6,
                velocity_limit=4,
                stiffness=1.0,
                damping=0.1,
            ),
            "finger_joints": ImplicitActuatorCfg(
                joint_names_expr=["middle_q1", "ring_q1", "index_q1", "pinky_q1"],
                effort_limit=6.0,
                velocity_limit=8,
                stiffness=1.0,
                damping=0.1,
            ),
            # "mimic_joints": ImplicitActuatorCfg(
            #     joint_names_expr=["middle_q2", "ring_q2", "index_q2", "pinky_q2"],
            #     effort_limit=6.0,
            #     velocity_limit=8,
            #     stiffness=1.0,
            #     damping=0.1,
            # )
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
    )
    
    use_tiled_camera: bool = False

class RednerEnv(DirectRLEnv):
    cfg: RenderEnvCfg

    def __init__(self, cfg: RenderEnvCfg,
                 render_mode: str | None = None, 
                 data = None,
                 **kwargs):
        self._data = data
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
        self._targets_joints = torch.zeros(self.num_envs,
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
            cfg_cuboid = sim_utils.MeshCuboidCfg(
                            size=(box_dim[j,1], box_dim[j,0], box_dim[j,2]),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                            )
            cfg_cuboid.func(f'{prim_path}/box_{j}',
                            cfg_cuboid,
                            translation=(box_poses[j,1],
                                        box_poses[j,0],
                                        box_poses[j,2]),
                            orientation=(
                                box_poses[j,6], 
                                box_poses[j,3],
                                box_poses[j,4],
                                box_poses[j,5]  
                            ))
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
       
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # create cabinet
        origins = self.scene.env_origins.detach().clone().cpu().numpy()
        print(self.scene.env_prim_paths)
        
        # for i in range(1, self.scene.num_envs):
        #     if False:
        #         self._create_boxes(self._data['box_dim'][i],
        #                         self._data['box_pose'][i],
        #                             self.scene.env_prim_paths[i],
        #                             origins[i], i)
        #     else:
        #         dest_path = F'/gate/boxes/0000/env_{i}/box.usd'
        #         if not Path(dest_path).exists():
        #             urdf_file = F'/gate/boxes/dump/env_{i}.urdf'
        #             xyzs = []
        #             rpys = []
        #             dims = []
        #             for i_part, part in enumerate(self._data['box_dim'][i]):
        #                 xyz = _list2str(self._data['box_pose'][i][i_part][:3])
        #                 euler = euler_xyz_from_quat(
        #                     torch.as_tensor(self._data['box_pose'][i][i_part][3:7]).roll(1,-1)[None])
        #                 euler = [d.squeeze(0) for d in euler]
        #                 rpy = _list2str(euler)
        #                 xyzs.append(xyz)
        #                 rpys.append(rpy)
        #                 dims.append(list(part))
        #             urdf_text: str = '''
        #             <robot name="robot">
        #             {box}
        #             </robot>
        #             '''.format(
        #             box=multi_box_link(
        #                 F'base_link',
        #                 dims,
        #                 xyzs,
        #                 rpys,
        #                 density=0.0
        #                 )
        #             )
        #             urdf_file = F'/gate/boxes/0000/env_{i}/box.urdf'
        #             os.makedirs(F'/gate/boxes/0000/env_{i}/',
        #                         exist_ok=True)
        #             with open(str(urdf_file), 'w') as fp:
        #                 fp.write(urdf_text)

        #             urdf_converter_cfg = UrdfConverterCfg(
        #                 asset_path=urdf_file,
        #                 usd_dir=os.path.dirname(dest_path),
        #                 usd_file_name=os.path.basename(dest_path),
        #                 fix_base=True,
        #                 convex_decompose_mesh=True,
        #                 merge_fixed_joints=True,
        #                 force_usd_conversion=True,
        #                 make_instanceable=False,
        #             )
        #             urdf_converter = UrdfConverter(urdf_converter_cfg)
        #         box_cfg = sim_utils.UsdFileCfg(usd_path=dest_path,
        #                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #                     #    collision_props=sim_utils.CollisionPropertiesCfg()
        #                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0,),
        #                                                                 emissive_color=(0.3,0.3,0.3),
        #                                                                 metallic=0.1,
        #                                                                 roughness=0.2))
        #         box_cfg.func(f"{self.scene.env_prim_paths[i]}/box", box_cfg, translation=(0, 0.0, 0.0))

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._mimic_sources = None
        self._mimic_targets = None

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if self._mimic_sources is None:
            mimic_sources = []
            mimic_targets = []
            ac = self._robot.actuators['finger_joints']
            print(ac.joint_names, ac.joint_indices)
            for idx in ac.joint_indices:
                mimic_sources.append(idx)
            # ac = self._robot.actuators['mimic_joints']
            # print(ac.joint_names, ac.joint_indices)
            # for idx in ac.joint_indices:
            #     mimic_targets.append(idx)
            mimic_targets, _ =self._robot.find_joints([s.replace('1','2')
                                           for s in ac.joint_names], preserve_order=True)

            self._mimic_sources = torch.as_tensor(mimic_sources, 
                                                dtype=torch.long,
                                                device=self.device)
            self._mimic_targets = torch.as_tensor(mimic_targets, 
                                                dtype=torch.long,
                                                device=self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, 
                                   dtype=torch.long,
                                   device=self.device)
        joint_pos = self._joints_inits[env_ids]
        # joint_pos[:, :4] = 0.
        # joint_pos[:,5:-1] = joint_pos[:,:4]*1.05851325 + 0.72349796
        joint_target = self._joints_inits.clone()
        joint_pos[:,
                      self._mimic_targets] = (joint_pos[:,
                                                self._mimic_sources] * 1.05851325
                                                # + 0.72349796)
                                                +0.0)
        joint_vel = torch.zeros_like(joint_pos)
        joint_target[:,
                      self._mimic_targets] = (joint_target[:,
                                                self._mimic_sources] * 1.05851325
                                                # + 0.72349796)
                                                +0.0)
        print(joint_pos[0], self._mimic_sources)
        print("!"*30)
        self._robot.set_joint_position_target(joint_target[env_ids[..., None],
                                                           self._mimic_sources],
                                              self._mimic_sources,
                                            # None,
                                              env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
    def _get_observations(self):
        return {}
    
    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self):
        joint_target = self._joints_inits.clone()
        joint_target[:,
                      self._mimic_sources] = self.episode_length_buf[..., None] * 0.03
        # print(joint_target[0, self._mimic_sources])
        divv = (self._robot.data.joint_pos[0][self._mimic_targets]/
              self._robot.data.joint_pos[0][self._mimic_sources])
        print(divv.min(), divv.max(),self._robot.data.joint_pos[0][self._mimic_targets],
              self._robot.data.joint_pos[0][self._mimic_sources])
        joint_target[:,
                      self._mimic_targets] = (joint_target[:,
                                                self._mimic_sources] * 1.05851325
                                                # + 0.72349796)
                                                +0.0)
        self._robot.set_joint_position_target(joint_target[:,
                                                self._mimic_sources],
                                              self._mimic_sources)
        # self._robot.set_joint_position_target(joint_target[:,
        #                                         self._mimic_sources],
        #                                       self._mimic_sources)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(truncated)
        return terminated, truncated
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs,
                           dtype=torch.float,
                           device=self.device)

def main():
    cfg = RenderEnvCfg()
    cfg.scene.num_envs=args_cli.num_envs
    print(cfg.scene.num_envs)
    with open("/input/traj/0000.pkl", "rb") as fp:
        data = pickle.load(fp)
    env = RednerEnv(cfg, data=data)
    # joint_data = torch.as_tensor(data['joint_pos'][:args_cli.num_envs],
    #                              device=env.device)
    joint_data = env._robot._data.joint_limits
    # print(joint_data)
    
    # joint_targets = torch.as_tensor(joint_targets,
    #                                 dtype=torch.long,
    #                                 device=env.device,
    #                                 ).sort()[0]
    targets_joints = torch.arange(env._robot.num_joints, dtype=torch.long,
                                 device=env.device)
    print(joint_data[0], joint_data.mean(-1)[0])
    # joint_data = torch.take_along_dim(joint_data.mean(-1)[0],
    #                                   joint_targets)
    joint_data = joint_data.mean(-1)
    joint_data[:] = 0.

    # joint_data[:,7:] = 0
    # joint_data[:] = 0.
    # joint_data[:,7:11] *= 0
    # joint_data[:,12:-1] = joint_data[:,7:11]*1.05851325 + 0.72349796
    # joint_data[:,11] = -1.
    # joint_data[:,-1] = 1.
    # joint_data[:, -2] = -1.
    print(joint_data, targets_joints)
    env._joints_inits = joint_data
    env._targets_joints = targets_joints
    print(env._robot.num_joints, env._robot.fixed_tendon_names)
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

    while simulation_app.is_running():
        env.step(torch.zeros(env.num_envs,
                             cfg.num_actions,
                             device=env.device))
        # print(env._robot.data.joint_pos[0,:4],
        #       env._robot.data.joint_pos[0, 6:])

if __name__ == "__main__":
    main()
    simulation_app.close()
