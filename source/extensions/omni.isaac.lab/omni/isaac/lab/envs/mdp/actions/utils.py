from typing import TYPE_CHECKING, Literal
import torch as th

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
import omni.isaac.lab.utils.math as math_utils

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

def get_base_frame(env,
                   asset_cfg: SceneEntityCfg,
                   base: Literal['root', 'body'],
                   op: Literal['mean', 'yaw_proj']|None = None
                   ):
    asset: RigidObject = env.scene[asset_cfg.name]
    if base == 'body':
        pos_w = asset.data.body_pos_w[..., asset_cfg.body_ids, :].clone()
        quat_w = asset.data.body_pos_w[..., asset_cfg.body_ids, :].clone()
        if op is None:
            pos_w = pos_w.squeeze()
            quat_w = quat_w.squeeze()
        elif op == 'mean':
            pos_w = pos_w.mean(dim=-2)
            quat_w = math_utils.slerp(quat_w[..., 0, :],
                                      quat_w[..., 1, :],
                                      steps=th.full((env.num_envs,),
                                                    0.5,
                                                    device=env.device))
        elif op == 'yaw_proj':
        # same as z-inv
            pos_w = pos_w.mean(dim=-2)
            pos_w[..., 2] = 0.0
            _, _, left_euler = math_utils.euler_xyz_from_quat(
               quat_w[..., 0, :]
            )
            _, _, right_euler = math_utils.euler_xyz_from_quat(
                quat_w[..., 1, :]
            )
            quat_w = math_utils.quat_from_euler_xyz(
                th.zeros_like(left_euler),
                th.zeros_like(left_euler),
                0.5*(left_euler + right_euler)
            )

    elif base == 'root':
        pos_w = asset.data.root_pos_w
        quat_w = asset.data.root_quat_w
        if op == 'yaw_proj':
            pos_w[..., 2] = 0.0
            _, _, euler = math_utils.euler_xyz_from_quat(
                quat_w
            )
            quat_w = math_utils.quat_from_euler_xyz(
                th.zeros_like(euler),
                th.zeros_like(euler),
                euler
            )

    return pos_w, quat_w