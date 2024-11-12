from __future__ import annotations

import torch as th
from collections.abc import Sequence
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import omni.log

import omni.isaac.lab.utils.string as string_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs.mdp.commands.humanoid_ik_command import (cart2sphere,
                                                                  sphere2cart)
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class CommandActionBase(ActionTerm):
    cfg: actions_cfg.CommandActionCfg
    _asset: Articulation
    scale: th.Tensor | float
    _offset: th.Tensor | float

    def __init__(self, cfg: actions_cfg.CommandActionCfg,
                 env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._raw_actions = th.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = th.zeros_like(self.raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        
    @property
    def raw_actions(self) -> th.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> th.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: th.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.rescale_to_limits:
            self._processed_actions = self.rescale_to_limits(self._processed_actions)
        self.map_action_to_command()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        pass

    @abstractmethod
    def map_action_to_command(self):
        pass

    @abstractmethod
    def rescale_to_limits(self, actions: th.Tensor) -> th.Tensor:
        pass

    @property
    @abstractmethod
    def command(self) -> th.Tensor:
        pass


class BodyCommandAction(CommandActionBase):
    def __init__(self, cfg: actions_cfg.BodyCommandActionCfg,
                 env: ManagerBasedEnv):
        self._action_dim = 0
        for k, v in cfg.control_type.items():
            if v == 'pos':
                self._action_dim += 3
            elif v == 'pose':
                self._action_dim += 6
            else:
                raise ValueError(f"infeasible type {v} for {k}")
        super().__init__(cfg, env)
        self.cfg = cfg

        self._body_ids, _ = self._asset.find_bodies(list(cfg.control_type.keys()),
                                                 preserve_order=True)
        target_ws: dict[str, th.Tensor] = dict()
        for k, v in cfg.control_type.items():
            if v == 'pos':
                target_w = th.zeros(env.num_envs, 3,
                                    device=self.device)
            elif v =='pose':
                target_w = th.zeros(env.num_envs, 7,
                                    device=self.device)
                target_w[..., 3] = 1
            target_ws[k] = target_w
        self._target_ws = target_ws

        if self.cfg.rescale_to_limits:
            assert (set(cfg.ranges.keys()) == set(cfg.control_type.keys()))
            bounds = list()
            for k in cfg.control_type.keys():
                bound = cfg.ranges[k]
                dx = bound.dx
                dr = bound.dr
                assert (self._filter_cfg(dx) and self._filter_cfg(dr))

                dx = th.as_tensor(dx, device=self.device)
                if len(dx.shape) == 1:
                    dx = dx[None].repeat(3, 1)
                if cfg.control_type[k] == 'pose':
                    dr = th.as_tensor(dr, device=self.device)
                    if len(dr.shape) == 1:
                        dr = dr[None].repeat(3, 1)
                    dx = th.cat([dx, dr], dim=0) # (6, 2)
                bounds.append(dx)
            self._bounds = th.cat(bounds, dim=0)[None].repeat(self._env.num_envs,
                                                              1, 1) # (n_env, n_act, 2)
            
        for k, v in self.cfg.frame_func_param.items():
            if isinstance(v, SceneEntityCfg):
                try:
                    v.resolve(self._env.scene)
                except ValueError as e:
                    raise ValueError(f"Error while parsing '{k}'. {e}")
                
        self._base_frame_pose_w = th.zeros(self._env.num_envs,
                                      7,
                                      device=self.device)

    def _filter_cfg(self, item) -> bool:
        # Check if the item is a tuple of two floats
        if (isinstance(item, tuple) and len(item) == 2
            and all(isinstance(i, float) for i in item)):
            return True
        # Check if the item is a list of exactly three tuples of two floats each
        elif (isinstance(item, list)
              and len(item) == 3):
            return all(isinstance(t, tuple) and len(t) == 2
                       and all(isinstance(i, float) for i in t)
                       for t in item)
        return False

    @property
    def action_dim(self) -> int:    
        return self._action_dim
    
    def rescale_to_limits(self, actions: th.Tensor) -> th.Tensor:
        actions = actions.clamp(-1.0, 1.0)

        actions = math_utils.unscale_transform(
            actions,
            self._bounds[..., 0],
            self._bounds[..., 1]
        )
        return actions

    
    def map_action_to_command(self):
        term_idx = 0
        for (k,v), idx in zip(self._target_ws.items(),
                          self._body_ids):
            body_pos_w = self._asset.data.body_pos_w[..., idx, :]
            if v.shape[-1] == 3:
                self._target_ws[k][:] = (body_pos_w
                                         + self._processed_actions[...,
                                                                   term_idx:term_idx+3])
                term_idx += 3
            elif v.shape[-1] == 7:
                body_quat_w = self._asset.data.body_quat_w[..., idx, :]
                self._target_ws[k][:] = math_utils.apply_delta_pose(
                    body_pos_w, body_quat_w,
                    self._processed_actions[...,term_idx:term_idx+7]
                )
                term_idx += 7

    def _update_base_frame(self):
        pos_w, quat_w = self.cfg.frame_func(self._env,
                            **self.cfg.frame_func_param)
        self._base_frame_pose_w[..., :3] = pos_w
        self._base_frame_pose_w[..., 3:7] = quat_w

    @property
    def command(self) -> th.Tensor:
        self._update_base_frame()
        errors = []
        for (k,v), idx in zip(self._target_ws.items(),
                          self._body_ids):
            body_pos_w = self._asset.data.body_pos_w[..., idx, :]
            body_pos_b, body_quat_b = math_utils.subtract_frame_transforms(
                self._base_frame_pose_w[..., :3],
                self._base_frame_pose_w[..., 3:7],
                body_pos_w[..., :3],
                body_pos_w[..., 3:7])
            goal_pos_b, goal_quat_b = math_utils.subtract_frame_transforms(
                self._base_frame_pose_w[..., :3],
                self._base_frame_pose_w[..., 3:7],
                v[:, :3],
                v[:, 3:7],
            )
            err_pos, err_quat = math_utils.compute_pose_error(
                body_pos_b,
                body_quat_b,
                goal_pos_b,
                goal_quat_b,
            )
            errors.append(err_pos)
            errors.append(err_quat)

        return th.cat(errors, dim=-1)