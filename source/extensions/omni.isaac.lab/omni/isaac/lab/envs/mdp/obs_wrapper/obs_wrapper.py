from __future__ import annotations

import inspect
import torch as th
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Sequence, Literal
from dataclasses import MISSING
"""
History.
"""

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import (ObservationTermCfg,
                                     ManagerTermBase,
                                     ManagerTermBaseCfg,
                                     SceneEntityCfg)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

@configclass
class HistoryObsCfg(ObservationTermCfg):
    history_len: int = MISSING
    num_input: int | None = None
    reset_type: Literal['zero', 'prev'] = 'zero'

    flatten: bool = True

    #target function to be wrapped
    func_target: Callable[..., th.Tensor] = MISSING
    func_target_cfg: ObservationTermCfg|None =  None


class HistoryObs(ManagerTermBase):
    """
    Wrapper that support input with arbitrary historical length
    """
    def __init__(self, cfg: HistoryObsCfg, env: ManagerBasedEnv):
        assert(cfg.history_len > 0)
        super().__init__(cfg, env)

        self._func = cfg.func_target
        # init if it is a class
        if isinstance(self._func,
                      ManagerTermBase):
            assert ((cfg.func_target_cfg is not None) 
                    and isinstance(cfg.func_target_cfg, ObservationTermCfg))
            self._func = self._func(cfg=cfg.func_target_cfg,
                                    env=self._env)
        if cfg.num_input is None:
            self._data = None
        else:
            self._data = th.zeros(self._env.num_envs,
                                  cfg.history_len,
                                  cfg.num_input,
                                  device=self.device)
            
        if cfg.reset_type == 'prev':
            self._is_reset = th.zeros(self._env.num_envs,
                                      dtype=th.bool,
                                      device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if isinstance(self._func,
                      ManagerTermBase):
            self._func.reset(env_ids)
        if self._data is not None:
            if self.cfg.reset_type =='zero':
                self._data[env_ids] = 0.0
        if self.cfg.reset_type =='prev':
            self._is_reset[env_ids] = True

    def __call__(self,
                  env: ManagerBasedRLEnv, 
                 *args,
                 **kwargs) -> Any:
        
        val = self._func(env=env, **kwargs)
        if self._data is None:
            self._data = th.zeros(
                self._env.num_envs,
                self.cfg.history_len,
                *val.shape[1:],
                device=self.device
            )
        self._data[:, 1:] = self._data[:, :-1].clone()
        self._data[:, 0] = val
        if self.cfg.reset_type =='prev':
            prev_done = th.nonzero(self._is_reset)[..., None]
            self._data[prev_done, 1:] = val[prev_done, None]
            self._is_reset[:] = False
        if self.cfg.flatten:
            return self._data.reshape(self._env.num_envs, -1)
        else:
            return self._data