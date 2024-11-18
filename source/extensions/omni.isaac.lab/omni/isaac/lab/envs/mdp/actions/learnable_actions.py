from __future__ import annotations

import gym
import torch as th
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING
import yaml
import pickle
from pathlib import Path

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from . import command_actions as cmd_action

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    

    from . import actions_cfg


class DummyEnvWrapper(VecEnv):
    """
        Dummy wrapper for setup
    """
    def __init__(self, env: ManagerBasedRLEnv,
                 obs_group_name: str = 'policy',
                 num_actions: int = 0,
                 num_obs: int = 0):
        self.env = env
        # store information required by wrapper
        self.num_envs = self.env.num_envs
        self.device = env.device
        
        self.num_actions = num_actions
        self.num_obs = num_obs
        
        self.num_privileged_obs = 0

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))
 
    def get_observations(self) -> tuple[th.Tensor, dict]:
        dummy_obs =  th.zeros(self.num_envs,
                        self.num_obs,
                        device=self.device)
        obs_dict = dict(policy=dummy_obs)
        return obs_dict["policy"], {"observations": obs_dict}
    
    def reset(self):
        pass
    def step(self, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, dict]:
        pass

class NetController(ActionTerm):
    """
    Base class for control based on pre-trained network
    """
    cfg: actions_cfg.NetControllerCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.NetControllerCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._raw_actions = th.zeros(self.num_envs, device=self.device)
        self._processed_actions = th.zeros_like(self.raw_actions)

        self._model = None
        self._runner = None

        self._prepare_terms()
        self._setup_controller()

        self._actions = th.zeros(env.num_envs,
                                 self._num_actions,
                                 device=env.device)
        
        print(self._terms,
              self._num_actions)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> th.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> th.Tensor:
        return self._processed_actions

    def _prepare_terms(self):
        """Prepares a list of action terms."""
        # parse action terms from the config
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()
        self._num_actions: int = 0

        # check if config is dict already
        if isinstance(self.cfg.actions, dict):
            cfg_items = self.cfg.actions.items()
        else:
            cfg_items = self.cfg.actions.__dict__.items()
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._term_names.append(term_name)
            self._terms[term_name] = term
            self._num_actions += term.action_dim

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._runner.alg.actor_critic.reset(env_ids)

    def _setup_controller(self):
        if self.cfg.loading_type == 'rsl_rl':
            agent_param_cfg = Path(self.cfg.ckpt).parent / 'params' / 'agent.yaml'    
            with open(agent_param_cfg, encoding="utf-8") as fp:
                param = yaml.full_load(fp)
        
            # agent_cfg = RslRlOnPolicyRunnerCfg()
            # agent_cfg.from_dict(param)
            agent_cfg = param
            # # FIXME hardcoded update should be changed
            # if 'class_name' not in agent_cfg['policy']:
            #     agent_cfg['policy']['class_name'] = "ActorCritic"
            self._runner: OnPolicyRunner = OnPolicyRunner(DummyEnvWrapper(self._env,
                                                                          self.cfg.obs_group,
                                                                          self._num_actions,
                                                                          self.cfg.obs_dim),
                                          agent_cfg,
                                          log_dir=None, 
                                          device=self.device)
            self._runner.load(self.cfg.ckpt)
            self._model: ActorCritic = self._runner.get_inference_policy(
                device=self.device)

    def process_actions(self, actions: th.Tensor):
        obs = self._env.observation_manager.compute_group(self.cfg.obs_group)
        if self.cfg.has_last_action_as_obs:
            obs['actions'] = self._actions
        obs = th.cat(list(obs.values()), dim=-1)
        with th.inference_mode():
            self._actions[:] = self._model(obs)
        idx = 0
        for term in self._terms.values():
            term_actions = self._actions[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def apply_actions(self) -> None:
        for term in self._terms.values():
            term.apply_actions()



class NetControllerWCommand(NetController):
    """
    pre-trained controller with command override

    Note: Obs group for this controller should not be flattened 
    to make command override easier
    """
    cfg: actions_cfg.NetControllerWCommandCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.NetControllerWCommandCfg,
                 env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prepare_command_terms()
        self._action_dim = None
        self._raw_actions = th.zeros(self.num_envs, self.action_dim,
                                     device=self.device)
        self._processed_actions = th.zeros_like(self.raw_actions)

    @property
    def action_dim(self) -> int:
        if self._action_dim is None:
           self._action_dim = 0
           for v in self._command_terms.values():
               self._action_dim += v.action_dim
        return self._action_dim
        
    def process_actions(self, actions: th.Tensor):
        obs = self._env.observation_manager.compute_group(self.cfg.obs_group)
        commands = dict()
        term_idx = 0
        for k,v in self._command_terms.items():
            d = v.action_dim
            v.process_actions(actions[..., term_idx:term_idx+d])
            command = v.command
            commands[k] = command
            term_idx += d
        obs.update(commands)
        obs = th.cat(list(obs.values()), dim=-1)
        
        with th.inference_mode():
            act = self._model(obs)
        idx = 0
        for term in self._terms.values():
            term_actions = act[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    def _prepare_command_terms(self):
        """Prepares a list of action terms."""
        # parse action terms from the config
        self._command_term_names: list[str] = list()
        self._command_terms: dict[str, cmd_action.CommandActionBase] = dict()

        # check if config is dict already
        if isinstance(self.cfg.command_actions, dict):
            cfg_items = self.cfg.command_actions.items()
        else:
            cfg_items = self.cfg.command_actions.__dict__.items()
        for term_name, term_cfg in cfg_items:
            # check if term config is None
            if term_cfg is None:
                continue
            # check valid type
            if not isinstance(term_cfg, actions_cfg.CommandActionCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, cmd_action.CommandActionBase):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type ActionType.")
            # add term name and parameters
            self._command_term_names.append(term_name)
            self._command_terms[term_name] = term