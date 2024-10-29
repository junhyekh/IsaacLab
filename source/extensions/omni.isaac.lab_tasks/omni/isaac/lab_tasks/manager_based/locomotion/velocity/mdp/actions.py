from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING
from collections.abc import Sequence
import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class PassiveIKAction(DifferentialInverseKinematicsAction):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.PassiveIKActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: th.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""

    """
    Properties.
    """
    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        self._raw_actions = th.zeros(self.num_envs, device=self.device)
        self._processed_actions = th.zeros_like(self.raw_actions)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> th.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> th.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: th.Tensor):
        # obtain quantities from simulation
        command_name: str = self.cfg.command_name 
        command: th.Tensor = self._env.command_manager.get_command(command_name)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # obtain quantities from simulation
        # command_name: str = self.cfg.command_name 
        # command: th.Tensor = self._env.command_manager.get_command(command_name)
        # ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # # set command into controller
        # self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)
