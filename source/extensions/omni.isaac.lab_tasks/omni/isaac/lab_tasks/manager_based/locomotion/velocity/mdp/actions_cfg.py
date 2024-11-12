from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.controllers import DifferentialIKControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from .actions import PassiveIKAction

@configclass
class PassiveIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """
    class_type: type[ActionTerm] = PassiveIKAction

    command_name: str = MISSING
    control_arm: Literal["left", "right"] = MISSING