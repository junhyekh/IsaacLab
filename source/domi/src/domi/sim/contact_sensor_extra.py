#!/usr/bin/env python3

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
import torch as th

from omni.isaac.lab.sensors import (
    ContactSensor,
    ContactSensorData
)
from omni.isaac.lab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from .contact_sensor_extra_cfg import ContactSensorExtraCfg

from pkm.util.math_util import (
    xyzw2wxyz,
    align_vectors
)


@dataclass
class ContactSensorExtraData(ContactSensorData):
    """
    Extra data based on contact filters.
    See:
    https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.objects.VisualCuboid.get_contact_force_data
    """
    c_force: th.Tensor | None = None
    c_point: th.Tensor | None = None
    c_normal: th.Tensor | None = None
    c_dist: th.Tensor | None = None
    c_num: th.Tensor | None = None
    c_idx: th.Tensor | None = None
    c_env: th.Tensor | None = None


class ContactSensorExtra(ContactSensor):
    cfg: 'ContactSensorExtraCfg'

    def __init__(self, cfg: 'ContactSensorExtraCfg'):
        super().__init__(cfg)
        # NOTE(ycho): override `self._data`
        self._data: ContactSensorExtraData = ContactSensorExtraData()

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        # th.isin(?,?)

        self._data.net_forces_w[env_ids] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()

        c: int = self.cfg.max_contact_data_count
        self._data.c_force = th.zeros(c, 3, device=self._device)
        self._data.c_point = th.zeros(c, 3, device=self._device)
        self._data.c_normal = th.zeros(c, 3, device=self._device)
        self._data.c_dist = th.zeros(c, 1, device=self._device)
        self._data.c_env = th.zeros(c, dtype=th.long, device=self._device)

        n: int = self._num_envs
        m: int = self.contact_physx_view.filter_count
        self._data.c_num = th.zeros(n, m, dtype=th.long, device=self._device)
        self._data.c_idx = th.zeros(n, m, dtype=th.long, device=self._device)

    def _update_buffers_impl(self,
                             env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        super()._update_buffers_impl(env_ids)

        # default to all sensors
        set_sparse: bool = False
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        else:
            set_sparse = True

        # NOTE(ycho): unlike other fields in the buffer,
        # these observations are _not_ partitioned in terms of
        # environments.
        c_view = self.contact_physx_view
        c_data = (c_view.get_contact_data(dt=self._sim_physics_dt))
        (force, point, normal, dist, num, idx) = c_data

        if set_sparse:
            # Only certain envs are set
            raise ValueError('set_sparse is not supported!!')
        else:
            # All fields are set
            c: int = self.cfg.max_contact_data_count
            num_f = num.reshape(-1)

            # NOTE(ycho): clip `num_f` with the contact buffer size
            # to ensure that the ranges denoted by `c_num / c_idx`
            # will be valid. Assumes the underlying `idx_f` is sorted.
            cum_f = th.cumsum(num_f, dim=0)
            msk_ovf = (cum_f >= c)
            num_f[msk_ovf] = (c - cum_f - num_f).clamp_min_(0)[msk_ovf].to(
                dtype=num_f.dtype)
            count: int = int(num.sum())

            self._data.c_force[...] = force
            self._data.c_point[...] = point
            self._data.c_normal[...] = normal
            self._data.c_dist[...] = dist
            self._data.c_num[...] = num
            self._data.c_idx[...] = idx
            self._data.c_env[:count] = th.repeat_interleave(
                th.arange(self._num_envs,
                          dtype=th.long,
                          device=self._device),
                num.reshape(-1),
                dim=-1,
                output_size=count)
            # NOTE(ycho): fill invalid indices with -1
            self._data.c_env[count:] = -1

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)

        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "contact_visualizer_2"):
                self.contact_visualizer_2 = VisualizationMarkers(
                    self.cfg.visualizer_cfg_2)
            # set their visibility to true
            self.contact_visualizer_2.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer_2"):
                self.contact_visualizer_2.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

        # Marked with presence of contact
        c_has = (self._data.c_env >= 0)
        point = self._data.c_point[c_has]

        x = th.zeros_like(self._data.c_normal[c_has])
        x[..., 0] = 1
        quat_xyzw = align_vectors(x, self._data.c_normal[c_has]).view(-1, 4)

        # FIXME(ycho): might not be the best visualization
        arrow_scale = th.tensor(
            [1.0, 0.01, 0.01],
            device=self.device).repeat(
            quat_xyzw.shape[0],
            1) * 6

        self.contact_visualizer_2.visualize(
            point.view(-1, 3),
            xyzw2wxyz(quat_xyzw),
            arrow_scale
        )
