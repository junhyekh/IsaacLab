import torch as th
from typing import Optional

@th.jit.script
def quat_rotate(q: th.Tensor, x: th.Tensor) -> th.Tensor:
    q_ax = q[..., 0:3]
    t = 2.0 * th.cross(q_ax, x, dim=-1)
    return x + q[..., 3:4] * t + th.cross(q_ax, t, dim=-1)

def _matrix_from_quaternion(x: th.Tensor, out: th.Tensor):
    # qx, qy, qz, qw = [x[..., i] for i in range(4)]
    qx = x[..., 0]
    qy = x[..., 1]
    qz = x[..., 2]
    qw = x[..., 3]

    tx = 2.0 * qx
    ty = 2.0 * qy
    tz = 2.0 * qz
    twx = tx * qw
    twy = ty * qw
    twz = tz * qw
    txx = tx * qx
    txy = ty * qx
    txz = tz * qx
    tyy = ty * qy
    tyz = tz * qy
    tzz = tz * qz

    # outr = out.reshape(-1, 9)
    # th.stack([
    #     1.0 - (tyy + tzz),
    #     txy - twx,
    #     txz + twy,
    #     txy + twz,
    #     1.0 - (txx + tzz),
    #     tyz - twx,
    #     txz - twy,
    #     tyz + twx,
    #     1.0 - (txx + tyy)], dim=-1, out=outr)
    out[..., 0, 0] = 1.0 - (tyy + tzz)
    out[..., 0, 1] = txy - twz
    out[..., 0, 2] = txz + twy
    out[..., 1, 0] = txy + twz
    out[..., 1, 1] = 1.0 - (txx + tzz)
    out[..., 1, 2] = tyz - twx
    out[..., 2, 0] = txz - twy
    out[..., 2, 1] = tyz + twx
    out[..., 2, 2] = 1.0 - (txx + tyy)

@th.jit.script
def _matrix_from_pose(p: th.Tensor, q: th.Tensor,
                      out: th.Tensor):
    _matrix_from_quaternion(q, out[..., :3, :3])
    out[..., :3, 3] = p
    out[..., 3, 3] = 1
    return out

def matrix_from_pose(p: th.Tensor, q: th.Tensor,
                     out: Optional[th.Tensor] = None):
    batch_shape = th.broadcast_shapes(p.shape[:-1], q.shape[:-1])
    if out is None:
        out = th.zeros(*batch_shape, 4, 4,
                       dtype=p.dtype,
                       device=p.device)
    _matrix_from_pose(p, q, out)
    return out