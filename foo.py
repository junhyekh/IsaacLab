import omni.isaac.lab.utils.math as math_utils
import torch as th
from icecream import ic


if __name__ == '__main__':
    a = th.Tensor([ 0.0058,  0.0000,  0.0000, -1.0000])
    b = th.Tensor([0.0058, 0.0000, 0.0000, 1.0000])
    zero = th.Tensor([0., 0., 0.])
    x = th.Tensor([1., 0., 0.])
    c = th.Tensor([0., 0., 0.])
    ic(math_utils.quat_from_euler_xyz(zero, zero, c))
    ic(math_utils.quat_from_euler_xyz(zero, zero, c))
    ic(math_utils.combine_frame_transforms(
        x,
        a,
        x,
        a,
    ))
    ic(math_utils.combine_frame_transforms(
        x,
        b,
        x,
        a,
    ))