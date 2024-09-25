import torch as th
import numpy as np

def dcn(x: th.Tensor) -> np.ndarray:
    """
    Convert torch tensor into numpy array.
    """
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return x