
from __future__ import annotations

import torch as th
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg
from omni.isaac.lab.sensors import ContactSensor
from domi.sim.contact_sensor_extra import ContactSensorExtra, ContactSensorExtraData
from icecream import ic
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg

import omni.isaac.lab.sim as sim_utils

from pkm.util.math_util import (
    xyzw2wxyz,
    align_vectors
)

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def process_contact_data(
        c_force: th.Tensor,
        c_normal: th.Tensor,
        c_point: th.Tensor,
        c_idx: th.Tensor,
        c_num: th.Tensor,
        M: int = 4,
        force_thresh: float = 1.
        ):
    """
    Process the contact data from the ContactSensorExtra
    Into a batch tensor.

    Args:
        c_force: th.Tensor [C, 1]
        c_normal: th.Tensor [C, 3]
        c_point: th.Tensor [C, 3]
        c_idx: th.Tensor [N, F]
        c_num: th.Tensor [N, F]
        M: int = 4 -> # of columns for the returns

    Returns:
        forces: th.Tensor [N, M, 3]
        points: th.Tensor [N, M, 3]
        valid_masks: th.Tensor [N, M]
    """

    if (c_num> M).any():
        raise ValueError(f'More than {M} contacts in an environment')

    N = c_num.shape[0]  
    device = c_force.device
    m = c_num.shape[1]  # Number of filters

    forces = th.zeros(N, M, 3, device=device)
    points = th.zeros(N, M, 3, device=device)
    valid_masks = th.zeros(N, M, dtype=th.bool, device=device)

    # Flatten c_num and c_idx
    c_num_flat = c_num.view(-1)   # [N * m]
    c_idx_flat = c_idx.view(-1)  # [N * m]
    c_env_flat = th.arange(N, 
        device=device).unsqueeze(1).repeat(1, m).view(-1)  # [N * m]

    # Filter out zero contact counts
    nonzero_mask = c_num_flat > 0
    c_num_nonzero = c_num_flat[nonzero_mask]  # [K]
    c_idx_nonzero = c_idx_flat[nonzero_mask]  # [K]
    c_env_nonzero = c_env_flat[nonzero_mask]  # [K]

    if c_num_nonzero.numel() == 0:
        return forces, points, valid_masks

    cum_counts_nonzero = th.cumsum(c_num_nonzero, dim=0)

    c_indices = th.repeat_interleave(c_idx_nonzero, c_num_nonzero) + \
                th.arange(cum_counts_nonzero[-1], device=device) - \
                th.repeat_interleave(
                    cum_counts_nonzero - c_num_nonzero, 
                    c_num_nonzero)

    c_envs = th.repeat_interleave(c_env_nonzero, c_num_nonzero)  

    c_force_vec = c_force[c_indices] * c_normal[c_indices]
    c_point_vec = c_point[c_indices]

    # Compute positions within each environment
    cum_counts = th.cumsum(c_num_flat, dim=0)
    env_offsets = th.cat([th.tensor([0], device=device), cum_counts[:-1]])
    idx_in_env = th.arange(cum_counts[-1], device=device) - \
                 th.repeat_interleave(env_offsets, c_num_flat)

    # Assign contact data to output tensors
    forces[c_envs, idx_in_env] = c_force_vec
    points[c_envs, idx_in_env] = c_point_vec
    valid_masks[c_envs, idx_in_env] = True

    # Mark contact forces smaller than threshold as invalid
    small_forces = forces.norm(dim=-1) < force_thresh
    forces[small_forces, :] = 0.
    points[small_forces, :] = 0.
    valid_masks[small_forces] = False

    return forces, points, valid_masks

def compute_com(asset: Articulation, device: str) -> th.Tensor:
    """
    Returns the COM in the world frame given the articulation
    assets.
    """
    # link_com_pose_b = asset.data.root_physx_view.get_coms().to(device)
    link_com_pose_b = asset.root_physx_view.get_coms().to(device)
    link_pose = asset.data.body_state_w[..., :7]
    link_mass = asset.data.default_mass.to(device)

    link_com_pos_w, link_com_quat_w = math_utils.combine_frame_transforms(
        link_pose[..., :3],
        xyzw2wxyz(link_pose[..., 3:7]),
        link_com_pose_b[..., :3],
        xyzw2wxyz(link_com_pose_b[..., 3:7]),
    )
    com_pos_w = \
        (link_com_pos_w * link_mass.unsqueeze(-1)).sum(dim=1) \
        / link_mass.sum(dim=-1).unsqueeze(-1)
    
    return com_pos_w

def compute_zmp(
        forces: th.Tensor,
        points: th.Tensor,
        valid_masks: th.Tensor,
        coms: th.Tensor
    ) -> th.Tensor:
    """
    Compute the Zero Moment Point (ZMP) given the contact forces, points, valid masks, and centers of mass.

    Args:
        forces: torch.Tensor [N, M, 3] - Contact forces for N samples and M contact points.
        points: torch.Tensor [N, M, 3] - Contact points for N samples and M contact points.
        valid_masks: torch.Tensor [N, M] - Boolean masks indicating valid contact points.
        coms: torch.Tensor [N, 3] - Centers of mass for N samples.

    Returns:
        zmp: torch.Tensor [N, 3] - Zero Moment Points for N samples.
    """
    masks = valid_masks.unsqueeze(-1)  # [N, M, 1]
    forces_valid = forces * masks      # [N, M, 3]
    points_valid = points * masks      # [N, M, 3]

    fz = forces_valid[:, :, 2]  # [N, M]
    px = points_valid[:, :, 0]  # [N, M]
    py = points_valid[:, :, 1]  # [N, M]

    epsilon = 1e-8
    fz_sum = th.sum(fz, dim=1, keepdim=True)  # [N, 1]

    all_zero_forces = th.all(fz_sum < 1, dim=1)  # [N]

    alpha = fz / (fz_sum + epsilon)# [N, M]
    zmp_x = th.sum(px * alpha, dim=1)  # [N]
    zmp_y = th.sum(py * alpha, dim=1)  # [N]

    zmp = th.zeros_like(coms)  # [N, 3]
    zmp[:, 0] = zmp_x
    zmp[:, 1] = zmp_y

    # Replace ZMP with center of mass if all contact forces are zero
    zmp[all_zero_forces, :2] = coms[all_zero_forces, :2]

    return zmp


def compute_2d_convex_hull(
        p_points: th.Tensor, 
        p_mask: th.Tensor) -> th.Tensor:
    """
    Computes 2d convex hull given batched points.
    Uses Gift wrapping algorith.: https://en.wikipedia.org/wiki/Gift_wrapping_algorith.

    Args:
        p_points: th.Tensor [N, M, 2]
        p_mask: th.Tensor [N, M]

    Returns:
        hull_points: th.Tensor [N, M, 2] -> convex hull nodes, will be Nan
            for the invalid(e.g. mask as invalid or interal) points
        hull_idx: th.Tensor [N, M] -> indices of convex hull nodes
    """
    N, M, P = p_points.shape
    if P != 2:
        raise ValueError("Must be 2d points")
    hull_indices = th.full((N, M), -1, dtype=th.long, device=p_points.device)

    px = p_points[..., 0].clone()  # [N, M]
    px[~p_mask] = float('inf')

    # Find th. leftmost point for each sample
    _, leftmost_indices = px.min(dim=1)  # [N]
    curr_indices = leftmost_indices.clone()  # [N]

    # Initialize previous edge vector as [0, 1] for all samples
    prev_edge = th.zeros(N, 2, device=p_points.device)
    prev_edge[:, 1] = -1.0  # [N, 2]

    valid_point_counts = p_mask.sum(dim=1)  # [N]
    completed = th.zeros(N, dtype=th.bool, device=p_points.device)

    # Handle samples with 0 valid points
    no_point_mask = valid_point_counts == 0
    if no_point_mask.any():
        completed[no_point_mask] = True

    # Handle samples with.1 valid point
    one_point_mask = valid_point_counts == 1
    if one_point_mask.any():
        sample_indices = th.where(one_point_mask)[0]
        point_indices = p_mask[one_point_mask].nonzero(as_tuple=True)[1]
        hull_indices[sample_indices, 0] = point_indices
        completed[sample_indices] = True

    # Keep track of starting point to detect when we complete the convex hull
    starting_indices = leftmost_indices.clone()

    for i in range(M):
        # For samples that are not yet completed
        active = ~completed  # [N]
        if not active.any():
            break  

        active_envs = th.where(active)[0]  # [N_active]
        curr_indices_a = curr_indices[active_envs]  # [N_active]
        prev_edge_a = prev_edge[active_envs]  # [N_active, 2]

        # Current points for active samples
        curr_points = p_points[active_envs, curr_indices_a]  # [N_active, 2]
        v_to_candidates = p_points[active_envs] - curr_points.unsqueeze(1)  # [N_active, M, 2]

        # Mask as invalid for (1) current point, and (2) invalid points
        valid_mask = p_mask[active_envs].clone()  # [N_active, M]
        valid_mask[th.arange(valid_mask.size(0), device=p_points.device), curr_indices_a] = False

        v_norm = v_to_candidates.norm(dim=2, keepdim=True)
        v_norm[v_norm == 0] = 1e-6  
        v_normalized = v_to_candidates / v_norm

        dot_product = th.einsum('...ij,...ij->...i', prev_edge_a.unsqueeze(1), v_normalized)
        dot_product[~valid_mask] = float('inf')

        # Find next point indices
        min_dot_product, next_point_indices = dot_product.min(dim=1)  # [N_active]

        # Detect envs for terminated
        no_valid_candidates = min_dot_product == float('inf')  # [N_active]
        completed[active_envs[no_valid_candidates]] = True

        # Samples with valid next points
        valid_next = ~no_valid_candidates  # [N_active]

        if valid_next.any():
            valid_indices_a = active_envs[valid_next]  # [N_valid_next]

            # Update convex hull indices
            hull_indices[valid_indices_a, i] = curr_indices[valid_indices_a]

            next_indices = next_point_indices[valid_next]  # [N_valid_next]
            new_curr_points = curr_points[valid_next]  # [N_valid_next, 2]
            next_points = p_points[valid_indices_a, next_indices]  # [N_valid_next, 2]

            curr_indices[valid_indices_a] = next_indices
            prev_edge[valid_indices_a] = new_curr_points - next_points

            returned_to_start = \
                curr_indices[valid_indices_a] == starting_indices[valid_indices_a]
            completed[valid_indices_a[returned_to_start]] = True
        else:
            if not (~completed).any():
                break

        curr_indices[completed] = -1
        prev_edge[completed] = 0

    # Create a mask for convex hull points
    is_hull = hull_indices != -1  # [N, M]
    hull_lengths = is_hull.sum(dim=1)  # [N]
    pos_in_hull = th.full(
        (N, M), fill_value=M + M, device=p_points.device)

    indices_i = th.arange(
        M, device=p_points.device).unsqueeze(0).expand(N, M)  # [N, M]
    valid_hull_mask = hull_indices != -1  # [N, M]

    hull_indices_valid = hull_indices[valid_hull_mask]  # [K_total]
    positions_i_valid = indices_i[valid_hull_mask]  # [K_total]
    batch_indices = th.arange(
        N, device=p_points.device).unsqueeze(1).expand(N, M)[valid_hull_mask]  # [K_total]
    pos_in_hull[batch_indices, hull_indices_valid] = positions_i_valid

    perm_indices = pos_in_hull.argsort(dim=1)
    hull_points = p_points[th.arange(N)[:, None], perm_indices]

    # Create polygon_idx starting from 0 for convex hull points
    positions = th.arange(
        M, device=p_points.device).unsqueeze(0).expand(N, M)  # [N, M]
    mask = positions < hull_lengths.unsqueeze(1)
    hull_idx = th.full((N, M), -1, dtype=th.long, device=p_points.device)
    hull_idx[mask] = positions[mask]

    return hull_points, hull_idx


def hull_point_signed_dist(
        hull_points: th.Tensor, 
        hull_idx: th.Tensor, 
        points: th.Tensor):
    """
    Args:
        hull_points: th.Tensor [N, M, 2] -> convex hull points
        hull_idx: th.Tensor [N, M] -> Indices of convex hull points, 
            -1 for invalid points.
        points: th.Tensor [N, 2] - Query points.
    Returns:
        signed_distance: th.Tensor [N] -> Signed distance from each 
            point to its convex hull.
    """
    N, M, _ = hull_points.shape
    device = hull_points.device

    is_hull = hull_idx != -1  # [N, M]
    hull_lengths = is_hull.sum(dim=1)  # [N]
    max_K = max(1, hull_lengths.max().item()) 

    hull_points = hull_points[:, :max_K, :]  # [N, max_K, 2]
    K = hull_lengths  # [N]

    # NOTE(ytcho): Make index tensor for start and the end
    # E.g. idx_start = [0, 1, 2, 3] then idx_end = [1, 2, 3, 0]
    idx_start = th.arange(max_K, device=device)[None, ...].expand(N, max_K)  # [N, max_K]
    valid = idx_start < K[..., None]  # [N, max_K], boolean mask

    # For end indices, wrap around using modulo K[n]
    K_expanded = K.unsqueeze(1).expand(N, max_K)  # [N, max_K]
    idx_end = (idx_start + 1) % th.clamp(K_expanded, min=1)
    idx_end[~valid] = idx_start[~valid]

    batch_idx = th.arange(N, device=device)[..., None].expand(N, max_K)  # [N, max_K]

    start_points = hull_points[batch_idx, idx_start]  # [N, max_K, 2]
    end_points = hull_points[batch_idx, idx_end]  # [N, max_K, 2]

    # Set zero for invalid points
    start_points[~valid] = 0
    end_points[~valid] = 0

    # Edge vector
    edge_v = end_points - start_points  # [N, max_K, 2]
    # Point to Verticies vector
    point_to_vertices = points.unsqueeze(1) - start_points  # [N, max_K, 2]

    edge_l_sq = (edge_v ** 2).sum(dim=2, keepdim=True)  # [N, max_K, 1]
    edge_l_sq[edge_l_sq == 0] = 1e-8

    # Projection scalar
    t = (point_to_vertices * edge_v).sum(dim=2, keepdim=True) / edge_l_sq  # [N, max_K, 1]
    t_clamped = t.clamp(0, 1)

    projected_point = start_points + t_clamped * edge_v  # [N, max_K, 2]
    dist = (points.unsqueeze(1) - projected_point).norm(dim=2)  # [N, max_K]
    dist[~valid] = float('inf')

    min_dist, _ = dist.min(dim=1)  # [N]

    # Now determine if the point lies inside the supp_polygon
    inside_mask = th.zeros(N, dtype=th.bool, device=device)

    # K >= 3
    mask_3 = K >= 3
    if mask_3.any():
        indices_3 = th.where(mask_3)[0]
        points_3 = points[indices_3]  # [N1, 2]
        hull_points_3 = hull_points[indices_3]  # [N1, max_K, 2]
        hull_size_3 = K[indices_3]  # [N1]

        inside_mask_3 = is_point_in_2d_hull(points_3, hull_points_3, hull_size_3)
        inside_mask[indices_3] = inside_mask_3

    # K == 1
    mask_1 = K == 1
    if mask_1.any():
        indices_1 = th.where(mask_1)[0]

        inside_mask_1 = min_dist[indices_1] < 1e-6
        inside_mask[indices_1] = inside_mask_1

    # K == 2
    mask_2 = K == 2
    if mask_2.any():
        indices_2 = th.where(mask_2)[0]
        t_2 = t_clamped[indices_2, :2, 0]  # [N3, 2]
        distances_eq_2 = dist[indices_2, :2]  # [N3, 2]

        on_segment = ((t_2 >= 0) & (t_2 <= 1) & (distances_eq_2 < 1e-6)).any(dim=1)
        inside_mask[indices_2] = on_segment

    signed_dist = min_dist
    signed_dist[inside_mask] *= -1

    # K == 0
    no_edges = K == 0
    signed_dist[no_edges] = 0.

    return signed_dist

def is_point_in_2d_hull(
        points:th.Tensor, 
        hull_points:th.Tensor, 
        num_vertices:th.Tensor):
    """
    Vectorized point-in-polygon test using the winding number method.
    Args:
        points: th.Tensor [N1, 2] - Query points.
        polygon: th.Tensor [N1, max_K, 2] - Polygon vertices.
        lengths: th.Tensor [N1] - Number of valid vertices per sample.
    Returns:
        inside: th.Tensor [N1] - Boolean mask indicating whether each point is inside its polygon.
    """
    point_to_hull = hull_points - points[:, None, :]  # [N, M, 2]

    angles = th.atan2(point_to_hull[..., 1], 
                      point_to_hull[..., 0])  # [N, M]

    N, max_K, _ = hull_points.shape
    indices = th.arange(max_K, device=points.device)[None, ...].expand(N, max_K)  # [N1, max_K]
    valid_mask = indices < num_vertices.unsqueeze(1)  # [N1, max_K]
    valid_angles = th.where(
        valid_mask,
        angles,
        th.tensor(float('nan'), device=points.device)
    )

    # Sort valid angles (NaNs at the end)
    valid_angles, _ = th.sort(valid_angles, dim=-1)  # [N, M]

    # Replace NaNs (invalid points) with the smallest valid angle of the row
    first_angle = valid_angles[:, 0:1]  # [N, 1]
    sorted_angles = th.where(
        th.isnan(valid_angles), first_angle, valid_angles)
    
    # Compute the differences between consecutive sorted valid angles
    delta_angles = th.diff(sorted_angles, dim=-1)  # [N, M-1]
    wrap_delta = \
        sorted_angles[:, 0] - sorted_angles[:, -1]  # [N]
    delta_angles = th.cat([delta_angles, wrap_delta[..., None]], dim=-1)  # [N, M]

    # Map from 0 to 2*pi
    delta_angles = th.where(
        delta_angles >= 0,
        delta_angles,
        delta_angles + 2 * th.pi)

    # If any delta_angle exceeds pi, meaning the point is outside the polygon
    is_outside = th.any(delta_angles > th.pi, dim=-1) \
        | th.isnan(valid_angles[..., 1]) # True if only one valid point in hull

    return ~is_outside


def zmp_supp_dist(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg,
        left_foot_sensor_cfg: SceneEntityCfg,
        right_foot_sensor_cfg: SceneEntityCfg
    ) -> th.Tensor:
    """
    Computes the distance/margin between the support polygon and the zmp
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    left_foot_sensor: ContactSensorExtra = env.scene.sensors[left_foot_sensor_cfg.name]
    right_foot_sensor: ContactSensorExtra = env.scene.sensors[right_foot_sensor_cfg.name]

    # ic(
    #     left_foot_sensor.data.c_force[:4],
    #     left_foot_sensor.data.c_normal[:4],
    #     left_foot_sensor.data.c_point[:4],
    # )
    left_forces, left_points, left_masks = \
        process_contact_data(
            left_foot_sensor.data.c_force,
            left_foot_sensor.data.c_normal,
            left_foot_sensor.data.c_point,
            left_foot_sensor.data.c_idx,
            left_foot_sensor.data.c_num)

    right_forces, right_points, right_masks = \
        process_contact_data(
            right_foot_sensor.data.c_force,
            right_foot_sensor.data.c_normal,
            right_foot_sensor.data.c_point,
            right_foot_sensor.data.c_idx,
            right_foot_sensor.data.c_num)
    
    # ic(asset.data.default_mass)
    com_pos_w = compute_com(asset, env.device)
    zmp_w = compute_zmp(
        th.cat([left_forces, right_forces], dim=1),
        th.cat([left_points, right_points], dim=1),
        th.cat([left_masks, right_masks], dim=1),
        com_pos_w
    )
    hull_points, hull_idx = compute_2d_convex_hull(
        th.cat([left_points, right_points], dim=1)[..., :2],
        th.cat([left_masks, right_masks], dim=1)
    )

    singed_zmp_dist = hull_point_signed_dist(
        hull_points, hull_idx, zmp_w[..., :2]
    )
    singed_com_dist = hull_point_signed_dist(
        hull_points, hull_idx, com_pos_w[..., :2]
    )
    # ic(singed_com_dist, singed_zmp_dist)
    proj_com = th.zeros_like(com_pos_w)
    proj_com[..., :2] = com_pos_w[..., :2]

    if not hasattr(env, "com_markers"):
        com_markers_cfg= VisualizationMarkersCfg(
            prim_path="/Visuals/com",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            }
        )
        env.com_markers = VisualizationMarkers(com_markers_cfg)
        zmp_markers_cfg= VisualizationMarkersCfg(
            prim_path="/Visuals/zmp",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.05,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            }
        )
        env.zmp_markers = VisualizationMarkers(zmp_markers_cfg)
    env.com_markers.visualize(
        proj_com
    )
    env.zmp_markers.visualize(
        zmp_w
    )
    return th.zeros(env.num_envs, device=env.device)

