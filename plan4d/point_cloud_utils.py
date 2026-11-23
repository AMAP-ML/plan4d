"""Point Cloud Generation Utilities for DA3 Output.

This module provides functions to:
1. Extract camera parameters (intrinsics + extrinsics) from ray map using DA3's camray_to_caminfo
2. Generate point cloud from depth + camera parameters
3. Save and visualize point clouds

Usage:
    from plan4d.point_cloud_utils import generate_point_cloud_from_da3_output
    
    results = generate_point_cloud_from_da3_output(
        pred_depth=pred_depth,
        pred_ray=pred_ray,
        rgb=rgb,
        output_dir='./output',
    )
"""

import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np
from einops import repeat
from typing import Dict, Optional, Tuple


# Optional matplotlib import (not required for core functionality)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Add DA3 source path
project_root = Path(__file__).parent.parent
da3_src_path = str(project_root / 'Depth-Anything-3' / 'src')
if da3_src_path not in sys.path:
    sys.path.insert(0, da3_src_path)

logger = logging.getLogger(__name__)


def _tensor_to_numpy_safe(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensors to NumPy arrays while handling bf16 tensors explicitly."""
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def _determinant_3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Compute a 3x3 determinant using basic tensor ops only.

    This avoids `torch.det`, which can invoke CUDA linalg backends and trip
    over cuSOLVER handle creation for tiny matrices inside training loops.
    """
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 matrix, got shape {tuple(matrix.shape)}")

    return (
        matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
        - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
        + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0])
    )


def procrustes_align(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve Procrustes alignment: find R, s, t such that s*R@source + t ≈ target.

    Args:
        source_points: [N, 3] points in source coordinate system
        target_points: [N, 3] points in target coordinate system

    Returns:
        rotation: [3, 3] rotation matrix
        scale: scalar scale factor
        translation: [3] translation vector
    """
    # SVD does not support bfloat16; disable autocast and cast to float32.
    # torch.autocast (used by FSDP/AMP) can silently override .float(),
    # so we must explicitly disable it for the entire computation block.
    original_dtype = source_points.dtype
    device_type = "cuda" if source_points.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        source_points = source_points.float()
        target_points = target_points.float()

        source_mean = source_points.mean(dim=0)
        target_mean = target_points.mean(dim=0)

        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        source_variance = (source_centered ** 2).sum()

        # Degenerate case: source points are nearly identical (e.g. training
        # early stage where pred_ray origins are the same across frames).
        # Must check BEFORE SVD — cuSOLVER crashes on degenerate/NaN matrices.
        # Return identity rotation, unit scale, and translation-only alignment
        # to avoid scale explosion (source_variance ≈ 0 → scale → ∞ → NaN).
        if source_variance < 1e-6:
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            scale = torch.ones(1, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation, scale.squeeze(), translation

        cross_covariance = target_centered.T @ source_centered  # [3, 3]

        # Guard against NaN/Inf in cross_covariance (e.g. from upstream NaN
        # in pred_ray or gt_ray).  Fall back to identity alignment.
        if not torch.isfinite(cross_covariance).all():
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            scale = torch.ones(1, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation, scale.squeeze(), translation

        # SVD can fail on CUDA with degenerate matrices (cuSOLVER internal
        # error).  Try CUDA first, fall back to CPU, then to identity.
        try:
            u_matrix, singular_values, vh_matrix = torch.linalg.svd(cross_covariance)
        except RuntimeError:
            try:
                cc_cpu = cross_covariance.cpu()
                u_cpu, s_cpu, vh_cpu = torch.linalg.svd(cc_cpu)
                u_matrix = u_cpu.to(source_points.device)
                singular_values = s_cpu.to(source_points.device)
                vh_matrix = vh_cpu.to(source_points.device)
            except RuntimeError:
                rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
                scale = torch.ones(1, device=source_points.device, dtype=torch.float32)
                translation = target_mean - source_mean
                return rotation, scale.squeeze(), translation

        # Handle reflection: ensure proper rotation (det(R) = +1)
        sign_correction = torch.ones(3, device=source_points.device, dtype=torch.float32)
        orthogonal_candidate = u_matrix @ vh_matrix
        if not torch.isfinite(orthogonal_candidate).all():
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            scale = torch.ones(1, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation, scale.squeeze(), translation
        if _determinant_3x3(orthogonal_candidate) < 0:
            sign_correction[-1] = -1.0

        rotation = u_matrix @ torch.diag(sign_correction) @ vh_matrix  # [3, 3]

        scale = (singular_values * sign_correction).sum() / source_variance.clamp(min=1e-12)

        translation = target_mean - scale * (rotation @ source_mean)

    # Cast back to original dtype
    rotation = rotation.to(original_dtype)
    scale = scale.to(original_dtype)
    translation = translation.to(original_dtype)

    return rotation, scale, translation


def rigid_align(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve rigid alignment: find R, t such that R@source + t ≈ target."""
    original_dtype = source_points.dtype
    device_type = "cuda" if source_points.is_cuda else "cpu"

    with torch.amp.autocast(device_type=device_type, enabled=False):
        source_points = source_points.float()
        target_points = target_points.float()

        source_mean = source_points.mean(dim=0)
        target_mean = target_points.mean(dim=0)

        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        source_variance = (source_centered ** 2).sum()
        if source_variance < 1e-6:
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation.to(original_dtype), translation.to(original_dtype)

        cross_covariance = target_centered.T @ source_centered
        if not torch.isfinite(cross_covariance).all():
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation.to(original_dtype), translation.to(original_dtype)

        try:
            u_matrix, _, vh_matrix = torch.linalg.svd(cross_covariance)
        except RuntimeError:
            try:
                cc_cpu = cross_covariance.cpu()
                u_cpu, _, vh_cpu = torch.linalg.svd(cc_cpu)
                u_matrix = u_cpu.to(source_points.device)
                vh_matrix = vh_cpu.to(source_points.device)
            except RuntimeError:
                rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
                translation = target_mean - source_mean
                return rotation.to(original_dtype), translation.to(original_dtype)

        sign_correction = torch.ones(3, device=source_points.device, dtype=torch.float32)
        orthogonal_candidate = u_matrix @ vh_matrix
        if not torch.isfinite(orthogonal_candidate).all():
            rotation = torch.eye(3, device=source_points.device, dtype=torch.float32)
            translation = target_mean - source_mean
            return rotation.to(original_dtype), translation.to(original_dtype)
        if _determinant_3x3(orthogonal_candidate) < 0:
            sign_correction[-1] = -1.0

        rotation = u_matrix @ torch.diag(sign_correction) @ vh_matrix
        translation = target_mean - rotation @ source_mean

    return rotation.to(original_dtype), translation.to(original_dtype)


def procrustes_align_with_orientation(
    pred_centers: torch.Tensor,
    gt_centers: torch.Tensor,
    pred_directions: torch.Tensor,
    gt_directions: torch.Tensor,
    direction_weight: float = 1.0,
    estimate_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Procrustes with orientation anchors to resolve rotation degeneracy.

    When camera centers are nearly collinear, standard Procrustes cannot
    determine the rotation component around the motion axis.  By appending
    synthetic "look-at" points (center + epsilon * direction) we inject
    orientation information so that the SVD-based alignment also constrains
    the viewing direction.

    Args:
        pred_centers: [S, 3] camera centers in pred world
        gt_centers: [S, 3] camera centers in GT world
        pred_directions: [S, 3] per-frame principal viewing direction (pred world)
        gt_directions: [S, 3] per-frame principal viewing direction (GT world)
        direction_weight: multiplier on the anchor offset (relative to scene spread)

    Returns:
        rotation: [3, 3] rotation matrix
        scale: scalar scale factor
        translation: [3] translation vector
    """
    center_spread = (gt_centers - gt_centers.mean(dim=0)).norm(dim=-1).mean()
    epsilon = center_spread.clamp(min=1e-6) * direction_weight

    pred_dir_normalized = torch.nn.functional.normalize(pred_directions, dim=-1)
    gt_dir_normalized = torch.nn.functional.normalize(gt_directions, dim=-1)

    pred_lookats = pred_centers + epsilon * pred_dir_normalized
    gt_lookats = gt_centers + epsilon * gt_dir_normalized

    augmented_pred = torch.cat([pred_centers, pred_lookats], dim=0)  # [2S, 3]
    augmented_gt = torch.cat([gt_centers, gt_lookats], dim=0)        # [2S, 3]

    if estimate_scale:
        rotation, scale, translation = procrustes_align(augmented_pred, augmented_gt)
    else:
        rotation, translation = rigid_align(augmented_pred, augmented_gt)
        scale = torch.ones((), device=rotation.device, dtype=rotation.dtype)

    return rotation, scale, translation


def align_pred_ray_to_gt(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    direction_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Align pred ray map to GT world coordinate system via rigid alignment.

    Both pred and GT ray are in format [B, S, 6, H, W] with [direction(3), origin(3)].
    Both use the same formula: direction = R_c2w @ K⁻¹ @ p, origin = camera_center_world,
    but in different world coordinate systems.

    This function uses orientation-augmented rigid alignment to avoid rotation
    degeneracy when camera centers are nearly collinear while preserving the
    original baseline scale. For each frame it appends a synthetic "look-at"
    point (center + epsilon * principal_direction) so the SVD also constrains
    the viewing direction, not just position.

    Steps:
    1. Extracts per-frame camera centers (spatial mean of origin channels)
    2. Extracts per-frame principal viewing direction (center pixel direction)
    3. Solves augmented rigid alignment (R_align, t_align) with scale fixed to 1
    4. Transforms pred ray: direction' = R_align @ direction,
       origin' = R_align @ origin + t

    Args:
        pred_ray: [B, S, 6, H, W] predicted ray map in DA3 canonical world
        gt_ray: [B, S, 6, H, W] ground truth ray map in dataset world
        direction_weight: weight for orientation anchors relative to scene spread

    Returns:
        aligned_pred_ray: [B, S, 6, H, W] pred ray aligned to GT world
        alignment_info: dict with R_align, s_align, t_align and diagnostics
    """
    batch_size, num_frames = pred_ray.shape[:2]
    height, width = pred_ray.shape[-2:]

    # Extract per-frame camera centers from origin channels
    pred_origins = pred_ray[:, :, 3:6, :, :]  # [B, S, 3, H, W]
    gt_origins = gt_ray[:, :, 3:6, :, :]

    pred_centers = pred_origins.mean(dim=(-1, -2))  # [B, S, 3]
    gt_centers = gt_origins.mean(dim=(-1, -2))

    # Extract per-frame principal viewing direction (center pixel)
    center_h, center_w = height // 2, width // 2
    pred_directions = pred_ray[:, :, :3, center_h, center_w]  # [B, S, 3]
    gt_directions = gt_ray[:, :, :3, center_h, center_w]      # [B, S, 3]

    aligned_rays = []
    alignment_infos = []

    for batch_idx in range(batch_size):
        rotation, scale, translation = procrustes_align_with_orientation(
            pred_centers=pred_centers[batch_idx],      # [S, 3]
            gt_centers=gt_centers[batch_idx],
            pred_directions=pred_directions[batch_idx],  # [S, 3]
            gt_directions=gt_directions[batch_idx],
            direction_weight=direction_weight,
            estimate_scale=False,
        )

        # Detach alignment params so gradients do NOT flow through SVD.
        # This prevents the model from "gaming the alignment" instead of
        # improving prediction quality.  The gradient ∂L/∂pred = R^T @ ∂L/∂aligned
        # simply rotates the gradient direction, which is mathematically equivalent
        # to comparing in the pred coordinate system — perfectly valid.
        rotation = rotation.detach()
        scale = scale.detach()
        translation = translation.detach()

        pred_direction = pred_ray[batch_idx, :, :3, :, :]  # [S, 3, H, W]
        pred_origin = pred_ray[batch_idx, :, 3:6, :, :]
        pred_centers_batch = pred_centers[batch_idx]
        gt_centers_batch = gt_centers[batch_idx]
        pred_directions_batch = pred_directions[batch_idx]
        gt_directions_batch = gt_directions[batch_idx]

        rotation_for_direction = rotation.to(device=pred_direction.device, dtype=pred_direction.dtype)
        rotation_for_origin = rotation.to(device=pred_origin.device, dtype=pred_origin.dtype)
        translation_for_origin = translation.to(device=pred_origin.device, dtype=pred_origin.dtype)
        rotation_for_centers = rotation.to(device=pred_centers_batch.device, dtype=pred_centers_batch.dtype)
        translation_for_centers = translation.to(device=pred_centers_batch.device, dtype=pred_centers_batch.dtype)
        rotation_for_center_dirs = rotation.to(
            device=pred_directions_batch.device,
            dtype=pred_directions_batch.dtype,
        )

        # direction: R_align @ pred_dir (rotation only, no scale — direction encodes R_c2w @ K⁻¹ @ p)
        aligned_direction = torch.einsum("ij,sjhw->sihw", rotation_for_direction, pred_direction)

        # origin: R_align @ pred_org + t_align (scale preserved)
        aligned_origin = (
            torch.einsum("ij,sjhw->sihw", rotation_for_origin, pred_origin)
            + translation_for_origin[:, None, None]
        )

        aligned_ray_batch = torch.cat([aligned_direction, aligned_origin], dim=1)  # [S, 6, H, W]
        aligned_rays.append(aligned_ray_batch)

        # Compute alignment residual on camera centers
        aligned_centers = (
            (rotation_for_centers @ pred_centers_batch.T).T
            + translation_for_centers
        )  # [S, 3]
        center_residual = (aligned_centers - gt_centers_batch).norm(dim=-1)  # [S]

        # Compute direction residual (cosine similarity at center pixel)
        aligned_center_dirs = (rotation_for_center_dirs @ pred_directions_batch.T).T  # [S, 3]
        gt_center_dirs = gt_directions_batch  # [S, 3]
        cos_sim = torch.nn.functional.cosine_similarity(aligned_center_dirs, gt_center_dirs, dim=-1)  # [S]

        alignment_infos.append({
            "rotation": _tensor_to_numpy_safe(rotation),
            "scale": float(scale.item()),
            "translation": _tensor_to_numpy_safe(translation),
            "center_residual_per_frame": _tensor_to_numpy_safe(center_residual),
            "center_residual_mean": float(center_residual.mean().item()),
            "direction_cosine_sim_per_frame": _tensor_to_numpy_safe(cos_sim),
            "direction_cosine_sim_mean": float(cos_sim.mean().item()),
        })

    aligned_pred_ray = torch.stack(aligned_rays, dim=0)  # [B, S, 6, H, W]

    combined_info = alignment_infos[0] if batch_size == 1 else alignment_infos
    return aligned_pred_ray, combined_info


def camray_to_camera_params(
    camray: torch.Tensor,
    image_height: int,
    image_width: int,
    confidence: torch.Tensor = None,
) -> dict:
    """
    Convert ray map to camera parameters (intrinsics + extrinsics) using DA3's camray_to_caminfo.
    
    Args:
        camray: Ray map tensor [B, S, H, W, 6] where channels are [direction(3), origin(3)]
        image_height: Original image height
        image_width: Original image width
        confidence: Optional confidence map [B, S, H, W]
        
    Returns:
        Dictionary containing camera parameters
    """
    try:
        from depth_anything_3.utils.ray_utils import camray_to_caminfo
        from depth_anything_3.utils.geometry import affine_inverse
        
        B, S, H, W, C = camray.shape
        
        if confidence is None:
            confidence = torch.ones(B, S, H, W, device=camray.device, dtype=camray.dtype)
        
        R, T, focal_lengths, principal_points = camray_to_caminfo(
            camray, 
            confidence=confidence,
            reproj_threshold=0.2,
            training=False
        )
        
        pred_extrinsic = torch.cat(
            [
                torch.cat([R, T.unsqueeze(-1)], dim=-1),
                repeat(
                    torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device),
                    "c -> b s 1 c",
                    b=R.shape[0],
                    s=R.shape[1],
                ),
            ],
            dim=-2,
        )
        pred_extrinsic = affine_inverse(pred_extrinsic)
        
        intrinsics = torch.zeros(B, S, 3, 3, device=camray.device, dtype=camray.dtype)
        intrinsics[:, :, 0, 0] = focal_lengths[:, :, 0] * image_width / 2.0
        intrinsics[:, :, 1, 1] = focal_lengths[:, :, 1] * image_height / 2.0
        intrinsics[:, :, 0, 2] = principal_points[:, :, 0] * image_width / 2.0
        intrinsics[:, :, 1, 2] = principal_points[:, :, 1] * image_height / 2.0
        intrinsics[:, :, 2, 2] = 1.0
        
        return {
            'extrinsics': pred_extrinsic,
            'intrinsics': intrinsics,
            'R': R,
            'T': T,
            'focal_lengths': focal_lengths,
            'principal_points': principal_points,
        }
        
    except ImportError as e:
        logger.warning(f"Could not import DA3 ray_utils: {e}")
        logger.warning("Using simplified camera parameter extraction")
        
        B, S, H, W, C = camray.shape
        ray_dir = camray[:, :, :, :, :3]
        ray_origin = camray[:, :, :, :, 3:]
        
        T = ray_origin.mean(dim=[2, 3])
        center_h, center_w = H // 2, W // 2
        forward = ray_dir[:, :, center_h, center_w, :]
        forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)
        
        up = torch.tensor([0.0, -1.0, 0.0], device=camray.device, dtype=camray.dtype)
        up = up.view(1, 1, 3).expand(B, S, 3)
        
        right = torch.cross(up, forward, dim=-1)
        right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-8)
        up = torch.cross(forward, right, dim=-1)
        
        R = torch.stack([right, up, forward], dim=-1)
        
        extrinsics = torch.eye(4, device=camray.device, dtype=camray.dtype)
        extrinsics = extrinsics.view(1, 1, 4, 4).expand(B, S, 4, 4).clone()
        extrinsics[:, :, :3, :3] = R
        extrinsics[:, :, :3, 3] = T
        
        fx = fy = image_width / 2.0
        cx, cy = image_width / 2.0, image_height / 2.0
        
        intrinsics = torch.zeros(B, S, 3, 3, device=camray.device, dtype=camray.dtype)
        intrinsics[:, :, 0, 0] = fx
        intrinsics[:, :, 1, 1] = fy
        intrinsics[:, :, 0, 2] = cx
        intrinsics[:, :, 1, 2] = cy
        intrinsics[:, :, 2, 2] = 1.0
        
        return {
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'R': R,
            'T': T,
            'focal_lengths': torch.tensor([[fx, fy]], device=camray.device).expand(B, S, 2),
            'principal_points': torch.tensor([[cx, cy]], device=camray.device).expand(B, S, 2),
        }


def depth_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    rgb: torch.Tensor = None,
    valid_mask: torch.Tensor = None,
    max_points: int = 100000,
) -> dict:
    """Convert depth map to point cloud using camera parameters."""
    if depth.dim() == 5:
        depth = depth.squeeze(2)
    
    B, S, H, W = depth.shape
    device = depth.device
    
    if extrinsics.shape[-2] == 3:
        ext_4x4 = torch.zeros(B, S, 4, 4, device=device, dtype=extrinsics.dtype)
        ext_4x4[:, :, :3, :4] = extrinsics
        ext_4x4[:, :, 3, 3] = 1.0
        extrinsics = ext_4x4
    
    u = torch.arange(W, device=device, dtype=depth.dtype)
    v = torch.arange(H, device=device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    
    u = u.view(1, 1, H, W).expand(B, S, H, W)
    v = v.view(1, 1, H, W).expand(B, S, H, W)
    ones = torch.ones_like(u)
    
    pixel_coords = torch.stack([u, v, ones], dim=-1)
    K_inv = torch.inverse(intrinsics)
    
    camera_coords = torch.einsum('bsij,bshwj->bshwi', K_inv, pixel_coords)
    camera_coords = camera_coords * depth.unsqueeze(-1)
    
    camera_coords_homo = torch.cat([
        camera_coords, 
        torch.ones(*camera_coords.shape[:-1], 1, device=device, dtype=camera_coords.dtype)
    ], dim=-1)
    
    c2w = torch.inverse(extrinsics)
    world_coords = torch.einsum('bsij,bshwj->bshwi', c2w, camera_coords_homo)
    world_coords = world_coords[..., :3]
    
    if valid_mask is None:
        valid_mask = (depth > 0) & torch.isfinite(depth)
    
    points = world_coords.reshape(-1, 3)
    valid = valid_mask.reshape(-1).bool()  # Ensure bool type for indexing
    points = points[valid]
    
    colors = None
    if rgb is not None:
        if rgb.shape[-1] != 3:
            rgb = rgb.permute(0, 1, 3, 4, 2)
        rgb_flat = rgb.reshape(-1, 3)
        colors = rgb_flat[valid]
        if colors.numel() > 0 and colors.max() > 1.0:
            colors = colors / 255.0
    
    if len(points) > max_points:
        indices = torch.randperm(len(points), device=device)[:max_points]
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
    
    return {
        'points': _tensor_to_numpy_safe(points),
        'colors': _tensor_to_numpy_safe(colors) if colors is not None else None,
    }


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, save_path: str):
    """Save point cloud to PLY file."""
    n_points = len(points)
    
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    
    with open(save_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {n_points}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        if colors is not None:
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
        f.write('end_header\n')
        
        for i in range(n_points):
            if colors is not None:
                f.write(f'{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} '
                       f'{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n')
            else:
                f.write(f'{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n')
    
    logger.info(f"Saved point cloud with {n_points} points to {save_path}")


def visualize_point_cloud_2d(
    points: np.ndarray,
    colors: np.ndarray = None,
    save_path: str = None,
    title: str = 'Point Cloud Projection',
) -> np.ndarray:
    """Visualize point cloud as 2D projections."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if colors is not None:
        if colors.max() > 1.0:
            colors = colors / 255.0
        point_colors = colors
    else:
        y_normalized = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-8)
        point_colors = plt.cm.viridis(y_normalized)[:, :3]
    
    max_vis_points = 50000
    if len(points) > max_vis_points:
        indices = np.random.choice(len(points), max_vis_points, replace=False)
        vis_points = points[indices]
        vis_colors = point_colors[indices]
    else:
        vis_points = points
        vis_colors = point_colors
    
    axes[0].scatter(vis_points[:, 0], vis_points[:, 2], c=vis_colors, s=0.5, alpha=0.5)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')
    axes[0].set_title('Top View (X-Z)')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(vis_points[:, 0], -vis_points[:, 1], c=vis_colors, s=0.5, alpha=0.5)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('-Y')
    axes[1].set_title('Front View (X-Y)')
    axes[1].set_aspect('equal')
    
    axes[2].scatter(vis_points[:, 2], -vis_points[:, 1], c=vis_colors, s=0.5, alpha=0.5)
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('-Y')
    axes[2].set_title('Side View (Z-Y)')
    axes[2].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved point cloud visualization to {save_path}")
    
    # Convert figure to numpy array (compatible with newer matplotlib versions)
    fig.canvas.draw()
    try:
        # Try newer matplotlib API first
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        # Fallback for older matplotlib versions
        try:
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Another fallback using PIL
            from PIL import Image
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            img = np.array(Image.open(buf))[:, :, :3]
    
    plt.close()
    return img


def generate_point_cloud_from_da3_output(
    pred_depth: torch.Tensor,
    pred_ray: torch.Tensor,
    rgb: torch.Tensor,
    gt_intrinsics: torch.Tensor = None,
    gt_extrinsics: torch.Tensor = None,
    valid_mask: torch.Tensor = None,
    output_dir: str = None,
    use_gt_camera: bool = False,
    max_points: int = 100000,
) -> dict:
    """
    Generate point cloud from DA3 output (depth + ray map).
    
    This function:
    1. Extracts camera parameters from ray map using camray_to_caminfo
    2. Generates point cloud using depth + camera parameters
    3. Optionally compares with GT camera parameters
    
    Args:
        pred_depth: Predicted depth [B, S, 1, H, W] or [B, S, H, W]
        pred_ray: Predicted ray map [B, S, H, W, 6] or [B, S, 6, H, W]
        rgb: RGB image [B, S, H, W, 3] or [B, S, 3, H, W]
        gt_intrinsics: Optional GT intrinsics [B, S, 3, 3]
        gt_extrinsics: Optional GT extrinsics [B, S, 4, 4]
        valid_mask: Optional valid depth mask [B, S, H, W]
        output_dir: Optional directory to save outputs
        use_gt_camera: If True, use GT camera params instead of extracted
        max_points: Maximum number of points
        
    Returns:
        Dictionary containing point cloud data and camera parameters
    """
    if pred_depth.dim() == 5:
        depth = pred_depth.squeeze(2)
    else:
        depth = pred_depth
    
    B, S, H, W = depth.shape
    
    if pred_ray.shape[-1] != 6:
        pred_ray = pred_ray.permute(0, 1, 3, 4, 2)
    
    if rgb.shape[-1] != 3:
        rgb = rgb.permute(0, 1, 3, 4, 2)
    
    results = {}
    
    logger.info("Extracting camera parameters from ray map...")
    cam_params = camray_to_camera_params(pred_ray, H, W)
    
    results['extracted_camera_params'] = {
        'R': _tensor_to_numpy_safe(cam_params['R']),
        'T': _tensor_to_numpy_safe(cam_params['T']),
        'focal_lengths': _tensor_to_numpy_safe(cam_params['focal_lengths']),
        'principal_points': _tensor_to_numpy_safe(cam_params['principal_points']),
    }
    
    logger.info(f"  Extracted R shape: {cam_params['R'].shape}")
    logger.info(f"  Extracted T: {_tensor_to_numpy_safe(cam_params['T'][0, 0])}")
    logger.info(f"  Extracted focal lengths: {_tensor_to_numpy_safe(cam_params['focal_lengths'][0, 0])}")
    
    if use_gt_camera and gt_intrinsics is not None and gt_extrinsics is not None:
        logger.info("Using GT camera parameters for point cloud generation")
        intrinsics = gt_intrinsics
        extrinsics = gt_extrinsics
        camera_source = 'gt'
    else:
        logger.info("Using extracted camera parameters for point cloud generation")
        intrinsics = cam_params['intrinsics']
        extrinsics = cam_params['extrinsics']
        camera_source = 'extracted'
    
    logger.info("Generating point cloud...")
    pc_data = depth_to_point_cloud(
        depth=depth,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        rgb=rgb,
        valid_mask=valid_mask,
        max_points=max_points,
    )
    
    results['points'] = pc_data['points']
    results['colors'] = pc_data['colors']
    results['camera_source'] = camera_source
    results['num_points'] = len(pc_data['points'])
    
    logger.info(f"  Generated {results['num_points']} points")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        ply_path = os.path.join(output_dir, f'point_cloud_{camera_source}.ply')
        save_point_cloud_ply(pc_data['points'], pc_data['colors'], ply_path)
        results['ply_path'] = ply_path
        
        vis_path = os.path.join(output_dir, f'point_cloud_{camera_source}_2d.png')
        vis_img = visualize_point_cloud_2d(
            pc_data['points'], 
            pc_data['colors'],
            save_path=vis_path,
            title=f'Point Cloud ({camera_source} camera params)'
        )
        results['visualization'] = vis_img
        results['vis_path'] = vis_path
    
    if not use_gt_camera and gt_intrinsics is not None and gt_extrinsics is not None:
        logger.info("Also generating point cloud with GT camera for comparison...")
        gt_pc_data = depth_to_point_cloud(
            depth=depth,
            intrinsics=gt_intrinsics,
            extrinsics=gt_extrinsics,
            rgb=rgb,
            valid_mask=valid_mask,
            max_points=max_points,
        )
        
        results['gt_points'] = gt_pc_data['points']
        results['gt_colors'] = gt_pc_data['colors']
        
        if output_dir:
            gt_ply_path = os.path.join(output_dir, 'point_cloud_gt.ply')
            save_point_cloud_ply(gt_pc_data['points'], gt_pc_data['colors'], gt_ply_path)
            results['gt_ply_path'] = gt_ply_path
            
            gt_vis_path = os.path.join(output_dir, 'point_cloud_gt_2d.png')
            visualize_point_cloud_2d(
                gt_pc_data['points'],
                gt_pc_data['colors'],
                save_path=gt_vis_path,
                title='Point Cloud (GT camera params)'
            )
            results['gt_vis_path'] = gt_vis_path
    
    return results
