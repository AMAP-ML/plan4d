"""Loss functions for GeometryAdapter VAE training.

This module implements the loss functions for training the GeometryAdapter VAE:
- Token distillation loss (L_token): MSE between original and reconstructed tokens
- Geometry reconstruction loss (L_geo): L1 loss for depth and camera-ray
- KL divergence loss (L_kl): Regularization for VAE latent space

Updates:
- Added `align_depth_least_squares()` for scale-invariant depth comparison
- Added `use_scale_invariant_depth` option in GeometryReconstructionLoss
- Added `supervise_ray_origin` option to control ray origin supervision
- Added `CyclicalKLScheduler` for KL weight annealing
"""

from contextlib import nullcontext

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Utility Functions for Global Scale Normalization
# ============================================================================

def calculate_adaptive_weight(
    target_loss: torch.Tensor,
    current_loss: torch.Tensor,
) -> float:
    """
    Scale current_loss by 10^n so its weighted magnitude matches target_loss.

    This is used for KL loss only when adaptive KL weighting is enabled.
    """
    target_val = float(target_loss.detach().cpu())
    current_val = float(current_loss.detach().cpu())

    if current_val > target_val:
        if current_val == 0 or target_val == 0:
            return 1e-2

        ratio = target_val / current_val
        n = math.floor(math.log10(abs(ratio)))
        return 10 ** n

    return 1.0


def _ensure_da3_src_importable() -> None:
    plan4d_root = Path(__file__).resolve().parents[1]
    da3_src_root = plan4d_root / "Depth-Anything-3" / "src"
    for candidate in (plan4d_root, da3_src_root):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _import_official_da3_camera_utils():
    _ensure_da3_src_importable()
    from depth_anything_3.utils.geometry import affine_inverse
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

    return affine_inverse, get_extrinsic_from_camray


def _import_official_da3_camera_token_utils():
    _ensure_da3_src_importable()
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse

    return pose_encoding_to_extri_intri, affine_inverse


def normalize_extrinsics_like_official_da3_api(
    extrinsics: torch.Tensor,
    eps: float = 1e-1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize world-to-camera extrinsics following DA3 API semantics.

    Returns:
        normalized_extrinsics: [B, T, 4, 4]
        pose_scale: [B] median c2w translation norm used for normalization
    """
    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            "Expected extrinsics with shape [B, T, 4, 4], "
            f"got {tuple(extrinsics.shape)}"
        )

    first_inverse = torch.linalg.inv(extrinsics[:, :1].float())
    normalized = torch.matmul(extrinsics.float(), first_inverse)
    c2ws = torch.linalg.inv(normalized)
    translations = c2ws[..., :3, 3]
    pose_scale = torch.median(translations.norm(dim=-1), dim=1).values.clamp(min=eps)

    normalized = normalized.to(dtype=extrinsics.dtype)
    normalized[..., :3, 3] = normalized[..., :3, 3] / pose_scale.to(
        device=normalized.device,
        dtype=normalized.dtype,
    ).view(-1, 1, 1)
    return normalized, pose_scale.to(device=extrinsics.device, dtype=extrinsics.dtype)


def generate_ray_map_batch(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Batched GT camray builder with DA3 dataset semantics:
    [direction(3), origin=center_world(3)] in world coordinates.
    """
    if intrinsics.dim() != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [B, T, 3, 3], got {tuple(intrinsics.shape)}")
    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics [B, T, 4, 4], got {tuple(extrinsics.shape)}")

    batch_size, num_frames = intrinsics.shape[:2]
    device = intrinsics.device
    dtype = intrinsics.dtype

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=0)
    pixels = pixels.view(1, 1, 3, height * width).expand(batch_size, num_frames, 3, height * width)

    k_inv = torch.linalg.inv(intrinsics.float()).to(dtype=dtype)
    directions_cam = torch.matmul(k_inv, pixels)

    rotation_w2c = extrinsics[..., :3, :3].to(dtype=dtype)
    translation_w2c = extrinsics[..., :3, 3].to(dtype=dtype)
    rotation_c2w = rotation_w2c.transpose(-2, -1)
    directions_world = torch.matmul(rotation_c2w, directions_cam)
    origin_world = -torch.matmul(rotation_c2w, translation_w2c.unsqueeze(-1)).squeeze(-1)

    directions_world = directions_world.view(batch_size, num_frames, 3, height, width)
    origins_world = origin_world.view(batch_size, num_frames, 3, 1, 1).expand(
        batch_size, num_frames, 3, height, width
    )
    return torch.cat([directions_world, origins_world], dim=2)


def compute_point_map_from_depth_and_ray(
    depth: torch.Tensor,
    ray: torch.Tensor,
) -> torch.Tensor:
    if depth.dim() == 4:
        depth = depth.unsqueeze(2)
    if depth.dim() != 5 or ray.dim() != 5:
        raise ValueError(f"Expected depth/ray to be 5D, got {tuple(depth.shape)} and {tuple(ray.shape)}")
    if ray.shape[2] < 6:
        raise ValueError(f"Expected ray with 6 channels, got {tuple(ray.shape)}")
    return ray[:, :, 3:, :, :] + depth * ray[:, :, :3, :, :]


def camera_center_from_w2c_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """Convert world-to-camera extrinsics to world-space camera centers."""
    if extrinsics.dim() != 4:
        raise ValueError(f"Expected extrinsics [B, T, 3/4, 4], got {tuple(extrinsics.shape)}")
    if extrinsics.shape[-2] < 3 or extrinsics.shape[-1] < 4:
        raise ValueError(f"Expected extrinsics [..., 3/4, 4], got {tuple(extrinsics.shape)}")

    rotation_w2c = extrinsics[..., :3, :3]
    translation_w2c = extrinsics[..., :3, 3]
    return -torch.einsum("...ji,...j->...i", rotation_w2c, translation_w2c)


def unproject_depth_map_with_camera(
    depth: torch.Tensor,
    *,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Dense world-space unprojection from depth and pinhole camera parameters."""
    if depth.dim() == 4:
        depth = depth.unsqueeze(2)
    if depth.dim() != 5 or depth.shape[2] != 1:
        raise ValueError(f"depth must be [B, T, 1, H, W], got {tuple(depth.shape)}")
    if intrinsics.dim() != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be [B, T, 3, 3], got {tuple(intrinsics.shape)}")
    if extrinsics.dim() != 4 or extrinsics.shape[-1] < 4 or extrinsics.shape[-2] < 3:
        raise ValueError(f"extrinsics must be [B, T, 3/4, 4], got {tuple(extrinsics.shape)}")

    batch_size, num_frames, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)  # [H, W, 3]

    k_inv = torch.linalg.inv(intrinsics.float()).to(device=device, dtype=dtype)
    directions_cam = torch.einsum("btij,hwj->bthwi", k_inv, pixels)
    points_cam = depth[:, :, 0, :, :][..., None] * directions_cam

    rotation_c2w = extrinsics[..., :3, :3].transpose(-1, -2).to(device=device, dtype=dtype)
    camera_center_world = camera_center_from_w2c_extrinsics(extrinsics).to(device=device, dtype=dtype)
    camera_center_world = camera_center_world[:, :, None, None, :]
    points_world = torch.einsum("btij,bthwj->bthwi", rotation_c2w, points_cam) + camera_center_world
    return points_world.permute(0, 1, 4, 2, 3).contiguous()


def compute_common_gt_scale_from_point_map(
    point_map: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if point_map.dim() != 5 or point_map.shape[2] != 3:
        raise ValueError(f"Expected point_map [B, T, 3, H, W], got {tuple(point_map.shape)}")

    point_norm = torch.linalg.norm(point_map, dim=2)
    if valid_mask is None:
        valid = torch.ones_like(point_norm, dtype=point_norm.dtype)
    else:
        if valid_mask.dim() == 5:
            valid = valid_mask.squeeze(2).to(dtype=point_norm.dtype)
        elif valid_mask.dim() == 4:
            valid = valid_mask.to(dtype=point_norm.dtype)
        else:
            raise ValueError(f"Unexpected valid_mask shape: {tuple(valid_mask.shape)}")
    denom = valid.sum(dim=(1, 2, 3)).clamp(min=1.0)
    scale = (point_norm * valid).sum(dim=(1, 2, 3)) / denom
    return scale.clamp(min=eps)


def reduce_loss_tensor_per_batch(
    loss: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    if loss.dim() < 1:
        raise ValueError(f"loss must have a batch dimension, got {tuple(loss.shape)}")

    batch_size = loss.shape[0]
    if valid_mask is None:
        weight = torch.ones_like(loss, dtype=loss.dtype)
    else:
        weight = valid_mask.to(device=loss.device, dtype=loss.dtype)
        if weight.shape != loss.shape:
            weight = weight.expand_as(loss)

    weighted_loss = loss * weight
    weighted_loss_flat = weighted_loss.reshape(batch_size, -1)
    weight_flat = weight.reshape(batch_size, -1)

    if reduction == "mean":
        return weighted_loss_flat.sum(dim=1) / weight_flat.sum(dim=1).clamp(min=1.0)
    if reduction == "sum":
        return weighted_loss_flat.sum(dim=1)
    if reduction == "none":
        return weighted_loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def reduce_scene_scale_reweighted_loss(
    loss_per_sample: torch.Tensor,
    scene_scale: torch.Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    if loss_per_sample.dim() != 1:
        raise ValueError(f"loss_per_sample must be [B], got {tuple(loss_per_sample.shape)}")

    scale = scene_scale.detach().to(
        device=loss_per_sample.device,
        dtype=loss_per_sample.dtype,
    ).clamp(min=eps)
    scaled_loss = loss_per_sample / scale

    if reduction == "mean":
        return scaled_loss.mean()
    if reduction == "sum":
        return scaled_loss.sum()
    if reduction == "none":
        return scaled_loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def normalize_intrinsics_params(
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    if intrinsics.dim() != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics [B, T, 3, 3], got {tuple(intrinsics.shape)}")
    normalized = torch.stack(
        [
            intrinsics[..., 0, 0] * (2.0 / width),
            intrinsics[..., 1, 1] * (2.0 / height),
            intrinsics[..., 0, 2] * (2.0 / width),
            intrinsics[..., 1, 2] * (2.0 / height),
        ],
        dim=-1,
    )
    return normalized


def decode_official_camera_from_ray(
    ray: torch.Tensor,
    ray_conf: Optional[torch.Tensor],
    *,
    image_height: int,
    image_width: int,
    training: bool = False,
) -> dict[str, torch.Tensor]:
    affine_inverse, get_extrinsic_from_camray = _import_official_da3_camera_utils()

    if ray.dim() != 5 or ray.shape[2] < 6:
        raise ValueError(f"Expected ray [B, T, 6, H, W], got {tuple(ray.shape)}")

    ray_hw_last = ray[:, :, :6].permute(0, 1, 3, 4, 2).contiguous()
    if ray_conf is None:
        ray_conf_hw = None
    elif ray_conf.dim() == 5 and ray_conf.shape[2] == 1:
        ray_conf_hw = ray_conf.squeeze(2)
    elif ray_conf.dim() == 4:
        ray_conf_hw = ray_conf
    else:
        raise ValueError(f"Unsupported ray_conf shape: {tuple(ray_conf.shape)}")

    pred_extrinsic_c2w, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
        ray_hw_last,
        ray_conf_hw,
        ray_hw_last.shape[-3],
        ray_hw_last.shape[-2],
        training=training,
    )
    pred_extrinsic_w2c = affine_inverse(pred_extrinsic_c2w)[:, :, :3, :]
    pred_intrinsics = torch.eye(
        3,
        device=pred_extrinsic_w2c.device,
        dtype=pred_extrinsic_w2c.dtype,
    )[None, None].repeat(pred_extrinsic_w2c.shape[0], pred_extrinsic_w2c.shape[1], 1, 1)
    pred_intrinsics[:, :, 0, 0] = pred_focal_lengths[:, :, 0] * image_width * 0.5
    pred_intrinsics[:, :, 1, 1] = pred_focal_lengths[:, :, 1] * image_height * 0.5
    pred_intrinsics[:, :, 0, 2] = pred_principal_points[:, :, 0] * image_width * 0.5
    pred_intrinsics[:, :, 1, 2] = pred_principal_points[:, :, 1] * image_height * 0.5
    return {
        "extrinsics": pred_extrinsic_w2c,
        "intrinsics": pred_intrinsics,
        "focal_norm": pred_focal_lengths,
        "principal_norm": pred_principal_points,
    }


def decode_official_camera_from_token(
    camera_tokens: torch.Tensor,
    camera_decoder: nn.Module,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    pose_encoding_to_extri_intri, affine_inverse = _import_official_da3_camera_token_utils()

    if camera_tokens.dim() != 3:
        raise ValueError(
            "Expected camera_tokens with shape [B, T, C], "
            f"got {tuple(camera_tokens.shape)}"
        )

    autocast_context = (
        torch.autocast(device_type=camera_tokens.device.type, enabled=False)
        if camera_tokens.device.type != "cpu"
        else nullcontext()
    )
    with autocast_context:
        pose_encoding = camera_decoder(camera_tokens.float())
        c2w_extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding,
            (image_height, image_width),
        )
        extrinsics = affine_inverse(c2w_extrinsics)[..., :3, :]

    return {
        "pose_encoding": pose_encoding,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def align_pred_ray_to_gt_loss_space(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align predicted rays into GT world gauge before supervision.

    This mirrors the `gt_aligned` diagnostic used in the debug script:
    a rigid alignment is estimated from per-frame camera centers and viewing
    directions, then applied to the predicted ray field before computing loss.
    """
    from plan4d.point_cloud_utils import align_pred_ray_to_gt

    if valid_mask is None:
        aligned_pred_ray, _ = align_pred_ray_to_gt(pred_ray=pred_ray, gt_ray=gt_ray)
        return aligned_pred_ray

    if valid_mask.dim() == 5:
        frame_mask = valid_mask.squeeze(2)
    elif valid_mask.dim() == 4:
        frame_mask = valid_mask
    else:
        raise ValueError(f"valid_mask must be [B, T, 1, H, W] or [B, T, H, W], got {tuple(valid_mask.shape)}")

    frame_mask = frame_mask.reshape(frame_mask.shape[0], frame_mask.shape[1], -1).sum(dim=-1) > 0
    if bool(frame_mask.all()):
        aligned_pred_ray, _ = align_pred_ray_to_gt(pred_ray=pred_ray, gt_ray=gt_ray)
        return aligned_pred_ray

    aligned_pred_ray = pred_ray.clone()
    for batch_idx in range(pred_ray.shape[0]):
        valid_indices = torch.nonzero(frame_mask[batch_idx], as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            continue
        aligned_valid, _ = align_pred_ray_to_gt(
            pred_ray=pred_ray[batch_idx : batch_idx + 1].index_select(1, valid_indices),
            gt_ray=gt_ray[batch_idx : batch_idx + 1].index_select(1, valid_indices),
        )
        aligned_pred_ray[batch_idx, valid_indices] = aligned_valid[0]
    return aligned_pred_ray


def point_cloud_l1_after_scale_shift_alignment(
    pred_points_world: torch.Tensor,
    target_points_world: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Dense point-cloud L1 after per-frame scale+shift alignment.

    The forward value matches the debug metric
    `point_cloud_loss_*_dense_after_scale_shift`, while scale/shift are detached
    so gradients optimize the aligned geometry itself rather than the closed-form
    alignment parameters.
    """
    if pred_points_world.dim() != 5 or target_points_world.dim() != 5:
        raise ValueError(
            "pred_points_world/target_points_world must be [B, T, 3, H, W], got "
            f"{tuple(pred_points_world.shape)} and {tuple(target_points_world.shape)}"
        )
    if pred_points_world.shape != target_points_world.shape:
        raise ValueError(
            "pred_points_world and target_points_world must share shape, got "
            f"{tuple(pred_points_world.shape)} and {tuple(target_points_world.shape)}"
        )

    if valid_mask.dim() == 5:
        valid_mask = valid_mask.squeeze(2)
    elif valid_mask.dim() != 4:
        raise ValueError(f"valid_mask must be [B, T, 1, H, W] or [B, T, H, W], got {tuple(valid_mask.shape)}")

    pred_flat = pred_points_world.permute(0, 1, 3, 4, 2).reshape(pred_points_world.shape[0], pred_points_world.shape[1], -1, 3)
    target_flat = target_points_world.permute(0, 1, 3, 4, 2).reshape(target_points_world.shape[0], target_points_world.shape[1], -1, 3)
    valid_flat = valid_mask.reshape(valid_mask.shape[0], valid_mask.shape[1], -1)

    finite_mask = torch.isfinite(pred_flat).all(dim=-1) & torch.isfinite(target_flat).all(dim=-1)
    valid_flat = (valid_flat > 0.5) & finite_mask
    valid_3d = valid_flat.unsqueeze(-1).to(dtype=pred_flat.dtype)

    count = valid_flat.sum(dim=-1).to(dtype=pred_flat.dtype)
    valid_frames = count > 0
    count_safe = count.clamp(min=1.0)

    pred_mean = (pred_flat * valid_3d).sum(dim=2, keepdim=True) / count_safe[..., None, None]
    target_mean = (target_flat * valid_3d).sum(dim=2, keepdim=True) / count_safe[..., None, None]
    pred_centered = (pred_flat - pred_mean) * valid_3d
    target_centered = (target_flat - target_mean) * valid_3d

    covariance = (pred_centered * target_centered).sum(dim=(2, 3))
    variance = pred_centered.square().sum(dim=(2, 3))
    can_align = valid_frames & (variance > eps)

    scale_align = torch.ones_like(count_safe)
    scale_est = (covariance + eps) / (variance + eps)
    scale_align = torch.where(can_align, scale_est.clamp(0.01, 100.0), scale_align).detach()

    shift_align = (target_mean - scale_align[..., None, None] * pred_mean).detach()
    aligned_pred = scale_align[..., None, None] * pred_flat + shift_align
    aligned_error = (aligned_pred - target_flat).abs() * valid_3d

    if reduction == "mean":
        denom = valid_3d.sum().clamp(min=1.0)
        return aligned_error.sum() / denom
    if reduction == "sum":
        return aligned_error.sum()
    if reduction == "none":
        return aligned_error.reshape(
            pred_points_world.shape[0],
            pred_points_world.shape[1],
            pred_points_world.shape[-2],
            pred_points_world.shape[-1],
            3,
        ).permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported reduction: {reduction}")


def symmetric_dense_point_set_distance(
    pred_points_world: torch.Tensor,
    target_points_world: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_points: int = 2048,
    reduction: str = "mean",
) -> torch.Tensor:
    """Approximate recon_unposed with dense point-set bidirectional NN distance.

    Official `recon_unposed` does TSDF fusion and then nearest-neighbor
    reconstruction metrics on fused point clouds. Training cannot afford full
    TSDF/Open3D in the loop, so we use the closest differentiable surrogate:
    after the official pose-scale alignment, flatten dense valid world points
    across the full sequence into point sets and compute symmetric nearest-
    neighbor distance.
    """
    if pred_points_world.dim() != 5 or target_points_world.dim() != 5:
        raise ValueError(
            "pred_points_world/target_points_world must be [B, T, 3, H, W], got "
            f"{tuple(pred_points_world.shape)} and {tuple(target_points_world.shape)}"
        )
    if pred_points_world.shape != target_points_world.shape:
        raise ValueError(
            "pred_points_world and target_points_world must share shape, got "
            f"{tuple(pred_points_world.shape)} and {tuple(target_points_world.shape)}"
        )
    if valid_mask.dim() == 5:
        valid_mask = valid_mask.squeeze(2)
    elif valid_mask.dim() != 4:
        raise ValueError(
            f"valid_mask must be [B, T, 1, H, W] or [B, T, H, W], got {tuple(valid_mask.shape)}"
        )

    batch_losses: list[torch.Tensor] = []
    for batch_idx in range(pred_points_world.shape[0]):
        pred_points_bt = pred_points_world[batch_idx].reshape(
            pred_points_world.shape[1],
            3,
            -1,
        )
        target_points_bt = target_points_world[batch_idx].reshape(
            target_points_world.shape[1],
            3,
            -1,
        )
        valid_bt = valid_mask[batch_idx].reshape(valid_mask.shape[1], -1) > 0.5

        pred_valid = valid_bt & torch.isfinite(pred_points_bt).all(dim=1)
        target_valid = valid_bt & torch.isfinite(target_points_bt).all(dim=1)
        pred_coords = pred_valid.nonzero(as_tuple=False)
        target_coords = target_valid.nonzero(as_tuple=False)

        if pred_coords.numel() == 0 or target_coords.numel() == 0:
            batch_losses.append(pred_points_world.new_zeros(()))
            continue

        if pred_coords.shape[0] > max_points:
            pred_take = evenly_spaced_index_sample(
                pred_coords.shape[0],
                max_points,
                device=pred_coords.device,
            )
            pred_coords = pred_coords.index_select(0, pred_take)
        if target_coords.shape[0] > max_points:
            target_take = evenly_spaced_index_sample(
                target_coords.shape[0],
                max_points,
                device=target_coords.device,
            )
            target_coords = target_coords.index_select(0, target_take)

        pred_points = pred_points_bt[pred_coords[:, 0], :, pred_coords[:, 1]]
        target_points = target_points_bt[target_coords[:, 0], :, target_coords[:, 1]]

        point_dists = torch.cdist(pred_points, target_points, p=2)
        pred_to_target = point_dists.min(dim=1).values
        target_to_pred = point_dists.min(dim=0).values
        batch_losses.append(0.5 * (pred_to_target.mean() + target_to_pred.mean()))

    stacked = torch.stack(batch_losses)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    if reduction == "none":
        return stacked
    raise ValueError(f"Unsupported reduction: {reduction}")


def evenly_spaced_index_sample(
    num_items: int,
    num_samples: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return monotonic integer sample indices without float rounding overflow."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if num_items <= num_samples:
        return torch.arange(num_items, device=device, dtype=torch.long)
    if num_samples == 1:
        return torch.zeros(1, device=device, dtype=torch.long)

    positions = torch.arange(num_samples, device=device, dtype=torch.long)
    return torch.div(
        positions * (num_items - 1),
        num_samples - 1,
        rounding_mode="floor",
    )


def pad_extrinsics_to_4x4(extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.dim() != 4 or extrinsics.shape[-1] != 4 or extrinsics.shape[-2] not in (3, 4):
        raise ValueError(f"Expected extrinsics [B, T, 3/4, 4], got {tuple(extrinsics.shape)}")
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padded = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).view(1, 1, 4, 4).repeat(
        extrinsics.shape[0], extrinsics.shape[1], 1, 1
    )
    padded[..., :3, :] = extrinsics
    return padded


def align_extrinsics_to_first_camera_torch(extrinsics: torch.Tensor) -> torch.Tensor:
    ext4 = pad_extrinsics_to_4x4(extrinsics)
    first_inv = torch.linalg.inv(ext4[:, :1].float())
    aligned = torch.matmul(ext4.float(), first_inv).to(dtype=ext4.dtype)
    return aligned


def align_extrinsics_umeyama_transform_like_official(
    target_extrinsics: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    *,
    ransac: bool = True,
    random_state: int = 42,
) -> dict[str, torch.Tensor]:
    _ensure_da3_src_importable()
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    target_ext4 = pad_extrinsics_to_4x4(target_extrinsics)
    pred_ext4 = pad_extrinsics_to_4x4(pred_extrinsics)

    aligned_list = []
    rotation_list = []
    translation_list = []
    scale_list = []
    for batch_idx in range(target_ext4.shape[0]):
        target_np = target_ext4[batch_idx].detach().float().cpu().numpy().astype(np.float64)
        pred_np = pred_ext4[batch_idx].detach().float().cpu().numpy().astype(np.float64)
        try:
            rotation, translation, scale, aligned_np = align_poses_umeyama(
                target_np,
                pred_np,
                return_aligned=True,
                ransac=ransac,
                random_state=random_state,
            )
        except Exception:
            rotation = np.eye(3, dtype=np.float64)
            translation = np.zeros(3, dtype=np.float64)
            scale = 1.0
            aligned_np = pred_np
        aligned_list.append(
            torch.from_numpy(np.asarray(aligned_np)).to(device=pred_extrinsics.device, dtype=pred_extrinsics.dtype)
        )
        rotation_list.append(
            torch.from_numpy(np.asarray(rotation)).to(device=pred_extrinsics.device, dtype=pred_extrinsics.dtype)
        )
        translation_list.append(
            torch.from_numpy(np.asarray(translation)).to(device=pred_extrinsics.device, dtype=pred_extrinsics.dtype)
        )
        scale_list.append(scale)

    return {
        "aligned_extrinsics": torch.stack(aligned_list, dim=0),
        "rotation": torch.stack(rotation_list, dim=0),
        "translation": torch.stack(translation_list, dim=0),
        "scale": torch.tensor(scale_list, device=pred_extrinsics.device, dtype=pred_extrinsics.dtype),
    }


def align_extrinsics_umeyama_like_official(
    target_extrinsics: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    *,
    ransac: bool = True,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    alignment = align_extrinsics_umeyama_transform_like_official(
        target_extrinsics,
        pred_extrinsics,
        ransac=ransac,
        random_state=random_state,
    )
    return alignment["aligned_extrinsics"], alignment["scale"]


def apply_umeyama_alignment_to_ray(
    ray: torch.Tensor,
    *,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if ray.dim() != 5 or ray.shape[2] < 6:
        raise ValueError(f"Expected ray [B, T, 6, H, W], got {tuple(ray.shape)}")

    batch_size, num_frames, _, height, width = ray.shape
    rotation = rotation.to(device=ray.device, dtype=ray.dtype)
    translation = translation.to(device=ray.device, dtype=ray.dtype)
    scale = scale.to(device=ray.device, dtype=ray.dtype)

    direction = ray[:, :, :3, :, :].reshape(batch_size, num_frames, 3, height * width)
    origin = ray[:, :, 3:6, :, :].reshape(batch_size, num_frames, 3, height * width)
    aligned_direction = torch.matmul(rotation[:, None], direction)
    aligned_origin = (
        scale.view(batch_size, 1, 1, 1) * torch.matmul(rotation[:, None], origin)
        + translation.view(batch_size, 1, 3, 1)
    )
    return torch.cat(
        [
            aligned_direction.reshape(batch_size, num_frames, 3, height, width),
            aligned_origin.reshape(batch_size, num_frames, 3, height, width),
        ],
        dim=2,
    )


def align_ray_to_official_camera_pose_space(
    pred_ray: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    pose_scale: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    ransac_view_thresh: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame_mask = _frame_valid_mask_from_dense_valid_mask(
        valid_mask,
        batch_size=pred_extrinsics.shape[0],
        num_frames=pred_extrinsics.shape[1],
        device=pred_extrinsics.device,
    )

    aligned_rays = []
    scales = []
    for batch_idx in range(pred_extrinsics.shape[0]):
        valid_indices = torch.nonzero(frame_mask[batch_idx], as_tuple=False).squeeze(-1)
        if valid_indices.numel() < 2:
            alignment = {
                "rotation": torch.eye(3, device=pred_extrinsics.device, dtype=pred_extrinsics.dtype).unsqueeze(0),
                "translation": pred_extrinsics.new_zeros(1, 3),
                "scale": pred_extrinsics.new_ones(1),
            }
        else:
            alignment = align_extrinsics_umeyama_transform_like_official(
                target_extrinsics=target_extrinsics[batch_idx : batch_idx + 1].index_select(1, valid_indices),
                pred_extrinsics=pred_extrinsics[batch_idx : batch_idx + 1].index_select(1, valid_indices),
                ransac=bool(valid_indices.numel() >= ransac_view_thresh),
            )
        aligned_world = apply_umeyama_alignment_to_ray(
            pred_ray[batch_idx : batch_idx + 1],
            rotation=alignment["rotation"],
            translation=alignment["translation"],
            scale=alignment["scale"],
        )
        aligned_pose = normalize_ray_like_official_da3_api(
            aligned_world,
            target_extrinsics[batch_idx : batch_idx + 1],
            pose_scale[batch_idx : batch_idx + 1],
        )
        aligned_rays.append(aligned_pose[0])
        scales.append(alignment["scale"][0])

    return torch.stack(aligned_rays, dim=0), torch.stack(scales, dim=0)


def _frame_valid_mask_from_dense_valid_mask(
    valid_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones(batch_size, num_frames, device=device, dtype=torch.bool)
    if valid_mask.dim() == 5:
        valid_mask = valid_mask.squeeze(2)
    elif valid_mask.dim() != 4:
        raise ValueError(
            f"valid_mask must be [B, T, 1, H, W] or [B, T, H, W], got {tuple(valid_mask.shape)}"
        )
    return valid_mask.reshape(batch_size, num_frames, -1).sum(dim=-1) > 0


def compute_depth_scale_from_official_camera_alignment(
    pred_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    ransac_view_thresh: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate the official DA3/bench scale that maps pred depth into GT depth.

    The alignment follows the bench semantics:
    1. anchor pred/gt trajectories to the first camera;
    2. solve Umeyama on the valid-view subset;
    3. use the returned Sim(3) scale for metric depth correction.
    """
    pred_ext4 = pad_extrinsics_to_4x4(pred_extrinsics)
    target_ext4 = pad_extrinsics_to_4x4(target_extrinsics)
    frame_mask = _frame_valid_mask_from_dense_valid_mask(
        valid_mask,
        batch_size=pred_ext4.shape[0],
        num_frames=pred_ext4.shape[1],
        device=pred_ext4.device,
    )

    scale_values: list[torch.Tensor] = []
    for batch_idx in range(pred_ext4.shape[0]):
        valid_indices = torch.nonzero(frame_mask[batch_idx], as_tuple=False).squeeze(-1)
        if valid_indices.numel() < 2:
            scale_values.append(pred_ext4.new_tensor(1.0))
            continue

        pred_valid = pred_ext4[batch_idx : batch_idx + 1].index_select(1, valid_indices)
        target_valid = target_ext4[batch_idx : batch_idx + 1].index_select(1, valid_indices)
        pred_valid = align_extrinsics_to_first_camera_torch(pred_valid)
        target_valid = align_extrinsics_to_first_camera_torch(target_valid)
        _, scale = align_extrinsics_umeyama_like_official(
            target_extrinsics=target_valid,
            pred_extrinsics=pred_valid,
            ransac=bool(valid_indices.numel() >= ransac_view_thresh),
        )
        scale_value = scale[0]
        if not torch.isfinite(scale_value) or float(scale_value.abs().item()) <= eps:
            scale_value = pred_ext4.new_tensor(1.0)
        scale_values.append(scale_value.clamp(min=eps))

    return torch.stack(scale_values, dim=0)


def align_depth_by_official_camera_scale(
    pred_depth: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    ransac_view_thresh: int = 10,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale predicted depth using the same camera-scale logic as official bench."""
    scale = compute_depth_scale_from_official_camera_alignment(
        pred_extrinsics=pred_extrinsics,
        target_extrinsics=target_extrinsics,
        valid_mask=valid_mask,
        ransac_view_thresh=ransac_view_thresh,
        eps=eps,
    ).detach()
    if pred_depth.dim() == 5:
        scale_view = scale.view(-1, 1, 1, 1, 1)
    elif pred_depth.dim() == 4:
        scale_view = scale.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected pred_depth shape: {tuple(pred_depth.shape)}")
    return pred_depth * scale_view, scale


def resolve_pred_camera_for_gt_geometry_alignment(
    *,
    pred_camera_intrinsics: Optional[torch.Tensor],
    pred_camera_extrinsics: Optional[torch.Tensor],
    pred_camera_tokens: Optional[torch.Tensor],
    camera_decoder: Optional[nn.Module],
    pred_camera_image_hw: Optional[Tuple[int, int]],
    target_image_hw: Tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Resolve GT-alignment camera from the DA3 camera-token branch only.

    Official DA3 / bench depth scaling is defined on the decoded camera branch,
    not on cameras re-estimated from the dense ray map. Therefore GT depth-scale
    alignment must consume either decoded camera-token outputs or raw camera
    tokens together with the DA3 camera decoder.
    """
    if pred_camera_intrinsics is not None or pred_camera_extrinsics is not None:
        if pred_camera_intrinsics is None or pred_camera_extrinsics is None:
            raise ValueError(
                "pred_camera_intrinsics and pred_camera_extrinsics must be provided together."
            )
        return {
            "intrinsics": pred_camera_intrinsics,
            "extrinsics": pred_camera_extrinsics,
        }

    if pred_camera_tokens is None:
        raise ValueError(
            "GT depth-scale alignment requires camera-token branch outputs: provide "
            "(pred_camera_intrinsics + pred_camera_extrinsics) or "
            "(pred_camera_tokens + camera_decoder). Ray-map camera decode is intentionally unsupported."
        )
    if camera_decoder is None:
        raise ValueError(
            "camera_decoder is required when pred_camera_tokens are provided for GT depth-scale alignment."
        )

    image_height, image_width = (
        pred_camera_image_hw
        if pred_camera_image_hw is not None
        else target_image_hw
    )
    return decode_official_camera_from_token(
        pred_camera_tokens,
        camera_decoder,
        image_height=int(image_height),
        image_width=int(image_width),
    )


def compute_translation_direction_loss(
    pred_translation: torch.Tensor,
    target_translation: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred_translation.shape != target_translation.shape:
        raise ValueError(
            f"pred_translation and target_translation must share shape, got {tuple(pred_translation.shape)} and {tuple(target_translation.shape)}"
        )
    pred_norm = torch.linalg.norm(pred_translation, dim=-1, keepdim=True)
    target_norm = torch.linalg.norm(target_translation, dim=-1, keepdim=True)
    valid = (pred_norm > eps) & (target_norm > eps)
    pred_unit = pred_translation / pred_norm.clamp(min=eps)
    target_unit = target_translation / target_norm.clamp(min=eps)
    loss = 1.0 - (pred_unit * target_unit).sum(dim=-1, keepdim=True).clamp(min=-1.0, max=1.0)

    if valid_mask is not None:
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        valid = valid & (valid_mask > 0.5)

    valid = valid.to(dtype=loss.dtype)
    if reduction == "mean":
        denom = valid.sum().clamp(min=1.0)
        return (loss * valid).sum() / denom
    if reduction == "sum":
        return (loss * valid).sum()
    if reduction == "none":
        return loss * valid
    raise ValueError(f"Unsupported reduction: {reduction}")


def normalize_ray_like_official_da3_api(
    ray: torch.Tensor,
    extrinsics: torch.Tensor,
    pose_scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Remove arbitrary GT world-gauge offsets using DA3's first-camera anchor.

    The provided GT ray already carries the dataset/world-coordinate semantics
    we want to supervise, so we keep its direction field unchanged up to the
    first-camera rotation, and only normalize the origin channels into DA3's
    anchored + pose-scale-normalized gauge.
    """
    if ray.dim() != 5 or ray.shape[2] < 6:
        raise ValueError(f"Expected ray [B, T, 6, H, W], got {tuple(ray.shape)}")
    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics [B, T, 4, 4], got {tuple(extrinsics.shape)}")

    batch_size, num_frames, _, height, width = ray.shape
    rotation_ref = extrinsics[:, :1, :3, :3].to(device=ray.device, dtype=ray.dtype)
    translation_ref = extrinsics[:, :1, :3, 3].to(device=ray.device, dtype=ray.dtype)
    rotation_ref = rotation_ref.expand(batch_size, num_frames, 3, 3)
    translation_ref = translation_ref.expand(batch_size, num_frames, 3)

    direction = ray[:, :, :3, :, :].reshape(batch_size, num_frames, 3, height * width)
    origin = ray[:, :, 3:6, :, :].reshape(batch_size, num_frames, 3, height * width)

    direction_anchored = torch.matmul(rotation_ref, direction)
    del pose_scale, eps
    origin_pose_normalized = torch.matmul(rotation_ref, origin) + translation_ref.unsqueeze(-1)

    return torch.cat(
        [
            direction_anchored.reshape(batch_size, num_frames, 3, height, width),
            origin_pose_normalized.reshape(batch_size, num_frames, 3, height, width),
        ],
        dim=2,
    )


def build_official_gt_geometry_targets(
    target_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    target_ray: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    if target_depth.dim() == 4:
        target_depth = target_depth.unsqueeze(2)
    if valid_mask.dim() == 4:
        valid_mask = valid_mask.unsqueeze(2)

    batch_size, num_frames, _, height, width = target_depth.shape
    normalized_extrinsics, pose_scale = normalize_extrinsics_like_official_da3_api(extrinsics)
    depth_pose_normalized = target_depth
    if target_ray is None:
        ray_pose_normalized = generate_ray_map_batch(
            intrinsics,
            normalized_extrinsics,
            height,
            width,
        )
    else:
        ray_pose_normalized = normalize_ray_like_official_da3_api(
            target_ray,
            extrinsics,
            pose_scale,
            eps=eps,
        )
    point_map_pose_normalized = compute_point_map_from_depth_and_ray(depth_pose_normalized, ray_pose_normalized)
    common_gt_scale = compute_common_gt_scale_from_point_map(
        point_map_pose_normalized,
        valid_mask=valid_mask,
        eps=eps,
    )
    common_gt_scale_view = common_gt_scale.view(batch_size, 1, 1, 1, 1)

    depth_common = depth_pose_normalized / common_gt_scale_view
    origin_common = ray_pose_normalized[:, :, 3:, :, :] / common_gt_scale_view
    ray_common = torch.cat([ray_pose_normalized[:, :, :3, :, :], origin_common], dim=2)
    point_map_common = point_map_pose_normalized / common_gt_scale_view

    return {
        "normalized_extrinsics": normalized_extrinsics,
        "pose_scale": pose_scale,
        "common_gt_scale": common_gt_scale,
        "target_depth_pose_normalized": depth_pose_normalized,
        "target_ray_pose_normalized": ray_pose_normalized,
        "target_depth_common": depth_common,
        "target_ray_common": ray_common,
        "target_point_map_common": point_map_common,
        "target_intrinsics_norm": normalize_intrinsics_params(intrinsics, height, width),
    }

def upsample_ray_with_renormalization(
    pred_ray: torch.Tensor,
    target_size: Tuple[int, int],
    mode: str = 'bilinear',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Upsample predicted ray map to target resolution.
    
    Direction vectors (first 3 channels) are NOT renormalized after interpolation
    because DA3's unnormalized directions encode intrinsic info (focal length in
    the vector magnitude). Bilinear interpolation of unnormalized vectors is valid.
    
    Args:
        pred_ray: Predicted ray map [B, T, 6, H_pred, W_pred] or [B, 6, H_pred, W_pred]
                  Format: [direction_x, direction_y, direction_z, origin_x, origin_y, origin_z]
        target_size: Target spatial size (H_gt, W_gt)
        mode: Interpolation mode ('bilinear' or 'nearest')
        eps: Small epsilon (unused, kept for API compatibility)
        
    Returns:
        Upsampled ray map [B, T, 6, H_gt, W_gt] or [B, 6, H_gt, W_gt]
    """
    has_time_dim = pred_ray.dim() == 5
    
    if has_time_dim:
        # [B, T, 6, H, W] -> [B*T, 6, H, W]
        B, T, C, H, W = pred_ray.shape
        pred_ray_flat = pred_ray.reshape(B * T, C, H, W)
    else:
        pred_ray_flat = pred_ray
    
    # Split into direction and origin
    direction = pred_ray_flat[:, :3, :, :]  # [B*T, 3, H, W]
    origin = pred_ray_flat[:, 3:, :, :]     # [B*T, 3, H, W]
    
    # Upsample both components
    direction_up = F.interpolate(
        direction, size=target_size, mode=mode, align_corners=False if mode == 'bilinear' else None
    )
    origin_up = F.interpolate(
        origin, size=target_size, mode=mode, align_corners=False if mode == 'bilinear' else None
    )
    
    # NOTE: Direction vectors are intentionally NOT renormalized after interpolation.
    # DA3's unnormalized directions encode intrinsic info (focal length in magnitude).
    # Bilinear interpolation of unnormalized vectors is geometrically valid.
    
    # Concatenate back
    ray_up = torch.cat([direction_up, origin_up], dim=1)  # [B*T, 6, H, W]
    
    if has_time_dim:
        ray_up = ray_up.reshape(B, T, C, target_size[0], target_size[1])
    
    return ray_up


def compute_scale_factor(
    depth: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute global scale normalization factor from depth and camera parameters.
    
    Following DA3 paper, the scale factor is computed as the average L2 norm
    of reprojected 3D points in world coordinates. This normalizes all distance-
    related quantities (depth, camera origin) to a common scale for training stability.
    
    Formula:
        P_cam = D(u,v) * K^{-1} @ [u, v, 1]^T
        P_world = R @ P_cam + t
        scale = mean(||P_world||_2) over valid pixels
    
    Args:
        depth: Depth map [B, T, H, W] or [B, 1, H, W]
        K: Camera intrinsics [B, T, 3, 3] or [B, 3, 3]
        R: Rotation matrix [B, T, 3, 3] or [B, 3, 3] (Camera-to-World)
        t: Translation vector [B, T, 3] or [B, 3] (Camera-to-World)
        valid_mask: Valid pixel mask [B, T, H, W] or [B, 1, H, W]
        
    Returns:
        scale_factor: [B] or [B, T] normalization scale factor
        
    Note:
        - If extrinsics are World-to-Camera, they will be automatically converted
        - Returns 1.0 for samples with no valid depth pixels
    """
    # Handle different input shapes
    if depth.dim() == 4 and depth.shape[1] == 1:
        # [B, 1, H, W] -> squeeze time dimension
        depth = depth.squeeze(1)  # [B, H, W]
        has_time_dim = False
    elif depth.dim() == 4:
        # [B, T, H, W]
        has_time_dim = True
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    
    B = depth.shape[0]
    if has_time_dim:
        T, H, W = depth.shape[1], depth.shape[2], depth.shape[3]
    else:
        H, W = depth.shape[1], depth.shape[2]
        T = 1
        depth = depth.unsqueeze(1)  # [B, 1, H, W]
    
    device = depth.device
    
    # Ensure K, R, t have time dimension
    if K.dim() == 3:  # [B, 3, 3]
        K = K.unsqueeze(1).expand(B, T, 3, 3)  # [B, T, 3, 3]
    if R.dim() == 3:  # [B, 3, 3]
        R = R.unsqueeze(1).expand(B, T, 3, 3)  # [B, T, 3, 3]
    if t.dim() == 2:  # [B, 3]
        t = t.unsqueeze(1).expand(B, T, 3)  # [B, T, 3]
    
    # Reshape for batch processing: [B*T, ...]
    depth_flat = depth.reshape(B * T, H, W)  # [B*T, H, W]
    K_flat = K.reshape(B * T, 3, 3).float()  # [B*T, 3, 3] - ensure float type
    R_flat = R.reshape(B * T, 3, 3).float()  # [B*T, 3, 3] - ensure float type
    t_flat = t.reshape(B * T, 3).float()  # [B*T, 3] - ensure float type
    
    if valid_mask is not None:
        if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)  # [B, H, W]
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1).expand(B, T, H, W)  # [B, T, H, W]
        valid_mask_flat = valid_mask.reshape(B * T, H, W)  # [B*T, H, W]
    else:
        valid_mask_flat = (depth_flat > 0).float()  # [B*T, H, W]
    
    # Generate pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    # Homogeneous coordinates [B*T, 3, H*W]
    pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=0)  # [3, H, W]
    pixels = pixels.view(3, -1).unsqueeze(0).expand(B * T, 3, H * W)  # [B*T, 3, H*W]
    
    # Back-project to camera coordinates: P_cam = D * K^{-1} @ p
    K_inv = torch.inverse(K_flat)  # [B*T, 3, 3]
    ray_dirs_cam = torch.bmm(K_inv, pixels)  # [B*T, 3, H*W]
    
    depth_flat_expanded = depth_flat.view(B * T, 1, -1)  # [B*T, 1, H*W]
    P_cam = depth_flat_expanded * ray_dirs_cam  # [B*T, 3, H*W]
    
    # Transform to world coordinates: P_world = R @ P_cam + t
    P_world = torch.bmm(R_flat, P_cam) + t_flat.unsqueeze(2)  # [B*T, 3, H*W]
    
    # Compute L2 norms
    point_norms = torch.norm(P_world, p=2, dim=1)  # [B*T, H*W]
    
    # Compute scale factor per sample
    valid_mask_flat_expanded = valid_mask_flat.view(B * T, -1) > 0.5  # [B*T, H*W]
    
    # Vectorized mean of valid norms (no Python for loop)
    masked_norms = point_norms * valid_mask_flat_expanded.float()
    count = valid_mask_flat_expanded.sum(dim=1)  # [B*T]
    has_valid = count > 0
    safe_count = count.clamp(min=1)
    scale_factors = masked_norms.sum(dim=1) / safe_count  # [B*T]
    # Fallback to 1.0 for samples with no valid pixels
    scale_factors = torch.where(has_valid, scale_factors, torch.ones_like(scale_factors))
    
    # Reshape to [B, T] or [B]
    if has_time_dim and T > 1:
        scale_factors = scale_factors.reshape(B, T)  # [B, T]
        # Average over time dimension for a single scale per batch
        scale_factors = scale_factors.mean(dim=1)  # [B]
    elif not has_time_dim:
        scale_factors = scale_factors.reshape(B)  # [B]
    
    return scale_factors


def compute_geometry_radius(
    depth: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute an origin-invariant scene scale from reprojected world points.

    Unlike `compute_scale_factor()`, this helper first centers the valid world
    points of each sample and then measures their mean radius. This makes the
    resulting scale robust to arbitrary world-origin choices across datasets.

    Args:
        depth: Depth map [B, T, H, W] or [B, 1, H, W]
        K: Camera intrinsics [B, T, 3, 3] or [B, 3, 3]
        R: Rotation matrix [B, T, 3, 3] or [B, 3, 3] (Camera-to-World)
        t: Translation vector [B, T, 3] or [B, 3] (Camera-to-World)
        valid_mask: Valid pixel mask [B, T, H, W] or [B, 1, H, W]
        eps: Minimum fallback scale

    Returns:
        geometry_radius: [B] centered scene scale
    """
    if depth.dim() == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)
        has_time_dim = False
    elif depth.dim() == 4:
        has_time_dim = True
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    B = depth.shape[0]
    if has_time_dim:
        T, H, W = depth.shape[1], depth.shape[2], depth.shape[3]
    else:
        H, W = depth.shape[1], depth.shape[2]
        T = 1
        depth = depth.unsqueeze(1)

    device = depth.device

    if K.dim() == 3:
        K = K.unsqueeze(1).expand(B, T, 3, 3)
    if R.dim() == 3:
        R = R.unsqueeze(1).expand(B, T, 3, 3)
    if t.dim() == 2:
        t = t.unsqueeze(1).expand(B, T, 3)

    depth_flat = depth.reshape(B * T, H, W)
    K_flat = K.reshape(B * T, 3, 3).float()
    R_flat = R.reshape(B * T, 3, 3).float()
    t_flat = t.reshape(B * T, 3).float()

    if valid_mask is not None:
        if valid_mask.dim() == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1).expand(B, T, H, W)
        valid_mask_flat = valid_mask.reshape(B * T, H, W) > 0.5
    else:
        valid_mask_flat = depth_flat > 0

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixels = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=0)
    pixels = pixels.view(3, -1).unsqueeze(0).expand(B * T, 3, H * W)

    K_inv = torch.inverse(K_flat)
    ray_dirs_cam = torch.bmm(K_inv, pixels)
    P_cam = depth_flat.view(B * T, 1, -1) * ray_dirs_cam
    P_world = torch.bmm(R_flat, P_cam) + t_flat.unsqueeze(2)
    P_world = P_world.transpose(1, 2).reshape(B, T * H * W, 3)
    valid_mask_flat = valid_mask_flat.reshape(B, T * H * W)

    geometry_radii = []
    for batch_idx in range(B):
        points = P_world[batch_idx][valid_mask_flat[batch_idx]]
        if points.numel() == 0:
            geometry_radii.append(torch.tensor(1.0, device=device, dtype=depth.dtype))
            continue

        center = points.mean(dim=0, keepdim=True)
        radii = torch.linalg.norm(points - center, dim=-1)
        radii = radii[torch.isfinite(radii)]
        if radii.numel() == 0:
            geometry_radii.append(torch.tensor(1.0, device=device, dtype=depth.dtype))
            continue

        geometry_radii.append(radii.mean().clamp(min=eps).to(dtype=depth.dtype))

    return torch.stack(geometry_radii, dim=0)


def normalize_geometry_targets(
    depth: torch.Tensor,
    ray: torch.Tensor,
    scale_factor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize geometry targets by scale factor.
    
    Following DA3 paper, all distance-related quantities should be normalized
    by the global scale factor before computing losses. This includes:
    - Depth values
    - Ray origins (camera position in world coordinates)
    
    Direction vectors and intrinsics are NOT normalized as they are scale-invariant.
    
    Args:
        depth: Depth map [B, T, 1, H, W] or [B, 1, H, W]
        ray: Ray map [B, T, 6, H, W] or [B, 6, H, W]
             Format: [direction_x, direction_y, direction_z, origin_x, origin_y, origin_z]
        scale_factor: Scale factor [B]
        
    Returns:
        normalized_depth: Normalized depth map (same shape as input)
        normalized_ray: Normalized ray map (same shape as input)
    """
    # Reshape scale_factor for broadcasting
    if depth.dim() == 5:  # [B, T, 1, H, W]
        scale = scale_factor.view(-1, 1, 1, 1, 1)
    elif depth.dim() == 4:  # [B, 1, H, W]
        scale = scale_factor.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    
    # Normalize depth
    normalized_depth = depth / scale
    
    # Normalize ray: only normalize origin (last 3 channels), not direction (first 3 channels)
    if ray.dim() == 5:  # [B, T, 6, H, W]
        direction = ray[:, :, :3, :, :]  # [B, T, 3, H, W]
        origin = ray[:, :, 3:, :, :]  # [B, T, 3, H, W]
        scale_ray = scale_factor.view(-1, 1, 1, 1, 1)
    elif ray.dim() == 4:  # [B, 6, H, W]
        direction = ray[:, :3, :, :]  # [B, 3, H, W]
        origin = ray[:, 3:, :, :]  # [B, 3, H, W]
        scale_ray = scale_factor.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected ray shape: {ray.shape}")
    
    normalized_origin = origin / scale_ray
    normalized_ray = torch.cat([direction, normalized_origin], dim=-3)
    
    return normalized_depth, normalized_ray


def normalize_depth_by_scene_scale(
    depth: torch.Tensor,
    scene_scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize depth-like tensors by a per-sample scene scale."""
    scene_scale = scene_scale.to(device=depth.device, dtype=depth.dtype).clamp(min=eps)
    if depth.dim() == 5:
        scale = scene_scale.view(-1, 1, 1, 1, 1)
    elif depth.dim() == 4:
        scale = scene_scale.view(-1, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth / scale


def extract_camera_center_from_origin_map(
    ray_origin: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Aggregate a dense 3-channel origin map into a per-frame vector.

    Args:
        ray_origin: Tensor [B, T, 3, H, W] or [B, 3, H, W]
        weights: Optional spatial weights [B, T, 1, H, W] / [B, 1, H, W]
        valid_mask: Optional valid mask [B, T, 1, H, W] / [B, 1, H, W]
        eps: Small epsilon for numerical stability

    Returns:
        aggregated_vector: [B, T, 3] or [B, 3] (matching time dimension of input)
    """
    has_time_dim = ray_origin.dim() == 5
    if ray_origin.dim() == 4:
        ray_origin = ray_origin.unsqueeze(1)  # [B, 1, 3, H, W]
    elif ray_origin.dim() != 5:
        raise ValueError(f"Unexpected ray_origin shape: {ray_origin.shape}")

    if ray_origin.shape[2] != 3:
        raise ValueError(f"ray_origin must have 3 channels, got {ray_origin.shape}")

    B, T, _, H, W = ray_origin.shape

    if weights is None:
        weights = torch.ones(
            (B, T, 1, H, W),
            dtype=ray_origin.dtype,
            device=ray_origin.device,
        )
    else:
        if weights.dim() == 4:
            weights = weights.unsqueeze(1)
        if weights.dim() != 5:
            raise ValueError(f"Unexpected weights shape: {weights.shape}")
        if weights.shape[2] != 1:
            raise ValueError(f"weights must be single-channel, got {weights.shape}")

    if valid_mask is not None:
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.dim() != 5:
            raise ValueError(f"Unexpected valid_mask shape: {valid_mask.shape}")
        if valid_mask.shape[2] != 1:
            raise ValueError(f"valid_mask must be single-channel, got {valid_mask.shape}")
        weights = weights * valid_mask.to(dtype=weights.dtype)

    weights = weights.clamp(min=0.0)
    weighted_sum = (ray_origin * weights).sum(dim=(-2, -1))  # [B, T, 3]
    weight_sum = weights.sum(dim=(-2, -1)).clamp(min=eps)    # [B, T, 1]
    aggregated_vector = weighted_sum / weight_sum

    if not has_time_dim:
        aggregated_vector = aggregated_vector.squeeze(1)  # [B, 3]

    return aggregated_vector


def compute_first_camera_baseline_loss(
    pred_camera_center: torch.Tensor,
    target_camera_center: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    scene_scale: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervise raw baseline lengths after anchoring trajectories to frame 0."""
    if pred_camera_center.dim() == 2:
        pred_camera_center = pred_camera_center.unsqueeze(1)
    if target_camera_center.dim() == 2:
        target_camera_center = target_camera_center.unsqueeze(1)

    if pred_camera_center.shape != target_camera_center.shape:
        raise ValueError(
            "pred_camera_center and target_camera_center must share shape, got "
            f"{tuple(pred_camera_center.shape)} and {tuple(target_camera_center.shape)}"
        )

    batch_size, num_frames, _ = pred_camera_center.shape
    if num_frames <= 1:
        return pred_camera_center.sum() * 0.0

    pred_baseline = torch.linalg.norm(
        pred_camera_center - pred_camera_center[:, :1],
        dim=-1,
    )
    target_baseline = torch.linalg.norm(
        target_camera_center - target_camera_center[:, :1],
        dim=-1,
    )

    baseline_error = torch.abs(pred_baseline - target_baseline)
    if scene_scale is not None:
        baseline_scale = scene_scale.to(
            device=baseline_error.device,
            dtype=baseline_error.dtype,
        ).clamp(min=eps).view(batch_size, 1)
        baseline_error = baseline_error / baseline_scale

    baseline_mask = torch.ones_like(baseline_error)
    baseline_mask[:, 0] = 0.0
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.squeeze(-1)
        if valid_mask.dim() != 2:
            raise ValueError(f"valid_mask must be [B, T] or [B, T, 1], got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.to(dtype=baseline_mask.dtype)
        baseline_mask = baseline_mask * valid_mask * valid_mask[:, :1]

    if reduction == "mean":
        denom = baseline_mask.sum().clamp(min=1.0)
        return (baseline_error * baseline_mask).sum() / denom
    if reduction == "sum":
        return (baseline_error * baseline_mask).sum()
    if reduction == "none":
        return baseline_error * baseline_mask
    raise ValueError(f"Unsupported reduction: {reduction}")


def compute_center_world_loss(
    pred_camera_center: torch.Tensor,
    target_camera_center: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    scene_scale: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervise per-frame camera centers after scene-scale normalization."""
    if pred_camera_center.dim() == 2:
        pred_camera_center = pred_camera_center.unsqueeze(1)
    if target_camera_center.dim() == 2:
        target_camera_center = target_camera_center.unsqueeze(1)

    if pred_camera_center.shape != target_camera_center.shape:
        raise ValueError(
            "pred_camera_center and target_camera_center must share shape, got "
            f"{tuple(pred_camera_center.shape)} and {tuple(target_camera_center.shape)}"
        )

    center_error = torch.abs(pred_camera_center - target_camera_center)
    if scene_scale is not None:
        center_scale = scene_scale.to(
            device=center_error.device,
            dtype=center_error.dtype,
        ).clamp(min=eps).view(pred_camera_center.shape[0], 1, 1)
        center_error = center_error / center_scale

    center_mask = torch.ones_like(center_error[..., :1])
    if valid_mask is not None:
        if valid_mask.dim() == 2:
            valid_mask = valid_mask.unsqueeze(-1)
        if valid_mask.dim() != 3:
            raise ValueError(f"valid_mask must be [B, T] or [B, T, 1], got {tuple(valid_mask.shape)}")
        center_mask = center_mask * valid_mask.to(dtype=center_mask.dtype)

    if reduction == "mean":
        denom = center_mask.sum().clamp(min=1.0) * center_error.shape[-1]
        return (center_error * center_mask).sum() / denom
    if reduction == "sum":
        return (center_error * center_mask).sum()
    if reduction == "none":
        return center_error * center_mask
    raise ValueError(f"Unsupported reduction: {reduction}")


def align_depth_least_squares(pred, target, valid_mask=None, eps=1e-8, sequence_level=False):
    """
    Vectorized least-squares depth alignment with automatic shape handling.

    Finds optimal ``(scale, shift)`` such that ``aligned = scale * pred + shift``
    minimises ``||aligned - target||^2`` over valid pixels.

    **Gradient policy**: ``scale`` and ``shift`` are ``.detach()``-ed so that
    gradients do **not** flow through the alignment parameters.  This prevents
    the model from "gaming" the alignment (optimising scale/shift instead of
    improving predictions).  Gradients still flow through ``pred`` in the
    expression ``scale * pred + shift``, which is equivalent to supervising
    in a linearly-transformed coordinate system — mathematically valid.

    Args:
        pred: Predicted depth.  Supported shapes: [B, T, 1, H, W], [B, 1, H, W], or [B, N].
        target: Target depth (same shape as *pred*).
        valid_mask: Optional validity mask (same shape as *pred*).
        eps: Small epsilon for numerical stability.
        sequence_level: When True **and** input is 5-D ``[B, T, 1, H, W]``,
            compute a **single** (scale, shift) pair per batch element across
            all T frames (``[B, T*H*W]``).  When False (default) each frame
            gets its own pair (``[B*T, H*W]``).  Has no effect for 4-D or
            2-D inputs.
    """
    # 1. Handle shapes automatically: preserve the original shape and restore it at the end
    mask_flat = None
    orig_shape = pred.shape
    
    # Flatten to the common [N_samples, N_pixels] layout
    if pred.dim() == 5: # [B, T, 1, H, W]
        if sequence_level:
            # Sequence-level: one scale/shift per batch element
            # [B, T, 1, H, W] -> [B, T*H*W]
            B = pred.shape[0]
            pred_flat = pred.reshape(B, -1)
            target_flat = target.reshape(B, -1)
            if valid_mask is not None:
                mask_flat = valid_mask.reshape(B, -1) > 0.5
        else:
            # Per-frame: one scale/shift per frame
            # [B, T, 1, H, W] -> [B*T, H*W]
            pred_flat = pred.flatten(0, 2).flatten(1)
            target_flat = target.flatten(0, 2).flatten(1)
            if valid_mask is not None:
                mask_flat = valid_mask.flatten(0, 2).flatten(1) > 0.5
    elif pred.dim() == 4: # [B, 1, H, W]
        pred_flat = pred.flatten(1)                   # [B, H*W]
        target_flat = target.flatten(1)
        if valid_mask is not None:
            mask_flat = valid_mask.flatten(1) > 0.5
    else:
        # Assume the input is already [B, N]
        pred_flat = pred
        target_flat = target
        mask_flat = valid_mask

    # Without a mask, use locations where both inputs are positive
    if mask_flat is None:
        mask_flat = (target_flat > 0) & (pred_flat > 0)

    # 2. Compute statistics in vectorized form
    # mask: [N_samples, N_pixels]
    pred_masked = pred_flat.masked_fill(~mask_flat, 0)
    target_masked = target_flat.masked_fill(~mask_flat, 0)
    
    # Count valid pixels
    count = mask_flat.sum(dim=1, keepdim=True) # [N_samples, 1]
    
    # Avoid division by zero and identify invalid samples
    valid_samples_mask = count >= 10  # Equivalent to the original fewer-than-10-valid-pixels check
    safe_count = count.clamp(min=1)

    p_mean = pred_masked.sum(dim=1, keepdim=True) / safe_count
    t_mean = target_masked.sum(dim=1, keepdim=True) / safe_count

    # Mask invalid regions again when computing variance and covariance
    # (pred - mean) * mask zeros invalid regions so they do not affect the sum
    p_centered = (pred_flat - p_mean) * mask_flat
    t_centered = (target_flat - t_mean) * mask_flat

    var_p = (p_centered ** 2).sum(dim=1, keepdim=True) + eps
    cov_pt = (p_centered * t_centered).sum(dim=1, keepdim=True)

    scale = (cov_pt / var_p).clamp(0.01, 100.0).detach()
    shift = (t_mean - scale * p_mean).detach()

    # 3. Apply alignment
    aligned_flat = scale * pred_flat + shift
    
    # 4. Fall back to the original values for samples with too few valid pixels
    # Use aligned results only where valid_samples_mask is True; otherwise retain the original values
    if not valid_samples_mask.all():
        aligned_flat = torch.where(valid_samples_mask, aligned_flat, pred_flat)

    # 5. Restore the original shape
    return aligned_flat.reshape(orig_shape)



class TokenDistillationLoss(nn.Module):
    """
    Token distillation loss for matching reconstructed tokens to original DA3 tokens.
    
    L_token = w_mse * L_MSE + w_cos * L_cos
    
    where:
    - L_MSE = MSE(stopgrad(F_da3), F_geo_recon)
    - L_cos = 1 - CosineSimilarity(stopgrad(F_da3), F_geo_recon)
    
    Uses stop-gradient on the target to prevent gradients flowing back to DA3 encoder.
    L_MSE recovers intensity while L_cos aligns semantic direction.
    """

    def __init__(
        self, 
        reduction: str = 'mean',
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0
    ):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            mse_weight: Weight for MSE loss component (default: 1.0)
            cosine_weight: Weight for cosine similarity loss component (default: 0.1)
        """
        super().__init__()
        self.reduction = reduction
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

    def forward(
        self, 
        pred_tokens: torch.Tensor, 
        target_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute token distillation loss combining MSE and cosine similarity.
        
        Args:
            pred_tokens: Predicted/reconstructed tokens [B, C, T, H, W]
            target_tokens: Target DA3 tokens (will be detached) [B, C, T, H, W]
            mask: Optional mask for valid regions [B, 1, T, H, W]
            
        Returns:
            Combined token distillation loss
        """
        # Stop gradient on target
        target = target_tokens.detach()
        
        # Compute MSE loss
        mse_loss = F.mse_loss(pred_tokens, target, reduction='none')
        
        # Compute cosine similarity loss
        # Reshape to [B*T*H*W, C] for cosine similarity computation
        B, C, T, H, W = pred_tokens.shape
        pred_flat = pred_tokens.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*T*H*W, C]
        target_flat = target.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*T*H*W, C]
        
        # Compute cosine similarity along channel dimension
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)  # [B*T*H*W]
        cos_loss = 1.0 - cos_sim  # Convert to loss
        cos_loss = cos_loss.reshape(B, T, H, W).unsqueeze(1)  # [B, 1, T, H, W]
        
        # Apply mask if provided
        if mask is not None:
            mse_loss = mse_loss * mask
            cos_loss = cos_loss * mask
            
            if self.reduction == 'mean':
                num_valid = mask.sum() * pred_tokens.shape[1] + 1e-8
                mse_loss_reduced = mse_loss.sum() / num_valid
                cos_loss_reduced = cos_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                mse_loss_reduced = mse_loss.sum()
                cos_loss_reduced = cos_loss.sum()
            else:
                # For 'none' reduction, return combined loss per element
                return self.mse_weight * mse_loss + self.cosine_weight * cos_loss
        else:
            if self.reduction == 'mean':
                mse_loss_reduced = mse_loss.mean()
                cos_loss_reduced = cos_loss.mean()
            elif self.reduction == 'sum':
                mse_loss_reduced = mse_loss.sum()
                cos_loss_reduced = cos_loss.sum()
            else:
                # For 'none' reduction, return combined loss per element
                return self.mse_weight * mse_loss + self.cosine_weight * cos_loss
        
        # Combine losses with weights
        total_loss = self.mse_weight * mse_loss_reduced + self.cosine_weight * cos_loss_reduced
        return total_loss



# ============================================================================
# VDA (Video-Depth-Anything) Loss Components
# Ported from Video-Depth-Anything/loss/loss.py for local use.
# Includes: TrimmedMAELoss, multi-scale GradientLoss, TrimmedProcrustesLoss,
#           TemporalGradientMatchingLoss, and supporting utilities.
# ============================================================================

def _vda_reduction_batch_based(image_loss, M):
    """Average of all valid pixels of the batch."""
    divisor = torch.sum(M)
    if divisor == 0:
        return torch.sum(image_loss) * 0.0
    else:
        return torch.sum(image_loss) / divisor


def _vda_reduction_image_based(image_loss, M):
    """Mean of average of valid pixels of an image."""
    valid = M.nonzero()
    image_loss[valid] = image_loss[valid] / M[valid]
    return torch.mean(image_loss)


def _vda_gradient_loss(prediction, target, mask, reduction=None, frame_id_mask=None):
    """Spatial gradient loss used by VDA multi-scale gradient."""
    if reduction is None:
        reduction = _vda_reduction_batch_based

    valid_id_mask_x = torch.ones_like(mask[:, :, 1:])
    valid_id_mask_y = torch.ones_like(mask[:, 1:, :])
    if frame_id_mask is not None:
        valid_id_mask_x = ((frame_id_mask[:, :, 1:] - frame_id_mask[:, :, :-1]) == 0).to(mask.dtype)
        valid_id_mask_y = ((frame_id_mask[:, 1:, :] - frame_id_mask[:, :-1, :]) == 0).to(mask.dtype)

    M = torch.sum(mask, (1, 2))
    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(torch.mul(mask[:, :, 1:], mask[:, :, :-1]), valid_id_mask_x)
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(torch.mul(mask[:, 1:, :], mask[:, :-1, :]), valid_id_mask_y)
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    return reduction(image_loss, M)


def _vda_normalize_prediction_robust(target, mask, ms=None):
    """Robust median-based normalization for scale-shift invariant comparison."""
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    if ms is None:
        m = torch.zeros_like(ssum)
        s = torch.ones_like(ssum)
        m[valid] = torch.median(
            (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
        ).values
    else:
        m, s = ms

    target = target - m.view(-1, 1, 1)

    if ms is None:
        sq = torch.sum(mask * target.abs(), (1, 2))
        s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1)), (m.detach(), s.detach())




class TrimmedMAELoss(nn.Module):
    """Trimmed MAE loss: sorts residuals and keeps only the smallest (1-trim) fraction."""

    def __init__(self, trim=0.2, reduction="batch-based"):
        super().__init__()
        self.trim = trim
        if reduction == "batch-based":
            self._reduction = _vda_reduction_batch_based
        else:
            self._reduction = _vda_reduction_image_based

    def forward(self, prediction, target, mask, weight_mask=None):
        if torch.sum(mask) == 0:
            return torch.sum(prediction) * 0.0
        M = torch.sum(mask, (1, 2))
        res = prediction - target
        if weight_mask is not None:
            res = res * weight_mask
        res = res[mask.bool()].abs()
        trimmed, _ = torch.sort(res.view(-1), descending=False)
        keep_num = int(len(res) * (1.0 - self.trim))
        if keep_num <= 0:
            return torch.sum(prediction) * 0.0
        trimmed = trimmed[:keep_num]
        return self._reduction(trimmed, M)


class MultiScaleGradientLoss(nn.Module):
    """Multi-scale gradient loss (VDA style): computes gradient loss at 4 spatial scales."""

    def __init__(self, scales=4, reduction="batch-based"):
        super().__init__()
        if reduction == "batch-based":
            self._reduction = _vda_reduction_batch_based
        else:
            self._reduction = _vda_reduction_image_based
        self._scales = scales

    def forward(self, prediction, target, mask, num_frame_h=1):
        total = 0

        frame_id_mask = None
        if num_frame_h > 1:
            frame_h = mask.shape[1] // num_frame_h
            frame_id_mask = torch.zeros_like(mask)
            for i in range(num_frame_h):
                frame_id_mask[:, i * frame_h:(i + 1) * frame_h, :] = i + 1

        for scale in range(self._scales):
            step = pow(2, scale)
            total += _vda_gradient_loss(
                prediction[:, ::step, ::step],
                target[:, ::step, ::step],
                mask[:, ::step, ::step],
                reduction=self._reduction,
                frame_id_mask=frame_id_mask[:, ::step, ::step] if num_frame_h > 1 else None,
            )
        return total


class TrimmedProcrustesLoss(nn.Module):
    """Combined trimmed MAE + multi-scale gradient loss with robust normalization.

    This is the spatial depth loss from Video-Depth-Anything.
    """

    def __init__(self, alpha=0.5, scales=4, trim=0.2, reduction="batch-based"):
        super().__init__()
        self._data_loss = TrimmedMAELoss(reduction=reduction, trim=trim)
        self._regularization_loss = MultiScaleGradientLoss(scales=scales, reduction=reduction)
        self._alpha = alpha

    def forward(self, prediction, target, mask, pred_ms=None, tar_ms=None,
                num_frame_h=1, no_norm=False):
        if no_norm:
            prediction_ssi = prediction
            target_ = target
        else:
            prediction_ssi, _ = _vda_normalize_prediction_robust(prediction, mask, ms=pred_ms)
            target_, _ = _vda_normalize_prediction_robust(target, mask, ms=tar_ms)

        total = self._data_loss(prediction_ssi, target_, mask)
        if self._alpha > 0:
            total += self._alpha * self._regularization_loss(
                prediction_ssi, target_, mask, num_frame_h=num_frame_h
            )
        return total


class TemporalGradientMatchingLoss(nn.Module):
    """Temporal gradient matching loss for video depth consistency.

    Ensures depth changes between adjacent frames are consistent with GT.
    Uses multi-temporal-scale finite differences with occlusion filtering.
    """

    def __init__(self, trim=0.2, temp_grad_scales=1, temp_grad_decay=0.5,
                 reduction="batch-based", diff_depth_th=0.05):
        super().__init__()
        trim_reduction = reduction if reduction != "none" else "batch-based"
        self.data_loss = TrimmedMAELoss(trim=trim, reduction=trim_reduction)
        self.temp_grad_scales = temp_grad_scales
        self.temp_grad_decay = temp_grad_decay
        self.diff_depth_th = diff_depth_th
        self.reduction = reduction

    def _forward_scalar(self, prediction, target, mask):
        total = 0
        cnt = 0

        min_target = torch.where(mask.bool(), target, torch.tensor(float('inf'), device=target.device)).min(-1).values.min(-1).values
        max_target = torch.where(mask.bool(), target, torch.tensor(float('-inf'), device=target.device)).max(-1).values.max(-1).values
        target_th = (max_target - min_target) * self.diff_depth_th

        for scale in range(self.temp_grad_scales):
            temp_stride = pow(2, scale)
            if temp_stride < prediction.shape[1]:
                pred_temp_grad = torch.diff(prediction[:, ::temp_stride, ...], dim=1)
                target_temp_grad = torch.diff(target[:, ::temp_stride, ...], dim=1)
                temp_mask = (
                    mask[:, ::temp_stride, ...][:, 1:, ...] &
                    mask[:, ::temp_stride, ...][:, :-1, ...]
                )

                valid_mask_from_target_th = (
                    target_temp_grad.abs() <
                    target_th.unsqueeze(-1).unsqueeze(-1)[:, ::temp_stride, ...][:, 1:, ...]
                )
                temp_mask = temp_mask & valid_mask_from_target_th

                total += self.data_loss(
                    prediction=pred_temp_grad.flatten(0, 1),
                    target=target_temp_grad.flatten(0, 1),
                    mask=temp_mask.flatten(0, 1).float(),
                ) * pow(self.temp_grad_decay, scale)
                cnt += 1

        if cnt == 0:
            return torch.tensor(0.0, device=prediction.device, requires_grad=True)
        return total / cnt

    def forward(self, prediction, target, mask):
        """
        Args:
            prediction: [B, T, H, W]
            target:     [B, T, H, W]
            mask:       [B, T, H, W] (bool)
        """
        if self.reduction == "none":
            batch_losses = [
                self._forward_scalar(
                    prediction[batch_idx : batch_idx + 1],
                    target[batch_idx : batch_idx + 1],
                    mask[batch_idx : batch_idx + 1],
                )
                for batch_idx in range(prediction.shape[0])
            ]
            return torch.stack(batch_losses, dim=0)
        return self._forward_scalar(prediction, target, mask)


def build_temporal_gradient_matching_loss(
    *,
    trim: float = 0.2,
    temporal_grad_scales: int = 1,
    temporal_grad_decay: float = 0.5,
    temporal_diff_depth_th: float = 0.05,
    reduction: str = "batch-based",
) -> TemporalGradientMatchingLoss:
    return TemporalGradientMatchingLoss(
        trim=trim,
        temp_grad_scales=temporal_grad_scales,
        temp_grad_decay=temporal_grad_decay,
        diff_depth_th=temporal_diff_depth_th,
        reduction=reduction,
    )


# ============================================================================
# Geometry-specific Loss Functions
# ============================================================================

class ConfidenceAwareDepthLoss(nn.Module):
    """
    Confidence-aware depth loss from DA3 training.
    
    L_D = (1/Z) * Σ m_p * (D_c,p * |D_pred - D_gt| - λ_c * log(D_c,p))
    
    Where:
    - D_c,p: Depth confidence at pixel p
    - m_p: Valid mask at pixel p
    - λ_c: Confidence regularization coefficient (prevents overconfidence)
    
    This loss encourages the model to:
    1. Minimize depth error weighted by confidence
    2. Avoid being overconfident (via log regularization)
    """

    def __init__(self, lambda_c: float = 0.5, reduction: str = 'mean'):
        """
        Args:
            lambda_c: Confidence regularization coefficient (default: 0.5)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.lambda_c = lambda_c
        self.reduction = reduction

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        depth_conf: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute confidence-aware depth loss.
        
        Args:
            pred_depth: Predicted depth [B, T, 1, H, W] or [B, 1, H, W]
            target_depth: Target depth [B, T, 1, H, W] or [B, 1, H, W]
            depth_conf: Depth confidence [B, T, 1, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, T, 1, H, W] or [B, 1, H, W] (optional)
            
        Returns:
            Confidence-aware depth loss
        """
        # Create valid mask: only compute loss where target_depth > 0
        if valid_mask is None:
            valid_mask = (target_depth > 0).float()
        else:
            valid_mask = valid_mask * (target_depth > 0).float()
        
        # Keep the historical max-depth guard for sky / far-range supervision,
        # but clamp target and prediction symmetrically so relative-depth
        # alignment is not penalized purely by the cap itself.
        pred_depth_clamped = pred_depth.clamp(min=1e-6, max=100.0)
        target_depth_clamped = target_depth.clamp(min=1e-6, max=100.0)

        # Compute L1 depth error
        depth_error = torch.abs(pred_depth_clamped - target_depth_clamped)
        
        # Clamp depth error to prevent extreme loss values
        # Max error of 10000.0 is reasonable for normalized depth
        depth_error = depth_error.clamp(max=10000.0)
        
        # Clamp confidence to match DA3's expp1 activation range [1, +inf)
        # DA3 uses conf_activation="expp1": conf = exp(x) + 1, so conf >= 1
        # Optimal solution: conf = lambda_c / |error| (high conf when error is small)
        depth_conf_clamped = depth_conf.clamp(min=1.0, max=1e4)
        
        # Confidence-weighted loss: D_c * |D_pred - D_gt|
        conf_weighted_loss = depth_conf_clamped * depth_error
        
        # Confidence regularization: -λ_c * log(D_c)
        # For conf >= 1: log(conf) >= 0, so -log(conf) <= 0
        # This rewards high confidence, balanced by the conf * error term
        conf_reg = -self.lambda_c * torch.log(depth_conf_clamped)
        # Total loss per pixel
        loss = conf_weighted_loss + conf_reg
        
        # Clamp per-pixel loss to non-negative to prevent confidence exploitation
        loss = loss.clamp(min=0.0)
        
        # Apply valid mask
        loss = loss * valid_mask

        # Normalize by number of valid pixels
        num_valid = valid_mask.sum() + 1e-8
        
        if self.reduction == 'mean':
            return loss.sum() / num_valid
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DepthGradientLoss(nn.Module):
    """
    Depth gradient loss for preserving edges and smoothness.
    
    L_grad = ||∇_x D_pred - ∇_x D_gt||_1 + ||∇_y D_pred - ∇_y D_gt||_1
    
    Where:
    - ∇_x: Horizontal gradient (finite difference)
    - ∇_y: Vertical gradient (finite difference)
    
    This loss encourages:
    1. Sharp edges where GT has sharp edges
    2. Smooth regions where GT is smooth
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute depth gradient loss.
        
        Args:
            pred_depth: Predicted depth [B, T, 1, H, W] or [B, 1, H, W]
            target_depth: Target depth [B, T, 1, H, W] or [B, 1, H, W]
            valid_mask: Valid pixel mask [B, T, 1, H, W] or [B, 1, H, W] (optional)
            
        Returns:
            Depth gradient loss
        """
        # Compute horizontal gradients (finite difference)
        pred_grad_x = pred_depth[..., :, 1:] - pred_depth[..., :, :-1]
        target_grad_x = target_depth[..., :, 1:] - target_depth[..., :, :-1]
        
        # Compute vertical gradients (finite difference)
        pred_grad_y = pred_depth[..., 1:, :] - pred_depth[..., :-1, :]
        target_grad_y = target_depth[..., 1:, :] - target_depth[..., :-1, :]
        
        # Compute L1 loss on gradients
        loss_x = torch.abs(pred_grad_x - target_grad_x)
        loss_y = torch.abs(pred_grad_y - target_grad_y)
        
        # Apply valid mask if provided
        batch_size = pred_depth.shape[0]
        if valid_mask is not None:
            # Adjust mask for gradient dimensions
            mask_x = valid_mask[..., :, 1:] * valid_mask[..., :, :-1]
            mask_y = valid_mask[..., 1:, :] * valid_mask[..., :-1, :]
            
            loss_x = loss_x * mask_x
            loss_y = loss_y * mask_y
            
            if self.reduction == 'mean':
                loss_x = loss_x.sum() / (mask_x.sum() + 1e-8)
                loss_y = loss_y.sum() / (mask_y.sum() + 1e-8)
            elif self.reduction == 'sum':
                loss_x = loss_x.sum()
                loss_y = loss_y.sum()
            elif self.reduction == 'none':
                loss_x = loss_x.reshape(batch_size, -1).sum(dim=1) / mask_x.reshape(batch_size, -1).sum(dim=1).clamp(min=1.0)
                loss_y = loss_y.reshape(batch_size, -1).sum(dim=1) / mask_y.reshape(batch_size, -1).sum(dim=1).clamp(min=1.0)
        else:
            if self.reduction == 'mean':
                loss_x = loss_x.mean()
                loss_y = loss_y.mean()
            elif self.reduction == 'sum':
                loss_x = loss_x.sum()
                loss_y = loss_y.sum()
            elif self.reduction == 'none':
                loss_x = loss_x.reshape(batch_size, -1).mean(dim=1)
                loss_y = loss_y.reshape(batch_size, -1).mean(dim=1)
        
        # Combine horizontal and vertical gradient losses
        return loss_x + loss_y


class ConfidenceAwareRayLoss(nn.Module):
    """Masked dense L1 loss on aligned ray directions."""

    def __init__(
        self,
        lambda_c: float = 0.5,
        lambda_origin_center: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_origin_center = lambda_origin_center
        self.reduction = reduction

    def forward(
        self,
        pred_ray: torch.Tensor,
        target_ray: torch.Tensor,
        ray_conf: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        pred_origin_center: Optional[torch.Tensor] = None,
        target_origin_center: Optional[torch.Tensor] = None,
        origin_center_mask: Optional[torch.Tensor] = None,
        geometry_scale_factor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute dense direction L1 against the official canonical GT ray target.

        `ray_conf` is intentionally not used in the main GT ray term. DA3's GT
        objective specifies an L1 ray-map loss `L_M`, while the exp-range
        confidence field is primarily consumed by the camera decoder. Using it as
        a multiplicative weight makes frozen DA3 look much worse than its actual
        decoded camera quality.
        """
        del ray_conf, pred_origin_center, target_origin_center, origin_center_mask, geometry_scale_factor

        if pred_ray.shape[2] < 3 or target_ray.shape[2] < 3:
            raise ValueError(
                f"pred_ray/target_ray must contain at least 3 ray channels, got "
                f"{pred_ray.shape} and {target_ray.shape}"
            )

        loss = torch.abs(pred_ray[:, :, :3, :, :] - target_ray[:, :, :3, :, :])

        if valid_mask is not None:
            valid_mask_expanded = valid_mask.expand_as(loss).to(dtype=loss.dtype)
            loss = loss * valid_mask_expanded
            if self.reduction == 'mean':
                dense_loss = loss.sum() / (valid_mask_expanded.sum() + 1e-8)
            elif self.reduction == 'sum':
                dense_loss = loss.sum()
            else:
                dense_loss = loss
        else:
            if self.reduction == 'mean':
                dense_loss = loss.mean()
            elif self.reduction == 'sum':
                dense_loss = loss.sum()
            else:
                dense_loss = loss
        return dense_loss


class PointCloudLoss(nn.Module):
    """Recon-unposed-style dense point-cloud loss surrogate."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.max_points = 2048

    def _fp32_autocast_context(self, device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _point_cloud_work_dtype(self, *tensors: Optional[torch.Tensor]) -> Optional[torch.dtype]:
        work_dtype: Optional[torch.dtype] = None
        for tensor in tensors:
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            work_dtype = tensor.dtype if work_dtype is None else torch.promote_types(work_dtype, tensor.dtype)
        if work_dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return work_dtype

    def _cast_floating_tensor(
        self,
        tensor: Optional[torch.Tensor],
        *,
        dtype: Optional[torch.dtype],
    ) -> Optional[torch.Tensor]:
        if tensor is None or not torch.is_tensor(tensor) or not tensor.is_floating_point() or dtype is None:
            return tensor
        return tensor.to(dtype=dtype)

    def _cast_camera_dict(
        self,
        pred_camera: dict[str, torch.Tensor],
        *,
        dtype: Optional[torch.dtype],
    ) -> dict[str, torch.Tensor]:
        if dtype is None:
            return pred_camera
        casted: dict[str, torch.Tensor] = {}
        for key, value in pred_camera.items():
            if torch.is_tensor(value) and value.is_floating_point():
                casted[key] = value.to(dtype=dtype)
            else:
                casted[key] = value
        return casted

    def _decode_camera_for_point_cloud(
        self,
        pred_ray: torch.Tensor,
        ray_conf: Optional[torch.Tensor],
        *,
        image_height: int,
        image_width: int,
    ) -> dict[str, torch.Tensor]:
        work_dtype = self._point_cloud_work_dtype(pred_ray, ray_conf)
        with self._fp32_autocast_context(pred_ray.device):
            pred_camera = decode_official_camera_from_ray(
                self._cast_floating_tensor(pred_ray, dtype=work_dtype),
                self._cast_floating_tensor(ray_conf, dtype=work_dtype),
                image_height=image_height,
                image_width=image_width,
                training=True,
            )
        return self._cast_camera_dict(pred_camera, dtype=work_dtype)

    def forward(
        self,
        pred_depth: torch.Tensor,
        pred_ray: torch.Tensor,
        gt_points_3d: Optional[torch.Tensor] = None,
        gt_points_2d: Optional[torch.Tensor] = None,
        points_valid_mask: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        pred_camera_center: Optional[torch.Tensor] = None,
        pred_translation_w2c: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        target_ray: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        target_extrinsics: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        gt_scene_scale: Optional[torch.Tensor] = None,
        ray_conf: Optional[torch.Tensor] = None,
        pred_camera_intrinsics: Optional[torch.Tensor] = None,
        pred_camera_extrinsics: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute a differentiable surrogate for official `recon_unposed`."""
        if pred_depth.dim() == 4:
            pred_depth = pred_depth.unsqueeze(2)

        if pred_ray.dim() != 5:
            raise ValueError(f"pred_ray must be [B, T, C, H, W], got {pred_ray.shape}")
        if pred_ray.shape[2] < 6:
            raise ValueError(f"pred_ray must have 6 channels (direction + origin), got {pred_ray.shape}")

        if target_depth is not None and valid_mask is not None:
            return self._forward_dense(
                pred_depth=pred_depth,
                pred_ray=pred_ray,
                target_depth=target_depth,
                target_ray=target_ray,
                target_intrinsics=target_intrinsics,
                target_extrinsics=target_extrinsics,
                valid_mask=valid_mask,
                gt_scene_scale=gt_scene_scale,
                ray_conf=ray_conf,
                pred_camera_intrinsics=pred_camera_intrinsics,
                pred_camera_extrinsics=pred_camera_extrinsics,
                eps=eps,
            )
        return pred_depth.sum() * 0.0

    def _forward_dense(
        self,
        pred_depth: torch.Tensor,
        pred_ray: torch.Tensor,
        target_depth: torch.Tensor,
        target_ray: torch.Tensor,
        target_intrinsics: Optional[torch.Tensor],
        target_extrinsics: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
        gt_scene_scale: Optional[torch.Tensor],
        ray_conf: Optional[torch.Tensor],
        pred_camera_intrinsics: Optional[torch.Tensor],
        pred_camera_extrinsics: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        if target_depth.dim() == 4:
            target_depth = target_depth.unsqueeze(2)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.unsqueeze(2)

        if pred_camera_intrinsics is not None or pred_camera_extrinsics is not None:
            if pred_camera_intrinsics is None or pred_camera_extrinsics is None:
                raise ValueError(
                    "pred_camera_intrinsics and pred_camera_extrinsics must be provided together for PointCloudLoss."
                )
            pred_camera = {
                "intrinsics": pred_camera_intrinsics,
                "extrinsics": pred_camera_extrinsics,
            }
        else:
            pred_camera = self._decode_camera_for_point_cloud(
                pred_ray,
                ray_conf,
                image_height=pred_depth.shape[-2],
                image_width=pred_depth.shape[-1],
            )
        del gt_scene_scale, eps

        if target_intrinsics is not None and target_extrinsics is not None:
            work_dtype = self._point_cloud_work_dtype(
                pred_depth,
                target_depth,
                target_intrinsics,
                target_extrinsics,
                pred_camera["intrinsics"],
                pred_camera["extrinsics"],
            )
            pred_depth = self._cast_floating_tensor(pred_depth, dtype=work_dtype)
            target_depth = self._cast_floating_tensor(target_depth, dtype=work_dtype)
            target_intrinsics = self._cast_floating_tensor(target_intrinsics, dtype=work_dtype)
            target_extrinsics = self._cast_floating_tensor(target_extrinsics, dtype=work_dtype)
            pred_camera = self._cast_camera_dict(pred_camera, dtype=work_dtype)

            with self._fp32_autocast_context(pred_depth.device):
                aligned_pred_extrinsics, pose_scale = align_extrinsics_umeyama_like_official(
                    target_extrinsics=target_extrinsics,
                    pred_extrinsics=pred_camera["extrinsics"],
                )
                pred_points_world = unproject_depth_map_with_camera(
                    pred_depth,
                    intrinsics=pred_camera["intrinsics"],
                    extrinsics=aligned_pred_extrinsics,
                )
                target_points_world = unproject_depth_map_with_camera(
                    target_depth,
                    intrinsics=target_intrinsics,
                    extrinsics=target_extrinsics,
                )
                del pose_scale
        else:
            if target_ray is None or target_ray.dim() != 5 or target_ray.shape[2] < 6:
                raise ValueError(
                    "target_ray must be [B, T, 6, H, W] when target_intrinsics/target_extrinsics are absent, "
                    f"got {None if target_ray is None else tuple(target_ray.shape)}"
                )
            pred_points_world = compute_point_map_from_depth_and_ray(pred_depth, pred_ray[:, :, :6, :, :])
            target_points_world = compute_point_map_from_depth_and_ray(target_depth, target_ray[:, :, :6, :, :])
        return symmetric_dense_point_set_distance(
            pred_points_world=pred_points_world,
            target_points_world=target_points_world,
            valid_mask=valid_mask,
            max_points=self.max_points,
            reduction=self.reduction,
        )


class DecodedCameraLoss(nn.Module):
    """Auxiliary camera loss using official pose-eval alignment semantics."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def _frame_valid_mask(
        self,
        valid_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(batch_size, num_frames, device=device, dtype=dtype)
        if valid_mask.dim() == 5:
            valid_mask = valid_mask.squeeze(2)
        frame_mask = (valid_mask.reshape(batch_size, num_frames, -1).sum(dim=-1) > 0).to(dtype=dtype)
        return frame_mask

    def _prepare_decoder_weights(
        self,
        conf: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        num_frames: int,
        image_height: int,
        image_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if conf is None:
            weights = torch.ones(
                batch_size,
                num_frames,
                image_height,
                image_width,
                device=device,
                dtype=dtype,
            )
        elif conf.dim() == 5 and conf.shape[2] == 1:
            weights = conf.squeeze(2).to(device=device, dtype=dtype)
        elif conf.dim() == 4:
            weights = conf.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported confidence shape: {tuple(conf.shape)}")

        if valid_mask is not None:
            if valid_mask.dim() == 5:
                valid_mask = valid_mask.squeeze(2)
            weights = weights * valid_mask.to(device=device, dtype=dtype)

        empty_frames = weights.reshape(batch_size, num_frames, -1).sum(dim=-1) <= 0
        if empty_frames.any():
            weights = weights.clone()
            weights[empty_frames] = 1.0
        return weights

    def _reduce(self, error: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        component_count = math.prod(error.shape[2:])
        view_shape = (frame_mask.shape[0], frame_mask.shape[1]) + (1,) * (error.dim() - 2)
        expanded_mask = frame_mask.view(view_shape).to(device=error.device, dtype=error.dtype)
        masked_error = error * expanded_mask

        if self.reduction == "mean":
            denom = frame_mask.sum().clamp(min=1.0) * component_count
            return masked_error.sum() / denom
        if self.reduction == "sum":
            return masked_error.sum()
        return masked_error

    def _resolve_pred_camera(
        self,
        *,
        pred_ray: Optional[torch.Tensor],
        ray_conf: Optional[torch.Tensor],
        pred_intrinsics: Optional[torch.Tensor],
        pred_extrinsics: Optional[torch.Tensor],
        pred_camera_tokens: Optional[torch.Tensor],
        camera_decoder: Optional[nn.Module],
        image_height: int,
        image_width: int,
        valid_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if pred_intrinsics is not None or pred_extrinsics is not None:
            if pred_intrinsics is None or pred_extrinsics is None:
                raise ValueError(
                    "pred_intrinsics and pred_extrinsics must be provided together."
                )
            return {
                "intrinsics": pred_intrinsics,
                "extrinsics": pred_extrinsics,
            }

        if pred_camera_tokens is not None:
            if camera_decoder is None:
                raise ValueError(
                    "camera_decoder is required when pred_camera_tokens are provided."
                )
            return decode_official_camera_from_token(
                pred_camera_tokens,
                camera_decoder,
                image_height=image_height,
                image_width=image_width,
            )

        if pred_ray is not None:
            batch_size, num_frames = pred_ray.shape[:2]
            pred_weights = self._prepare_decoder_weights(
                ray_conf,
                valid_mask,
                batch_size=batch_size,
                num_frames=num_frames,
                image_height=image_height,
                image_width=image_width,
                device=pred_ray.device,
                dtype=pred_ray.dtype,
            )
            return decode_official_camera_from_ray(
                pred_ray,
                pred_weights,
                image_height=image_height,
                image_width=image_width,
                training=False,
            )

        raise ValueError(
            "DecodedCameraLoss requires one of: "
            "(pred_intrinsics + pred_extrinsics), "
            "(pred_camera_tokens + camera_decoder), or pred_ray."
        )

    def forward(
        self,
        pred_ray: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        target_extrinsics: Optional[torch.Tensor] = None,
        ray_conf: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        geometry_scale_factor: Optional[torch.Tensor] = None,
        pred_intrinsics: Optional[torch.Tensor] = None,
        pred_extrinsics: Optional[torch.Tensor] = None,
        pred_camera_tokens: Optional[torch.Tensor] = None,
        camera_decoder: Optional[nn.Module] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        if target_intrinsics is None or target_extrinsics is None:
            raise ValueError("target_intrinsics and target_extrinsics are required.")
        if target_intrinsics.dim() != 4 or target_intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected target_intrinsics [B, T, 3, 3], got {tuple(target_intrinsics.shape)}"
            )
        if target_extrinsics.dim() != 4 or target_extrinsics.shape[-1] < 4 or target_extrinsics.shape[-2] < 3:
            raise ValueError(
                f"Expected target_extrinsics [B, T, 3/4, 4], got {tuple(target_extrinsics.shape)}"
            )

        if pred_ray is not None and (pred_ray.dim() != 5 or pred_ray.shape[2] < 6):
            raise ValueError(f"Expected pred_ray [B, T, 6, H, W], got {tuple(pred_ray.shape)}")

        batch_size, num_frames = target_intrinsics.shape[:2]
        if image_height is None or image_width is None:
            if pred_ray is not None:
                image_height = pred_ray.shape[-2]
                image_width = pred_ray.shape[-1]
            else:
                image_height = int(target_intrinsics[..., 1, 2].float().mean().item() * 2.0)
                image_width = int(target_intrinsics[..., 0, 2].float().mean().item() * 2.0)

        frame_mask = self._frame_valid_mask(
            valid_mask,
            batch_size=batch_size,
            num_frames=num_frames,
            device=target_intrinsics.device,
            dtype=target_intrinsics.dtype,
        )

        pred_camera = self._resolve_pred_camera(
            pred_ray=pred_ray,
            ray_conf=ray_conf,
            pred_intrinsics=pred_intrinsics,
            pred_extrinsics=pred_extrinsics,
            pred_camera_tokens=pred_camera_tokens,
            camera_decoder=camera_decoder,
            image_height=image_height,
            image_width=image_width,
            valid_mask=valid_mask,
        )
        pred_extrinsics_raw = pred_camera["extrinsics"]
        pred_intrinsics_raw = pred_camera["intrinsics"]
        finite_camera_mask = (
            torch.isfinite(pred_extrinsics_raw).all(dim=(-1, -2))
            & torch.isfinite(pred_intrinsics_raw).all(dim=(-1, -2))
        ).to(device=frame_mask.device, dtype=frame_mask.dtype)
        frame_mask = frame_mask * finite_camera_mask

        if frame_mask.sum() <= 0:
            zero = target_intrinsics.new_zeros(())
            return {
                "camera_loss": zero,
                "camera_rotation_loss": zero,
                "camera_translation_loss": zero,
                "camera_intrinsics_loss": zero,
                "camera_center_loss": zero,
            }

        pred_extrinsics_eval = align_extrinsics_to_first_camera_torch(pred_extrinsics_raw)
        target_extrinsics_eval = align_extrinsics_to_first_camera_torch(target_extrinsics)
        pred_intrinsics_norm = normalize_intrinsics_params(
            pred_intrinsics_raw,
            image_height,
            image_width,
        )
        target_intrinsics_norm = normalize_intrinsics_params(
            target_intrinsics,
            image_height,
            image_width,
        )
        rotation_loss = self._reduce(
            torch.abs(pred_extrinsics_eval[..., :3, :3] - target_extrinsics_eval[..., :3, :3]),
            frame_mask,
        )
        translation_loss = self._reduce(
            torch.abs(pred_extrinsics_eval[..., :3, 3] - target_extrinsics_eval[..., :3, 3]),
            frame_mask,
        )
        pose_translation_dir_loss = compute_translation_direction_loss(
            pred_extrinsics_eval[..., :3, 3],
            target_extrinsics_eval[..., :3, 3],
            valid_mask=frame_mask,
            reduction=self.reduction,
        )
        intrinsics_loss = self._reduce(
            torch.abs(pred_intrinsics_norm - target_intrinsics_norm),
            frame_mask,
        )
        del geometry_scale_factor
        total_loss = rotation_loss + pose_translation_dir_loss + intrinsics_loss
        return {
            "camera_loss": total_loss,
            "camera_rotation_loss": rotation_loss,
            "camera_translation_loss": translation_loss,
            "camera_intrinsics_loss": intrinsics_loss,
            "camera_center_loss": pose_translation_dir_loss,
        }


class GeometryReconstructionLoss(nn.Module):
    """DA3-style GT geometry loss with canonical ray targets and camera-scale depth alignment."""

    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_depth_grad: float = 0.1,
        lambda_ray: float = 1.0,
        lambda_point_cloud: float = 1.0,
        lambda_camera: float = 0.0,
        lambda_c_depth: float = 0.5,
        lambda_c_ray: float = 0.5,
        lambda_ray_origin_center: float = 1.0,
        lambda_ray_raw_baseline: float = 1.0,
        is_metric_depth: bool = True,
        use_scale_invariant_depth: bool = False,
        reduction: str = 'mean',
        # VDA-style loss parameters (default False for backward compatibility)
        use_trimmed_loss: bool = False,
        lambda_trimmed_procrustes: float = 1.0,
        trim: float = 0.2,
        trimmed_alpha: float = 0.5,
        trimmed_scales: int = 4,
        use_temporal_loss: bool = False,
        lambda_temporal: float = 10.0,
        temporal_trim: float = 0.2,
        temporal_grad_scales: int = 1,
        temporal_grad_decay: float = 0.5,
        temporal_diff_depth_th: float = 0.05,
    ):
        """
        Args:
            lambda_depth: Weight for confidence-aware depth loss
            lambda_depth_grad: Weight for depth gradient loss
            lambda_ray: Weight for confidence-aware ray loss
            lambda_point_cloud: Weight for point cloud loss
            lambda_camera: Weight for decoded-camera supervision from the DA3 camera-token branch
            lambda_c_depth: Confidence regularization for depth
            lambda_c_ray: Confidence regularization for ray
            lambda_ray_origin_center: Internal weight of the absolute ray-origin scale term inside ray loss
            lambda_ray_raw_baseline: Deprecated no-op retained for backward compatibility
            is_metric_depth: Whether GT depth is metric (kept for backward compat)
            use_scale_invariant_depth: Whether to align pred depth (kept for backward compat)
            reduction: Reduction method
            use_trimmed_loss: Enable VDA TrimmedProcrustes spatial depth loss
            lambda_trimmed_procrustes: Weight for trimmed procrustes loss
            trim: Fraction of outlier pixels to discard (0.2 = drop top 20%)
            trimmed_alpha: Weight for multi-scale gradient in TrimmedProcrustesLoss
            trimmed_scales: Number of spatial scales for gradient loss
            use_temporal_loss: Enable VDA temporal gradient matching loss
            lambda_temporal: Weight for temporal loss (VDA default: 10)
            temporal_trim: Trim ratio for temporal loss MAE
            temporal_grad_scales: Number of temporal scales
            temporal_grad_decay: Decay factor per temporal scale
            temporal_diff_depth_th: Threshold for filtering large depth changes
        """
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_depth_grad = lambda_depth_grad
        self.lambda_ray = lambda_ray
        self.lambda_point_cloud = lambda_point_cloud
        self.lambda_camera = lambda_camera
        self.reduction = reduction
        self.is_metric_depth = is_metric_depth
        self.use_scale_invariant_depth = use_scale_invariant_depth
        self.lambda_ray_origin_center = lambda_ray_origin_center
        self.lambda_ray_raw_baseline = lambda_ray_raw_baseline
        
        # VDA-style loss flags
        self.use_trimmed_loss = use_trimmed_loss
        self.lambda_trimmed_procrustes = lambda_trimmed_procrustes
        self.use_temporal_loss = use_temporal_loss
        self.lambda_temporal = lambda_temporal
        
        # Initialize sub-losses
        self.depth_loss = ConfidenceAwareDepthLoss(
            lambda_c=lambda_c_depth,
            reduction=reduction
        )
        self.depth_loss_per_pixel = ConfidenceAwareDepthLoss(
            lambda_c=lambda_c_depth,
            reduction="none",
        )
        self.depth_grad_loss = DepthGradientLoss(reduction=reduction)
        self.depth_grad_loss_per_sample = DepthGradientLoss(reduction="none")
        self.ray_loss = ConfidenceAwareRayLoss(
            lambda_c=lambda_c_ray,
            lambda_origin_center=0.0,
            reduction=reduction
        )
        self.point_cloud_loss = PointCloudLoss(reduction=reduction)
        self.point_cloud_loss_per_sample = PointCloudLoss(reduction="none")
        self.camera_loss = DecodedCameraLoss(reduction=reduction)

        # VDA-style losses (conditionally initialized)
        if use_trimmed_loss:
            self.trimmed_procrustes_loss_fn = TrimmedProcrustesLoss(
                alpha=trimmed_alpha,
                scales=trimmed_scales,
                trim=trim,
                reduction="batch-based",
            )
        
        if use_temporal_loss:
            self.temporal_loss_fn = build_temporal_gradient_matching_loss(
                trim=temporal_trim,
                temporal_grad_scales=temporal_grad_scales,
                temporal_grad_decay=temporal_grad_decay,
                temporal_diff_depth_th=temporal_diff_depth_th,
                reduction="batch-based",
            )
            self.temporal_loss_fn_per_sample = build_temporal_gradient_matching_loss(
                trim=temporal_trim,
                temporal_grad_scales=temporal_grad_scales,
                temporal_grad_decay=temporal_grad_decay,
                temporal_diff_depth_th=temporal_diff_depth_th,
                reduction="none",
            )

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        pred_ray: torch.Tensor,
        target_ray: torch.Tensor,
        depth_conf: torch.Tensor,
        ray_conf: torch.Tensor,
        valid_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        gt_points_3d: Optional[torch.Tensor] = None,
        gt_points_2d: Optional[torch.Tensor] = None,
        points_valid_mask: Optional[torch.Tensor] = None,
        pred_camera_intrinsics: Optional[torch.Tensor] = None,
        pred_camera_extrinsics: Optional[torch.Tensor] = None,
        pred_camera_tokens: Optional[torch.Tensor] = None,
        camera_decoder: Optional[nn.Module] = None,
        pred_camera_image_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometry reconstruction loss with global scale normalization.
        
        Args:
            pred_depth: Predicted depth [B, T, 1, H_pred, W_pred]
            target_depth: Target depth [B, T, 1, H_gt, W_gt]
            pred_ray: Predicted ray [B, T, 6, H_pred, W_pred]
            target_ray: Target ray [B, T, 6, H_gt, W_gt]
            depth_conf: Depth confidence [B, T, 1, H_pred, W_pred]
            ray_conf: Ray confidence [B, T, 1, H_pred, W_pred]
            valid_mask: Valid pixel mask [B, T, 1, H_gt, W_gt]
            intrinsics: Camera intrinsics [B, T, 3, 3]
            extrinsics: Camera extrinsics [B, T, 4, 4] (world-to-camera)
            gt_points_3d: GT 3D points [B, T, N, 3] (world coordinates, optional)
            gt_points_2d: GT 2D points [B, T, N, 2] (pixel coordinates, optional)
            points_valid_mask: Valid point mask [B, T, N] (optional)
            pred_camera_intrinsics: Optional decoded student intrinsics from camera-token branch
            pred_camera_extrinsics: Optional decoded student extrinsics from camera-token branch
            pred_camera_tokens: Optional raw student camera tokens (decoded when camera_decoder is provided)
            camera_decoder: Optional DA3 `cam_dec` used to decode pred_camera_tokens
            pred_camera_image_hw: Optional image size used by camera-token decode
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}

        if target_depth.dim() == 4:
            target_depth = target_depth.unsqueeze(2)
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.unsqueeze(2)

        if pred_ray is not None:
            if pred_ray.shape[-1] == 6 and pred_ray.shape[2] != 6:
                pred_ray = pred_ray.permute(0, 1, 4, 2, 3)
            assert pred_ray.dim() == 5 and pred_ray.shape[2] == 6, \
                f"pred_ray should be [B, T, 6, H, W], got {pred_ray.shape}"
        
        if target_ray is not None:
            if target_ray.shape[-1] == 6 and target_ray.shape[2] != 6:
                target_ray = target_ray.permute(0, 1, 4, 2, 3)
            assert target_ray.dim() == 5 and target_ray.shape[2] == 6, \
                f"target_ray should be [B, T, 6, H, W], got {target_ray.shape}"
        
        target_h, target_w = target_depth.shape[-2], target_depth.shape[-1]
        pred_depth_h, pred_depth_w = pred_depth.shape[-2], pred_depth.shape[-1]
        pred_ray_h, pred_ray_w = pred_ray.shape[-2], pred_ray.shape[-1]

        if pred_depth_h != target_h or pred_depth_w != target_w:
            pred_depth = F.interpolate(
                pred_depth.reshape(-1, 1, pred_depth_h, pred_depth_w),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).reshape(pred_depth.shape[0], pred_depth.shape[1], 1, target_h, target_w)
            
            # Upsample depth confidence if needed
            if depth_conf is not None:
                depth_conf_h, depth_conf_w = depth_conf.shape[-2], depth_conf.shape[-1]
                if depth_conf_h != target_h or depth_conf_w != target_w:
                    depth_conf = F.interpolate(
                        depth_conf.reshape(-1, 1, depth_conf_h, depth_conf_w),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).reshape(depth_conf.shape[0], depth_conf.shape[1], 1, target_h, target_w)

        if pred_ray_h != target_h or pred_ray_w != target_w:
            pred_ray = upsample_ray_with_renormalization(
                pred_ray, target_size=(target_h, target_w), mode='bilinear'
            )
            if ray_conf is not None:
                ray_conf_h, ray_conf_w = ray_conf.shape[-2], ray_conf.shape[-1]
                if ray_conf_h != target_h or ray_conf_w != target_w:
                    ray_conf = F.interpolate(
                        ray_conf.reshape(-1, 1, ray_conf_h, ray_conf_w),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    ).reshape(ray_conf.shape[0], ray_conf.shape[1], 1, target_h, target_w)

        gt_camera_ray = generate_ray_map_batch(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            height=target_h,
            width=target_w,
        )
        gt_targets = build_official_gt_geometry_targets(
            target_depth=target_depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            valid_mask=valid_mask,
            target_ray=gt_camera_ray,
        )
        common_gt_scale = gt_targets["common_gt_scale"]
        target_ray_pose_normalized = gt_targets["target_ray_pose_normalized"]
        ray_supervision_enabled = self.lambda_ray > 0
        needs_official_depth_alignment = (
            self.lambda_depth > 0
            or self.lambda_depth_grad > 0
            or ray_supervision_enabled
            or self.lambda_point_cloud > 0
            or (self.use_trimmed_loss and self.lambda_trimmed_procrustes > 0)
            or (self.use_temporal_loss and self.lambda_temporal > 0 and pred_depth.shape[1] > 1)
        )
        pred_camera_for_gt_alignment = None
        pred_depth_aligned = pred_depth
        depth_umeyama_scale = pred_depth.new_ones(pred_depth.shape[0])
        if needs_official_depth_alignment:
            pred_camera_for_gt_alignment = resolve_pred_camera_for_gt_geometry_alignment(
                pred_camera_intrinsics=pred_camera_intrinsics,
                pred_camera_extrinsics=pred_camera_extrinsics,
                pred_camera_tokens=pred_camera_tokens,
                camera_decoder=camera_decoder,
                pred_camera_image_hw=pred_camera_image_hw,
                target_image_hw=(target_h, target_w),
            )
            pred_depth_aligned, depth_umeyama_scale = align_depth_by_official_camera_scale(
                pred_depth,
                pred_camera_for_gt_alignment["extrinsics"],
                extrinsics,
                valid_mask=valid_mask,
            )

        losses["common_gt_scale"] = common_gt_scale
        losses["scale_factor"] = common_gt_scale
        losses["depth_umeyama_scale"] = depth_umeyama_scale.mean()

        raw_depth_loss = self.depth_loss_per_pixel(
            pred_depth=pred_depth_aligned,
            target_depth=target_depth,
            depth_conf=depth_conf,
            valid_mask=valid_mask
        )
        loss_depth = reduce_scene_scale_reweighted_loss(
            reduce_loss_tensor_per_batch(
                raw_depth_loss,
                valid_mask,
                reduction=self.reduction,
            ),
            common_gt_scale,
            reduction=self.reduction,
        )
        if self.use_trimmed_loss:
            loss_depth_grad = torch.tensor(0.0, device=pred_depth.device)
        else:
            raw_depth_grad_loss = self.depth_grad_loss_per_sample(
                pred_depth=pred_depth_aligned,
                target_depth=target_depth,
                valid_mask=valid_mask
            )
            loss_depth_grad = reduce_scene_scale_reweighted_loss(
                raw_depth_grad_loss,
                common_gt_scale,
                reduction=self.reduction,
            )

        losses['depth_loss'] = loss_depth
        losses['depth_grad_loss'] = loss_depth_grad

        if self.use_trimmed_loss:
            loss_trimmed = self.trimmed_procrustes_loss_fn(
                prediction=pred_depth_aligned.squeeze(2).flatten(0, 1),  # [B*T, H, W]
                target=target_depth.squeeze(2).flatten(0, 1),            # [B*T, H, W]
                mask=valid_mask.squeeze(2).flatten(0, 1).float(),        # [B*T, H, W]
                no_norm=True,
            )
            losses['trimmed_procrustes_loss'] = loss_trimmed
        else:
            losses['trimmed_procrustes_loss'] = torch.tensor(0.0, device=pred_depth.device)

        if ray_supervision_enabled:
            pred_ray_aligned, ray_umeyama_scale = align_ray_to_official_camera_pose_space(
                pred_ray,
                pred_camera_for_gt_alignment["extrinsics"],
                extrinsics,
                gt_targets["pose_scale"],
                valid_mask=valid_mask,
            )
            losses["ray_umeyama_scale"] = ray_umeyama_scale.mean()

            loss_ray_direction = self.ray_loss(
                pred_ray=pred_ray_aligned,
                target_ray=target_ray_pose_normalized,
                ray_conf=ray_conf,
                valid_mask=valid_mask,
            )

            if self.lambda_ray_origin_center > 0:
                raw_origin_scale_loss = torch.abs(
                    pred_ray_aligned[:, :, 3:, :, :]
                    - target_ray_pose_normalized[:, :, 3:, :, :]
                )
                origin_scale_mask = valid_mask.expand_as(raw_origin_scale_loss)
                origin_scale_loss_per_sample = reduce_loss_tensor_per_batch(
                    raw_origin_scale_loss,
                    origin_scale_mask,
                    reduction="mean" if self.reduction != "none" else "none",
                )
                if self.reduction == "mean":
                    loss_ray_origin_scale = origin_scale_loss_per_sample.mean()
                elif self.reduction == "sum":
                    loss_ray_origin_scale = (
                        raw_origin_scale_loss * origin_scale_mask.to(dtype=raw_origin_scale_loss.dtype)
                    ).sum()
                else:
                    loss_ray_origin_scale = origin_scale_loss_per_sample
            else:
                loss_ray_origin_scale = torch.tensor(0.0, device=pred_depth.device)
            loss_ray = loss_ray_direction + self.lambda_ray_origin_center * loss_ray_origin_scale
        else:
            loss_ray_direction = torch.tensor(0.0, device=pred_depth.device)
            loss_ray_origin_scale = torch.tensor(0.0, device=pred_depth.device)
            loss_ray = torch.tensor(0.0, device=pred_depth.device)
        losses["ray_direction_loss"] = loss_ray_direction
        losses["ray_origin_scale_loss"] = loss_ray_origin_scale
        losses['ray_loss'] = loss_ray

        if self.lambda_point_cloud > 0:
            raw_point_cloud_loss = self.point_cloud_loss_per_sample(
                pred_depth=pred_depth_aligned,
                pred_ray=pred_ray,
                gt_points_3d=gt_points_3d,
                gt_points_2d=gt_points_2d,
                points_valid_mask=points_valid_mask,
                extrinsics=extrinsics,
                target_depth=target_depth,
                target_ray=gt_camera_ray,
                target_intrinsics=intrinsics,
                target_extrinsics=extrinsics,
                valid_mask=valid_mask,
                gt_scene_scale=None,
                ray_conf=ray_conf,
                pred_camera_intrinsics=(
                    pred_camera_for_gt_alignment["intrinsics"] if pred_camera_for_gt_alignment is not None else pred_camera_intrinsics
                ),
                pred_camera_extrinsics=(
                    pred_camera_for_gt_alignment["extrinsics"] if pred_camera_for_gt_alignment is not None else pred_camera_extrinsics
                ),
            )
            loss_point_cloud = reduce_scene_scale_reweighted_loss(
                raw_point_cloud_loss,
                common_gt_scale,
                reduction=self.reduction,
            )
        else:
            loss_point_cloud = torch.tensor(0.0, device=pred_depth.device)
        losses['point_cloud_loss'] = loss_point_cloud

        camera_supervision_enabled = (
            self.lambda_camera > 0
            and (
                (pred_camera_intrinsics is not None and pred_camera_extrinsics is not None)
                or pred_camera_tokens is not None
            )
        )
        if camera_supervision_enabled:
            if pred_camera_image_hw is None:
                camera_image_hw = (target_h, target_w)
            else:
                camera_image_hw = (
                    int(pred_camera_image_hw[0]),
                    int(pred_camera_image_hw[1]),
                )
            camera_supervision_pred_intrinsics = pred_camera_intrinsics
            camera_supervision_pred_extrinsics = pred_camera_extrinsics
            camera_supervision_tokens = pred_camera_tokens
            camera_supervision_decoder = camera_decoder
            if pred_camera_for_gt_alignment is not None:
                camera_supervision_pred_intrinsics = pred_camera_for_gt_alignment["intrinsics"]
                camera_supervision_pred_extrinsics = pred_camera_for_gt_alignment["extrinsics"]
                camera_supervision_tokens = None
                camera_supervision_decoder = None
            camera_loss_dict = self.camera_loss(
                target_intrinsics=intrinsics,
                target_extrinsics=extrinsics,
                valid_mask=valid_mask,
                pred_intrinsics=camera_supervision_pred_intrinsics,
                pred_extrinsics=camera_supervision_pred_extrinsics,
                pred_camera_tokens=camera_supervision_tokens,
                camera_decoder=camera_supervision_decoder,
                image_height=camera_image_hw[0],
                image_width=camera_image_hw[1],
            )
            losses.update(camera_loss_dict)

        losses['total_loss'] = (
            self.lambda_depth * loss_depth +
            self.lambda_depth_grad * loss_depth_grad +
            self.lambda_ray * loss_ray +
            self.lambda_point_cloud * loss_point_cloud
        )
        if camera_supervision_enabled:
            losses['total_loss'] = losses['total_loss'] + self.lambda_camera * losses['camera_loss']

        if self.use_trimmed_loss:
            losses['total_loss'] = (
                losses['total_loss'] +
                self.lambda_trimmed_procrustes * losses['trimmed_procrustes_loss']
            )

        if self.use_temporal_loss and pred_depth.shape[1] > 1:
            pred_seq_aligned_temporal = pred_depth_aligned.squeeze(2)  # [B, T, H, W]
            target_depth_4d = target_depth.squeeze(2)                  # [B, T, H, W]
            valid_mask_4d = valid_mask.squeeze(2)        # [B, T, H, W]
            raw_temporal_loss = self.temporal_loss_fn_per_sample(
                prediction=pred_seq_aligned_temporal,
                target=target_depth_4d,
                mask=valid_mask_4d.bool(),
            )
            loss_temporal = reduce_scene_scale_reweighted_loss(
                raw_temporal_loss,
                common_gt_scale,
                reduction=self.reduction,
            )
            losses['temporal_loss'] = loss_temporal
            losses['total_loss'] = losses['total_loss'] + self.lambda_temporal * loss_temporal
        else:
            losses['temporal_loss'] = torch.tensor(0.0, device=pred_depth.device)
        
        return losses


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss between two Gaussian distributions.
    
    L_kl = KL(q_adapter(z|F_da3) || q_longcat(z|video))
         = 0.5 * sum(log(σ_q²/σ_p²) + (σ_p² + (μ_p - μ_q)²)/σ_q² - 1)
    
    Where:
    - q_adapter: GeometryAdapter VAE's latent distribution (P)
    - q_longcat: LongCat-Video VAE's latent distribution (Q, target)
    
    Note: q_longcat can be provided either as:
    1. Explicit tensors (mu_q, log_var_q) - for online computation
    2. Pre-computed statistics (latents_mean, latents_std) - for efficiency
    """

    def __init__(
        self,
        reduction: str = 'sum',
        free_bits: float = 0.0,
        latents_mean: Optional[List[float]] = None,
        latents_std: Optional[List[float]] = None
    ):
        """
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            free_bits: Minimum KL per dimension (for free-bits regularization)
            latents_mean: Pre-computed mean of LongCat VAE latent distribution (16 values)
            latents_std: Pre-computed std of LongCat VAE latent distribution (16 values)
        """
        super().__init__()
        self.reduction = reduction
        self.free_bits = free_bits
        
        # Register pre-computed statistics as buffers (will be moved to device automatically)
        if latents_mean is not None and latents_std is not None:
            self.register_buffer('target_mu', torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1))
            self.register_buffer('target_log_var', torch.log(torch.tensor(latents_std, dtype=torch.float32).pow(2)).view(1, -1, 1, 1, 1))
            self.use_precomputed = True
        else:
            self.use_precomputed = False

    def forward(
        self, 
        mu_p: torch.Tensor, 
        log_var_p: torch.Tensor,
        mu_q: Optional[torch.Tensor] = None,
        log_var_q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussian distributions.
        
        KL(P||Q) = 0.5 * sum(log(σ_q²/σ_p²) + (σ_p² + (μ_p - μ_q)²)/σ_q² - 1)
        
        Args:
            mu_p: Mean of adapter distribution (P) [B, C, T, H, W]
            log_var_p: Log variance of adapter distribution (P) [B, C, T, H, W]
            mu_q: Mean of LongCat VAE distribution (Q, target) [B, C, T, H, W] (optional if using pre-computed)
            log_var_q: Log variance of LongCat VAE distribution (Q, target) [B, C, T, H, W] (optional if using pre-computed)
            
        Returns:
            KL divergence loss
        """
        # Use pre-computed statistics if available and no explicit target provided
        if self.use_precomputed and mu_q is None and log_var_q is None:
            # Ensure pre-computed statistics are on the same device as input
            # Broadcast pre-computed statistics to match input shape [B, C, T, H, W]
            mu_q = self.target_mu.to(mu_p.device).expand_as(mu_p)
            log_var_q = self.target_log_var.to(log_var_p.device).expand_as(log_var_p)
        elif mu_q is None or log_var_q is None:
            raise ValueError("Either provide mu_q/log_var_q or initialize with latents_mean/latents_std")
        
        # Clamp log_var for numerical stability
        log_var_p = log_var_p.clamp(-30.0, 20.0)
        log_var_q = log_var_q.clamp(-30.0, 20.0)
        
        # Compute variance
        var_p = log_var_p.exp()
        var_q = log_var_q.exp()
        
        # KL divergence between two Gaussians:
        # KL(P||Q) = 0.5 * (log(var_q/var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1)
        #          = 0.5 * (log_var_q - log_var_p + var_p/var_q + (mu_p - mu_q)^2/var_q - 1)
        kl = 0.5 * (
            log_var_q - log_var_p +
            var_p / var_q +
            (mu_p - mu_q).pow(2) / var_q -
            1.0
        )
        
        # Apply free bits (minimum KL per dimension)
        if self.free_bits > 0:
            kl = torch.clamp(kl, min=self.free_bits)
        
        # Apply reduction
        if self.reduction == "mean":
            return kl.mean()
        elif self.reduction == "sum":
            return kl.sum()
        else:
            return kl


# ============================================================================
# KL Annealing Schedulers
# ============================================================================

class CyclicalKLScheduler:
    """
    Cyclical Annealing scheduler for KL divergence weight.
    
    Implements the cyclical annealing strategy from "Cyclical Annealing Schedule: 
    A Simple Approach to Mitigating KL Vanishing" (Fu et al., 2019).
    
    The scheduler alternates between:
    1. Reconstruction phase: β ≈ 0 (focus on reconstruction)
    2. Regularization phase: β gradually increases (enforce latent structure)
    
    This prevents KL Vanishing and Posterior Collapse by giving the model
    time to learn meaningful reconstructions before enforcing the prior.
    
    Formula:
        β(step) = min(β_max, β_max * (step % cycle_length) / warmup_steps)
    
    Where:
    - cycle_length: Number of steps in one complete cycle
    - warmup_steps: Number of steps to reach β_max within each cycle
    - β_max: Maximum KL weight (target weight from config)
    """
    
    def __init__(
        self,
        kl_weight_max: float = 1.0,
        cycle_length: int = 5000,
        warmup_ratio: float = 0.5,
        n_cycles: int = 4,
        start_step: int = 0,
    ):
        """
        Args:
            kl_weight_max: Maximum KL weight (β_max)
            cycle_length: Number of training steps in one complete cycle
            warmup_ratio: Ratio of cycle spent warming up (0.5 = half cycle warmup)
            n_cycles: Number of cycles to perform (after which β stays at max)
            start_step: Starting step (for resuming training)
        """
        self.kl_weight_max = kl_weight_max
        self.cycle_length = cycle_length
        self.warmup_steps = int(cycle_length * warmup_ratio)
        self.n_cycles = n_cycles
        self.total_annealing_steps = cycle_length * n_cycles
        self.current_step = start_step
    
    def step(self) -> float:
        """
        Get current KL weight and advance to next step.
        
        Returns:
            Current KL weight (β)
        """
        kl_weight = self.get_kl_weight()
        self.current_step += 1
        return kl_weight
    
    def get_kl_weight(self) -> float:
        """
        Calculate KL weight for current step without advancing.
        
        Returns:
            Current KL weight (β)
        """
        # After all cycles complete, use maximum weight
        if self.current_step >= self.total_annealing_steps:
            return self.kl_weight_max
        
        # Calculate position within current cycle
        step_in_cycle = self.current_step % self.cycle_length
        
        # Linear warmup within cycle
        if step_in_cycle < self.warmup_steps:
            # Linear increase from 0 to kl_weight_max
            return self.kl_weight_max * (step_in_cycle / self.warmup_steps)
        else:
            # Stay at maximum for rest of cycle
            return self.kl_weight_max
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'kl_weight_max': self.kl_weight_max,
            'cycle_length': self.cycle_length,
            'warmup_steps': self.warmup_steps,
            'n_cycles': self.n_cycles,
            'total_annealing_steps': self.total_annealing_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.kl_weight_max = state_dict['kl_weight_max']
        self.cycle_length = state_dict['cycle_length']
        self.warmup_steps = state_dict['warmup_steps']
        self.n_cycles = state_dict['n_cycles']
        self.total_annealing_steps = state_dict['total_annealing_steps']


class MonotonicKLScheduler:
    """
    Monotonic annealing scheduler for KL divergence weight.
    
    Implements simple linear warmup from 0 to target KL weight.
    Useful as a baseline or for initial training phases.
    
    Formula:
        β(step) = min(β_max, β_max * step / warmup_steps)
    """
    
    def __init__(
        self,
        kl_weight_max: float = 1.0,
        warmup_steps: int = 10000,
        start_step: int = 0,
    ):
        """
        Args:
            kl_weight_max: Maximum KL weight (β_max)
            warmup_steps: Number of steps to reach maximum weight
            start_step: Starting step (for resuming training)
        """
        self.kl_weight_max = kl_weight_max
        self.warmup_steps = warmup_steps
        self.current_step = start_step
    
    def step(self) -> float:
        """Get current KL weight and advance to next step."""
        kl_weight = self.get_kl_weight()
        self.current_step += 1
        return kl_weight
    
    def get_kl_weight(self) -> float:
        """Calculate KL weight for current step without advancing."""
        if self.current_step >= self.warmup_steps:
            return self.kl_weight_max
        return self.kl_weight_max * (self.current_step / self.warmup_steps)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'kl_weight_max': self.kl_weight_max,
            'warmup_steps': self.warmup_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.kl_weight_max = state_dict['kl_weight_max']
        self.warmup_steps = state_dict['warmup_steps']


# ============================================================================
# Combined Loss Function
# ============================================================================

class GeometryAdapterLoss(nn.Module):
    """
    Combined loss for GeometryAdapter VAE training.
    
    L_total = λ_token * L_token + β(t) * L_kl + λ_geo * L_geo
    
    Where:
    - L_token: Token reconstruction loss (MSE between reconstructed and original tokens)
    - L_kl: KL divergence loss (regularization for VAE latent space)
    - L_geo: Geometry reconstruction loss (hierarchical: depth + depth_grad + ray)
    - β(t): Dynamic KL weight controlled by scheduler (prevents KL vanishing)
    
    Key Features:
    - Supports cyclical/monotonic KL annealing to prevent posterior collapse
    - Supports is_metric_depth flag for controlling GT depth normalization
    - Supports use_scale_invariant_depth for pred depth alignment
    """

    def __init__(
        self,
        token_weight: float = 1.0,
        kl_weight: float = 0.001,
        geo_weight: float = 0.1,
        kl_reduction: str = 'sum',
        use_adaptive_kl_weight: bool = False,
        kl_free_bits: float = 0.0,
        latents_mean: Optional[List[float]] = None,
        latents_std: Optional[List[float]] = None,
        # GeometryReconstructionLoss parameters
        lambda_depth: float = 1.0,
        lambda_depth_grad: float = 0.1,
        lambda_ray: float = 1.0,
        lambda_point_cloud: float = 1.0,
        lambda_camera: float = 0.0,
        lambda_c_depth: float = 0.5,
        lambda_c_ray: float = 0.5,
        lambda_ray_origin_center: float = 1.0,
        lambda_ray_raw_baseline: float = 1.0,
        is_metric_depth: bool = True,
        use_scale_invariant_depth: bool = False,
        # VDA-style loss parameters (default False for backward compatibility)
        use_trimmed_loss: bool = False,
        lambda_trimmed_procrustes: float = 1.0,
        trim: float = 0.2,
        trimmed_alpha: float = 0.5,
        trimmed_scales: int = 4,
        use_temporal_loss: bool = False,
        lambda_temporal: float = 10.0,
        temporal_trim: float = 0.2,
        temporal_grad_scales: int = 1,
        temporal_grad_decay: float = 0.5,
        temporal_diff_depth_th: float = 0.05,
    ):
        """
        Args:
            token_weight: Weight for token reconstruction loss
            kl_weight: Base weight for KL divergence loss (used when adaptive KL is disabled)
            geo_weight: Weight for geometry reconstruction loss (total)
            kl_reduction: Reduction method for KL divergence ('mean', 'sum', 'none')
            use_adaptive_kl_weight: Match weighted KL loss order to weighted token loss
            kl_free_bits: Minimum KL per dimension
            latents_mean: Pre-computed mean of LongCat VAE latent distribution
            latents_std: Pre-computed std of LongCat VAE latent distribution
            lambda_depth: Weight for confidence-aware depth loss (within geo loss)
            lambda_depth_grad: Weight for depth gradient loss (within geo loss)
            lambda_ray: Weight for confidence-aware ray loss (within geo loss)
            lambda_point_cloud: Weight for point cloud loss (within geo loss)
            lambda_c_depth: Confidence regularization for depth
            lambda_c_ray: Confidence regularization for ray
            lambda_ray_origin_center: Internal weight of the absolute ray-origin scale term inside ray loss
            lambda_ray_raw_baseline: Weight for raw baseline-length supervision
            is_metric_depth: Whether GT depth is metric (kept for backward compat)
            use_scale_invariant_depth: Whether to align pred depth (kept for backward compat)
            use_trimmed_loss: Enable VDA TrimmedProcrustes spatial depth loss
            lambda_trimmed_procrustes: Weight for trimmed procrustes loss
            trim: Fraction of outlier pixels to discard
            trimmed_alpha: Weight for multi-scale gradient in TrimmedProcrustesLoss
            trimmed_scales: Number of spatial scales for gradient loss
            use_temporal_loss: Enable VDA temporal gradient matching loss
            lambda_temporal: Weight for temporal loss
            temporal_trim: Trim ratio for temporal loss MAE
            temporal_grad_scales: Number of temporal scales
            temporal_grad_decay: Decay factor per temporal scale
            temporal_diff_depth_th: Threshold for filtering large depth changes
        """
        super().__init__()
        if use_adaptive_kl_weight and kl_reduction == 'none':
            raise ValueError("Adaptive KL weighting requires scalar KL reduction, got 'none'.")

        self.token_weight = token_weight
        self.kl_weight = kl_weight  # Base KL weight (maximum target)
        self.geo_weight = geo_weight
        self.use_adaptive_kl_weight = use_adaptive_kl_weight
        
        # Current effective KL weight (modulated by scheduler or adaptive matching)
        self.current_kl_weight = 0.0 if use_adaptive_kl_weight else kl_weight
        
        # Loss functions
        self.token_loss_fn = TokenDistillationLoss(reduction='mean')
        self.kl_loss_fn = KLDivergenceLoss(
            reduction=kl_reduction,
            free_bits=kl_free_bits,
            latents_mean=latents_mean,
            latents_std=latents_std
        )
        self.geometry_loss_fn = GeometryReconstructionLoss(
            lambda_depth=lambda_depth,
            lambda_depth_grad=lambda_depth_grad,
            lambda_ray=lambda_ray,
            lambda_point_cloud=lambda_point_cloud,
            lambda_camera=lambda_camera,
            lambda_c_depth=lambda_c_depth,
            lambda_c_ray=lambda_c_ray,
            lambda_ray_origin_center=lambda_ray_origin_center,
            lambda_ray_raw_baseline=lambda_ray_raw_baseline,
            is_metric_depth=is_metric_depth,
            use_scale_invariant_depth=use_scale_invariant_depth,
            reduction='mean',
            # VDA-style loss parameters
            use_trimmed_loss=use_trimmed_loss,
            lambda_trimmed_procrustes=lambda_trimmed_procrustes,
            trim=trim,
            trimmed_alpha=trimmed_alpha,
            trimmed_scales=trimmed_scales,
            use_temporal_loss=use_temporal_loss,
            lambda_temporal=lambda_temporal,
            temporal_trim=temporal_trim,
            temporal_grad_scales=temporal_grad_scales,
            temporal_grad_decay=temporal_grad_decay,
            temporal_diff_depth_th=temporal_diff_depth_th,
        )
    
    def set_kl_weight(self, kl_weight: float) -> None:
        """
        Set current KL weight (called by scheduler).
        
        Args:
            kl_weight: Current KL weight from scheduler
        """
        self.current_kl_weight = kl_weight

    def forward(
        self,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        pred_depth: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        pred_ray: Optional[torch.Tensor] = None,
        target_ray: Optional[torch.Tensor] = None,
        depth_conf: Optional[torch.Tensor] = None,
        ray_conf: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        mu_longcat: Optional[torch.Tensor] = None,
        log_var_longcat: Optional[torch.Tensor] = None,
        gt_points_3d: Optional[torch.Tensor] = None,
        gt_points_2d: Optional[torch.Tensor] = None,
        points_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_tokens: Predicted tokens [B, C, T, H, W]
            target_tokens: Target tokens [B, C, T, H, W]
            mu: Mean of adapter latent distribution [B, 16, T, H, W]
            log_var: Log variance of adapter latent distribution [B, 16, T, H, W]
            pred_depth: Predicted depth [B, T, 1, H, W] (optional)
            target_depth: Target depth [B, T, 1, H, W] (optional)
            pred_ray: Predicted ray [B, T, 6, H, W] (optional)
            target_ray: Target ray [B, T, 6, H, W] (optional)
            depth_conf: Depth confidence [B, T, 1, H, W] (optional)
            ray_conf: Ray confidence [B, T, 1, H, W] (optional)
            valid_mask: Valid mask [B, T, 1, H, W] (optional)
            intrinsics: Camera intrinsics [B, T, 3, 3] (optional, required for geo loss)
            extrinsics: Camera extrinsics [B, T, 4, 4] (optional, required for geo loss)
            mu_longcat: Mean of LongCat VAE distribution (optional, for online computation)
            log_var_longcat: Log variance of LongCat VAE distribution (optional, for online computation)
            gt_points_3d: GT 3D points [B, T, N, 3] (world coordinates, optional)
            gt_points_2d: GT 2D points [B, T, N, 2] (pixel coordinates, optional)
            points_valid_mask: Valid point mask [B, T, N] (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # 1. Token reconstruction loss
        token_loss = self.token_loss_fn(
            pred_tokens=pred_tokens,
            target_tokens=target_tokens,
            mask=None
        )
        losses['token_loss'] = token_loss 
        losses['token_loss_weighted'] = token_loss * self.token_weight
        # 2. KL divergence loss
        kl_loss = self.kl_loss_fn(
            mu_p=mu,
            log_var_p=log_var,
            mu_q=mu_longcat,
            log_var_q=log_var_longcat
        )
        effective_kl_weight = self.current_kl_weight
        if self.use_adaptive_kl_weight:
            effective_kl_weight = calculate_adaptive_weight(
                self.token_weight * token_loss,
                kl_loss,
            )
            self.current_kl_weight = effective_kl_weight

        losses['kl_loss'] = kl_loss
        losses['kl_weight'] = effective_kl_weight

        # 3. Geometry reconstruction loss (optional)
        if (pred_depth is not None and target_depth is not None and
            pred_ray is not None and target_ray is not None and
            valid_mask is not None and
            intrinsics is not None and extrinsics is not None):
            
            # Use default confidence of 1.0 if not provided
            if depth_conf is None:
                depth_conf = torch.ones_like(target_depth)
            if ray_conf is None:
                ray_conf = torch.ones_like(target_depth)  # [B,T,1,H,W]
            
            geo_losses = self.geometry_loss_fn(
                pred_depth=pred_depth,
                target_depth=target_depth,
                pred_ray=pred_ray,
                target_ray=target_ray,
                depth_conf=depth_conf,
                ray_conf=ray_conf,
                valid_mask=valid_mask,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                gt_points_3d=gt_points_3d,
                gt_points_2d=gt_points_2d,
                points_valid_mask=points_valid_mask,
            )
            
            # Add geometry sub-losses to output
            losses['geometry_loss'] = geo_losses['total_loss']
            losses['depth_loss'] = geo_losses['depth_loss']
            losses['depth_grad_loss'] = geo_losses['depth_grad_loss']
            losses['ray_loss'] = geo_losses['ray_loss']
            losses['point_cloud_loss'] = geo_losses['point_cloud_loss']
            losses['trimmed_procrustes_loss'] = geo_losses['trimmed_procrustes_loss']
            losses['temporal_loss'] = geo_losses['temporal_loss']
            losses['scale_factor'] = geo_losses['scale_factor']
            losses['common_gt_scale'] = geo_losses['common_gt_scale']
            geometry_loss = geo_losses['total_loss']
        else:
            # No geometry supervision
            geometry_loss = torch.tensor(0.0, device=mu.device)
            losses['geometry_loss'] = geometry_loss
            losses['depth_loss'] = torch.tensor(0.0, device=mu.device)
            losses['depth_grad_loss'] = torch.tensor(0.0, device=mu.device)
            losses['ray_loss'] = torch.tensor(0.0, device=mu.device)
            losses['point_cloud_loss'] = torch.tensor(0.0, device=mu.device)
            losses['trimmed_procrustes_loss'] = torch.tensor(0.0, device=mu.device)
            losses['temporal_loss'] = torch.tensor(0.0, device=mu.device)
        
        # 4. Clamp geometry loss to prevent loss explosion from outlier batches
        # Normal geometry_loss is O(1~10); values >3000 indicate degenerate samples.
        # Use torch.clamp unconditionally instead of .item() + conditional branch:
        # .item() triggers GPU-CPU synchronisation which is a performance killer
        # in distributed training (FSDP/DDP).
        geo_loss_max = 3000.0
        losses['geometry_loss_unclamped'] = geometry_loss.detach()
        geometry_loss = geometry_loss.clamp(min=0.0, max=geo_loss_max)

        # 5. Total loss
        losses['total_loss'] = (
            self.token_weight * token_loss +
            effective_kl_weight * kl_loss +
            self.geo_weight * geometry_loss
        )
        
        return losses
