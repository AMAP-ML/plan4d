from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_btchw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[2] == 3:
        return tensor
    if tensor.shape[-1] == 3:
        return tensor.permute(0, 1, 4, 2, 3)
    raise ValueError(f"Cannot infer channel axis for shape={tuple(tensor.shape)}")


def _sample_scalar_per_batch(
    value: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), float(default), device=device, dtype=dtype)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.reshape(1).expand(batch_size)
    if tensor.shape[0] != batch_size:
        if tensor.numel() == 1:
            return tensor.reshape(1).expand(batch_size)
        raise ValueError(f"Expected scalar or batch scalar with B={batch_size}, got shape={tuple(tensor.shape)}")
    return tensor.reshape(batch_size, -1).mean(dim=1)


def compute_scene_flow_depth_warp_loss(
    pred: torch.Tensor,
    *,
    gt_depth: torch.Tensor,
    gt_depth_valid_mask: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    deform_mask: torch.Tensor,
    scene_flow_scale: torch.Tensor | float,
    flow_target_amplification: torch.Tensor | float = 1.0,
    loss_type: str = 'l1',
    depth_warp_space: str = 'metric',
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    pred = _to_btchw(pred)
    batch_size, num_frames, _, height, width = pred.shape
    zero = pred.new_zeros(())
    if num_frames < 2:
        raise ValueError(f"Depth warp loss requires at least 2 frames, got T={num_frames}")
    if gt_depth is None or gt_depth_valid_mask is None or intrinsics is None or extrinsics is None:
        raise ValueError("Depth warp loss requires gt_depth, gt_depth_valid_mask, intrinsics, and extrinsics")
    if gt_depth.ndim != 4:
        raise ValueError(f"Expected gt_depth shape [B, T, H, W], got {tuple(gt_depth.shape)}")
    if gt_depth.shape != (batch_size, num_frames, height, width):
        raise ValueError(
            f"Expected gt_depth shape {(batch_size, num_frames, height, width)}, got {tuple(gt_depth.shape)}"
        )
    if intrinsics.shape != (batch_size, num_frames, 3, 3):
        raise ValueError(f"Expected intrinsics shape [B, T, 3, 3], got {tuple(intrinsics.shape)}")
    if extrinsics.shape != (batch_size, num_frames, 4, 4):
        raise ValueError(f"Expected extrinsics shape [B, T, 4, 4], got {tuple(extrinsics.shape)}")
    if deform_mask.shape != (batch_size, num_frames, height, width):
        raise ValueError(f"Expected deform_mask shape [B, T, H, W], got {tuple(deform_mask.shape)}")

    gt_depth = gt_depth.to(device=pred.device, dtype=pred.dtype)
    intrinsics = intrinsics.to(device=pred.device, dtype=pred.dtype)
    extrinsics = extrinsics.to(device=pred.device, dtype=pred.dtype)
    deform_mask = deform_mask.to(device=pred.device)
    if gt_depth_valid_mask.shape != gt_depth.shape:
        raise ValueError(
            f"Expected gt_depth_valid_mask shape {tuple(gt_depth.shape)}, got {tuple(gt_depth_valid_mask.shape)}"
        )
    gt_depth_valid_mask = gt_depth_valid_mask.to(device=pred.device) > 0.5

    scale = _sample_scalar_per_batch(scene_flow_scale, batch_size, pred.device, pred.dtype, 1.0)
    amplification = _sample_scalar_per_batch(
        flow_target_amplification,
        batch_size,
        pred.device,
        pred.dtype,
        1.0,
    ).clamp_min(eps)

    if depth_warp_space == 'relative':
        R = scale.clamp_min(eps).view(batch_size, 1, 1, 1)
        gt_depth = gt_depth / R
        extrinsics = extrinsics.clone()
        extrinsics[:, :, :3, 3] = extrinsics[:, :, :3, 3] / R.view(batch_size, 1, 1)
        flow_scale = (1.0 / amplification).view(batch_size, 1, 1, 1, 1)
    elif depth_warp_space == 'metric':
        flow_scale = (scale / amplification).view(batch_size, 1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported depth_warp_space: {depth_warp_space}")
    warp_flow = pred[:, :-1].permute(0, 1, 3, 4, 2) * flow_scale

    depth_t = gt_depth[:, :-1]
    depth_t1 = gt_depth[:, 1:]
    intrinsics_t = intrinsics[:, :-1]
    intrinsics_t1 = intrinsics[:, 1:]
    extrinsics_t = extrinsics[:, :-1]
    extrinsics_t1 = extrinsics[:, 1:]
    pair_count = num_frames - 1

    ys, xs = torch.meshgrid(
        torch.arange(height, device=pred.device, dtype=pred.dtype),
        torch.arange(width, device=pred.device, dtype=pred.dtype),
        indexing='ij',
    )
    xs = xs.view(1, 1, height, width)
    ys = ys.view(1, 1, height, width)
    fx_t = intrinsics_t[..., 0, 0].view(batch_size, pair_count, 1, 1).clamp_min(eps)
    fy_t = intrinsics_t[..., 1, 1].view(batch_size, pair_count, 1, 1).clamp_min(eps)
    cx_t = intrinsics_t[..., 0, 2].view(batch_size, pair_count, 1, 1)
    cy_t = intrinsics_t[..., 1, 2].view(batch_size, pair_count, 1, 1)

    points_cam_t = torch.stack(
        [
            (xs - cx_t) * depth_t / fx_t,
            (ys - cy_t) * depth_t / fy_t,
            depth_t,
        ],
        dim=-1,
    )
    flat_points = points_cam_t.reshape(batch_size, pair_count, height * width, 3)
    flat_flow = warp_flow.reshape(batch_size, pair_count, height * width, 3)

    rotation_t = extrinsics_t[..., :3, :3]
    translation_t = extrinsics_t[..., :3, 3]
    points_world = torch.matmul(flat_points - translation_t.unsqueeze(-2), rotation_t)

    reference_w2c = extrinsics[:, 0:1]
    reference_rotation = reference_w2c[..., :3, :3]
    reference_translation = reference_w2c[..., :3, 3]
    points_ref = torch.matmul(points_world, reference_rotation.transpose(-1, -2)) + reference_translation.unsqueeze(-2)
    next_points_ref = points_ref + flat_flow
    next_points_world = torch.matmul(next_points_ref - reference_translation.unsqueeze(-2), reference_rotation)

    rotation_t1 = extrinsics_t1[..., :3, :3]
    translation_t1 = extrinsics_t1[..., :3, 3]
    next_points_cam = torch.matmul(next_points_world, rotation_t1.transpose(-1, -2)) + translation_t1.unsqueeze(-2)
    next_points_cam = next_points_cam.reshape(batch_size, pair_count, height, width, 3)
    predicted_depth_t1 = next_points_cam[..., 2]

    fx_t1 = intrinsics_t1[..., 0, 0].view(batch_size, pair_count, 1, 1)
    fy_t1 = intrinsics_t1[..., 1, 1].view(batch_size, pair_count, 1, 1)
    cx_t1 = intrinsics_t1[..., 0, 2].view(batch_size, pair_count, 1, 1)
    cy_t1 = intrinsics_t1[..., 1, 2].view(batch_size, pair_count, 1, 1)
    positive_depth = predicted_depth_t1 > eps
    projected_x = fx_t1 * (next_points_cam[..., 0] / predicted_depth_t1.clamp_min(eps)) + cx_t1
    projected_y = fy_t1 * (next_points_cam[..., 1] / predicted_depth_t1.clamp_min(eps)) + cy_t1
    in_bounds = (
        (projected_x >= 0.0)
        & (projected_x <= float(width - 1))
        & (projected_y >= 0.0)
        & (projected_y <= float(height - 1))
    )

    if width > 1:
        grid_x = projected_x / float(width - 1) * 2.0 - 1.0
    else:
        grid_x = projected_x.new_zeros(projected_x.shape)
    if height > 1:
        grid_y = projected_y / float(height - 1) * 2.0 - 1.0
    else:
        grid_y = projected_y.new_zeros(projected_y.shape)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(batch_size * pair_count, height, width, 2)

    sampled_depth_t1 = F.grid_sample(
        depth_t1.reshape(batch_size * pair_count, 1, height, width),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    ).reshape(batch_size, pair_count, height, width)
    sampled_valid_t1 = F.grid_sample(
        gt_depth_valid_mask[:, 1:].to(dtype=pred.dtype).reshape(batch_size * pair_count, 1, height, width),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    ).reshape(batch_size, pair_count, height, width) > 0.999

    valid = (
        (deform_mask[:, :-1] > 0.5)
        & gt_depth_valid_mask[:, :-1]
        & sampled_valid_t1
        & positive_depth
        & in_bounds
        & torch.isfinite(sampled_depth_t1)
        & torch.isfinite(predicted_depth_t1)
    )
    valid_ratio = valid.float().mean() if valid.numel() else zero
    abs_diff = (predicted_depth_t1 - sampled_depth_t1).abs()

    if valid.any():
        depth_warp_abs = abs_diff[valid].mean()
        if loss_type == 'l1':
            depth_warp_loss = depth_warp_abs
        elif loss_type == 'l2':
            depth_warp_loss = abs_diff[valid].square().mean()
        else:
            raise ValueError(f"Unsupported depth warp loss_type: {loss_type}")
    else:
        depth_warp_abs = zero
        depth_warp_loss = zero

    return {
        'loss_depth_warp': depth_warp_loss,
        'depth_warp_abs': depth_warp_abs,
        'depth_warp_valid_ratio': valid_ratio,
    }


def compute_masked_scene_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    deform_mask: torch.Tensor,
    loss_type: str = 'l1',
    zero_last_weight: float = 0.0,
    eps: float = 1e-8,
    scene_scale_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    pred = _to_btchw(pred)
    target = _to_btchw(target)

    if deform_mask.ndim != 4:
        raise ValueError(f"Expected deform_mask shape [B, T, H, W], got {tuple(deform_mask.shape)}")

    valid = deform_mask > 0.5
    valid_ratio = valid.float().mean()

    diff = pred - target
    per_pixel_l1 = diff.abs().sum(dim=2)
    per_pixel_l2 = diff.square().sum(dim=2)
    per_pixel_epe = torch.linalg.norm(diff, dim=2)

    if valid.any():
        if loss_type == 'l1':
            base_loss = per_pixel_l1[valid].mean()
        elif loss_type == 'l2':
            base_loss = per_pixel_l2[valid].mean()
        elif loss_type == 'epe':
            base_loss = per_pixel_epe[valid].mean()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        epe = per_pixel_epe[valid].mean()
    else:
        base_loss = pred.new_zeros(())
        epe = pred.new_zeros(())

    # Per-sample scene-scale weighting: loss * (1/geometry_radius).
    # This normalizes gradient contribution by scene geometry scale,
    # so indoor (small radius) and outdoor (large radius) scenes
    # contribute equally in relative terms.
    base_loss = base_loss * scene_scale_weight

    zero_last = torch.linalg.norm(pred[:, -1], dim=1).mean()
    total = base_loss + zero_last_weight * zero_last

    return {
        'loss': total,
        'loss_sf': base_loss,
        'loss_zero_last': zero_last_weight * zero_last,
        'epe': epe,
        'valid_ratio': valid_ratio.clamp_min(eps),
        'last_frame_zero_epe': zero_last,
    }


class SceneFlowLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = 'l1',
        zero_last_weight: float = 0.0,
        scale_invariant: bool = False,
        depth_warp_weight: float = 0.0,
        depth_warp_loss_type: str = 'l1',
        depth_warp_space: str = 'metric',
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.zero_last_weight = zero_last_weight
        self.scale_invariant = scale_invariant
        self.depth_warp_weight = float(depth_warp_weight)
        self.depth_warp_loss_type = depth_warp_loss_type
        self.depth_warp_space = depth_warp_space

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        deform_mask: torch.Tensor,
        scene_flow_scale: torch.Tensor | float = 0.0,
        *,
        gt_depth: Optional[torch.Tensor] = None,
        gt_depth_valid_mask: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        flow_target_amplification: torch.Tensor | float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        if scene_flow_scale is None:
            scene_flow_scale = 0.0
        # Compute scene_scale_weight = 1/geometry_radius when scale_invariant is enabled.
        scale_tensor = torch.as_tensor(scene_flow_scale, device=pred.device, dtype=pred.dtype)
        if self.scale_invariant and torch.any(scale_tensor > 0):
            scene_scale_weight = 1.0 / scale_tensor[scale_tensor > 0].mean().item()
        else:
            scene_scale_weight = 1.0

        loss_dict = compute_masked_scene_flow_loss(
            pred=pred,
            target=target,
            deform_mask=deform_mask,
            loss_type=self.loss_type,
            zero_last_weight=self.zero_last_weight,
            scene_scale_weight=scene_scale_weight,
        )
        if self.depth_warp_weight > 0.0:
            depth_warp_dict = compute_scene_flow_depth_warp_loss(
                pred=pred,
                gt_depth=gt_depth,
                gt_depth_valid_mask=gt_depth_valid_mask,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                deform_mask=deform_mask,
                scene_flow_scale=scene_flow_scale,
                flow_target_amplification=flow_target_amplification,
                loss_type=self.depth_warp_loss_type,
                depth_warp_space=self.depth_warp_space,
            )
            weighted_depth_warp = depth_warp_dict['loss_depth_warp'] * self.depth_warp_weight
            loss_dict['loss'] = loss_dict['loss'] + weighted_depth_warp
            loss_dict['loss_depth_warp'] = weighted_depth_warp
            loss_dict['depth_warp_abs'] = depth_warp_dict['depth_warp_abs']
            loss_dict['depth_warp_valid_ratio'] = depth_warp_dict['depth_warp_valid_ratio']
        else:
            zero = pred.new_zeros(())
            loss_dict['loss_depth_warp'] = zero
            loss_dict['depth_warp_abs'] = zero
            loss_dict['depth_warp_valid_ratio'] = zero
        return loss_dict
