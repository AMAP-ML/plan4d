from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


def _ensure_btchw(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dim() != 5:
        raise ValueError(f"{name} must be [B, T, C, H, W], got {tuple(tensor.shape)}")
    return tensor


def _ensure_bcthw(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if tensor.dim() != 5:
        raise ValueError(f"{name} must be [B, C, T, H, W], got {tuple(tensor.shape)}")
    return tensor


def _normalize_valid_mask(
    valid_mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones((batch_size, num_frames, 1, height, width), device=device, dtype=dtype)

    if valid_mask.dim() == 4:
        valid_mask = valid_mask.unsqueeze(2)
    if valid_mask.dim() != 5:
        raise ValueError(f"valid_mask must be [B, T, 1, H, W], got {tuple(valid_mask.shape)}")
    if valid_mask.shape[2] != 1:
        raise ValueError(f"valid_mask channel dimension must be 1, got {tuple(valid_mask.shape)}")
    if valid_mask.shape[:2] != (batch_size, num_frames):
        raise ValueError(
            "valid_mask batch/time dimensions must match prediction/target, "
            f"got {tuple(valid_mask.shape[:2])} vs {(batch_size, num_frames)}"
        )
    if valid_mask.shape[-2:] != (height, width):
        raise ValueError(
            f"valid_mask spatial shape must match prediction/target, got {tuple(valid_mask.shape[-2:])} vs {(height, width)}"
        )
    return valid_mask.to(device=device, dtype=dtype)


def _masked_mean(loss_map: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = valid_mask.expand_as(loss_map)
    denom = expanded_mask.sum().clamp(min=1.0)
    return (loss_map * expanded_mask).sum() / denom


def _downsample_btchw(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    tensor = _ensure_btchw(tensor, name="tensor")
    if scale <= 1:
        return tensor
    flat = tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    down = F.avg_pool2d(flat, kernel_size=scale, stride=scale, ceil_mode=False)
    return down.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], down.shape[-2], down.shape[-1])


def _downsample_mask_btchw(valid_mask: torch.Tensor, scale: int) -> torch.Tensor:
    valid_mask = _ensure_btchw(valid_mask, name="valid_mask")
    if scale <= 1:
        return valid_mask
    flat = valid_mask.reshape(-1, 1, valid_mask.shape[3], valid_mask.shape[4])
    down = F.max_pool2d(flat, kernel_size=scale, stride=scale, ceil_mode=False)
    return down.reshape(valid_mask.shape[0], valid_mask.shape[1], 1, down.shape[-2], down.shape[-1])


def _finite_difference_x(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., :, 1:] - tensor[..., :, :-1]


def _finite_difference_y(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[..., 1:, :] - tensor[..., :-1, :]


def _laplacian_2d(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _ensure_btchw(tensor, name="tensor")
    flat = tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=flat.device,
        dtype=flat.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(flat.shape[1], 1, 1, 1)
    lap = F.conv2d(flat, kernel, padding=1, groups=flat.shape[1])
    return lap.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], tensor.shape[4])


def per_channel_standardize_btchw(
    tensor: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    tensor = _ensure_btchw(tensor, name="tensor")
    batch_size, num_frames, channels, height, width = tensor.shape
    mask = _normalize_valid_mask(
        valid_mask,
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    expanded_mask = mask.expand_as(tensor)
    denom = expanded_mask.sum(dim=(0, 1, 3, 4), keepdim=True).clamp(min=1.0)
    mean = (tensor * expanded_mask).sum(dim=(0, 1, 3, 4), keepdim=True) / denom
    centered = (tensor - mean) * expanded_mask
    var = centered.square().sum(dim=(0, 1, 3, 4), keepdim=True) / denom
    std = var.sqrt().clamp(min=eps)
    return (tensor - mean) / std


def compute_spatiotemporal_probe_metrics(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    scales: Iterable[int] = (1, 2, 4),
    detail_floor_lambda_t: float = 1.0,
) -> dict[str, float]:
    prediction = _ensure_btchw(prediction, name="prediction")
    target = _ensure_btchw(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {tuple(prediction.shape)} and {tuple(target.shape)}")

    batch_size, num_frames, _, height, width = prediction.shape
    mask = _normalize_valid_mask(
        valid_mask,
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=prediction.device,
        dtype=prediction.dtype,
    )

    gradient_terms: list[torch.Tensor] = []
    laplacian_terms: list[torch.Tensor] = []
    temporal_terms: list[torch.Tensor] = []
    pred_energies: list[torch.Tensor] = []
    target_energies: list[torch.Tensor] = []

    for scale in tuple(int(scale) for scale in scales):
        scaled_pred = _downsample_btchw(prediction, scale)
        scaled_target = _downsample_btchw(target, scale)
        scaled_mask = _downsample_mask_btchw(mask, scale)

        grad_x_pred = _finite_difference_x(scaled_pred)
        grad_x_target = _finite_difference_x(scaled_target)
        grad_y_pred = _finite_difference_y(scaled_pred)
        grad_y_target = _finite_difference_y(scaled_target)

        mask_x = scaled_mask[..., :, 1:] * scaled_mask[..., :, :-1]
        mask_y = scaled_mask[..., 1:, :] * scaled_mask[..., :-1, :]
        gradient_terms.append(_masked_mean((grad_x_pred - grad_x_target).abs(), mask_x))
        gradient_terms.append(_masked_mean((grad_y_pred - grad_y_target).abs(), mask_y))

        lap_pred = _laplacian_2d(scaled_pred)
        lap_target = _laplacian_2d(scaled_target)
        laplacian_terms.append(_masked_mean((lap_pred - lap_target).abs(), scaled_mask))

        spatial_energy_pred = 0.5 * (
            _masked_mean(grad_x_pred.square(), mask_x) + _masked_mean(grad_y_pred.square(), mask_y)
        )
        spatial_energy_target = 0.5 * (
            _masked_mean(grad_x_target.square(), mask_x) + _masked_mean(grad_y_target.square(), mask_y)
        )

        if scaled_pred.shape[1] > 1:
            temporal_pred = scaled_pred[:, 1:] - scaled_pred[:, :-1]
            temporal_target = scaled_target[:, 1:] - scaled_target[:, :-1]
            temporal_mask = scaled_mask[:, 1:] * scaled_mask[:, :-1]
            temporal_terms.append(_masked_mean((temporal_pred - temporal_target).abs(), temporal_mask))
            temporal_energy_pred = _masked_mean(temporal_pred.square(), temporal_mask)
            temporal_energy_target = _masked_mean(temporal_target.square(), temporal_mask)
            pred_energies.append(spatial_energy_pred + detail_floor_lambda_t * temporal_energy_pred)
            target_energies.append(spatial_energy_target + detail_floor_lambda_t * temporal_energy_target)
        else:
            pred_energies.append(spatial_energy_pred)
            target_energies.append(spatial_energy_target)

    gradient_l1 = torch.stack(gradient_terms).mean() if gradient_terms else prediction.new_tensor(0.0)
    laplacian_l1 = torch.stack(laplacian_terms).mean() if laplacian_terms else prediction.new_tensor(0.0)
    temporal_diff_l1 = torch.stack(temporal_terms).mean() if temporal_terms else prediction.new_tensor(0.0)
    pred_detail_energy = torch.stack(pred_energies).mean() if pred_energies else prediction.new_tensor(0.0)
    target_detail_energy = torch.stack(target_energies).mean() if target_energies else prediction.new_tensor(0.0)
    detail_floor_absolute_gap = torch.relu(target_detail_energy - pred_detail_energy)
    target_has_detail = target_detail_energy > 1e-8
    detail_floor_ratio = torch.where(
        target_has_detail,
        pred_detail_energy / target_detail_energy.clamp(min=1e-8),
        torch.ones_like(pred_detail_energy),
    )
    detail_floor_relative_gap = torch.where(
        target_has_detail,
        torch.relu(1.0 - detail_floor_ratio),
        torch.zeros_like(pred_detail_energy),
    )

    return {
        "gradient_l1": float(gradient_l1.item()),
        "laplacian_l1": float(laplacian_l1.item()),
        "temporal_diff_l1": float(temporal_diff_l1.item()),
        "pred_detail_energy": float(pred_detail_energy.item()),
        "target_detail_energy": float(target_detail_energy.item()),
        "detail_floor_gap": float(detail_floor_relative_gap.item()),
        "detail_floor_relative_gap": float(detail_floor_relative_gap.item()),
        "detail_floor_absolute_gap": float(detail_floor_absolute_gap.item()),
        "detail_floor_ratio": float(detail_floor_ratio.item()),
    }


def compute_geometry_probe_metrics(
    *,
    pred_depth_aligned: torch.Tensor,
    target_depth: torch.Tensor,
    pred_ray_aligned: torch.Tensor,
    target_ray: torch.Tensor,
    valid_mask: torch.Tensor,
    scales: Iterable[int] = (1, 2, 4),
) -> dict[str, float]:
    depth_metrics = compute_spatiotemporal_probe_metrics(
        prediction=pred_depth_aligned,
        target=target_depth,
        valid_mask=valid_mask,
        scales=scales,
    )
    ray_metrics = compute_spatiotemporal_probe_metrics(
        prediction=pred_ray_aligned[:, :, :3],
        target=target_ray[:, :, :3],
        valid_mask=valid_mask,
        scales=scales,
    )
    result = {f"depth.{key}": value for key, value in depth_metrics.items()}
    result.update({f"ray_direction.{key}": value for key, value in ray_metrics.items()})
    return result


def apply_contiguous_time_mask(
    tensor: torch.Tensor,
    *,
    mask_ratio: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    tensor = _ensure_bcthw(tensor, name="tensor")
    num_frames = tensor.shape[2]
    masked_length = int(round(num_frames * float(mask_ratio)))
    masked_length = max(0, min(num_frames, masked_length))
    if masked_length == 0:
        return tensor.clone(), {
            "start_index": 0,
            "end_index": 0,
            "masked_length": 0,
        }

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    start = 0 if num_frames == masked_length else int(
        torch.randint(0, num_frames - masked_length + 1, (1,), generator=generator).item()
    )
    end = start + masked_length

    masked = tensor.clone()
    masked[:, :, start:end] = 0
    return masked, {
        "start_index": start,
        "end_index": end,
        "masked_length": masked_length,
    }


def apply_channel_dropout(
    tensor: torch.Tensor,
    *,
    drop_ratio: float,
    seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    tensor = _ensure_bcthw(tensor, name="tensor")
    channels = tensor.shape[1]
    dropped_count = int(round(channels * float(drop_ratio)))
    dropped_count = max(0, min(channels, dropped_count))
    if dropped_count == 0:
        return tensor.clone(), {
            "dropped_count": 0,
            "dropped_channels": [],
        }

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    dropped_channels = torch.randperm(channels, generator=generator)[:dropped_count].tolist()

    dropped = tensor.clone()
    dropped[:, dropped_channels] = 0
    return dropped, {
        "dropped_count": dropped_count,
        "dropped_channels": dropped_channels,
    }


def reshape_token_sequence_to_grid(
    tokens: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
) -> torch.Tensor:
    if tokens.dim() != 4:
        raise ValueError(f"tokens must be [B, T, N, C], got {tuple(tokens.shape)}")
    grid_h, grid_w = int(grid_hw[0]), int(grid_hw[1])
    if tokens.shape[2] != grid_h * grid_w:
        raise ValueError(
            f"Token count {tokens.shape[2]} does not match requested grid {grid_hw}"
        )
    return tokens.reshape(tokens.shape[0], tokens.shape[1], grid_h, grid_w, tokens.shape[3]).permute(0, 1, 4, 2, 3).contiguous()


def compute_channel_utilization_metrics(
    tensor: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> dict[str, object]:
    tensor = _ensure_bcthw(tensor, name="tensor")
    flat = tensor.permute(0, 2, 3, 4, 1).reshape(-1, tensor.shape[1]).to(dtype=torch.float32)
    flat = flat - flat.mean(dim=0, keepdim=True)
    channel_var = flat.var(dim=0, unbiased=False)
    total_var = channel_var.sum().clamp(min=eps)
    normalized = channel_var / total_var

    participation_ratio = float((channel_var.sum().square() / channel_var.square().sum().clamp(min=eps)).item())

    covariance = (flat.transpose(0, 1) @ flat) / max(flat.shape[0], 1)
    eigvals = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
    eigvals_sum = eigvals.sum().clamp(min=eps)
    eig_prob = eigvals / eigvals_sum
    effective_rank = float(torch.exp(-(eig_prob * torch.log(eig_prob.clamp(min=eps))).sum()).item())

    return {
        "channel_var": channel_var.detach().cpu().tolist(),
        "channel_std": channel_var.sqrt().detach().cpu().tolist(),
        "participation_ratio": participation_ratio,
        "effective_rank": effective_rank,
        "active_channels_gt_1pct": int((normalized > 0.01).sum().item()),
    }
