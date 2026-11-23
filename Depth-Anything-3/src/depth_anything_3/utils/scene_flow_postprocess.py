from __future__ import annotations

import cv2
import numpy as np
import torch


def scene_flow_filter_period(value: float | str) -> float:
    period = float(value)
    if not np.isfinite(period) or (period != 0 and period < 2):
        raise ValueError("Scene-flow filter period must be 0 or a finite value >= 2.")
    return period


def filter_scene_flow_periodic(
    flow: np.ndarray | torch.Tensor,
    period: float = 14.0,
    *,
    channel_dim: int = -3,
) -> np.ndarray | torch.Tensor:
    """Filter spatial patch harmonics in inference flow, preserving layout and dtype."""
    period = scene_flow_filter_period(period)
    if period == 0:
        return flow
    if not isinstance(flow, (np.ndarray, torch.Tensor)):
        raise TypeError("Scene flow must be a NumPy array or Torch tensor.")
    if flow.ndim not in (4, 5):
        raise ValueError("Expected scene flow with shape [T,3,H,W] or [B,T,3,H,W], or channels last.")
    if channel_dim not in (-3, -1, flow.ndim - 3, flow.ndim - 1):
        raise ValueError("channel_dim must select the first or last of the three spatial/component axes.")
    channel_axis = channel_dim % flow.ndim
    if flow.shape[channel_axis] != 3:
        raise ValueError("Scene flow must have exactly three vector components.")

    is_tensor = isinstance(flow, torch.Tensor)
    if is_tensor:
        if not flow.is_floating_point():
            raise TypeError("Scene flow must have a real floating-point dtype.")
        working_dtype = torch.float64 if flow.dtype == torch.float64 else torch.float32
        array = flow.detach().to(device="cpu", dtype=working_dtype).numpy()
    else:
        if not np.issubdtype(flow.dtype, np.floating):
            raise TypeError("Scene flow must have a real floating-point dtype.")
        array = flow.astype(np.float64 if flow.dtype == np.float64 else np.float32, copy=False)
    channel_last = np.moveaxis(array, channel_axis, -1)
    height, width = channel_last.shape[-3:-1]
    if height < 1 or width < 1:
        raise ValueError("Scene-flow spatial dimensions must be positive.")
    frames = channel_last.reshape(-1, height, width, 3)

    from scipy.fft import irfft2, rfft2

    padding = int(np.ceil(period * 2))
    padded_height, padded_width = height + 2 * padding, width + 2 * padding
    vertical_frequency = np.abs(np.fft.fftfreq(padded_height))
    horizontal_frequency = np.fft.rfftfreq(padded_width)
    bandwidth = 0.14 / period
    vertical_mask = np.ones_like(vertical_frequency)
    horizontal_mask = np.ones_like(horizontal_frequency)
    for harmonic in range(1, int(period / 2) + 1):
        frequency = harmonic / period
        vertical_mask *= 1.0 - np.exp(-0.5 * ((vertical_frequency - frequency) / bandwidth) ** 2)
        horizontal_mask *= 1.0 - np.exp(-0.5 * ((horizontal_frequency - frequency) / bandwidth) ** 2)
    mask = (vertical_mask[:, None] * horizontal_mask[None, :]).astype(np.float32)
    mask[0, 0] = 1.0

    result = np.empty_like(frames)
    for frame_index, frame in enumerate(frames):
        if not np.isfinite(frame).all():
            raise ValueError("Nonfinite scene flow cannot be passed to the periodic filter.")
        padded = cv2.copyMakeBorder(
            frame, padding, padding, padding, padding, cv2.BORDER_REFLECT_101
        )
        spectrum = rfft2(padded, axes=(0, 1), workers=1)
        restored = irfft2(
            spectrum * mask[..., None], s=(padded_height, padded_width),
            axes=(0, 1), workers=1,
        )
        cropped = restored[padding:padding + height, padding:padding + width]
        cropped += (
            np.mean(frame, axis=(0, 1), dtype=np.float64)
            - np.mean(cropped, axis=(0, 1), dtype=np.float64)
        )
        result[frame_index] = cropped

    result = np.moveaxis(result.reshape(channel_last.shape), -1, channel_axis)
    if is_tensor:
        return torch.from_numpy(np.ascontiguousarray(result)).to(device=flow.device, dtype=flow.dtype)
    return result.astype(flow.dtype, copy=False)
