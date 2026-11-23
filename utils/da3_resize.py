from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


DA3_PATCH_SIZE = 14


@dataclass(frozen=True)
class ResizeOnlyTransform:
    original_height: int
    original_width: int
    target_size: int
    boundary_resized_height: int
    boundary_resized_width: int
    resized_height: int
    resized_width: int
    scale_y: float
    scale_x: float

    def to_export_dict(self) -> dict[str, np.ndarray]:
        return {
            "original_height": np.array(self.original_height, dtype=np.int32),
            "original_width": np.array(self.original_width, dtype=np.int32),
            "target_size": np.array(self.target_size, dtype=np.int32),
            "boundary_resized_height": np.array(self.boundary_resized_height, dtype=np.int32),
            "boundary_resized_width": np.array(self.boundary_resized_width, dtype=np.int32),
            "resized_height": np.array(self.resized_height, dtype=np.int32),
            "resized_width": np.array(self.resized_width, dtype=np.int32),
            "scale_y": np.array(self.scale_y, dtype=np.float32),
            "scale_x": np.array(self.scale_x, dtype=np.float32),
        }


def nearest_multiple(value: int, divisor: int) -> int:
    down = (value // divisor) * divisor
    up = down + divisor
    if down <= 0:
        return max(divisor, up)
    return up if abs(up - value) <= abs(value - down) else down


def build_upper_bound_resize_transform(
    *,
    height: int,
    width: int,
    target_size: int,
    patch_size: int = DA3_PATCH_SIZE,
) -> ResizeOnlyTransform:
    longest = max(height, width)
    if longest <= 0:
        raise ValueError("Image size must be positive.")
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    scale = target_size / float(longest)
    boundary_resized_height = max(1, int(round(height * scale)))
    boundary_resized_width = max(1, int(round(width * scale)))
    resized_height = max(patch_size, nearest_multiple(boundary_resized_height, patch_size))
    resized_width = max(patch_size, nearest_multiple(boundary_resized_width, patch_size))
    return ResizeOnlyTransform(
        original_height=height,
        original_width=width,
        target_size=target_size,
        boundary_resized_height=boundary_resized_height,
        boundary_resized_width=boundary_resized_width,
        resized_height=resized_height,
        resized_width=resized_width,
        scale_y=resized_height / float(height),
        scale_x=resized_width / float(width),
    )


def apply_resize_only_to_intrinsics(
    intrinsics: torch.Tensor | np.ndarray,
    *,
    original_height: int,
    original_width: int,
    resized_height: int,
    resized_width: int,
) -> torch.Tensor | np.ndarray:
    if torch.is_tensor(intrinsics):
        resized = intrinsics.clone()
    else:
        resized = np.array(intrinsics, copy=True)

    scale_x = resized_width / float(original_width)
    scale_y = resized_height / float(original_height)
    resized[..., 0, :] *= scale_x
    resized[..., 1, :] *= scale_y
    return resized


def resize_video_frames(
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
    mode: str,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] frames, got {tuple(frames.shape)}")
    if frames.shape[-2:] == (height, width):
        return frames
    interpolate_kwargs = {
        "input": frames,
        "size": (height, width),
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    return F.interpolate(**interpolate_kwargs)


def resize_video_frames_upper_bound(
    frames: torch.Tensor,
    *,
    target_size: int,
    mode: str = "bilinear",
    patch_size: int = DA3_PATCH_SIZE,
) -> tuple[torch.Tensor, ResizeOnlyTransform]:
    transform = build_upper_bound_resize_transform(
        height=int(frames.shape[-2]),
        width=int(frames.shape[-1]),
        target_size=target_size,
        patch_size=patch_size,
    )
    resized = resize_video_frames(
        frames,
        height=transform.resized_height,
        width=transform.resized_width,
        mode=mode,
    )
    return resized, transform


def apply_resize_only_to_array(
    array: np.ndarray,
    *,
    transform: ResizeOnlyTransform,
    interpolation: int,
) -> np.ndarray:
    return cv2.resize(
        array,
        (transform.resized_width, transform.resized_height),
        interpolation=interpolation,
    )


def coerce_image_hw(image_hw: Any) -> tuple[int, int] | None:
    if image_hw is None:
        return None
    if torch.is_tensor(image_hw):
        if image_hw.dim() == 1 and image_hw.numel() == 2:
            return int(image_hw[0].item()), int(image_hw[1].item())
        if image_hw.dim() == 2 and image_hw.shape[-1] == 2:
            first = image_hw[0]
            if not torch.equal(image_hw, first.unsqueeze(0).expand_as(image_hw)):
                raise ValueError("All image_hw entries in the batch must be identical.")
            return int(first[0].item()), int(first[1].item())
        raise ValueError(f"Unsupported image_hw tensor shape: {tuple(image_hw.shape)}")
    if isinstance(image_hw, (tuple, list)) and len(image_hw) == 2:
        if all(torch.is_tensor(value) for value in image_hw):
            height, width = image_hw
            if height.dim() == 0 and width.dim() == 0:
                return int(height.item()), int(width.item())
            if height.dim() == 1 and width.dim() == 1:
                if not torch.equal(height, height[0].expand_as(height)):
                    raise ValueError("All image_hw height entries in the batch must be identical.")
                if not torch.equal(width, width[0].expand_as(width)):
                    raise ValueError("All image_hw width entries in the batch must be identical.")
                return int(height[0].item()), int(width[0].item())
            raise ValueError(
                "Unsupported collated image_hw tensor shapes: "
                f"{tuple(height.shape)} and {tuple(width.shape)}"
            )
        return int(image_hw[0]), int(image_hw[1])
    raise ValueError(f"Unsupported image_hw type: {type(image_hw)}")
