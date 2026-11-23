from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from training.datasets.point_odyssey_scene_flow import (
    _ensure_extrinsic_4x4,
    _pad_last_frame,
    _pad_zero,
    adjust_intrinsics_for_resize_and_crop,
    compute_geometry_scale_from_points,
    compute_reference_points_from_depth_sequence,
    compute_scene_flow_scale,
    normalize_scene_flow_with_scale,
    resize_and_center_crop_image,
)

REQUIRED_METADATA_FILES = ("features.json", "dataset_info.json")


def _load_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow_datasets is required to read MOVi-F. Install it in the active environment first."
        ) from exc
    return tfds


def _has_tfds_metadata(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in REQUIRED_METADATA_FILES)


def resolve_builder_dir(builder_dir: str | Path) -> Path:
    candidate = Path(builder_dir).expanduser().resolve()
    if _has_tfds_metadata(candidate):
        return candidate

    matches = sorted(
        {
            metadata.parent
            for metadata in candidate.rglob("features.json")
            if (metadata.parent / "dataset_info.json").is_file()
        }
    )
    if not matches:
        required = ", ".join(REQUIRED_METADATA_FILES)
        raise FileNotFoundError(f"Could not find a TFDS builder directory with {required} under {candidate}")
    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Found multiple TFDS builder directories under {candidate}: {joined}")
    return matches[0]


def _decode_quantized(array: Any, value_range: Any | None) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if value_range is None:
        return values
    min_value, max_value = np.asarray(value_range, dtype=np.float32)
    scale = (max_value - min_value) / np.float32(65535.0)
    return values * scale + min_value


def _decode_depth(array: Any, metadata: dict[str, Any]) -> np.ndarray:
    values = np.asarray(array)
    if np.issubdtype(values.dtype, np.integer):
        values = _decode_quantized(values, metadata.get("depth_range"))
    else:
        values = values.astype(np.float32)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3:
        raise ValueError(f"Expected depth with shape [T, H, W] or [T, H, W, 1], got {values.shape}")
    return values.astype(np.float32)


def _decode_flow(array: Any, metadata: dict[str, Any], range_key: str) -> np.ndarray:
    values = np.asarray(array)
    if np.issubdtype(values.dtype, np.integer):
        values = _decode_quantized(values, metadata.get(range_key))
    else:
        values = values.astype(np.float32)
    if values.ndim != 4 or values.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape [T, H, W, 2], got {values.shape}")
    row_delta = values[..., 0]
    col_delta = values[..., 1]
    return np.stack([col_delta, row_delta], axis=-1).astype(np.float32)


def build_movi_f_intrinsics(
    *,
    focal_length: float,
    sensor_width: float,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    focal_length = float(focal_length)
    sensor_width = float(sensor_width)
    if sensor_width <= 0.0:
        raise ValueError(f"sensor_width must be positive, got {sensor_width}")
    focal_px = focal_length * float(image_width) / sensor_width
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = focal_px
    intrinsics[1, 1] = focal_px
    intrinsics[0, 2] = (float(image_width) - 1.0) * 0.5
    intrinsics[1, 2] = (float(image_height) - 1.0) * 0.5
    return intrinsics


def _normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return vector / np.float32(norm)


def _build_look_at_w2c(position: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32)
    if position.shape != (3,):
        raise ValueError(f"Camera position must have shape (3,), got {position.shape}")
    target_vec = np.zeros(3, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32)
    forward = _normalize_vector(target_vec - position)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(world_up, forward))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(world_up, forward).astype(np.float32)
    right = _normalize_vector(right)
    down = -np.cross(forward, right).astype(np.float32)
    down = _normalize_vector(down)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return np.linalg.inv(c2w).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = _ensure_extrinsic_4x4(transform)
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    transformed = flat @ rotation.T + translation[None]
    return transformed.reshape(points.shape)


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected image with 2 or 3 dims, got {image.shape}")

    height, width = image.shape[:2]
    valid = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & (xs >= 0.0)
        & (xs <= float(width - 1))
        & (ys >= 0.0)
        & (ys <= float(height - 1))
    )
    sampled_shape = xs.shape + (() if image.ndim == 2 else (image.shape[2],))
    sampled = np.zeros(sampled_shape, dtype=np.float32)
    if not np.any(valid):
        return sampled, valid

    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)

    wx = (xs - x0.astype(np.float32)).astype(np.float32)
    wy = (ys - y0.astype(np.float32)).astype(np.float32)
    w00 = (1.0 - wx) * (1.0 - wy)
    w01 = wx * (1.0 - wy)
    w10 = (1.0 - wx) * wy
    w11 = wx * wy

    if image.ndim == 2:
        sampled = (
            image[y0, x0].astype(np.float32) * w00
            + image[y0, x1].astype(np.float32) * w01
            + image[y1, x0].astype(np.float32) * w10
            + image[y1, x1].astype(np.float32) * w11
        ).astype(np.float32)
    else:
        sampled = (
            image[y0, x0].astype(np.float32) * w00[..., None]
            + image[y0, x1].astype(np.float32) * w01[..., None]
            + image[y1, x0].astype(np.float32) * w10[..., None]
            + image[y1, x1].astype(np.float32) * w11[..., None]
        ).astype(np.float32)
    sampled[~valid] = 0.0
    return sampled, valid


def _unproject_radial_depth_at_pixels(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    x_norm = (xs - np.float32(cx)) / np.float32(max(fx, 1e-8))
    y_norm = (ys - np.float32(cy)) / np.float32(max(fy, 1e-8))
    direction = np.stack([x_norm, y_norm, np.ones_like(x_norm, dtype=np.float32)], axis=-1)
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True).astype(np.float32)
    direction_norm = np.maximum(direction_norm, np.float32(1e-8))
    return direction * (depth[..., None] / direction_norm)


def reconstruct_movi_f_scene_flow_pair(
    *,
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    forward_flow_t: np.ndarray,
    backward_flow_t1: np.ndarray,
    intrinsics_t: np.ndarray,
    intrinsics_t1: np.ndarray,
    extrinsic_t: np.ndarray,
    extrinsic_t1: np.ndarray,
    reference_w2c: np.ndarray,
    fb_threshold_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    depth_t = np.asarray(depth_t, dtype=np.float32)
    depth_t1 = np.asarray(depth_t1, dtype=np.float32)
    forward_flow_t = np.asarray(forward_flow_t, dtype=np.float32)
    backward_flow_t1 = np.asarray(backward_flow_t1, dtype=np.float32)
    if depth_t.shape != depth_t1.shape:
        raise ValueError(f"depth_t and depth_t1 must have the same shape, got {depth_t.shape} and {depth_t1.shape}")
    if forward_flow_t.shape != depth_t.shape + (2,):
        raise ValueError(f"forward_flow_t must have shape {depth_t.shape + (2,)}, got {forward_flow_t.shape}")
    if backward_flow_t1.shape != depth_t.shape + (2,):
        raise ValueError(f"backward_flow_t1 must have shape {depth_t.shape + (2,)}, got {backward_flow_t1.shape}")

    height, width = depth_t.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    target_xs = xs + forward_flow_t[..., 0]
    target_ys = ys + forward_flow_t[..., 1]

    sampled_depth_t1, target_valid = _sample_bilinear(depth_t1, target_xs, target_ys)
    sampled_backward_t1, backward_valid = _sample_bilinear(backward_flow_t1, target_xs, target_ys)

    point_t_cam = _unproject_radial_depth_at_pixels(depth_t, intrinsics_t, xs, ys)
    point_t1_cam = _unproject_radial_depth_at_pixels(sampled_depth_t1, intrinsics_t1, target_xs, target_ys)

    point_t_world = _transform_points(point_t_cam, np.linalg.inv(_ensure_extrinsic_4x4(extrinsic_t)).astype(np.float32))
    point_t1_world = _transform_points(point_t1_cam, np.linalg.inv(_ensure_extrinsic_4x4(extrinsic_t1)).astype(np.float32))
    point_t_ref = _transform_points(point_t_world, reference_w2c)
    point_t1_ref = _transform_points(point_t1_world, reference_w2c)

    scene_flow = (point_t1_ref - point_t_ref).astype(np.float32)

    fb_error = np.linalg.norm(forward_flow_t + sampled_backward_t1, axis=-1).astype(np.float32)
    valid = (
        target_valid
        & backward_valid
        & np.isfinite(depth_t)
        & np.isfinite(sampled_depth_t1)
        & (depth_t > 1e-6)
        & (sampled_depth_t1 > 1e-6)
        & np.isfinite(scene_flow).all(axis=-1)
        & (fb_error <= np.float32(fb_threshold_px))
    )
    scene_flow[~valid] = 0.0
    return scene_flow.astype(np.float32), valid.astype(bool)


def _load_num_examples(builder_dir: Path, split: str) -> Optional[int]:
    info_path = builder_dir / "dataset_info.json"
    if not info_path.is_file():
        return None
    with open(info_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    for split_info in payload.get("splits", []):
        if str(split_info.get("name")) != str(split):
            continue
        if "numExamples" in split_info:
            return int(split_info["numExamples"])
        shard_lengths = split_info.get("shardLengths")
        if shard_lengths:
            return int(sum(int(length) for length in shard_lengths))
    return None


def build_movi_f_manifest_entries(
    *,
    builder_dir: str | Path,
    splits: Sequence[str],
    chunk_size: int,
    max_scenes: Optional[int] = None,
    pad_to_chunk_size: bool = True,
    fb_threshold_px: float = 1.0,
) -> list["SceneFlowManifestEntry"]:
    from training.datasets.multi_scene_flow import SceneFlowManifestEntry

    resolved_builder_dir = resolve_builder_dir(builder_dir)
    entries: list[SceneFlowManifestEntry] = []
    for split in splits:
        split = str(split)
        num_examples = _load_num_examples(resolved_builder_dir, split)
        if num_examples is None:
            continue
        if max_scenes is not None:
            num_examples = min(int(num_examples), int(max_scenes))
        frame_indices = tuple(range(int(chunk_size)))
        for sample_index in range(int(num_examples)):
            scene_id = f"{split}_{sample_index:06d}"
            entries.append(
                SceneFlowManifestEntry(
                    dataset_name="movi_f",
                    source_split=split,
                    scene_id=scene_id,
                    scene_relpath=str(Path(split) / scene_id),
                    clip_uid=f"{split}:{sample_index:06d}",
                    frame_indices=frame_indices,
                    adapter_kwargs={
                        "builder_dir": str(resolved_builder_dir),
                        "split": split,
                        "sample_index": int(sample_index),
                        "pad_to_chunk_size": bool(pad_to_chunk_size),
                        "fb_threshold_px": float(fb_threshold_px),
                    },
                    has_gt_flow=True,
                    flow_source="depth_and_optical_flow_reconstruction",
                    enabled_for_train=True,
                    quality_score=1.0,
                    loss_weight=1.0,
                    domain_tags={"synthetic": True, "dense": True, "movi_f": True},
                    meta={"num_frames": int(chunk_size), "sample_index": int(sample_index)},
                )
            )
    return entries


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    value = metadata.get(key, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stringify_clip_uid(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class MOViFSceneFlowAdapter(IterableDataset):
    def __init__(
        self,
        *,
        builder_dir: str | Path,
        split: str = "validation",
        flow_target_space: str = "metric",
        relative_scale_source: str = "flow_p95",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        chunk_size: Optional[int] = None,
        pad_to_chunk_size: bool = True,
        fb_threshold_px: float = 1.0,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        flow_target_amplification: float = 1.0,
    ) -> None:
        super().__init__()
        self.builder_dir = resolve_builder_dir(builder_dir)
        self.split = str(split)
        self.flow_target_space = str(flow_target_space)
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = str(flow_scale_stat)
        self.flow_scale_eps = float(flow_scale_eps)
        self.flow_target_amplification = float(flow_target_amplification)
        self.chunk_size = None if chunk_size is None else int(chunk_size)
        self.pad_to_chunk_size = bool(pad_to_chunk_size and self.chunk_size is not None)
        self.fb_threshold_px = float(fb_threshold_px)
        self.target_height = None if target_height is None else int(target_height)
        self.target_width = None if target_width is None else int(target_width)
        if self.flow_target_space not in {"metric", "relative"}:
            raise ValueError(f"Unsupported flow_target_space: {self.flow_target_space}")
        if self.relative_scale_source not in {"flow_p95", "geometry_radius"}:
            raise ValueError(f"Unsupported relative_scale_source: {self.relative_scale_source}")
        self._num_examples = _load_num_examples(self.builder_dir, self.split)

    def __len__(self) -> int:
        if self._num_examples is None:
            raise TypeError("MOViFSceneFlowAdapter length is unknown")
        return self._num_examples

    def _build_intrinsics_sequence(self, camera: dict[str, Any], image_height: int, image_width: int, num_frames: int) -> np.ndarray:
        intrinsics = build_movi_f_intrinsics(
            focal_length=float(np.asarray(camera["focal_length"]).item()),
            sensor_width=float(np.asarray(camera["sensor_width"]).item()),
            image_height=image_height,
            image_width=image_width,
        )
        return np.repeat(intrinsics[None], num_frames, axis=0).astype(np.float32)

    def _build_extrinsics_sequence(self, camera: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(camera["positions"], dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"camera.positions must have shape [T, 3], got {positions.shape}")
        extrinsics = np.stack([_build_look_at_w2c(position) for position in positions], axis=0)
        return extrinsics.astype(np.float32)

    def _validate_target_size(self, image_height: int, image_width: int) -> None:
        if self.target_height is not None and self.target_height <= 0:
            raise ValueError(f"target_height must be positive, got {self.target_height}")
        if self.target_width is not None and self.target_width <= 0:
            raise ValueError(f"target_width must be positive, got {self.target_width}")

    def _resize_dense_sequence(
        self,
        array: np.ndarray,
        *,
        target_height: int,
        target_width: int,
        interpolation: int,
    ) -> np.ndarray:
        frames: list[np.ndarray] = []
        for frame in np.asarray(array):
            resized, _ = resize_and_center_crop_image(
                frame,
                target_height,
                target_width,
                interpolation=interpolation,
            )
            frames.append(resized)
        return np.stack(frames, axis=0)

    def _resize_sample_to_target(
        self,
        *,
        rgbs: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        scene_flow: np.ndarray,
        deform_mask: np.ndarray,
        valid_depth_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        src_height = int(rgbs.shape[1])
        src_width = int(rgbs.shape[2])
        target_height = src_height if self.target_height is None else int(self.target_height)
        target_width = src_width if self.target_width is None else int(self.target_width)
        if (src_height, src_width) == (target_height, target_width):
            return rgbs, depth, intrinsics, scene_flow, deform_mask, valid_depth_mask

        rgbs_resized = self._resize_dense_sequence(
            rgbs,
            target_height=target_height,
            target_width=target_width,
            interpolation=1,
        ).astype(np.uint8)
        depth_resized = self._resize_dense_sequence(
            depth,
            target_height=target_height,
            target_width=target_width,
            interpolation=0,
        ).astype(np.float32)
        scene_flow_resized = self._resize_dense_sequence(
            scene_flow,
            target_height=target_height,
            target_width=target_width,
            interpolation=1,
        ).astype(np.float32)
        deform_mask_resized = self._resize_dense_sequence(
            deform_mask.astype(np.float32),
            target_height=target_height,
            target_width=target_width,
            interpolation=0,
        ).astype(np.float32)
        valid_depth_mask_resized = self._resize_dense_sequence(
            valid_depth_mask.astype(np.float32),
            target_height=target_height,
            target_width=target_width,
            interpolation=0,
        ) > 0.5

        intrinsics_resized = []
        for intrinsic in intrinsics:
            _, params = resize_and_center_crop_image(
                np.zeros((src_height, src_width), dtype=np.uint8),
                target_height,
                target_width,
                interpolation=0,
            )
            resize_height, resize_width, crop_top, crop_left, _, _ = params
            intrinsics_resized.append(
                adjust_intrinsics_for_resize_and_crop(
                    intrinsic,
                    src_height,
                    src_width,
                    resize_height,
                    resize_width,
                    crop_top,
                    crop_left,
                )
            )

        return (
            rgbs_resized,
            depth_resized,
            np.stack(intrinsics_resized, axis=0).astype(np.float32),
            scene_flow_resized,
            deform_mask_resized,
            valid_depth_mask_resized,
        )

    def _finalize_sample(
        self,
        *,
        example: dict[str, Any],
        rgbs: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        scene_flow: np.ndarray,
        deform_mask: np.ndarray,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        native_rgbs = np.asarray(rgbs, dtype=np.uint8)
        native_depth = np.asarray(depth, dtype=np.float32)
        native_intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        native_scene_flow = np.asarray(scene_flow, dtype=np.float32)
        native_deform_mask = np.asarray(deform_mask, dtype=np.float32)
        native_valid_depth_mask = np.isfinite(native_depth) & (native_depth > 1e-6)

        rgbs = native_rgbs
        depth = native_depth
        intrinsics = native_intrinsics
        scene_flow = native_scene_flow
        deform_mask = native_deform_mask
        rgbs, depth, intrinsics, scene_flow, deform_mask, valid_depth_mask = self._resize_sample_to_target(
            rgbs=rgbs,
            depth=depth,
            intrinsics=intrinsics,
            scene_flow=scene_flow,
            deform_mask=deform_mask,
            valid_depth_mask=native_valid_depth_mask,
        )

        sequence_length = int(native_rgbs.shape[0])
        frame_ids = np.arange(sequence_length, dtype=np.int64)
        flow_p95_scale = compute_scene_flow_scale(
            native_scene_flow,
            native_deform_mask,
            self.flow_scale_stat,
            self.flow_scale_eps,
        )
        scene_flow_scale = flow_p95_scale
        if self.flow_target_space == "relative":
            if self.relative_scale_source == "geometry_radius":
                reference_points = compute_reference_points_from_depth_sequence(
                    depths=native_depth,
                    intrinsics=native_intrinsics,
                    extrinsics=extrinsics,
                    valid_masks=native_valid_depth_mask.astype(np.float32),
                    reference_w2c=extrinsics[0],
                )
                scene_flow_scale = compute_geometry_scale_from_points(
                    reference_points,
                    native_valid_depth_mask.astype(np.float32),
                    scale_stat=self.geometry_scale_stat,
                    scale_eps=self.flow_scale_eps,
                )
            native_scene_flow = normalize_scene_flow_with_scale(native_scene_flow, scene_flow_scale)
            scene_flow = normalize_scene_flow_with_scale(scene_flow, scene_flow_scale)
        if self.flow_target_amplification != 1.0:
            amp = np.float32(self.flow_target_amplification)
            native_scene_flow = native_scene_flow * amp
            scene_flow = scene_flow * amp

        if self.pad_to_chunk_size and sequence_length < int(self.chunk_size):
            pad_length = int(self.chunk_size)
            rgbs = _pad_last_frame(rgbs, pad_length)
            depth = _pad_last_frame(depth, pad_length)
            intrinsics = _pad_last_frame(intrinsics, pad_length)
            extrinsics = _pad_last_frame(extrinsics, pad_length)
            scene_flow = _pad_zero(scene_flow, pad_length)
            deform_mask = _pad_zero(deform_mask, pad_length)
            valid_depth_mask = _pad_last_frame(valid_depth_mask.astype(np.float32), pad_length) > 0.5
            native_rgbs = _pad_last_frame(native_rgbs, pad_length)
            native_depth = _pad_last_frame(native_depth, pad_length)
            native_intrinsics = _pad_last_frame(native_intrinsics, pad_length)
            native_scene_flow = _pad_zero(native_scene_flow, pad_length)
            native_deform_mask = _pad_zero(native_deform_mask, pad_length)
            native_valid_depth_mask = _pad_last_frame(native_valid_depth_mask.astype(np.float32), pad_length) > 0.5
            frame_ids = _pad_last_frame(frame_ids[:, None], pad_length).reshape(-1)

        clip_uid = _stringify_clip_uid(_metadata_value(metadata, "video_name", "unknown"))
        return {
            "rgbs": torch.from_numpy(rgbs),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "extrinsics": torch.from_numpy(extrinsics).float(),
            "scene_flow": torch.from_numpy(scene_flow).float(),
            "deform_mask": torch.from_numpy(deform_mask).float(),
            "gt_depth": torch.from_numpy(depth).float(),
            "gt_depth_valid_mask": torch.from_numpy(valid_depth_mask.astype(np.float32)).float(),
            "eval_rgbs": torch.from_numpy(native_rgbs),
            "eval_intrinsics": torch.from_numpy(native_intrinsics).float(),
            "eval_scene_flow": torch.from_numpy(native_scene_flow).float(),
            "eval_deform_mask": torch.from_numpy(native_deform_mask).float(),
            "eval_gt_depth": torch.from_numpy(native_depth).float(),
            "eval_gt_depth_valid_mask": torch.from_numpy(native_valid_depth_mask.astype(np.float32)).float(),
            "scene_flow_scale": torch.tensor(float(scene_flow_scale), dtype=torch.float32),
            "flow_p95_scale": torch.tensor(float(flow_p95_scale), dtype=torch.float32),
            "flow_target_amplification": torch.tensor(float(self.flow_target_amplification), dtype=torch.float32),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
            "dataset_name": "movi_f",
            "source_split": self.split,
            "scene_name": clip_uid,
            "clip_uid": clip_uid,
            "frame_ids": torch.from_numpy(frame_ids).long(),
            "quality_score": torch.tensor(1.0, dtype=torch.float32),
            "loss_weight": torch.tensor(1.0, dtype=torch.float32),
            "has_gt_flow": torch.tensor(True),
            "flow_source": "depth_and_optical_flow_reconstruction",
            "domain_tags": {"synthetic": True, "dense": True},
            "meta": {
                "num_frames": int(_metadata_value(metadata, "num_frames", sequence_length)),
                "height": int(_metadata_value(metadata, "height", rgbs.shape[1])),
                "width": int(_metadata_value(metadata, "width", rgbs.shape[2])),
                "camera_positions": np.asarray(example["camera"]["positions"], dtype=np.float32),
                "camera_quaternions": np.asarray(example["camera"]["quaternions"], dtype=np.float32),
            },
        }

    def _convert_example(self, example: dict[str, Any]) -> dict[str, Any]:
        rgbs = np.asarray(example["video"], dtype=np.uint8)
        if rgbs.ndim != 4 or rgbs.shape[-1] != 3:
            raise ValueError(f"Expected video with shape [T, H, W, 3], got {rgbs.shape}")
        metadata = dict(example.get("metadata") or {})
        depth = _decode_depth(example["depth"], metadata)
        forward_flow = _decode_flow(example["forward_flow"], metadata, "forward_flow_range")
        backward_flow = _decode_flow(example["backward_flow"], metadata, "backward_flow_range")

        sequence_length, image_height, image_width = depth.shape
        self._validate_target_size(image_height, image_width)

        intrinsics = self._build_intrinsics_sequence(example["camera"], image_height, image_width, sequence_length)
        extrinsics = self._build_extrinsics_sequence(example["camera"])

        scene_flow = np.zeros((sequence_length, image_height, image_width, 3), dtype=np.float32)
        deform_mask = np.zeros((sequence_length, image_height, image_width), dtype=np.float32)
        reference_w2c = extrinsics[0]
        for frame_idx in range(sequence_length - 1):
            pair_flow, pair_mask = reconstruct_movi_f_scene_flow_pair(
                depth_t=depth[frame_idx],
                depth_t1=depth[frame_idx + 1],
                forward_flow_t=forward_flow[frame_idx],
                backward_flow_t1=backward_flow[frame_idx + 1],
                intrinsics_t=intrinsics[frame_idx],
                intrinsics_t1=intrinsics[frame_idx + 1],
                extrinsic_t=extrinsics[frame_idx],
                extrinsic_t1=extrinsics[frame_idx + 1],
                reference_w2c=reference_w2c,
                fb_threshold_px=self.fb_threshold_px,
            )
            scene_flow[frame_idx] = pair_flow
            deform_mask[frame_idx] = pair_mask.astype(np.float32)

        return self._finalize_sample(
            example=example,
            rgbs=rgbs,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            scene_flow=scene_flow,
            deform_mask=deform_mask,
            metadata=metadata,
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        tfds = _load_tfds()
        builder = tfds.builder_from_directory(str(self.builder_dir))
        dataset = builder.as_dataset(split=self.split)
        worker_info = get_worker_info()

        for example_index, example in enumerate(tfds.as_numpy(dataset)):
            if worker_info is not None and (example_index % worker_info.num_workers) != worker_info.id:
                continue
            yield self._convert_example(example)


__all__ = [
    "MOViFSceneFlowAdapter",
    "build_movi_f_manifest_entries",
    "build_movi_f_intrinsics",
    "reconstruct_movi_f_scene_flow_pair",
    "resolve_builder_dir",
]
