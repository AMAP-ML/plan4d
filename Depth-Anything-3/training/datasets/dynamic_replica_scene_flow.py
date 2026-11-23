from __future__ import annotations

import gzip
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.datasets.point_odyssey_scene_flow import (
    _ensure_extrinsic_4x4,
    _pad_last_frame,
    _pad_zero,
    adjust_intrinsics_for_resize_and_crop,
    compute_geometry_scale_from_points,
    compute_reference_points_from_depth_sequence,
    compute_scene_flow_scale,
    get_trunk_indices,
    normalize_scene_flow_with_scale,
    resize_and_center_crop_image,
)


PYTORCH3D_TO_OPENCV = np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))


@dataclass(frozen=True)
class DynamicReplicaClipInfo:
    scene_name: str
    camera_name: str
    frame_indices: tuple[int, ...]
    frame_annotations: tuple[Mapping[str, Any], ...]


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a json list in {path}, got {type(payload)}")
    return [dict(item) for item in payload]


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_float16_png(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(f"Failed to read float16 png from {path}")
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype != np.uint16:
        raise ValueError(f"Expected uint16 png at {path}, got {array.dtype}")
    return np.ascontiguousarray(array).view(np.float16).astype(np.float32)


def _load_depth(path: Path) -> np.ndarray:
    return _load_float16_png(path)


def _load_flow(path: Path) -> np.ndarray:
    packed = _load_float16_png(path)
    height, packed_width = packed.shape
    if packed_width % 2 != 0:
        raise ValueError(f"Expected flow png width to be even, got {packed.shape} at {path}")
    width = packed_width // 2
    return np.stack([packed[:, :width], packed[:, width:]], axis=-1).astype(np.float32)


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask from {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)


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


def _unproject_depth_at_pixels(depth: np.ndarray, intrinsics: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (xs - np.float32(cx)) * depth.astype(np.float32) / np.float32(max(fx, 1e-8))
    y = (ys - np.float32(cy)) * depth.astype(np.float32) / np.float32(max(fy, 1e-8))
    return np.stack([x, y, depth.astype(np.float32)], axis=-1).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = _ensure_extrinsic_4x4(transform)
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    flat = points.reshape(-1, 3).astype(np.float32)
    transformed = flat @ rotation.T + translation[None]
    return transformed.reshape(points.shape)


def dynamic_replica_viewpoint_to_intrinsics(viewpoint: Mapping[str, Any], image_height: int, image_width: int) -> np.ndarray:
    focal_length = np.asarray(viewpoint["focal_length"], dtype=np.float32)
    principal_point = np.asarray(viewpoint["principal_point"], dtype=np.float32)
    half_image_size = np.array([image_width, image_height], dtype=np.float32) / 2.0

    intrinsics_format = str(viewpoint["intrinsics_format"]).lower()
    if intrinsics_format == "ndc_norm_image_bounds":
        rescale = half_image_size
    elif intrinsics_format == "ndc_isotropic":
        rescale = np.float32(half_image_size.min())
    else:
        raise ValueError(f"Unsupported intrinsics_format: {viewpoint['intrinsics_format']}")

    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = float(focal_length_px[0])
    intrinsic[1, 1] = float(focal_length_px[1])
    intrinsic[0, 2] = float(principal_point_px[0])
    intrinsic[1, 2] = float(principal_point_px[1])
    return intrinsic


def dynamic_replica_viewpoint_to_extrinsic(viewpoint: Mapping[str, Any]) -> np.ndarray:
    rotation = np.asarray(viewpoint["R"], dtype=np.float32)
    translation = np.asarray(viewpoint["T"], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = PYTORCH3D_TO_OPENCV @ rotation
    extrinsic[:3, 3] = PYTORCH3D_TO_OPENCV @ translation
    return extrinsic


def reconstruct_dynamic_replica_scene_flow_pair(
    *,
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    forward_flow_t: np.ndarray,
    forward_mask_t: np.ndarray,
    intrinsics_t: np.ndarray,
    intrinsics_t1: np.ndarray,
    extrinsic_t: np.ndarray,
    extrinsic_t1: np.ndarray,
    reference_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth_t = np.asarray(depth_t, dtype=np.float32)
    depth_t1 = np.asarray(depth_t1, dtype=np.float32)
    forward_flow_t = np.asarray(forward_flow_t, dtype=np.float32)
    forward_mask_t = np.asarray(forward_mask_t, dtype=np.float32) > 0.5

    height, width = depth_t.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    target_xs = xs + forward_flow_t[..., 0]
    target_ys = ys + forward_flow_t[..., 1]

    sampled_depth_t1, target_valid = _sample_bilinear(depth_t1, target_xs, target_ys)
    point_t_cam = _unproject_depth_at_pixels(depth_t, intrinsics_t, xs, ys)
    point_t1_cam = _unproject_depth_at_pixels(sampled_depth_t1, intrinsics_t1, target_xs, target_ys)

    point_t_world = _transform_points(point_t_cam, np.linalg.inv(_ensure_extrinsic_4x4(extrinsic_t)).astype(np.float32))
    point_t1_world = _transform_points(point_t1_cam, np.linalg.inv(_ensure_extrinsic_4x4(extrinsic_t1)).astype(np.float32))
    point_t_ref = _transform_points(point_t_world, reference_w2c)
    point_t1_ref = _transform_points(point_t1_world, reference_w2c)

    scene_flow = (point_t1_ref - point_t_ref).astype(np.float32)
    valid = (
        forward_mask_t
        & target_valid
        & np.isfinite(depth_t)
        & np.isfinite(sampled_depth_t1)
        & (depth_t > 1e-6)
        & (sampled_depth_t1 > 1e-6)
        & np.isfinite(scene_flow).all(axis=-1)
    )
    scene_flow[~valid] = 0.0
    return scene_flow.astype(np.float32), valid.astype(np.float32)


class DynamicReplicaSceneFlowAdapter(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "test",
        chunk_size: int = 81,
        overlap: int = 0,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        pad_to_chunk_size: bool = True,
        flow_target_space: str = "relative",
        relative_scale_source: str = "geometry_radius",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        max_scenes: Optional[int] = None,
        camera_names: Optional[Sequence[str]] = None,
        flow_target_amplification: float = 1.0,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.target_height = target_height
        self.target_width = target_width
        self.process_res = None if process_res is None else int(process_res)
        self.process_res_method = None if process_res_method is None else str(process_res_method)
        self.pad_to_chunk_size = bool(pad_to_chunk_size)
        self.flow_target_space = str(flow_target_space)
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = str(flow_scale_stat)
        self.flow_scale_eps = float(flow_scale_eps)
        self.flow_target_amplification = float(flow_target_amplification)
        self.camera_names = None if camera_names is None else {str(name) for name in camera_names}
        self._warned_invalid_flow_pairs: set[str] = set()

        annotations_path = self.data_root / split / f"frame_annotations_{split}.jgz"
        annotations = _load_annotations(annotations_path)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for annotation in annotations:
            camera_name = str(annotation.get("camera_name", "left"))
            if self.camera_names is not None and camera_name not in self.camera_names:
                continue
            image_path = annotation.get("image", {}).get("path")
            if not image_path:
                continue
            scene_name = Path(str(image_path)).parts[0]
            grouped.setdefault(scene_name, []).append(annotation)

        scene_names = sorted(grouped.keys())
        if max_scenes is not None:
            scene_names = scene_names[: int(max_scenes)]

        self.samples: list[DynamicReplicaClipInfo] = []
        for scene_name in scene_names:
            scene_annotations = sorted(grouped[scene_name], key=lambda item: int(item["frame_number"]))
            if len(scene_annotations) < 2:
                continue
            camera_name = str(scene_annotations[0].get("camera_name", "left"))
            for start_idx, end_idx in get_trunk_indices(len(scene_annotations), self.chunk_size, self.overlap):
                clip_annotations = tuple(scene_annotations[start_idx:end_idx])
                frame_indices = tuple(int(item["frame_number"]) for item in clip_annotations)
                self.samples.append(
                    DynamicReplicaClipInfo(
                        scene_name=scene_name,
                        camera_name=camera_name,
                        frame_indices=frame_indices,
                        frame_annotations=clip_annotations,
                    )
                )

        if not self.samples:
            raise RuntimeError(f"No valid Dynamic Replica clips found under {self.data_root / split}")

    def __len__(self) -> int:
        return len(self.samples)

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
                process_res=self.process_res,
                process_res_method=self.process_res_method,
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
        if self.process_res_method is None and (src_height, src_width) == (target_height, target_width):
            return rgbs, depth, intrinsics, scene_flow, deform_mask, valid_depth_mask

        rgbs_resized = self._resize_dense_sequence(
            rgbs,
            target_height=target_height,
            target_width=target_width,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        depth_resized = self._resize_dense_sequence(
            depth,
            target_height=target_height,
            target_width=target_width,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        scene_flow_resized = self._resize_dense_sequence(
            scene_flow,
            target_height=target_height,
            target_width=target_width,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        deform_mask_resized = self._resize_dense_sequence(
            deform_mask.astype(np.float32),
            target_height=target_height,
            target_width=target_width,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        valid_depth_mask_resized = self._resize_dense_sequence(
            valid_depth_mask.astype(np.float32),
            target_height=target_height,
            target_width=target_width,
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5

        intrinsics_resized = []
        for intrinsic in intrinsics:
            _, params = resize_and_center_crop_image(
                np.zeros((src_height, src_width), dtype=np.uint8),
                target_height,
                target_width,
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                interpolation=cv2.INTER_NEAREST,
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
        sample: DynamicReplicaClipInfo,
        rgbs: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        scene_flow: np.ndarray,
        deform_mask: np.ndarray,
    ) -> Dict[str, Any]:
        native_rgbs = np.asarray(rgbs, dtype=np.uint8)
        native_depth = np.asarray(depth, dtype=np.float32)
        native_intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        native_scene_flow = np.asarray(scene_flow, dtype=np.float32)
        native_deform_mask = np.asarray(deform_mask, dtype=np.float32)
        native_valid_depth_mask = np.isfinite(native_depth) & (native_depth > 1e-6)

        rgbs, depth, intrinsics, scene_flow, deform_mask, valid_depth_mask = self._resize_sample_to_target(
            rgbs=native_rgbs,
            depth=native_depth,
            intrinsics=native_intrinsics,
            scene_flow=native_scene_flow,
            deform_mask=native_deform_mask,
            valid_depth_mask=native_valid_depth_mask,
        )

        sequence_length = int(native_rgbs.shape[0])
        frame_ids = np.asarray(sample.frame_indices, dtype=np.int64)
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

        if self.pad_to_chunk_size and sequence_length < self.chunk_size:
            rgbs = _pad_last_frame(rgbs, self.chunk_size)
            depth = _pad_last_frame(depth, self.chunk_size)
            intrinsics = _pad_last_frame(intrinsics, self.chunk_size)
            extrinsics = _pad_last_frame(extrinsics, self.chunk_size)
            scene_flow = _pad_zero(scene_flow, self.chunk_size)
            deform_mask = _pad_zero(deform_mask, self.chunk_size)
            valid_depth_mask = _pad_last_frame(valid_depth_mask.astype(np.float32), self.chunk_size) > 0.5
            native_rgbs = _pad_last_frame(native_rgbs, self.chunk_size)
            native_depth = _pad_last_frame(native_depth, self.chunk_size)
            native_intrinsics = _pad_last_frame(native_intrinsics, self.chunk_size)
            native_scene_flow = _pad_zero(native_scene_flow, self.chunk_size)
            native_deform_mask = _pad_zero(native_deform_mask, self.chunk_size)
            native_valid_depth_mask = _pad_last_frame(native_valid_depth_mask.astype(np.float32), self.chunk_size) > 0.5
            frame_ids = _pad_last_frame(frame_ids[:, None], self.chunk_size).reshape(-1)

        clip_uid = f"{sample.scene_name}:{int(sample.frame_indices[0]):05d}:{int(sample.frame_indices[-1]) + 1:05d}"
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
            "dataset_name": "dynamic_replica",
            "source_split": self.split,
            "scene_name": sample.scene_name,
            "camera_name": sample.camera_name,
            "clip_uid": clip_uid,
            "frame_ids": torch.from_numpy(frame_ids).long(),
            "quality_score": torch.tensor(1.0, dtype=torch.float32),
            "loss_weight": torch.tensor(1.0, dtype=torch.float32),
            "has_gt_flow": torch.tensor(True),
            "flow_source": "depth_and_optical_flow_reconstruction",
            "domain_tags": {"synthetic": True, "dense": True, "dynamic_stereo": True},
        }

    def _warn_invalid_flow_pair(
        self,
        sample: DynamicReplicaClipInfo,
        frame_idx: int,
        flow_path: Path,
        reason: str,
    ) -> None:
        warning_key = f"{sample.scene_name}:{sample.camera_name}:{int(sample.frame_indices[frame_idx])}:{flow_path}"
        if warning_key in self._warned_invalid_flow_pairs:
            return
        self._warned_invalid_flow_pairs.add(warning_key)
        warnings.warn(
            (
                "Skipping Dynamic Replica flow pair "
                f"{sample.scene_name} ({sample.camera_name}) frame {int(sample.frame_indices[frame_idx])} "
                f"because flow data is invalid: {reason}"
            ),
            RuntimeWarning,
        )

    def _load_sample(self, sample: DynamicReplicaClipInfo) -> Dict[str, Any]:
        rgbs: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []

        for annotation in sample.frame_annotations:
            image_path = self.data_root / self.split / str(annotation["image"]["path"])
            depth_path = self.data_root / self.split / str(annotation["depth"]["path"])
            viewpoint = annotation["viewpoint"]

            rgb = _load_rgb(image_path)
            depth = _load_depth(depth_path)
            if rgb.shape[:2] != depth.shape:
                raise ValueError(
                    f"RGB/depth shape mismatch for {image_path}: rgb={rgb.shape[:2]} depth={depth.shape}"
                )

            rgbs.append(rgb)
            depths.append(depth)
            intrinsics.append(dynamic_replica_viewpoint_to_intrinsics(viewpoint, depth.shape[0], depth.shape[1]))
            extrinsics.append(dynamic_replica_viewpoint_to_extrinsic(viewpoint))

        rgbs_arr = np.stack(rgbs, axis=0).astype(np.uint8)
        depths_arr = np.stack(depths, axis=0).astype(np.float32)
        intrinsics_arr = np.stack(intrinsics, axis=0).astype(np.float32)
        extrinsics_arr = np.stack(extrinsics, axis=0).astype(np.float32)

        sequence_length = len(sample.frame_annotations)
        image_height, image_width = depths_arr.shape[1:]
        scene_flow = np.zeros((sequence_length, image_height, image_width, 3), dtype=np.float32)
        deform_mask = np.zeros((sequence_length, image_height, image_width), dtype=np.float32)
        reference_w2c = extrinsics_arr[0]

        for frame_idx in range(sequence_length - 1):
            annotation = sample.frame_annotations[frame_idx]
            flow_path_str = annotation.get("flow_forward", {}).get("path")
            flow_mask_path_str = annotation.get("flow_forward_mask", {}).get("path")
            if not flow_path_str or not flow_mask_path_str:
                continue

            flow_path = self.data_root / self.split / str(flow_path_str)
            flow_mask_path = self.data_root / self.split / str(flow_mask_path_str)
            try:
                flow = _load_flow(flow_path)
                flow_mask = _load_binary_mask(flow_mask_path)
                pair_flow, pair_mask = reconstruct_dynamic_replica_scene_flow_pair(
                    depth_t=depths_arr[frame_idx],
                    depth_t1=depths_arr[frame_idx + 1],
                    forward_flow_t=flow,
                    forward_mask_t=flow_mask,
                    intrinsics_t=intrinsics_arr[frame_idx],
                    intrinsics_t1=intrinsics_arr[frame_idx + 1],
                    extrinsic_t=extrinsics_arr[frame_idx],
                    extrinsic_t1=extrinsics_arr[frame_idx + 1],
                    reference_w2c=reference_w2c,
                )
            except (FileNotFoundError, ValueError, IndexError, RuntimeError) as exc:
                self._warn_invalid_flow_pair(sample, frame_idx, flow_path, str(exc))
                continue
            scene_flow[frame_idx] = pair_flow
            deform_mask[frame_idx] = pair_mask

        return self._finalize_sample(
            sample=sample,
            rgbs=rgbs_arr,
            depth=depths_arr,
            intrinsics=intrinsics_arr,
            extrinsics=extrinsics_arr,
            scene_flow=scene_flow,
            deform_mask=deform_mask,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_sample(self.samples[idx])


__all__ = [
    "DynamicReplicaSceneFlowAdapter",
    "dynamic_replica_viewpoint_to_extrinsic",
    "dynamic_replica_viewpoint_to_intrinsics",
    "reconstruct_dynamic_replica_scene_flow_pair",
]
