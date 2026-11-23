from __future__ import annotations

import csv
import gzip
import json
import warnings
import zipfile
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.datasets.dynamic_replica_scene_flow import DynamicReplicaClipInfo, DynamicReplicaSceneFlowAdapter
from training.datasets.movi_f_scene_flow import MOViFSceneFlowAdapter, _load_tfds
from training.datasets.point_odyssey_scene_flow import (
    PointOdysseySceneFlowDataset,
    SampleInfo,
    _ensure_extrinsic_4x4,
    _pad_last_frame,
    _pad_zero,
    adjust_intrinsics_for_resize_and_crop,
    compute_geometry_scale_from_points,
    compute_reference_points_from_depth_sequence,
    compute_resize_params,
    compute_scene_flow_scale,
    get_trunk_indices,
    normalize_scene_flow_with_scale,
)

BLENDER_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
SCENE_FLOW_MANIFEST_VERSION = 1
_MANIFEST_RUNTIME_KEYS = {"scene_dir", "variation_dir", "target_height", "target_width", "chunk_size"}
_MANIFEST_PATH_KEY_BY_DATASET = {
    "point_odyssey": "scene_dir",
    "blinkvision": "scene_dir",
    "omniworld": "scene_dir",
    "virtual_kitti2": "variation_dir",
}


@dataclass(frozen=True)
class SceneFlowManifestEntry:
    dataset_name: str
    source_split: str
    scene_id: str
    clip_uid: str
    frame_indices: Tuple[int, ...]
    scene_relpath: str = ""
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    has_gt_flow: bool = True
    flow_source: str = "metric_gt"
    enabled_for_train: bool = True
    quality_score: float = 1.0
    loss_weight: float = 1.0
    domain_tags: Dict[str, bool] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def _resolve_scene_relpath(self, data_root: str | Path | None = None) -> str:
        if self.scene_relpath:
            return str(self.scene_relpath)
        if data_root is None:
            raise ValueError("data_root is required to derive scene_relpath")
        path_key = _MANIFEST_PATH_KEY_BY_DATASET.get(self.dataset_name)
        if path_key is None or path_key not in self.adapter_kwargs:
            raise ValueError(f"Cannot derive scene_relpath for dataset {self.dataset_name}")
        scene_path = Path(self.adapter_kwargs[path_key]).resolve()
        return str(scene_path.relative_to(Path(data_root).resolve()))

    def to_json_dict(self, data_root: str | Path | None = None) -> Dict[str, Any]:
        adapter_kwargs = {
            key: value
            for key, value in dict(self.adapter_kwargs).items()
            if key not in _MANIFEST_RUNTIME_KEYS
        }
        return {
            "manifest_version": SCENE_FLOW_MANIFEST_VERSION,
            "dataset_name": self.dataset_name,
            "source_split": self.source_split,
            "scene_id": self.scene_id,
            "scene_relpath": self._resolve_scene_relpath(data_root),
            "clip_uid": self.clip_uid,
            "frame_indices": [int(frame_id) for frame_id in self.frame_indices],
            "adapter_kwargs": _jsonify_value(adapter_kwargs),
            "has_gt_flow": bool(self.has_gt_flow),
            "flow_source": self.flow_source,
            "enabled_for_train": bool(self.enabled_for_train),
            "quality_score": float(self.quality_score),
            "loss_weight": float(self.loss_weight),
            "domain_tags": {str(key): bool(value) for key, value in dict(self.domain_tags).items()},
            "meta": _jsonify_value(dict(self.meta)),
        }

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        data_root: str | Path | None = None,
    ) -> "SceneFlowManifestEntry":
        dataset_name = str(payload["dataset_name"])
        adapter_kwargs = dict(payload.get("adapter_kwargs") or {})
        scene_relpath = str(payload.get("scene_relpath", ""))
        if data_root is not None and scene_relpath:
            path_key = _MANIFEST_PATH_KEY_BY_DATASET.get(dataset_name)
            if path_key is not None:
                adapter_kwargs[path_key] = str((Path(data_root) / scene_relpath).resolve())
        if dataset_name == "dynamic_replica" and data_root is not None:
            adapter_kwargs["data_root"] = str(Path(data_root).expanduser().resolve())
        return cls(
            dataset_name=dataset_name,
            source_split=str(payload["source_split"]),
            scene_id=str(payload["scene_id"]),
            clip_uid=str(payload["clip_uid"]),
            frame_indices=tuple(int(frame_id) for frame_id in payload.get("frame_indices", [])),
            scene_relpath=scene_relpath,
            adapter_kwargs=adapter_kwargs,
            has_gt_flow=bool(payload.get("has_gt_flow", True)),
            flow_source=str(payload.get("flow_source", "metric_gt")),
            enabled_for_train=bool(payload.get("enabled_for_train", True)),
            quality_score=float(payload.get("quality_score", 1.0)),
            loss_weight=float(payload.get("loss_weight", 1.0)),
            domain_tags={str(key): bool(value) for key, value in dict(payload.get("domain_tags") or {}).items()},
            meta=dict(payload.get("meta") or {}),
        )


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonify_value(item) for item in value]
    return value


def save_scene_flow_manifest(
    entries: Sequence[SceneFlowManifestEntry],
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry.to_json_dict(data_root), ensure_ascii=False))
            file.write("\n")


def load_scene_flow_manifest(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> List[SceneFlowManifestEntry]:
    entries: List[SceneFlowManifestEntry] = []
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            entries.append(SceneFlowManifestEntry.from_json_dict(json.loads(line), data_root=data_root))
    return entries


def _read_npz_array_shape(npz_path: Path, key: str) -> Optional[Tuple[int, ...]]:
    member_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        if member_name not in archive.namelist():
            return None
        with archive.open(member_name) as file:
            magic = file.read(6)
            if magic != b"\x93NUMPY":
                return None
            major = file.read(1)[0]
            _minor = file.read(1)[0]
            header_len = int.from_bytes(file.read(2 if major == 1 else 4), "little")
            header = literal_eval(file.read(header_len).decode("latin1"))
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        return None
    return tuple(int(dim) for dim in shape)


def _validate_point_odyssey_annotation_header(npz_path: Path) -> Tuple[bool, str, int]:
    intrinsics_shape = _read_npz_array_shape(npz_path, "intrinsics")
    if intrinsics_shape is None or len(intrinsics_shape) != 3 or tuple(intrinsics_shape[1:]) != (3, 3):
        return False, "intrinsics", 0
    total_frames = int(intrinsics_shape[0])

    extrinsics_shape = _read_npz_array_shape(npz_path, "extrinsics")
    if extrinsics_shape is None or len(extrinsics_shape) != 3 or tuple(extrinsics_shape[1:]) not in {(4, 4), (3, 4)}:
        return False, "extrinsics", total_frames

    trajs_2d_shape = _read_npz_array_shape(npz_path, "trajs_2d")
    if trajs_2d_shape is None or len(trajs_2d_shape) != 3 or trajs_2d_shape[0] < total_frames or trajs_2d_shape[-1] != 2:
        return False, "trajs_2d", total_frames

    trajs_3d_shape = _read_npz_array_shape(npz_path, "trajs_3d")
    if trajs_3d_shape is None or len(trajs_3d_shape) != 3 or trajs_3d_shape[0] < total_frames or trajs_3d_shape[-1] != 3:
        return False, "trajs_3d", total_frames

    if trajs_2d_shape[1] != trajs_3d_shape[1]:
        return False, "trajs_points_mismatch", total_frames

    valids_shape = _read_npz_array_shape(npz_path, "valids")
    if valids_shape is None or len(valids_shape) != 2 or valids_shape[0] < total_frames or valids_shape[1] != trajs_3d_shape[1]:
        return False, "valids", total_frames

    visibs_shape = _read_npz_array_shape(npz_path, "visibs")
    if visibs_shape is not None and (
        len(visibs_shape) != 2 or visibs_shape[0] < total_frames or visibs_shape[1] != trajs_3d_shape[1]
    ):
        return False, "visibs", total_frames

    return True, "", total_frames


def build_point_odyssey_manifest_entries(
    *,
    data_root: str | Path,
    splits: Sequence[str],
    chunk_size: int,
    overlap: int,
    max_scenes: Optional[int] = None,
) -> List[SceneFlowManifestEntry]:
    root = Path(data_root)
    entries: List[SceneFlowManifestEntry] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            warnings.warn(f"PointOdyssey split not found: {split_dir}", RuntimeWarning)
            continue
        scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if max_scenes is not None:
            scene_dirs = scene_dirs[:int(max_scenes)]
        for scene_dir in scene_dirs:
            anno_path = scene_dir / "anno.npz"
            if not anno_path.exists():
                continue
            is_valid, invalid_reason, total_frames = _validate_point_odyssey_annotation_header(anno_path)
            if not is_valid:
                warnings.warn(
                    f"Skipping scene {scene_dir.name}: malformed {invalid_reason} in {anno_path}",
                    RuntimeWarning,
                )
                continue
            for start_idx, end_idx in get_trunk_indices(total_frames, chunk_size, overlap):
                entries.append(
                    SceneFlowManifestEntry(
                        dataset_name="point_odyssey",
                        source_split=str(split),
                        scene_id=scene_dir.name,
                        clip_uid=f"{scene_dir.name}:{start_idx}:{end_idx}",
                        frame_indices=tuple(range(start_idx, end_idx)),
                        scene_relpath=str(scene_dir.relative_to(root)),
                        adapter_kwargs={"scene_dir": str(scene_dir)},
                        has_gt_flow=True,
                        flow_source="traj3d_rasterized",
                        enabled_for_train=True,
                        quality_score=1.0,
                        domain_tags={"synthetic": True, "indoor": False, "outdoor": True, "sparse": True},
                        meta={"num_frames": total_frames},
                    )
                )
    return entries


def _dynamic_replica_clip_uid(scene_name: str, frame_indices: Sequence[int]) -> str:
    return f"{scene_name}:{int(frame_indices[0]):05d}:{int(frame_indices[-1]) + 1:05d}"


def _dynamic_replica_real_has_dense_flow(real_root: Path) -> bool:
    for annotation_path in sorted(real_root.glob("*/test/frame_annotations_test.jgz")):
        with gzip.open(annotation_path, "rt", encoding="utf-8") as file:
            payload = json.load(file)
        for annotation in payload:
            flow_info = annotation.get("flow_forward") or {}
            flow_mask_info = annotation.get("flow_forward_mask") or {}
            if flow_info.get("path") and flow_mask_info.get("path"):
                return True
    return False


def build_dynamic_replica_manifest_entries(
    *,
    data_root: str | Path,
    splits: Sequence[str],
    chunk_size: int,
    overlap: int,
    max_scenes: Optional[int] = None,
    camera_names: Optional[Sequence[str]] = None,
) -> List[SceneFlowManifestEntry]:
    root = Path(data_root).expanduser().resolve()
    entries: List[SceneFlowManifestEntry] = []

    for split in splits:
        split = str(split)
        annotations_path = root / split / f"frame_annotations_{split}.jgz"
        if not annotations_path.exists():
            if split == "real":
                if _dynamic_replica_real_has_dense_flow(root / "real"):
                    warnings.warn(
                        "Dynamic Replica real split stores nested per-scene annotations and is not supported by manifest export yet.",
                        RuntimeWarning,
                    )
                else:
                    warnings.warn(
                        "Skipping Dynamic Replica real split because its annotations do not provide flow_forward/flow_forward_mask for dense scene-flow supervision.",
                        RuntimeWarning,
                    )
            else:
                warnings.warn(f"Dynamic Replica split not found: {annotations_path}", RuntimeWarning)
            continue

        try:
            dataset = DynamicReplicaSceneFlowAdapter(
                data_root=root,
                split=split,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                target_height=None,
                target_width=None,
                pad_to_chunk_size=True,
                flow_target_space="metric",
                relative_scale_source="geometry_radius",
                geometry_scale_stat="mean_radius",
                flow_scale_stat="p95",
                flow_scale_eps=1e-6,
                max_scenes=max_scenes,
                camera_names=camera_names,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            warnings.warn(f"Skipping Dynamic Replica split {split}: {exc}", RuntimeWarning)
            continue

        for sample in dataset.samples:
            frame_indices = tuple(int(frame_id) for frame_id in sample.frame_indices)
            if len(frame_indices) <= 1:
                continue
            entries.append(
                SceneFlowManifestEntry(
                    dataset_name="dynamic_replica",
                    source_split=split,
                    scene_id=sample.scene_name,
                    scene_relpath=str(Path(split) / sample.scene_name),
                    clip_uid=_dynamic_replica_clip_uid(sample.scene_name, frame_indices),
                    frame_indices=frame_indices,
                    adapter_kwargs={
                        "data_root": str(root),
                        "split": split,
                        "camera_name": sample.camera_name,
                        "overlap": int(overlap),
                    },
                    has_gt_flow=True,
                    flow_source="depth_and_optical_flow_reconstruction",
                    enabled_for_train=True,
                    quality_score=1.0,
                    loss_weight=1.0,
                    domain_tags={"synthetic": True, "dense": True, "dynamic_stereo": True},
                    meta={"camera_name": sample.camera_name},
                )
            )

    return entries


def compute_dataset_sampling_probs(
    dataset_sizes: Sequence[int],
    alpha: float,
    dataset_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if not dataset_sizes:
        raise ValueError("dataset_sizes must not be empty")
    sizes = np.asarray(dataset_sizes, dtype=np.float64)
    if np.any(sizes <= 0):
        raise ValueError(f"dataset_sizes must all be positive, got {dataset_sizes}")
    weights = np.power(sizes, float(alpha))
    if dataset_weights is not None:
        scale = np.asarray(dataset_weights, dtype=np.float64)
        if scale.shape != weights.shape:
            raise ValueError("dataset_weights shape mismatch")
        if np.any(scale <= 0):
            raise ValueError(f"dataset_weights must all be positive, got {dataset_weights}")
        weights *= scale
    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        raise ValueError(f"Invalid dataset sampling weights: {weights}")
    return weights / weights_sum


def _resize_crop(
    array: np.ndarray,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
    crop_height: int,
    crop_width: int,
    interpolation: int,
) -> np.ndarray:
    resized = cv2.resize(array, (resize_width, resize_height), interpolation=interpolation)
    return resized[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]


def _resize_crop_flow(
    flow_uv: np.ndarray,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
    crop_height: int,
    crop_width: int,
) -> np.ndarray:
    src_height, src_width = flow_uv.shape[:2]
    resized = cv2.resize(flow_uv, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized[..., 0] *= np.float32(resize_width / max(src_width, 1))
    resized[..., 1] *= np.float32(resize_height / max(src_height, 1))
    return resized[crop_top:crop_top + crop_height, crop_left:crop_left + crop_width]


def _resize_crop_mask(
    mask: np.ndarray,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
    crop_height: int,
    crop_width: int,
) -> np.ndarray:
    return _resize_crop(
        mask.astype(np.float32),
        resize_height,
        resize_width,
        crop_top,
        crop_left,
        crop_height,
        crop_width,
        interpolation=cv2.INTER_NEAREST,
    ) > 0.5


def _unproject_from_pixels(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pixels_x: np.ndarray,
    pixels_y: np.ndarray,
) -> np.ndarray:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (pixels_x - np.float32(cx)) * depth / np.float32(max(fx, 1e-8))
    y = (pixels_y - np.float32(cy)) * depth / np.float32(max(fy, 1e-8))
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


def _unproject_depth_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    return _unproject_from_pixels(depth, intrinsics, xs, ys)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = _ensure_extrinsic_4x4(transform)
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    flat = points.reshape(-1, 3).astype(np.float32)
    transformed = flat @ rotation.T + translation[None]
    return transformed.reshape(points.shape)


def flow_depth_to_metric_scene_flow(
    intrinsics_t: np.ndarray,
    intrinsics_t1: np.ndarray,
    extrinsics_w2c: np.ndarray,
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    flow_uv: np.ndarray,
    valid_mask_t: np.ndarray,
    valid_mask_t1: np.ndarray,
    occlusion_mask: Optional[np.ndarray],
    reference_w2c: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_t = np.asarray(depth_t, dtype=np.float32)
    depth_t1 = np.asarray(depth_t1, dtype=np.float32)
    flow_uv = np.asarray(flow_uv, dtype=np.float32)
    valid_mask_t = np.asarray(valid_mask_t)
    valid_mask_t1 = np.asarray(valid_mask_t1)
    if extrinsics_w2c.shape[0] < 2:
        raise ValueError(f"extrinsics_w2c must contain at least 2 frames, got {extrinsics_w2c.shape}")
    if flow_uv.shape[:2] != depth_t.shape or flow_uv.shape[-1] != 2:
        raise ValueError(f"flow_uv must have shape [H, W, 2], got {flow_uv.shape}")
    if depth_t1.shape != depth_t.shape:
        raise ValueError(f"depth_t, depth_t1, valid_mask_t, and valid_mask_t1 must share the same shape, got {depth_t.shape}, {depth_t1.shape}, {valid_mask_t.shape}, {valid_mask_t1.shape}")
    if valid_mask_t.shape != depth_t.shape or valid_mask_t1.shape != depth_t.shape:
        raise ValueError(f"depth_t, depth_t1, valid_mask_t, and valid_mask_t1 must share the same shape, got {depth_t.shape}, {depth_t1.shape}, {valid_mask_t.shape}, {valid_mask_t1.shape}")

    height, width = depth_t.shape
    reference_w2c = _ensure_extrinsic_4x4(reference_w2c if reference_w2c is not None else extrinsics_w2c[0])
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    map_x = xs + flow_uv[..., 0].astype(np.float32)
    map_y = ys + flow_uv[..., 1].astype(np.float32)
    in_bounds = (map_x >= 0.0) & (map_x <= np.float32(width - 1)) & (map_y >= 0.0) & (map_y <= np.float32(height - 1))

    warped_depth_t1 = cv2.remap(
        depth_t1.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_valid_t1 = cv2.remap(
        valid_mask_t1.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ) > 0.5

    valid = (
        valid_mask_t.astype(bool)
        & warped_valid_t1.astype(bool)
        & in_bounds
        & np.isfinite(depth_t)
        & np.isfinite(warped_depth_t1)
        & (depth_t > 1e-6)
        & (warped_depth_t1 > 1e-6)
    )
    if occlusion_mask is not None:
        occ = occlusion_mask[..., 0] if occlusion_mask.ndim == 3 and occlusion_mask.shape[-1] == 1 else occlusion_mask
        valid &= np.asarray(occ, dtype=np.float32) < 0.5

    current_c2w = np.linalg.inv(_ensure_extrinsic_4x4(extrinsics_w2c[0])).astype(np.float32)
    next_c2w = np.linalg.inv(_ensure_extrinsic_4x4(extrinsics_w2c[1])).astype(np.float32)
    points_t_cam = _unproject_depth_map(depth_t.astype(np.float32), intrinsics_t.astype(np.float32))
    points_t1_cam = _unproject_from_pixels(
        warped_depth_t1.astype(np.float32),
        intrinsics_t1.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
    )
    points_t_ref = _transform_points(_transform_points(points_t_cam, current_c2w), reference_w2c)
    points_t1_ref = _transform_points(_transform_points(points_t1_cam, next_c2w), reference_w2c)
    scene_flow = (points_t1_ref - points_t_ref).astype(np.float32)
    scene_flow[~valid] = 0.0
    return scene_flow, valid.astype(bool)


def _safe_npz(path: Path, key: Optional[str] = None) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        if key is not None:
            return np.asarray(payload[key])
        if not payload.files:
            raise ValueError(f"No arrays found in {path}")
        return np.asarray(payload[payload.files[0]])


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz[None, :]
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternions with last dim 4, got {quat_wxyz.shape}")

    quat_norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    quat_norm = np.clip(quat_norm, 1e-8, None)
    quat_wxyz = quat_wxyz / quat_norm

    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    matrix = np.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        axis=-1,
    ).reshape(quat_wxyz.shape[:-1] + (3, 3))
    return matrix.astype(np.float32)


def _euler_xyz_to_matrix(euler_xyz: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in euler_xyz]
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def blinkvision_pose_to_w2c(frame_pose: np.ndarray, unit_scale: float) -> np.ndarray:
    translation = np.asarray(frame_pose[0], dtype=np.float32)
    rotation_matrix = _euler_xyz_to_matrix(np.asarray(frame_pose[1], dtype=np.float32))
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation_matrix.T
    w2c[:3, 3] = -rotation_matrix.T @ translation
    w2c = BLENDER_TO_OPENCV @ w2c
    w2c[:3, 3] *= np.float32(unit_scale)
    return w2c


def decode_vkitti_scene_flow(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read scene flow from {path}")
    return ((2.0 / (2**16 - 1.0) * bgr[..., ::-1].astype(np.float32)) - 1.0) * 10.0


def decode_omniworld_flow_png(flow_u_path: Path, flow_v_path: Path) -> np.ndarray:
    flow_u = cv2.imread(str(flow_u_path), cv2.IMREAD_UNCHANGED)
    flow_v = cv2.imread(str(flow_v_path), cv2.IMREAD_UNCHANGED)
    if flow_u is None:
        raise FileNotFoundError(f"Failed to read OmniWorld flow_u from {flow_u_path}")
    if flow_v is None:
        raise FileNotFoundError(f"Failed to read OmniWorld flow_v from {flow_v_path}")
    flow_u = ((flow_u.astype(np.float32) / 65535.0) * 600.0) - 300.0
    flow_v = ((flow_v.astype(np.float32) / 65535.0) * 600.0) - 300.0
    return np.stack([flow_u, flow_v], axis=-1).astype(np.float32)


def decode_omniworld_depth(path: Path, metric_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    depth_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(f"Failed to read OmniWorld depth from {path}")

    depth_png = depth_png.astype(np.float32)
    normalized = depth_png / 65535.0
    near_mask = normalized < 0.0015
    far_mask = depth_png > 65500.0
    near, far = 1.0, 1000.0
    depth = normalized / (far - normalized * (far - near)) / 0.004
    depth *= np.float32(metric_scale)

    valid = (~near_mask) & (~far_mask) & np.isfinite(depth) & (depth > 1e-6)
    depth = depth.astype(np.float32)
    depth[~valid] = -1.0
    return depth, valid.astype(bool)


def split_into_consecutive_runs(frame_indices: Sequence[int]) -> List[Tuple[int, ...]]:
    if not frame_indices:
        return []
    ordered = [int(frame_id) for frame_id in frame_indices]
    runs: List[Tuple[int, ...]] = []
    current_run = [ordered[0]]
    for frame_id in ordered[1:]:
        if frame_id == current_run[-1] + 1:
            current_run.append(frame_id)
            continue
        runs.append(tuple(current_run))
        current_run = [frame_id]
    runs.append(tuple(current_run))
    return runs


def _parse_omniworld_requested_split_ids(splits: Sequence[Any]) -> Optional[set[int]]:
    if not splits:
        return None

    requested: set[int] = set()
    for split in splits:
        if isinstance(split, str) and split.lower() in {"all", "*"}:
            return None
        if isinstance(split, str) and split.startswith("split_"):
            requested.add(int(split.split("_", 1)[1]))
            continue
        requested.add(int(split))
    return requested


def _load_omniworld_metadata(root: Path) -> Dict[str, Dict[str, str]]:
    metadata_path = root / "metadata" / "omniworld_game_metadata.csv"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return {str(row["UID"]): dict(row) for row in reader if row.get("UID")}


def _parse_omniworld_metric_scale(metadata_row: Optional[Mapping[str, Any]]) -> float:
    if not metadata_row:
        return 1.0
    raw_value = metadata_row.get("Metric Scale")
    if raw_value in {None, ""}:
        return 1.0
    return float(raw_value)


def _load_vkitti_depth(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth from {path}")
    invalid = depth > 65500
    depth = depth.astype(np.float32) / 100.0
    valid = (~invalid) & np.isfinite(depth) & (depth > 1e-6)
    return depth, valid.astype(bool)


def _rotate_vector_map_to_reference(vector_map: np.ndarray, current_w2c: np.ndarray, reference_w2c: np.ndarray) -> np.ndarray:
    current_w2c = _ensure_extrinsic_4x4(current_w2c)
    reference_w2c = _ensure_extrinsic_4x4(reference_w2c)
    current_c2w_rot = current_w2c[:3, :3].T.astype(np.float32)
    reference_rot = reference_w2c[:3, :3].astype(np.float32)
    return np.einsum("ij,hwj->hwi", reference_rot @ current_c2w_rot, vector_map.astype(np.float32)).astype(np.float32)


def _vkitti2_scene_flow_to_reference(
    raw_scene_flow: np.ndarray,
    depth_t: np.ndarray,
    intrinsics_t: np.ndarray,
    current_w2c: np.ndarray,
    next_w2c: np.ndarray,
    reference_w2c: np.ndarray,
) -> np.ndarray:
    """Convert VKITTI2 mixed-frame scene flow to reference-frame displacement.

    VKITTI2 defines sf = P(cam_{t+1}) - P(cam_t) with endpoints in different
    camera frames, so we must reproject each endpoint separately.
    """
    p_cam_t = _unproject_depth_map(depth_t.astype(np.float32), intrinsics_t.astype(np.float32))
    p_cam_t1 = p_cam_t + raw_scene_flow.astype(np.float32)
    c2w_t = np.linalg.inv(_ensure_extrinsic_4x4(current_w2c)).astype(np.float32)
    c2w_t1 = np.linalg.inv(_ensure_extrinsic_4x4(next_w2c)).astype(np.float32)
    ref_w2c = _ensure_extrinsic_4x4(reference_w2c)
    p_ref_t = _transform_points(_transform_points(p_cam_t, c2w_t), ref_w2c)
    p_ref_t1 = _transform_points(_transform_points(p_cam_t1, c2w_t1), ref_w2c)
    return (p_ref_t1 - p_ref_t).astype(np.float32)


class BaseManifestSceneFlowAdapter(Dataset):
    def __init__(
        self,
        entries: Sequence[SceneFlowManifestEntry],
        target_height: int,
        target_width: int,
        chunk_size: int,
        flow_target_space: str,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        relative_scale_source: str = "flow_p95",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        pad_to_chunk_size: bool = True,
        flow_target_amplification: float = 1.0,
    ) -> None:
        super().__init__()
        self.entries = list(entries)
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.chunk_size = int(chunk_size)
        self.flow_target_space = flow_target_space
        self.process_res = None if process_res is None else int(process_res)
        self.process_res_method = None if process_res_method is None else str(process_res_method)
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = flow_scale_stat
        self.flow_scale_eps = float(flow_scale_eps)
        self.pad_to_chunk_size = bool(pad_to_chunk_size)
        self.flow_target_amplification = float(flow_target_amplification)
        if self.flow_target_space not in {"metric", "relative"}:
            raise ValueError(f"Unsupported flow_target_space: {self.flow_target_space}")
        if self.relative_scale_source not in {"flow_p95", "geometry_radius"}:
            raise ValueError(f"Unsupported relative_scale_source: {self.relative_scale_source}")

    def __len__(self) -> int:
        return len(self.entries)

    def _finalize_sample(
        self,
        entry: SceneFlowManifestEntry,
        rgbs: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        scene_flow: np.ndarray,
        deform_mask: np.ndarray,
        frame_ids: Sequence[int],
        depths: Optional[np.ndarray] = None,
        valid_masks: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        rgbs = np.asarray(rgbs, dtype=np.uint8)
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        extrinsics = np.asarray(extrinsics, dtype=np.float32)
        scene_flow = np.asarray(scene_flow, dtype=np.float32)
        deform_mask = np.asarray(deform_mask, dtype=np.float32)
        depths_arr = None if depths is None else np.asarray(depths, dtype=np.float32)
        valid_masks_arr = None if valid_masks is None else np.asarray(valid_masks, dtype=np.float32)
        frame_ids_np = np.asarray(frame_ids, dtype=np.int64)
        sequence_length = int(rgbs.shape[0])
        flow_p95_scale = compute_scene_flow_scale(scene_flow, deform_mask, self.flow_scale_stat, self.flow_scale_eps)
        scene_flow_scale = flow_p95_scale
        # Always compute geometry_radius for scale-invariant loss weighting.
        geometry_radius = float(self.flow_scale_eps)
        if depths_arr is not None and valid_masks_arr is not None:
            reference_points = compute_reference_points_from_depth_sequence(
                depths=depths_arr,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                valid_masks=valid_masks_arr,
                reference_w2c=extrinsics[0],
            )
            geometry_radius = compute_geometry_scale_from_points(
                reference_points,
                valid_masks_arr,
                scale_stat=self.geometry_scale_stat,
                scale_eps=self.flow_scale_eps,
            )
        if self.flow_target_space == "relative":
            if self.relative_scale_source == "geometry_radius":
                scene_flow_scale = geometry_radius
            scene_flow = normalize_scene_flow_with_scale(scene_flow, scene_flow_scale)
        if self.flow_target_amplification != 1.0:
            scene_flow = scene_flow * np.float32(self.flow_target_amplification)
        if self.pad_to_chunk_size and sequence_length < self.chunk_size:
            rgbs = _pad_last_frame(rgbs, self.chunk_size)
            intrinsics = _pad_last_frame(intrinsics, self.chunk_size)
            extrinsics = _pad_last_frame(extrinsics, self.chunk_size)
            scene_flow = _pad_zero(scene_flow, self.chunk_size)
            deform_mask = _pad_zero(deform_mask, self.chunk_size)
            if depths_arr is not None:
                depths_arr = _pad_last_frame(depths_arr, self.chunk_size)
            if valid_masks_arr is not None:
                valid_masks_arr = _pad_last_frame(valid_masks_arr, self.chunk_size)
            frame_ids_np = _pad_last_frame(frame_ids_np[:, None], self.chunk_size).reshape(-1)
        sample_dict = {
            "rgbs": torch.from_numpy(rgbs),
            "intrinsics": torch.from_numpy(intrinsics).float(),
            "extrinsics": torch.from_numpy(extrinsics).float(),
            "scene_flow": torch.from_numpy(scene_flow).float(),
            "scene_flow_scale": torch.tensor(scene_flow_scale, dtype=torch.float32),
            "flow_p95_scale": torch.tensor(flow_p95_scale, dtype=torch.float32),
            "geometry_radius": torch.tensor(float(geometry_radius), dtype=torch.float32),
            "flow_target_amplification": torch.tensor(float(self.flow_target_amplification), dtype=torch.float32),
            "deform_mask": torch.from_numpy(deform_mask).float(),
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
            "dataset_name": entry.dataset_name,
            "source_split": entry.source_split,
            "scene_name": entry.scene_id,
            "clip_uid": entry.clip_uid,
            "frame_ids": torch.from_numpy(frame_ids_np).long(),
            "quality_score": torch.tensor(float(entry.quality_score), dtype=torch.float32),
            "loss_weight": torch.tensor(float(entry.loss_weight), dtype=torch.float32),
            "has_gt_flow": torch.tensor(bool(entry.has_gt_flow)),
            "flow_source": entry.flow_source,
            "domain_tags": {key: bool(value) for key, value in entry.domain_tags.items()},
            "meta": dict(entry.meta),
        }
        if depths_arr is not None and valid_masks_arr is not None:
            sample_dict["gt_depth"] = torch.from_numpy(depths_arr).float()
            sample_dict["gt_depth_valid_mask"] = torch.from_numpy(valid_masks_arr).float()
        return sample_dict


class BlinkVisionSceneFlowAdapter(BaseManifestSceneFlowAdapter):
    def __init__(self, *args: Any, default_rgb_dir: str = "clean_uint8", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.default_rgb_dir = default_rgb_dir
        self._scene_cache: Dict[Path, Dict[str, Any]] = {}

    @staticmethod
    def build_manifest_entries(
        *,
        data_root: str | Path,
        splits: Sequence[str],
        chunk_size: int,
        overlap: int,
        target_height: int,
        target_width: int,
        max_scenes_per_split: Optional[int] = None,
        max_depth_m: float = 30.0,
        rgb_dir: str = "clean_uint8",
    ) -> List[SceneFlowManifestEntry]:
        root = Path(data_root)
        entries: List[SceneFlowManifestEntry] = []
        for split in splits:
            split_dir = root / split
            if not split_dir.exists():
                warnings.warn(f"BlinkVision split not found: {split_dir}", RuntimeWarning)
                continue
            scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            if max_scenes_per_split is not None:
                scene_dirs = scene_dirs[:int(max_scenes_per_split)]
            for scene_dir in scene_dirs:
                meta_path = scene_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path, "r", encoding="utf-8") as file:
                    metadata = json.load(file)
                num_frames = int(metadata.get("num_frames", 0))
                if num_frames <= 1:
                    continue
                if not BlinkVisionSceneFlowAdapter._scene_has_required_assets(
                    scene_dir,
                    num_frames=num_frames,
                    rgb_dir=rgb_dir,
                ):
                    continue
                for start_idx, end_idx in get_trunk_indices(num_frames, chunk_size, overlap):
                    entries.append(
                        SceneFlowManifestEntry(
                            dataset_name="blinkvision",
                            source_split=split,
                            scene_id=scene_dir.name,
                            clip_uid=f"{split}:{scene_dir.name}:{start_idx}:{end_idx}",
                            frame_indices=tuple(range(start_idx, end_idx)),
                            adapter_kwargs={
                                "scene_dir": str(scene_dir),
                                "target_height": int(target_height),
                                "target_width": int(target_width),
                                "chunk_size": int(chunk_size),
                                "max_depth_m": float(max_depth_m),
                                "rgb_dir": rgb_dir,
                            },
                            has_gt_flow=True,
                            flow_source="flow_depth_reprojected",
                            enabled_for_train=True,
                            quality_score=1.0,
                            domain_tags={"synthetic": True, "indoor": split.startswith("indoor"), "outdoor": split.startswith("outdoor"), "sparse": False},
                            meta={"num_frames": num_frames},
                        )
                    )
        return entries

    @staticmethod
    def _scene_has_required_assets(
        scene_dir: Path,
        *,
        num_frames: int,
        rgb_dir: str,
    ) -> bool:
        rgb_root = scene_dir / rgb_dir
        depth_root = scene_dir / "depth"
        flow_root = scene_dir / "forward_flow"
        for frame_id in range(int(num_frames)):
            if not (rgb_root / f"{frame_id:06d}.png").exists():
                return False
            if not (depth_root / f"{frame_id:06d}.npz").exists():
                return False
            if frame_id < int(num_frames) - 1 and not (flow_root / f"{frame_id:06d}.npz").exists():
                return False
        return True

    def _load_scene_cache(self, scene_dir: Path) -> Dict[str, Any]:
        if scene_dir in self._scene_cache:
            return self._scene_cache[scene_dir]
        with open(scene_dir / "metadata.json", "r", encoding="utf-8") as file:
            metadata = json.load(file)
        payload = np.load(scene_dir / "poses.npz", allow_pickle=True)
        cache = {
            "metadata": metadata,
            "camera_poses": np.asarray(payload["camera_poses"], dtype=np.float32),
            "intrinsics": np.asarray(metadata["K_matrix"], dtype=np.float32),
            "unit_scale": float(metadata.get("unit_scale", 1.0)),
        }
        self._scene_cache[scene_dir] = cache
        return cache

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        scene_dir = Path(entry.adapter_kwargs["scene_dir"])
        cache = self._load_scene_cache(scene_dir)
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        rgb_dir = scene_dir / entry.adapter_kwargs.get("rgb_dir", self.default_rgb_dir)

        first_rgb = cv2.imread(str(rgb_dir / f"{frame_ids[0]:06d}.png"), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise FileNotFoundError(f"Missing BlinkVision RGB at {rgb_dir / f'{frame_ids[0]:06d}.png'}")
        first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
        src_height, src_width = first_rgb.shape[:2]
        resize_height, resize_width, crop_top, crop_left, crop_height, crop_width = compute_resize_params(
            src_height,
            src_width,
            self.target_height,
            self.target_width,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )

        rgbs, intrinsics, extrinsics, depths, valid_masks = [], [], [], [], []
        adjusted_intrinsic = adjust_intrinsics_for_resize_and_crop(
            cache["intrinsics"],
            src_height,
            src_width,
            resize_height,
            resize_width,
            crop_top,
            crop_left,
        ).astype(np.float32)
        max_depth_m = float(entry.adapter_kwargs.get("max_depth_m", 30.0))

        for frame_id in frame_ids:
            rgb = cv2.imread(str(rgb_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Missing BlinkVision RGB frame: {scene_dir}:{frame_id}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = _resize_crop(rgb, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_LINEAR)
            depth = _safe_npz(scene_dir / "depth" / f"{frame_id:06d}.npz", "depth").astype(np.float32)
            depth = _resize_crop(depth, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = np.isfinite(depth) & (depth > 1e-6) & (depth < max_depth_m)

            rgbs.append(rgb.astype(np.uint8))
            intrinsics.append(adjusted_intrinsic.copy())
            depths.append(depth)
            valid_masks.append(valid_mask.astype(bool))
            extrinsics.append(blinkvision_pose_to_w2c(cache["camera_poses"][frame_id], cache["unit_scale"]))

        rgbs_arr = np.stack(rgbs, axis=0).astype(np.uint8)
        intrinsics_arr = np.stack(intrinsics, axis=0).astype(np.float32)
        extrinsics_arr = np.stack(extrinsics, axis=0).astype(np.float32)
        scene_flow = np.zeros((len(frame_ids), crop_height, crop_width, 3), dtype=np.float32)
        deform_mask = np.zeros((len(frame_ids), crop_height, crop_width), dtype=np.float32)

        for local_idx in range(len(frame_ids) - 1):
            frame_id = frame_ids[local_idx]
            flow_uv = _safe_npz(scene_dir / "forward_flow" / f"{frame_id:06d}.npz", "forward_flow").astype(np.float32)
            flow_uv = _resize_crop_flow(flow_uv, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width)
            occ = None
            occ_path = scene_dir / "occlusion_map" / f"{frame_id:06d}.npz"
            if occ_path.exists():
                occ = _safe_npz(occ_path, "occlusion_map").astype(np.float32)
                occ = _resize_crop(occ, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_NEAREST).astype(np.float32)
                # BlinkVision stores visibility in this field: 1 means visible, 0 means occluded.
                occ = 1.0 - occ
            pair_flow, pair_mask = flow_depth_to_metric_scene_flow(
                intrinsics_t=intrinsics_arr[local_idx],
                intrinsics_t1=intrinsics_arr[local_idx + 1],
                extrinsics_w2c=extrinsics_arr[local_idx:local_idx + 2],
                depth_t=depths[local_idx],
                depth_t1=depths[local_idx + 1],
                flow_uv=flow_uv,
                valid_mask_t=valid_masks[local_idx],
                valid_mask_t1=valid_masks[local_idx + 1],
                occlusion_mask=occ,
                reference_w2c=extrinsics_arr[0],
            )
            scene_flow[local_idx] = pair_flow
            deform_mask[local_idx] = pair_mask.astype(np.float32)

        return self._finalize_sample(
            entry,
            rgbs_arr,
            intrinsics_arr,
            extrinsics_arr,
            scene_flow,
            deform_mask,
            frame_ids,
            depths=np.stack(depths, axis=0).astype(np.float32),
            valid_masks=np.stack(valid_masks, axis=0).astype(np.float32),
        )


class VirtualKITTI2SceneFlowAdapter(BaseManifestSceneFlowAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cache: Dict[Tuple[Path, str, str], Dict[int, np.ndarray]] = {}

    @staticmethod
    def build_manifest_entries(
        *,
        data_root: str | Path,
        chunk_size: int,
        overlap: int,
        target_height: int,
        target_width: int,
        scenes: Optional[Sequence[str]] = None,
        variations: Optional[Sequence[str]] = None,
        cameras: Optional[Sequence[str]] = None,
    ) -> List[SceneFlowManifestEntry]:
        root = Path(data_root)
        scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("Scene"))
        if scenes is not None:
            scene_dirs = [path for path in scene_dirs if path.name in set(scenes)]
        entries: List[SceneFlowManifestEntry] = []
        for scene_dir in scene_dirs:
            variation_dirs = sorted(path for path in scene_dir.iterdir() if path.is_dir())
            if variations is not None:
                variation_dirs = [path for path in variation_dirs if path.name in set(variations)]
            for variation_dir in variation_dirs:
                rgb_root = variation_dir / "frames" / "rgb"
                if not rgb_root.exists():
                    continue
                camera_dirs = sorted(path for path in rgb_root.iterdir() if path.is_dir())
                if cameras is not None:
                    camera_dirs = [path for path in camera_dirs if path.name in set(cameras)]
                for camera_dir in camera_dirs:
                    frame_paths = sorted(camera_dir.glob("rgb_*.jpg"))
                    if len(frame_paths) <= 1:
                        continue
                    for start_idx, end_idx in get_trunk_indices(len(frame_paths), chunk_size, overlap):
                        entries.append(
                            SceneFlowManifestEntry(
                                dataset_name="virtual_kitti2",
                                source_split="all",
                                scene_id=f"{scene_dir.name}/{variation_dir.name}/{camera_dir.name}",
                                clip_uid=f"{scene_dir.name}:{variation_dir.name}:{camera_dir.name}:{start_idx}:{end_idx}",
                                frame_indices=tuple(range(start_idx, end_idx)),
                                adapter_kwargs={
                                    "variation_dir": str(variation_dir),
                                    "camera": camera_dir.name,
                                    "target_height": int(target_height),
                                    "target_width": int(target_width),
                                    "chunk_size": int(chunk_size),
                                },
                                has_gt_flow=True,
                                flow_source="forward_scene_flow_png",
                                enabled_for_train=True,
                                quality_score=1.0,
                                domain_tags={"synthetic": True, "indoor": False, "outdoor": True, "sparse": False},
                                meta={"num_frames": len(frame_paths)},
                            )
                        )
        return entries

    def _parse_intrinsics(self, variation_dir: Path, camera: str) -> Dict[int, np.ndarray]:
        cache_key = (variation_dir, camera, "intrinsics")
        if cache_key in self._cache:
            return self._cache[cache_key]
        result: Dict[int, np.ndarray] = {}
        camera_id = int(camera.split("_")[-1])
        with open(variation_dir / "intrinsic.txt", "r", encoding="utf-8") as file:
            next(file)
            for line in file:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                frame, cam = int(parts[0]), int(parts[1])
                if cam != camera_id:
                    continue
                fx, fy, cx, cy = map(float, parts[2:6])
                result[frame] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        self._cache[cache_key] = result
        return result

    def _parse_extrinsics(self, variation_dir: Path, camera: str) -> Dict[int, np.ndarray]:
        cache_key = (variation_dir, camera, "extrinsics")
        if cache_key in self._cache:
            return self._cache[cache_key]
        result: Dict[int, np.ndarray] = {}
        camera_id = int(camera.split("_")[-1])
        with open(variation_dir / "extrinsic.txt", "r", encoding="utf-8") as file:
            next(file)
            for line in file:
                parts = line.strip().split()
                if len(parts) < 18:
                    continue
                frame, cam = int(parts[0]), int(parts[1])
                if cam != camera_id:
                    continue
                values = np.asarray([float(value) for value in parts[2:18]], dtype=np.float32).reshape(4, 4)
                result[frame] = values
        self._cache[cache_key] = result
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        variation_dir = Path(entry.adapter_kwargs["variation_dir"])
        camera = str(entry.adapter_kwargs["camera"])
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)

        rgb_dir = variation_dir / "frames" / "rgb" / camera
        first_rgb = cv2.imread(str(rgb_dir / f"rgb_{frame_ids[0]:05d}.jpg"), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise FileNotFoundError(f"Missing Virtual KITTI 2 RGB at {rgb_dir / f'rgb_{frame_ids[0]:05d}.jpg'}")
        first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
        src_height, src_width = first_rgb.shape[:2]
        resize_height, resize_width, crop_top, crop_left, crop_height, crop_width = compute_resize_params(
            src_height,
            src_width,
            self.target_height,
            self.target_width,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )

        intrinsics_by_frame = self._parse_intrinsics(variation_dir, camera)
        extrinsics_by_frame = self._parse_extrinsics(variation_dir, camera)
        rgbs, intrinsics, extrinsics, depths, valid_masks, raw_scene_flows = [], [], [], [], [], []

        for local_idx, frame_id in enumerate(frame_ids):
            rgb = cv2.imread(str(rgb_dir / f"rgb_{frame_id:05d}.jpg"), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Missing Virtual KITTI 2 RGB frame: {variation_dir}:{camera}:{frame_id}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = _resize_crop(rgb, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_LINEAR)

            depth, valid_mask = _load_vkitti_depth(
                variation_dir / "frames" / "depth" / camera / f"depth_{frame_id:05d}.png"
            )
            depth = _resize_crop(
                depth,
                resize_height,
                resize_width,
                crop_top,
                crop_left,
                crop_height,
                crop_width,
                cv2.INTER_NEAREST,
            ).astype(np.float32)
            valid_mask = _resize_crop_mask(valid_mask, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width)
            intrinsic = adjust_intrinsics_for_resize_and_crop(
                intrinsics_by_frame[frame_id],
                src_height,
                src_width,
                resize_height,
                resize_width,
                crop_top,
                crop_left,
            ).astype(np.float32)

            rgbs.append(rgb.astype(np.uint8))
            intrinsics.append(intrinsic)
            extrinsics.append(_ensure_extrinsic_4x4(extrinsics_by_frame[frame_id]))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(bool))

            if local_idx < len(frame_ids) - 1:
                raw_scene_flow = decode_vkitti_scene_flow(
                    variation_dir / "frames" / "forwardSceneFlow" / camera / f"sceneFlow_{frame_id:05d}.png"
                )
                raw_scene_flow = _resize_crop(
                    raw_scene_flow,
                    resize_height,
                    resize_width,
                    crop_top,
                    crop_left,
                    crop_height,
                    crop_width,
                    cv2.INTER_LINEAR,
                ).astype(np.float32)
            else:
                raw_scene_flow = np.zeros((crop_height, crop_width, 3), dtype=np.float32)
            raw_scene_flows.append(raw_scene_flow)

        rgbs_arr = np.stack(rgbs, axis=0).astype(np.uint8)
        intrinsics_arr = np.stack(intrinsics, axis=0).astype(np.float32)
        extrinsics_arr = np.stack(extrinsics, axis=0).astype(np.float32)
        scene_flow = np.zeros((len(frame_ids), crop_height, crop_width, 3), dtype=np.float32)
        deform_mask = np.zeros((len(frame_ids), crop_height, crop_width), dtype=np.float32)

        for local_idx in range(len(frame_ids) - 1):
            ref_flow = _vkitti2_scene_flow_to_reference(
                raw_scene_flows[local_idx],
                depth_t=depths[local_idx],
                intrinsics_t=intrinsics_arr[local_idx],
                current_w2c=extrinsics_arr[local_idx],
                next_w2c=extrinsics_arr[local_idx + 1],
                reference_w2c=extrinsics_arr[0],
            )
            valid = valid_masks[local_idx] & np.isfinite(ref_flow).all(axis=-1)
            ref_flow[~valid] = 0.0
            scene_flow[local_idx] = ref_flow
            deform_mask[local_idx] = valid.astype(np.float32)

        return self._finalize_sample(
            entry,
            rgbs_arr,
            intrinsics_arr,
            extrinsics_arr,
            scene_flow,
            deform_mask,
            frame_ids,
            depths=np.stack(depths, axis=0).astype(np.float32),
            valid_masks=np.stack(valid_masks, axis=0).astype(np.float32),
        )


class OmniWorldSceneFlowAdapter(BaseManifestSceneFlowAdapter):
    def __init__(self, *args: Any, default_pose_source: str = "camera", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.default_pose_source = str(default_pose_source)
        self._pose_cache: Dict[Tuple[Path, int, str, float], Dict[str, Any]] = {}

    @staticmethod
    def _window_has_required_assets(scene_dir: Path, frame_ids: Sequence[int]) -> bool:
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth"
        flow_dir = scene_dir / "flow"
        for frame_id in frame_ids:
            if not (color_dir / f"{int(frame_id):06d}.png").exists():
                return False
            if not (depth_dir / f"{int(frame_id):06d}.png").exists():
                return False
        for frame_id in frame_ids[:-1]:
            pair_dir = flow_dir / f"{int(frame_id):06d}"
            if not (pair_dir / "flow_u_16.png").exists():
                return False
            if not (pair_dir / "flow_v_16.png").exists():
                return False
        return True

    @staticmethod
    def build_manifest_entries(
        *,
        data_root: str | Path,
        splits: Sequence[Any],
        chunk_size: int,
        overlap: int,
        target_height: int,
        target_width: int,
        max_scenes: Optional[int] = None,
        pose_source: str = "camera",
    ) -> List[SceneFlowManifestEntry]:
        root = Path(data_root)
        metadata_by_uid = _load_omniworld_metadata(root)
        requested_split_ids = _parse_omniworld_requested_split_ids(splits)
        scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "metadata")
        if max_scenes is not None:
            scene_dirs = scene_dirs[:int(max_scenes)]

        entries: List[SceneFlowManifestEntry] = []
        for scene_dir in scene_dirs:
            if not (scene_dir / "flow").exists():
                continue
            if not (scene_dir / "color").exists() or not (scene_dir / "depth").exists():
                continue
            if not (scene_dir / "camera").exists() and not (scene_dir / "droidclib").exists():
                continue

            split_info_path = scene_dir / "split_info.json"
            if not split_info_path.exists():
                continue
            with open(split_info_path, "r", encoding="utf-8") as file:
                split_payload = json.load(file)

            metadata_row = metadata_by_uid.get(scene_dir.name)
            metric_scale = _parse_omniworld_metric_scale(metadata_row)
            fps = float((metadata_row or {}).get("FPS", 0.0) or 0.0)
            test_split_indices = [
                int(token.strip())
                for token in str((metadata_row or {}).get("Test Split Index", "")).split(",")
                if token.strip()
            ]

            for split_idx, split_frames in enumerate(split_payload.get("split", [])):
                if requested_split_ids is not None and split_idx not in requested_split_ids:
                    continue
                split_frames = [int(frame_id) for frame_id in split_frames]
                for run_id, run_frames in enumerate(split_into_consecutive_runs(split_frames)):
                    if len(run_frames) <= 1:
                        continue
                    for start_idx, end_idx in get_trunk_indices(len(run_frames), chunk_size, overlap):
                        frame_ids = tuple(int(frame_id) for frame_id in run_frames[start_idx:end_idx])
                        if len(frame_ids) <= 1:
                            continue
                        if not OmniWorldSceneFlowAdapter._window_has_required_assets(scene_dir, frame_ids):
                            continue
                        entries.append(
                            SceneFlowManifestEntry(
                                dataset_name="omniworld",
                                source_split=f"split_{split_idx}",
                                scene_id=scene_dir.name,
                                clip_uid=f"{scene_dir.name}:split_{split_idx}:run_{run_id}:{frame_ids[0]}:{frame_ids[-1] + 1}",
                                frame_indices=frame_ids,
                                adapter_kwargs={
                                    "scene_dir": str(scene_dir),
                                    "split_idx": int(split_idx),
                                    "metric_scale": float(metric_scale),
                                    "pose_source": str(pose_source),
                                    "target_height": int(target_height),
                                    "target_width": int(target_width),
                                },
                                has_gt_flow=True,
                                flow_source="flow_depth_reprojected",
                                enabled_for_train=True,
                                quality_score=1.0,
                                domain_tags={"synthetic": True, "indoor": False, "outdoor": True, "sparse": False},
                                meta={
                                    "fps": fps,
                                    "metric_scale": float(metric_scale),
                                    "split_idx": int(split_idx),
                                    "run_id": int(run_id),
                                    "test_split_indices": test_split_indices,
                                },
                            )
                        )
        return entries

    @staticmethod
    def _expand_per_frame_scalar(value: Any, frame_count: int, field_name: str) -> np.ndarray:
        values = np.asarray(value, dtype=np.float32)
        if values.ndim == 0:
            values = np.full((frame_count,), float(values), dtype=np.float32)
        if values.shape[0] != frame_count:
            raise ValueError(f"OmniWorld field {field_name} expected {frame_count} values, got {values.shape}")
        return values.astype(np.float32)

    def _load_camera_pose_bundle(self, scene_dir: Path, split_idx: int, metric_scale: float) -> Dict[str, Any]:
        with open(scene_dir / "split_info.json", "r", encoding="utf-8") as file:
            split_payload = json.load(file)
        split_frames = tuple(int(frame_id) for frame_id in split_payload["split"][split_idx])

        with open(scene_dir / "camera" / f"split_{split_idx}.json", "r", encoding="utf-8") as file:
            camera_payload = json.load(file)

        frame_count = len(split_frames)
        focals = self._expand_per_frame_scalar(camera_payload["focals"], frame_count, "focals")
        cx = self._expand_per_frame_scalar(camera_payload.get("cx", 640.0), frame_count, "cx")
        cy = self._expand_per_frame_scalar(camera_payload.get("cy", 360.0), frame_count, "cy")

        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
        intrinsics[:, 0, 0] = focals
        intrinsics[:, 1, 1] = focals
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        quats = np.asarray(camera_payload["quats"], dtype=np.float32)
        trans = np.asarray(camera_payload["trans"], dtype=np.float32)
        if quats.shape != (frame_count, 4):
            raise ValueError(f"Unexpected OmniWorld quats shape: {quats.shape}")
        if trans.shape != (frame_count, 3):
            raise ValueError(f"Unexpected OmniWorld trans shape: {trans.shape}")

        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
        extrinsics[:, :3, :3] = _quat_wxyz_to_matrix(quats)
        extrinsics[:, :3, 3] = trans.astype(np.float32) * np.float32(metric_scale)

        frame_to_local = {int(frame_id): local_idx for local_idx, frame_id in enumerate(split_frames)}
        invalid_positions = set()
        for raw_index in camera_payload.get("invalid_frame", []):
            frame_or_pos = int(raw_index)
            if 0 <= frame_or_pos < frame_count:
                invalid_positions.add(frame_or_pos)
                continue
            if frame_or_pos in frame_to_local:
                invalid_positions.add(frame_to_local[frame_or_pos])

        return {
            "split_frames": split_frames,
            "frame_to_local": frame_to_local,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "invalid_positions": invalid_positions,
            "pose_source": "camera",
        }

    def _load_droidclib_pose_bundle(self, scene_dir: Path, split_idx: int, metric_scale: float) -> Dict[str, Any]:
        with open(scene_dir / "droidclib" / f"split_{split_idx}.json", "r", encoding="utf-8") as file:
            droid_payload = json.load(file)

        split_frames = tuple(int(frame_id) for frame_id in droid_payload.get("split", []))
        frame_count = len(split_frames)
        if frame_count == 0:
            raise ValueError(f"Empty OmniWorld droidclib split: {scene_dir} split_{split_idx}")

        intrinsic_payload = droid_payload.get("orig_intrinsic") or droid_payload.get("crop_intrinsic")
        if not intrinsic_payload:
            raise ValueError(f"Missing OmniWorld intrinsic payload in {scene_dir} split_{split_idx}")

        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
        intrinsics[:, 0, 0] = np.float32(intrinsic_payload["fx"])
        intrinsics[:, 1, 1] = np.float32(intrinsic_payload["fy"])
        intrinsics[:, 0, 2] = np.float32(intrinsic_payload["cx"])
        intrinsics[:, 1, 2] = np.float32(intrinsic_payload["cy"])

        extrinsics = np.asarray(droid_payload["extrinsics"], dtype=np.float32)
        if extrinsics.shape != (frame_count, 4, 4):
            raise ValueError(f"Unexpected OmniWorld droidclib extrinsics shape: {extrinsics.shape}")
        extrinsics = extrinsics.copy()
        extrinsics[:, :3, 3] *= np.float32(metric_scale)

        return {
            "split_frames": split_frames,
            "frame_to_local": {int(frame_id): local_idx for local_idx, frame_id in enumerate(split_frames)},
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "invalid_positions": set(),
            "pose_source": "droidclib",
        }

    def _load_pose_bundle(
        self,
        scene_dir: Path,
        split_idx: int,
        metric_scale: float,
        pose_source: str,
    ) -> Dict[str, Any]:
        cache_key = (scene_dir, int(split_idx), str(pose_source), float(metric_scale))
        if cache_key in self._pose_cache:
            return self._pose_cache[cache_key]

        loaders = []
        if pose_source == "droidclib":
            loaders = [self._load_droidclib_pose_bundle, self._load_camera_pose_bundle]
        else:
            loaders = [self._load_camera_pose_bundle, self._load_droidclib_pose_bundle]

        last_error: Optional[BaseException] = None
        for loader in loaders:
            try:
                bundle = loader(scene_dir, split_idx, metric_scale)
                self._pose_cache[cache_key] = bundle
                return bundle
            except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load OmniWorld pose bundle for {scene_dir} split_{split_idx}") from last_error

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        scene_dir = Path(entry.adapter_kwargs["scene_dir"])
        split_idx = int(entry.adapter_kwargs["split_idx"])
        metric_scale = float(entry.adapter_kwargs.get("metric_scale", 1.0))
        pose_source = str(entry.adapter_kwargs.get("pose_source", self.default_pose_source))
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)

        pose_bundle = self._load_pose_bundle(scene_dir, split_idx, metric_scale, pose_source)
        first_rgb = cv2.imread(str(scene_dir / "color" / f"{frame_ids[0]:06d}.png"), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise FileNotFoundError(f"Missing OmniWorld RGB frame: {scene_dir}:{frame_ids[0]}")
        first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
        src_height, src_width = first_rgb.shape[:2]
        resize_height, resize_width, crop_top, crop_left, crop_height, crop_width = compute_resize_params(
            src_height,
            src_width,
            self.target_height,
            self.target_width,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )

        rgbs, intrinsics, extrinsics, depths, valid_masks = [], [], [], [], []
        for frame_id in frame_ids:
            if frame_id not in pose_bundle["frame_to_local"]:
                raise KeyError(f"OmniWorld frame {frame_id} is not present in pose bundle for {scene_dir} split_{split_idx}")
            pose_local_idx = int(pose_bundle["frame_to_local"][frame_id])

            rgb = cv2.imread(str(scene_dir / "color" / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Missing OmniWorld RGB frame: {scene_dir}:{frame_id}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = _resize_crop(rgb, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_LINEAR)

            depth, valid_mask = decode_omniworld_depth(scene_dir / "depth" / f"{frame_id:06d}.png", metric_scale=metric_scale)
            depth = _resize_crop(depth, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = _resize_crop_mask(valid_mask, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width)
            if pose_local_idx in pose_bundle["invalid_positions"]:
                valid_mask &= False

            intrinsic = adjust_intrinsics_for_resize_and_crop(
                pose_bundle["intrinsics"][pose_local_idx],
                src_height,
                src_width,
                resize_height,
                resize_width,
                crop_top,
                crop_left,
            ).astype(np.float32)

            rgbs.append(rgb.astype(np.uint8))
            intrinsics.append(intrinsic)
            extrinsics.append(_ensure_extrinsic_4x4(pose_bundle["extrinsics"][pose_local_idx]))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(bool))

        rgbs_arr = np.stack(rgbs, axis=0).astype(np.uint8)
        intrinsics_arr = np.stack(intrinsics, axis=0).astype(np.float32)
        extrinsics_arr = np.stack(extrinsics, axis=0).astype(np.float32)
        scene_flow = np.zeros((len(frame_ids), crop_height, crop_width, 3), dtype=np.float32)
        deform_mask = np.zeros((len(frame_ids), crop_height, crop_width), dtype=np.float32)

        for local_idx in range(len(frame_ids) - 1):
            flow_dir = scene_dir / "flow" / f"{frame_ids[local_idx]:06d}"
            flow_uv = decode_omniworld_flow_png(flow_dir / "flow_u_16.png", flow_dir / "flow_v_16.png")
            flow_uv = _resize_crop_flow(flow_uv, resize_height, resize_width, crop_top, crop_left, crop_height, crop_width)

            pair_flow, pair_mask = flow_depth_to_metric_scene_flow(
                intrinsics_t=intrinsics_arr[local_idx],
                intrinsics_t1=intrinsics_arr[local_idx + 1],
                extrinsics_w2c=extrinsics_arr[local_idx:local_idx + 2],
                depth_t=depths[local_idx],
                depth_t1=depths[local_idx + 1],
                flow_uv=flow_uv,
                valid_mask_t=valid_masks[local_idx],
                valid_mask_t1=valid_masks[local_idx + 1],
                occlusion_mask=None,
                reference_w2c=extrinsics_arr[0],
            )
            scene_flow[local_idx] = pair_flow
            deform_mask[local_idx] = pair_mask.astype(np.float32)

        sample = self._finalize_sample(
            entry,
            rgbs_arr,
            intrinsics_arr,
            extrinsics_arr,
            scene_flow,
            deform_mask,
            frame_ids,
            depths=np.stack(depths, axis=0).astype(np.float32),
            valid_masks=np.stack(valid_masks, axis=0).astype(np.float32),
        )
        sample["meta"] = dict(sample["meta"], pose_source=pose_bundle["pose_source"])
        return sample


class PointOdysseySceneFlowAdapter(PointOdysseySceneFlowDataset):
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(super().__getitem__(idx))
        sample["dataset_name"] = "point_odyssey"
        sample["source_split"] = self.split
        sample["clip_uid"] = f"{sample['scene_name']}:{int(sample['start_idx'])}:{int(sample['end_idx'])}"
        sample["quality_score"] = torch.tensor(1.0, dtype=torch.float32)
        sample["loss_weight"] = torch.tensor(1.0, dtype=torch.float32)
        sample["has_gt_flow"] = torch.tensor(True)
        sample["flow_source"] = "traj3d_rasterized"
        sample["domain_tags"] = {"synthetic": True, "indoor": False, "outdoor": True, "sparse": True}
        sample["meta"] = {}
        return sample


class PointOdysseyManifestSceneFlowAdapter(PointOdysseySceneFlowDataset):
    def __init__(
        self,
        *,
        entries: Sequence[SceneFlowManifestEntry],
        target_height: int,
        target_width: int,
        chunk_size: int,
        flow_target_space: str,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        relative_scale_source: str = "flow_p95",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        pad_to_chunk_size: bool = True,
        flow_target_amplification: float = 1.0,
    ) -> None:
        Dataset.__init__(self)
        self.data_root = Path(".")
        self.split = "manifest"
        self.chunk_size = int(chunk_size)
        self.overlap = 0
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.process_res = None if process_res is None else int(process_res)
        self.process_res_method = None if process_res_method is None else str(process_res_method)
        self.max_scenes = None
        self.pad_to_chunk_size = bool(pad_to_chunk_size)
        self.flow_target_space = flow_target_space
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = flow_scale_stat
        self.flow_scale_eps = float(flow_scale_eps)
        self.flow_target_amplification = float(flow_target_amplification)
        self._skipped_scene_names: set[str] = set()
        self.entries = list(entries)
        self.samples: List[SampleInfo] = []

        for entry in self.entries:
            frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
            if not frame_ids:
                raise ValueError(f"PointOdyssey manifest entry {entry.clip_uid} has no frame indices")
            expected = tuple(range(frame_ids[0], frame_ids[-1] + 1))
            if frame_ids != expected:
                raise ValueError(f"PointOdyssey manifest entry {entry.clip_uid} must use consecutive frame indices")
            scene_dir_raw = entry.adapter_kwargs.get("scene_dir")
            if not scene_dir_raw:
                raise ValueError(f"PointOdyssey manifest entry {entry.clip_uid} is missing scene_dir")
            self.samples.append(
                SampleInfo(
                    scene_name=entry.scene_id,
                    scene_dir=Path(scene_dir_raw),
                    start_idx=frame_ids[0],
                    end_idx=frame_ids[-1] + 1,
                )
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.samples:
            raise RuntimeError("No valid PointOdyssey manifest samples are available")

        for offset in range(len(self.samples)):
            actual_idx = (idx + offset) % len(self.samples)
            sample_info = self.samples[actual_idx]
            entry = self.entries[actual_idx]
            frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
            try:
                sample = dict(self._load_sample(sample_info))
            except (FileNotFoundError, ValueError, IndexError, RuntimeError) as exc:
                self._warn_skip(sample_info.scene_name, str(exc))
                continue

            frame_ids_np = np.asarray(frame_ids, dtype=np.int64)
            if self.pad_to_chunk_size and len(frame_ids) < self.chunk_size:
                frame_ids_np = _pad_last_frame(frame_ids_np[:, None], self.chunk_size).reshape(-1)

            sample["dataset_name"] = "point_odyssey"
            sample["source_split"] = entry.source_split
            sample["clip_uid"] = entry.clip_uid
            sample["scene_dir"] = str(sample_info.scene_dir)
            sample["frame_ids"] = torch.from_numpy(frame_ids_np).long()
            sample["quality_score"] = torch.tensor(float(entry.quality_score), dtype=torch.float32)
            sample["loss_weight"] = torch.tensor(float(entry.loss_weight), dtype=torch.float32)
            sample["has_gt_flow"] = torch.tensor(bool(entry.has_gt_flow))
            sample["flow_source"] = entry.flow_source
            sample["domain_tags"] = {key: bool(value) for key, value in entry.domain_tags.items()}
            sample["meta"] = dict(entry.meta)
            return sample

        raise RuntimeError("All PointOdyssey manifest samples are invalid or unreadable")


class DynamicReplicaManifestSceneFlowAdapter(DynamicReplicaSceneFlowAdapter):
    def __init__(
        self,
        *,
        entries: Sequence[SceneFlowManifestEntry],
        target_height: int,
        target_width: int,
        chunk_size: int,
        flow_target_space: str,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        relative_scale_source: str = "geometry_radius",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        pad_to_chunk_size: bool = True,
        flow_target_amplification: float = 1.0,
    ) -> None:
        Dataset.__init__(self)
        if not entries:
            raise ValueError("DynamicReplicaManifestSceneFlowAdapter requires at least one manifest entry")

        self.entries = list(entries)
        data_roots = {
            str(entry.adapter_kwargs.get("data_root", "")).strip()
            for entry in self.entries
            if str(entry.adapter_kwargs.get("data_root", "")).strip()
        }
        if len(data_roots) != 1:
            raise ValueError("Dynamic Replica manifest entries must share a single data_root")
        self.data_root = Path(next(iter(data_roots))).expanduser().resolve()

        split_names = {
            str(entry.adapter_kwargs.get("split", entry.source_split)).strip()
            for entry in self.entries
        }
        if len(split_names) != 1:
            raise ValueError("Dynamic Replica manifest entries must belong to exactly one split")
        self.split = next(iter(split_names))

        self.chunk_size = int(chunk_size)
        self.overlap = int(self.entries[0].adapter_kwargs.get("overlap", 0))
        self.target_height = int(target_height)
        self.target_width = int(target_width)
        self.process_res = None if process_res is None else int(process_res)
        self.process_res_method = None if process_res_method is None else str(process_res_method)
        self.pad_to_chunk_size = bool(pad_to_chunk_size)
        self.flow_target_space = str(flow_target_space)
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = str(flow_scale_stat)
        self.flow_scale_eps = float(flow_scale_eps)
        self.flow_target_amplification = float(flow_target_amplification)
        self.camera_names = None
        self._warned_invalid_flow_pairs: set[str] = set()

        annotations_path = self.data_root / self.split / f"frame_annotations_{self.split}.jgz"
        with gzip.open(annotations_path, "rt", encoding="utf-8") as file:
            annotations = json.load(file)

        grouped: Dict[Tuple[str, str], Dict[int, Mapping[str, Any]]] = {}
        for annotation in annotations:
            image_path = annotation.get("image", {}).get("path")
            if not image_path:
                continue
            scene_name = Path(str(image_path)).parts[0]
            camera_name = str(annotation.get("camera_name", "left"))
            frame_number = int(annotation["frame_number"])
            grouped.setdefault((scene_name, camera_name), {})[frame_number] = dict(annotation)

        self.samples: List[DynamicReplicaClipInfo] = []
        for entry in self.entries:
            camera_name = str(entry.adapter_kwargs.get("camera_name") or entry.meta.get("camera_name", "left"))
            frame_lookup = grouped.get((entry.scene_id, camera_name))
            if not frame_lookup:
                raise ValueError(f"Missing Dynamic Replica annotations for {entry.scene_id} ({camera_name})")
            frame_indices = tuple(int(frame_id) for frame_id in entry.frame_indices)
            frame_annotations = []
            for frame_id in frame_indices:
                annotation = frame_lookup.get(frame_id)
                if annotation is None:
                    raise ValueError(
                        f"Missing Dynamic Replica frame {frame_id} for {entry.scene_id} ({camera_name})"
                    )
                frame_annotations.append(annotation)
            self.samples.append(
                DynamicReplicaClipInfo(
                    scene_name=entry.scene_id,
                    camera_name=camera_name,
                    frame_indices=frame_indices,
                    frame_annotations=tuple(frame_annotations),
                )
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(super().__getitem__(idx))
        entry = self.entries[idx]
        sample["source_split"] = entry.source_split
        sample["clip_uid"] = entry.clip_uid
        sample["quality_score"] = torch.tensor(float(entry.quality_score), dtype=torch.float32)
        sample["loss_weight"] = torch.tensor(float(entry.loss_weight), dtype=torch.float32)
        sample["has_gt_flow"] = torch.tensor(bool(entry.has_gt_flow))
        sample["flow_source"] = entry.flow_source
        sample["domain_tags"] = {key: bool(value) for key, value in entry.domain_tags.items()}
        sample["meta"] = dict(entry.meta)
        return sample


class MOViFManifestSceneFlowAdapter(Dataset):
    def __init__(
        self,
        *,
        entries: Sequence[SceneFlowManifestEntry],
        target_height: int,
        target_width: int,
        chunk_size: int,
        flow_target_space: str,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        relative_scale_source: str = "flow_p95",
        geometry_scale_stat: str = "mean_radius",
        flow_scale_stat: str = "p95",
        flow_scale_eps: float = 1e-6,
        flow_target_amplification: float = 1.0,
    ) -> None:
        super().__init__()
        if process_res is not None or process_res_method is not None:
            warnings.warn(
                "MOVi-F manifest adapter ignores process_res/process_res_method and uses target_height/target_width.",
                RuntimeWarning,
            )
        if not entries:
            raise ValueError("MOViFManifestSceneFlowAdapter requires at least one manifest entry")
        builder_dirs = {
            str(entry.adapter_kwargs.get("builder_dir", "")).strip()
            for entry in entries
            if str(entry.adapter_kwargs.get("builder_dir", "")).strip()
        }
        if len(builder_dirs) != 1:
            raise ValueError("MOVi-F manifest entries must share a single builder_dir")
        splits = {
            str(entry.adapter_kwargs.get("split", entry.source_split)).strip()
            for entry in entries
        }
        if len(splits) != 1:
            raise ValueError("MOVi-F manifest entries must belong to exactly one split")
        self.entries = list(entries)
        self.sample_indices = tuple(
            int(entry.adapter_kwargs.get("sample_index", entry.meta.get("sample_index", idx)))
            for idx, entry in enumerate(self.entries)
        )
        self.adapter = MOViFSceneFlowAdapter(
            builder_dir=next(iter(builder_dirs)),
            split=next(iter(splits)),
            flow_target_space=flow_target_space,
            relative_scale_source=relative_scale_source,
            geometry_scale_stat=geometry_scale_stat,
            flow_scale_stat=flow_scale_stat,
            flow_scale_eps=flow_scale_eps,
            chunk_size=chunk_size,
            pad_to_chunk_size=bool(self.entries[0].adapter_kwargs.get("pad_to_chunk_size", True)),
            fb_threshold_px=float(self.entries[0].adapter_kwargs.get("fb_threshold_px", 1.0)),
            target_height=target_height,
            target_width=target_width,
            flow_target_amplification=flow_target_amplification,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _read_example_at_index(self, sample_index: int) -> Dict[str, Any]:
        tfds = _load_tfds()
        builder = tfds.builder_from_directory(str(self.adapter.builder_dir))
        dataset = builder.as_dataset(split=self.adapter.split)
        for example_index, example in enumerate(tfds.as_numpy(dataset)):
            if example_index == sample_index:
                return self.adapter._convert_example(example)
        raise IndexError(f"MOVi-F sample_index {sample_index} is out of range for split {self.adapter.split}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        sample = dict(self._read_example_at_index(self.sample_indices[idx]))
        sample["source_split"] = entry.source_split
        sample["clip_uid"] = entry.clip_uid
        sample["scene_name"] = entry.scene_id
        sample["quality_score"] = torch.tensor(float(entry.quality_score), dtype=torch.float32)
        sample["loss_weight"] = torch.tensor(float(entry.loss_weight), dtype=torch.float32)
        sample["has_gt_flow"] = torch.tensor(bool(entry.has_gt_flow))
        sample["flow_source"] = entry.flow_source
        sample["domain_tags"] = {key: bool(value) for key, value in entry.domain_tags.items()}
        sample["meta"] = {**dict(sample.get("meta") or {}), **dict(entry.meta)}
        return sample


class MixedSceneFlowDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], dataset_names: Sequence[str]) -> None:
        super().__init__()
        if len(datasets) != len(dataset_names):
            raise ValueError("datasets and dataset_names must have the same length")
        self.datasets = list(datasets)
        self.dataset_names = list(dataset_names)
        self.dataset_sizes = [len(dataset) for dataset in self.datasets]
        self.dataset_offsets: List[int] = []
        self._index: List[Tuple[int, int]] = []
        offset = 0
        for dataset_id, dataset in enumerate(self.datasets):
            self.dataset_offsets.append(offset)
            for local_idx in range(len(dataset)):
                self._index.append((dataset_id, local_idx))
            offset += len(dataset)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        dataset_id, local_idx = self._index[global_idx]
        sample = dict(self.datasets[dataset_id][local_idx])
        sample.setdefault("dataset_name", self.dataset_names[dataset_id])
        sample["dataset_id"] = torch.tensor(dataset_id, dtype=torch.long)
        return sample


def _create_manifest_adapter(dataset_name: str, entries: Sequence[SceneFlowManifestEntry], common_cfg: Mapping[str, Any]) -> Dataset:
    kwargs = {
        "entries": entries,
        "target_height": int(common_cfg["target_height"]),
        "target_width": int(common_cfg["target_width"]),
        "chunk_size": int(common_cfg["chunk_size"]),
        "flow_target_space": common_cfg.get("flow_target_space", "relative"),
        "process_res": common_cfg.get("process_res"),
        "process_res_method": common_cfg.get("process_res_method"),
        "relative_scale_source": common_cfg.get("relative_scale_source", "flow_p95"),
        "geometry_scale_stat": common_cfg.get("geometry_scale_stat", "mean_radius"),
        "flow_scale_stat": common_cfg.get("flow_scale_stat", "p95"),
        "flow_scale_eps": float(common_cfg.get("flow_scale_eps", 1e-6)),
        "flow_target_amplification": float(common_cfg.get("flow_target_amplification", 1.0)),
    }
    if dataset_name == "point_odyssey":
        return PointOdysseyManifestSceneFlowAdapter(**kwargs)
    if dataset_name == "blinkvision":
        return BlinkVisionSceneFlowAdapter(**kwargs)
    if dataset_name == "omniworld":
        return OmniWorldSceneFlowAdapter(**kwargs)
    if dataset_name == "virtual_kitti2":
        return VirtualKITTI2SceneFlowAdapter(**kwargs)
    if dataset_name == "dynamic_replica":
        return DynamicReplicaManifestSceneFlowAdapter(**kwargs)
    if dataset_name == "movi_f":
        return MOViFManifestSceneFlowAdapter(**kwargs)
    raise ValueError(f"Unsupported manifest adapter dataset: {dataset_name}")


def _load_manifest_entries_from_spec(spec: Mapping[str, Any], manifest_key: str) -> List[SceneFlowManifestEntry]:
    manifest_path = spec.get(manifest_key)
    if not manifest_path:
        return []
    return load_scene_flow_manifest(manifest_path, data_root=spec.get("data_root"))


def _iter_manifest_specs(
    spec: Mapping[str, Any],
    *,
    singular_key: str,
    plural_key: str,
) -> List[Tuple[str, str]]:
    manifests = spec.get(plural_key)
    if manifests:
        if not isinstance(manifests, Mapping):
            raise TypeError(f"{plural_key} must be a mapping of split->manifest_path")
        return [
            (str(split), str(manifest_path))
            for split, manifest_path in manifests.items()
            if manifest_path
        ]

    manifest_path = spec.get(singular_key)
    if not manifest_path:
        return []
    return [("", str(manifest_path))]


def _dataset_component_name(base_name: str, split_name: str, total_components: int) -> str:
    if total_components <= 1 or not split_name:
        return base_name
    return f"{base_name}:{split_name}"


def build_scene_flow_train_components(
    data_cfg: Mapping[str, Any],
) -> Tuple[Dataset, Dict[str, Any]]:
    if "datasets" not in data_cfg:
        dataset = PointOdysseySceneFlowAdapter(
            data_root=data_cfg["data_root"],
            split=data_cfg["split"],
            chunk_size=int(data_cfg["chunk_size"]),
            overlap=int(data_cfg["overlap"]),
            target_height=int(data_cfg["target_height"]),
            target_width=int(data_cfg["target_width"]),
            process_res=data_cfg.get("process_res"),
            process_res_method=data_cfg.get("process_res_method"),
            max_scenes=data_cfg.get("max_scenes"),
            flow_target_space=data_cfg.get("flow_target_space", "relative"),
            relative_scale_source=data_cfg.get("relative_scale_source", "flow_p95"),
            geometry_scale_stat=data_cfg.get("geometry_scale_stat", "mean_radius"),
            flow_scale_stat=data_cfg.get("flow_scale_stat", "p95"),
            flow_scale_eps=float(data_cfg.get("flow_scale_eps", 1e-6)),
            flow_target_amplification=float(data_cfg.get("flow_target_amplification", 1.0)),
        )
        return dataset, {"dataset_names": ["point_odyssey"], "dataset_sizes": [len(dataset)], "dataset_offsets": [0], "dataset_probs": np.array([1.0], dtype=np.float64)}

    common_cfg = {
        "chunk_size": int(data_cfg["chunk_size"]),
        "overlap": int(data_cfg.get("overlap", max(0, int(data_cfg["chunk_size"]) // 2))),
        "target_height": int(data_cfg["target_height"]),
        "target_width": int(data_cfg["target_width"]),
        "process_res": data_cfg.get("process_res"),
        "process_res_method": data_cfg.get("process_res_method"),
        "flow_target_space": data_cfg.get("flow_target_space", "relative"),
        "relative_scale_source": data_cfg.get("relative_scale_source", "flow_p95"),
        "geometry_scale_stat": data_cfg.get("geometry_scale_stat", "mean_radius"),
        "flow_scale_stat": data_cfg.get("flow_scale_stat", "p95"),
        "flow_scale_eps": float(data_cfg.get("flow_scale_eps", 1e-6)),
        "flow_target_amplification": float(data_cfg.get("flow_target_amplification", 1.0)),
    }
    datasets: List[Dataset] = []
    dataset_names: List[str] = []
    dataset_weights: List[float] = []

    for spec in data_cfg["datasets"]:
        name = str(spec["name"])
        if not bool(spec.get("enabled_for_train", True)):
            continue
        train_manifest_specs = _iter_manifest_specs(spec, singular_key="train_manifest", plural_key="train_manifests")
        if train_manifest_specs:
            total_manifests = len(train_manifest_specs)
            for split_name, manifest_path in train_manifest_specs:
                entries = load_scene_flow_manifest(manifest_path, data_root=spec.get("data_root"))
                if not entries:
                    warnings.warn(
                        f"No manifest entries loaded for {name} from {manifest_path}",
                        RuntimeWarning,
                    )
                    continue
                datasets.append(_create_manifest_adapter(name, entries, common_cfg))
                dataset_names.append(_dataset_component_name(name, split_name, total_manifests))
                dataset_weights.append(float(spec.get("mixture_weight", 1.0)))
            continue
        if name == "point_odyssey":
            train_splits = spec.get("train_splits", [spec.get("split", "train")])
            for split in train_splits:
                dataset = PointOdysseySceneFlowAdapter(
                    data_root=spec["data_root"],
                    split=split,
                    chunk_size=common_cfg["chunk_size"],
                    overlap=common_cfg["overlap"],
                    target_height=common_cfg["target_height"],
                    target_width=common_cfg["target_width"],
                    process_res=common_cfg["process_res"],
                    process_res_method=common_cfg["process_res_method"],
                    max_scenes=spec.get("max_scenes"),
                    flow_target_space=common_cfg["flow_target_space"],
                    relative_scale_source=common_cfg["relative_scale_source"],
                    geometry_scale_stat=common_cfg["geometry_scale_stat"],
                    flow_scale_stat=common_cfg["flow_scale_stat"],
                    flow_scale_eps=common_cfg["flow_scale_eps"],
                    flow_target_amplification=common_cfg.get("flow_target_amplification", 1.0),
                )
                datasets.append(dataset)
                dataset_names.append(name if len(train_splits) == 1 else f"{name}:{split}")
                dataset_weights.append(float(spec.get("mixture_weight", 1.0)))
            continue
        if name == "blinkvision":
            entries = BlinkVisionSceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                splits=spec.get("train_splits", []),
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                max_scenes_per_split=spec.get("max_scenes_per_split"),
                max_depth_m=float(spec.get("max_depth_m", 30.0)),
                rgb_dir=str(spec.get("rgb_dir", "clean_uint8")),
            )
        elif name == "omniworld":
            entries = OmniWorldSceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                splits=spec.get("train_splits", []),
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                max_scenes=spec.get("max_scenes"),
                pose_source=str(spec.get("pose_source", "camera")),
            )
        elif name == "virtual_kitti2":
            entries = VirtualKITTI2SceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                scenes=spec.get("train_scenes"),
                variations=spec.get("variations"),
                cameras=spec.get("cameras"),
            )
        else:
            continue
        if not entries:
            warnings.warn(f"No manifest entries built for {name}", RuntimeWarning)
            continue
        datasets.append(_create_manifest_adapter(name, entries, common_cfg))
        dataset_names.append(name)
        dataset_weights.append(float(spec.get("mixture_weight", 1.0)))

    if not datasets:
        raise RuntimeError("No enabled scene-flow datasets were built")
    mixed_dataset = MixedSceneFlowDataset(datasets, dataset_names)
    dataset_probs = compute_dataset_sampling_probs(mixed_dataset.dataset_sizes, float(data_cfg.get("mixture_alpha", 0.5)), dataset_weights)
    return mixed_dataset, {
        "dataset_names": mixed_dataset.dataset_names,
        "dataset_sizes": mixed_dataset.dataset_sizes,
        "dataset_offsets": mixed_dataset.dataset_offsets,
        "dataset_probs": dataset_probs,
    }


def build_scene_flow_val_loaders(data_cfg: Mapping[str, Any], num_workers_override: Optional[int] = None) -> Dict[str, DataLoader]:
    num_workers = int(data_cfg.get("num_workers", 0) if num_workers_override is None else num_workers_override)
    batch_size = int(data_cfg.get("batch_size", 1))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    if "datasets" not in data_cfg:
        dataset = PointOdysseySceneFlowAdapter(
            data_root=data_cfg["data_root"],
            split=data_cfg.get("val_split", data_cfg["split"]),
            chunk_size=int(data_cfg["chunk_size"]),
            overlap=int(data_cfg["overlap"]),
            target_height=int(data_cfg["target_height"]),
            target_width=int(data_cfg["target_width"]),
            process_res=data_cfg.get("process_res"),
            process_res_method=data_cfg.get("process_res_method"),
            max_scenes=data_cfg.get("max_scenes"),
            flow_target_space=data_cfg.get("flow_target_space", "relative"),
            relative_scale_source=data_cfg.get("relative_scale_source", "flow_p95"),
            geometry_scale_stat=data_cfg.get("geometry_scale_stat", "mean_radius"),
            flow_scale_stat=data_cfg.get("flow_scale_stat", "p95"),
            flow_scale_eps=float(data_cfg.get("flow_scale_eps", 1e-6)),
            flow_target_amplification=float(data_cfg.get("flow_target_amplification", 1.0)),
        )
        return {"point_odyssey": DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)}

    common_cfg = {
        "chunk_size": int(data_cfg["chunk_size"]),
        "overlap": int(data_cfg.get("overlap", max(0, int(data_cfg["chunk_size"]) // 2))),
        "target_height": int(data_cfg["target_height"]),
        "target_width": int(data_cfg["target_width"]),
        "process_res": data_cfg.get("process_res"),
        "process_res_method": data_cfg.get("process_res_method"),
        "flow_target_space": data_cfg.get("flow_target_space", "relative"),
        "relative_scale_source": data_cfg.get("relative_scale_source", "flow_p95"),
        "geometry_scale_stat": data_cfg.get("geometry_scale_stat", "mean_radius"),
        "flow_scale_stat": data_cfg.get("flow_scale_stat", "p95"),
        "flow_scale_eps": float(data_cfg.get("flow_scale_eps", 1e-6)),
        "flow_target_amplification": float(data_cfg.get("flow_target_amplification", 1.0)),
    }
    val_loaders: Dict[str, DataLoader] = {}
    for spec in data_cfg["datasets"]:
        name = str(spec["name"])
        if not bool(spec.get("enabled_for_val", False)):
            continue
        val_manifest_specs = _iter_manifest_specs(spec, singular_key="val_manifest", plural_key="val_manifests")
        if val_manifest_specs:
            total_manifests = len(val_manifest_specs)
            for split_name, manifest_path in val_manifest_specs:
                entries = load_scene_flow_manifest(manifest_path, data_root=spec.get("data_root"))
                if not entries:
                    continue
                dataset = _create_manifest_adapter(name, entries, common_cfg)
                key = _dataset_component_name(name, split_name, total_manifests)
                val_loaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
            continue
        if name == "point_odyssey":
            val_splits = spec.get("val_splits", [])
            for split in val_splits:
                dataset = PointOdysseySceneFlowAdapter(
                    data_root=spec["data_root"],
                    split=split,
                    chunk_size=common_cfg["chunk_size"],
                    overlap=common_cfg["overlap"],
                    target_height=common_cfg["target_height"],
                    target_width=common_cfg["target_width"],
                    process_res=common_cfg["process_res"],
                    process_res_method=common_cfg["process_res_method"],
                    max_scenes=spec.get("max_val_scenes"),
                    flow_target_space=common_cfg["flow_target_space"],
                    relative_scale_source=common_cfg["relative_scale_source"],
                    geometry_scale_stat=common_cfg["geometry_scale_stat"],
                    flow_scale_stat=common_cfg["flow_scale_stat"],
                    flow_scale_eps=common_cfg["flow_scale_eps"],
                    flow_target_amplification=common_cfg.get("flow_target_amplification", 1.0),
                )
                key = name if len(val_splits) == 1 else f"{name}:{split}"
                val_loaders[key] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
            continue
        if name == "movi_f":
            val_splits = spec.get("val_splits", [spec.get("split", "validation")])
            for split in val_splits:
                dataset = MOViFSceneFlowAdapter(
                    builder_dir=spec["builder_dir"],
                    split=split,
                    chunk_size=common_cfg["chunk_size"],
                    pad_to_chunk_size=bool(spec.get("pad_to_chunk_size", True)),
                    flow_target_space=common_cfg["flow_target_space"],
                    relative_scale_source=common_cfg["relative_scale_source"],
                    geometry_scale_stat=common_cfg["geometry_scale_stat"],
                    flow_scale_stat=common_cfg["flow_scale_stat"],
                    flow_scale_eps=common_cfg["flow_scale_eps"],
                    flow_target_amplification=common_cfg.get("flow_target_amplification", 1.0),
                    fb_threshold_px=float(spec.get("fb_threshold_px", 1.0)),
                    target_height=common_cfg["target_height"],
                    target_width=common_cfg["target_width"],
                )
                key = name if len(val_splits) == 1 else f"{name}:{split}"
                val_loaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
            continue
        if name == "dynamic_replica":
            val_splits = spec.get("val_splits", [spec.get("split", "test")])
            for split in val_splits:
                dataset = DynamicReplicaSceneFlowAdapter(
                    data_root=spec["data_root"],
                    split=split,
                    chunk_size=common_cfg["chunk_size"],
                    overlap=common_cfg["overlap"],
                    target_height=common_cfg["target_height"],
                    target_width=common_cfg["target_width"],
                    process_res=common_cfg["process_res"],
                    process_res_method=common_cfg["process_res_method"],
                    pad_to_chunk_size=bool(spec.get("pad_to_chunk_size", True)),
                    flow_target_space=common_cfg["flow_target_space"],
                    relative_scale_source=common_cfg["relative_scale_source"],
                    geometry_scale_stat=common_cfg["geometry_scale_stat"],
                    flow_scale_stat=common_cfg["flow_scale_stat"],
                    flow_scale_eps=common_cfg["flow_scale_eps"],
                    flow_target_amplification=common_cfg.get("flow_target_amplification", 1.0),
                    max_scenes=spec.get("max_val_scenes"),
                    camera_names=spec.get("camera_names"),
                )
                key = name if len(val_splits) == 1 else f"{name}:{split}"
                val_loaders[key] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                )
            continue
        if name == "blinkvision":
            entries = BlinkVisionSceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                splits=spec.get("val_splits", []),
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                max_scenes_per_split=spec.get("max_val_scenes_per_split"),
                max_depth_m=float(spec.get("max_depth_m", 30.0)),
                rgb_dir=str(spec.get("rgb_dir", "clean_uint8")),
            )
        elif name == "omniworld":
            entries = OmniWorldSceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                splits=spec.get("val_splits", []),
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                max_scenes=spec.get("max_val_scenes"),
                pose_source=str(spec.get("pose_source", "camera")),
            )
        elif name == "virtual_kitti2":
            entries = VirtualKITTI2SceneFlowAdapter.build_manifest_entries(
                data_root=spec["data_root"],
                chunk_size=common_cfg["chunk_size"],
                overlap=common_cfg["overlap"],
                target_height=common_cfg["target_height"],
                target_width=common_cfg["target_width"],
                scenes=spec.get("val_scenes"),
                variations=spec.get("variations"),
                cameras=spec.get("cameras"),
            )
        else:
            continue
        if not entries:
            continue
        dataset = _create_manifest_adapter(name, entries, common_cfg)
        val_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return val_loaders


__all__ = [
    "BlinkVisionSceneFlowAdapter",
    "DynamicReplicaSceneFlowAdapter",
    "MixedSceneFlowDataset",
    "MOViFSceneFlowAdapter",
    "MOViFManifestSceneFlowAdapter",
    "OmniWorldSceneFlowAdapter",
    "DynamicReplicaManifestSceneFlowAdapter",
    "PointOdysseyManifestSceneFlowAdapter",
    "PointOdysseySceneFlowAdapter",
    "SceneFlowManifestEntry",
    "VirtualKITTI2SceneFlowAdapter",
    "build_dynamic_replica_manifest_entries",
    "build_point_odyssey_manifest_entries",
    "blinkvision_pose_to_w2c",
    "build_scene_flow_train_components",
    "build_scene_flow_val_loaders",
    "compute_dataset_sampling_probs",
    "decode_omniworld_depth",
    "decode_omniworld_flow_png",
    "decode_vkitti_scene_flow",
    "flow_depth_to_metric_scene_flow",
    "load_scene_flow_manifest",
    "save_scene_flow_manifest",
    "split_into_consecutive_runs",
]
