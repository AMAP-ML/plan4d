from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset
from utils.da3_resize import (
    apply_resize_only_to_array,
    apply_resize_only_to_intrinsics,
    build_upper_bound_resize_transform,
    resize_video_frames_upper_bound,
)


STITCHING_MANIFEST_VERSION = 1
SCANNETPP_FALLBACK_MANIFEST_NAMES = frozenset(
    {
        "scannetpp_nvs_sem_train.jsonl",
        "scannetpp_nvs_sem_val.jsonl",
    }
)
BLENDER_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
MVS_SYNTH_FLIP_Y = np.diag(np.array([1.0, -1.0, 1.0, 1.0], dtype=np.float32))
PYTORCH3D_TO_OPENCV = np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))


def _resolve_plan4d_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
    )
    for candidate in candidates:
        if (candidate / "plan4d" / "point_odyssey.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate plan4d root containing plan4d/point_odyssey.py")


def _resolve_da3_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1] / "Depth-Anything-3",
    )
    for candidate in candidates:
        if (candidate / "training" / "datasets" / "multi_scene_flow.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate plan4d Depth-Anything-3 root containing training/datasets/multi_scene_flow.py")


def _ensure_da3_training_imports() -> None:
    da3_root = _resolve_da3_root()
    da3_src_root = da3_root / "src"
    da3_training_root = da3_root / "training"
    for path in (da3_root, da3_src_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    training_pkg = sys.modules.get("training")
    if training_pkg is not None and hasattr(training_pkg, "__path__") and da3_training_root.exists():
        training_paths = [str(path) for path in training_pkg.__path__]
        if str(da3_training_root) not in training_paths:
            training_pkg.__path__.append(str(da3_training_root))


@lru_cache(maxsize=1)
def _load_multi_scene_flow_module():
    _ensure_da3_training_imports()
    module_path = _resolve_da3_root() / "training" / "datasets" / "multi_scene_flow.py"
    module_name = "plan4d_plan4d_multi_scene_flow"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_point_odyssey_utils_module():
    module_path = _resolve_plan4d_root() / "plan4d" / "point_odyssey.py"
    spec = importlib.util.spec_from_file_location("plan4d_plan4d_point_odyssey_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_resize_params(
    src_height: int,
    src_width: int,
    target_height: int,
    target_width: int,
) -> Tuple[int, int, int, int, int, int]:
    return _load_point_odyssey_utils_module().compute_resize_params(
        src_height,
        src_width,
        target_height,
        target_width,
    )


def adjust_intrinsics_for_resize_and_crop(
    intrinsics: np.ndarray,
    src_height: int,
    src_width: int,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
) -> np.ndarray:
    return _load_point_odyssey_utils_module().adjust_intrinsics_for_resize_and_crop(
        intrinsics,
        src_height,
        src_width,
        resize_height,
        resize_width,
        crop_top,
        crop_left,
    )


def get_trunk_indices(
    total_frames: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    return _load_point_odyssey_utils_module().get_trunk_indices(total_frames, chunk_size, overlap)


def generate_ray_map(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    return _load_point_odyssey_utils_module().generate_ray_map(
        intrinsics,
        extrinsics,
        height,
        width,
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


def _ensure_extrinsic_4x4(extrinsic: np.ndarray) -> np.ndarray:
    array = np.asarray(extrinsic, dtype=np.float32)
    if array.shape == (4, 4):
        return array.astype(np.float32)
    if array.shape == (3, 4):
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return np.concatenate([array, bottom], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported extrinsic shape: {array.shape}")


def _normalize_mvs_synth_extrinsic_to_opencv(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic_4x4 = _ensure_extrinsic_4x4(extrinsic)
    det = float(np.linalg.det(extrinsic_4x4[:3, :3]))
    if np.isclose(det, 1.0, atol=1e-4):
        return extrinsic_4x4.astype(np.float32)
    if not np.isclose(det, -1.0, atol=1e-3):
        raise ValueError(f"Unexpected MVS-Synth rotation determinant: {det}")
    return (MVS_SYNTH_FLIP_Y @ extrinsic_4x4).astype(np.float32)


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


def _resize_video_frames(
    frames_01: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    if frames_01.shape[-2:] == (resolution, resolution):
        return frames_01
    return F.interpolate(
        frames_01,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )


def _pad_last_frame(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.shape[0] >= target_length:
        return array
    pad_count = int(target_length - array.shape[0])
    last = np.repeat(array[-1:], pad_count, axis=0)
    return np.concatenate([array, last], axis=0)


def _pad_zero(array: np.ndarray, target_length: int) -> np.ndarray:
    if array.shape[0] >= target_length:
        return array
    pad_shape = (int(target_length - array.shape[0]),) + tuple(array.shape[1:])
    pad = np.zeros(pad_shape, dtype=array.dtype)
    return np.concatenate([array, pad], axis=0)


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


def decode_omniworld_depth(path: Path, metric_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
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
    depth[~valid] = 0.0
    return depth, valid.astype(bool)


def _load_vkitti_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth from {path}")
    invalid = depth > 65500
    depth = depth.astype(np.float32) / 100.0
    valid = (~invalid) & np.isfinite(depth) & (depth > 1e-6) & (depth <= 80.0)
    return depth, valid.astype(bool)


def _load_mvs_synth_depth_exr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth from {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    depth = depth.copy()
    depth[~valid] = 0.0
    return depth, valid.astype(bool)


def split_into_consecutive_runs(frame_indices: Sequence[int]) -> list[tuple[int, ...]]:
    if not frame_indices:
        return []
    ordered = [int(frame_id) for frame_id in frame_indices]
    runs: list[tuple[int, ...]] = []
    current_run = [ordered[0]]
    for frame_id in ordered[1:]:
        if frame_id == current_run[-1] + 1:
            current_run.append(frame_id)
            continue
        runs.append(tuple(current_run))
        current_run = [frame_id]
    runs.append(tuple(current_run))
    return runs


def _load_omniworld_metadata(root: Path) -> dict[str, dict[str, str]]:
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


def _resolve_relative_path(path: Path, relative_to: str | Path) -> str:
    root = Path(relative_to).resolve()
    return str(path.resolve().relative_to(root))


def _frame_id_from_name(name: str) -> int:
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame id from {name}")
    return int(digits)


def _extract_virtual_kitti2_camera(*values: Any) -> Optional[str]:
    for value in values:
        if value in {None, ""}:
            continue
        tokens = str(value).replace(":", "/").split("/")
        for token in tokens:
            if token.startswith("Camera_"):
                return token
    return None


def _extract_virtual_kitti2_sequence_dir(payload: Mapping[str, Any]) -> Optional[str]:
    sequence_dir = payload.get("sequence_dir")
    if sequence_dir not in {None, ""}:
        return str(sequence_dir)

    camera_dir = payload.get("camera_dir")
    if camera_dir not in {None, ""}:
        camera_dir_path = Path(str(camera_dir))
        try:
            return str(camera_dir_path.parents[2])
        except IndexError:
            pass

    scene_dir = payload.get("scene_dir")
    variation = payload.get("variation")
    if scene_dir not in {None, ""} and variation not in {None, ""}:
        return str(Path(str(scene_dir)) / str(variation))

    return None


@dataclass(frozen=True)
class StitchingManifestEntry:
    dataset_name: str
    source_split: str
    scene_id: str
    clip_uid: str
    frame_indices: Tuple[int, ...]
    scene_relpath: str
    supervision_mode: str
    raw_frame_count: int
    is_padded: bool
    has_gt_geometry: bool
    scene_flow_available: Optional[bool] = None
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        if Path(self.scene_relpath).is_absolute():
            raise ValueError(f"scene_relpath must be relative, got {self.scene_relpath}")
        payload = {
            "manifest_version": STITCHING_MANIFEST_VERSION,
            "dataset_name": self.dataset_name,
            "source_split": self.source_split,
            "scene_id": self.scene_id,
            "clip_uid": self.clip_uid,
            "frame_indices": [int(frame_id) for frame_id in self.frame_indices],
            "scene_relpath": self.scene_relpath,
            "supervision_mode": self.supervision_mode,
            "raw_frame_count": int(self.raw_frame_count),
            "is_padded": bool(self.is_padded),
            "has_gt_geometry": bool(self.has_gt_geometry),
            "adapter_kwargs": _jsonify_value(self.adapter_kwargs),
            "meta": _jsonify_value(self.meta),
        }
        if self.scene_flow_available is not None:
            payload["scene_flow_available"] = bool(self.scene_flow_available)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> "StitchingManifestEntry":
        dataset_name = str(payload["dataset_name"])
        adapter_kwargs = dict(payload.get("adapter_kwargs") or {})
        if dataset_name == "virtual_kitti2" and "camera" not in adapter_kwargs:
            camera = _extract_virtual_kitti2_camera(
                payload.get("camera"),
                payload.get("meta", {}).get("camera") if isinstance(payload.get("meta"), Mapping) else None,
                payload.get("scene_id"),
                payload.get("clip_uid"),
            )
            if camera is not None:
                adapter_kwargs["camera"] = camera
        return cls(
            dataset_name=dataset_name,
            source_split=str(payload["source_split"]),
            scene_id=str(payload["scene_id"]),
            clip_uid=str(payload["clip_uid"]),
            frame_indices=tuple(int(frame_id) for frame_id in payload.get("frame_indices", [])),
            scene_relpath=str(payload["scene_relpath"]),
            supervision_mode=str(payload.get("supervision_mode", "distill")),
            raw_frame_count=int(payload.get("raw_frame_count", len(payload.get("frame_indices", [])))),
            is_padded=bool(payload.get("is_padded", False)),
            has_gt_geometry=bool(payload.get("has_gt_geometry", payload.get("supervision_mode") == "gt")),
            scene_flow_available=(
                bool(payload["scene_flow_available"])
                if payload.get("scene_flow_available") is not None
                else None
            ),
            adapter_kwargs=adapter_kwargs,
            meta=dict(payload.get("meta") or {}),
        )

    def resolve_scene_path(self, manifest_data_root: str | Path) -> Path:
        root = Path(manifest_data_root).expanduser().resolve()
        scene_path = Path(self.scene_relpath)
        if scene_path.is_absolute():
            return scene_path.resolve()

        resolved = (root / scene_path).resolve()
        if resolved.exists():
            return resolved

        if (
            self.dataset_name == "omniworld"
            and scene_path.parts
            and scene_path.parts[0] == "OmniWorld-Game-extracted"
        ):
            fallback = (root / "OmniWorld" / scene_path).resolve()
            if fallback.exists():
                return fallback

        return resolved


def _infer_scannetpp_source_split(manifest_name: str) -> str:
    stem = manifest_name[:-6] if manifest_name.endswith(".jsonl") else manifest_name
    prefix = "scannetpp_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def _collect_scannetpp_frame_ids(scene_dir: Path) -> list[int]:
    rgb_dir = scene_dir / "iphone" / "rgb"
    frame_paths = sorted(rgb_dir.glob("frame_*.jpg"))
    return [int(path.stem.split("_")[-1]) for path in frame_paths]


def _is_valid_scannetpp_local_scene(
    scene_dir: Path,
    *,
    chunk_size: int,
) -> bool:
    pose_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    rgb_dir = scene_dir / "iphone" / "rgb"
    depth_dir = scene_dir / "iphone" / "depth"
    if not pose_path.exists() or not rgb_dir.exists() or not depth_dir.exists():
        return False

    frame_ids = _collect_scannetpp_frame_ids(scene_dir)
    if len(frame_ids) < chunk_size:
        return False
    target_frame_ids = frame_ids[:chunk_size]
    return all((depth_dir / f"frame_{frame_id:06d}.png").exists() for frame_id in target_frame_ids)


def _build_scannetpp_local_fallback_entries(
    *,
    scannetpp_local_root: str | Path,
    manifest_data_root: str | Path,
    manifest_name: str,
    source_split: str,
    chunk_size: int,
    sample_count: int,
) -> list[StitchingManifestEntry]:
    local_root = Path(scannetpp_local_root).expanduser().resolve()
    manifest_root = Path(manifest_data_root).expanduser().resolve()
    if not local_root.exists():
        raise FileNotFoundError(f"ScanNet++ local root does not exist: {local_root}")

    valid_scene_dirs = [
        scene_dir
        for scene_dir in sorted(path for path in local_root.iterdir() if path.is_dir())
        if _is_valid_scannetpp_local_scene(scene_dir, chunk_size=chunk_size)
    ]
    offset = 0 if "train" in manifest_name else sample_count
    selected_scene_dirs = valid_scene_dirs[offset: offset + sample_count]
    if len(selected_scene_dirs) < sample_count:
        raise ValueError(
            f"ScanNet++ local fallback under {local_root} only found {len(valid_scene_dirs)} valid scenes; "
            f"need at least {offset + sample_count} for {manifest_name}."
        )

    entries = []
    for scene_dir in selected_scene_dirs:
        frame_ids = tuple(_collect_scannetpp_frame_ids(scene_dir)[:chunk_size])
        scene_relpath = str(scene_dir.resolve().relative_to(manifest_root))
        entries.append(
            StitchingManifestEntry(
                dataset_name="scannetpp",
                source_split=source_split,
                scene_id=scene_dir.name,
                clip_uid=f"{scene_dir.name}:0:{len(frame_ids)}:local_fallback",
                frame_indices=frame_ids,
                scene_relpath=scene_relpath,
                supervision_mode="gt",
                raw_frame_count=len(frame_ids),
                is_padded=len(frame_ids) < chunk_size,
                has_gt_geometry=True,
                adapter_kwargs={},
                meta={"fallback_source": "local_scannetpp_data"},
            )
        )
    return entries


def save_stitching_manifest(
    entries: Sequence[StitchingManifestEntry],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")


def _build_action100m_manifest_entry(
    payload: Mapping[str, Any],
    *,
    default_dataset_name: str | None = None,
) -> StitchingManifestEntry:
    dataset_name = str(default_dataset_name or "action100m")
    video_path = str(payload["video"])
    sample_id = str(payload.get("sample_id") or Path(video_path).stem)
    frame_count = int(payload.get("num_frames", 0) or 0)
    if frame_count <= 0:
        raise ValueError(f"Action100M row must provide a positive num_frames, got {frame_count}")
    return StitchingManifestEntry(
        dataset_name=dataset_name,
        source_split=str(payload.get("source_split", "train")),
        scene_id=sample_id,
        clip_uid=sample_id,
        frame_indices=tuple(range(frame_count)),
        scene_relpath=video_path,
        supervision_mode="distill",
        raw_frame_count=frame_count,
        is_padded=False,
        has_gt_geometry=False,
        adapter_kwargs={"manifest_format": "action100m"},
        meta={
            "prompt": payload.get("prompt"),
            "task_type": payload.get("task_type"),
            "cond_frames": payload.get("cond_frames"),
        },
    )


GT_CAPABLE_RAW_MANIFEST_DATASETS = frozenset(
    {
        "point_odyssey",
        "omniworld",
        "virtual_kitti2",
        "scannetpp",
        "mvs_synth",
        "blinkvision",
        "dynamic_replica",
    }
)

SCENE_FLOW_SUPERVISION_DATASETS = frozenset(
    {
        "point_odyssey",
        "dynamic_replica",
        "blinkvision",
        "virtual_kitti2",
        "omniworld",
    }
)
SCENE_FLOW_TARGET_SPACE = "relative"
SCENE_FLOW_RELATIVE_SCALE_SOURCE = "geometry_radius"
SCENE_FLOW_GEOMETRY_SCALE_STAT = "mean_radius"
SCENE_FLOW_SCALE_STAT = "p95"
SCENE_FLOW_SCALE_EPS = 1e-6
SCENE_FLOW_SOURCE_BY_DATASET = {
    "point_odyssey": "traj3d_rasterized",
    "dynamic_replica": "depth_and_optical_flow_reconstruction",
    "blinkvision": "flow_depth_reprojected",
    "virtual_kitti2": "forward_scene_flow_png",
    "omniworld": "flow_depth_reprojected",
}


@dataclass(frozen=True)
class _SceneFlowAdapterRef:
    adapter: Any
    adapter_index: int
    expected_clip_uid: str


class _ManifestSceneFlowProvider:
    def __init__(
        self,
        *,
        entries: Sequence[StitchingManifestEntry],
        manifest_data_root: Path,
        target_resolution: int,
        chunk_size: int,
        process_res_method: str,
    ) -> None:
        self._adapter_refs: dict[int, _SceneFlowAdapterRef] = {}
        scene_flow_module = _load_multi_scene_flow_module()
        grouped_entries: dict[tuple[str, str], list[tuple[int, Any]]] = {}

        for entry_idx, entry in enumerate(entries):
            converted_entry = self._convert_entry(
                scene_flow_module,
                entry=entry,
                manifest_data_root=manifest_data_root,
            )
            if converted_entry is None:
                continue
            group_key = self._build_group_key(converted_entry)
            grouped_entries.setdefault(group_key, []).append((entry_idx, converted_entry))

        process_res = target_resolution if process_res_method == "upper_bound_resize" else None
        adapter_kwargs = {
            "target_height": int(target_resolution),
            "target_width": int(target_resolution),
            "chunk_size": int(chunk_size),
            "flow_target_space": SCENE_FLOW_TARGET_SPACE,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "relative_scale_source": SCENE_FLOW_RELATIVE_SCALE_SOURCE,
            "geometry_scale_stat": SCENE_FLOW_GEOMETRY_SCALE_STAT,
            "flow_scale_stat": SCENE_FLOW_SCALE_STAT,
            "flow_scale_eps": SCENE_FLOW_SCALE_EPS,
            "pad_to_chunk_size": True,
        }

        for grouped in grouped_entries.values():
            dataset_name = grouped[0][1].dataset_name
            adapter = self._build_adapter(
                scene_flow_module,
                dataset_name=dataset_name,
                entries=[converted_entry for _, converted_entry in grouped],
                adapter_kwargs=adapter_kwargs,
            )
            for adapter_index, (entry_idx, converted_entry) in enumerate(grouped):
                self._adapter_refs[entry_idx] = _SceneFlowAdapterRef(
                    adapter=adapter,
                    adapter_index=adapter_index,
                    expected_clip_uid=str(converted_entry.clip_uid),
                )

    def get(self, entry_idx: int) -> dict[str, Any] | None:
        adapter_ref = self._adapter_refs.get(entry_idx)
        if adapter_ref is None:
            return None
        try:
            sample = adapter_ref.adapter[adapter_ref.adapter_index]
        except (FileNotFoundError, ValueError, KeyError, IndexError, RuntimeError):
            return None
        if str(sample.get("clip_uid", "")) != adapter_ref.expected_clip_uid:
            return None
        return sample

    @staticmethod
    def _build_group_key(entry) -> tuple[str, str]:
        if entry.dataset_name == "dynamic_replica":
            data_root = str(entry.adapter_kwargs.get("data_root", ""))
            split = str(entry.adapter_kwargs.get("split", entry.source_split))
            return entry.dataset_name, f"{data_root}::{split}"
        return entry.dataset_name, ""

    @staticmethod
    def _build_adapter(scene_flow_module, *, dataset_name: str, entries: Sequence[Any], adapter_kwargs: dict[str, Any]):
        if dataset_name == "point_odyssey":
            return scene_flow_module.PointOdysseyManifestSceneFlowAdapter(
                entries=entries,
                **adapter_kwargs,
            )
        if dataset_name == "dynamic_replica":
            return scene_flow_module.DynamicReplicaManifestSceneFlowAdapter(
                entries=entries,
                **adapter_kwargs,
            )
        if dataset_name == "blinkvision":
            return scene_flow_module.BlinkVisionSceneFlowAdapter(
                entries=entries,
                **adapter_kwargs,
            )
        if dataset_name == "virtual_kitti2":
            return scene_flow_module.VirtualKITTI2SceneFlowAdapter(
                entries=entries,
                **adapter_kwargs,
            )
        if dataset_name == "omniworld":
            return scene_flow_module.OmniWorldSceneFlowAdapter(
                entries=entries,
                **adapter_kwargs,
            )
        raise ValueError(f"Unsupported scene-flow supervision dataset: {dataset_name}")

    @staticmethod
    def _convert_entry(scene_flow_module, *, entry: StitchingManifestEntry, manifest_data_root: Path):
        if entry.dataset_name not in SCENE_FLOW_SUPERVISION_DATASETS:
            return None
        if entry.scene_flow_available is not True:
            return None

        scene_path = entry.resolve_scene_path(manifest_data_root)
        adapter_kwargs: dict[str, Any]

        if entry.dataset_name == "point_odyssey":
            adapter_kwargs = {"scene_dir": str(scene_path)}
        elif entry.dataset_name == "dynamic_replica":
            camera_name = str(
                entry.adapter_kwargs.get("camera_name")
                or entry.meta.get("camera_name")
                or ("right" if entry.scene_id.endswith("_right") else "left")
            )
            adapter_kwargs = {
                "data_root": str(scene_path.parents[1]),
                "split": str(entry.source_split),
                "camera_name": camera_name,
                "overlap": 0,
            }
        elif entry.dataset_name == "blinkvision":
            adapter_kwargs = {
                "scene_dir": str(scene_path),
                "rgb_dir": str(entry.adapter_kwargs.get("rgb_dir", entry.meta.get("rgb_dir", "clean_uint8"))),
            }
        elif entry.dataset_name == "virtual_kitti2":
            camera = _extract_virtual_kitti2_camera(
                entry.adapter_kwargs.get("camera"),
                entry.meta.get("camera"),
                entry.scene_id,
                entry.clip_uid,
            )
            if camera is None:
                return None
            adapter_kwargs = {
                "variation_dir": str(scene_path),
                "camera": camera,
            }
        elif entry.dataset_name == "omniworld":
            split_idx = entry.adapter_kwargs.get("split_idx", entry.meta.get("split_idx"))
            if split_idx is None:
                return None
            adapter_kwargs = {
                "scene_dir": str(scene_path),
                "split_idx": int(split_idx),
                "metric_scale": float(entry.adapter_kwargs.get("metric_scale", entry.meta.get("metric_scale", 1.0))),
                "pose_source": str(entry.adapter_kwargs.get("pose_source", entry.meta.get("pose_source", "camera"))),
            }
        else:
            return None

        return scene_flow_module.SceneFlowManifestEntry(
            dataset_name=str(entry.dataset_name),
            source_split=str(entry.source_split),
            scene_id=str(entry.scene_id),
            clip_uid=str(entry.clip_uid),
            frame_indices=tuple(int(frame_id) for frame_id in entry.frame_indices),
            scene_relpath=str(entry.scene_relpath),
            adapter_kwargs=adapter_kwargs,
            has_gt_flow=True,
            flow_source=SCENE_FLOW_SOURCE_BY_DATASET[str(entry.dataset_name)],
            enabled_for_train=True,
            quality_score=1.0,
            loss_weight=1.0,
            domain_tags={},
            meta=dict(entry.meta),
        )


def _coerce_frame_indices(payload: Mapping[str, Any]) -> tuple[int, ...]:
    if "frame_indices" in payload and payload["frame_indices"] is not None:
        return tuple(int(frame_id) for frame_id in payload["frame_indices"])
    if "start_idx" in payload and "end_idx" in payload:
        start_idx = int(payload["start_idx"])
        end_idx = int(payload["end_idx"])
        return tuple(range(start_idx, end_idx))
    frame_count = int(payload.get("num_frames", 0) or 0)
    if frame_count <= 0:
        raise ValueError(
            "Raw manifest row must provide frame_indices or a positive num_frames."
        )
    return tuple(range(frame_count))


def _build_raw_caption_manifest_entry(
    payload: Mapping[str, Any],
    *,
    default_dataset_name: str | None = None,
) -> StitchingManifestEntry:
    dataset_name = str(
        payload.get("dataset")
        or payload.get("dataset_name")
        or default_dataset_name
        or ""
    ).strip()
    if dataset_name not in GT_CAPABLE_RAW_MANIFEST_DATASETS:
        raise ValueError(f"Unsupported raw caption manifest dataset: {dataset_name!r}")

    frame_indices = _coerce_frame_indices(payload)
    scene_relpath_value = payload.get("scene_relpath")
    if scene_relpath_value in {None, ""} and dataset_name == "virtual_kitti2":
        scene_relpath_value = _extract_virtual_kitti2_sequence_dir(payload)
    scene_relpath = str(
        scene_relpath_value
        or payload.get("scene_dir")
        or payload.get("video")
        or ""
    ).strip()
    if scene_relpath == "":
        raise ValueError(f"Raw caption manifest row missing scene path: keys={sorted(payload.keys())}")

    scene_id = str(
        payload.get("scene_id")
        or payload.get("scene_name")
        or Path(scene_relpath).name
        or Path(scene_relpath).stem
    )
    sample_id = str(payload.get("sample_id") or f"{scene_id}:{frame_indices[0]}:{len(frame_indices)}")
    source_split = str(payload.get("source_split") or payload.get("split") or "train")

    adapter_kwargs: dict[str, Any] = {}
    for key in ("split_idx", "metric_scale", "pose_source"):
        if key in payload and payload[key] is not None:
            adapter_kwargs[key] = payload[key]
    if dataset_name == "virtual_kitti2":
        camera = _extract_virtual_kitti2_camera(
            payload.get("camera"),
            payload.get("camera_name"),
            payload.get("scene_id"),
            payload.get("sample_id"),
        )
        if camera is not None:
            adapter_kwargs["camera"] = camera
    if dataset_name == "dynamic_replica":
        camera_name = payload.get("camera_name")
        if not camera_name:
            camera_name = "right" if str(scene_id).endswith("_right") else "left"
        adapter_kwargs["camera_name"] = camera_name

    scene_flow_available: Optional[bool]
    if payload.get("has_gt_scene_flow") is not None:
        scene_flow_available = bool(payload["has_gt_scene_flow"])
    elif payload.get("scene_flow_available") is not None:
        scene_flow_available = bool(payload["scene_flow_available"])
    else:
        scene_flow_available = None

    return StitchingManifestEntry(
        dataset_name=dataset_name,
        source_split=source_split,
        scene_id=scene_id,
        clip_uid=sample_id,
        frame_indices=frame_indices,
        scene_relpath=scene_relpath,
        supervision_mode="gt",
        raw_frame_count=int(payload.get("num_frames", len(frame_indices)) or len(frame_indices)),
        is_padded=False,
        has_gt_geometry=True,
        scene_flow_available=scene_flow_available,
        adapter_kwargs=adapter_kwargs,
        meta={
            "prompt": payload.get("prompt"),
            "task_type": payload.get("task_type"),
            "cond_frames": payload.get("cond_frames"),
            "scene_flow_match_key": payload.get("scene_flow_match_key"),
            "scene_flow_source": payload.get("scene_flow_source"),
        },
    )


def load_stitching_manifest(
    path: str | Path,
    *,
    default_dataset_name: str | None = None,
) -> list[StitchingManifestEntry]:
    entries: list[StitchingManifestEntry] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if "clip_uid" in payload and "scene_relpath" in payload:
                serialized_payload = dict(payload)
                if "dataset_name" not in serialized_payload and default_dataset_name is not None:
                    serialized_payload["dataset_name"] = default_dataset_name
                entries.append(StitchingManifestEntry.from_json_dict(serialized_payload))
                continue
            if "video" in payload and (default_dataset_name in {None, "action100m"}):
                entries.append(
                    _build_action100m_manifest_entry(
                        payload,
                        default_dataset_name=default_dataset_name,
                    )
                )
                continue
            raw_dataset_name = str(
                payload.get("dataset")
                or payload.get("dataset_name")
                or default_dataset_name
                or ""
            ).strip()
            if raw_dataset_name in GT_CAPABLE_RAW_MANIFEST_DATASETS:
                entries.append(
                    _build_raw_caption_manifest_entry(
                        payload,
                        default_dataset_name=default_dataset_name,
                    )
                )
                continue
            raise ValueError(f"Unsupported manifest row format in {path}: keys={sorted(payload.keys())}")
    return entries


def filter_manifest_entries_for_stage(
    entries: Sequence[StitchingManifestEntry],
    *,
    training_stage: str,
) -> list[StitchingManifestEntry]:
    training_stage = str(training_stage)
    if training_stage == "distill":
        return list(entries)
    if training_stage in {"gt", "hybrid"}:
        return [entry for entry in entries if entry.has_gt_geometry and entry.supervision_mode == "gt"]
    raise ValueError(f"Unsupported training stage: {training_stage}")


def _load_gzip_json_list(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f'Expected list payload in {path}, got {type(payload)}')
    return [dict(item) for item in payload]


def _load_float16_png(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise FileNotFoundError(f'Failed to read float16 png from {path}')
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype != np.uint16:
        raise ValueError(f'Expected uint16 png at {path}, got {array.dtype}')
    return np.ascontiguousarray(array).view(np.float16).astype(np.float32)


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f'Failed to read mask from {path}')
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)


def _dynamic_replica_viewpoint_to_intrinsics(
    viewpoint: Mapping[str, Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
    focal_length = np.asarray(viewpoint['focal_length'], dtype=np.float32)
    principal_point = np.asarray(viewpoint['principal_point'], dtype=np.float32)
    half_image_size = np.array([image_width, image_height], dtype=np.float32) / 2.0

    intrinsics_format = str(viewpoint['intrinsics_format']).lower()
    if intrinsics_format == 'ndc_norm_image_bounds':
        rescale = half_image_size
    elif intrinsics_format == 'ndc_isotropic':
        rescale = np.float32(half_image_size.min())
    else:
        raise ValueError(f'Unsupported intrinsics_format: {viewpoint["intrinsics_format"]}')

    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = float(focal_length_px[0])
    intrinsic[1, 1] = float(focal_length_px[1])
    intrinsic[0, 2] = float(principal_point_px[0])
    intrinsic[1, 2] = float(principal_point_px[1])
    return intrinsic


def _dynamic_replica_viewpoint_to_extrinsic(viewpoint: Mapping[str, Any]) -> np.ndarray:
    rotation = np.asarray(viewpoint['R'], dtype=np.float32)
    translation = np.asarray(viewpoint['T'], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = PYTORCH3D_TO_OPENCV @ rotation
    extrinsic[:3, 3] = PYTORCH3D_TO_OPENCV @ translation
    return extrinsic


def _normalize_dynamic_replica_clip_uid(scene_id: str, frame_indices: Sequence[int]) -> str:
    frame_ids = tuple(int(frame_id) for frame_id in frame_indices)
    if not frame_ids:
        raise ValueError(f'Dynamic Replica clip for {scene_id} has no frames')
    return f'{scene_id}:{frame_ids[0]}:{frame_ids[-1] + 1}'


def build_dynamic_replica_manifest_entries_from_scene_flow_manifest(
    source_manifest_path: str | Path,
    *,
    data_root: str | Path,
    relative_to: str | Path,
) -> list[StitchingManifestEntry]:
    source_manifest_path = Path(source_manifest_path)
    data_root = Path(data_root).resolve()
    entries: list[StitchingManifestEntry] = []
    with open(source_manifest_path, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            dataset_name = str(payload.get('dataset_name', ''))
            if dataset_name != 'dynamic_replica':
                raise ValueError(
                    f'Dynamic Replica conversion only supports dynamic_replica rows, got {dataset_name!r} in {source_manifest_path}'
                )
            frame_indices = tuple(int(frame_id) for frame_id in payload.get('frame_indices', []))
            source_split = str(payload.get('source_split') or payload.get('adapter_kwargs', {}).get('split') or 'train')
            scene_id = str(payload['scene_id'])
            scene_relpath = str(payload['scene_relpath']).strip()
            if not scene_relpath:
                raise ValueError(f'Dynamic Replica row in {source_manifest_path} is missing scene_relpath')
            camera_name = str(
                payload.get('meta', {}).get('camera_name')
                or payload.get('adapter_kwargs', {}).get('camera_name')
                or ('right' if scene_id.endswith('_right') else 'left')
            )
            resolved_scene_dir = data_root / scene_relpath
            entries.append(
                StitchingManifestEntry(
                    dataset_name='dynamic_replica',
                    source_split=source_split,
                    scene_id=scene_id,
                    clip_uid=_normalize_dynamic_replica_clip_uid(scene_id, frame_indices),
                    frame_indices=frame_indices,
                    scene_relpath=_resolve_relative_path(resolved_scene_dir, relative_to),
                    supervision_mode='gt',
                    raw_frame_count=len(frame_indices),
                    is_padded=bool(payload.get('is_padded', False)),
                    has_gt_geometry=True,
                    adapter_kwargs={'camera_name': camera_name},
                    meta={'camera_name': camera_name},
                )
            )
    return entries


def _point_odyssey_window_has_gt(scene_dir: Path, frame_ids: Sequence[int]) -> tuple[bool, Optional[str]]:
    anno_path = scene_dir / "anno.npz"
    if not anno_path.exists():
        return False, "missing_anno"
    for frame_id in frame_ids:
        depth_path = scene_dir / "depths" / f"depth_{int(frame_id):05d}.png"
        if not depth_path.exists():
            return False, f"missing_depth:{frame_id}"
    return True, None


def build_point_odyssey_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    splits: Sequence[str],
    chunk_size: int,
    overlap: int,
    max_scenes: Optional[int] = None,
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    entries: list[StitchingManifestEntry] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            warnings.warn(f"Point Odyssey split not found: {split_dir}", RuntimeWarning)
            continue
        scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if max_scenes is not None:
            scene_dirs = scene_dirs[: int(max_scenes)]
        for scene_dir in scene_dirs:
            rgb_files = sorted(scene_dir.glob("rgbs/rgb_*.jpg")) or sorted(scene_dir.glob("rgbs/rgb_*.png"))
            total_frames = len(rgb_files)
            if total_frames <= 1:
                continue
            for start_idx, end_idx in get_trunk_indices(total_frames, chunk_size, overlap):
                frame_ids = tuple(range(start_idx, end_idx))
                if len(frame_ids) <= 1:
                    continue
                has_gt_geometry, reason = _point_odyssey_window_has_gt(scene_dir, frame_ids)
                supervision_mode = "gt" if has_gt_geometry else "distill"
                meta = {"num_frames": total_frames}
                if reason is not None:
                    meta["gt_unavailable_reason"] = reason
                entries.append(
                    StitchingManifestEntry(
                        dataset_name="point_odyssey",
                        source_split=str(split),
                        scene_id=scene_dir.name,
                        clip_uid=f"{scene_dir.name}:{start_idx}:{end_idx}",
                        frame_indices=frame_ids,
                        scene_relpath=_resolve_relative_path(scene_dir, relative_to),
                        supervision_mode=supervision_mode,
                        raw_frame_count=len(frame_ids),
                        is_padded=len(frame_ids) < int(chunk_size),
                        has_gt_geometry=has_gt_geometry,
                        adapter_kwargs={},
                        meta=meta,
                    )
                )
    return entries


def _blinkvision_window_has_gt(
    scene_dir: Path,
    frame_ids: Sequence[int],
    *,
    rgb_dir: str,
) -> tuple[bool, Optional[str]]:
    if not (scene_dir / "metadata.json").exists():
        return False, "missing_metadata"
    if not (scene_dir / "poses.npz").exists():
        return False, "missing_poses"
    for frame_id in frame_ids:
        if not (scene_dir / rgb_dir / f"{int(frame_id):06d}.png").exists():
            return False, f"missing_rgb:{frame_id}"
        if not (scene_dir / "depth" / f"{int(frame_id):06d}.npz").exists():
            return False, f"missing_depth:{frame_id}"
    return True, None


def build_blinkvision_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    splits: Sequence[str],
    chunk_size: int,
    overlap: int,
    max_scenes_per_split: Optional[int] = None,
    rgb_dir: str = "clean_uint8",
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    entries: list[StitchingManifestEntry] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            warnings.warn(f"BlinkVision split not found: {split_dir}", RuntimeWarning)
            continue
        scene_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if max_scenes_per_split is not None:
            scene_dirs = scene_dirs[: int(max_scenes_per_split)]
        for scene_dir in scene_dirs:
            meta_path = scene_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            num_frames = int(metadata.get("num_frames", 0))
            if num_frames <= 1:
                continue
            rgb_root = scene_dir / rgb_dir
            if not rgb_root.exists():
                continue
            for start_idx, end_idx in get_trunk_indices(num_frames, chunk_size, overlap):
                frame_ids = tuple(range(start_idx, end_idx))
                if len(frame_ids) <= 1:
                    continue
                if not all((rgb_root / f"{int(frame_id):06d}.png").exists() for frame_id in frame_ids):
                    continue
                has_gt_geometry, reason = _blinkvision_window_has_gt(scene_dir, frame_ids, rgb_dir=rgb_dir)
                supervision_mode = "gt" if has_gt_geometry else "distill"
                meta = {"num_frames": num_frames}
                if reason is not None:
                    meta["gt_unavailable_reason"] = reason
                entries.append(
                    StitchingManifestEntry(
                        dataset_name="blinkvision",
                        source_split=str(split),
                        scene_id=scene_dir.name,
                        clip_uid=f"{split}:{scene_dir.name}:{start_idx}:{end_idx}",
                        frame_indices=frame_ids,
                        scene_relpath=_resolve_relative_path(scene_dir, relative_to),
                        supervision_mode=supervision_mode,
                        raw_frame_count=len(frame_ids),
                        is_padded=len(frame_ids) < int(chunk_size),
                        has_gt_geometry=has_gt_geometry,
                        adapter_kwargs={"rgb_dir": rgb_dir},
                        meta=meta,
                    )
                )
    return entries


def build_mvs_synth_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    chunk_size: int,
    overlap: int,
    max_scenes: Optional[int] = None,
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if max_scenes is not None:
        scene_dirs = scene_dirs[: int(max_scenes)]
    entries: list[StitchingManifestEntry] = []
    for scene_dir in scene_dirs:
        image_files = sorted((scene_dir / "images").glob("*.png"))
        total_frames = len(image_files)
        if total_frames <= 1:
            continue
        for start_idx, end_idx in get_trunk_indices(total_frames, chunk_size, overlap):
            frame_ids = tuple(range(start_idx, end_idx))
            if len(frame_ids) <= 1:
                continue
            if not all((scene_dir / "images" / f"{int(frame_id):04d}.png").exists() for frame_id in frame_ids):
                continue
            has_gt_geometry = all((scene_dir / "depths" / f"{int(frame_id):04d}.exr").exists() for frame_id in frame_ids) and all(
                (scene_dir / "poses" / f"{int(frame_id):04d}.json").exists() for frame_id in frame_ids
            )
            supervision_mode = "gt" if has_gt_geometry else "distill"
            meta: dict[str, Any] = {"num_frames": total_frames}
            if not has_gt_geometry:
                meta["gt_unavailable_reason"] = "missing_depth_or_pose"
            entries.append(
                StitchingManifestEntry(
                    dataset_name="mvs_synth",
                    source_split="train",
                    scene_id=scene_dir.name,
                    clip_uid=f"{scene_dir.name}:{start_idx}:{end_idx}",
                    frame_indices=frame_ids,
                    scene_relpath=_resolve_relative_path(scene_dir, relative_to),
                    supervision_mode=supervision_mode,
                    raw_frame_count=len(frame_ids),
                    is_padded=len(frame_ids) < int(chunk_size),
                    has_gt_geometry=has_gt_geometry,
                    adapter_kwargs={},
                    meta=meta,
                )
            )
    return entries


def _omniworld_has_color(scene_dir: Path, frame_ids: Sequence[int]) -> bool:
    return all((scene_dir / "color" / f"{int(frame_id):06d}.png").exists() for frame_id in frame_ids)


def _omniworld_has_gt(
    scene_dir: Path,
    frame_ids: Sequence[int],
    *,
    pose_source: str,
    split_idx: int,
) -> tuple[bool, Optional[str]]:
    if not all((scene_dir / "depth" / f"{int(frame_id):06d}.png").exists() for frame_id in frame_ids):
        return False, "missing_depth"
    if pose_source == "droidclib":
        pose_path = scene_dir / "droidclib" / f"split_{split_idx}.json"
        fallback_path = scene_dir / "camera" / f"split_{split_idx}.json"
    else:
        pose_path = scene_dir / "camera" / f"split_{split_idx}.json"
        fallback_path = scene_dir / "droidclib" / f"split_{split_idx}.json"
    if pose_path.exists() or fallback_path.exists():
        return True, None
    return False, "missing_pose"


def build_omniworld_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    splits: Sequence[Any],
    chunk_size: int,
    overlap: int,
    max_scenes: Optional[int] = None,
    pose_source: str = "camera",
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    metadata_by_uid = _load_omniworld_metadata(root)
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "metadata")
    if max_scenes is not None:
        scene_dirs = scene_dirs[: int(max_scenes)]
    requested_split_ids = None if not splits else {
        int(split.split("_", 1)[1]) if isinstance(split, str) and split.startswith("split_") else int(split)
        for split in splits
    }
    entries: list[StitchingManifestEntry] = []
    for scene_dir in scene_dirs:
        split_info_path = scene_dir / "split_info.json"
        if not split_info_path.exists():
            continue
        with open(split_info_path, "r", encoding="utf-8") as file:
            split_payload = json.load(file)
        metadata_row = metadata_by_uid.get(scene_dir.name)
        metric_scale = _parse_omniworld_metric_scale(metadata_row)
        for split_idx, split_frames in enumerate(split_payload.get("split", [])):
            if requested_split_ids is not None and split_idx not in requested_split_ids:
                continue
            split_frames = [int(frame_id) for frame_id in split_frames]
            for run_id, run_frames in enumerate(split_into_consecutive_runs(split_frames)):
                if len(run_frames) <= 1:
                    continue
                for start_idx, end_idx in get_trunk_indices(len(run_frames), chunk_size, overlap):
                    frame_ids = tuple(int(frame_id) for frame_id in run_frames[start_idx:end_idx])
                    if len(frame_ids) <= 1 or not _omniworld_has_color(scene_dir, frame_ids):
                        continue
                    has_gt_geometry, reason = _omniworld_has_gt(
                        scene_dir,
                        frame_ids,
                        pose_source=pose_source,
                        split_idx=split_idx,
                    )
                    supervision_mode = "gt" if has_gt_geometry else "distill"
                    meta = {
                        "metric_scale": float(metric_scale),
                        "run_id": int(run_id),
                    }
                    if reason is not None:
                        meta["gt_unavailable_reason"] = reason
                    entries.append(
                        StitchingManifestEntry(
                            dataset_name="omniworld",
                            source_split=f"split_{split_idx}",
                            scene_id=scene_dir.name,
                            clip_uid=f"{scene_dir.name}:split_{split_idx}:run_{run_id}:{frame_ids[0]}:{frame_ids[-1] + 1}",
                            frame_indices=frame_ids,
                            scene_relpath=_resolve_relative_path(scene_dir, relative_to),
                            supervision_mode=supervision_mode,
                            raw_frame_count=len(frame_ids),
                            is_padded=len(frame_ids) < int(chunk_size),
                            has_gt_geometry=has_gt_geometry,
                            adapter_kwargs={
                                "split_idx": int(split_idx),
                                "pose_source": str(pose_source),
                                "metric_scale": float(metric_scale),
                            },
                            meta=meta,
                        )
                    )
    return entries


def build_scannetpp_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    split_files: Sequence[str],
    chunk_size: int,
    overlap: int,
    max_scenes_per_split: Optional[int] = None,
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    entries: list[StitchingManifestEntry] = []
    for split_name in split_files:
        split_path = root / "splits" / f"{split_name}.txt"
        if not split_path.exists():
            warnings.warn(f"ScanNet++ split not found: {split_path}", RuntimeWarning)
            continue
        scene_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_scenes_per_split is not None:
            scene_ids = scene_ids[: int(max_scenes_per_split)]
        for scene_id in scene_ids:
            scene_dir = root / "data" / scene_id
            rgb_dir = scene_dir / "iphone" / "rgb"
            if not rgb_dir.exists():
                continue
            frame_paths = sorted(rgb_dir.glob("frame_*.jpg"))
            if len(frame_paths) <= 1:
                continue
            frame_ids = [_frame_id_from_name(path.name) for path in frame_paths]
            pose_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
            depth_dir = scene_dir / "iphone" / "depth"
            has_pose = pose_path.exists()
            has_depth = depth_dir.exists()
            for start_idx, end_idx in get_trunk_indices(len(frame_ids), chunk_size, overlap):
                clip_frame_ids = tuple(frame_ids[start_idx:end_idx])
                if len(clip_frame_ids) <= 1:
                    continue
                has_gt_geometry = has_pose and has_depth and all(
                    (depth_dir / f"frame_{int(frame_id):06d}.png").exists() for frame_id in clip_frame_ids
                )
                supervision_mode = "gt" if has_gt_geometry else "distill"
                meta: dict[str, Any] = {"num_frames": len(frame_ids)}
                if not has_gt_geometry:
                    meta["gt_unavailable_reason"] = "missing_depth_or_pose"
                entries.append(
                    StitchingManifestEntry(
                        dataset_name="scannetpp",
                        source_split=split_name,
                        scene_id=scene_id,
                        clip_uid=f"{scene_id}:{start_idx}:{end_idx}",
                        frame_indices=clip_frame_ids,
                        scene_relpath=_resolve_relative_path(scene_dir, relative_to),
                        supervision_mode=supervision_mode,
                        raw_frame_count=len(clip_frame_ids),
                        is_padded=len(clip_frame_ids) < int(chunk_size),
                        has_gt_geometry=has_gt_geometry,
                        adapter_kwargs={},
                        meta=meta,
                    )
                )
    return entries


def _vkitti_window_has_gt(
    variation_dir: Path,
    camera: str,
    frame_ids: Sequence[int],
) -> tuple[bool, Optional[str]]:
    if not (variation_dir / "intrinsic.txt").exists():
        return False, "missing_intrinsic"
    if not (variation_dir / "extrinsic.txt").exists():
        return False, "missing_extrinsic"
    for frame_id in frame_ids:
        if not (variation_dir / "frames" / "depth" / camera / f"depth_{int(frame_id):05d}.png").exists():
            return False, f"missing_depth:{frame_id}"
    return True, None


def build_virtual_kitti2_manifest_entries(
    *,
    data_root: str | Path,
    relative_to: str | Path,
    chunk_size: int,
    overlap: int,
    scenes: Optional[Sequence[str]] = None,
    variations: Optional[Sequence[str]] = None,
    cameras: Optional[Sequence[str]] = None,
) -> list[StitchingManifestEntry]:
    root = Path(data_root)
    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("Scene"))
    if scenes is not None:
        scene_dirs = [path for path in scene_dirs if path.name in set(scenes)]
    entries: list[StitchingManifestEntry] = []
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
                frame_ids = [_frame_id_from_name(path.name) for path in frame_paths]
                for start_idx, end_idx in get_trunk_indices(len(frame_ids), chunk_size, overlap):
                    clip_frame_ids = tuple(frame_ids[start_idx:end_idx])
                    if len(clip_frame_ids) <= 1:
                        continue
                    has_gt_geometry, reason = _vkitti_window_has_gt(variation_dir, camera_dir.name, clip_frame_ids)
                    supervision_mode = "gt" if has_gt_geometry else "distill"
                    meta = {"num_frames": len(frame_ids)}
                    if reason is not None:
                        meta["gt_unavailable_reason"] = reason
                    entries.append(
                        StitchingManifestEntry(
                            dataset_name="virtual_kitti2",
                            source_split="all",
                            scene_id=f"{scene_dir.name}/{variation_dir.name}/{camera_dir.name}",
                            clip_uid=f"{scene_dir.name}:{variation_dir.name}:{camera_dir.name}:{start_idx}:{end_idx}",
                            frame_indices=clip_frame_ids,
                            scene_relpath=_resolve_relative_path(variation_dir, relative_to),
                            supervision_mode=supervision_mode,
                            raw_frame_count=len(clip_frame_ids),
                            is_padded=len(clip_frame_ids) < int(chunk_size),
                            has_gt_geometry=has_gt_geometry,
                            adapter_kwargs={"camera": camera_dir.name},
                            meta=meta,
                        )
                    )
    return entries


def _load_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image from {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class StitchingManifestDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[StitchingManifestEntry],
        *,
        manifest_data_root: str | Path,
        image_resolution: int,
        feedforward_image_resolution: int,
        chunk_size: int = 81,
        training_stage: Optional[str] = None,
        process_res_method: str = "upper_bound_resize",
    ) -> None:
        super().__init__()
        self.entries = list(entries)
        self.manifest_data_root = Path(manifest_data_root).expanduser().resolve()
        self.image_resolution = int(image_resolution)
        self.feedforward_image_resolution = int(feedforward_image_resolution)
        self.chunk_size = int(chunk_size)
        self.training_stage = training_stage
        self.process_res_method = str(process_res_method)
        if self.process_res_method not in {"square_center_crop", "upper_bound_resize"}:
            raise ValueError(f"Unsupported process_res_method: {self.process_res_method}")

        self._blinkvision_cache: dict[Path, dict[str, Any]] = {}
        self._dynamic_replica_annotation_cache: dict[Path, dict[tuple[str, str], dict[int, Mapping[str, Any]]]] = {}
        self._omni_pose_cache: dict[tuple[Path, int, str, float], dict[str, Any]] = {}
        self._omni_split_idx_cache: dict[tuple[Path, tuple[int, ...], str], int] = {}
        self._scannet_pose_cache: dict[Path, dict[str, Any]] = {}
        self._vkitti_intrinsics_cache: dict[tuple[Path, str], dict[int, np.ndarray]] = {}
        self._vkitti_extrinsics_cache: dict[tuple[Path, str], dict[int, np.ndarray]] = {}
        self._scene_flow_provider = (
            _ManifestSceneFlowProvider(
                entries=self.entries,
                manifest_data_root=self.manifest_data_root,
                target_resolution=self.feedforward_image_resolution,
                chunk_size=self.chunk_size,
                process_res_method=self.process_res_method,
            )
            if self.training_stage != "distill"
            else None
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _build_feedforward_resize_config(self, src_height: int, src_width: int) -> dict[str, Any]:
        if self.process_res_method == "upper_bound_resize":
            transform = build_upper_bound_resize_transform(
                height=src_height,
                width=src_width,
                target_size=self.feedforward_image_resolution,
            )
            return {
                "src_height": int(src_height),
                "src_width": int(src_width),
                "out_height": int(transform.resized_height),
                "out_width": int(transform.resized_width),
                "upper_bound_transform": transform,
            }

        resize_height, resize_width, crop_top, crop_left, crop_height, crop_width = compute_resize_params(
            src_height,
            src_width,
            self.feedforward_image_resolution,
            self.feedforward_image_resolution,
        )
        return {
            "src_height": int(src_height),
            "src_width": int(src_width),
            "out_height": int(crop_height),
            "out_width": int(crop_width),
            "resize_height": int(resize_height),
            "resize_width": int(resize_width),
            "crop_top": int(crop_top),
            "crop_left": int(crop_left),
            "crop_height": int(crop_height),
            "crop_width": int(crop_width),
            "upper_bound_transform": None,
        }

    def _attach_scene_flow_supervision(
        self,
        *,
        entry_idx: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if self._scene_flow_provider is None:
            return result
        sample = self._scene_flow_provider.get(entry_idx)
        if sample is None:
            return result

        scene_flow = sample.get("scene_flow")
        deform_mask = sample.get("deform_mask")
        if not torch.is_tensor(scene_flow) or not torch.is_tensor(deform_mask):
            return result

        if scene_flow.ndim == 4 and scene_flow.shape[-1] == 3:
            scene_flow = scene_flow.permute(0, 3, 1, 2).contiguous()
        if deform_mask.ndim == 3:
            deform_mask = deform_mask.unsqueeze(1).contiguous()
        elif deform_mask.ndim == 4 and deform_mask.shape[-1] == 1:
            deform_mask = deform_mask.permute(0, 3, 1, 2).contiguous()

        if scene_flow.ndim != 4 or deform_mask.ndim != 4:
            return result

        target_t = int(result["depth"].shape[0])
        target_hw = tuple(int(dim) for dim in result["depth"].shape[-2:])
        if tuple(scene_flow.shape[-2:]) != target_hw:
            scene_flow = F.interpolate(
                scene_flow,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if tuple(deform_mask.shape[-2:]) != target_hw:
            deform_mask = F.interpolate(
                deform_mask.float(),
                size=target_hw,
                mode="nearest",
            )
        if int(scene_flow.shape[0]) != target_t:
            scene_flow = scene_flow[:target_t]
        if int(deform_mask.shape[0]) != target_t:
            deform_mask = deform_mask[:target_t]

        result["scene_flow"] = scene_flow.float().contiguous()
        result["deform_mask"] = deform_mask.float().contiguous()
        if torch.is_tensor(sample.get("scene_flow_scale")):
            result["scene_flow_scale"] = sample["scene_flow_scale"].float()
        if torch.is_tensor(sample.get("flow_p95_scale")):
            result["flow_p95_scale"] = sample["flow_p95_scale"].float()
        elif "scene_flow_scale" in result:
            result["flow_p95_scale"] = result["scene_flow_scale"]
        if sample.get("has_gt_flow") is not None:
            result["has_gt_flow"] = sample["has_gt_flow"]
        if sample.get("flow_source") is not None:
            result["flow_source"] = sample["flow_source"]
        return result

    def _resize_array(
        self,
        array: np.ndarray,
        resize_config: Mapping[str, Any],
        interpolation: int,
    ) -> np.ndarray:
        transform = resize_config.get("upper_bound_transform")
        if transform is not None:
            return apply_resize_only_to_array(
                array,
                transform=transform,
                interpolation=interpolation,
            )
        return _resize_crop(
            array,
            int(resize_config["resize_height"]),
            int(resize_config["resize_width"]),
            int(resize_config["crop_top"]),
            int(resize_config["crop_left"]),
            int(resize_config["crop_height"]),
            int(resize_config["crop_width"]),
            interpolation=interpolation,
        )

    def _resize_mask(
        self,
        mask: np.ndarray,
        resize_config: Mapping[str, Any],
    ) -> np.ndarray:
        transform = resize_config.get("upper_bound_transform")
        if transform is not None:
            resized = apply_resize_only_to_array(
                mask.astype(np.float32),
                transform=transform,
                interpolation=cv2.INTER_NEAREST,
            )
            return resized > 0.5
        return _resize_crop_mask(
            mask,
            int(resize_config["resize_height"]),
            int(resize_config["resize_width"]),
            int(resize_config["crop_top"]),
            int(resize_config["crop_left"]),
            int(resize_config["crop_height"]),
            int(resize_config["crop_width"]),
        )

    def _resize_intrinsics(
        self,
        intrinsics: np.ndarray,
        resize_config: Mapping[str, Any],
    ) -> np.ndarray:
        transform = resize_config.get("upper_bound_transform")
        if transform is not None:
            return apply_resize_only_to_intrinsics(
                intrinsics,
                original_height=int(resize_config["src_height"]),
                original_width=int(resize_config["src_width"]),
                resized_height=int(resize_config["out_height"]),
                resized_width=int(resize_config["out_width"]),
            ).astype(np.float32)
        return adjust_intrinsics_for_resize_and_crop(
            intrinsics,
            int(resize_config["src_height"]),
            int(resize_config["src_width"]),
            int(resize_config["resize_height"]),
            int(resize_config["resize_width"]),
            int(resize_config["crop_top"]),
            int(resize_config["crop_left"]),
        ).astype(np.float32)

    def _load_dynamic_replica_annotations(
        self,
        annotations_path: Path,
    ) -> dict[tuple[str, str], dict[int, Mapping[str, Any]]]:
        if annotations_path in self._dynamic_replica_annotation_cache:
            return self._dynamic_replica_annotation_cache[annotations_path]

        grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
        for annotation in _load_gzip_json_list(annotations_path):
            image_path = annotation.get('image', {}).get('path')
            if not image_path:
                continue
            scene_name = Path(str(image_path)).parts[0]
            camera_name = str(annotation.get('camera_name', 'left'))
            frame_number = int(annotation['frame_number'])
            grouped.setdefault((scene_name, camera_name), {})[frame_number] = annotation

        self._dynamic_replica_annotation_cache[annotations_path] = grouped
        return grouped

    def _load_dynamic_replica_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        split_dir = scene_dir.parent
        annotations_path = split_dir / f'frame_annotations_{entry.source_split}.jgz'
        camera_name = str(entry.adapter_kwargs.get('camera_name') or entry.meta.get('camera_name') or 'left')
        scene_annotations = self._load_dynamic_replica_annotations(annotations_path).get((scene_dir.name, camera_name))
        if not scene_annotations:
            raise ValueError(
                f'Missing Dynamic Replica annotations for {scene_dir.name} ({camera_name}) in {annotations_path}'
            )

        first_annotation = scene_annotations.get(frame_ids[0])
        if first_annotation is None:
            raise ValueError(
                f'Missing Dynamic Replica frame {frame_ids[0]} for {scene_dir.name} ({camera_name})'
            )
        first_rgb = _load_image_rgb(split_dir / str(first_annotation['image']['path']))
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs: list[np.ndarray] = []
        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            annotation = scene_annotations.get(frame_id)
            if annotation is None:
                raise ValueError(
                    f'Missing Dynamic Replica frame {frame_id} for {scene_dir.name} ({camera_name})'
                )
            rgb = _load_image_rgb(split_dir / str(annotation['image']['path']))
            if self.process_res_method != 'upper_bound_resize':
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

            if not include_geometry:
                continue

            depth = _load_float16_png(split_dir / str(annotation['depth']['path']))
            depth *= float(annotation.get('depth', {}).get('scale_adjustment', 1.0) or 1.0)
            intrinsic = _dynamic_replica_viewpoint_to_intrinsics(
                annotation['viewpoint'],
                depth.shape[0],
                depth.shape[1],
            )
            intrinsic = self._resize_intrinsics(intrinsic, resize_config)
            extrinsic = _ensure_extrinsic_4x4(_dynamic_replica_viewpoint_to_extrinsic(annotation['viewpoint']))
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)

            valid_mask = (np.isfinite(depth) & (depth > 1e-6)).astype(np.float32)
            depth = depth.astype(np.float32)
            depth[valid_mask <= 0.0] = 0.0

            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw: dict[str, Any] = {'rgbs': np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        raw.update(
            {
                'intrinsics': np.stack(intrinsics, axis=0).astype(np.float32),
                'extrinsics': np.stack(extrinsics, axis=0).astype(np.float32),
                'depth': np.stack(depths, axis=0).astype(np.float32),
                'valid_mask': np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _load_point_odyssey_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        first_rgb_path = scene_dir / "rgbs" / f"rgb_{frame_ids[0]:05d}.jpg"
        if not first_rgb_path.exists():
            first_rgb_path = scene_dir / "rgbs" / f"rgb_{frame_ids[0]:05d}.png"
        first_rgb = _load_image_rgb(first_rgb_path)
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs: list[np.ndarray] = []
        for frame_id in frame_ids:
            rgb_path = scene_dir / "rgbs" / f"rgb_{frame_id:05d}.jpg"
            if not rgb_path.exists():
                rgb_path = scene_dir / "rgbs" / f"rgb_{frame_id:05d}.png"
            rgb = _load_image_rgb(rgb_path)
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        anno = np.load(scene_dir / "anno.npz")
        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            intrinsic = self._resize_intrinsics(
                np.asarray(anno["intrinsics"][frame_id], dtype=np.float32),
                resize_config,
            )
            extrinsic = _ensure_extrinsic_4x4(np.asarray(anno["extrinsics"][frame_id], dtype=np.float32))
            depth = cv2.imread(str(scene_dir / "depths" / f"depth_{frame_id:05d}.png"), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Failed to read Point Odyssey depth from {scene_dir}:{frame_id}")
            depth = depth.astype(np.float32) / 1000.0
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = depth > 0.0

            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )

        return raw

    def _load_blinkvision_scene_cache(self, scene_dir: Path) -> dict[str, Any]:
        if scene_dir in self._blinkvision_cache:
            return self._blinkvision_cache[scene_dir]
        with open(scene_dir / "metadata.json", "r", encoding="utf-8") as file:
            metadata = json.load(file)
        payload = np.load(scene_dir / "poses.npz", allow_pickle=True)
        cache = {
            "metadata": metadata,
            "camera_poses": np.asarray(payload["camera_poses"], dtype=np.float32),
            "intrinsics": np.asarray(metadata["K_matrix"], dtype=np.float32),
            "unit_scale": float(metadata.get("unit_scale", 1.0)),
        }
        self._blinkvision_cache[scene_dir] = cache
        return cache

    def _load_blinkvision_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        rgb_dir = scene_dir / str(entry.adapter_kwargs.get("rgb_dir", "clean_uint8"))
        first_rgb = _load_image_rgb(rgb_dir / f"{frame_ids[0]:06d}.png")
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs: list[np.ndarray] = []
        for frame_id in frame_ids:
            rgb = _load_image_rgb(rgb_dir / f"{frame_id:06d}.png")
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        cache = self._load_blinkvision_scene_cache(scene_dir)
        adjusted_intrinsic = self._resize_intrinsics(
            cache["intrinsics"],
            resize_config,
        )
        max_depth_m = float(entry.meta.get("max_depth_m", 30.0))
        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            depth = np.load(scene_dir / "depth" / f"{frame_id:06d}.npz")["depth"].astype(np.float32)
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = np.isfinite(depth) & (depth > 1e-6) & (depth < max_depth_m)
            intrinsics.append(adjusted_intrinsic.copy())
            extrinsics.append(blinkvision_pose_to_w2c(cache["camera_poses"][frame_id], cache["unit_scale"]))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))
        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _load_mvs_synth_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        first_rgb = _load_image_rgb(scene_dir / "images" / f"{frame_ids[0]:04d}.png")
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)
        rgbs = []
        for frame_id in frame_ids:
            rgb = _load_image_rgb(scene_dir / "images" / f"{frame_id:04d}.png")
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))
        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            pose_path = scene_dir / "poses" / f"{frame_id:04d}.json"
            with open(pose_path, "r", encoding="utf-8") as file:
                pose_payload = json.load(file)

            intrinsic = np.array(
                [
                    [float(pose_payload["f_x"]), 0.0, float(pose_payload["c_x"])],
                    [0.0, float(pose_payload["f_y"]), float(pose_payload["c_y"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            intrinsic = self._resize_intrinsics(intrinsic, resize_config)

            depth, valid_mask = _load_mvs_synth_depth_exr(scene_dir / "depths" / f"{frame_id:04d}.exr")
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = self._resize_mask(valid_mask, resize_config)

            intrinsics.append(intrinsic)
            extrinsics.append(_normalize_mvs_synth_extrinsic_to_opencv(np.asarray(pose_payload["extrinsic"], dtype=np.float32)))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _load_omni_pose_bundle(
        self,
        scene_dir: Path,
        split_idx: int,
        metric_scale: float,
        pose_source: str,
    ) -> dict[str, Any]:
        cache_key = (scene_dir, int(split_idx), str(pose_source), float(metric_scale))
        if cache_key in self._omni_pose_cache:
            return self._omni_pose_cache[cache_key]

        def _expand_per_frame_scalar(value: Any, frame_count: int, field_name: str) -> np.ndarray:
            values = np.asarray(value, dtype=np.float32)
            if values.ndim == 0:
                values = np.full((frame_count,), float(values), dtype=np.float32)
            if values.shape[0] != frame_count:
                raise ValueError(f"OmniWorld field {field_name} expected {frame_count} values, got {values.shape}")
            return values.astype(np.float32)

        def _load_camera() -> dict[str, Any]:
            with open(scene_dir / "split_info.json", "r", encoding="utf-8") as file:
                split_payload = json.load(file)
            split_frames = tuple(int(frame_id) for frame_id in split_payload["split"][split_idx])

            with open(scene_dir / "camera" / f"split_{split_idx}.json", "r", encoding="utf-8") as file:
                camera_payload = json.load(file)

            frame_count = len(split_frames)
            focals = _expand_per_frame_scalar(camera_payload["focals"], frame_count, "focals")
            cx = _expand_per_frame_scalar(camera_payload.get("cx", 640.0), frame_count, "cx")
            cy = _expand_per_frame_scalar(camera_payload.get("cy", 360.0), frame_count, "cy")

            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
            intrinsics[:, 0, 0] = focals
            intrinsics[:, 1, 1] = focals
            intrinsics[:, 0, 2] = cx
            intrinsics[:, 1, 2] = cy

            quats = np.asarray(camera_payload["quats"], dtype=np.float32)
            trans = np.asarray(camera_payload["trans"], dtype=np.float32)
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], frame_count, axis=0)
            extrinsics[:, :3, :3] = _quat_wxyz_to_matrix(quats)
            extrinsics[:, :3, 3] = trans.astype(np.float32) * np.float32(metric_scale)

            frame_to_local = {int(frame_id): local_idx for local_idx, frame_id in enumerate(split_frames)}
            invalid_positions = set()
            for raw_index in camera_payload.get("invalid_frame", []):
                frame_or_pos = int(raw_index)
                if 0 <= frame_or_pos < frame_count:
                    invalid_positions.add(frame_or_pos)
                elif frame_or_pos in frame_to_local:
                    invalid_positions.add(frame_to_local[frame_or_pos])

            return {
                "frame_to_local": frame_to_local,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "invalid_positions": invalid_positions,
            }

        def _load_droidclib() -> dict[str, Any]:
            with open(scene_dir / "droidclib" / f"split_{split_idx}.json", "r", encoding="utf-8") as file:
                payload = json.load(file)
            split_frames = tuple(int(frame_id) for frame_id in payload.get("split", []))
            frame_count = len(split_frames)
            intrinsic_payload = payload.get("orig_intrinsic") or payload.get("crop_intrinsic")
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
            intrinsics[:, 0, 0] = np.float32(intrinsic_payload["fx"])
            intrinsics[:, 1, 1] = np.float32(intrinsic_payload["fy"])
            intrinsics[:, 0, 2] = np.float32(intrinsic_payload["cx"])
            intrinsics[:, 1, 2] = np.float32(intrinsic_payload["cy"])

            extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32).copy()
            extrinsics[:, :3, 3] *= np.float32(metric_scale)
            return {
                "frame_to_local": {int(frame_id): local_idx for local_idx, frame_id in enumerate(split_frames)},
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "invalid_positions": set(),
            }

        loaders = [_load_camera, _load_droidclib] if pose_source != "droidclib" else [_load_droidclib, _load_camera]
        last_error: Optional[BaseException] = None
        for loader in loaders:
            try:
                bundle = loader()
                self._omni_pose_cache[cache_key] = bundle
                return bundle
            except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load OmniWorld pose bundle for {scene_dir} split_{split_idx}") from last_error

    def _infer_omniworld_split_idx(
        self,
        scene_dir: Path,
        frame_ids: Sequence[int],
        pose_source: str,
    ) -> int:
        cache_key = (scene_dir, tuple(int(frame_id) for frame_id in frame_ids), str(pose_source))
        if cache_key in self._omni_split_idx_cache:
            return self._omni_split_idx_cache[cache_key]

        target_frame_ids = set(int(frame_id) for frame_id in frame_ids)

        def _contains_all(split_frames: Sequence[int]) -> bool:
            return target_frame_ids.issubset({int(frame_id) for frame_id in split_frames})

        if pose_source != "droidclib":
            split_info_path = scene_dir / "split_info.json"
            if split_info_path.exists():
                with open(split_info_path, "r", encoding="utf-8") as file:
                    split_payload = json.load(file)
                for split_idx, split_frames in enumerate(split_payload.get("split", [])):
                    if _contains_all(split_frames):
                        self._omni_split_idx_cache[cache_key] = int(split_idx)
                        return int(split_idx)

        droidclib_dir = scene_dir / "droidclib"
        if droidclib_dir.exists():
            for split_path in sorted(droidclib_dir.glob("split_*.json")):
                with open(split_path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                if _contains_all(payload.get("split", [])):
                    split_idx = int(split_path.stem.split("_")[-1])
                    self._omni_split_idx_cache[cache_key] = split_idx
                    return split_idx

        raise RuntimeError(
            f"Failed to infer OmniWorld split_idx for {scene_dir} and frames {tuple(frame_ids)}"
        )

    def _load_omniworld_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        first_rgb = _load_image_rgb(scene_dir / "color" / f"{frame_ids[0]:06d}.png")
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs = []
        for frame_id in frame_ids:
            rgb = _load_image_rgb(scene_dir / "color" / f"{frame_id:06d}.png")
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        pose_source = str(entry.adapter_kwargs.get("pose_source", "camera"))
        raw_split_idx = entry.adapter_kwargs.get("split_idx")
        split_idx = (
            int(raw_split_idx)
            if raw_split_idx is not None
            else self._infer_omniworld_split_idx(scene_dir, frame_ids, pose_source)
        )
        metric_scale = float(entry.adapter_kwargs.get("metric_scale", 1.0))
        pose_bundle = self._load_omni_pose_bundle(scene_dir, split_idx, metric_scale, pose_source)

        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            local_idx = pose_bundle["frame_to_local"][frame_id]
            depth, valid_mask = decode_omniworld_depth(
                scene_dir / "depth" / f"{frame_id:06d}.png",
                metric_scale=metric_scale,
            )
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = self._resize_mask(valid_mask, resize_config)
            if local_idx in pose_bundle["invalid_positions"]:
                valid_mask &= False
            intrinsic = self._resize_intrinsics(
                pose_bundle["intrinsics"][local_idx],
                resize_config,
            )

            intrinsics.append(intrinsic)
            extrinsics.append(_ensure_extrinsic_4x4(pose_bundle["extrinsics"][local_idx]))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _load_scannet_pose_cache(self, scene_dir: Path) -> dict[str, Any]:
        if scene_dir in self._scannet_pose_cache:
            return self._scannet_pose_cache[scene_dir]
        with open(scene_dir / "iphone" / "pose_intrinsic_imu.json", "r", encoding="utf-8") as file:
            payload = json.load(file)
        self._scannet_pose_cache[scene_dir] = payload
        return payload

    def _load_scannetpp_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        rgb_dir = scene_dir / "iphone" / "rgb"
        first_rgb = _load_image_rgb(rgb_dir / f"frame_{frame_ids[0]:06d}.jpg")
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs = []
        for frame_id in frame_ids:
            rgb = _load_image_rgb(rgb_dir / f"frame_{frame_id:06d}.jpg")
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        pose_payload = self._load_scannet_pose_cache(scene_dir)
        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            frame_key = f"frame_{frame_id:06d}"
            frame_payload = pose_payload[frame_key]
            intrinsic = np.asarray(frame_payload["intrinsic"], dtype=np.float32)
            pose = np.asarray(frame_payload.get("aligned_pose") or frame_payload["pose"], dtype=np.float32)
            c2w = _ensure_extrinsic_4x4(pose)
            depth = cv2.imread(
                str(scene_dir / "iphone" / "depth" / f"frame_{frame_id:06d}.png"),
                cv2.IMREAD_UNCHANGED,
            )
            if depth is None:
                raise FileNotFoundError(f"Failed to read ScanNet++ depth from {scene_dir}:{frame_id}")
            depth = depth.astype(np.float32) / 1000.0
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = np.isfinite(depth) & (depth > 0.0)
            intrinsic = self._resize_intrinsics(
                intrinsic,
                resize_config,
            )

            intrinsics.append(intrinsic)
            extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _parse_vkitti_intrinsics(self, variation_dir: Path, camera: str) -> dict[int, np.ndarray]:
        cache_key = (variation_dir, camera)
        if cache_key in self._vkitti_intrinsics_cache:
            return self._vkitti_intrinsics_cache[cache_key]
        result: dict[int, np.ndarray] = {}
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
        self._vkitti_intrinsics_cache[cache_key] = result
        return result

    def _parse_vkitti_extrinsics(self, variation_dir: Path, camera: str) -> dict[int, np.ndarray]:
        cache_key = (variation_dir, camera)
        if cache_key in self._vkitti_extrinsics_cache:
            return self._vkitti_extrinsics_cache[cache_key]
        result: dict[int, np.ndarray] = {}
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
        self._vkitti_extrinsics_cache[cache_key] = result
        return result

    def _resolve_virtual_kitti2_camera(self, entry: StitchingManifestEntry) -> str:
        camera = _extract_virtual_kitti2_camera(
            entry.adapter_kwargs.get("camera"),
            entry.meta.get("camera"),
            entry.scene_id,
            entry.clip_uid,
        )
        if camera is None:
            raise KeyError(
                f"Missing virtual_kitti2 camera for clip_uid={entry.clip_uid} scene_id={entry.scene_id}"
            )
        return camera

    def _load_virtual_kitti2_raw(self, entry: StitchingManifestEntry, scene_dir: Path, include_geometry: bool) -> dict[str, Any]:
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        camera = self._resolve_virtual_kitti2_camera(entry)
        rgb_dir = scene_dir / "frames" / "rgb" / camera
        first_rgb = _load_image_rgb(rgb_dir / f"rgb_{frame_ids[0]:05d}.jpg")
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)

        rgbs = []
        for frame_id in frame_ids:
            rgb = _load_image_rgb(rgb_dir / f"rgb_{frame_id:05d}.jpg")
            if self.process_res_method != "upper_bound_resize":
                rgb = self._resize_array(rgb, resize_config, cv2.INTER_LINEAR)
            rgbs.append(rgb.astype(np.uint8))

        raw: dict[str, Any] = {"rgbs": np.stack(rgbs, axis=0).astype(np.uint8)}
        if not include_geometry:
            return raw

        intrinsics_by_frame = self._parse_vkitti_intrinsics(scene_dir, camera)
        extrinsics_by_frame = self._parse_vkitti_extrinsics(scene_dir, camera)
        intrinsics, extrinsics, depths, valid_masks = [], [], [], []
        for frame_id in frame_ids:
            depth, valid_mask = _load_vkitti_depth(
                scene_dir / "frames" / "depth" / camera / f"depth_{frame_id:05d}.png"
            )
            depth = self._resize_array(depth, resize_config, cv2.INTER_NEAREST).astype(np.float32)
            valid_mask = self._resize_mask(valid_mask, resize_config)
            intrinsic = self._resize_intrinsics(
                intrinsics_by_frame[frame_id],
                resize_config,
            )
            intrinsics.append(intrinsic)
            extrinsics.append(_ensure_extrinsic_4x4(extrinsics_by_frame[frame_id]))
            depths.append(depth)
            valid_masks.append(valid_mask.astype(np.float32))

        raw.update(
            {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),
                "extrinsics": np.stack(extrinsics, axis=0).astype(np.float32),
                "depth": np.stack(depths, axis=0).astype(np.float32),
                "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
            }
        )
        return raw

    def _load_action100m_raw(self, entry: StitchingManifestEntry, video_path: Path, include_geometry: bool) -> dict[str, Any]:
        if include_geometry:
            raise ValueError(f"Action100M sample does not provide GT geometry: {entry.clip_uid}")

        frames = [np.asarray(frame)[..., :3] for frame in iio.imiter(video_path)]
        if not frames:
            raise IndexError(f"Action100M sample {entry.clip_uid} has no decodable frames: {video_path}")
        frame_ids = tuple(int(frame_id) for frame_id in entry.frame_indices)
        last_valid_frame_idx = len(frames) - 1
        selected_frames = [frames[min(frame_id, last_valid_frame_idx)].astype(np.uint8) for frame_id in frame_ids]
        first_rgb = selected_frames[0]
        src_height, src_width = first_rgb.shape[:2]
        resize_config = self._build_feedforward_resize_config(src_height, src_width)
        if self.process_res_method != "upper_bound_resize":
            selected_frames = [
                self._resize_array(frame, resize_config, cv2.INTER_LINEAR).astype(np.uint8)
                for frame in selected_frames
            ]
        stacked = np.stack(selected_frames, axis=0).astype(np.uint8)
        return {
            "rgbs": stacked,
            "teacher_rgbs": stacked,
            "student_uses_feedforward_image": False,
        }

    def _load_raw_sample(self, entry: StitchingManifestEntry, include_geometry: bool) -> dict[str, Any]:
        scene_path = entry.resolve_scene_path(self.manifest_data_root)
        if entry.dataset_name == "point_odyssey":
            return self._load_point_odyssey_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "dynamic_replica":
            return self._load_dynamic_replica_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "blinkvision":
            return self._load_blinkvision_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "mvs_synth":
            return self._load_mvs_synth_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "omniworld":
            return self._load_omniworld_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "scannetpp":
            return self._load_scannetpp_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "virtual_kitti2":
            return self._load_virtual_kitti2_raw(entry, scene_path, include_geometry)
        if entry.dataset_name == "action100m":
            return self._load_action100m_raw(entry, scene_path, include_geometry)
        raise ValueError(f"Unsupported manifest dataset: {entry.dataset_name}")

    def _finalize_common_sample(
        self,
        entry: StitchingManifestEntry,
        raw: dict[str, Any],
        include_geometry: bool,
    ) -> dict[str, Any]:
        rgbs = np.asarray(raw["rgbs"], dtype=np.uint8)
        teacher_rgbs = np.asarray(raw.get("teacher_rgbs", rgbs), dtype=np.uint8)
        sequence_length = int(rgbs.shape[0])
        if sequence_length < self.chunk_size:
            rgbs = _pad_last_frame(rgbs, self.chunk_size)
            teacher_rgbs = _pad_last_frame(teacher_rgbs, self.chunk_size)
        feedforward_rgb = torch.from_numpy(rgbs).float() / 255.0
        feedforward_rgb = feedforward_rgb.permute(0, 3, 1, 2).contiguous()
        teacher_feedforward_rgb = torch.from_numpy(teacher_rgbs).float() / 255.0
        teacher_feedforward_rgb = teacher_feedforward_rgb.permute(0, 3, 1, 2).contiguous()
        if self.process_res_method == "upper_bound_resize":
            feedforward_rgb, _ = resize_video_frames_upper_bound(
                feedforward_rgb,
                target_size=self.feedforward_image_resolution,
                mode="bilinear",
            )
            teacher_feedforward_rgb, _ = resize_video_frames_upper_bound(
                teacher_feedforward_rgb,
                target_size=self.feedforward_image_resolution,
                mode="bilinear",
            )
            vae_rgb, _ = resize_video_frames_upper_bound(
                torch.from_numpy(rgbs).float().permute(0, 3, 1, 2).contiguous() / 255.0,
                target_size=self.image_resolution,
                mode="bilinear",
            )
        else:
            vae_rgb = _resize_video_frames(feedforward_rgb, self.image_resolution)

        result: dict[str, Any] = {
            "dataset_name": entry.dataset_name,
            "source_split": entry.source_split,
            "scene_name": entry.scene_id,
            "clip_uid": entry.clip_uid,
            "supervision_mode": entry.supervision_mode,
            "sequence_length": torch.tensor(sequence_length, dtype=torch.long),
            "is_padded": torch.tensor(bool(entry.is_padded), dtype=torch.bool),
            "vae_image_tensor": (vae_rgb * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous(),
            "teacher_feedforward_image_tensor": (
                teacher_feedforward_rgb * 2.0 - 1.0
            ).permute(1, 0, 2, 3).contiguous(),
            "image_hw": torch.tensor(
                [int(teacher_feedforward_rgb.shape[-2]), int(teacher_feedforward_rgb.shape[-1])],
                dtype=torch.long,
            ),
        }
        if "prompt" in entry.meta and entry.meta["prompt"] is not None:
            result["prompt"] = entry.meta["prompt"]
        if "task_type" in entry.meta and entry.meta["task_type"] is not None:
            result["task_type"] = entry.meta["task_type"]
        if "cond_frames" in entry.meta and entry.meta["cond_frames"] is not None:
            result["cond_frames"] = entry.meta["cond_frames"]
        if raw.get("student_uses_feedforward_image", True):
            result["feedforward_image_tensor"] = (
                feedforward_rgb * 2.0 - 1.0
            ).permute(1, 0, 2, 3).contiguous()

        if not include_geometry:
            return result

        intrinsics = np.asarray(raw["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(raw["extrinsics"], dtype=np.float32)
        depth = np.asarray(raw["depth"], dtype=np.float32)
        valid_mask = np.asarray(raw["valid_mask"], dtype=np.float32)
        if sequence_length < self.chunk_size:
            intrinsics = _pad_last_frame(intrinsics, self.chunk_size)
            extrinsics = _pad_last_frame(extrinsics, self.chunk_size)
            depth = _pad_last_frame(depth, self.chunk_size)
            valid_mask = _pad_zero(valid_mask, self.chunk_size)

        ray_maps = []
        for frame_idx in range(intrinsics.shape[0]):
            ray_map = generate_ray_map(
                torch.from_numpy(intrinsics[frame_idx]).float(),
                torch.from_numpy(extrinsics[frame_idx]).float(),
                depth.shape[1],
                depth.shape[2],
            )
            ray_maps.append(ray_map)
        ray = torch.stack(ray_maps, dim=0).contiguous()

        result.update(
            {
                "depth": torch.from_numpy(depth).unsqueeze(1).float().contiguous(),
                "ray": ray.float(),
                "valid_mask": torch.from_numpy(valid_mask).unsqueeze(1).float().contiguous(),
                "intrinsics": torch.from_numpy(intrinsics).float().contiguous(),
                "extrinsics": torch.from_numpy(extrinsics).float().contiguous(),
            }
        )

        if "points_3d" in raw and "points_2d" in raw and "points_valid" in raw:
            points_3d = np.asarray(raw["points_3d"], dtype=np.float32)
            points_2d = np.asarray(raw["points_2d"], dtype=np.float32)
            points_valid = np.asarray(raw["points_valid"], dtype=np.float32)
            if sequence_length < self.chunk_size:
                points_3d = _pad_last_frame(points_3d, self.chunk_size)
                points_2d = _pad_last_frame(points_2d, self.chunk_size)
                points_valid = _pad_zero(points_valid, self.chunk_size)
            result["points_3d"] = torch.from_numpy(points_3d).float().contiguous()
            result["points_2d"] = torch.from_numpy(points_2d).float().contiguous()
            result["points_valid"] = torch.from_numpy(points_valid).float().contiguous()

        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        if self.training_stage == "gt" and not entry.has_gt_geometry:
            raise ValueError(f"GT stage received manifest sample without GT geometry: {entry.clip_uid}")
        include_geometry = entry.has_gt_geometry and self.training_stage != "distill"
        raw = self._load_raw_sample(entry, include_geometry=include_geometry)
        result = self._finalize_common_sample(entry, raw, include_geometry=include_geometry)
        if include_geometry:
            result = self._attach_scene_flow_supervision(entry_idx=idx, result=result)
        return result


class StitchingManifestPointOdysseyDataset(StitchingManifestDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        invalid = [entry.dataset_name for entry in self.entries if entry.dataset_name != "point_odyssey"]
        if invalid:
            raise ValueError(f"PointOdyssey manifest dataset received non-PointOdyssey entries: {invalid[:3]}")


def build_manifest_training_dataset(
    manifest_specs: Sequence[tuple[str, Path]],
    *,
    manifest_data_root: str | Path,
    image_resolution: int,
    feedforward_image_resolution: int,
    chunk_size: int,
    training_stage: str,
    process_res_method: str = "upper_bound_resize",
    samples_per_manifest: int | None = None,
    scannetpp_local_root: str | Path | None = None,
) -> Dataset:
    if samples_per_manifest is not None and int(samples_per_manifest) <= 0:
        raise ValueError("samples_per_manifest must be a positive integer when provided.")

    sample_limit = int(samples_per_manifest) if samples_per_manifest is not None else None
    datasets: list[Dataset] = []
    for dataset_name, manifest_path in manifest_specs:
        manifest_name = manifest_path.name
        if sample_limit is not None and manifest_name in SCANNETPP_FALLBACK_MANIFEST_NAMES:
            if scannetpp_local_root is None:
                raise ValueError(
                    "scannetpp_local_root is required when limiting "
                    "scannetpp_nvs_sem_train/val manifests."
                )
            filtered_entries = _build_scannetpp_local_fallback_entries(
                scannetpp_local_root=scannetpp_local_root,
                manifest_data_root=manifest_data_root,
                manifest_name=manifest_name,
                source_split=_infer_scannetpp_source_split(manifest_name),
                chunk_size=chunk_size,
                sample_count=sample_limit,
            )
        else:
            entries = load_stitching_manifest(
                manifest_path,
                default_dataset_name=dataset_name,
            )
            if dataset_name:
                mismatched = {
                    entry.dataset_name
                    for entry in entries
                    if entry.dataset_name != dataset_name
                }
                if mismatched:
                    warnings.warn(
                        f"Manifest label {dataset_name} does not exactly match entry dataset names "
                        f"{sorted(mismatched)} in {manifest_path}; continuing with manifest contents.",
                        RuntimeWarning,
                    )
            filtered_entries = filter_manifest_entries_for_stage(entries, training_stage=training_stage)
            if sample_limit is not None:
                filtered_entries = filtered_entries[:sample_limit]
        if not filtered_entries:
            warnings.warn(
                f"Manifest {manifest_path} has no usable samples for training_stage={training_stage}",
                RuntimeWarning,
            )
            continue
        datasets.append(
            StitchingManifestDataset(
                entries=filtered_entries,
                manifest_data_root=manifest_data_root,
                image_resolution=image_resolution,
                feedforward_image_resolution=feedforward_image_resolution,
                chunk_size=chunk_size,
                training_stage=training_stage,
                process_res_method=process_res_method,
            )
        )
    if not datasets:
        raise ValueError("No manifest datasets available after stage filtering.")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
