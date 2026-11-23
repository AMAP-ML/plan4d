from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import open3d as o3d
import torch
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for import_path in (DA3_SRC_ROOT, PLAN4D_ROOT, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from depth_anything_3.bench.utils import evaluate_3d_reconstruction, write_point_cloud_with_cameras
from evaluation.dynamic_replica_ray_benchmark import (
    DEFAULT_DA3_SCENE_FLOW_PRESET,
    DEFAULT_DOWN_SAMPLE,
    DEFAULT_EVAL_THRESHOLD,
    DEFAULT_SCENE_FLOW_EVAL_SPACE,
    DEFAULT_SCENE_FLOW_TARGET_SPACE,
    DEFAULT_SCENE_FLOW_THRESHOLD,
    _aggregate_metric_records,
    _as_homogeneous_extrinsics,
    _compute_clip_alignment,
    _compute_depth_metrics_np,
    _resize_intrinsics_to_original,
    _sanitize_depth,
    _write_raw_camera_params_npz,
    _write_results_npz,
    chunk_frame_indices,
    compute_dynamic_replica_scene_flow_metrics,
    default_da3_base_model_path,
    default_da3_scene_flow_head_checkpoint_path,
    dump_json,
)
from evaluation.eth3d_ray_benchmark import (
    compute_pose_with_gt,
    compute_umeyama_aligned_pose_with_gt,
    compute_raw_camera_scale_metrics,
    default_wan_vae_path,
    load_anyview_api,
    load_stitched_runner,
    run_da3_anyview_scene,
    run_da3_stitched_scene,
)


DEFAULT_INPUT_RESOLUTION = 504
DEFAULT_CLIP_LENGTH = 24
DEFAULT_SPLIT = "validation"
DEFAULT_LAZY_MODE = False
DEFAULT_SCENE_FLOW_EVAL = False
DEFAULT_MAX_POINT_SAMPLES = 200_000
DEFAULT_DEPTH_ALIGNMENT_STRATEGY = "median"
DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE = 8
DEPTH_ALIGNMENT_STRATEGIES = ("pose", "median", "none")
DEFAULT_DA3_STREAMING_MODEL_PATH = str((DA3_ROOT.parent / "checkpoints" / "da3").resolve())

_MOVI_F_SAMPLE_CACHE: dict[tuple[Path, str, str], list[dict[str, Any]]] = {}


@dataclass(frozen=True)
class MOViFClip:
    split: str
    scene_name: str
    sample_index: int
    frame_indices: tuple[int, ...]

    @property
    def start_frame(self) -> int:
        return int(self.frame_indices[0])

    @property
    def end_frame(self) -> int:
        return int(self.frame_indices[-1]) + 1

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)

    @property
    def clip_uid(self) -> str:
        return f"{self.scene_name}:{self.start_frame:05d}:{self.end_frame:05d}"

    @property
    def clip_dirname(self) -> str:
        return f"{self.start_frame:05d}_{self.end_frame:05d}"

    @property
    def clip_key(self) -> str:
        return f"{self.scene_name}/{self.clip_dirname}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "scene_name": self.scene_name,
            "sample_index": int(self.sample_index),
            "frame_indices": [int(frame_id) for frame_id in self.frame_indices],
            "clip_uid": self.clip_uid,
            "clip_dirname": self.clip_dirname,
            "num_frames": self.num_frames,
        }


def default_movi_f_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "MOVi-F-Val").resolve())


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar_int(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item()) if value.shape == () else int(value.reshape(-1)[0])
    return int(value)


def _to_scalar_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.shape == () else float(value.reshape(-1)[0])
    return float(value)


def _get_sample_name(sample: Mapping[str, Any]) -> str:
    for key in ("scene_name", "clip_uid", "video_name"):
        if key in sample:
            value = sample[key]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
    return "unknown"


def _get_sample_split(sample: Mapping[str, Any], fallback: str = DEFAULT_SPLIT) -> str:
    return str(sample.get("source_split", fallback))


def _get_sample_sequence_length(sample: Mapping[str, Any]) -> int:
    if "sequence_length" in sample:
        return _to_scalar_int(sample["sequence_length"])
    if "eval_rgbs" in sample:
        return int(_to_numpy(sample["eval_rgbs"]).shape[0])
    if "rgbs" in sample:
        return int(_to_numpy(sample["rgbs"]).shape[0])
    raise ValueError("MOVi-F sample does not contain sequence_length, eval_rgbs, or rgbs.")


def _get_sample_frame_ids(sample: Mapping[str, Any]) -> np.ndarray:
    sequence_length = _get_sample_sequence_length(sample)
    if "frame_ids" not in sample:
        return np.arange(sequence_length, dtype=np.int64)
    frame_ids = _to_numpy(sample["frame_ids"]).astype(np.int64).reshape(-1)
    return frame_ids[:sequence_length]


def _positions_for_frame_ids(sample: Mapping[str, Any], frame_indices: Sequence[int]) -> np.ndarray:
    sample_frame_ids = _get_sample_frame_ids(sample)
    positions: list[int] = []
    for frame_id in frame_indices:
        matches = np.nonzero(sample_frame_ids == int(frame_id))[0]
        if len(matches) == 0:
            raise ValueError(f"Frame id {frame_id} is missing from MOVi-F sample {_get_sample_name(sample)}")
        positions.append(int(matches[0]))
    return np.asarray(positions, dtype=np.int64)


def build_movi_f_clips_from_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    scenes: Sequence[str] | None = None,
    max_clips: int = -1,
    lazy_mode: bool = DEFAULT_LAZY_MODE,
) -> list[MOViFClip]:
    if clip_length <= 0:
        raise ValueError("clip_length must be positive.")
    requested_scenes = {str(scene_name) for scene_name in scenes} if scenes else None
    clips: list[MOViFClip] = []
    for sample_index, sample in enumerate(samples):
        scene_name = _get_sample_name(sample)
        if requested_scenes is not None and scene_name not in requested_scenes:
            continue
        frame_ids = tuple(int(frame_id) for frame_id in _get_sample_frame_ids(sample))
        split = _get_sample_split(sample)
        for clip_frame_ids in chunk_frame_indices(frame_ids, int(clip_length)):
            clips.append(
                MOViFClip(
                    split=split,
                    scene_name=scene_name,
                    sample_index=int(sample_index),
                    frame_indices=tuple(int(frame_id) for frame_id in clip_frame_ids),
                )
            )

    if lazy_mode:
        first_clip_by_scene: dict[str, MOViFClip] = {}
        for clip in clips:
            first_clip_by_scene.setdefault(clip.scene_name, clip)
        clips = list(first_clip_by_scene.values())
    if max_clips > 0:
        clips = clips[: int(max_clips)]
    return clips


def _load_movi_f_samples(
    dataset_root: str | Path,
    *,
    split: str,
    target_space: str,
) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    cache_key = (root, str(split), str(target_space))
    if cache_key not in _MOVI_F_SAMPLE_CACHE:
        MOViFSceneFlowAdapter = _load_movi_f_scene_flow_adapter_class()
        adapter = MOViFSceneFlowAdapter(
            builder_dir=root,
            split=str(split),
            flow_target_space=str(target_space),
            chunk_size=None,
            pad_to_chunk_size=False,
            target_height=None,
            target_width=None,
        )
        _MOVI_F_SAMPLE_CACHE[cache_key] = [dict(sample) for sample in adapter]
    return _MOVI_F_SAMPLE_CACHE[cache_key]


def _load_movi_f_scene_flow_adapter_class():
    module_path = DA3_ROOT / "training" / "datasets" / "movi_f_scene_flow.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"MOVi-F adapter not found: {module_path}")

    old_path = list(sys.path)
    saved_training_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "training" or name.startswith("training.")
    }
    for name in list(saved_training_modules):
        sys.modules.pop(name, None)

    try:
        sys.path.insert(0, str(DA3_ROOT))
        spec = importlib.util.spec_from_file_location("_plan4d_movi_f_scene_flow", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MOViFSceneFlowAdapter
    finally:
        sys.path[:] = old_path
        for name in [name for name in list(sys.modules) if name == "training" or name.startswith("training.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_training_modules)


def build_movi_f_clips(
    dataset_root: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    scenes: Sequence[str] | None = None,
    max_clips: int = -1,
    lazy_mode: bool = DEFAULT_LAZY_MODE,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
) -> list[MOViFClip]:
    samples = _load_movi_f_samples(dataset_root, split=split, target_space=target_space)
    clips = build_movi_f_clips_from_samples(
        samples,
        clip_length=int(clip_length),
        scenes=scenes,
        max_clips=int(max_clips),
        lazy_mode=bool(lazy_mode),
    )
    if scenes:
        available = {_get_sample_name(sample) for sample in samples}
        missing = sorted(str(scene_name) for scene_name in scenes if str(scene_name) not in available)
        if missing:
            raise ValueError(f"Requested MOVi-F scenes are missing from split={split}: {missing}")
    return clips


def radial_depth_to_camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W], got {depth.shape}")
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_norm = (xs - np.float32(cx)) / np.float32(max(fx, 1e-8))
    y_norm = (ys - np.float32(cy)) / np.float32(max(fy, 1e-8))
    direction = np.stack([x_norm, y_norm, np.ones_like(x_norm, dtype=np.float32)], axis=-1)
    direction_norm = np.maximum(
        np.linalg.norm(direction, axis=-1, keepdims=True).astype(np.float32),
        np.float32(1e-8),
    )
    return (direction / direction_norm) * depth[..., None]


def z_depth_to_camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W], got {depth.shape}")
    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_norm = (xs - np.float32(cx)) / np.float32(max(fx, 1e-8))
    y_norm = (ys - np.float32(cy)) / np.float32(max(fy, 1e-8))
    return np.stack([x_norm * depth, y_norm * depth, depth], axis=-1).astype(np.float32)


def z_depth_to_radial_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = z_depth_to_camera_points(depth, intrinsics)
    return np.linalg.norm(points, axis=-1).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = _as_homogeneous_extrinsics(np.asarray(transform, dtype=np.float32))
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    transformed = flat @ rotation.T + translation[None]
    return transformed.reshape(points.shape).astype(np.float32)


def _subsample_point_indices(num_points: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or int(num_points) <= int(max_points):
        return np.arange(int(num_points), dtype=np.int64)
    return np.linspace(0, int(num_points) - 1, int(max_points), dtype=np.int64)


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return points[_subsample_point_indices(len(points), int(max_points))]


def _normalize_rgb_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors).reshape(-1, 3)
    if colors.size == 0:
        return colors.astype(np.float64)
    colors = colors.astype(np.float64)
    if float(np.nanmax(colors)) > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def _rgb_frame_for_mask(scene_data, frame_index: int, mask: np.ndarray) -> np.ndarray:
    rgb = np.asarray(scene_data.rgbs[int(frame_index)])
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError(f"Expected MOVi-F RGB frame with shape [H,W,3], got {rgb.shape}")
    target_height, target_width = np.asarray(mask).shape
    if rgb.shape[:2] != (target_height, target_width):
        rgb = cv2.resize(
            rgb[..., :3],
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.ascontiguousarray(rgb[..., :3])


def _point_cloud_from_points(
    points: np.ndarray,
    max_points: int,
    *,
    colors: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    indices = _subsample_point_indices(len(points), int(max_points))
    sampled = points[indices]
    pcd = o3d.geometry.PointCloud()
    if len(sampled) > 0:
        pcd.points = o3d.utility.Vector3dVector(sampled.astype(np.float64))
        if colors is not None:
            colors = np.asarray(colors).reshape(-1, 3)
            if len(colors) != len(points):
                raise ValueError(
                    f"Expected one RGB color per point, got {len(colors)} colors for {len(points)} points."
                )
            pcd.colors = o3d.utility.Vector3dVector(_normalize_rgb_colors(colors[indices]))
    return pcd


def _write_point_cloud(
    *,
    output_path: Path,
    pcd: o3d.geometry.PointCloud,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    image_sizes: Sequence[tuple[int, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    write_point_cloud_with_cameras(
        fuse_path=str(output_path),
        pcd=pcd,
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        extrinsics=_as_homogeneous_extrinsics(np.asarray(extrinsics, dtype=np.float32)),
        image_sizes=list(image_sizes),
    )


def _slice_sample_array(sample: Mapping[str, Any], key: str, positions: np.ndarray) -> np.ndarray:
    if key not in sample:
        raise KeyError(f"MOVi-F sample missing key: {key}")
    return _to_numpy(sample[key])[positions]


def load_movi_f_clip_scene_data(
    dataset_root: str | Path,
    clip: MOViFClip,
    *,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
):
    samples = _load_movi_f_samples(
        dataset_root,
        split=clip.split,
        target_space=str(target_space),
    )
    if int(clip.sample_index) >= len(samples):
        raise IndexError(f"MOVi-F sample_index={clip.sample_index} exceeds loaded sample count {len(samples)}")
    sample = samples[int(clip.sample_index)]
    positions = _positions_for_frame_ids(sample, clip.frame_indices)

    rgb_key = "eval_rgbs" if "eval_rgbs" in sample else "rgbs"
    depth_key = "eval_gt_depth" if "eval_gt_depth" in sample else "gt_depth"
    depth_valid_key = "eval_gt_depth_valid_mask" if "eval_gt_depth_valid_mask" in sample else "gt_depth_valid_mask"
    intrinsics_key = "eval_intrinsics" if "eval_intrinsics" in sample else "intrinsics"
    scene_flow_key = "eval_scene_flow" if "eval_scene_flow" in sample else "scene_flow"
    deform_mask_key = "eval_deform_mask" if "eval_deform_mask" in sample else "deform_mask"

    rgbs = _slice_sample_array(sample, rgb_key, positions).astype(np.uint8)
    gt_depths = _slice_sample_array(sample, depth_key, positions).astype(np.float32)
    gt_depth_valid_masks = _slice_sample_array(sample, depth_valid_key, positions).astype(np.float32) > 0.5
    intrinsics = _slice_sample_array(sample, intrinsics_key, positions).astype(np.float32)
    extrinsics = _slice_sample_array(sample, "extrinsics", positions).astype(np.float32)
    scene_flow = _slice_sample_array(sample, scene_flow_key, positions).astype(np.float32)
    deform_mask = _slice_sample_array(sample, deform_mask_key, positions).astype(np.float32)
    scene_flow_scale = _to_scalar_float(sample.get("scene_flow_scale", 1.0))
    image_files = [np.ascontiguousarray(frame) for frame in rgbs]

    return SimpleNamespace(
        image_files=image_files,
        rgbs=rgbs,
        gt_depths=gt_depths,
        gt_depth_valid_masks=gt_depth_valid_masks,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        scene_flow_scale=scene_flow_scale,
        aux=SimpleNamespace(
            split=clip.split,
            scene_name=clip.scene_name,
            clip_uid=clip.clip_uid,
            sample_index=int(clip.sample_index),
            frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
            image_sizes=[tuple(int(dim) for dim in frame.shape[:2]) for frame in rgbs],
        ),
    )


def _write_gt_meta(export_dir: Path, clip: MOViFClip, scene_data) -> None:
    meta_path = export_dir / "exports" / "gt_meta.npz"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        meta_path,
        scene_name=np.array(clip.scene_name, dtype=object),
        clip_uid=np.array(clip.clip_uid, dtype=object),
        split=np.array(clip.split, dtype=object),
        sample_index=np.array(int(clip.sample_index), dtype=np.int64),
        frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
        extrinsics=np.asarray(scene_data.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(scene_data.intrinsics, dtype=np.float32),
        image_sizes=np.asarray(scene_data.aux.image_sizes, dtype=np.int32),
    )


def _prediction_paths_for_clip(work_dir: str | Path, clip: MOViFClip) -> dict[str, Path]:
    base_dir = Path(work_dir) / "model_results" / "movi_f" / clip.scene_name / clip.clip_dirname
    return {
        "unposed_result": base_dir / "unposed" / "exports" / "mini_npz" / "results.npz",
        "posed_result": base_dir / "posed" / "exports" / "mini_npz" / "results.npz",
        "unposed_raw": base_dir / "unposed" / "exports" / "raw_camera_params.npz",
        "posed_raw": base_dir / "posed" / "exports" / "raw_camera_params.npz",
        "gt_fuse": base_dir / "gt_fuse" / "pcd.ply",
        "unposed_fuse": base_dir / "unposed" / "exports" / "fuse" / "pcd.ply",
        "posed_fuse": base_dir / "posed" / "exports" / "fuse" / "pcd.ply",
    }


def write_movi_f_clip_exports(
    work_dir: str | Path,
    clip: MOViFClip,
    unposed_prediction: Mapping[str, Any],
    posed_prediction: Mapping[str, Any],
    scene_data,
    *,
    unposed_preprocess_meta: Mapping[str, Any] | None = None,
    posed_preprocess_meta: Mapping[str, Any] | None = None,
) -> None:
    base_dir = Path(work_dir) / "model_results" / "movi_f" / clip.scene_name / clip.clip_dirname
    for suffix, prediction, preprocess_meta in (
        ("unposed", unposed_prediction, unposed_preprocess_meta),
        ("posed", posed_prediction, posed_preprocess_meta),
    ):
        export_dir = base_dir / suffix
        _write_results_npz(export_dir, prediction)
        _write_raw_camera_params_npz(export_dir, prediction)
        _write_gt_meta(export_dir, clip, scene_data)
        if preprocess_meta:
            preprocess_path = export_dir / "exports" / "preprocess_meta.npz"
            preprocess_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(preprocess_path, **preprocess_meta)


def _build_gt_clip_point_cloud(
    scene_data,
    gt_fuse_path: Path,
    *,
    max_point_samples: int,
    reuse_existing_fuse: bool,
) -> o3d.geometry.PointCloud:
    if reuse_existing_fuse and gt_fuse_path.exists() and gt_fuse_path.stat().st_size > 0:
        return o3d.io.read_point_cloud(str(gt_fuse_path))

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    for frame_index, (depth, intrinsic, extrinsic, valid) in enumerate(
        zip(
            scene_data.gt_depths,
            scene_data.intrinsics,
            scene_data.extrinsics,
            scene_data.gt_depth_valid_masks,
        )
    ):
        sanitized_depth, depth_valid = _sanitize_depth(depth)
        frame_valid = valid & depth_valid
        points_cam = radial_depth_to_camera_points(sanitized_depth, intrinsic)
        points_world = transform_points(points_cam, np.linalg.inv(_as_homogeneous_extrinsics(extrinsic)))
        if frame_valid.any():
            point_chunks.append(points_world[frame_valid])
            color_chunks.append(_rgb_frame_for_mask(scene_data, frame_index, frame_valid)[frame_valid])
        else:
            logger.warning("MOVi-F GT frame {} has no valid radial depth pixels.", frame_index)

    all_points = np.concatenate(point_chunks, axis=0) if point_chunks else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.zeros((0, 3), dtype=np.uint8)
    pcd = _point_cloud_from_points(all_points, int(max_point_samples), colors=all_colors)
    _write_point_cloud(
        output_path=gt_fuse_path,
        pcd=pcd,
        intrinsics=scene_data.intrinsics,
        extrinsics=scene_data.extrinsics,
        image_sizes=scene_data.aux.image_sizes,
    )
    return pcd


def _validate_depth_alignment_strategy(strategy: str) -> str:
    strategy = str(strategy)
    if strategy not in DEPTH_ALIGNMENT_STRATEGIES:
        raise ValueError(
            f"Unsupported MOVi-F depth alignment strategy: {strategy}. "
            f"Expected one of {DEPTH_ALIGNMENT_STRATEGIES}."
        )
    return strategy


def _intrinsic_for_prediction_depth_mode(
    *,
    mode: str,
    pred_intrinsic: np.ndarray,
    gt_intrinsic: np.ndarray,
    model_height: int,
    model_width: int,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    if mode in {"recon_unposed", "depth_unposed"}:
        return _resize_intrinsics_to_original(
            pred_intrinsic,
            model_height=model_height,
            model_width=model_width,
            original_height=image_height,
            original_width=image_width,
        ).astype(np.float32)
    if mode in {"recon_posed", "depth_posed"}:
        return np.asarray(gt_intrinsic, dtype=np.float32)
    raise ValueError(f"Unsupported mode: {mode}")


def _compute_median_depth_alignment_scale(
    *,
    scene_data,
    pred_depths: np.ndarray,
    pred_intrinsics: np.ndarray,
    mode: str,
    sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
    eps: float = 1e-6,
) -> float:
    if sample_stride <= 0:
        raise ValueError(f"sample_stride must be positive, got {sample_stride}")

    model_height, model_width = pred_depths.shape[1:3]
    pred_values: list[np.ndarray] = []
    gt_values: list[np.ndarray] = []
    for frame_index, gt_depth in enumerate(scene_data.gt_depths):
        image_height, image_width = gt_depth.shape
        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        intrinsic = _intrinsic_for_prediction_depth_mode(
            mode=mode,
            pred_intrinsic=pred_intrinsics[frame_index],
            gt_intrinsic=scene_data.intrinsics[frame_index],
            model_height=model_height,
            model_width=model_width,
            image_height=image_height,
            image_width=image_width,
        )
        pred_radial_depth = z_depth_to_radial_depth(pred_depth, intrinsic)
        gt_depth_sanitized, gt_valid = _sanitize_depth(gt_depth)
        valid = gt_valid & pred_valid & scene_data.gt_depth_valid_masks[frame_index]
        if sample_stride > 1:
            valid = valid[::sample_stride, ::sample_stride]
            pred_radial_depth = pred_radial_depth[::sample_stride, ::sample_stride]
            gt_depth_sanitized = gt_depth_sanitized[::sample_stride, ::sample_stride]
        if valid.any():
            pred_values.append(pred_radial_depth[valid])
            gt_values.append(gt_depth_sanitized[valid])

    if not pred_values:
        return 1.0
    pred_concat = np.concatenate(pred_values, axis=0)
    gt_concat = np.concatenate(gt_values, axis=0)
    pred_median = float(np.median(pred_concat))
    if not np.isfinite(pred_median) or pred_median <= eps:
        return 1.0
    scale = float(np.median(gt_concat) / pred_median)
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _compute_depth_alignment_scales(
    *,
    scene_data,
    pred_depths: np.ndarray,
    pred_extrinsics: np.ndarray,
    pred_intrinsics: np.ndarray,
    mode: str,
    strategy: str,
    sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> tuple[float, float]:
    strategy = _validate_depth_alignment_strategy(strategy)
    pose_scale, _ = _compute_clip_alignment(pred_extrinsics, scene_data.extrinsics)
    if strategy == "pose":
        return float(pose_scale), float(pose_scale)
    if strategy == "none":
        return 1.0, float(pose_scale)
    return (
        _compute_median_depth_alignment_scale(
            scene_data=scene_data,
            pred_depths=pred_depths,
            pred_intrinsics=pred_intrinsics,
            mode=mode,
            sample_stride=sample_stride,
        ),
        float(pose_scale),
    )


def _compute_depth_alignment_scales_for_result(
    *,
    scene_data,
    result_path: Path,
    mode: str,
    strategy: str,
    sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> tuple[float, float]:
    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)
        pred_intrinsics = np.asarray(pred_npz["intrinsics"], dtype=np.float32)
    return _compute_depth_alignment_scales(
        scene_data=scene_data,
        pred_depths=pred_depths,
        pred_extrinsics=pred_extrinsics,
        pred_intrinsics=pred_intrinsics,
        mode=mode,
        strategy=strategy,
        sample_stride=sample_stride,
    )


def _fuse_prediction_for_mode(
    *,
    scene_data,
    result_path: Path,
    fuse_path: Path,
    mode: str,
    max_point_samples: int,
    reuse_existing_fuse: bool,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
    depth_alignment_scale: float | None = None,
) -> o3d.geometry.PointCloud:
    if reuse_existing_fuse and fuse_path.exists() and fuse_path.stat().st_size > 0:
        return o3d.io.read_point_cloud(str(fuse_path))

    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)
        pred_intrinsics = np.asarray(pred_npz["intrinsics"], dtype=np.float32)

    if depth_alignment_scale is None:
        depth_alignment_scale, _ = _compute_depth_alignment_scales(
            scene_data=scene_data,
            pred_depths=pred_depths,
            pred_extrinsics=pred_extrinsics,
            pred_intrinsics=pred_intrinsics,
            mode=mode,
            strategy=depth_alignment_strategy,
            sample_stride=depth_alignment_sample_stride,
        )
    _, aligned_extrinsics = _compute_clip_alignment(pred_extrinsics, scene_data.extrinsics)
    scale = float(depth_alignment_scale)
    model_height, model_width = pred_depths.shape[1:3]

    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []
    fused_intrinsics: list[np.ndarray] = []
    for frame_index, gt_depth in enumerate(scene_data.gt_depths):
        image_height, image_width = gt_depth.shape
        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        _, gt_valid = _sanitize_depth(gt_depth)
        gt_mask = np.asarray(scene_data.gt_depth_valid_masks[frame_index], dtype=bool)
        frame_valid = pred_valid & gt_valid & gt_mask
        pred_depth = pred_depth * scale

        if mode in {"recon_unposed", "recon_posed"}:
            intrinsic = _intrinsic_for_prediction_depth_mode(
                mode=mode,
                pred_intrinsic=pred_intrinsics[frame_index],
                gt_intrinsic=scene_data.intrinsics[frame_index],
                model_height=model_height,
                model_width=model_width,
                image_height=image_height,
                image_width=image_width,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "recon_unposed":
            extrinsic = aligned_extrinsics[frame_index]
        elif mode == "recon_posed":
            extrinsic = np.asarray(scene_data.extrinsics[frame_index], dtype=np.float32)

        points_cam = z_depth_to_camera_points(pred_depth, intrinsic)
        points_world = transform_points(points_cam, np.linalg.inv(_as_homogeneous_extrinsics(extrinsic)))
        if frame_valid.any():
            point_chunks.append(points_world[frame_valid])
            color_chunks.append(_rgb_frame_for_mask(scene_data, frame_index, frame_valid)[frame_valid])
        fused_intrinsics.append(intrinsic.astype(np.float32))

    all_points = np.concatenate(point_chunks, axis=0) if point_chunks else np.zeros((0, 3), dtype=np.float32)
    all_colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.zeros((0, 3), dtype=np.uint8)
    fused_extrinsics = aligned_extrinsics if mode == "recon_unposed" else np.asarray(scene_data.extrinsics, dtype=np.float32)
    pcd = _point_cloud_from_points(all_points, int(max_point_samples), colors=all_colors)
    _write_point_cloud(
        output_path=fuse_path,
        pcd=pcd,
        intrinsics=np.stack(fused_intrinsics, axis=0).astype(np.float32),
        extrinsics=fused_extrinsics,
        image_sizes=scene_data.aux.image_sizes,
    )
    return pcd


def _evaluate_depth_for_mode(
    *,
    scene_data,
    result_path: Path,
    mode: str,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
    depth_alignment_scale: float | None = None,
    depth_pose_alignment_scale: float | None = None,
) -> dict[str, float]:
    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)
        pred_intrinsics = np.asarray(pred_npz["intrinsics"], dtype=np.float32)

    if depth_alignment_scale is None or depth_pose_alignment_scale is None:
        computed_scale, computed_pose_scale = _compute_depth_alignment_scales(
            scene_data=scene_data,
            pred_depths=pred_depths,
            pred_extrinsics=pred_extrinsics,
            pred_intrinsics=pred_intrinsics,
            mode=mode,
            strategy=depth_alignment_strategy,
            sample_stride=depth_alignment_sample_stride,
        )
        if depth_alignment_scale is None:
            depth_alignment_scale = computed_scale
        if depth_pose_alignment_scale is None:
            depth_pose_alignment_scale = computed_pose_scale
    scale = float(depth_alignment_scale)
    model_height, model_width = pred_depths.shape[1:3]
    pred_values: list[np.ndarray] = []
    gt_values: list[np.ndarray] = []

    for frame_index, gt_depth in enumerate(scene_data.gt_depths):
        image_height, image_width = gt_depth.shape
        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        pred_depth = pred_depth * scale
        intrinsic = _intrinsic_for_prediction_depth_mode(
            mode=mode,
            pred_intrinsic=pred_intrinsics[frame_index],
            gt_intrinsic=scene_data.intrinsics[frame_index],
            model_height=model_height,
            model_width=model_width,
            image_height=image_height,
            image_width=image_width,
        )
        pred_radial_depth = z_depth_to_radial_depth(pred_depth, intrinsic)
        gt_depth_sanitized, gt_valid = _sanitize_depth(gt_depth)
        valid = gt_valid & pred_valid & scene_data.gt_depth_valid_masks[frame_index]
        if valid.any():
            pred_values.append(pred_radial_depth[valid])
            gt_values.append(gt_depth_sanitized[valid])

    if pred_values:
        pred_concat = np.concatenate(pred_values, axis=0)
        gt_concat = np.concatenate(gt_values, axis=0)
    else:
        pred_concat = np.zeros((0,), dtype=np.float32)
        gt_concat = np.zeros((0,), dtype=np.float32)
    metrics = _compute_depth_metrics_np(pred_concat, gt_concat)
    metrics["depth_alignment_scale"] = float(scale)
    metrics["depth_scale_from_pose_alignment"] = float(depth_pose_alignment_scale)
    metrics["num_frames"] = float(len(scene_data.image_files))
    metrics["depth_gt_type"] = 1.0
    return metrics


def _evaluate_movi_f_clip_metrics(
    *,
    dataset_root: str | Path,
    work_dir: Path,
    clip: MOViFClip,
    max_point_samples: int,
    eval_threshold: float,
    down_sample: float,
    reuse_existing_fuse: bool,
    target_space: str,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> dict[str, dict[str, float]]:
    clip_paths = _prediction_paths_for_clip(work_dir, clip)
    scene_data = load_movi_f_clip_scene_data(dataset_root, clip, target_space=str(target_space))

    pose_metrics = compute_pose_with_gt(
        result_path=clip_paths["unposed_result"],
        gt_extrinsics=scene_data.extrinsics,
    )
    aligned_pose_metrics = compute_umeyama_aligned_pose_with_gt(
        result_path=clip_paths["unposed_result"],
        gt_extrinsics=scene_data.extrinsics,
    )
    unposed_depth_scale, unposed_pose_scale = _compute_depth_alignment_scales_for_result(
        scene_data=scene_data,
        result_path=clip_paths["unposed_result"],
        mode="recon_unposed",
        strategy=depth_alignment_strategy,
        sample_stride=depth_alignment_sample_stride,
    )
    posed_depth_scale, posed_pose_scale = _compute_depth_alignment_scales_for_result(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
        mode="recon_posed",
        strategy=depth_alignment_strategy,
        sample_stride=depth_alignment_sample_stride,
    )
    gt_pcd = _build_gt_clip_point_cloud(
        scene_data,
        clip_paths["gt_fuse"],
        max_point_samples=int(max_point_samples),
        reuse_existing_fuse=bool(reuse_existing_fuse),
    )
    pred_unposed_pcd = _fuse_prediction_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["unposed_result"],
        fuse_path=clip_paths["unposed_fuse"],
        mode="recon_unposed",
        max_point_samples=int(max_point_samples),
        reuse_existing_fuse=bool(reuse_existing_fuse),
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=unposed_depth_scale,
    )
    pred_posed_pcd = _fuse_prediction_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
        fuse_path=clip_paths["posed_fuse"],
        mode="recon_posed",
        max_point_samples=int(max_point_samples),
        reuse_existing_fuse=bool(reuse_existing_fuse),
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=posed_depth_scale,
    )

    recon_unposed_metrics = {
        str(key): float(value)
        for key, value in evaluate_3d_reconstruction(
            pred_unposed_pcd,
            gt_pcd,
            threshold=float(eval_threshold),
            down_sample=float(down_sample),
        ).items()
    }
    recon_posed_metrics = {
        str(key): float(value)
        for key, value in evaluate_3d_reconstruction(
            pred_posed_pcd,
            gt_pcd,
            threshold=float(eval_threshold),
            down_sample=float(down_sample),
        ).items()
    }
    recon_unposed_metrics["depth_alignment_scale"] = float(unposed_depth_scale)
    recon_unposed_metrics["depth_pose_alignment_scale"] = float(unposed_pose_scale)
    recon_posed_metrics["depth_alignment_scale"] = float(posed_depth_scale)
    recon_posed_metrics["depth_pose_alignment_scale"] = float(posed_pose_scale)
    depth_unposed_metrics = _evaluate_depth_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["unposed_result"],
        mode="depth_unposed",
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=unposed_depth_scale,
        depth_pose_alignment_scale=unposed_pose_scale,
    )
    depth_posed_metrics = _evaluate_depth_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
        mode="depth_posed",
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=posed_depth_scale,
        depth_pose_alignment_scale=posed_pose_scale,
    )
    return {
        "pose": pose_metrics,
        "aligned_pose": aligned_pose_metrics,
        "recon_unposed": recon_unposed_metrics,
        "recon_posed": recon_posed_metrics,
        "depth_unposed": depth_unposed_metrics,
        "depth_posed": depth_posed_metrics,
    }


def evaluate_movi_f_work_dir(
    dataset_root: str | Path,
    work_dir: str | Path,
    clips: Sequence[MOViFClip],
    *,
    max_point_samples: int,
    eval_threshold: float,
    down_sample: float,
    reuse_existing_fuse: bool,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    pose_per_clip: dict[str, dict[str, float]] = {}
    aligned_pose_per_clip: dict[str, dict[str, float]] = {}
    recon_unposed_per_clip: dict[str, dict[str, float]] = {}
    recon_posed_per_clip: dict[str, dict[str, float]] = {}
    depth_unposed_per_clip: dict[str, dict[str, float]] = {}
    depth_posed_per_clip: dict[str, dict[str, float]] = {}

    for clip in clips:
        logger.info("Evaluating MOVi-F clip {}", clip.clip_uid)
        clip_metrics = _evaluate_movi_f_clip_metrics(
            dataset_root=dataset_root,
            work_dir=work_dir,
            clip=clip,
            max_point_samples=int(max_point_samples),
            eval_threshold=float(eval_threshold),
            down_sample=float(down_sample),
            reuse_existing_fuse=bool(reuse_existing_fuse),
            target_space=str(target_space),
            depth_alignment_strategy=depth_alignment_strategy,
            depth_alignment_sample_stride=depth_alignment_sample_stride,
        )
        pose_per_clip[clip.clip_key] = clip_metrics["pose"]
        aligned_pose_per_clip[clip.clip_key] = clip_metrics["aligned_pose"]
        recon_unposed_per_clip[clip.clip_key] = clip_metrics["recon_unposed"]
        recon_posed_per_clip[clip.clip_key] = clip_metrics["recon_posed"]
        depth_unposed_per_clip[clip.clip_key] = clip_metrics["depth_unposed"]
        depth_posed_per_clip[clip.clip_key] = clip_metrics["depth_posed"]
        gc.collect()

    metric_dir = work_dir / "metric_results"
    pose_payload = _aggregate_metric_records(pose_per_clip)
    aligned_pose_payload = _aggregate_metric_records(aligned_pose_per_clip)
    recon_unposed_payload = _aggregate_metric_records(recon_unposed_per_clip)
    recon_posed_payload = _aggregate_metric_records(recon_posed_per_clip)
    depth_unposed_payload = _aggregate_metric_records(depth_unposed_per_clip)
    depth_posed_payload = _aggregate_metric_records(depth_posed_per_clip)

    dump_json(metric_dir / "movi_f_pose.json", pose_payload)
    dump_json(metric_dir / "movi_f_aligned_pose.json", aligned_pose_payload)
    dump_json(metric_dir / "movi_f_recon_unposed.json", recon_unposed_payload)
    dump_json(metric_dir / "movi_f_recon_posed.json", recon_posed_payload)
    dump_json(metric_dir / "movi_f_depth_unposed.json", depth_unposed_payload)
    dump_json(metric_dir / "movi_f_depth_posed.json", depth_posed_payload)

    return {
        "pose": pose_payload,
        "aligned_pose": aligned_pose_payload,
        "recon_unposed": recon_unposed_payload,
        "recon_posed": recon_posed_payload,
        "depth_unposed": depth_unposed_payload,
        "depth_posed": depth_posed_payload,
    }


def evaluate_movi_f_raw_camera_diagnostics(
    work_dir: str | Path,
    clips: Sequence[MOViFClip],
    *,
    dataset_root: str | Path,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    raw_camera_pose_unposed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_pose_posed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_scale_unposed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_scale_posed_per_clip: dict[str, dict[str, float]] = {}

    for clip in clips:
        clip_paths = _prediction_paths_for_clip(work_dir, clip)
        if not clip_paths["unposed_raw"].exists() or not clip_paths["posed_raw"].exists():
            logger.warning("Skipping raw camera diagnostics for MOVi-F clip {} because raw_camera_params are missing.", clip.clip_uid)
            continue
        scene_data = load_movi_f_clip_scene_data(dataset_root, clip, target_space=str(target_space))
        raw_camera_pose_unposed_per_clip[clip.clip_key] = compute_pose_with_gt(
            result_path=clip_paths["unposed_raw"],
            gt_extrinsics=scene_data.extrinsics,
        )
        raw_camera_pose_posed_per_clip[clip.clip_key] = compute_pose_with_gt(
            result_path=clip_paths["posed_raw"],
            gt_extrinsics=scene_data.extrinsics,
        )
        with np.load(clip_paths["unposed_raw"]) as unposed_raw_npz:
            raw_camera_scale_unposed_per_clip[clip.clip_key] = compute_raw_camera_scale_metrics(
                unposed_raw_npz["extrinsics"],
                scene_data.extrinsics,
            )
        with np.load(clip_paths["posed_raw"]) as posed_raw_npz:
            raw_camera_scale_posed_per_clip[clip.clip_key] = compute_raw_camera_scale_metrics(
                posed_raw_npz["extrinsics"],
                scene_data.extrinsics,
            )

    metric_dir = work_dir / "metric_results"
    pose_unposed_payload = _aggregate_metric_records(raw_camera_pose_unposed_per_clip)
    pose_posed_payload = _aggregate_metric_records(raw_camera_pose_posed_per_clip)
    scale_unposed_payload = _aggregate_metric_records(raw_camera_scale_unposed_per_clip)
    scale_posed_payload = _aggregate_metric_records(raw_camera_scale_posed_per_clip)
    dump_json(metric_dir / "movi_f_raw_camera_pose_unposed.json", pose_unposed_payload)
    dump_json(metric_dir / "movi_f_raw_camera_pose_posed.json", pose_posed_payload)
    dump_json(metric_dir / "movi_f_raw_camera_scale_unposed.json", scale_unposed_payload)
    dump_json(metric_dir / "movi_f_raw_camera_scale_posed.json", scale_posed_payload)
    return {
        "raw_camera_pose_unposed": pose_unposed_payload,
        "raw_camera_pose_posed": pose_posed_payload,
        "raw_camera_scale_unposed": scale_unposed_payload,
        "raw_camera_scale_posed": scale_posed_payload,
    }


def compute_movi_f_scene_flow_metrics(
    pred_flow: np.ndarray,
    target_flow: np.ndarray,
    deform_mask: np.ndarray,
    *,
    scene_flow_scale: float | None,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
    eval_space: str = DEFAULT_SCENE_FLOW_EVAL_SPACE,
    threshold: float = DEFAULT_SCENE_FLOW_THRESHOLD,
) -> dict[str, float]:
    return compute_dynamic_replica_scene_flow_metrics(
        pred_flow,
        target_flow,
        deform_mask,
        scene_flow_scale=scene_flow_scale,
        target_space=target_space,
        eval_space=eval_space,
        threshold=threshold,
    )


def load_movi_f_clip_scene_flow_gt(
    dataset_root: str | Path,
    clip: MOViFClip,
    *,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
) -> dict[str, Any]:
    scene_data = load_movi_f_clip_scene_data(dataset_root, clip, target_space=str(target_space))
    return {
        "scene_flow": np.asarray(scene_data.scene_flow, dtype=np.float32),
        "deform_mask": np.asarray(scene_data.deform_mask, dtype=np.float32),
        "scene_flow_scale": float(scene_data.scene_flow_scale),
    }


def evaluate_movi_f_scene_flow_work_dir(
    *,
    dataset_root: str | Path,
    work_dir: str | Path,
    clips: Sequence[MOViFClip],
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
    eval_space: str = DEFAULT_SCENE_FLOW_EVAL_SPACE,
    threshold: float = DEFAULT_SCENE_FLOW_THRESHOLD,
    scene_flow_gt_loader: Any | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    gt_loader = scene_flow_gt_loader or load_movi_f_clip_scene_flow_gt
    per_mode_clip_metrics: dict[str, dict[str, dict[str, float]]] = {
        "unposed": {},
        "posed": {},
    }

    for clip in clips:
        clip_paths = _prediction_paths_for_clip(work_dir, clip)
        gt_payload = gt_loader(
            dataset_root,
            clip,
            target_space=str(target_space),
        )
        for mode in ("unposed", "posed"):
            result_path = clip_paths[f"{mode}_result"]
            if not result_path.exists():
                logger.warning("Skipping MOVi-F scene-flow {} metrics for missing export {}", mode, result_path)
                continue
            with np.load(result_path, allow_pickle=True) as pred_npz:
                if "scene_flow" not in pred_npz:
                    logger.warning("Skipping MOVi-F scene-flow {} metrics for {} because scene_flow is missing.", mode, clip.clip_uid)
                    continue
                pred_scene_flow = np.asarray(pred_npz["scene_flow"], dtype=np.float32)
            metrics = compute_movi_f_scene_flow_metrics(
                pred_scene_flow,
                gt_payload["scene_flow"],
                gt_payload["deform_mask"],
                scene_flow_scale=float(gt_payload.get("scene_flow_scale", 1.0)),
                target_space=str(target_space),
                eval_space=str(eval_space),
                threshold=float(threshold),
            )
            metrics["num_frames"] = float(len(clip.frame_indices))
            per_mode_clip_metrics[mode][clip.clip_key] = metrics

    if not per_mode_clip_metrics["unposed"] and not per_mode_clip_metrics["posed"]:
        raise RuntimeError(
            f"No scene_flow predictions were found under {work_dir}. "
            "Rerun inference with --da3_enable_scene_flow_head; "
            "older exports only contain depth/camera arrays."
        )

    metric_dir = work_dir / "metric_results"
    unposed_payload = _aggregate_metric_records(per_mode_clip_metrics["unposed"])
    posed_payload = _aggregate_metric_records(per_mode_clip_metrics["posed"])
    dump_json(metric_dir / "movi_f_scene_flow_unposed.json", unposed_payload)
    dump_json(metric_dir / "movi_f_scene_flow_posed.json", posed_payload)
    return {
        "scene_flow_unposed": unposed_payload,
        "scene_flow_posed": posed_payload,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MOVi-F-Val benchmark with reconstruction, radial-depth, camera pose, "
            "and optional scene-flow metrics for raw DA3 anyview-base and/or PLAN4D DA3 stitched."
        )
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="both",
        choices=["da3_anyview_base", "da3_stitched", "both"],
        help="Which model path to evaluate.",
    )
    parser.add_argument(
        "--movi_f_root",
        type=str,
        default=default_movi_f_root(),
        help="MOVi-F TFDS builder root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="MOVi-F split. Defaults to validation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for benchmark exports and metric JSON files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Stitched checkpoint path. Required when model_kind includes da3_stitched.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=DEFAULT_CLIP_LENGTH,
        help="Clip length for sequential evaluation. Tail clips shorter than this are kept.",
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=DEFAULT_INPUT_RESOLUTION,
        help="Target input resolution passed to the model-side preprocessing.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional override for stitched loader temporal length. Defaults to the longest selected clip.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of MOVi-F video names.",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=-1,
        help="Optional clip cap for debugging. Use -1 to evaluate all clips.",
    )
    parser.add_argument(
        "--lazy_mode",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LAZY_MODE,
        help="When enabled, evaluate only the first clip per MOVi-F video.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip inference and only evaluate existing exports under --output_dir.",
    )
    parser.add_argument(
        "--reuse_existing_fuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse existing fused PLY files when present.",
    )
    parser.add_argument(
        "--movi_f_depth_alignment_strategy",
        type=str,
        default=DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
        choices=DEPTH_ALIGNMENT_STRATEGIES,
        help=(
            "Depth scale alignment for MOVi-F depth/reconstruction metrics. "
            "Use median to avoid camera-trajectory scale bias on dynamic clips."
        ),
    )
    parser.add_argument(
        "--movi_f_depth_alignment_sample_stride",
        type=int,
        default=DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
        help="Pixel stride used when estimating median depth alignment.",
    )
    parser.add_argument(
        "--da3_model_path",
        type=str,
        default=default_da3_base_model_path(),
        help="Local DA3 weights directory.",
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default=default_wan_vae_path(),
        help="Local Wan VAE directory for stitched inference.",
    )
    parser.add_argument(
        "--da3_config_name",
        type=str,
        default="config.json",
        help="DA3 config filename under --da3_model_path.",
    )
    parser.add_argument(
        "--da3_checkpoint_name",
        type=str,
        default="model.safetensors",
        help="DA3 checkpoint filename under --da3_model_path.",
    )
    parser.add_argument(
        "--da3_enable_scene_flow_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach the pretrained DA3 scene-flow head for stitched DA3 inference.",
    )
    parser.add_argument(
        "--da3_scene_flow_head_checkpoint_path",
        type=str,
        default=default_da3_scene_flow_head_checkpoint_path(),
        help="Checkpoint path for the DA3 scene-flow head.",
    )
    parser.add_argument(
        "--da3_scene_flow_preset",
        type=str,
        default=DEFAULT_DA3_SCENE_FLOW_PRESET,
        help="DA3 registry preset that defines the scene-flow head and adapter.",
    )
    parser.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="first",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view strategy shared by raw anyview and stitched DA3.",
    )
    parser.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_1",
        help="Fallback stitched layer location when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--stitching_layer_config",
        type=str,
        default="conv3d_k5x3x3_o1536_s1x2x2_p2x1x1",
        help="Fallback stitched layer spec when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="r512,a256,d0.05,f0",
        help="Fallback stitched LoRA config when checkpoint metadata is absent.",
    )
    parser.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set when the stitched checkpoint stores fully fine-tuned stitched parts.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint passed through to stitched loader when local paths are missing.",
    )
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help="Preprocess semantics for stitched DA3 inference.",
    )
    parser.add_argument(
        "--stitched_da3_sky_postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply official DA3 mono sky postprocess for stitched predictions.",
    )
    parser.add_argument(
        "--stitched_da3_posed_rerun_with_gt_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, posed evaluation reruns stitched inference with GT camera tokens.",
    )
    parser.add_argument(
        "--stitched_da3_camera_param_source",
        type=str,
        default="ray_map",
        choices=["ray_map", "camera_token"],
        help="Camera parameter decode source for stitched DA3.",
    )
    parser.add_argument(
        "--max_point_samples",
        type=int,
        default=DEFAULT_MAX_POINT_SAMPLES,
        help="Maximum dense point samples per clip for point-cloud reconstruction metrics.",
    )
    parser.add_argument(
        "--eval_threshold",
        type=float,
        default=DEFAULT_EVAL_THRESHOLD,
        help="Distance threshold for reconstruction precision/recall/F-score.",
    )
    parser.add_argument(
        "--down_sample",
        type=float,
        default=DEFAULT_DOWN_SAMPLE,
        help="Voxel size for reconstruction metric down-sampling.",
    )
    parser.add_argument(
        "--enable_scene_flow_eval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SCENE_FLOW_EVAL,
        help="Evaluate stitched DA3 scene-flow predictions against MOVi-F reconstructed GT flow.",
    )
    parser.add_argument(
        "--scene_flow_target_space",
        type=str,
        default=DEFAULT_SCENE_FLOW_TARGET_SPACE,
        choices=["relative", "metric"],
        help="Space of the saved model scene-flow predictions and MOVi-F target flow.",
    )
    parser.add_argument(
        "--scene_flow_eval_space",
        type=str,
        default=DEFAULT_SCENE_FLOW_EVAL_SPACE,
        choices=["auto", "native", "metric"],
        help="Scene-flow metric space. Use metric for relative predictions restored by scene_flow_scale.",
    )
    parser.add_argument(
        "--scene_flow_threshold",
        type=float,
        default=DEFAULT_SCENE_FLOW_THRESHOLD,
        help="APD threshold for scene-flow EPE metrics.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.movi_f_root).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"MOVi-F root does not exist: {dataset_root}")
    if args.clip_length <= 0:
        raise ValueError("--clip_length must be positive.")
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be positive when provided.")
    if args.max_clips == 0:
        raise ValueError("--max_clips must be positive or -1.")
    if args.max_point_samples <= 0:
        raise ValueError("--max_point_samples must be positive.")
    if args.movi_f_depth_alignment_sample_stride <= 0:
        raise ValueError("--movi_f_depth_alignment_sample_stride must be positive.")
    if args.model_kind in {"da3_stitched", "both"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for da3_stitched or both.")
    if args.scene_flow_threshold <= 0:
        raise ValueError("--scene_flow_threshold must be positive.")
    if (
        args.enable_scene_flow_eval
        and not args.eval_only
        and args.model_kind in {"da3_stitched", "both"}
        and not args.da3_enable_scene_flow_head
    ):
        logger.warning(
            "--enable_scene_flow_eval is set but --da3_enable_scene_flow_head is disabled; "
            "new stitched exports will not contain scene_flow."
        )


def resolve_runtime_max_frames(args: argparse.Namespace, clips: Sequence[MOViFClip]) -> int:
    if not clips:
        raise ValueError("No MOVi-F clips selected for evaluation.")
    required_max_frames = max(int(clip.num_frames) for clip in clips)
    if args.max_frames is None:
        args.max_frames = int(required_max_frames)
    elif int(args.max_frames) < required_max_frames:
        raise ValueError(
            f"--max_frames={args.max_frames} is smaller than the longest selected clip "
            f"({required_max_frames} frames)."
        )
    else:
        args.max_frames = int(args.max_frames)
    return int(args.max_frames)


def _build_run_metadata(args: argparse.Namespace, *, model_kind: str, clips: Sequence[MOViFClip]) -> dict[str, Any]:
    return {
        "benchmark": "movi_f",
        "model_kind": model_kind,
        "movi_f_root": str(Path(args.movi_f_root).expanduser().resolve()),
        "split": str(args.split),
        "clip_length": int(args.clip_length),
        "input_resolution": int(args.input_resolution),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "num_clips": int(len(clips)),
        "clips": [clip.to_dict() for clip in clips],
        "da3_enable_scene_flow_head": bool(args.da3_enable_scene_flow_head),
        "da3_scene_flow_head_checkpoint_path": str(args.da3_scene_flow_head_checkpoint_path),
        "da3_scene_flow_preset": str(args.da3_scene_flow_preset),
        "enable_scene_flow_eval": bool(args.enable_scene_flow_eval),
        "scene_flow_target_space": str(args.scene_flow_target_space),
        "scene_flow_eval_space": str(args.scene_flow_eval_space),
        "scene_flow_threshold": float(args.scene_flow_threshold),
        "movi_f_depth_alignment_strategy": str(args.movi_f_depth_alignment_strategy),
        "movi_f_depth_alignment_sample_stride": int(args.movi_f_depth_alignment_sample_stride),
    }


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    clips: Sequence[MOViFClip],
) -> dict[str, Any]:
    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)
    dump_json(output_root / "run_meta.json", _build_run_metadata(args, model_kind=model_kind, clips=clips))

    runner = None
    if model_kind == "da3_anyview_base":
        runner = load_anyview_api(args)
    elif model_kind == "da3_stitched":
        runner = load_stitched_runner(
            args,
            da3_process_res_method=getattr(args, "da3_process_res_method", "upper_bound_resize"),
        )
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    if not args.eval_only:
        for clip in clips:
            logger.info("Running MOVi-F inference for clip {}", clip.clip_uid)
            scene_data = load_movi_f_clip_scene_data(
                args.movi_f_root,
                clip,
                target_space=str(args.scene_flow_target_space),
            )
            if model_kind == "da3_anyview_base":
                unposed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=args.input_resolution,
                    ref_view_strategy=args.da3_ref_view_strategy,
                    input_images=scene_data.image_files,
                )
                posed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=args.input_resolution,
                    ref_view_strategy=args.da3_ref_view_strategy,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                    input_images=scene_data.image_files,
                )
                unposed_preprocess_meta = None
                posed_preprocess_meta = None
            elif model_kind == "da3_stitched":
                unposed_prediction, unposed_preprocess_meta = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=args,
                    process_res_method=getattr(args, "da3_process_res_method", "upper_bound_resize"),
                    input_images=scene_data.image_files,
                )
                if getattr(args, "stitched_da3_posed_rerun_with_gt_camera", True):
                    posed_prediction, posed_preprocess_meta = run_da3_stitched_scene(
                        runner,
                        scene_data=scene_data,
                        args=args,
                        extrinsics=scene_data.extrinsics,
                        intrinsics=scene_data.intrinsics,
                        process_res_method=getattr(args, "da3_process_res_method", "upper_bound_resize"),
                        input_images=scene_data.image_files,
                    )
                else:
                    posed_prediction = dict(unposed_prediction)
                    posed_preprocess_meta = dict(unposed_preprocess_meta) if unposed_preprocess_meta else None
            else:
                raise ValueError(f"Unsupported model_kind: {model_kind}")

            write_movi_f_clip_exports(
                output_root,
                clip,
                unposed_prediction,
                posed_prediction,
                scene_data,
                unposed_preprocess_meta=unposed_preprocess_meta,
                posed_preprocess_meta=posed_preprocess_meta,
            )
            gc.collect()

    metrics = evaluate_movi_f_work_dir(
        args.movi_f_root,
        output_root,
        clips,
        max_point_samples=args.max_point_samples,
        eval_threshold=args.eval_threshold,
        down_sample=args.down_sample,
        reuse_existing_fuse=args.reuse_existing_fuse,
        target_space=str(args.scene_flow_target_space),
        depth_alignment_strategy=args.movi_f_depth_alignment_strategy,
        depth_alignment_sample_stride=args.movi_f_depth_alignment_sample_stride,
    )
    metrics.update(
        evaluate_movi_f_raw_camera_diagnostics(
            output_root,
            clips,
            dataset_root=args.movi_f_root,
            target_space=str(args.scene_flow_target_space),
        )
    )
    if args.enable_scene_flow_eval:
        if model_kind != "da3_stitched":
            logger.warning("Scene-flow evaluation is only enabled for da3_stitched exports; skipping {}.", model_kind)
        else:
            metrics.update(
                evaluate_movi_f_scene_flow_work_dir(
                    dataset_root=args.movi_f_root,
                    work_dir=output_root,
                    clips=clips,
                    target_space=str(args.scene_flow_target_space),
                    eval_space=str(args.scene_flow_eval_space),
                    threshold=float(args.scene_flow_threshold),
                )
            )
    dump_json(output_root / "metric_results" / "movi_f_summary.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    clips = build_movi_f_clips(
        args.movi_f_root,
        split=args.split,
        clip_length=args.clip_length,
        scenes=args.scenes,
        max_clips=int(args.max_clips),
        lazy_mode=bool(args.lazy_mode),
        target_space=str(args.scene_flow_target_space),
    )
    if not clips:
        raise ValueError("No MOVi-F clips selected for evaluation.")
    resolve_runtime_max_frames(args, clips)
    logger.info(
        "MOVi-F benchmark uses {} clips from {} videos.",
        len(clips),
        len({clip.scene_name for clip in clips}),
    )

    requested_model_kinds = (
        ["da3_anyview_base", "da3_stitched"]
        if args.model_kind == "both"
        else [str(args.model_kind)]
    )
    all_metrics: dict[str, Any] = {}
    for model_kind in requested_model_kinds:
        logger.info("Evaluating model_kind={}", model_kind)
        all_metrics[model_kind] = evaluate_model_kind(
            args,
            model_kind=model_kind,
            clips=clips,
        )
    if args.model_kind == "both":
        dump_json(Path(args.output_dir) / "movi_f_summary_all_models.json", all_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
