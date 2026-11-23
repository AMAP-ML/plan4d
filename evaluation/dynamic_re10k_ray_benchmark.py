from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

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

for import_path in (PROJECT_ROOT, PLAN4D_ROOT, DA3_SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from depth_anything_3.bench.utils import evaluate_3d_reconstruction, write_point_cloud_with_cameras
from depth_anything_3.utils.pose_align import align_poses_umeyama, transform_points_sim3
from evaluation.dynamic_replica_ray_benchmark import (
    _aggregate_metric_records,
    _compute_depth_metrics_np,
    _integrate_depth_sequence_to_pcd,
    _resize_intrinsics_to_original,
    _sanitize_depth,
)
from evaluation.eth3d_ray_benchmark import (
    _write_raw_camera_params_npz,
    _write_results_npz,
    clone_prediction_dict,
    compute_pose_with_gt,
    compute_raw_camera_scale_metrics,
    compute_umeyama_aligned_pose_with_gt,
    default_wan_vae_path,
    dump_json,
    load_anyview_api,
    load_stitched_runner,
    mean_of_dicts,
    run_da3_anyview_scene,
    run_da3_stitched_scene,
)


DEFAULT_INPUT_RESOLUTION = 504
DEFAULT_COLMAP_POINT_ERROR_MAX = 2.5
DEFAULT_COLMAP_POINT_TRACK_MIN = 4
DEFAULT_MIN_COLMAP_REGISTERED_FRAMES = 4
DEFAULT_MIN_FILTERED_POINTS = 100
DEFAULT_MIN_VALID_SPARSE_SAMPLES = 64
DEFAULT_RECON_THRESHOLD_RATIO = 0.01
DEFAULT_MODEL_KIND = "both"
DEFAULT_DA3_MODEL_PATH = str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())
DEFAULT_DA3_STREAMING_MODEL_PATH = str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())


@dataclass(frozen=True)
class DynamicRE10KSceneData:
    scene_name: str
    frame_indices: np.ndarray
    image_files: list[str]
    mask_files: list[str]
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    aux: SimpleNamespace


@dataclass(frozen=True)
class DynamicRE10KColmapData:
    frame_indices: np.ndarray
    image_ids: np.ndarray
    image_names: list[str]
    camera_ids: np.ndarray
    extrinsics: np.ndarray
    cameras: dict[int, dict[str, Any]]
    points_by_id: dict[int, dict[str, Any]]
    filtered_point_ids: list[int]
    filtered_points: np.ndarray
    filtered_colors: np.ndarray
    frame_to_observations: dict[int, list[dict[str, Any]]]
    registered_frames: int
    filtered_point_count: int
    bbox_diagonal: float


def default_dynamic_re10k_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "Dynamic-RE10K").resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Dynamic-RE10K benchmark with COLMAP-camera pose metrics and "
            "COLMAP sparse static depth/reconstruction diagnostics."
        )
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default=DEFAULT_MODEL_KIND,
        choices=["da3_anyview_base", "da3_stitched", "both"],
        help="Which model path to evaluate.",
    )
    parser.add_argument(
        "--dynamic_re10k_root",
        type=str,
        default=default_dynamic_re10k_root(),
        help="Dynamic-RE10K dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for exports and metric JSON files.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Stitched checkpoint path. Required when model_kind includes da3_stitched.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Optional maximum frames per scene after metadata-order truncation. -1 keeps all metadata frames.",
    )
    parser.add_argument(
        "--eval_only",
        "--reuse_predictions",
        dest="eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip model inference and only evaluate existing prediction exports under "
            "--output_dir/<model_kind>/model_results/."
        ),
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=DEFAULT_INPUT_RESOLUTION,
        help="Target model-side input resolution.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of Dynamic-RE10K scene names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--da3_model_path",
        type=str,
        default=DEFAULT_DA3_MODEL_PATH,
        help="Local DA3 anyview weights directory.",
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
        help="Apply official DA3 mono sky postprocess on stitched predictions.",
    )
    parser.add_argument(
        "--stitched_da3_posed_rerun_with_gt_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, posed evaluation reruns stitched inference with metadata GT camera tokens.",
    )
    parser.add_argument(
        "--stitched_da3_camera_param_source",
        type=str,
        default="ray_map",
        choices=["ray_map", "camera_token"],
        help="Which stitched DA3 output to decode camera parameters from.",
    )
    parser.add_argument(
        "--dynamic_re10k_recon_aggressive_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, reconstruction fusion applies sequence-global scale+shift alignment "
            "before TSDF fusion. Disabled by default to keep reconstruction alignment scale-only."
        ),
    )
    parser.add_argument(
        "--dynamic_re10k_colmap_point_error_max",
        type=float,
        default=DEFAULT_COLMAP_POINT_ERROR_MAX,
        help="Maximum COLMAP sparse point reprojection error kept for sparse diagnostics.",
    )
    parser.add_argument(
        "--dynamic_re10k_colmap_point_track_min",
        type=int,
        default=DEFAULT_COLMAP_POINT_TRACK_MIN,
        help="Minimum COLMAP sparse point track length kept for sparse diagnostics.",
    )
    parser.add_argument(
        "--dynamic_re10k_min_colmap_registered_frames",
        type=int,
        default=DEFAULT_MIN_COLMAP_REGISTERED_FRAMES,
        help="Minimum overlapping COLMAP-registered frames required for COLMAP-based diagnostics.",
    )
    parser.add_argument(
        "--dynamic_re10k_min_filtered_points",
        type=int,
        default=DEFAULT_MIN_FILTERED_POINTS,
        help="Minimum filtered COLMAP sparse points required for COLMAP-based diagnostics.",
    )
    parser.add_argument(
        "--dynamic_re10k_min_valid_sparse_samples",
        type=int,
        default=DEFAULT_MIN_VALID_SPARSE_SAMPLES,
        help="Minimum valid sparse depth samples required for COLMAP-based depth/reconstruction diagnostics.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dynamic_re10k_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dynamic-RE10K root does not exist: {dataset_root}")
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.max_frames == 0:
        raise ValueError("--max_frames must be positive or -1 to disable truncation.")
    if args.model_kind in {"da3_stitched", "both"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for da3_stitched or both.")
    if args.dynamic_re10k_colmap_point_error_max <= 0.0:
        raise ValueError("--dynamic_re10k_colmap_point_error_max must be positive.")
    if args.dynamic_re10k_colmap_point_track_min <= 0:
        raise ValueError("--dynamic_re10k_colmap_point_track_min must be positive.")


def _to_float_dict(values: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def list_dynamic_re10k_scenes(dataset_root: str | Path) -> list[str]:
    metadata_root = Path(dataset_root) / "metadata"
    return sorted(path.stem for path in metadata_root.glob("*.json"))


def _intrinsics_from_fxfycxcy(values: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = [float(value) for value in values]
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = fx
    intrinsics[1, 1] = fy
    intrinsics[0, 2] = cx
    intrinsics[1, 2] = cy
    return intrinsics


def load_dynamic_re10k_scene_data(
    dataset_root: str | Path,
    scene: str,
    max_frames: int = -1,
) -> DynamicRE10KSceneData:
    dataset_root = Path(dataset_root)
    metadata_path = dataset_root / "metadata" / f"{scene}.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    frames = sorted(payload["frames"], key=lambda item: int(item["frame_idx"]))
    if max_frames > 0:
        frames = frames[: int(max_frames)]

    frame_indices: list[int] = []
    image_files: list[str] = []
    mask_files: list[str] = []
    extrinsics: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []

    for frame_item in frames:
        frame_idx = int(frame_item["frame_idx"])
        image_path = dataset_root / "images" / scene / f"{frame_idx:05d}.png"
        mask_path = dataset_root / "binary_masks" / scene / f"{frame_idx:05d}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Dynamic-RE10K image is missing: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Dynamic-RE10K mask is missing: {mask_path}")
        frame_indices.append(frame_idx)
        image_files.append(str(image_path))
        mask_files.append(str(mask_path))
        intrinsics.append(_intrinsics_from_fxfycxcy(frame_item["fxfycxcy"]))
        extrinsics.append(np.asarray(frame_item["w2c"], dtype=np.float32))

    return DynamicRE10KSceneData(
        scene_name=scene,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        image_files=image_files,
        mask_files=mask_files,
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        aux=SimpleNamespace(metadata_path=str(metadata_path)),
    )


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qz * qx + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qz * qx - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _parse_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": np.asarray([float(value) for value in parts[4:]], dtype=np.float32),
        }
    return cameras


def _parse_colmap_points(path: Path) -> dict[int, dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        track_pairs = []
        for index in range(8, len(parts), 2):
            if index + 1 >= len(parts):
                break
            track_pairs.append((int(parts[index]), int(parts[index + 1])))
        points[point_id] = {
            "point3d_id": point_id,
            "xyz": np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32),
            "rgb": np.asarray([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8),
            "error": float(parts[7]),
            "track": tuple(track_pairs),
            "track_length": len(track_pairs),
        }
    return points


def _parse_colmap_images(
    path: Path,
    points_by_id: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    image_rows: list[dict[str, Any]] = []
    frame_to_observations: dict[int, list[dict[str, Any]]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        header = lines[index].strip()
        index += 1
        if not header or header.startswith("#"):
            continue
        parts = header.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = [float(value) for value in parts[1:5]]
        tx, ty, tz = [float(value) for value in parts[5:8]]
        camera_id = int(parts[8])
        image_name = parts[9]
        frame_idx = int(Path(image_name).stem)
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = _quat_to_rotmat(qw, qx, qy, qz)
        extrinsic[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float32)

        observation_line = lines[index].strip() if index < len(lines) else ""
        index += 1
        observation_values = observation_line.split() if observation_line else []
        observations: list[dict[str, Any]] = []
        for obs_index in range(0, len(observation_values), 3):
            if obs_index + 2 >= len(observation_values):
                break
            point3d_id = int(observation_values[obs_index + 2])
            observations.append(
                {
                    "x": float(observation_values[obs_index]),
                    "y": float(observation_values[obs_index + 1]),
                    "point3d_id": point3d_id,
                    "is_filtered_point": point3d_id in points_by_id,
                    "image_id": image_id,
                }
            )

        image_rows.append(
            {
                "image_id": image_id,
                "frame_idx": frame_idx,
                "image_name": image_name,
                "camera_id": camera_id,
                "extrinsic": extrinsic,
            }
        )
        frame_to_observations[frame_idx] = observations

    image_rows.sort(key=lambda item: item["frame_idx"])
    return image_rows, frame_to_observations


def load_dynamic_re10k_colmap_data(
    dataset_root: str | Path,
    scene: str,
    *,
    point_error_max: float = DEFAULT_COLMAP_POINT_ERROR_MAX,
    point_track_min: int = DEFAULT_COLMAP_POINT_TRACK_MIN,
) -> DynamicRE10KColmapData:
    sparse_root = Path(dataset_root) / "colmap" / scene / "sparse" / "0_text"
    cameras = _parse_colmap_cameras(sparse_root / "cameras.txt")
    all_points = _parse_colmap_points(sparse_root / "points3D.txt")
    filtered_points = {
        point_id: point
        for point_id, point in all_points.items()
        if point["error"] <= float(point_error_max) and point["track_length"] >= int(point_track_min)
    }
    image_rows, frame_to_observations = _parse_colmap_images(sparse_root / "images.txt", filtered_points)

    filtered_point_ids = sorted(filtered_points)
    if filtered_point_ids:
        filtered_xyz = np.stack([filtered_points[point_id]["xyz"] for point_id in filtered_point_ids], axis=0)
        filtered_rgb = np.stack([filtered_points[point_id]["rgb"] for point_id in filtered_point_ids], axis=0)
        bbox_diagonal = float(np.linalg.norm(filtered_xyz.max(axis=0) - filtered_xyz.min(axis=0)))
    else:
        filtered_xyz = np.zeros((0, 3), dtype=np.float32)
        filtered_rgb = np.zeros((0, 3), dtype=np.uint8)
        bbox_diagonal = 0.0

    return DynamicRE10KColmapData(
        frame_indices=np.asarray([item["frame_idx"] for item in image_rows], dtype=np.int64),
        image_ids=np.asarray([item["image_id"] for item in image_rows], dtype=np.int64),
        image_names=[str(item["image_name"]) for item in image_rows],
        camera_ids=np.asarray([item["camera_id"] for item in image_rows], dtype=np.int64),
        extrinsics=np.asarray([item["extrinsic"] for item in image_rows], dtype=np.float32),
        cameras=cameras,
        points_by_id=filtered_points,
        filtered_point_ids=filtered_point_ids,
        filtered_points=filtered_xyz,
        filtered_colors=filtered_rgb,
        frame_to_observations=frame_to_observations,
        registered_frames=len(image_rows),
        filtered_point_count=len(filtered_point_ids),
        bbox_diagonal=bbox_diagonal,
    )


def gate_dynamic_re10k_colmap_diagnostics(
    *,
    registered_frames: int,
    filtered_point_count: int,
    valid_sparse_samples: int,
    min_registered_frames: int,
    min_filtered_points: int,
    min_valid_sparse_samples: int,
) -> dict[str, Any]:
    if registered_frames < min_registered_frames:
        reason = "registered_frames_insufficient"
    elif filtered_point_count < min_filtered_points:
        reason = "filtered_points_insufficient"
    elif valid_sparse_samples < min_valid_sparse_samples:
        reason = "valid_sparse_samples_insufficient"
    else:
        reason = "ok"
    return {
        "enabled": reason == "ok",
        "reason": reason,
        "registered_frames": int(registered_frames),
        "filtered_point_count": int(filtered_point_count),
        "valid_sparse_samples": int(valid_sparse_samples),
        "min_registered_frames": int(min_registered_frames),
        "min_filtered_points": int(min_filtered_points),
        "min_valid_sparse_samples": int(min_valid_sparse_samples),
    }


def _write_dynamic_re10k_gt_meta(export_dir: Path, scene_data: DynamicRE10KSceneData) -> None:
    meta_path = export_dir / "exports" / "gt_meta.npz"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        meta_path,
        scene_name=np.array(scene_data.scene_name, dtype=object),
        frame_indices=np.asarray(scene_data.frame_indices, dtype=np.int64),
        image_files=np.asarray(scene_data.image_files, dtype=object),
        mask_files=np.asarray(scene_data.mask_files, dtype=object),
        extrinsics=np.asarray(scene_data.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(scene_data.intrinsics, dtype=np.float32),
        metadata_path=np.array(scene_data.aux.metadata_path, dtype=object),
    )


def _load_saved_gt_meta(meta_path: Path) -> DynamicRE10KSceneData:
    with np.load(meta_path, allow_pickle=True) as payload:
        return DynamicRE10KSceneData(
            scene_name=str(payload["scene_name"].item()),
            frame_indices=np.asarray(payload["frame_indices"], dtype=np.int64),
            image_files=[str(path) for path in payload["image_files"]],
            mask_files=[str(path) for path in payload["mask_files"]],
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            aux=SimpleNamespace(metadata_path=str(payload["metadata_path"].item())),
        )


def _scene_paths(work_dir: str | Path, scene: str) -> dict[str, Path]:
    base_dir = Path(work_dir) / "model_results" / "dynamic_re10k" / scene
    return {
        "base_dir": base_dir,
        "unposed_result": base_dir / "unposed" / "exports" / "mini_npz" / "results.npz",
        "posed_result": base_dir / "posed" / "exports" / "mini_npz" / "results.npz",
        "unposed_raw": base_dir / "unposed" / "exports" / "raw_camera_params.npz",
        "posed_raw": base_dir / "posed" / "exports" / "raw_camera_params.npz",
        "gt_meta": base_dir / "unposed" / "exports" / "gt_meta.npz",
        "gt_sparse_fuse": base_dir / "gt_colmap_sparse" / "pcd.ply",
        "unposed_fuse": base_dir / "unposed" / "exports" / "fuse" / "pcd.ply",
        "posed_fuse": base_dir / "posed" / "exports" / "fuse" / "pcd.ply",
    }


def write_dynamic_re10k_scene_exports(
    work_dir: str | Path,
    scene_name: str,
    unposed_prediction: dict[str, np.ndarray],
    posed_prediction: dict[str, np.ndarray],
    scene_data: DynamicRE10KSceneData,
) -> None:
    base_dir = Path(work_dir) / "model_results" / "dynamic_re10k" / scene_name
    for suffix, prediction in (("unposed", unposed_prediction), ("posed", posed_prediction)):
        export_dir = base_dir / suffix
        _write_results_npz(export_dir, prediction)
        _write_raw_camera_params_npz(export_dir, prediction)
        _write_dynamic_re10k_gt_meta(export_dir, scene_data)


def _assert_existing_scene_exports(work_dir: str | Path, scenes: Sequence[str]) -> None:
    required_keys = ("unposed_result", "posed_result", "unposed_raw", "posed_raw", "gt_meta")
    missing_entries: list[str] = []
    for scene in scenes:
        paths = _scene_paths(work_dir, scene)
        for key in required_keys:
            if not paths[key].exists():
                missing_entries.append(f"{scene}:{key}:{paths[key]}")
    if missing_entries:
        preview = "\n".join(missing_entries[:10])
        if len(missing_entries) > 10:
            preview += f"\n... ({len(missing_entries) - 10} more)"
        raise FileNotFoundError(
            "Dynamic-RE10K eval_only/reuse_predictions requires existing scene exports. Missing:\n"
            f"{preview}"
        )


def _as_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(np.asarray(_as_homogeneous_extrinsics(extrinsics), dtype=np.float64))
    return np.asarray(c2w[:, :3, 3], dtype=np.float64)


def _has_field(payload: Mapping[str, Any] | Any, key: str) -> bool:
    if isinstance(payload, Mapping):
        return key in payload
    return hasattr(payload, key)


def _lookup_field(payload: Mapping[str, Any] | Any, key: str):
    if isinstance(payload, Mapping):
        return payload[key]
    return getattr(payload, key)


def _registered_frame_indices_from_colmap_data(colmap_data: Mapping[str, Any] | Any) -> np.ndarray:
    key = "registered_frame_indices" if _has_field(colmap_data, "registered_frame_indices") else "frame_indices"
    return np.asarray(_lookup_field(colmap_data, key), dtype=np.int64)


def _is_placeholder_metadata_pose_sequence(extrinsics: np.ndarray) -> bool:
    extrinsics = np.asarray(_as_homogeneous_extrinsics(extrinsics), dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[0] == 0:
        return False
    if np.max(np.abs(extrinsics[:, :3, :3] - np.eye(3, dtype=np.float64)[None])) > 1e-6:
        return False
    translations = extrinsics[:, :3, 3]
    if np.max(np.abs(translations[:, 0])) > 1e-6 or np.max(np.abs(translations[:, 1])) > 1e-6:
        return False
    expected_z = -0.05 * np.arange(extrinsics.shape[0], dtype=np.float64)
    return bool(np.max(np.abs(translations[:, 2] - expected_z)) <= 1e-6)


def _compute_pose_metrics_from_arrays(pred_extrinsics: np.ndarray, gt_extrinsics: np.ndarray) -> dict[str, float]:
    from depth_anything_3.bench.utils import compute_pose

    return _to_float_dict(
        compute_pose(
            torch.from_numpy(_as_homogeneous_extrinsics(pred_extrinsics)),
            torch.from_numpy(_as_homogeneous_extrinsics(gt_extrinsics)),
        )
    )


def _compute_umeyama_aligned_pose_metrics_from_arrays(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
) -> dict[str, float]:
    from depth_anything_3.bench.utils import compute_pose

    pred_extrinsics = _as_homogeneous_extrinsics(pred_extrinsics)
    gt_extrinsics = _as_homogeneous_extrinsics(gt_extrinsics)
    _, _, scale, aligned_extrinsics = align_poses_umeyama(
        gt_extrinsics.copy(),
        pred_extrinsics.copy(),
        return_aligned=True,
        ransac=len(gt_extrinsics) >= 10,
        random_state=42,
    )
    aligned_extrinsics = _as_homogeneous_extrinsics(aligned_extrinsics)
    metrics = _to_float_dict(
        compute_pose(
            torch.from_numpy(aligned_extrinsics),
            torch.from_numpy(gt_extrinsics),
        )
    )
    pred_centers = _camera_centers_from_extrinsics(pred_extrinsics)
    aligned_centers = _camera_centers_from_extrinsics(aligned_extrinsics)
    gt_centers = _camera_centers_from_extrinsics(gt_extrinsics)
    center_errors = np.linalg.norm(aligned_centers - gt_centers, axis=-1)
    metrics["umeyama_scale"] = float(scale)
    metrics["center_rmse_before_umeyama"] = float(
        np.sqrt(np.mean(np.linalg.norm(pred_centers - gt_centers, axis=-1) ** 2))
    )
    metrics["center_rmse_after_umeyama"] = float(np.sqrt(np.mean(center_errors ** 2)))
    metrics["center_mean_after_umeyama"] = float(np.mean(center_errors))
    return metrics


def _load_mask_static_valid(mask_path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Dynamic-RE10K mask is missing: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask == 0


def _fit_affine_scale_shift(pred_values: np.ndarray, gt_values: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    pred = pred[valid]
    gt = gt[valid]
    if pred.size < 2:
        return 1.0, 0.0
    design = np.stack([pred, np.ones_like(pred)], axis=1)
    scale, shift = np.linalg.lstsq(design, gt, rcond=None)[0]
    if not np.isfinite(scale):
        scale = 1.0
    if not np.isfinite(shift):
        shift = 0.0
    return float(scale), float(shift)


def _apply_depth_affine(depth: np.ndarray, *, scale: float, shift: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    aligned = scale * depth + shift
    aligned[~np.isfinite(aligned)] = 0.0
    aligned[aligned <= 0.0] = 0.0
    return aligned.astype(np.float32)


def _project_world_points(
    points_world: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float32)
    rotated = points_world @ np.asarray(extrinsic[:3, :3], dtype=np.float32).T + np.asarray(
        extrinsic[:3, 3], dtype=np.float32
    )
    depth = rotated[:, 2]
    pixels = np.zeros((len(points_world), 2), dtype=np.float32)
    valid = depth > 1e-6
    if valid.any():
        pixels_valid = rotated[valid, :2] / depth[valid, None]
        pixels[valid, 0] = pixels_valid[:, 0] * float(intrinsic[0, 0]) + float(intrinsic[0, 2])
        pixels[valid, 1] = pixels_valid[:, 1] * float(intrinsic[1, 1]) + float(intrinsic[1, 2])
    return pixels, depth


def _intrinsics_from_colmap_camera(camera: Mapping[str, Any]) -> np.ndarray:
    params = np.asarray(camera["params"], dtype=np.float32)
    model = str(camera["model"]).upper()
    single_focal_models = {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL",
        "RADIAL_FISHEYE",
        "FOV",
    }
    if model in single_focal_models:
        if params.size < 3:
            raise ValueError(f"COLMAP camera {camera['camera_id']} has too few params for model {model}: {params}")
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    else:
        if params.size < 4:
            raise ValueError(f"COLMAP camera {camera['camera_id']} has too few params for model {model}: {params}")
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def _prepare_colmap_subset(
    scene_data: DynamicRE10KSceneData,
    colmap_data: DynamicRE10KColmapData,
) -> dict[str, Any]:
    frame_to_metadata_index = {
        int(frame_idx): index for index, frame_idx in enumerate(np.asarray(scene_data.frame_indices, dtype=np.int64))
    }
    metadata_indices: list[int] = []
    colmap_indices: list[int] = []
    for colmap_index, frame_idx in enumerate(np.asarray(colmap_data.frame_indices, dtype=np.int64)):
        if int(frame_idx) in frame_to_metadata_index:
            metadata_indices.append(frame_to_metadata_index[int(frame_idx)])
            colmap_indices.append(colmap_index)

    if not metadata_indices:
        return {
            "metadata_indices": np.zeros((0,), dtype=np.int64),
            "colmap_indices": np.zeros((0,), dtype=np.int64),
            "registered_frame_indices": np.zeros((0,), dtype=np.int64),
            "registered_image_ids": np.zeros((0,), dtype=np.int64),
            "colmap_camera_ids": np.zeros((0,), dtype=np.int64),
            "colmap_intrinsics": np.zeros((0, 3, 3), dtype=np.float32),
            "colmap_extrinsics": np.zeros((0, 4, 4), dtype=np.float32),
            "frame_to_point_ids": {},
            "frame_to_observations": {},
            "frame_to_colmap_extrinsic": {},
            "points_colmap_by_id": {},
            "points_colmap": np.zeros((0, 3), dtype=np.float32),
            "colors_colmap": np.zeros((0, 3), dtype=np.uint8),
            "filtered_point_ids": [],
            "bbox_diagonal": 0.0,
        }

    valid_colmap_indices = np.asarray(colmap_indices, dtype=np.int64)
    valid_image_ids = np.asarray(colmap_data.image_ids, dtype=np.int64)[valid_colmap_indices]
    valid_frame_indices = np.asarray(colmap_data.frame_indices, dtype=np.int64)[valid_colmap_indices]
    valid_camera_ids = np.asarray(colmap_data.camera_ids, dtype=np.int64)[valid_colmap_indices]
    valid_image_id_set = {int(image_id) for image_id in valid_image_ids}

    filtered_point_ids: list[int] = []
    points_colmap_by_id: dict[int, np.ndarray] = {}
    colors_colmap_by_id: dict[int, np.ndarray] = {}
    for point_id, point in colmap_data.points_by_id.items():
        if any(int(image_id) in valid_image_id_set for image_id, _ in point["track"]):
            filtered_point_ids.append(int(point_id))
            points_colmap_by_id[int(point_id)] = np.asarray(point["xyz"], dtype=np.float32)
            colors_colmap_by_id[int(point_id)] = np.asarray(point["rgb"], dtype=np.uint8)

    if filtered_point_ids:
        points_colmap = np.stack([points_colmap_by_id[point_id] for point_id in filtered_point_ids], axis=0)
        colors_colmap = np.stack([colors_colmap_by_id[point_id] for point_id in filtered_point_ids], axis=0)
        bbox_diagonal = float(np.linalg.norm(points_colmap.max(axis=0) - points_colmap.min(axis=0)))
    else:
        points_colmap = np.zeros((0, 3), dtype=np.float32)
        colors_colmap = np.zeros((0, 3), dtype=np.uint8)
        bbox_diagonal = 0.0

    frame_to_point_ids: dict[int, list[int]] = {}
    frame_to_observations: dict[int, list[dict[str, Any]]] = {}
    frame_to_colmap_extrinsic: dict[int, np.ndarray] = {}
    colmap_extrinsics = np.asarray(colmap_data.extrinsics, dtype=np.float32)[valid_colmap_indices]
    for local_index, frame_idx in enumerate(valid_frame_indices):
        observations = [
            dict(observation)
            for observation in colmap_data.frame_to_observations.get(int(frame_idx), [])
            if int(observation["point3d_id"]) in points_colmap_by_id
        ]
        frame_to_observations[int(frame_idx)] = observations
        frame_to_point_ids[int(frame_idx)] = [int(observation["point3d_id"]) for observation in observations]
        frame_to_colmap_extrinsic[int(frame_idx)] = np.asarray(colmap_extrinsics[local_index], dtype=np.float32)

    return {
        "metadata_indices": np.asarray(metadata_indices, dtype=np.int64),
        "colmap_indices": valid_colmap_indices,
        "registered_frame_indices": valid_frame_indices,
        "registered_image_ids": valid_image_ids,
        "colmap_camera_ids": valid_camera_ids,
        "colmap_intrinsics": np.stack(
            [_intrinsics_from_colmap_camera(colmap_data.cameras[int(camera_id)]) for camera_id in valid_camera_ids],
            axis=0,
        ).astype(np.float32),
        "colmap_extrinsics": colmap_extrinsics,
        "frame_to_point_ids": frame_to_point_ids,
        "frame_to_observations": frame_to_observations,
        "frame_to_colmap_extrinsic": frame_to_colmap_extrinsic,
        "points_colmap_by_id": points_colmap_by_id,
        "points_colmap": points_colmap,
        "colors_colmap": colors_colmap,
        "filtered_point_ids": filtered_point_ids,
        "bbox_diagonal": bbox_diagonal,
    }


def _collect_sparse_depth_samples_from_colmap_observations(
    scene_data: DynamicRE10KSceneData,
    colmap_data: Mapping[str, Any] | Any,
    pred_depths: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[int, int]]:
    registered_frame_indices = _registered_frame_indices_from_colmap_data(colmap_data)
    frame_to_observations = _lookup_field(colmap_data, "frame_to_observations")
    frame_to_colmap_extrinsic = (
        _lookup_field(colmap_data, "frame_to_colmap_extrinsic")
        if _has_field(colmap_data, "frame_to_colmap_extrinsic")
        else {
            int(frame_idx): np.asarray(_lookup_field(colmap_data, "extrinsics"), dtype=np.float32)[index]
            for index, frame_idx in enumerate(registered_frame_indices.tolist())
        }
    )
    points_colmap_by_id = (
        _lookup_field(colmap_data, "points_colmap_by_id")
        if _has_field(colmap_data, "points_colmap_by_id")
        else _lookup_field(colmap_data, "points_by_id")
    )

    frame_to_metadata_index = {
        int(frame_idx): index for index, frame_idx in enumerate(np.asarray(scene_data.frame_indices, dtype=np.int64))
    }
    pred_samples: list[np.ndarray] = []
    gt_samples: list[np.ndarray] = []
    valid_counts_by_frame: dict[int, int] = {}

    for frame_idx in registered_frame_indices:
        metadata_index = frame_to_metadata_index.get(int(frame_idx))
        if metadata_index is None:
            continue
        image_path = scene_data.image_files[metadata_index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Dynamic-RE10K image is missing: {image_path}")
        original_height, original_width = image.shape[:2]

        pred_depth = cv2.resize(
            np.asarray(pred_depths[metadata_index], dtype=np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        static_valid = _load_mask_static_valid(scene_data.mask_files[metadata_index])

        observations = frame_to_observations.get(int(frame_idx), [])
        frame_extrinsic = np.asarray(frame_to_colmap_extrinsic.get(int(frame_idx)), dtype=np.float32)
        if frame_extrinsic.size == 0 or not observations:
            valid_counts_by_frame[int(frame_idx)] = 0
            continue

        frame_pred_values: list[float] = []
        frame_gt_values: list[float] = []
        rot = np.asarray(frame_extrinsic[:3, :3], dtype=np.float32)
        trans = np.asarray(frame_extrinsic[:3, 3], dtype=np.float32)
        for observation in observations:
            point_world = points_colmap_by_id.get(int(observation["point3d_id"]))
            if point_world is None:
                continue
            if isinstance(point_world, Mapping):
                point_world = point_world.get("xyz")
            if point_world is None:
                continue
            point_camera = np.asarray(point_world, dtype=np.float32) @ rot.T + trans
            gt_depth = float(point_camera[2])
            if gt_depth <= 1e-6:
                continue
            pixel_x = int(np.rint(float(observation["x"])))
            pixel_y = int(np.rint(float(observation["y"])))
            if pixel_x < 0 or pixel_x >= original_width or pixel_y < 0 or pixel_y >= original_height:
                continue
            if not static_valid[pixel_y, pixel_x] or not pred_valid[pixel_y, pixel_x]:
                continue
            frame_pred_values.append(float(pred_depth[pixel_y, pixel_x]))
            frame_gt_values.append(gt_depth)

        valid_counts_by_frame[int(frame_idx)] = len(frame_gt_values)
        if not frame_gt_values:
            continue
        pred_samples.append(np.asarray(frame_pred_values, dtype=np.float32))
        gt_samples.append(np.asarray(frame_gt_values, dtype=np.float32))

    return pred_samples, gt_samples, valid_counts_by_frame


def _collect_sparse_depth_samples(
    scene_data: DynamicRE10KSceneData,
    colmap_subset: Mapping[str, Any],
    pred_depths: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[int, int]]:
    if "frame_to_observations" in colmap_subset:
        return _collect_sparse_depth_samples_from_colmap_observations(
            scene_data=scene_data,
            colmap_data=colmap_subset,
            pred_depths=pred_depths,
        )

    frame_to_metadata_index = {
        int(frame_idx): index for index, frame_idx in enumerate(np.asarray(scene_data.frame_indices, dtype=np.int64))
    }
    pred_samples: list[np.ndarray] = []
    gt_samples: list[np.ndarray] = []
    valid_counts_by_frame: dict[int, int] = {}

    for frame_idx in np.asarray(colmap_subset["registered_frame_indices"], dtype=np.int64):
        metadata_index = frame_to_metadata_index.get(int(frame_idx))
        if metadata_index is None:
            continue
        image_path = scene_data.image_files[metadata_index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Dynamic-RE10K image is missing: {image_path}")
        original_height, original_width = image.shape[:2]

        pred_depth = cv2.resize(
            np.asarray(pred_depths[metadata_index], dtype=np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        static_valid = _load_mask_static_valid(scene_data.mask_files[metadata_index])

        point_ids = colmap_subset["frame_to_point_ids"].get(int(frame_idx), [])
        if not point_ids:
            valid_counts_by_frame[int(frame_idx)] = 0
            continue
        points_world = np.stack(
            [colmap_subset["points_metadata_by_id"][int(point_id)] for point_id in point_ids],
            axis=0,
        ).astype(np.float32)
        pixels, gt_depth = _project_world_points(
            points_world,
            scene_data.extrinsics[metadata_index],
            scene_data.intrinsics[metadata_index],
        )
        pixel_x = np.rint(pixels[:, 0]).astype(np.int64)
        pixel_y = np.rint(pixels[:, 1]).astype(np.int64)
        valid = (
            (gt_depth > 1e-6)
            & (pixel_x >= 0)
            & (pixel_x < original_width)
            & (pixel_y >= 0)
            & (pixel_y < original_height)
        )
        if valid.any():
            visible_valid = static_valid[pixel_y[valid], pixel_x[valid]] & pred_valid[pixel_y[valid], pixel_x[valid]]
            valid_indices = np.flatnonzero(valid)
            valid = valid.copy()
            valid[valid_indices] = visible_valid
        valid_counts_by_frame[int(frame_idx)] = int(np.count_nonzero(valid))
        if not valid.any():
            continue
        pred_samples.append(pred_depth[pixel_y[valid], pixel_x[valid]].astype(np.float32))
        gt_samples.append(gt_depth[valid].astype(np.float32))

    return pred_samples, gt_samples, valid_counts_by_frame


def _evaluate_sparse_depth_for_mode(

    *,
    scene_data: DynamicRE10KSceneData,
    colmap_subset: Mapping[str, Any],
    result_path: Path,
    alignment_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    if alignment_mode not in {"sequence_scale_shift", "per_frame_scale_shift"}:
        raise ValueError(f"Unsupported depth alignment mode: {alignment_mode}")

    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)

    pred_samples, gt_samples, valid_counts_by_frame = _collect_sparse_depth_samples_from_colmap_observations(
        scene_data=scene_data,
        colmap_data=colmap_subset,
        pred_depths=pred_depths,
    )
    total_valid = int(sum(valid_counts_by_frame.values()))
    if total_valid == 0 or not pred_samples:
        empty = _compute_depth_metrics_np(np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        empty["depth_alignment_scale"] = 1.0
        empty["depth_alignment_shift"] = 0.0
        empty["frames_with_valid_samples"] = 0.0
        return empty, {
            "valid_sparse_samples": total_valid,
            "frames_with_valid_samples": 0,
            "valid_counts_by_frame": valid_counts_by_frame,
        }

    if alignment_mode == "sequence_scale_shift":
        scale, shift = _fit_affine_scale_shift(
            np.concatenate(pred_samples, axis=0),
            np.concatenate(gt_samples, axis=0),
        )
        aligned_pred = [np.clip(scale * values + shift, 1e-6, None) for values in pred_samples]
        metrics = _compute_depth_metrics_np(
            np.concatenate(aligned_pred, axis=0),
            np.concatenate(gt_samples, axis=0),
        )
        metrics["depth_alignment_scale"] = float(scale)
        metrics["depth_alignment_shift"] = float(shift)
        metrics["frames_with_valid_samples"] = float(sum(count > 0 for count in valid_counts_by_frame.values()))
        return metrics, {
            "valid_sparse_samples": total_valid,
            "frames_with_valid_samples": int(sum(count > 0 for count in valid_counts_by_frame.values())),
            "valid_counts_by_frame": valid_counts_by_frame,
            "scale": float(scale),
            "shift": float(shift),
        }

    aligned_pred_values: list[np.ndarray] = []
    aligned_gt_values: list[np.ndarray] = []
    scales: list[float] = []
    shifts: list[float] = []
    for pred_values, gt_values in zip(pred_samples, gt_samples):
        if len(pred_values) >= 2:
            frame_scale, frame_shift = _fit_affine_scale_shift(pred_values, gt_values)
        else:
            frame_scale, frame_shift = 1.0, 0.0
        aligned_pred_values.append(np.clip(frame_scale * pred_values + frame_shift, 1e-6, None))
        aligned_gt_values.append(gt_values)
        scales.append(float(frame_scale))
        shifts.append(float(frame_shift))
    metrics = _compute_depth_metrics_np(
        np.concatenate(aligned_pred_values, axis=0),
        np.concatenate(aligned_gt_values, axis=0),
    )
    metrics["depth_alignment_scale"] = float(np.mean(scales)) if scales else 1.0
    metrics["depth_alignment_shift"] = float(np.mean(shifts)) if shifts else 0.0
    metrics["frames_with_valid_samples"] = float(sum(count > 0 for count in valid_counts_by_frame.values()))
    return metrics, {
        "valid_sparse_samples": total_valid,
        "frames_with_valid_samples": int(sum(count > 0 for count in valid_counts_by_frame.values())),
        "valid_counts_by_frame": valid_counts_by_frame,
        "scale": float(np.mean(scales)) if scales else 1.0,
        "shift": float(np.mean(shifts)) if shifts else 0.0,
    }


def _build_sparse_gt_point_cloud(
    colmap_subset: Mapping[str, Any],
    output_path: Path,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(colmap_subset["points_colmap"], dtype=np.float64)
    colors = np.asarray(colmap_subset["colors_colmap"], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors / 255.0, 0.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    if len(np.asarray(colmap_subset["colmap_intrinsics"], dtype=np.float32)) and len(
        np.asarray(colmap_subset["colmap_extrinsics"], dtype=np.float32)
    ):
        write_point_cloud_with_cameras(
            fuse_path=str(output_path),
            pcd=pcd,
            intrinsics=np.asarray(colmap_subset["colmap_intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(colmap_subset["colmap_extrinsics"], dtype=np.float32),
            image_sizes=None,
        )
    return pcd


def _fuse_prediction_for_mode(
    *,
    scene_data: DynamicRE10KSceneData,
    colmap_subset: Mapping[str, Any],
    result_path: Path,
    fuse_path: Path,
    mode: str,
    recon_scale: float,
    recon_shift: float,
) -> o3d.geometry.PointCloud:
    if mode not in {"recon_unposed", "recon_posed"}:
        raise ValueError(f"Unsupported fusion mode: {mode}")

    metadata_indices = np.asarray(colmap_subset["metadata_indices"], dtype=np.int64)
    gt_colmap_extrinsics = np.asarray(colmap_subset["colmap_extrinsics"], dtype=np.float32)
    if metadata_indices.size == 0 or gt_colmap_extrinsics.size == 0:
        empty = o3d.geometry.PointCloud()
        fuse_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(fuse_path), empty)
        return empty

    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)[metadata_indices]
        pred_intrinsics = np.asarray(pred_npz["intrinsics"], dtype=np.float32)[metadata_indices]

    _, _, _, fused_extrinsics = align_poses_umeyama(
        _as_homogeneous_extrinsics(gt_colmap_extrinsics).copy(),
        _as_homogeneous_extrinsics(pred_extrinsics).copy(),
        return_aligned=True,
        ransac=len(gt_colmap_extrinsics) >= 10,
        random_state=42,
    )

    fused_depths: list[np.ndarray] = []
    fused_intrinsics: list[np.ndarray] = []
    fused_image_files: list[str] = []
    for local_index, metadata_index in enumerate(metadata_indices.tolist()):
        image_path = scene_data.image_files[metadata_index]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Dynamic-RE10K image is missing: {image_path}")
        original_height, original_width = image.shape[:2]
        static_valid = _load_mask_static_valid(scene_data.mask_files[metadata_index])
        pred_depth = cv2.resize(
            pred_depths[local_index],
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        pred_depth = _apply_depth_affine(pred_depth, scale=float(recon_scale), shift=float(recon_shift))
        pred_depth[~static_valid] = 0.0
        fused_depths.append(pred_depth.astype(np.float32))
        fused_intrinsics.append(
            _resize_intrinsics_to_original(
                pred_intrinsics[local_index],
                model_height=pred_depths.shape[1],
                model_width=pred_depths.shape[2],
                original_height=original_height,
                original_width=original_width,
            ).astype(np.float32)
        )
        fused_image_files.append(image_path)

    pcd, image_sizes = _integrate_depth_sequence_to_pcd(
        depth_maps=fused_depths,
        image_files=fused_image_files,
        intrinsics=np.asarray(fused_intrinsics, dtype=np.float32),
        extrinsics=np.asarray(fused_extrinsics, dtype=np.float32),
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        max_depth=10.0,
        sampling_number=1_000_000,
    )
    fuse_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(fuse_path), pcd)
    write_point_cloud_with_cameras(
        fuse_path=str(fuse_path),
        pcd=pcd,
        intrinsics=np.asarray(fused_intrinsics, dtype=np.float32),
        extrinsics=np.asarray(fused_extrinsics, dtype=np.float32),
        image_sizes=image_sizes,
    )
    return pcd


def _build_skip_payload(

    *,
    scenes: Sequence[str],
    per_scene: Mapping[str, Mapping[str, float]],
    skipped_scenes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    per_scene_ordered = {scene: dict(per_scene[scene]) for scene in sorted(per_scene)}
    return {
        "per_scene": per_scene_ordered,
        "mean": mean_of_dicts(list(per_scene_ordered.values())) if per_scene_ordered else {},
        "num_scenes_total": len(list(scenes)),
        "num_scenes_evaluated": len(per_scene_ordered),
        "skipped_scenes": {scene: dict(payload) for scene, payload in sorted(skipped_scenes.items())},
    }


def evaluate_dynamic_re10k_work_dir(
    dataset_root: str | Path,
    work_dir: str | Path,
    scenes: Sequence[str],
    *,
    min_colmap_registered_frames: int,
    min_filtered_points: int,
    min_valid_sparse_samples: int,
    colmap_point_error_max: float,
    colmap_point_track_min: int,
    recon_aggressive_alignment: bool,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    camera_metadata_pose_per_scene: dict[str, dict[str, float]] = {}
    camera_metadata_aligned_pose_per_scene: dict[str, dict[str, float]] = {}
    camera_colmap_pose_per_scene: dict[str, dict[str, float]] = {}
    camera_colmap_aligned_pose_per_scene: dict[str, dict[str, float]] = {}
    depth_sequence_unposed_per_scene: dict[str, dict[str, float]] = {}
    depth_sequence_posed_per_scene: dict[str, dict[str, float]] = {}
    depth_per_frame_unposed_per_scene: dict[str, dict[str, float]] = {}
    depth_per_frame_posed_per_scene: dict[str, dict[str, float]] = {}
    recon_unposed_per_scene: dict[str, dict[str, float]] = {}
    recon_posed_per_scene: dict[str, dict[str, float]] = {}

    metadata_pose_skips: dict[str, dict[str, Any]] = {}
    colmap_camera_skips: dict[str, dict[str, Any]] = {}
    sparse_metric_skips: dict[str, dict[str, Any]] = {}

    for scene in scenes:
        paths = _scene_paths(work_dir, scene)
        scene_data = _load_saved_gt_meta(paths["gt_meta"])
        if _is_placeholder_metadata_pose_sequence(scene_data.extrinsics):
            metadata_pose_skips[scene] = {
                "enabled": False,
                "reason": "metadata_pose_placeholder",
                "num_frames": int(len(scene_data.frame_indices)),
            }
        else:
            camera_metadata_pose_per_scene[scene] = compute_pose_with_gt(
                result_path=paths["unposed_result"],
                gt_extrinsics=scene_data.extrinsics,
            )
            camera_metadata_aligned_pose_per_scene[scene] = compute_umeyama_aligned_pose_with_gt(
                result_path=paths["unposed_result"],
                gt_extrinsics=scene_data.extrinsics,
            )

        colmap_data = load_dynamic_re10k_colmap_data(
            dataset_root,
            scene,
            point_error_max=colmap_point_error_max,
            point_track_min=colmap_point_track_min,
        )
        colmap_subset = _prepare_colmap_subset(scene_data, colmap_data)
        registered_overlap = int(len(_registered_frame_indices_from_colmap_data(colmap_data)))
        if registered_overlap >= min_colmap_registered_frames:
            with np.load(paths["unposed_result"], allow_pickle=True) as pred_unposed:
                pred_extrinsics = np.asarray(pred_unposed["extrinsics"], dtype=np.float32)[
                    np.asarray(colmap_subset["metadata_indices"], dtype=np.int64)
                ]
            gt_colmap_extrinsics = np.asarray(colmap_subset["colmap_extrinsics"], dtype=np.float32)
            camera_colmap_pose_per_scene[scene] = _compute_pose_metrics_from_arrays(pred_extrinsics, gt_colmap_extrinsics)
            camera_colmap_aligned_pose_per_scene[scene] = _compute_umeyama_aligned_pose_metrics_from_arrays(
                pred_extrinsics,
                gt_colmap_extrinsics,
            )
        else:
            colmap_camera_skips[scene] = {
                "enabled": False,
                "reason": "registered_frames_insufficient",
                "registered_frames": int(registered_overlap),
                "min_registered_frames": int(min_colmap_registered_frames),
            }

        depth_sequence_unposed, depth_sequence_unposed_meta = _evaluate_sparse_depth_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["unposed_result"],
            alignment_mode="sequence_scale_shift",
        )
        sparse_gate = gate_dynamic_re10k_colmap_diagnostics(
            registered_frames=registered_overlap,
            filtered_point_count=len(colmap_subset["filtered_point_ids"]),
            valid_sparse_samples=int(depth_sequence_unposed_meta["valid_sparse_samples"]),
            min_registered_frames=min_colmap_registered_frames,
            min_filtered_points=min_filtered_points,
            min_valid_sparse_samples=min_valid_sparse_samples,
        )
        if not sparse_gate["enabled"]:
            sparse_metric_skips[scene] = sparse_gate
            continue

        depth_sequence_posed, _ = _evaluate_sparse_depth_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["posed_result"],
            alignment_mode="sequence_scale_shift",
        )
        depth_per_frame_unposed, _ = _evaluate_sparse_depth_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["unposed_result"],
            alignment_mode="per_frame_scale_shift",
        )
        depth_per_frame_posed, _ = _evaluate_sparse_depth_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["posed_result"],
            alignment_mode="per_frame_scale_shift",
        )
        depth_sequence_unposed_per_scene[scene] = depth_sequence_unposed
        depth_sequence_posed_per_scene[scene] = depth_sequence_posed
        depth_per_frame_unposed_per_scene[scene] = depth_per_frame_unposed
        depth_per_frame_posed_per_scene[scene] = depth_per_frame_posed

        gt_sparse_pcd = _build_sparse_gt_point_cloud(colmap_subset, paths["gt_sparse_fuse"])

        unposed_scale = float(depth_sequence_unposed["depth_alignment_scale"])
        unposed_shift = float(depth_sequence_unposed["depth_alignment_shift"]) if recon_aggressive_alignment else 0.0
        posed_scale = float(depth_sequence_posed["depth_alignment_scale"])
        posed_shift = float(depth_sequence_posed["depth_alignment_shift"]) if recon_aggressive_alignment else 0.0
        pred_unposed_pcd = _fuse_prediction_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["unposed_result"],
            fuse_path=paths["unposed_fuse"],
            mode="recon_unposed",
            recon_scale=unposed_scale,
            recon_shift=unposed_shift,
        )
        pred_posed_pcd = _fuse_prediction_for_mode(
            scene_data=scene_data,
            colmap_subset=colmap_subset,
            result_path=paths["posed_result"],
            fuse_path=paths["posed_fuse"],
            mode="recon_posed",
            recon_scale=posed_scale,
            recon_shift=posed_shift,
        )
        threshold = max(float(colmap_subset["bbox_diagonal"]) * DEFAULT_RECON_THRESHOLD_RATIO, 1e-6)
        recon_unposed_metrics = _to_float_dict(
            evaluate_3d_reconstruction(
                pred_unposed_pcd,
                gt_sparse_pcd,
                threshold=threshold,
                down_sample=0.0,
            )
        )
        recon_posed_metrics = _to_float_dict(
            evaluate_3d_reconstruction(
                pred_posed_pcd,
                gt_sparse_pcd,
                threshold=threshold,
                down_sample=0.0,
            )
        )
        recon_unposed_metrics["eval_threshold"] = float(threshold)
        recon_unposed_metrics["depth_alignment_scale"] = unposed_scale
        recon_unposed_metrics["depth_alignment_shift"] = unposed_shift
        recon_posed_metrics["eval_threshold"] = float(threshold)
        recon_posed_metrics["depth_alignment_scale"] = posed_scale
        recon_posed_metrics["depth_alignment_shift"] = posed_shift
        recon_unposed_per_scene[scene] = recon_unposed_metrics
        recon_posed_per_scene[scene] = recon_posed_metrics

    metric_dir = work_dir / "metric_results"
    camera_metadata_pose_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=camera_metadata_pose_per_scene,
        skipped_scenes=metadata_pose_skips,
    )
    camera_metadata_aligned_pose_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=camera_metadata_aligned_pose_per_scene,
        skipped_scenes=metadata_pose_skips,
    )
    camera_colmap_pose_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=camera_colmap_pose_per_scene,
        skipped_scenes=colmap_camera_skips,
    )
    camera_colmap_aligned_pose_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=camera_colmap_aligned_pose_per_scene,
        skipped_scenes=colmap_camera_skips,
    )
    depth_sequence_unposed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=depth_sequence_unposed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )
    depth_sequence_posed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=depth_sequence_posed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )
    depth_per_frame_unposed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=depth_per_frame_unposed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )
    depth_per_frame_posed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=depth_per_frame_posed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )
    recon_unposed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=recon_unposed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )
    recon_posed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=recon_posed_per_scene,
        skipped_scenes=sparse_metric_skips,
    )

    dump_json(metric_dir / "dynamic_re10k_camera_metadata_pose.json", camera_metadata_pose_payload)
    dump_json(metric_dir / "dynamic_re10k_camera_metadata_aligned_pose.json", camera_metadata_aligned_pose_payload)
    dump_json(metric_dir / "dynamic_re10k_camera_colmap_pose.json", camera_colmap_pose_payload)
    dump_json(metric_dir / "dynamic_re10k_camera_colmap_aligned_pose.json", camera_colmap_aligned_pose_payload)
    dump_json(metric_dir / "dynamic_re10k_depth_colmap_sparse_sequence_scale_shift_unposed.json", depth_sequence_unposed_payload)
    dump_json(metric_dir / "dynamic_re10k_depth_colmap_sparse_sequence_scale_shift_posed.json", depth_sequence_posed_payload)
    dump_json(metric_dir / "dynamic_re10k_depth_colmap_sparse_per_frame_scale_shift_unposed.json", depth_per_frame_unposed_payload)
    dump_json(metric_dir / "dynamic_re10k_depth_colmap_sparse_per_frame_scale_shift_posed.json", depth_per_frame_posed_payload)
    dump_json(metric_dir / "dynamic_re10k_recon_colmap_sparse_static_unposed.json", recon_unposed_payload)
    dump_json(metric_dir / "dynamic_re10k_recon_colmap_sparse_static_posed.json", recon_posed_payload)

    return {
        "camera_metadata_pose": camera_metadata_pose_payload,
        "camera_metadata_aligned_pose": camera_metadata_aligned_pose_payload,
        "camera_colmap_pose": camera_colmap_pose_payload,
        "camera_colmap_aligned_pose": camera_colmap_aligned_pose_payload,
        "depth_colmap_sparse_sequence_scale_shift_unposed": depth_sequence_unposed_payload,
        "depth_colmap_sparse_sequence_scale_shift_posed": depth_sequence_posed_payload,
        "depth_colmap_sparse_per_frame_scale_shift_unposed": depth_per_frame_unposed_payload,
        "depth_colmap_sparse_per_frame_scale_shift_posed": depth_per_frame_posed_payload,
        "recon_colmap_sparse_static_unposed": recon_unposed_payload,
        "recon_colmap_sparse_static_posed": recon_posed_payload,
    }


def evaluate_dynamic_re10k_raw_camera_diagnostics_work_dir(
    work_dir: str | Path,
    scenes: Sequence[str],
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    raw_camera_pose_unposed_per_scene: dict[str, dict[str, float]] = {}
    raw_camera_pose_posed_per_scene: dict[str, dict[str, float]] = {}
    raw_camera_scale_unposed_per_scene: dict[str, dict[str, float]] = {}
    raw_camera_scale_posed_per_scene: dict[str, dict[str, float]] = {}
    raw_camera_skips: dict[str, dict[str, Any]] = {}

    for scene in scenes:
        paths = _scene_paths(work_dir, scene)
        if not paths["unposed_raw"].exists() or not paths["posed_raw"].exists():
            continue
        scene_data = _load_saved_gt_meta(paths["gt_meta"])
        if _is_placeholder_metadata_pose_sequence(scene_data.extrinsics):
            raw_camera_skips[scene] = {
                "enabled": False,
                "reason": "metadata_pose_placeholder",
                "num_frames": int(len(scene_data.frame_indices)),
            }
            continue
        raw_camera_pose_unposed_per_scene[scene] = compute_pose_with_gt(
            result_path=paths["unposed_raw"],
            gt_extrinsics=scene_data.extrinsics,
        )
        raw_camera_pose_posed_per_scene[scene] = compute_pose_with_gt(
            result_path=paths["posed_raw"],
            gt_extrinsics=scene_data.extrinsics,
        )
        with np.load(paths["unposed_raw"], allow_pickle=True) as raw_npz:
            raw_camera_scale_unposed_per_scene[scene] = compute_raw_camera_scale_metrics(
                np.asarray(raw_npz["extrinsics"], dtype=np.float32),
                scene_data.extrinsics,
            )
        with np.load(paths["posed_raw"], allow_pickle=True) as raw_npz:
            raw_camera_scale_posed_per_scene[scene] = compute_raw_camera_scale_metrics(
                np.asarray(raw_npz["extrinsics"], dtype=np.float32),
                scene_data.extrinsics,
            )

    metric_dir = work_dir / "metric_results"
    raw_camera_pose_unposed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=raw_camera_pose_unposed_per_scene,
        skipped_scenes=raw_camera_skips,
    )
    raw_camera_pose_posed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=raw_camera_pose_posed_per_scene,
        skipped_scenes=raw_camera_skips,
    )
    raw_camera_scale_unposed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=raw_camera_scale_unposed_per_scene,
        skipped_scenes=raw_camera_skips,
    )
    raw_camera_scale_posed_payload = _build_skip_payload(
        scenes=scenes,
        per_scene=raw_camera_scale_posed_per_scene,
        skipped_scenes=raw_camera_skips,
    )

    dump_json(metric_dir / "dynamic_re10k_raw_camera_pose_unposed.json", raw_camera_pose_unposed_payload)
    dump_json(metric_dir / "dynamic_re10k_raw_camera_pose_posed.json", raw_camera_pose_posed_payload)
    dump_json(metric_dir / "dynamic_re10k_raw_camera_scale_unposed.json", raw_camera_scale_unposed_payload)
    dump_json(metric_dir / "dynamic_re10k_raw_camera_scale_posed.json", raw_camera_scale_posed_payload)
    return {
        "raw_camera_pose_unposed": raw_camera_pose_unposed_payload,
        "raw_camera_pose_posed": raw_camera_pose_posed_payload,
        "raw_camera_scale_unposed": raw_camera_scale_unposed_payload,
        "raw_camera_scale_posed": raw_camera_scale_posed_payload,
    }


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    scenes: Sequence[str],
) -> dict[str, Any]:
    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)
    effective_args = SimpleNamespace(**vars(args))
    selected_scenes = list(scenes)
    eval_only = bool(getattr(effective_args, "eval_only", False))
    metadata_pose_placeholder_scenes: list[str] = []

    if eval_only:
        logger.info("Reusing existing predictions for {} on {} scenes", model_kind, len(selected_scenes))
        _assert_existing_scene_exports(output_root, selected_scenes)
        for scene in selected_scenes:
            scene_data = _load_saved_gt_meta(_scene_paths(output_root, scene)["gt_meta"])
            if _is_placeholder_metadata_pose_sequence(scene_data.extrinsics):
                metadata_pose_placeholder_scenes.append(scene)
    else:
        if model_kind == "da3_anyview_base":
            runner = load_anyview_api(effective_args)

            def run_scene(scene_data: DynamicRE10KSceneData):
                metadata_pose_is_placeholder = _is_placeholder_metadata_pose_sequence(scene_data.extrinsics)
                unposed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=effective_args.input_resolution,
                    ref_view_strategy=effective_args.da3_ref_view_strategy,
                )
                if metadata_pose_is_placeholder:
                    logger.warning(
                        "Dynamic-RE10K scene {} uses placeholder metadata cameras; posed anyview rerun is disabled.",
                        scene_data.scene_name,
                    )
                    return unposed_prediction, clone_prediction_dict(unposed_prediction)
                posed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=effective_args.input_resolution,
                    ref_view_strategy=effective_args.da3_ref_view_strategy,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                )
                return unposed_prediction, posed_prediction

        else:
            effective_args.da3_model_path = getattr(effective_args, "da3_model_path", DEFAULT_DA3_STREAMING_MODEL_PATH)
            runner = load_stitched_runner(
                effective_args,
                da3_process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
            )

            def run_scene(scene_data: DynamicRE10KSceneData):
                metadata_pose_is_placeholder = _is_placeholder_metadata_pose_sequence(scene_data.extrinsics)
                unposed_prediction, _ = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=effective_args,
                    process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
                )
                if metadata_pose_is_placeholder:
                    logger.warning(
                        "Dynamic-RE10K scene {} uses placeholder metadata cameras; posed stitched rerun is disabled.",
                        scene_data.scene_name,
                    )
                    posed_prediction = clone_prediction_dict(unposed_prediction)
                elif getattr(effective_args, "stitched_da3_posed_rerun_with_gt_camera", True):
                    posed_prediction, _ = run_da3_stitched_scene(
                        runner,
                        scene_data=scene_data,
                        args=effective_args,
                        extrinsics=scene_data.extrinsics,
                        intrinsics=scene_data.intrinsics,
                        process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
                    )
                else:
                    posed_prediction = clone_prediction_dict(unposed_prediction)
                return unposed_prediction, posed_prediction

        for scene in selected_scenes:
            scene_data = load_dynamic_re10k_scene_data(
                effective_args.dynamic_re10k_root,
                scene,
                max_frames=effective_args.max_frames,
            )
            if _is_placeholder_metadata_pose_sequence(scene_data.extrinsics):
                metadata_pose_placeholder_scenes.append(scene)
            logger.info("Evaluating {} scene {} with {} frames", model_kind, scene, len(scene_data.image_files))
            unposed_prediction, posed_prediction = run_scene(scene_data)
            write_dynamic_re10k_scene_exports(
                work_dir=output_root,
                scene_name=scene,
                unposed_prediction=unposed_prediction,
                posed_prediction=posed_prediction,
                scene_data=scene_data,
            )

    metrics = evaluate_dynamic_re10k_work_dir(
        effective_args.dynamic_re10k_root,
        output_root,
        selected_scenes,
        min_colmap_registered_frames=int(effective_args.dynamic_re10k_min_colmap_registered_frames),
        min_filtered_points=int(effective_args.dynamic_re10k_min_filtered_points),
        min_valid_sparse_samples=int(effective_args.dynamic_re10k_min_valid_sparse_samples),
        colmap_point_error_max=float(effective_args.dynamic_re10k_colmap_point_error_max),
        colmap_point_track_min=int(effective_args.dynamic_re10k_colmap_point_track_min),
        recon_aggressive_alignment=bool(effective_args.dynamic_re10k_recon_aggressive_alignment),
    )
    metrics.update(evaluate_dynamic_re10k_raw_camera_diagnostics_work_dir(output_root, selected_scenes))
    dump_json(output_root / "metric_results" / "dynamic_re10k_summary.json", metrics)
    dump_json(
        output_root / "run_meta.json",
        {
            "model_kind": model_kind,
            "dataset_name": "dynamic_re10k",
            "dynamic_re10k_root": str(Path(effective_args.dynamic_re10k_root).resolve()),
            "input_resolution": int(effective_args.input_resolution),
            "max_frames": int(effective_args.max_frames),
            "da3_ref_view_strategy": str(effective_args.da3_ref_view_strategy),
            "da3_process_res_method": str(getattr(effective_args, "da3_process_res_method", "upper_bound_resize")),
            "stitched_da3_sky_postprocess": bool(getattr(effective_args, "stitched_da3_sky_postprocess", True)),
            "stitched_da3_posed_rerun_with_gt_camera": bool(
                getattr(effective_args, "stitched_da3_posed_rerun_with_gt_camera", True)
            ),
            "stitched_da3_camera_param_source": str(
                getattr(effective_args, "stitched_da3_camera_param_source", "ray_map")
            ),
            "dynamic_re10k_recon_aggressive_alignment": bool(
                effective_args.dynamic_re10k_recon_aggressive_alignment
            ),
            "eval_only": bool(eval_only),
            "reuse_predictions": bool(eval_only),
            "dynamic_re10k_colmap_point_error_max": float(effective_args.dynamic_re10k_colmap_point_error_max),
            "dynamic_re10k_colmap_point_track_min": int(effective_args.dynamic_re10k_colmap_point_track_min),
            "dynamic_re10k_min_colmap_registered_frames": int(
                effective_args.dynamic_re10k_min_colmap_registered_frames
            ),
            "dynamic_re10k_min_filtered_points": int(effective_args.dynamic_re10k_min_filtered_points),
            "dynamic_re10k_min_valid_sparse_samples": int(
                effective_args.dynamic_re10k_min_valid_sparse_samples
            ),
            "dynamic_re10k_metadata_pose_is_placeholder": bool(selected_scenes)
            and len(metadata_pose_placeholder_scenes) == len(selected_scenes),
            "dynamic_re10k_metadata_pose_placeholder_scenes": sorted(set(metadata_pose_placeholder_scenes)),
            "scenes": selected_scenes,
            "device": str(effective_args.device),
            "ray_pose_only": True,
        },
    )
    return metrics


def main(
argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    available_scenes = list_dynamic_re10k_scenes(args.dynamic_re10k_root)
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [scene for scene in available_scenes if scene in wanted]
    else:
        scenes = available_scenes
    if not scenes:
        raise ValueError(f"No Dynamic-RE10K scenes selected under {args.dynamic_re10k_root}")

    model_kinds = ["da3_anyview_base", "da3_stitched"] if args.model_kind == "both" else [args.model_kind]
    for model_kind in model_kinds:
        logger.info("Running Dynamic-RE10K benchmark for {} on {} scenes", model_kind, len(scenes))
        evaluate_model_kind(args, model_kind=model_kind, scenes=scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
