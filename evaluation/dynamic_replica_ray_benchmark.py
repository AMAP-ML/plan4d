from __future__ import annotations

import argparse
import gc
import importlib
import gzip
import json
import sys
from collections import defaultdict
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

from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
    sample_points_from_mesh,
    write_point_cloud_with_cameras,
)
from depth_anything_3.utils.pose_align import align_poses_umeyama
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


PYTORCH3D_TO_OPENCV = np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))

DEFAULT_INPUT_RESOLUTION = 504
DEFAULT_CLIP_LENGTH = 81
DEFAULT_SPLIT = "test"
DEFAULT_LAZY_MODE = False
DEFAULT_VOXEL_LENGTH = 4.0 / 512.0
DEFAULT_SDF_TRUNC = 0.04
DEFAULT_MAX_DEPTH = 10.0
DEFAULT_SAMPLING_NUMBER = 1_000_000
DEFAULT_EVAL_THRESHOLD = 0.05
DEFAULT_DOWN_SAMPLE = 4.0 / 512.0
DEFAULT_DEPTH_ALIGNMENT_STRATEGY = "median"
DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE = 8
DEPTH_ALIGNMENT_STRATEGIES = ("pose", "median", "none")
DEFAULT_SCENE_FLOW_EVAL = False
DEFAULT_SCENE_FLOW_TARGET_SPACE = "relative"
DEFAULT_SCENE_FLOW_EVAL_SPACE = "metric"
DEFAULT_SCENE_FLOW_THRESHOLD = 0.05
DEFAULT_DA3_SCENE_FLOW_PRESET = "da3-giant-sceneflow"

_ANNOTATION_CACHE: dict[tuple[Path, str], dict[tuple[str, str], dict[int, Mapping[str, Any]]]] = {}
_SCENE_FLOW_ADAPTER_CACHE: dict[tuple[Path, str, int, str, str], Any] = {}
_DA3_DYNAMIC_REPLICA_SCENE_FLOW_ADAPTER_CLASS: Any | None = None


@dataclass(frozen=True)
class DynamicReplicaClip:
    split: str
    scene_name: str
    camera_name: str
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
            "camera_name": self.camera_name,
            "frame_indices": [int(frame_id) for frame_id in self.frame_indices],
            "clip_uid": self.clip_uid,
            "clip_dirname": self.clip_dirname,
            "num_frames": self.num_frames,
        }


def default_dynamic_replica_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "dynamic_stereo" / "dynamic_replica_data").resolve())


def default_da3_base_model_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())


def default_da3_scene_flow_head_checkpoint_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "scene_flow_head.bin").resolve())


def _to_float_dict(payload: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def mean_of_dicts(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    values = [dict(item) for item in values if item]
    if not values:
        return {}
    keys = sorted({key for item in values for key in item.keys()})
    return {
        key: float(np.mean([float(item[key]) for item in values if key in item]))
        for key in keys
    }


def clone_prediction_dict(prediction: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, dict):
            cloned[key] = {
                inner_key: inner_value.copy() if isinstance(inner_value, np.ndarray) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            cloned[key] = value
    return cloned


def _as_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")


def _load_gzip_json_list(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise TypeError(f"Expected list payload in {path}, got {type(payload).__name__}")
    return payload


def _load_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_float16_png(path: str | Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(f"Failed to read float16 PNG: {path}")
    if encoded.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG for {path}, got {encoded.dtype}")
    return encoded.view(np.float16).astype(np.float32)


def dynamic_replica_viewpoint_to_intrinsics(
    viewpoint: Mapping[str, Any],
    image_height: int,
    image_width: int,
) -> np.ndarray:
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


def chunk_frame_indices(frame_indices: Sequence[int], clip_length: int) -> list[tuple[int, ...]]:
    if clip_length <= 0:
        raise ValueError(f"clip_length must be positive, got {clip_length}")
    ordered = tuple(int(frame_id) for frame_id in frame_indices)
    if not ordered:
        return []
    return [
        ordered[start_idx : start_idx + clip_length]
        for start_idx in range(0, len(ordered), clip_length)
    ]


def load_dynamic_replica_annotation_index(
    dataset_root: str | Path,
    split: str,
) -> dict[tuple[str, str], dict[int, Mapping[str, Any]]]:
    root = Path(dataset_root).expanduser().resolve()
    cache_key = (root, str(split))
    if cache_key in _ANNOTATION_CACHE:
        return _ANNOTATION_CACHE[cache_key]

    annotations_path = root / str(split) / f"frame_annotations_{split}.jgz"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Dynamic Replica annotations not found: {annotations_path}")

    grouped: dict[tuple[str, str], dict[int, Mapping[str, Any]]] = {}
    for annotation in _load_gzip_json_list(annotations_path):
        image_path = annotation.get("image", {}).get("path")
        if not image_path:
            continue
        scene_name = Path(str(image_path)).parts[0]
        camera_name = str(annotation.get("camera_name", "left"))
        frame_number = int(annotation["frame_number"])
        grouped.setdefault((scene_name, camera_name), {})[frame_number] = annotation

    _ANNOTATION_CACHE[cache_key] = grouped
    return grouped


def build_dynamic_replica_clips(
    dataset_root: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
    clip_length: int = DEFAULT_CLIP_LENGTH,
    scenes: Sequence[str] | None = None,
    camera_name: str = "left",
) -> list[DynamicReplicaClip]:
    annotation_index = load_dynamic_replica_annotation_index(dataset_root, split)
    available_scene_names = sorted(
        scene_name
        for scene_name, camera in annotation_index.keys()
        if camera == camera_name
    )
    requested_scenes = set(str(scene_name) for scene_name in scenes) if scenes else None
    if requested_scenes is not None:
        missing = sorted(scene_name for scene_name in requested_scenes if scene_name not in set(available_scene_names))
        if missing:
            raise ValueError(f"Requested scenes are missing from split={split}: {missing}")
        scene_names = [scene_name for scene_name in available_scene_names if scene_name in requested_scenes]
    else:
        scene_names = available_scene_names

    clips: list[DynamicReplicaClip] = []
    for scene_name in scene_names:
        scene_annotations = annotation_index[(scene_name, camera_name)]
        frame_indices = tuple(sorted(int(frame_id) for frame_id in scene_annotations.keys()))
        for clip_frames in chunk_frame_indices(frame_indices, clip_length):
            clips.append(
                DynamicReplicaClip(
                    split=str(split),
                    scene_name=scene_name,
                    camera_name=camera_name,
                    frame_indices=clip_frames,
                )
            )
    return clips


def select_dynamic_replica_clips(
    clips: Sequence[DynamicReplicaClip],
    *,
    lazy_mode: bool = DEFAULT_LAZY_MODE,
    max_clips: int = -1,
) -> list[DynamicReplicaClip]:
    selected = list(clips)
    if lazy_mode:
        first_clip_by_scene: dict[str, DynamicReplicaClip] = {}
        for clip in selected:
            first_clip_by_scene.setdefault(clip.scene_name, clip)
        selected = list(first_clip_by_scene.values())
    if max_clips > 0:
        selected = selected[: int(max_clips)]
    return selected


def load_dynamic_replica_clip_scene_data(
    dataset_root: str | Path,
    clip: DynamicReplicaClip,
):
    root = Path(dataset_root).expanduser().resolve()
    annotation_index = load_dynamic_replica_annotation_index(root, clip.split)
    frame_annotations = annotation_index.get((clip.scene_name, clip.camera_name))
    if not frame_annotations:
        raise ValueError(
            f"Missing annotations for scene={clip.scene_name} camera={clip.camera_name} split={clip.split}"
        )

    image_files: list[str] = []
    depth_files: list[str] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []
    annotations_out: list[Mapping[str, Any]] = []

    split_root = root / clip.split
    for frame_id in clip.frame_indices:
        annotation = frame_annotations.get(int(frame_id))
        if annotation is None:
            raise ValueError(f"Missing frame {frame_id} for clip {clip.clip_uid}")
        image_path = split_root / str(annotation["image"]["path"])
        depth_path = split_root / str(annotation["depth"]["path"])
        image_size = tuple(int(value) for value in annotation["image"]["size"])
        image_height, image_width = image_size

        image_files.append(str(image_path))
        depth_files.append(str(depth_path))
        intrinsics.append(
            dynamic_replica_viewpoint_to_intrinsics(
                annotation["viewpoint"],
                image_height=image_height,
                image_width=image_width,
            )
        )
        extrinsics.append(dynamic_replica_viewpoint_to_extrinsic(annotation["viewpoint"]))
        image_sizes.append((image_height, image_width))
        annotations_out.append(annotation)

    return SimpleNamespace(
        image_files=image_files,
        depth_files=depth_files,
        intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
        extrinsics=np.stack(extrinsics, axis=0).astype(np.float32),
        aux=SimpleNamespace(
            split=clip.split,
            scene_name=clip.scene_name,
            camera_name=clip.camera_name,
            clip_uid=clip.clip_uid,
            frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
            image_sizes=image_sizes,
            annotations=annotations_out,
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Dynamic Replica left-camera 81-frame clip benchmark with reconstruction, "
            "camera pose, and aligned depth metrics for raw DA3 anyview-base and/or PLAN4D DA3 stitched."
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
        "--dynamic_replica_root",
        type=str,
        default=default_dynamic_replica_root(),
        help="Dynamic Replica dataset root directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Dynamic Replica split. Expected to be test for this benchmark.",
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
        help="Optional subset of left-camera scene names.",
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
        help=(
            "When enabled, evaluate only the first left-camera clip per scene. "
            "With the default --clip_length=81 this is the first 81-frame clip."
        ),
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
        "--voxel_length",
        type=float,
        default=DEFAULT_VOXEL_LENGTH,
        help="TSDF voxel length used for GT and predicted fusion.",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=DEFAULT_SDF_TRUNC,
        help="TSDF truncation distance used for GT and predicted fusion.",
    )
    parser.add_argument(
        "--max_depth_value",
        type=float,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum depth used during TSDF integration.",
    )
    parser.add_argument(
        "--sampling_number",
        type=int,
        default=DEFAULT_SAMPLING_NUMBER,
        help="Number of points sampled from fused meshes for reconstruction metrics.",
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
        "--dynamic_replica_depth_alignment_strategy",
        type=str,
        default=DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
        choices=DEPTH_ALIGNMENT_STRATEGIES,
        help=(
            "Depth scale used before Dynamic Replica depth/reconstruction metrics: "
            "pose uses ray-camera trajectory scale, median uses GT-depth median scale, none keeps raw depth."
        ),
    )
    parser.add_argument(
        "--dynamic_replica_depth_alignment_sample_stride",
        type=int,
        default=DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
        help="Pixel stride used when estimating median GT-depth scale.",
    )
    parser.add_argument(
        "--enable_scene_flow_eval",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SCENE_FLOW_EVAL,
        help="Evaluate stitched DA3 scene-flow predictions against Dynamic Replica reconstructed GT flow.",
    )
    parser.add_argument(
        "--scene_flow_target_space",
        type=str,
        default=DEFAULT_SCENE_FLOW_TARGET_SPACE,
        choices=["relative", "metric"],
        help="Space of the saved model scene-flow predictions and Dynamic Replica target flow.",
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
    dataset_root = Path(args.dynamic_replica_root)
    split_path = dataset_root / str(args.split) / f"frame_annotations_{args.split}.jgz"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dynamic Replica root does not exist: {dataset_root}")
    if not split_path.exists():
        raise FileNotFoundError(f"Dynamic Replica annotations do not exist: {split_path}")
    if args.clip_length <= 0:
        raise ValueError("--clip_length must be positive.")
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be positive when provided.")
    if args.max_clips == 0:
        raise ValueError("--max_clips must be positive or -1.")
    if args.model_kind in {"da3_stitched", "both"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for da3_stitched or both.")
    if args.scene_flow_threshold <= 0:
        raise ValueError("--scene_flow_threshold must be positive.")
    if args.dynamic_replica_depth_alignment_sample_stride <= 0:
        raise ValueError("--dynamic_replica_depth_alignment_sample_stride must be positive.")
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


def resolve_runtime_max_frames(
    args: argparse.Namespace,
    clips: Sequence[DynamicReplicaClip],
) -> int:
    if not clips:
        raise ValueError("No Dynamic Replica clips selected for evaluation.")
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


def _write_results_npz(export_dir: Path, prediction: Mapping[str, Any]) -> None:
    result_path = export_dir / "exports" / "mini_npz" / "results.npz"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"depth": np.asarray(prediction["depth"], dtype=np.float32)}
    if prediction.get("conf") is not None:
        save_dict["conf"] = np.asarray(prediction["conf"], dtype=np.float32)
    if prediction.get("extrinsics") is not None:
        save_dict["extrinsics"] = np.asarray(prediction["extrinsics"], dtype=np.float32)
    if prediction.get("intrinsics") is not None:
        save_dict["intrinsics"] = np.asarray(prediction["intrinsics"], dtype=np.float32)
    if prediction.get("scene_flow") is not None:
        save_dict["scene_flow"] = np.asarray(prediction["scene_flow"], dtype=np.float32)
    if prediction.get("scene_flow_cam") is not None:
        save_dict["scene_flow_cam"] = np.asarray(prediction["scene_flow_cam"], dtype=np.float32)
    np.savez_compressed(result_path, **save_dict)


def _write_raw_camera_params_npz(export_dir: Path, prediction: Mapping[str, Any]) -> None:
    raw_camera_params = prediction.get("raw_camera_params")
    if not raw_camera_params:
        return
    result_path = export_dir / "exports" / "raw_camera_params.npz"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in raw_camera_params.items()
        if value is not None
    }
    if save_dict:
        np.savez_compressed(result_path, **save_dict)


def _write_gt_meta(export_dir: Path, clip: DynamicReplicaClip, scene_data) -> None:
    meta_path = export_dir / "exports" / "gt_meta.npz"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        meta_path,
        scene_name=np.array(clip.scene_name, dtype=object),
        clip_uid=np.array(clip.clip_uid, dtype=object),
        camera_name=np.array(clip.camera_name, dtype=object),
        split=np.array(clip.split, dtype=object),
        frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
        extrinsics=np.asarray(scene_data.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(scene_data.intrinsics, dtype=np.float32),
        image_files=np.asarray(scene_data.image_files, dtype=object),
        depth_files=np.asarray(scene_data.depth_files, dtype=object),
    )


def write_dynamic_replica_clip_exports(
    work_dir: str | Path,
    clip: DynamicReplicaClip,
    unposed_prediction: Mapping[str, Any],
    posed_prediction: Mapping[str, Any],
    scene_data,
    *,
    unposed_preprocess_meta: Mapping[str, Any] | None = None,
    posed_preprocess_meta: Mapping[str, Any] | None = None,
) -> None:
    base_dir = Path(work_dir) / "model_results" / "dynamic_replica" / clip.scene_name / clip.clip_dirname
    export_specs = (
        ("unposed", unposed_prediction, unposed_preprocess_meta),
        ("posed", posed_prediction, posed_preprocess_meta),
    )
    for suffix, prediction, preprocess_meta in export_specs:
        export_dir = base_dir / suffix
        _write_results_npz(export_dir, prediction)
        _write_raw_camera_params_npz(export_dir, prediction)
        _write_gt_meta(export_dir, clip, scene_data)
        if preprocess_meta:
            preprocess_path = export_dir / "exports" / "preprocess_meta.npz"
            preprocess_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(preprocess_path, **preprocess_meta)


def _load_saved_gt_meta(result_path: Path) -> SimpleNamespace:
    export_dir = result_path.parent.parent
    gt_meta = np.load(export_dir / "gt_meta.npz", allow_pickle=True)
    return SimpleNamespace(
        scene_name=str(gt_meta["scene_name"].item()) if gt_meta["scene_name"].shape == () else str(gt_meta["scene_name"]),
        clip_uid=str(gt_meta["clip_uid"].item()) if gt_meta["clip_uid"].shape == () else str(gt_meta["clip_uid"]),
        camera_name=str(gt_meta["camera_name"].item()) if gt_meta["camera_name"].shape == () else str(gt_meta["camera_name"]),
        split=str(gt_meta["split"].item()) if gt_meta["split"].shape == () else str(gt_meta["split"]),
        frame_indices=np.asarray(gt_meta["frame_indices"], dtype=np.int64),
        image_files=[str(path) for path in gt_meta["image_files"]],
        depth_files=[str(path) for path in gt_meta["depth_files"]],
        extrinsics=np.asarray(gt_meta["extrinsics"], dtype=np.float32),
        intrinsics=np.asarray(gt_meta["intrinsics"], dtype=np.float32),
    )


def _prediction_paths_for_clip(work_dir: str | Path, clip: DynamicReplicaClip) -> dict[str, Path]:
    base_dir = Path(work_dir) / "model_results" / "dynamic_replica" / clip.scene_name / clip.clip_dirname
    return {
        "unposed_result": base_dir / "unposed" / "exports" / "mini_npz" / "results.npz",
        "posed_result": base_dir / "posed" / "exports" / "mini_npz" / "results.npz",
        "unposed_raw": base_dir / "unposed" / "exports" / "raw_camera_params.npz",
        "posed_raw": base_dir / "posed" / "exports" / "raw_camera_params.npz",
        "gt_fuse": base_dir / "gt_fuse" / "pcd.ply",
        "unposed_fuse": base_dir / "unposed" / "exports" / "fuse" / "pcd.ply",
        "posed_fuse": base_dir / "posed" / "exports" / "fuse" / "pcd.ply",
    }


def _resize_intrinsics_to_original(
    intrinsic: np.ndarray,
    *,
    model_height: int,
    model_width: int,
    original_height: int,
    original_width: int,
) -> np.ndarray:
    resized = np.asarray(intrinsic, dtype=np.float32).copy()
    resized[0, :] *= float(original_width) / float(model_width)
    resized[1, :] *= float(original_height) / float(model_height)
    return resized


def _sanitize_depth(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    sanitized = np.zeros_like(depth, dtype=np.float32)
    sanitized[valid] = depth[valid]
    return sanitized, valid


def _load_gt_depth_and_valid_mask(depth_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    depth = _load_float16_png(depth_path)
    # Dynamic Replica's mask PNG is a sparse foreground/object mask rather than a generic
    # "valid depth" mask, so benchmarking uses finite positive GT depth as the valid region.
    return _sanitize_depth(depth)


def _compute_clip_alignment(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
) -> tuple[float, np.ndarray]:
    _, _, scale, aligned_extrinsics = align_poses_umeyama(
        _as_homogeneous_extrinsics(gt_extrinsics.copy()),
        _as_homogeneous_extrinsics(pred_extrinsics.copy()),
        return_aligned=True,
        ransac=len(gt_extrinsics) >= 10,
        random_state=42,
    )
    return float(scale), np.asarray(aligned_extrinsics, dtype=np.float32)


def _validate_depth_alignment_strategy(strategy: str) -> str:
    strategy = str(strategy)
    if strategy not in DEPTH_ALIGNMENT_STRATEGIES:
        raise ValueError(
            f"Unsupported Dynamic Replica depth alignment strategy: {strategy}. "
            f"Expected one of {DEPTH_ALIGNMENT_STRATEGIES}."
        )
    return strategy


def _compute_median_depth_alignment_scale(
    *,
    scene_data,
    pred_depths: np.ndarray,
    sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
    eps: float = 1e-6,
) -> float:
    if sample_stride <= 0:
        raise ValueError(f"sample_stride must be positive, got {sample_stride}")

    pred_values: list[np.ndarray] = []
    gt_values: list[np.ndarray] = []
    for frame_index, depth_file in enumerate(scene_data.depth_files):
        gt_depth, gt_valid = _load_gt_depth_and_valid_mask(depth_file)
        image_height, image_width = gt_depth.shape
        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        valid = gt_valid & pred_valid
        if sample_stride > 1:
            valid = valid[::sample_stride, ::sample_stride]
            pred_depth = pred_depth[::sample_stride, ::sample_stride]
            gt_depth = gt_depth[::sample_stride, ::sample_stride]
        if valid.any():
            pred_values.append(pred_depth[valid])
            gt_values.append(gt_depth[valid])

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
            sample_stride=sample_stride,
        ),
        float(pose_scale),
    )


def _compute_depth_alignment_scales_for_result(
    *,
    scene_data,
    result_path: Path,
    strategy: str,
    sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> tuple[float, float]:
    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)
    return _compute_depth_alignment_scales(
        scene_data=scene_data,
        pred_depths=pred_depths,
        pred_extrinsics=pred_extrinsics,
        strategy=strategy,
        sample_stride=sample_stride,
    )


def _integrate_depth_sequence_to_pcd(
    *,
    depth_maps: Sequence[np.ndarray],
    image_files: Sequence[str],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
    sampling_number: int,
) -> tuple[o3d.geometry.PointCloud, list[tuple[int, int]]]:
    volume = create_tsdf_volume(
        voxel_length=float(voxel_length),
        sdf_trunc=float(sdf_trunc),
    )
    image_sizes: list[tuple[int, int]] = []
    for depth_map, image_file, intrinsic, extrinsic in zip(depth_maps, image_files, intrinsics, extrinsics):
        color = _load_rgb_image(image_file)
        image_height, image_width = color.shape[:2]
        image_sizes.append((image_height, image_width))

        sanitized_depth, _ = _sanitize_depth(depth_map)
        depth_o3d = o3d.geometry.Image(sanitized_depth.astype(np.float32))
        color_o3d = o3d.geometry.Image(color.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_trunc=float(max_depth),
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            image_width,
            image_height,
            float(intrinsic[0, 0]),
            float(intrinsic[1, 1]),
            float(intrinsic[0, 2]),
            float(intrinsic[1, 2]),
        )
        volume.integrate(rgbd, pinhole, _as_homogeneous_extrinsics(extrinsic))

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = sample_points_from_mesh(mesh, int(sampling_number))
    else:
        pcd = volume.extract_point_cloud()
    return pcd, image_sizes


def _write_pcd_with_cameras(
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
        extrinsics=_as_homogeneous_extrinsics(extrinsics),
        image_sizes=list(image_sizes),
    )


def _build_gt_clip_point_cloud(
    scene_data,
    gt_fuse_path: Path,
    *,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
    sampling_number: int,
    reuse_existing_fuse: bool,
) -> o3d.geometry.PointCloud:
    if reuse_existing_fuse and gt_fuse_path.exists() and gt_fuse_path.stat().st_size > 0:
        return o3d.io.read_point_cloud(str(gt_fuse_path))

    gt_depth_maps = []
    for depth_file in scene_data.depth_files:
        gt_depth, _ = _load_gt_depth_and_valid_mask(depth_file)
        gt_depth_maps.append(gt_depth)

    gt_pcd, image_sizes = _integrate_depth_sequence_to_pcd(
        depth_maps=gt_depth_maps,
        image_files=scene_data.image_files,
        intrinsics=scene_data.intrinsics,
        extrinsics=scene_data.extrinsics,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        max_depth=max_depth,
        sampling_number=sampling_number,
    )
    _write_pcd_with_cameras(
        output_path=gt_fuse_path,
        pcd=gt_pcd,
        intrinsics=scene_data.intrinsics,
        extrinsics=scene_data.extrinsics,
        image_sizes=image_sizes,
    )
    return gt_pcd


def _fuse_prediction_for_mode(
    *,
    scene_data,
    result_path: Path,
    fuse_path: Path,
    mode: str,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
    sampling_number: int,
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
            strategy=depth_alignment_strategy,
            sample_stride=depth_alignment_sample_stride,
        )
    _, aligned_extrinsics = _compute_clip_alignment(pred_extrinsics, scene_data.extrinsics)
    scale = float(depth_alignment_scale)
    model_height, model_width = pred_depths.shape[1:3]

    fused_depths: list[np.ndarray] = []
    fused_intrinsics: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []
    for frame_index, image_file in enumerate(scene_data.image_files):
        image = _load_rgb_image(image_file)
        original_height, original_width = image.shape[:2]
        image_sizes.append((original_height, original_width))

        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        _, gt_valid = _load_gt_depth_and_valid_mask(scene_data.depth_files[frame_index])
        if gt_valid.shape != pred_depth.shape:
            gt_valid = cv2.resize(
                gt_valid.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        pred_depth = pred_depth * float(scale)
        pred_depth[~(pred_valid & gt_valid)] = 0.0
        fused_depths.append(pred_depth)

        if mode == "recon_unposed":
            intrinsic = _resize_intrinsics_to_original(
                pred_intrinsics[frame_index],
                model_height=model_height,
                model_width=model_width,
                original_height=original_height,
                original_width=original_width,
            )
            fused_intrinsics.append(intrinsic.astype(np.float32))
        elif mode == "recon_posed":
            fused_intrinsics.append(np.asarray(scene_data.intrinsics[frame_index], dtype=np.float32))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    fused_extrinsics = aligned_extrinsics if mode == "recon_unposed" else np.asarray(
        scene_data.extrinsics,
        dtype=np.float32,
    )
    pred_pcd, _ = _integrate_depth_sequence_to_pcd(
        depth_maps=fused_depths,
        image_files=scene_data.image_files,
        intrinsics=np.stack(fused_intrinsics, axis=0).astype(np.float32),
        extrinsics=fused_extrinsics,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        max_depth=max_depth,
        sampling_number=sampling_number,
    )
    _write_pcd_with_cameras(
        output_path=fuse_path,
        pcd=pred_pcd,
        intrinsics=np.stack(fused_intrinsics, axis=0).astype(np.float32),
        extrinsics=fused_extrinsics,
        image_sizes=image_sizes,
    )
    return pred_pcd


def _compute_depth_metrics_np(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
) -> dict[str, float]:
    pred = np.clip(np.asarray(pred_values, dtype=np.float64), 1e-6, None)
    gt = np.clip(np.asarray(gt_values, dtype=np.float64), 1e-6, None)
    if pred.size == 0 or gt.size == 0:
        return {
            "depth_abs_rel": 0.0,
            "depth_sq_rel": 0.0,
            "depth_rmse": 0.0,
            "depth_rmse_log": 0.0,
            "depth_a1": 0.0,
            "depth_a2": 0.0,
            "depth_a3": 0.0,
            "valid_pixel_count": 0.0,
        }

    thresh = np.maximum(pred / gt, gt / pred)
    return {
        "depth_abs_rel": float(np.mean(np.abs(pred - gt) / gt)),
        "depth_sq_rel": float(np.mean(((pred - gt) ** 2) / gt)),
        "depth_rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "depth_rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        "depth_a1": float(np.mean(thresh < 1.25)),
        "depth_a2": float(np.mean(thresh < 1.25 ** 2)),
        "depth_a3": float(np.mean(thresh < 1.25 ** 3)),
        "valid_pixel_count": float(pred.size),
    }


def _numpy_from_maybe_tensor(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _scene_flow_to_tchw(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"Expected scene flow with 4 dims, got {array.shape}")
    if array.shape[1] == 3:
        return array
    if array.shape[-1] == 3:
        return np.transpose(array, (0, 3, 1, 2)).copy()
    raise ValueError(f"Expected scene-flow channel dimension of 3, got {array.shape}")


def _deform_mask_to_thw(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Expected deform mask shape [T,H,W], [T,1,H,W], or [T,H,W,1], got {array.shape}")


def _resize_scene_flow_tchw(flow: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.shape[-2:] == target_hw:
        return flow
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    resized = np.empty((flow.shape[0], flow.shape[1], target_height, target_width), dtype=np.float32)
    for frame_idx in range(flow.shape[0]):
        for channel_idx in range(flow.shape[1]):
            resized[frame_idx, channel_idx] = cv2.resize(
                flow[frame_idx, channel_idx],
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
    return resized


def _resize_deform_mask_thw(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape[-2:] == target_hw:
        return mask
    target_height, target_width = (int(target_hw[0]), int(target_hw[1]))
    resized = np.empty((mask.shape[0], target_height, target_width), dtype=np.float32)
    for frame_idx in range(mask.shape[0]):
        resized[frame_idx] = cv2.resize(
            mask[frame_idx],
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return resized


def _resolve_scene_flow_eval_space(target_space: str, eval_space: str) -> str:
    if target_space not in {"relative", "metric"}:
        raise ValueError(f"Unsupported scene_flow target_space: {target_space}")
    if eval_space not in {"auto", "native", "metric"}:
        raise ValueError(f"Unsupported scene_flow eval_space: {eval_space}")
    if eval_space == "auto":
        return "metric" if target_space == "relative" else "native"
    return eval_space


def _convert_scene_flow_to_eval_space(
    pred_flow: np.ndarray,
    target_flow: np.ndarray,
    *,
    scene_flow_scale: float | None,
    target_space: str,
    eval_space: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    resolved_eval_space = _resolve_scene_flow_eval_space(target_space, eval_space)
    if resolved_eval_space == "metric" and target_space == "relative":
        if scene_flow_scale is None:
            raise ValueError("metric scene-flow evaluation for relative predictions requires scene_flow_scale")
        scale = float(scene_flow_scale)
        return pred_flow * scale, target_flow * scale, scale
    return pred_flow, target_flow, 1.0


def compute_dynamic_replica_scene_flow_metrics(
    pred_flow: np.ndarray,
    target_flow: np.ndarray,
    deform_mask: np.ndarray,
    *,
    scene_flow_scale: float | None,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
    eval_space: str = DEFAULT_SCENE_FLOW_EVAL_SPACE,
    threshold: float = DEFAULT_SCENE_FLOW_THRESHOLD,
) -> dict[str, float]:
    pred = _scene_flow_to_tchw(pred_flow)
    target = _scene_flow_to_tchw(target_flow)
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError(f"pred_flow and target_flow temporal/channel mismatch: {pred.shape} vs {target.shape}")
    if pred.shape[-2:] != target.shape[-2:]:
        pred = _resize_scene_flow_tchw(pred, tuple(target.shape[-2:]))

    mask = _deform_mask_to_thw(deform_mask)
    if mask.shape[0] != target.shape[0]:
        raise ValueError(f"deform_mask temporal mismatch: {mask.shape} vs {target.shape}")
    if mask.shape[-2:] != target.shape[-2:]:
        mask = _resize_deform_mask_thw(mask, tuple(target.shape[-2:]))

    pred_eval, target_eval, applied_scale = _convert_scene_flow_to_eval_space(
        pred,
        target,
        scene_flow_scale=scene_flow_scale,
        target_space=target_space,
        eval_space=eval_space,
    )
    valid_mask = mask > 0.5
    finite = np.isfinite(pred_eval).all(axis=1) & np.isfinite(target_eval).all(axis=1)
    valid = valid_mask & finite
    valid_ratio = float(np.mean(valid_mask)) if valid_mask.size else 0.0
    last_frame_zero_epe = (
        float(np.linalg.norm(pred_eval[-1], axis=0).mean()) if pred_eval.shape[0] > 0 else 0.0
    )
    if not valid.any():
        return {
            "scene_flow_loss": 0.0,
            "scene_flow_l1": 0.0,
            "scene_flow_epe": 0.0,
            "scene_flow_apd_0.05": 0.0,
            "scene_flow_valid_ratio": valid_ratio,
            "scene_flow_valid_count": 0.0,
            "scene_flow_last_frame_zero_epe": last_frame_zero_epe,
            "scene_flow_pred_mag_mean": 0.0,
            "scene_flow_gt_mag_mean": 0.0,
            "scene_flow_scale": applied_scale,
        }

    diff = pred_eval - target_eval
    per_pixel_l1 = np.abs(diff).sum(axis=1)
    per_pixel_epe = np.linalg.norm(diff, axis=1)
    pred_mag = np.linalg.norm(pred_eval, axis=1)
    target_mag = np.linalg.norm(target_eval, axis=1)
    valid_epe = per_pixel_epe[valid]
    valid_l1 = per_pixel_l1[valid]
    return {
        "scene_flow_loss": float(np.mean(valid_l1)),
        "scene_flow_l1": float(np.mean(valid_l1)),
        "scene_flow_epe": float(np.mean(valid_epe)),
        "scene_flow_apd_0.05": float(np.mean(valid_epe < float(threshold))),
        "scene_flow_valid_ratio": valid_ratio,
        "scene_flow_valid_count": float(valid_epe.size),
        "scene_flow_last_frame_zero_epe": last_frame_zero_epe,
        "scene_flow_pred_mag_mean": float(np.mean(pred_mag[valid])),
        "scene_flow_gt_mag_mean": float(np.mean(target_mag[valid])),
        "scene_flow_scale": applied_scale,
    }


def _evaluate_depth_for_mode(
    *,
    scene_data,
    result_path: Path,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
    depth_alignment_scale: float | None = None,
    depth_pose_alignment_scale: float | None = None,
) -> dict[str, float]:
    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)

    if depth_alignment_scale is None or depth_pose_alignment_scale is None:
        computed_scale, computed_pose_scale = _compute_depth_alignment_scales(
            scene_data=scene_data,
            pred_depths=pred_depths,
            pred_extrinsics=pred_extrinsics,
            strategy=depth_alignment_strategy,
            sample_stride=depth_alignment_sample_stride,
        )
        if depth_alignment_scale is None:
            depth_alignment_scale = computed_scale
        if depth_pose_alignment_scale is None:
            depth_pose_alignment_scale = computed_pose_scale
    scale = float(depth_alignment_scale)
    pred_values: list[np.ndarray] = []
    gt_values: list[np.ndarray] = []

    for frame_index, depth_file in enumerate(scene_data.depth_files):
        gt_depth, gt_valid = _load_gt_depth_and_valid_mask(depth_file)
        image_height, image_width = gt_depth.shape
        pred_depth = cv2.resize(
            pred_depths[frame_index],
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        pred_depth, pred_valid = _sanitize_depth(pred_depth)
        pred_depth = pred_depth * float(scale)
        valid = gt_valid & pred_valid
        if valid.any():
            pred_values.append(pred_depth[valid])
            gt_values.append(gt_depth[valid])

    if pred_values:
        pred_concat = np.concatenate(pred_values, axis=0)
        gt_concat = np.concatenate(gt_values, axis=0)
    else:
        pred_concat = np.zeros((0,), dtype=np.float32)
        gt_concat = np.zeros((0,), dtype=np.float32)
    metrics = _compute_depth_metrics_np(pred_concat, gt_concat)
    metrics["depth_alignment_scale"] = float(scale)
    metrics["depth_pose_alignment_scale"] = float(depth_pose_alignment_scale)
    metrics["depth_scale_from_pose_alignment"] = float(depth_pose_alignment_scale)
    metrics["num_frames"] = float(len(scene_data.depth_files))
    return metrics


def _aggregate_metric_records(
    per_clip: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    per_scene: dict[str, list[dict[str, float]]] = defaultdict(list)
    for clip_key, metrics in per_clip.items():
        scene_name = clip_key.split("/", 1)[0]
        per_scene[scene_name].append(dict(metrics))

    per_scene_mean = {
        scene_name: mean_of_dicts(metric_list)
        for scene_name, metric_list in sorted(per_scene.items())
    }
    overall_mean = mean_of_dicts(list(per_scene_mean.values()))
    return {
        "per_clip": {key: dict(value) for key, value in per_clip.items()},
        "per_scene": per_scene_mean,
        "mean": overall_mean,
    }


def _evaluate_dynamic_replica_clip_metrics(
    *,
    dataset_root: str | Path,
    work_dir: Path,
    clip: DynamicReplicaClip,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
    sampling_number: int,
    eval_threshold: float,
    down_sample: float,
    reuse_existing_fuse: bool,
    depth_alignment_strategy: str = DEFAULT_DEPTH_ALIGNMENT_STRATEGY,
    depth_alignment_sample_stride: int = DEFAULT_DEPTH_ALIGNMENT_SAMPLE_STRIDE,
) -> dict[str, dict[str, float]]:
    clip_paths = _prediction_paths_for_clip(work_dir, clip)
    scene_data = load_dynamic_replica_clip_scene_data(dataset_root, clip)

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
        strategy=depth_alignment_strategy,
        sample_stride=depth_alignment_sample_stride,
    )
    posed_depth_scale, posed_pose_scale = _compute_depth_alignment_scales_for_result(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
        strategy=depth_alignment_strategy,
        sample_stride=depth_alignment_sample_stride,
    )
    gt_pcd = _build_gt_clip_point_cloud(
        scene_data,
        clip_paths["gt_fuse"],
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        max_depth=max_depth,
        sampling_number=sampling_number,
        reuse_existing_fuse=reuse_existing_fuse,
    )
    pred_unposed_pcd = _fuse_prediction_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["unposed_result"],
        fuse_path=clip_paths["unposed_fuse"],
        mode="recon_unposed",
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        max_depth=max_depth,
        sampling_number=sampling_number,
        reuse_existing_fuse=reuse_existing_fuse,
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=unposed_depth_scale,
    )
    pred_posed_pcd = _fuse_prediction_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
        fuse_path=clip_paths["posed_fuse"],
        mode="recon_posed",
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        max_depth=max_depth,
        sampling_number=sampling_number,
        reuse_existing_fuse=reuse_existing_fuse,
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=posed_depth_scale,
    )

    recon_unposed_metrics = _to_float_dict(
        evaluate_3d_reconstruction(
            pred_unposed_pcd,
            gt_pcd,
            threshold=float(eval_threshold),
            down_sample=float(down_sample),
        )
    )
    recon_posed_metrics = _to_float_dict(
        evaluate_3d_reconstruction(
            pred_posed_pcd,
            gt_pcd,
            threshold=float(eval_threshold),
            down_sample=float(down_sample),
        )
    )
    recon_unposed_metrics["depth_alignment_scale"] = float(unposed_depth_scale)
    recon_unposed_metrics["depth_pose_alignment_scale"] = float(unposed_pose_scale)
    recon_posed_metrics["depth_alignment_scale"] = float(posed_depth_scale)
    recon_posed_metrics["depth_pose_alignment_scale"] = float(posed_pose_scale)
    depth_unposed_metrics = _evaluate_depth_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["unposed_result"],
        depth_alignment_strategy=depth_alignment_strategy,
        depth_alignment_sample_stride=depth_alignment_sample_stride,
        depth_alignment_scale=unposed_depth_scale,
        depth_pose_alignment_scale=unposed_pose_scale,
    )
    depth_posed_metrics = _evaluate_depth_for_mode(
        scene_data=scene_data,
        result_path=clip_paths["posed_result"],
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


def _load_dynamic_replica_scene_flow_adapter_class():
    global _DA3_DYNAMIC_REPLICA_SCENE_FLOW_ADAPTER_CLASS
    if _DA3_DYNAMIC_REPLICA_SCENE_FLOW_ADAPTER_CLASS is not None:
        return _DA3_DYNAMIC_REPLICA_SCENE_FLOW_ADAPTER_CLASS

    saved_training_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "training" or name.startswith("training.")
    }
    for name in saved_training_modules:
        sys.modules.pop(name, None)

    da3_root_str = str(DA3_ROOT)
    inserted_da3_root = False
    if da3_root_str not in sys.path:
        sys.path.insert(0, da3_root_str)
        inserted_da3_root = True
    else:
        sys.path.remove(da3_root_str)
        sys.path.insert(0, da3_root_str)

    try:
        module = importlib.import_module("training.datasets.dynamic_replica_scene_flow")
        adapter_class = module.DynamicReplicaSceneFlowAdapter
    finally:
        for name in list(sys.modules.keys()):
            if name == "training" or name.startswith("training."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_training_modules)
        if inserted_da3_root:
            try:
                sys.path.remove(da3_root_str)
            except ValueError:
                pass

    _DA3_DYNAMIC_REPLICA_SCENE_FLOW_ADAPTER_CLASS = adapter_class
    return adapter_class


def _get_dynamic_replica_scene_flow_adapter(
    dataset_root: str | Path,
    *,
    split: str,
    clip_length: int,
    target_space: str,
    camera_name: str,
):
    DynamicReplicaSceneFlowAdapter = _load_dynamic_replica_scene_flow_adapter_class()

    root = Path(dataset_root).expanduser().resolve()
    key = (root, str(split), int(clip_length), str(target_space), str(camera_name))
    if key not in _SCENE_FLOW_ADAPTER_CACHE:
        _SCENE_FLOW_ADAPTER_CACHE[key] = DynamicReplicaSceneFlowAdapter(
            data_root=root,
            split=str(split),
            chunk_size=int(clip_length),
            overlap=0,
            target_height=None,
            target_width=None,
            process_res=None,
            process_res_method=None,
            pad_to_chunk_size=False,
            flow_target_space=str(target_space),
            camera_names=[str(camera_name)],
        )
    return _SCENE_FLOW_ADAPTER_CACHE[key]


def load_dynamic_replica_clip_scene_flow_gt(
    dataset_root: str | Path,
    clip: DynamicReplicaClip,
    *,
    clip_length: int,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
) -> dict[str, Any]:
    adapter = _get_dynamic_replica_scene_flow_adapter(
        dataset_root,
        split=clip.split,
        clip_length=int(clip_length),
        target_space=str(target_space),
        camera_name=clip.camera_name,
    )
    frame_indices = tuple(int(frame_id) for frame_id in clip.frame_indices)
    sample_index = None
    for idx, sample in enumerate(adapter.samples):
        if (
            sample.scene_name == clip.scene_name
            and sample.camera_name == clip.camera_name
            and tuple(int(frame_id) for frame_id in sample.frame_indices) == frame_indices
        ):
            sample_index = idx
            break
    if sample_index is None:
        raise ValueError(f"Dynamic Replica scene-flow GT sample not found for clip {clip.clip_uid}")

    sample_payload = adapter[sample_index]
    scene_flow = sample_payload.get("eval_scene_flow", sample_payload.get("scene_flow"))
    deform_mask = sample_payload.get("eval_deform_mask", sample_payload.get("deform_mask"))
    if scene_flow is None or deform_mask is None:
        raise ValueError(f"Dynamic Replica scene-flow GT payload is incomplete for clip {clip.clip_uid}")
    scene_flow_scale = sample_payload.get("scene_flow_scale", 1.0)
    if torch.is_tensor(scene_flow_scale):
        scene_flow_scale = float(scene_flow_scale.detach().cpu().item())
    return {
        "scene_flow": _numpy_from_maybe_tensor(scene_flow).astype(np.float32),
        "deform_mask": _numpy_from_maybe_tensor(deform_mask).astype(np.float32),
        "scene_flow_scale": float(scene_flow_scale),
    }


def evaluate_dynamic_replica_scene_flow_work_dir(
    *,
    dataset_root: str | Path,
    work_dir: str | Path,
    clips: Sequence[DynamicReplicaClip],
    clip_length: int,
    target_space: str = DEFAULT_SCENE_FLOW_TARGET_SPACE,
    eval_space: str = DEFAULT_SCENE_FLOW_EVAL_SPACE,
    threshold: float = DEFAULT_SCENE_FLOW_THRESHOLD,
    scene_flow_gt_loader: Any | None = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    gt_loader = scene_flow_gt_loader or load_dynamic_replica_clip_scene_flow_gt
    per_mode_clip_metrics: dict[str, dict[str, dict[str, float]]] = {
        "unposed": {},
        "posed": {},
    }

    for clip in clips:
        clip_paths = _prediction_paths_for_clip(work_dir, clip)
        gt_payload = gt_loader(
            dataset_root,
            clip,
            clip_length=int(clip_length),
            target_space=str(target_space),
        )
        for mode in ("unposed", "posed"):
            result_path = clip_paths[f"{mode}_result"]
            if not result_path.exists():
                logger.warning("Skipping scene-flow {} metrics for missing export {}", mode, result_path)
                continue
            with np.load(result_path, allow_pickle=True) as pred_npz:
                if "scene_flow" not in pred_npz:
                    logger.warning("Skipping scene-flow {} metrics for {} because scene_flow is missing.", mode, clip.clip_uid)
                    continue
                pred_scene_flow = np.asarray(pred_npz["scene_flow"], dtype=np.float32)
            metrics = compute_dynamic_replica_scene_flow_metrics(
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
            "Rerun inference with --da3_enable_scene_flow_head after this fix; "
            "older exports only contain depth/camera arrays."
        )

    metric_dir = work_dir / "metric_results"
    unposed_payload = _aggregate_metric_records(per_mode_clip_metrics["unposed"])
    posed_payload = _aggregate_metric_records(per_mode_clip_metrics["posed"])
    dump_json(metric_dir / "dynamic_replica_scene_flow_unposed.json", unposed_payload)
    dump_json(metric_dir / "dynamic_replica_scene_flow_posed.json", posed_payload)
    return {
        "scene_flow_unposed": unposed_payload,
        "scene_flow_posed": posed_payload,
    }


def evaluate_dynamic_replica_work_dir(
    dataset_root: str | Path,
    work_dir: str | Path,
    clips: Sequence[DynamicReplicaClip],
    *,
    voxel_length: float,
    sdf_trunc: float,
    max_depth: float,
    sampling_number: int,
    eval_threshold: float,
    down_sample: float,
    reuse_existing_fuse: bool,
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
        logger.info("Evaluating Dynamic Replica clip {}", clip.clip_uid)
        clip_metrics = _evaluate_dynamic_replica_clip_metrics(
            dataset_root=dataset_root,
            work_dir=work_dir,
            clip=clip,
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            max_depth=max_depth,
            sampling_number=sampling_number,
            eval_threshold=eval_threshold,
            down_sample=down_sample,
            reuse_existing_fuse=reuse_existing_fuse,
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

    dump_json(metric_dir / "dynamic_replica_pose.json", pose_payload)
    dump_json(metric_dir / "dynamic_replica_aligned_pose.json", aligned_pose_payload)
    dump_json(metric_dir / "dynamic_replica_recon_unposed.json", recon_unposed_payload)
    dump_json(metric_dir / "dynamic_replica_recon_posed.json", recon_posed_payload)
    dump_json(metric_dir / "dynamic_replica_depth_unposed.json", depth_unposed_payload)
    dump_json(metric_dir / "dynamic_replica_depth_posed.json", depth_posed_payload)

    return {
        "pose": pose_payload,
        "aligned_pose": aligned_pose_payload,
        "recon_unposed": recon_unposed_payload,
        "recon_posed": recon_posed_payload,
        "depth_unposed": depth_unposed_payload,
        "depth_posed": depth_posed_payload,
    }


def evaluate_dynamic_replica_raw_camera_diagnostics(
    work_dir: str | Path,
    clips: Sequence[DynamicReplicaClip],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    raw_camera_pose_unposed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_pose_posed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_scale_unposed_per_clip: dict[str, dict[str, float]] = {}
    raw_camera_scale_posed_per_clip: dict[str, dict[str, float]] = {}

    for clip in clips:
        clip_paths = _prediction_paths_for_clip(work_dir, clip)
        if not clip_paths["unposed_raw"].exists() or not clip_paths["posed_raw"].exists():
            logger.warning("Skipping raw camera diagnostics for clip {} because raw_camera_params are missing.", clip.clip_uid)
            continue

        scene_data = load_dynamic_replica_clip_scene_data(dataset_root, clip)
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

    dump_json(metric_dir / "dynamic_replica_raw_camera_pose_unposed.json", pose_unposed_payload)
    dump_json(metric_dir / "dynamic_replica_raw_camera_pose_posed.json", pose_posed_payload)
    dump_json(metric_dir / "dynamic_replica_raw_camera_scale_unposed.json", scale_unposed_payload)
    dump_json(metric_dir / "dynamic_replica_raw_camera_scale_posed.json", scale_posed_payload)

    return {
        "raw_camera_pose_unposed": pose_unposed_payload,
        "raw_camera_pose_posed": pose_posed_payload,
        "raw_camera_scale_unposed": scale_unposed_payload,
        "raw_camera_scale_posed": scale_posed_payload,
    }


def _build_run_metadata(
    args: argparse.Namespace,
    *,
    model_kind: str,
    clips: Sequence[DynamicReplicaClip],
) -> dict[str, Any]:
    return {
        "model_kind": model_kind,
        "dynamic_replica_root": str(Path(args.dynamic_replica_root).expanduser().resolve()),
        "split": str(args.split),
        "input_resolution": int(args.input_resolution),
        "clip_length": int(args.clip_length),
        "max_frames": int(args.max_frames),
        "lazy_mode": bool(args.lazy_mode),
        "max_clips": int(args.max_clips),
        "clip_count": len(clips),
        "scenes": sorted({clip.scene_name for clip in clips}),
        "stitched_da3_camera_param_source": str(args.stitched_da3_camera_param_source),
        "stitched_da3_sky_postprocess": bool(args.stitched_da3_sky_postprocess),
        "stitched_da3_posed_rerun_with_gt_camera": bool(args.stitched_da3_posed_rerun_with_gt_camera),
        "da3_process_res_method": str(args.da3_process_res_method),
        "da3_enable_scene_flow_head": bool(args.da3_enable_scene_flow_head),
        "da3_scene_flow_head_checkpoint_path": str(args.da3_scene_flow_head_checkpoint_path),
        "da3_scene_flow_preset": str(args.da3_scene_flow_preset),
        "enable_scene_flow_eval": bool(args.enable_scene_flow_eval),
        "scene_flow_target_space": str(args.scene_flow_target_space),
        "scene_flow_eval_space": str(args.scene_flow_eval_space),
        "scene_flow_threshold": float(args.scene_flow_threshold),
        "dynamic_replica_depth_alignment_strategy": str(args.dynamic_replica_depth_alignment_strategy),
        "dynamic_replica_depth_alignment_sample_stride": int(args.dynamic_replica_depth_alignment_sample_stride),
        "voxel_length": float(args.voxel_length),
        "sdf_trunc": float(args.sdf_trunc),
        "max_depth_value": float(args.max_depth_value),
        "sampling_number": int(args.sampling_number),
        "eval_threshold": float(args.eval_threshold),
        "down_sample": float(args.down_sample),
        "clips": [clip.to_dict() for clip in clips],
    }


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    clips: Sequence[DynamicReplicaClip],
) -> dict[str, Any]:
    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)
    run_meta = _build_run_metadata(args, model_kind=model_kind, clips=clips)
    dump_json(output_root / "run_meta.json", run_meta)

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
            logger.info("Running Dynamic Replica inference for clip {}", clip.clip_uid)
            scene_data = load_dynamic_replica_clip_scene_data(args.dynamic_replica_root, clip)
            if model_kind == "da3_anyview_base":
                unposed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=args.input_resolution,
                    ref_view_strategy=args.da3_ref_view_strategy,
                )
                posed_prediction = run_da3_anyview_scene(
                    runner,
                    scene_data=scene_data,
                    process_res=args.input_resolution,
                    ref_view_strategy=args.da3_ref_view_strategy,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                )
                unposed_preprocess_meta = None
                posed_preprocess_meta = None
            elif model_kind == "da3_stitched":
                unposed_prediction, unposed_preprocess_meta = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=args,
                    process_res_method=getattr(args, "da3_process_res_method", "upper_bound_resize"),
                )
                if getattr(args, "stitched_da3_posed_rerun_with_gt_camera", True):
                    posed_prediction, posed_preprocess_meta = run_da3_stitched_scene(
                        runner,
                        scene_data=scene_data,
                        args=args,
                        extrinsics=scene_data.extrinsics,
                        intrinsics=scene_data.intrinsics,
                        process_res_method=getattr(args, "da3_process_res_method", "upper_bound_resize"),
                    )
                else:
                    posed_prediction = clone_prediction_dict(unposed_prediction)
                    posed_preprocess_meta = dict(unposed_preprocess_meta) if unposed_preprocess_meta else None
            else:
                raise ValueError(f"Unsupported model_kind: {model_kind}")

            write_dynamic_replica_clip_exports(
                output_root,
                clip,
                unposed_prediction,
                posed_prediction,
                scene_data,
                unposed_preprocess_meta=unposed_preprocess_meta,
                posed_preprocess_meta=posed_preprocess_meta,
            )
            gc.collect()

    metrics: dict[str, Any] = {}
    if args.enable_scene_flow_eval:
        if model_kind != "da3_stitched":
            logger.warning("Scene-flow evaluation is only enabled for da3_stitched exports; skipping {}.", model_kind)
        else:
            metrics.update(
                evaluate_dynamic_replica_scene_flow_work_dir(
                    dataset_root=args.dynamic_replica_root,
                    work_dir=output_root,
                    clips=clips,
                    clip_length=int(args.clip_length),
                    target_space=str(args.scene_flow_target_space),
                    eval_space=str(args.scene_flow_eval_space),
                    threshold=float(args.scene_flow_threshold),
                )
            )

    metrics.update(
        evaluate_dynamic_replica_work_dir(
            args.dynamic_replica_root,
            output_root,
            clips,
            voxel_length=args.voxel_length,
            sdf_trunc=args.sdf_trunc,
            max_depth=args.max_depth_value,
            sampling_number=args.sampling_number,
            eval_threshold=args.eval_threshold,
            down_sample=args.down_sample,
            reuse_existing_fuse=args.reuse_existing_fuse,
            depth_alignment_strategy=args.dynamic_replica_depth_alignment_strategy,
            depth_alignment_sample_stride=args.dynamic_replica_depth_alignment_sample_stride,
        )
    )
    raw_camera_metrics = evaluate_dynamic_replica_raw_camera_diagnostics(
        output_root,
        clips,
        dataset_root=args.dynamic_replica_root,
    )
    metrics.update(raw_camera_metrics)
    dump_json(output_root / "metric_results" / "dynamic_replica_summary.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    clips = build_dynamic_replica_clips(
        args.dynamic_replica_root,
        split=args.split,
        clip_length=args.clip_length,
        scenes=args.scenes,
        camera_name="left",
    )
    clips = select_dynamic_replica_clips(
        clips,
        lazy_mode=bool(args.lazy_mode),
        max_clips=int(args.max_clips),
    )

    logger.info(
        "Dynamic Replica benchmark uses {} left-camera clips from {} scenes.",
        len(clips),
        len({clip.scene_name for clip in clips}),
    )
    if not clips:
        raise ValueError("No Dynamic Replica clips selected for evaluation.")
    resolve_runtime_max_frames(args, clips)

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
        dump_json(Path(args.output_dir) / "dynamic_replica_summary_all_models.json", all_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
