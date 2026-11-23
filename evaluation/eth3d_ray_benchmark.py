from __future__ import annotations

import argparse
import gc
import importlib
import json
import multiprocessing
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for import_path in (PROJECT_ROOT, PLAN4D_ROOT, DA3_SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from utils.da3_resize import (
    apply_resize_only_to_intrinsics,
    build_upper_bound_resize_transform,
)


DEFAULT_INPUT_RESOLUTION = 532
DEFAULT_STITCHED_INPUT_RESOLUTION = 504
DEFAULT_MAX_FRAMES = 100
DEFAULT_STITCHED_INPUT_ORDER_STRATEGY = "dataset"
STITCHED_INPUT_ORDER_STRATEGIES = ("dataset", "camera_path")


def _aux_to_dict(aux: Any) -> dict[str, Any]:
    if aux is None:
        return {}
    if isinstance(aux, dict):
        return dict(aux)
    if hasattr(aux, "items"):
        return dict(aux.items())
    return dict(vars(aux))


@dataclass(frozen=True)
class SquareResizeCropTransform:
    original_height: int
    original_width: int
    target_size: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    scale_y: float
    scale_x: float

    def to_export_dict(self) -> dict[str, np.ndarray]:
        return {
            "original_height": np.array(self.original_height, dtype=np.int32),
            "original_width": np.array(self.original_width, dtype=np.int32),
            "target_size": np.array(self.target_size, dtype=np.int32),
            "resized_height": np.array(self.resized_height, dtype=np.int32),
            "resized_width": np.array(self.resized_width, dtype=np.int32),
            "crop_top": np.array(self.crop_top, dtype=np.int32),
            "crop_left": np.array(self.crop_left, dtype=np.int32),
            "scale_y": np.array(self.scale_y, dtype=np.float32),
            "scale_x": np.array(self.scale_x, dtype=np.float32),
        }


@dataclass(frozen=True)
class TemporalPadInfo:
    original_frames: int
    padded_frames: int
    pad_count: int
    pad_source_indices: tuple[int, ...] = ()

    def to_export_dict(self) -> dict[str, np.ndarray]:
        return {
            "original_frames": np.array(self.original_frames, dtype=np.int32),
            "padded_frames": np.array(self.padded_frames, dtype=np.int32),
            "pad_count": np.array(self.pad_count, dtype=np.int32),
            "pad_source_indices": np.asarray(self.pad_source_indices, dtype=np.int32),
        }


def default_eth3d_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "DA3-bench" / "eth3d").resolve())


def default_da3_model_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())


def default_wan_vae_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "longcat" / "vae").resolve())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ETH3D full benchmark with configurable camera-parameter sources for raw DA3 "
            "anyview and/or PLAN4D DA3 stitched models."
        )
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="both",
        choices=["da3_anyview", "da3_anyview_square", "da3_stitched", "da3_stitched_square", "both"],
        help="Which model path to evaluate.",
    )
    parser.add_argument(
        "--eth3d_root",
        type=str,
        default=default_eth3d_root(),
        help="ETH3D benchmark root directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for benchmark-compatible exports and metric JSON files.",
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
        default=DEFAULT_MAX_FRAMES,
        help="Maximum frames per ETH3D scene. Uses the benchmark's fixed seed sampling.",
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=DEFAULT_INPUT_RESOLUTION,
        help="Target resolution. Upper-bound resize paths treat it as long-side target; square paths treat it as crop size.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional subset of ETH3D scene names.",
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
        default=default_da3_model_path(),
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
        help="Default preprocess semantics for stitched DA3. Legacy square benchmark overrides this to square_center_crop.",
    )
    parser.add_argument(
        "--stitched_da3_sky_postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply official DA3 mono sky postprocess on ETH3D stitched predictions.",
    )
    parser.add_argument(
        "--stitched_da3_posed_rerun_with_gt_camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, ETH3D stitched posed evaluation reruns student inference with GT "
            "camera tokens. Enabled by default for the stitched ETH3D benchmark entrypoints."
        ),
    )
    parser.add_argument(
        "--stitched_da3_train_style_eval_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, align stitched DA3 ETH3D outputs into the GT train-style loss space "
            "before decoding cameras and exporting depths."
        ),
    )
    parser.add_argument(
        "--eth3d_depth_scale_shift_alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, fit one global affine depth transform d_gt = scale * d_pred + shift "
            "per ETH3D scene before reconstruction fusion, and write aligned_pose.json diagnostics."
        ),
    )
    parser.add_argument(
        "--eth3d_disable_middle_frame_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable ETH3D fixed-seed middle-frame subsampling. When enabled, the benchmark "
            "keeps the selected input order and only truncates the tail with --max_frames, "
            "while slicing GT cameras and image metadata with the same indices."
        ),
    )
    parser.add_argument(
        "--stitched_da3_camera_param_source",
        type=str,
        default="ray_map",
        choices=["ray_map", "camera_token"],
        help=(
            "Which stitched DA3 output to decode camera parameters from. "
            "'ray_map' preserves the legacy ETH3D behavior; 'camera_token' uses cam_dec outputs."
        ),
    )
    parser.add_argument(
        "--stitched_da3_input_order_strategy",
        type=str,
        default=DEFAULT_STITCHED_INPUT_ORDER_STRATEGY,
        choices=STITCHED_INPUT_ORDER_STRATEGIES,
        help=(
            "Input image ordering for stitched DA3. 'dataset' preserves the ETH3D/COLMAP order; "
            "'camera_path' greedily orders views by nearest camera-center trajectory for video VAE input."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.eth3d_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"ETH3D root does not exist: {dataset_root}")
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.max_frames == 0:
        raise ValueError("--max_frames must be positive or -1 to disable sampling.")
    if args.model_kind in {"da3_stitched", "da3_stitched_square", "both"} and not args.checkpoint_path:
        raise ValueError("--checkpoint_path is required for da3_stitched, da3_stitched_square, or both.")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _to_float_dict(values: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def mean_of_dicts(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = values[0].keys()
    return {key: float(np.mean([item[key] for item in values]).item()) for key in keys}


def clone_prediction_dict(prediction: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, dict):
            cloned[key] = {
                nested_key: nested_value.copy() if isinstance(nested_value, np.ndarray) else nested_value
                for nested_key, nested_value in value.items()
            }
        else:
            cloned[key] = value
    return cloned


def list_eth3d_scenes(dataset_root: str | Path) -> list[str]:
    return sorted(path.name for path in Path(dataset_root).iterdir() if path.is_dir())


def _slice_scene_data(scene_data, indices: list[int]) -> SimpleNamespace:
    aux_dict = _aux_to_dict(scene_data.aux)
    total_frames = len(scene_data.image_files)
    index_array = np.asarray(indices, dtype=np.int64)

    sliced_aux = {}
    for key, value in aux_dict.items():
        if isinstance(value, list) and len(value) == total_frames:
            sliced_aux[key] = [value[idx] for idx in indices]
        elif isinstance(value, np.ndarray) and len(value) == total_frames:
            sliced_aux[key] = value[index_array]
        else:
            sliced_aux[key] = value

    return SimpleNamespace(
        image_files=[scene_data.image_files[idx] for idx in indices],
        extrinsics=np.asarray(scene_data.extrinsics)[index_array],
        intrinsics=np.asarray(scene_data.intrinsics)[index_array],
        aux=SimpleNamespace(**sliced_aux),
    )


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1] < 3 or extrinsics.shape[2] < 4:
        raise ValueError(f"Expected extrinsics with shape [N,3|4,4], got {tuple(extrinsics.shape)}")
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)


def camera_path_order_indices(extrinsics: np.ndarray) -> list[int]:
    centers = camera_centers_from_extrinsics(extrinsics)
    num_frames = int(centers.shape[0])
    if num_frames <= 2:
        return list(range(num_frames))
    if not np.isfinite(centers).all():
        return list(range(num_frames))

    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    best_order: list[int] | None = None
    best_score: tuple[float, float, tuple[int, ...]] | None = None
    for start_index in range(num_frames):
        unvisited = set(range(num_frames))
        unvisited.remove(start_index)
        order = [start_index]
        current_index = start_index
        while unvisited:
            next_index = min(
                unvisited,
                key=lambda candidate: (float(distances[current_index, candidate]), candidate),
            )
            unvisited.remove(next_index)
            order.append(next_index)
            current_index = next_index

        step_distances = distances[order[:-1], order[1:]]
        score = (
            float(np.max(step_distances)),
            float(np.sum(step_distances)),
            tuple(order),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_order = order

    if best_order is None:
        return list(range(num_frames))
    return best_order


def resolve_scene_order_indices(scene_data, order_strategy: str) -> list[int]:
    if order_strategy == "dataset":
        return list(range(len(scene_data.image_files)))
    if order_strategy == "camera_path":
        return camera_path_order_indices(np.asarray(scene_data.extrinsics))
    raise ValueError(f"Unsupported input order strategy: {order_strategy}")


def sample_scene_data(
    scene_data,
    max_frames: int,
    *,
    order_strategy: str = "dataset",
    disable_middle_frame_filter: bool = False,
) -> SimpleNamespace:
    total_frames = len(scene_data.image_files)
    if disable_middle_frame_filter or order_strategy != "dataset":
        indices = resolve_scene_order_indices(scene_data, order_strategy)
        if max_frames > 0 and len(indices) > max_frames:
            indices = indices[:max_frames]
        return _slice_scene_data(scene_data, indices)

    if max_frames <= 0 or len(scene_data.image_files) <= max_frames:
        return _slice_scene_data(scene_data, list(range(total_frames)))

    random.seed(42)
    indices = list(range(len(scene_data.image_files)))
    random.shuffle(indices)
    indices = sorted(indices[:max_frames])
    return _slice_scene_data(scene_data, indices)


def largest_temporal_length_4k_plus_1_not_exceeding(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    return 1 + ((num_frames - 1) // 4) * 4


def resolve_stitched_eval_max_frames(
    *,
    total_frames: int,
    requested_max_frames: int,
) -> int:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if requested_max_frames <= 0:
        available_frames = total_frames
    else:
        available_frames = min(total_frames, requested_max_frames)
    return available_frames


def load_rgb_frames(image_files: list[Any]) -> torch.Tensor:
    frames = []
    for image_input in image_files:
        if isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            frame = imageio.imread(image_input)
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC image input, got shape {tuple(frame.shape)}")
        frame = np.ascontiguousarray(frame[..., :3])
        frames.append(torch.from_numpy(frame))
    return rearrange(torch.stack(frames, dim=0), "t h w c -> t c h w").contiguous()


def next_temporal_length_4k_plus_1(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    remainder = (num_frames - 1) % 4
    if remainder == 0:
        return num_frames
    return num_frames + (4 - remainder)


def sample_temporal_pad_source_indices(
    original_frames: int,
    padded_frames: int,
) -> tuple[int, ...]:
    if original_frames <= 0:
        raise ValueError("original_frames must be positive")
    if padded_frames < original_frames:
        raise ValueError("padded_frames must be greater than or equal to original_frames")
    pad_count = padded_frames - original_frames
    if pad_count == 0:
        return ()
    return tuple(random.Random(42).choices(range(original_frames), k=pad_count))


def pad_frames_for_stitched_inference(frames: torch.Tensor) -> tuple[torch.Tensor, TemporalPadInfo]:
    if frames.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W] frames, got {tuple(frames.shape)}")
    original_frames = int(frames.shape[0])
    padded_frames = next_temporal_length_4k_plus_1(original_frames)
    pad_count = padded_frames - original_frames
    if pad_count == 0:
        return frames, TemporalPadInfo(
            original_frames=original_frames,
            padded_frames=padded_frames,
            pad_count=0,
        )

    pad_source_indices = sample_temporal_pad_source_indices(original_frames, padded_frames)
    pad_frames = frames[list(pad_source_indices)]
    padded = torch.cat([frames, pad_frames], dim=0)
    return padded, TemporalPadInfo(
        original_frames=original_frames,
        padded_frames=padded_frames,
        pad_count=pad_count,
        pad_source_indices=pad_source_indices,
    )


def build_square_resize_crop_transform(
    height: int,
    width: int,
    target_size: int,
) -> SquareResizeCropTransform:
    scale = target_size / min(height, width)
    resized_height = round(height * scale)
    resized_width = round(width * scale)
    crop_top = max(0, (resized_height - target_size) // 2)
    crop_left = max(0, (resized_width - target_size) // 2)
    scale_y = resized_height / height
    scale_x = resized_width / width
    return SquareResizeCropTransform(
        original_height=height,
        original_width=width,
        target_size=target_size,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=crop_top,
        crop_left=crop_left,
        scale_y=scale_y,
        scale_x=scale_x,
    )


def _normalize_bt_hw(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() == 5 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if tensor.dim() != 4:
        raise ValueError(f"Expected [B,T,H,W] or [B,T,1,H,W], got {tuple(tensor.shape)}")
    return tensor.float()


def apply_square_resize_crop_to_intrinsics(
    intrinsics: torch.Tensor | np.ndarray,
    transform: SquareResizeCropTransform,
) -> torch.Tensor | np.ndarray:
    resized = intrinsics.clone() if torch.is_tensor(intrinsics) else intrinsics.copy()
    resized[..., 0, :] *= transform.scale_x
    resized[..., 1, :] *= transform.scale_y
    resized[..., 0, 2] -= transform.crop_left
    resized[..., 1, 2] -= transform.crop_top
    return resized


def _apply_square_resize_crop(
    array: np.ndarray,
    transform: SquareResizeCropTransform,
    *,
    interpolation: int,
) -> np.ndarray:
    resized = cv2.resize(
        array,
        (transform.resized_width, transform.resized_height),
        interpolation=interpolation,
    )
    return resized[
        transform.crop_top : transform.crop_top + transform.target_size,
        transform.crop_left : transform.crop_left + transform.target_size,
    ]


def _build_square_rgb_inputs(
    image_files: list[str],
    target_size: int,
) -> tuple[list[np.ndarray], SquareResizeCropTransform]:
    frames = load_rgb_frames(image_files)
    original_height, original_width = frames.shape[-2:]
    transform = build_square_resize_crop_transform(
        height=original_height,
        width=original_width,
        target_size=target_size,
    )
    square_images = [
        _apply_square_resize_crop(
            frame.permute(1, 2, 0).cpu().numpy(),
            transform,
            interpolation=cv2.INTER_LINEAR,
        )
        for frame in frames
    ]
    return square_images, transform


def _load_square_preprocess_transform(preprocess_path: Path) -> SquareResizeCropTransform:
    meta = np.load(preprocess_path)
    return SquareResizeCropTransform(
        original_height=int(meta["original_height"]),
        original_width=int(meta["original_width"]),
        target_size=int(meta["target_size"]),
        resized_height=int(meta["resized_height"]),
        resized_width=int(meta["resized_width"]),
        crop_top=int(meta["crop_top"]),
        crop_left=int(meta["crop_left"]),
        scale_y=float(meta["scale_y"]),
        scale_x=float(meta["scale_x"]),
    )


def _pad_camera_sequence_for_stitched_inference(
    values: np.ndarray,
    padded_frames: int,
    *,
    pad_source_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    original_frames = int(values.shape[0])
    if original_frames >= padded_frames:
        return values
    pad_count = padded_frames - original_frames
    if pad_source_indices is None:
        pad_source_indices = sample_temporal_pad_source_indices(original_frames, padded_frames)
    if len(pad_source_indices) != pad_count:
        raise ValueError(
            f"Expected {pad_count} pad_source_indices for padded_frames={padded_frames}, "
            f"got {len(pad_source_indices)}"
        )
    pad_values = values[np.asarray(pad_source_indices, dtype=np.int64)]
    return np.concatenate([values, pad_values], axis=0)


def _align_prediction_to_input_camera_scale(
    depth: torch.Tensor,
    pred_extrinsics: torch.Tensor,
    input_extrinsics: torch.Tensor,
    input_intrinsics: torch.Tensor,
    *,
    ransac_view_thresh: int = 10,
) -> dict[str, np.ndarray]:
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    pred_extrinsics_np = pred_extrinsics.squeeze(0).detach().cpu().numpy()
    input_extrinsics_np = input_extrinsics.squeeze(0).detach().cpu().numpy()
    input_intrinsics_np = input_intrinsics.squeeze(0).detach().cpu().numpy()
    _, _, scale, _ = align_poses_umeyama(
        pred_extrinsics_np,
        input_extrinsics_np,
        ransac=len(input_extrinsics_np) >= ransac_view_thresh,
        return_aligned=True,
        random_state=42,
    )
    return {
        "depth": (depth.squeeze(0).detach().cpu().numpy() / float(scale)).astype(np.float32),
        "extrinsics": input_extrinsics_np[:, :3, :].astype(np.float32),
        "intrinsics": input_intrinsics_np.astype(np.float32),
        "scale": float(scale),
    }


def _camera_params_to_numpy(
    camera_params: dict[str, Any],
    *,
    num_frames: int,
) -> dict[str, np.ndarray]:
    export: dict[str, np.ndarray] = {}
    for key, value in camera_params.items():
        if value is None:
            continue
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.dim() >= 2 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            export[key] = tensor[:num_frames].numpy().astype(np.float32)
        else:
            export[key] = np.asarray(value, dtype=np.float32)[:num_frames]
    return export


def _intrinsics_to_normalized_camera_params(
    intrinsics: np.ndarray,
    *,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    focal_lengths = np.stack(
        [
            intrinsics[..., 0, 0] * 2.0 / float(image_width),
            intrinsics[..., 1, 1] * 2.0 / float(image_height),
        ],
        axis=-1,
    ).astype(np.float32)
    principal_points = np.stack(
        [
            intrinsics[..., 0, 2] * 2.0 / float(image_width),
            intrinsics[..., 1, 2] * 2.0 / float(image_height),
        ],
        axis=-1,
    ).astype(np.float32)
    return focal_lengths, principal_points


def _intrinsics_tensor_to_normalized_camera_params(
    intrinsics: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = intrinsics.float()
    focal_lengths = torch.stack(
        [
            intrinsics[..., 0, 0] * 2.0 / float(image_width),
            intrinsics[..., 1, 1] * 2.0 / float(image_height),
        ],
        dim=-1,
    )
    principal_points = torch.stack(
        [
            intrinsics[..., 0, 2] * 2.0 / float(image_width),
            intrinsics[..., 1, 2] * 2.0 / float(image_height),
        ],
        dim=-1,
    )
    return focal_lengths, principal_points


def _build_camera_params_from_token_outputs(
    stitched_outputs: dict[str, Any],
    *,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    extrinsics = stitched_outputs.get("camera_extrinsics")
    intrinsics = stitched_outputs.get("camera_intrinsics")
    if extrinsics is None or intrinsics is None:
        raise RuntimeError(
            "Stitched inference did not return camera-token decoded extrinsics/intrinsics."
        )
    focal_lengths, principal_points = _intrinsics_tensor_to_normalized_camera_params(
        intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "focal_lengths": focal_lengths,
        "principal_points": principal_points,
    }


def _prediction_to_raw_camera_params(
    prediction: Any,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray]:
    extrinsics = getattr(prediction, "extrinsics", None)
    intrinsics = getattr(prediction, "intrinsics", None)
    if extrinsics is None or intrinsics is None:
        raise RuntimeError("DA3 anyview raw prediction did not return extrinsics/intrinsics.")

    raw_intrinsics = np.asarray(intrinsics, dtype=np.float32)
    focal_lengths = getattr(prediction, "focal_lengths", None)
    principal_points = getattr(prediction, "principal_points", None)
    if focal_lengths is None or principal_points is None:
        focal_lengths, principal_points = _intrinsics_to_normalized_camera_params(
            raw_intrinsics,
            image_height=image_height,
            image_width=image_width,
        )

    return {
        "extrinsics": np.asarray(extrinsics, dtype=np.float32),
        "intrinsics": raw_intrinsics,
        "focal_lengths": np.asarray(focal_lengths, dtype=np.float32),
        "principal_points": np.asarray(principal_points, dtype=np.float32),
    }


def _run_anyview_inference_with_raw_camera(
    api,
    image_inputs: list[Any],
    *,
    process_res: int,
    ref_view_strategy: str,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> tuple[Any, dict[str, np.ndarray]]:
    inference_kwargs = dict(
        process_res=process_res,
        process_res_method="upper_bound_resize",
        use_ray_pose=True,
        ref_view_strategy=ref_view_strategy,
        autocast_enabled=False,
    )

    helper_names = (
        "_preprocess_inputs",
        "_prepare_model_inputs",
        "_normalize_extrinsics",
        "_run_model_forward",
        "_convert_to_prediction",
        "_align_to_input_extrinsics_intrinsics",
        "_add_processed_images",
    )
    if all(hasattr(api, name) for name in helper_names):
        imgs_cpu, extrinsics_cpu, intrinsics_cpu = api._preprocess_inputs(
            image_inputs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
            process_res_method="upper_bound_resize",
        )
        imgs, ex_t, in_t = api._prepare_model_inputs(imgs_cpu, extrinsics_cpu, intrinsics_cpu)
        ex_t_norm = api._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        raw_output = api._run_model_forward(
            imgs,
            ex_t_norm,
            in_t,
            [],
            False,
            True,
            ref_view_strategy,
            False,
            autocast_enabled=False,
        )
        prediction = api._convert_to_prediction(raw_output)
        raw_camera_params = _prediction_to_raw_camera_params(
            prediction,
            image_height=int(imgs_cpu.shape[-2]),
            image_width=int(imgs_cpu.shape[-1]),
        )
        prediction = api._align_to_input_extrinsics_intrinsics(
            extrinsics_cpu,
            intrinsics_cpu,
            prediction,
            True,
        )
        prediction = api._add_processed_images(prediction, imgs_cpu)
        return prediction, raw_camera_params

    inference_args: list[Any] = [image_inputs]
    if extrinsics is not None or intrinsics is not None:
        if extrinsics is None or intrinsics is None:
            raise ValueError("extrinsics and intrinsics must be provided together")
        inference_args.extend([extrinsics, intrinsics])
    prediction = api.inference(*inference_args, **inference_kwargs)
    image_height = int(prediction.depth.shape[-2])
    image_width = int(prediction.depth.shape[-1])
    raw_camera_params = _prediction_to_raw_camera_params(
        prediction,
        image_height=image_height,
        image_width=image_width,
    )
    return prediction, raw_camera_params


def restore_square_prediction_to_original(
    depth: torch.Tensor,
    conf: torch.Tensor | None,
    intrinsics: torch.Tensor,
    transform: SquareResizeCropTransform,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    depth = _normalize_bt_hw(depth)
    conf = _normalize_bt_hw(conf)

    batch_size, num_frames, _, _ = depth.shape
    resized_depth = torch.zeros(
        batch_size,
        num_frames,
        transform.resized_height,
        transform.resized_width,
        dtype=depth.dtype,
        device=depth.device,
    )
    resized_depth[
        :,
        :,
        transform.crop_top : transform.crop_top + transform.target_size,
        transform.crop_left : transform.crop_left + transform.target_size,
    ] = depth
    restored_depth = F.interpolate(
        resized_depth.reshape(-1, 1, transform.resized_height, transform.resized_width),
        size=(transform.original_height, transform.original_width),
        mode="nearest",
    ).reshape(batch_size, num_frames, transform.original_height, transform.original_width)

    restored_conf = None
    if conf is not None:
        resized_conf = torch.zeros_like(resized_depth)
        resized_conf[
            :,
            :,
            transform.crop_top : transform.crop_top + transform.target_size,
            transform.crop_left : transform.crop_left + transform.target_size,
        ] = conf
        restored_conf = F.interpolate(
            resized_conf.reshape(-1, 1, transform.resized_height, transform.resized_width),
            size=(transform.original_height, transform.original_width),
            mode="nearest",
        ).reshape(batch_size, num_frames, transform.original_height, transform.original_width)

    restored_intrinsics = intrinsics.clone().float()
    restored_intrinsics[..., 0, 2] += transform.crop_left
    restored_intrinsics[..., 1, 2] += transform.crop_top
    restored_intrinsics[..., 0, :] /= transform.scale_x
    restored_intrinsics[..., 1, :] /= transform.scale_y

    return restored_depth, restored_conf, restored_intrinsics


def _write_results_npz(export_dir: Path, prediction: dict[str, np.ndarray]) -> None:
    result_path = export_dir / "exports" / "mini_npz" / "results.npz"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {"depth": prediction["depth"].astype(np.float32)}
    if prediction.get("conf") is not None:
        save_dict["conf"] = prediction["conf"].astype(np.float32)
    if prediction.get("extrinsics") is not None:
        save_dict["extrinsics"] = prediction["extrinsics"].astype(np.float32)
    if prediction.get("intrinsics") is not None:
        save_dict["intrinsics"] = prediction["intrinsics"].astype(np.float32)
    np.savez_compressed(result_path, **save_dict)


def _write_raw_camera_params_npz(export_dir: Path, prediction: dict[str, Any]) -> None:
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


def _write_gt_meta(export_dir: Path, scene_data) -> None:
    meta_path = export_dir / "exports" / "gt_meta.npz"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        meta_path,
        extrinsics=np.asarray(scene_data.extrinsics, dtype=np.float32),
        intrinsics=np.asarray(scene_data.intrinsics, dtype=np.float32),
        image_files=np.asarray(scene_data.image_files, dtype=object),
    )


def write_eth3d_scene_exports(
    work_dir: str | Path,
    scene_name: str,
    unposed_prediction: dict[str, np.ndarray],
    posed_prediction: dict[str, np.ndarray],
    scene_data,
    preprocess_meta: dict[str, Any] | None = None,
) -> None:
    base_dir = Path(work_dir) / "model_results" / "eth3d" / scene_name
    for suffix, prediction in (("unposed", unposed_prediction), ("posed", posed_prediction)):
        export_dir = base_dir / suffix
        _write_results_npz(export_dir, prediction)
        _write_raw_camera_params_npz(export_dir, prediction)
        _write_gt_meta(export_dir, scene_data)
        if preprocess_meta is not None:
            preprocess_path = export_dir / "exports" / "preprocess_meta.npz"
            preprocess_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(preprocess_path, **preprocess_meta)


def compute_pose_with_gt(result_path: Path, gt_extrinsics: np.ndarray) -> dict[str, float]:
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous

    pred = np.load(result_path)
    return _to_float_dict(
        compute_pose(
            torch.from_numpy(as_homogeneous(pred["extrinsics"])),
            torch.from_numpy(as_homogeneous(gt_extrinsics)),
        )
    )


def _as_homogeneous_extrinsics_np(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float64)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")


def _camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(_as_homogeneous_extrinsics_np(extrinsics))
    return np.asarray(c2w[..., :3, 3], dtype=np.float64)


def _camera_center_alignment_metrics(
    pred_extrinsics: np.ndarray,
    aligned_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
) -> dict[str, float]:
    gt_centers = _camera_centers_from_extrinsics(gt_extrinsics)
    pred_centers = _camera_centers_from_extrinsics(pred_extrinsics)
    aligned_centers = _camera_centers_from_extrinsics(aligned_extrinsics)
    before = np.linalg.norm(pred_centers - gt_centers, axis=-1)
    after = np.linalg.norm(aligned_centers - gt_centers, axis=-1)
    return {
        "center_rmse_before_umeyama": float(np.sqrt(np.mean(before ** 2))) if before.size else 0.0,
        "center_rmse_after_umeyama": float(np.sqrt(np.mean(after ** 2))) if after.size else 0.0,
        "center_mean_after_umeyama": float(np.mean(after)) if after.size else 0.0,
        "center_median_after_umeyama": float(np.median(after)) if after.size else 0.0,
        "center_p90_after_umeyama": float(np.percentile(after, 90)) if after.size else 0.0,
        "center_max_after_umeyama": float(np.max(after)) if after.size else 0.0,
    }


def compute_umeyama_aligned_pose_with_gt(result_path: Path, gt_extrinsics: np.ndarray) -> dict[str, float]:
    from depth_anything_3.bench.utils import compute_pose
    from depth_anything_3.utils.geometry import as_homogeneous
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    with np.load(result_path) as pred:
        pred_extrinsics = np.asarray(pred["extrinsics"], dtype=np.float32)
    gt_extrinsics = np.asarray(gt_extrinsics, dtype=np.float32)
    _, _, scale, aligned_extrinsics = align_poses_umeyama(
        as_homogeneous(gt_extrinsics.copy()),
        as_homogeneous(pred_extrinsics.copy()),
        return_aligned=True,
        ransac=len(gt_extrinsics) >= 10,
        random_state=42,
    )
    metrics = _to_float_dict(
        compute_pose(
            torch.from_numpy(as_homogeneous(np.asarray(aligned_extrinsics, dtype=np.float32))),
            torch.from_numpy(as_homogeneous(gt_extrinsics)),
        )
    )
    metrics["umeyama_scale"] = float(scale)
    metrics.update(
        _camera_center_alignment_metrics(
            pred_extrinsics=pred_extrinsics,
            aligned_extrinsics=np.asarray(aligned_extrinsics, dtype=np.float32),
            gt_extrinsics=gt_extrinsics,
        )
    )
    return metrics


def _fit_sequence_depth_scale_shift_alignment(
    pred_depths: Sequence[np.ndarray],
    gt_depths: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    *,
    eps: float = 1e-6,
) -> tuple[float, float]:
    pred_values: list[np.ndarray] = []
    gt_values: list[np.ndarray] = []
    for pred_depth, gt_depth, valid_mask in zip(pred_depths, gt_depths, valid_masks):
        pred = np.asarray(pred_depth, dtype=np.float32)
        gt = np.asarray(gt_depth, dtype=np.float32)
        valid = np.asarray(valid_mask, dtype=bool)
        valid = valid & np.isfinite(pred) & np.isfinite(gt) & (pred > eps) & (gt > eps)
        if valid.any():
            pred_values.append(pred[valid].reshape(-1).astype(np.float64))
            gt_values.append(gt[valid].reshape(-1).astype(np.float64))

    if not pred_values:
        return 1.0, 0.0
    pred_concat = np.concatenate(pred_values, axis=0)
    gt_concat = np.concatenate(gt_values, axis=0)
    if pred_concat.size < 2 or float(np.var(pred_concat)) <= eps:
        pred_median = float(np.median(pred_concat)) if pred_concat.size else 0.0
        gt_median = float(np.median(gt_concat)) if gt_concat.size else 0.0
        if pred_median > eps and np.isfinite(pred_median) and np.isfinite(gt_median):
            return float(gt_median / pred_median), 0.0
        return 1.0, 0.0

    design = np.stack([pred_concat, np.ones_like(pred_concat)], axis=1)
    scale, shift = np.linalg.lstsq(design, gt_concat, rcond=None)[0]
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= eps:
        return 1.0, 0.0
    return float(scale), float(shift)


def _apply_depth_scale_shift_alignment(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    *,
    scale: float,
    shift: float,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    aligned = depth * np.float32(scale) + np.float32(shift)
    keep = valid & np.isfinite(aligned) & (aligned > 0.0)
    return np.where(keep, aligned, 0.0).astype(np.float32)


def _fit_depth_scale_shift_from_stats(
    count: int,
    sum_pred: float,
    sum_gt: float,
    sum_pred_sq: float,
    sum_pred_gt: float,
    *,
    eps: float = 1e-6,
) -> tuple[float, float]:
    if count < 2:
        return 1.0, 0.0
    denom = float(count) * sum_pred_sq - sum_pred * sum_pred
    if abs(denom) <= eps:
        pred_mean = sum_pred / float(count)
        gt_mean = sum_gt / float(count)
        if pred_mean > eps and np.isfinite(pred_mean) and np.isfinite(gt_mean):
            return float(gt_mean / pred_mean), 0.0
        return 1.0, 0.0
    scale = (float(count) * sum_pred_gt - sum_pred * sum_gt) / denom
    shift = (sum_gt - scale * sum_pred) / float(count)
    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= eps:
        return 1.0, 0.0
    return float(scale), float(shift)


def _load_eth3d_original_gt_depth_and_valid_mask(
    image_input: Any,
    *,
    original_height: int,
    original_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    gt_depth_path, gt_mask_path = _resolve_eth3d_gt_paths_from_image_path(image_input)
    gt_depth = np.zeros((original_height, original_width), dtype=np.float32)
    valid_mask = np.zeros((original_height, original_width), dtype=bool)
    if gt_depth_path is None or not gt_depth_path.exists():
        return gt_depth, valid_mask

    gt_depth = np.fromfile(gt_depth_path, dtype=np.float32)
    expected_size = original_height * original_width
    if gt_depth.size != expected_size:
        raise ValueError(
            f"Unexpected GT depth size for {gt_depth_path}: got {gt_depth.size}, "
            f"expected {expected_size}"
        )
    gt_depth = gt_depth.reshape(original_height, original_width).astype(np.float32)

    if gt_mask_path is not None and gt_mask_path.exists():
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise RuntimeError(f"Failed to read ETH3D mask: {gt_mask_path}")
    else:
        gt_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    valid_mask = np.logical_not(gt_mask == 1)
    valid_mask &= np.isfinite(gt_depth)
    valid_mask &= gt_depth > 0
    gt_depth = np.where(valid_mask, gt_depth, 0.0).astype(np.float32)
    return gt_depth, valid_mask


def _fit_regular_eth3d_depth_scale_shift_alignment(
    pred_depths: np.ndarray,
    image_files: Sequence[Any],
) -> tuple[float, float, int]:
    count = 0
    sum_pred = 0.0
    sum_gt = 0.0
    sum_pred_sq = 0.0
    sum_pred_gt = 0.0
    for index, image_path in enumerate(image_files):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read ETH3D image: {image_path}")
        original_height, original_width = image.shape[:2]
        depth = cv2.resize(
            pred_depths[index],
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        gt_depth, valid_mask = _load_eth3d_original_gt_depth_and_valid_mask(
            image_path,
            original_height=original_height,
            original_width=original_width,
        )
        valid = valid_mask & np.isfinite(depth) & (depth > 1e-6)
        if not valid.any():
            continue
        pred_values = depth[valid].astype(np.float64)
        gt_values = gt_depth[valid].astype(np.float64)
        count += int(pred_values.size)
        sum_pred += float(pred_values.sum())
        sum_gt += float(gt_values.sum())
        sum_pred_sq += float(np.square(pred_values).sum())
        sum_pred_gt += float((pred_values * gt_values).sum())

    scale, shift = _fit_depth_scale_shift_from_stats(
        count,
        sum_pred,
        sum_gt,
        sum_pred_sq,
        sum_pred_gt,
    )
    return scale, shift, count


def _snapshot_eth3d_dataset_config(dataset) -> dict[str, Any]:
    return {
        "data_root": str(dataset.data_root),
        "voxel_length": float(dataset.voxel_length),
        "sdf_trunc": float(dataset.sdf_trunc),
        "max_depth": float(dataset.max_depth),
        "sampling_number": int(dataset.sampling_number),
        "eval_threshold": float(dataset.eval_threshold),
        "down_sample": dataset.down_sample,
    }


def _restore_eth3d_dataset_from_config(config: dict[str, Any]):
    from depth_anything_3.bench.datasets.eth3d import ETH3D

    dataset = ETH3D()
    dataset.data_root = config["data_root"]
    dataset.voxel_length = config["voxel_length"]
    dataset.sdf_trunc = config["sdf_trunc"]
    dataset.max_depth = config["max_depth"]
    dataset.sampling_number = config["sampling_number"]
    dataset.eval_threshold = config["eval_threshold"]
    dataset.down_sample = config["down_sample"]
    return dataset


def _resolve_eth3d_gt_meta_for_result(dataset, scene: str, result_path: str | Path) -> SimpleNamespace:
    result_path = Path(result_path)
    saved_meta_path = result_path.parent.parent / "gt_meta.npz"
    if saved_meta_path.exists():
        return _load_saved_gt_meta(result_path)

    if hasattr(dataset, "_load_gt_meta"):
        gt_data = dataset._load_gt_meta(str(result_path))
        if gt_data is not None and getattr(gt_data, "image_files", None) is not None:
            return gt_data

    return dataset.get_data(scene)


def _evaluate_eth3d_scene_metrics_local(
    work_dir: str | Path,
    scene: str,
    dataset_config: dict[str, Any],
    *,
    reuse_existing_fuse: bool = False,
    depth_scale_shift_alignment: bool = False,
) -> dict[str, dict[str, float]]:
    dataset = _restore_eth3d_dataset_from_config(dataset_config)
    base_dir = Path(work_dir) / "model_results" / "eth3d" / scene
    unposed_result_path = base_dir / "unposed" / "exports" / "mini_npz" / "results.npz"
    posed_result_path = base_dir / "posed" / "exports" / "mini_npz" / "results.npz"
    gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)

    pose_metrics = compute_pose_with_gt(
        result_path=unposed_result_path,
        gt_extrinsics=gt_meta["extrinsics"],
    )

    unposed_fuse_path = base_dir / "unposed" / "exports" / "fuse" / "pcd.ply"
    if not (reuse_existing_fuse and unposed_fuse_path.exists() and unposed_fuse_path.stat().st_size > 0):
        if depth_scale_shift_alignment:
            _fuse_regular_eth3d_scene_streaming(
                dataset,
                scene,
                unposed_result_path,
                unposed_fuse_path,
                "recon_unposed",
                depth_scale_shift_alignment=True,
            )
        else:
            _fuse_regular_eth3d_scene_streaming(
                dataset,
                scene,
                unposed_result_path,
                unposed_fuse_path,
                "recon_unposed",
            )
    recon_unposed_metrics = _to_float_dict(dataset.eval3d(scene, str(unposed_fuse_path)))

    posed_fuse_path = base_dir / "posed" / "exports" / "fuse" / "pcd.ply"
    if not (reuse_existing_fuse and posed_fuse_path.exists() and posed_fuse_path.stat().st_size > 0):
        if depth_scale_shift_alignment:
            _fuse_regular_eth3d_scene_streaming(
                dataset,
                scene,
                posed_result_path,
                posed_fuse_path,
                "recon_posed",
                depth_scale_shift_alignment=True,
            )
        else:
            _fuse_regular_eth3d_scene_streaming(
                dataset,
                scene,
                posed_result_path,
                posed_fuse_path,
                "recon_posed",
            )
    recon_posed_metrics = _to_float_dict(dataset.eval3d(scene, str(posed_fuse_path)))

    return {
        "pose": pose_metrics,
        "recon_unposed": recon_unposed_metrics,
        "recon_posed": recon_posed_metrics,
    }


def _fuse_regular_eth3d_scene_streaming(
    dataset,
    scene: str,
    result_path: str | Path,
    fuse_path: str | Path,
    mode: str,
    *,
    depth_scale_shift_alignment: bool = False,
) -> None:
    from depth_anything_3.bench.dataset import _wait_for_file_ready
    from depth_anything_3.bench.utils import (
        create_tsdf_volume,
        sample_points_from_mesh,
        write_point_cloud_with_cameras,
    )
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    result_path = Path(result_path)
    fuse_path = Path(fuse_path)

    gt_data = _resolve_eth3d_gt_meta_for_result(dataset, scene, result_path)

    image_files = [str(path) for path in gt_data.image_files]
    gt_extrinsics = np.asarray(gt_data.extrinsics, dtype=np.float32)
    gt_intrinsics = np.asarray(gt_data.intrinsics, dtype=np.float32)

    _wait_for_file_ready(str(result_path))
    with np.load(result_path) as pred_npz:
        pred_depths = np.asarray(pred_npz["depth"], dtype=np.float32)
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float32)
        pred_intrinsics = np.asarray(pred_npz["intrinsics"], dtype=np.float32)

    if mode == "recon_unposed":
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            gt_extrinsics.copy(),
            pred_extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
    elif mode == "recon_posed":
        depth_scale_extrinsics = _select_depth_scale_extrinsics_for_fusion(
            result_path,
            pred_extrinsics,
            mode=mode,
        )
        _, _, scale, _ = align_poses_umeyama(
            gt_extrinsics.copy(),
            depth_scale_extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        aligned_extrinsics = gt_extrinsics.copy()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    depth_alignment_scale = float(scale)
    depth_alignment_shift = 0.0
    use_depth_scale_shift_alignment = bool(depth_scale_shift_alignment)
    if use_depth_scale_shift_alignment:
        depth_alignment_scale, depth_alignment_shift, valid_pixel_count = (
            _fit_regular_eth3d_depth_scale_shift_alignment(pred_depths, image_files)
        )
        if valid_pixel_count < 2:
            logger.warning(
                "ETH3D depth scale+shift alignment found no valid GT samples for scene {} mode {}; falling back to Umeyama scale-only.",
                scene,
                mode,
            )
            use_depth_scale_shift_alignment = False
            depth_alignment_scale = float(scale)
            depth_alignment_shift = 0.0
        else:
            logger.info(
                "ETH3D depth scale+shift alignment for scene {} mode {}: scale={}, shift={}, valid_pixels={}",
                scene,
                mode,
                depth_alignment_scale,
                depth_alignment_shift,
                valid_pixel_count,
            )

    model_height, model_width = pred_depths.shape[1], pred_depths.shape[2]
    volume = create_tsdf_volume(
        voxel_length=dataset.voxel_length,
        sdf_trunc=dataset.sdf_trunc,
    )
    intrinsics_out: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []

    for index, image_path in enumerate(image_files):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read ETH3D image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        image_sizes.append((original_height, original_width))

        depth = cv2.resize(
            pred_depths[index],
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        if use_depth_scale_shift_alignment:
            _, depth_valid_mask = _load_eth3d_original_gt_depth_and_valid_mask(
                image_path,
                original_height=original_height,
                original_width=original_width,
            )
            depth_valid_mask = depth_valid_mask & np.isfinite(depth) & (depth > 0)
            depth = _apply_depth_scale_shift_alignment(
                depth,
                depth_valid_mask,
                scale=depth_alignment_scale,
                shift=depth_alignment_shift,
            )
        else:
            gt_zero_mask = None
            if hasattr(dataset, "_load_gt_mask"):
                gt_zero_mask = dataset._load_gt_mask(
                    scene,
                    Path(image_path).name,
                    (original_height, original_width),
                )
            if hasattr(dataset, "_mask_invalid_depth"):
                depth = dataset._mask_invalid_depth(depth, gt_zero_mask)
            else:
                pred_invalid = np.isnan(depth) | np.isinf(depth)
                if gt_zero_mask is not None:
                    depth = depth * np.logical_and(gt_zero_mask, np.logical_not(pred_invalid)).astype(
                        np.float32
                    )
                else:
                    depth[pred_invalid | (depth <= 0)] = 0.0
            depth = depth * float(scale)

        if mode == "recon_unposed":
            intrinsic = pred_intrinsics[index].copy()
            intrinsic[0, :] *= original_width / model_width
            intrinsic[1, :] *= original_height / model_height
        else:
            intrinsic = gt_intrinsics[index].copy()
        intrinsics_out.append(intrinsic.astype(np.float32))

        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(image.astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_trunc=dataset.max_depth,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            original_width,
            original_height,
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        )
        volume.integrate(rgbd, pinhole, aligned_extrinsics[index])

        del image, depth, depth_o3d, color_o3d, rgbd, pinhole
        if (index + 1) % 8 == 0:
            gc.collect()

    mesh = volume.extract_triangle_mesh()
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)
    fuse_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(fuse_path), pcd)
    write_point_cloud_with_cameras(
        fuse_path=str(fuse_path),
        pcd=pcd,
        intrinsics=np.stack(intrinsics_out, axis=0).astype(np.float32),
        extrinsics=np.asarray(aligned_extrinsics, dtype=np.float32),
        image_sizes=image_sizes,
    )


def _eth3d_scene_metrics_worker(
    queue,
    work_dir: str,
    scene: str,
    dataset_config: dict[str, Any],
    reuse_existing_fuse: bool,
    depth_scale_shift_alignment: bool,
) -> None:
    try:
        queue.put(
            {
                "ok": True,
                "result": _evaluate_eth3d_scene_metrics_local(
                    work_dir=work_dir,
                    scene=scene,
                    dataset_config=dataset_config,
                    reuse_existing_fuse=reuse_existing_fuse,
                    depth_scale_shift_alignment=depth_scale_shift_alignment,
                ),
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _evaluate_eth3d_scene_metrics(
    work_dir: str | Path,
    scene: str,
    dataset_config: dict[str, Any],
    *,
    isolate_process: bool = True,
    reuse_existing_fuse: bool = False,
    depth_scale_shift_alignment: bool = False,
) -> dict[str, dict[str, float]]:
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", None)
    if isolate_process and (main_path is None or not Path(main_path).exists()):
        logger.warning(
            "Falling back to in-process ETH3D metric evaluation for scene {} because "
            "the current __main__ entrypoint is not a real file: {}",
            scene,
            main_path,
        )
        isolate_process = False

    if not isolate_process:
        if depth_scale_shift_alignment:
            return _evaluate_eth3d_scene_metrics_local(
                work_dir=work_dir,
                scene=scene,
                dataset_config=dataset_config,
                reuse_existing_fuse=reuse_existing_fuse,
                depth_scale_shift_alignment=True,
            )
        return _evaluate_eth3d_scene_metrics_local(
            work_dir=work_dir,
            scene=scene,
            dataset_config=dataset_config,
            reuse_existing_fuse=reuse_existing_fuse,
        )

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_eth3d_scene_metrics_worker,
        args=(
            queue,
            str(work_dir),
            scene,
            dataset_config,
            reuse_existing_fuse,
            depth_scale_shift_alignment,
        ),
    )
    process.start()
    process.join()

    payload = None
    if not queue.empty():
        payload = queue.get()
    queue.close()
    queue.join_thread()

    if process.exitcode != 0:
        error_message = None if payload is None else payload.get("error")
        raise RuntimeError(
            f"ETH3D scene metric worker failed for scene={scene} with exitcode={process.exitcode}. "
            f"{error_message or 'No payload returned from worker.'}"
        )
    if payload is None or not payload.get("ok", False):
        raise RuntimeError(
            f"ETH3D scene metric worker returned invalid payload for scene={scene}: {payload}"
        )
    return payload["result"]


def evaluate_eth3d_work_dir(
    work_dir: str | Path,
    dataset,
    scenes: list[str],
    *,
    isolate_process: bool = True,
    reuse_existing_fuse: bool = False,
    depth_scale_shift_alignment: bool = False,
    aligned_pose_diagnostics: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    work_dir = Path(work_dir)
    pose_results: dict[str, dict[str, float]] = {}
    recon_unposed_results: dict[str, dict[str, float]] = {}
    recon_posed_results: dict[str, dict[str, float]] = {}
    aligned_pose_results: dict[str, dict[str, float]] = {}
    dataset_config = _snapshot_eth3d_dataset_config(dataset)

    for scene in scenes:
        logger.info("Evaluating ETH3D metrics for scene {}", scene)
        metric_kwargs = {
            "work_dir": work_dir,
            "scene": scene,
            "dataset_config": dataset_config,
            "isolate_process": isolate_process,
            "reuse_existing_fuse": reuse_existing_fuse,
        }
        if depth_scale_shift_alignment:
            metric_kwargs["depth_scale_shift_alignment"] = True
        scene_metrics = _evaluate_eth3d_scene_metrics(**metric_kwargs)
        pose_results[scene] = scene_metrics["pose"]
        recon_unposed_results[scene] = scene_metrics["recon_unposed"]
        recon_posed_results[scene] = scene_metrics["recon_posed"]
        if aligned_pose_diagnostics:
            base_dir = work_dir / "model_results" / "eth3d" / scene
            unposed_result_path = base_dir / "unposed" / "exports" / "mini_npz" / "results.npz"
            gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)
            aligned_pose_results[scene] = compute_umeyama_aligned_pose_with_gt(
                result_path=unposed_result_path,
                gt_extrinsics=gt_meta["extrinsics"],
            )

    pose_results["mean"] = mean_of_dicts(list(pose_results.values()))
    recon_unposed_results["mean"] = mean_of_dicts(list(recon_unposed_results.values()))
    recon_posed_results["mean"] = mean_of_dicts(list(recon_posed_results.values()))
    if aligned_pose_diagnostics:
        aligned_pose_results["mean"] = mean_of_dicts(list(aligned_pose_results.values()))

    metric_dir = work_dir / "metric_results"
    dump_json(metric_dir / "eth3d_pose.json", pose_results)
    dump_json(metric_dir / "eth3d_recon_unposed.json", recon_unposed_results)
    dump_json(metric_dir / "eth3d_recon_posed.json", recon_posed_results)
    if aligned_pose_diagnostics:
        dump_json(metric_dir / "aligned_pose.json", aligned_pose_results)

    metrics = {
        "pose": pose_results,
        "recon_unposed": recon_unposed_results,
        "recon_posed": recon_posed_results,
    }
    if aligned_pose_diagnostics:
        metrics["aligned_pose"] = aligned_pose_results
    return metrics


def _load_saved_gt_meta(result_path: Path) -> SimpleNamespace:
    export_dir = result_path.parent.parent
    gt_meta = np.load(export_dir / "gt_meta.npz", allow_pickle=True)
    return SimpleNamespace(
        image_files=list(gt_meta["image_files"]),
        extrinsics=np.asarray(gt_meta["extrinsics"], dtype=np.float32),
        intrinsics=np.asarray(gt_meta["intrinsics"], dtype=np.float32),
    )


def _select_depth_scale_extrinsics_for_fusion(
    result_path: str | Path,
    pred_extrinsics: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    pred_extrinsics = np.asarray(pred_extrinsics, dtype=np.float32)
    if mode != "recon_posed":
        return pred_extrinsics

    raw_camera_path = Path(result_path).parent.parent / "raw_camera_params.npz"
    if not raw_camera_path.exists():
        return pred_extrinsics

    try:
        with np.load(raw_camera_path) as raw_camera_params:
            if "extrinsics" not in raw_camera_params:
                return pred_extrinsics
            raw_extrinsics = np.asarray(raw_camera_params["extrinsics"], dtype=np.float32)
    except Exception as exc:
        logger.warning(
            "Failed to load raw camera params for ETH3D posed depth scale from {}: {}",
            raw_camera_path,
            exc,
        )
        return pred_extrinsics

    if raw_extrinsics.shape[0] != pred_extrinsics.shape[0]:
        logger.warning(
            "Ignoring raw camera params for ETH3D posed depth scale because frame count differs: raw={}, exported={}",
            raw_extrinsics.shape[0],
            pred_extrinsics.shape[0],
        )
        return pred_extrinsics
    return raw_extrinsics


def _load_square_zero_mask(
    dataset,
    scene: str,
    image_name: str,
    *,
    original_height: int,
    original_width: int,
    transform: SquareResizeCropTransform,
) -> np.ndarray:
    _, square_zero_mask = _load_square_gt_depth_and_mask(
        dataset,
        scene,
        image_name,
        original_height=original_height,
        original_width=original_width,
        transform=transform,
    )
    return square_zero_mask


def _load_square_gt_depth_and_mask(
    dataset,
    scene: str,
    image_name: str,
    *,
    original_height: int,
    original_width: int,
    transform: SquareResizeCropTransform,
) -> tuple[np.ndarray, np.ndarray]:
    mask_name = (
        image_name.replace(".JPG", ".png")
        .replace(".jpg", ".png")
        .replace(".jpeg", ".png")
    )
    gt_mask_path = Path(dataset.data_root) / scene / "masks_for_images" / "dslr_images" / mask_name
    gt_depth_path = Path(dataset.data_root) / scene / "ground_truth_depth" / "dslr_images" / image_name

    if gt_depth_path.exists():
        gt_depth = np.fromfile(gt_depth_path, dtype=np.float32).reshape(original_height, original_width)
    else:
        gt_depth = np.ones((original_height, original_width), dtype=np.float32)

    if gt_mask_path.exists():
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = np.asarray(gt_mask)
    else:
        gt_mask = np.zeros((original_height, original_width), dtype=np.uint8)

    invalid_mask_from_gt = gt_mask == 1
    gt_depth_copy = gt_depth.copy()
    gt_depth_copy[gt_mask == 1] = 0
    invalid_mask_from_gt_depth = np.logical_or(gt_depth_copy == 0, gt_depth_copy == np.inf)
    zero_mask = np.logical_and(
        np.logical_not(invalid_mask_from_gt),
        np.logical_not(invalid_mask_from_gt_depth),
    ).astype(np.float32)
    square_gt_depth = _apply_square_resize_crop(
        gt_depth,
        transform,
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    square_zero_mask = _apply_square_resize_crop(
        zero_mask,
        transform,
        interpolation=cv2.INTER_NEAREST,
    ) > 0.5
    pred_invalid = np.isnan(square_gt_depth) | np.isinf(square_gt_depth) | (square_gt_depth <= 0)
    square_gt_depth[pred_invalid] = 0.0
    square_zero_mask = np.logical_and(square_zero_mask, np.logical_not(pred_invalid))
    square_gt_depth = square_gt_depth * square_zero_mask.astype(np.float32)
    return square_gt_depth, square_zero_mask


def _load_train_style_alignment_ops() -> dict[str, Any]:
    losses_module = importlib.import_module("plan4d.geometry_losses")
    return {
        "align_depth_least_squares": losses_module.align_depth_least_squares,
        "align_pred_ray_to_gt_loss_space": losses_module.align_pred_ray_to_gt_loss_space,
        "generate_ray_map_batch": losses_module.generate_ray_map_batch,
    }


def _as_homogeneous_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float32)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Unsupported extrinsics shape: {tuple(extrinsics.shape)}")


def _resolve_eth3d_gt_paths_from_image_path(
    image_input: Any,
) -> tuple[Path | None, Path | None]:
    if not isinstance(image_input, (str, Path)):
        return None, None

    image_path = Path(image_input)
    if image_path.parent.name == "images":
        scene_root = image_path.parent.parent
    else:
        parts = image_path.parts
        if "images" not in parts:
            return None, None
        images_idx = len(parts) - 1 - parts[::-1].index("images")
        scene_root = Path(*parts[:images_idx])

    image_name = image_path.name
    mask_name = (
        image_name.replace(".JPG", ".png")
        .replace(".jpg", ".png")
        .replace(".jpeg", ".png")
    )
    gt_depth_path = scene_root / "ground_truth_depth" / "dslr_images" / image_name
    gt_mask_path = scene_root / "masks_for_images" / "dslr_images" / mask_name
    return gt_depth_path, gt_mask_path


def _load_eth3d_gt_depth_and_valid_mask_for_alignment(
    image_input: Any,
    *,
    process_res_method: str,
    transform: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_height = int(transform.original_height)
    original_width = int(transform.original_width)
    gt_depth_path, gt_mask_path = _resolve_eth3d_gt_paths_from_image_path(image_input)

    gt_depth = np.zeros((original_height, original_width), dtype=np.float32)
    valid_mask = np.zeros((original_height, original_width), dtype=np.float32)

    if gt_depth_path is not None and gt_depth_path.exists():
        gt_depth = np.fromfile(gt_depth_path, dtype=np.float32)
        expected_size = original_height * original_width
        if gt_depth.size != expected_size:
            raise ValueError(
                f"Unexpected GT depth size for {gt_depth_path}: got {gt_depth.size}, "
                f"expected {expected_size}"
            )
        gt_depth = gt_depth.reshape(original_height, original_width)

        if gt_mask_path is not None and gt_mask_path.exists():
            gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                raise RuntimeError(f"Failed to read ETH3D mask: {gt_mask_path}")
        else:
            gt_mask = np.zeros((original_height, original_width), dtype=np.uint8)

        valid_mask = np.logical_not(gt_mask == 1)
        valid_mask &= np.isfinite(gt_depth)
        valid_mask &= gt_depth > 0
        valid_mask = valid_mask.astype(np.float32)
        gt_depth = np.where(valid_mask > 0.5, gt_depth, 0.0).astype(np.float32)

    if process_res_method == "upper_bound_resize":
        gt_depth = cv2.resize(
            gt_depth,
            (int(transform.resized_width), int(transform.resized_height)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        valid_mask = cv2.resize(
            valid_mask,
            (int(transform.resized_width), int(transform.resized_height)),
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
    elif process_res_method == "square_center_crop":
        gt_depth = _apply_square_resize_crop(
            gt_depth,
            transform,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        valid_mask = _apply_square_resize_crop(
            valid_mask.astype(np.float32),
            transform,
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {process_res_method}")

    invalid_depth = np.isnan(gt_depth) | np.isinf(gt_depth) | (gt_depth <= 0)
    valid_mask = np.logical_and(valid_mask, np.logical_not(invalid_depth))
    gt_depth = np.where(valid_mask, gt_depth, 0.0).astype(np.float32)
    return torch.from_numpy(gt_depth), torch.from_numpy(valid_mask.astype(np.float32))


def _build_processed_gt_alignment_targets(
    scene_data,
    *,
    process_res_method: str,
    transform: Any,
    original_num_frames: int,
) -> dict[str, torch.Tensor]:
    intrinsics = np.asarray(scene_data.intrinsics, dtype=np.float32)[:original_num_frames]
    extrinsics = _as_homogeneous_extrinsics(
        np.asarray(scene_data.extrinsics, dtype=np.float32)[:original_num_frames]
    )

    if process_res_method == "upper_bound_resize":
        processed_intrinsics = np.stack(
            [
                apply_resize_only_to_intrinsics(
                    item,
                    original_height=int(transform.original_height),
                    original_width=int(transform.original_width),
                    resized_height=int(transform.resized_height),
                    resized_width=int(transform.resized_width),
                )
                for item in intrinsics
            ],
            axis=0,
        ).astype(np.float32)
    elif process_res_method == "square_center_crop":
        processed_intrinsics = np.stack(
            [apply_square_resize_crop_to_intrinsics(item, transform) for item in intrinsics],
            axis=0,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {process_res_method}")

    gt_depths = []
    valid_masks = []
    for image_input in scene_data.image_files[:original_num_frames]:
        gt_depth, valid_mask = _load_eth3d_gt_depth_and_valid_mask_for_alignment(
            image_input,
            process_res_method=process_res_method,
            transform=transform,
        )
        gt_depths.append(gt_depth.float())
        valid_masks.append(valid_mask.float())

    return {
        "target_depth": torch.stack(gt_depths, dim=0).unsqueeze(0).unsqueeze(2),
        "valid_mask": torch.stack(valid_masks, dim=0).unsqueeze(0).unsqueeze(2),
        "intrinsics": torch.from_numpy(processed_intrinsics).unsqueeze(0),
        "extrinsics": torch.from_numpy(extrinsics).unsqueeze(0),
    }


def _apply_train_style_gt_alignment_to_stitched_outputs(
    outputs: dict[str, Any],
    scene_data,
    *,
    args: argparse.Namespace,
    process_res_method: str,
    transform: Any,
    original_num_frames: int,
) -> dict[str, Any]:
    del args

    alignment_ops = _load_train_style_alignment_ops()
    gt_targets = _build_processed_gt_alignment_targets(
        scene_data,
        process_res_method=process_res_method,
        transform=transform,
        original_num_frames=original_num_frames,
    )

    pred_depth = outputs["depth"][:, :original_num_frames].float()
    pred_ray = outputs["ray"][:, :original_num_frames].float()
    depth_conf = outputs.get("depth_conf")
    ray_conf = outputs.get("ray_conf")
    trimmed_depth_conf = None if depth_conf is None else depth_conf[:, :original_num_frames]
    trimmed_ray_conf = None if ray_conf is None else ray_conf[:, :original_num_frames]

    device = pred_depth.device
    target_depth = gt_targets["target_depth"].to(device=device, dtype=pred_depth.dtype)
    valid_mask = gt_targets["valid_mask"].to(device=device, dtype=pred_depth.dtype)
    target_intrinsics = gt_targets["intrinsics"].to(device=device, dtype=pred_depth.dtype)
    target_extrinsics = gt_targets["extrinsics"].to(device=device, dtype=pred_depth.dtype)

    gt_camera_ray = alignment_ops["generate_ray_map_batch"](
        target_intrinsics,
        target_extrinsics,
        int(target_depth.shape[-2]),
        int(target_depth.shape[-1]),
    )
    aligned_depth = alignment_ops["align_depth_least_squares"](
        pred_depth,
        target_depth,
        valid_mask=valid_mask,
        sequence_level=True,
    )
    aligned_ray = alignment_ops["align_pred_ray_to_gt_loss_space"](
        pred_ray,
        gt_camera_ray,
        valid_mask=valid_mask,
    )

    da3_stitched_module = importlib.import_module("models.da3_stitched")
    aligned_camera_params = da3_stitched_module.decode_official_da3_camera_params(
        aligned_ray,
        None if trimmed_ray_conf is None else trimmed_ray_conf.to(device=device, dtype=pred_depth.dtype),
        image_height=int(target_depth.shape[-2]),
        image_width=int(target_depth.shape[-1]),
    )

    updated_outputs = dict(outputs)
    updated_outputs["depth"] = aligned_depth.detach().cpu()
    if trimmed_depth_conf is not None:
        updated_outputs["depth_conf"] = trimmed_depth_conf.detach().cpu()
    updated_outputs["ray"] = aligned_ray.detach().cpu()
    if trimmed_ray_conf is not None:
        updated_outputs["ray_conf"] = trimmed_ray_conf.detach().cpu()
    updated_outputs["camera_params"] = {
        key: value.detach().cpu()
        for key, value in aligned_camera_params.items()
    }
    if "sky" in outputs and torch.is_tensor(outputs["sky"]):
        updated_outputs["sky"] = outputs["sky"][:, :original_num_frames].detach().cpu()
    updated_outputs["camera_param_source"] = "ray_map_train_style_eval_alignment"
    return updated_outputs


def _build_square_gt_point_cloud(
    dataset,
    scene: str,
    result_path: Path,
):
    from depth_anything_3.bench.utils import create_tsdf_volume, fuse_depth_to_tsdf, sample_points_from_mesh

    gt_data = _load_saved_gt_meta(result_path)
    transform = _load_square_preprocess_transform(result_path.parent.parent / "preprocess_meta.npz")

    square_images: list[np.ndarray] = []
    square_gt_depths: list[np.ndarray] = []
    square_gt_intrinsics: list[np.ndarray] = []

    for image_path, intrinsic in zip(gt_data.image_files, gt_data.intrinsics):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        square_images.append(
            _apply_square_resize_crop(image, transform, interpolation=cv2.INTER_LINEAR)
        )
        square_gt_intrinsics.append(
            apply_square_resize_crop_to_intrinsics(intrinsic, transform).astype(np.float32)
        )
        square_gt_depth, _ = _load_square_gt_depth_and_mask(
            dataset,
            scene,
            Path(image_path).name,
            original_height=original_height,
            original_width=original_width,
            transform=transform,
        )
        square_gt_depths.append(square_gt_depth)

    mesh = fuse_depth_to_tsdf(
        create_tsdf_volume(
            voxel_length=dataset.voxel_length,
            sdf_trunc=dataset.sdf_trunc,
        ),
        np.stack(square_gt_depths, axis=0).astype(np.float32),
        np.stack(square_images, axis=0),
        np.stack(square_gt_intrinsics, axis=0).astype(np.float32),
        np.asarray(gt_data.extrinsics, dtype=np.float32),
        max_depth=dataset.max_depth,
    )
    return sample_points_from_mesh(mesh, dataset.sampling_number)


def _fuse_stitched_square_scene(
    dataset,
    scene: str,
    result_path: Path,
    fuse_path: Path,
    mode: str,
    *,
    depth_scale_shift_alignment: bool = False,
) -> None:
    from depth_anything_3.bench.utils import (
        create_tsdf_volume,
        fuse_depth_to_tsdf,
        sample_points_from_mesh,
        write_point_cloud_with_cameras,
    )
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    pred_data = SimpleNamespace(**{key: value for key, value in np.load(result_path).items()})
    gt_data = _load_saved_gt_meta(result_path)
    transform = _load_square_preprocess_transform(result_path.parent.parent / "preprocess_meta.npz")

    square_images: list[np.ndarray] = []
    square_gt_intrinsics: list[np.ndarray] = []
    square_gt_depths: list[np.ndarray] = []
    square_zero_masks: list[np.ndarray] = []
    original_sizes: list[tuple[int, int]] = []

    for image_path, intrinsic in zip(gt_data.image_files, gt_data.intrinsics):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        original_sizes.append((original_height, original_width))
        square_images.append(
            _apply_square_resize_crop(image, transform, interpolation=cv2.INTER_LINEAR)
        )
        square_gt_intrinsics.append(
            apply_square_resize_crop_to_intrinsics(intrinsic, transform).astype(np.float32)
        )
        square_gt_depth, square_zero_mask = _load_square_gt_depth_and_mask(
            dataset,
            scene,
            Path(image_path).name,
            original_height=original_height,
            original_width=original_width,
            transform=transform,
        )
        square_gt_depths.append(square_gt_depth)
        square_zero_masks.append(square_zero_mask)

    if mode == "recon_unposed":
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        intrinsics = np.asarray(pred_data.intrinsics, dtype=np.float32)
    elif mode == "recon_posed":
        depth_scale_extrinsics = _select_depth_scale_extrinsics_for_fusion(
            result_path,
            pred_data.extrinsics,
            mode=mode,
        )
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            depth_scale_extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )
        extrinsics = gt_data.extrinsics.copy()
        intrinsics = np.stack(square_gt_intrinsics, axis=0).astype(np.float32)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    depth_alignment_scale = float(scale)
    depth_alignment_shift = 0.0
    use_depth_scale_shift_alignment = bool(depth_scale_shift_alignment)
    if use_depth_scale_shift_alignment:
        valid_pixel_count = int(
            sum(
                np.logical_and.reduce(
                    (
                        np.asarray(mask, dtype=bool),
                        np.isfinite(np.asarray(depth, dtype=np.float32)),
                        np.asarray(depth, dtype=np.float32) > 1e-6,
                        np.asarray(gt_depth, dtype=np.float32) > 1e-6,
                    )
                ).sum()
                for depth, gt_depth, mask in zip(pred_data.depth, square_gt_depths, square_zero_masks)
            )
        )
        if valid_pixel_count < 2:
            logger.warning(
                "ETH3D square depth scale+shift alignment found no valid GT samples for scene {} mode {}; falling back to Umeyama scale-only.",
                scene,
                mode,
            )
            use_depth_scale_shift_alignment = False
        else:
            depth_alignment_scale, depth_alignment_shift = _fit_sequence_depth_scale_shift_alignment(
                [np.asarray(depth, dtype=np.float32) for depth in pred_data.depth],
                square_gt_depths,
                square_zero_masks,
            )
            logger.info(
                "ETH3D square depth scale+shift alignment for scene {} mode {}: scale={}, shift={}, valid_pixels={}",
                scene,
                mode,
                depth_alignment_scale,
                depth_alignment_shift,
                valid_pixel_count,
            )

    depths_out = []
    for depth, zero_mask in zip(pred_data.depth, square_zero_masks):
        depth = np.asarray(depth, dtype=np.float32).copy()
        if use_depth_scale_shift_alignment:
            valid_mask = np.asarray(zero_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
            depth = _apply_depth_scale_shift_alignment(
                depth,
                valid_mask,
                scale=depth_alignment_scale,
                shift=depth_alignment_shift,
            )
        else:
            pred_invalid = np.isnan(depth) | np.isinf(depth)
            combined_mask = np.logical_and(zero_mask, np.logical_not(pred_invalid))
            depth = depth * combined_mask.astype(np.float32)
            depth = depth * float(scale)
        depths_out.append(depth)

    mesh = fuse_depth_to_tsdf(
        create_tsdf_volume(
            voxel_length=dataset.voxel_length,
            sdf_trunc=dataset.sdf_trunc,
        ),
        np.stack(depths_out, axis=0),
        np.stack(square_images, axis=0),
        intrinsics,
        extrinsics,
        max_depth=dataset.max_depth,
    )
    pcd = sample_points_from_mesh(mesh, dataset.sampling_number)
    fuse_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(fuse_path), pcd)
    write_point_cloud_with_cameras(
        fuse_path=str(fuse_path),
        pcd=pcd,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_sizes=[image.shape[:2] for image in square_images],
    )


def _evaluate_square_reconstruction(
    dataset,
    scene: str,
    result_path: Path,
    fuse_path: Path,
    mode: str,
    *,
    gt_square_pcd=None,
    depth_scale_shift_alignment: bool = False,
) -> dict[str, float]:
    from depth_anything_3.bench.utils import evaluate_3d_reconstruction

    _fuse_stitched_square_scene(
        dataset,
        scene,
        result_path,
        fuse_path,
        mode,
        depth_scale_shift_alignment=depth_scale_shift_alignment,
    )
    pred_pcd = o3d.io.read_point_cloud(str(fuse_path))
    if gt_square_pcd is None:
        gt_square_pcd = _build_square_gt_point_cloud(dataset, scene, result_path)
    return _to_float_dict(
        evaluate_3d_reconstruction(
            pred_pcd,
            gt_square_pcd,
            threshold=dataset.eval_threshold,
            down_sample=dataset.down_sample,
        )
    )


def compute_raw_camera_scale_metrics(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
    *,
    eps: float = 1e-6,
) -> dict[str, float]:
    from depth_anything_3.utils.geometry import affine_inverse_np, as_homogeneous
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    pred_ext = as_homogeneous(np.asarray(pred_extrinsics, dtype=np.float32))
    gt_ext = as_homogeneous(np.asarray(gt_extrinsics, dtype=np.float32))
    _, _, pred_to_gt_scale, _ = align_poses_umeyama(
        gt_ext,
        pred_ext,
        return_aligned=True,
        ransac=len(gt_ext) >= 10,
        random_state=42,
    )

    pred_first_inv = affine_inverse_np(pred_ext[:1])[0]
    gt_first_inv = affine_inverse_np(gt_ext[:1])[0]
    pred_rel = pred_ext @ pred_first_inv
    gt_rel = gt_ext @ gt_first_inv
    pred_centers = affine_inverse_np(pred_rel)[:, :3, 3]
    gt_centers = affine_inverse_np(gt_rel)[:, :3, 3]

    pred_dists = np.linalg.norm(pred_centers, axis=-1)
    gt_dists = np.linalg.norm(gt_centers, axis=-1)
    valid = gt_dists > eps
    if valid.any():
        ratios = pred_dists[valid] / gt_dists[valid]
    else:
        ratios = np.array([float("inf")], dtype=np.float32)

    return {
        "pred_to_gt_umeyama_scale": float(pred_to_gt_scale),
        "pred_camera_center_dist_mean": float(pred_dists[valid].mean()) if valid.any() else 0.0,
        "pred_camera_center_dist_median": float(np.median(pred_dists[valid])) if valid.any() else 0.0,
        "gt_camera_center_dist_mean": float(gt_dists[valid].mean()) if valid.any() else 0.0,
        "gt_camera_center_dist_median": float(np.median(gt_dists[valid])) if valid.any() else 0.0,
        "camera_center_dist_ratio_mean": float(ratios.mean()),
        "camera_center_dist_ratio_median": float(np.median(ratios)),
        "camera_center_dist_ratio_min": float(ratios.min()),
        "camera_center_dist_ratio_max": float(ratios.max()),
    }


def evaluate_raw_camera_diagnostics_work_dir(
    work_dir: str | Path,
    scenes: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    work_dir = Path(work_dir)
    raw_camera_pose_unposed: dict[str, dict[str, float]] = {}
    raw_camera_pose_posed: dict[str, dict[str, float]] = {}
    raw_camera_scale_unposed: dict[str, dict[str, float]] = {}
    raw_camera_scale_posed: dict[str, dict[str, float]] = {}

    for scene in scenes:
        base_dir = work_dir / "model_results" / "eth3d" / scene
        gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)
        gt_extrinsics = np.asarray(gt_meta["extrinsics"], dtype=np.float32)

        unposed_raw_path = base_dir / "unposed" / "exports" / "raw_camera_params.npz"
        posed_raw_path = base_dir / "posed" / "exports" / "raw_camera_params.npz"

        raw_camera_pose_unposed[scene] = compute_pose_with_gt(
            result_path=unposed_raw_path,
            gt_extrinsics=gt_extrinsics,
        )
        raw_camera_pose_posed[scene] = compute_pose_with_gt(
            result_path=posed_raw_path,
            gt_extrinsics=gt_extrinsics,
        )

        unposed_raw = np.load(unposed_raw_path)
        posed_raw = np.load(posed_raw_path)
        raw_camera_scale_unposed[scene] = compute_raw_camera_scale_metrics(
            unposed_raw["extrinsics"],
            gt_extrinsics,
        )
        raw_camera_scale_posed[scene] = compute_raw_camera_scale_metrics(
            posed_raw["extrinsics"],
            gt_extrinsics,
        )

    raw_camera_pose_unposed["mean"] = mean_of_dicts(list(raw_camera_pose_unposed.values()))
    raw_camera_pose_posed["mean"] = mean_of_dicts(list(raw_camera_pose_posed.values()))
    raw_camera_scale_unposed["mean"] = mean_of_dicts(list(raw_camera_scale_unposed.values()))
    raw_camera_scale_posed["mean"] = mean_of_dicts(list(raw_camera_scale_posed.values()))

    metric_dir = work_dir / "metric_results"
    dump_json(metric_dir / "eth3d_raw_camera_pose_unposed.json", raw_camera_pose_unposed)
    dump_json(metric_dir / "eth3d_raw_camera_pose_posed.json", raw_camera_pose_posed)
    dump_json(metric_dir / "eth3d_raw_camera_scale_unposed.json", raw_camera_scale_unposed)
    dump_json(metric_dir / "eth3d_raw_camera_scale_posed.json", raw_camera_scale_posed)

    return {
        "raw_camera_pose_unposed": raw_camera_pose_unposed,
        "raw_camera_pose_posed": raw_camera_pose_posed,
        "raw_camera_scale_unposed": raw_camera_scale_unposed,
        "raw_camera_scale_posed": raw_camera_scale_posed,
    }


def evaluate_stitched_square_eth3d_work_dir(
    work_dir: str | Path,
    dataset,
    scenes: list[str],
    *,
    depth_scale_shift_alignment: bool = False,
    aligned_pose_diagnostics: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    work_dir = Path(work_dir)
    pose_results: dict[str, dict[str, float]] = {}
    recon_unposed_results: dict[str, dict[str, float]] = {}
    recon_posed_results: dict[str, dict[str, float]] = {}
    aligned_pose_results: dict[str, dict[str, float]] = {}

    for scene in scenes:
        base_dir = work_dir / "model_results" / "eth3d" / scene
        unposed_result_path = base_dir / "unposed" / "exports" / "mini_npz" / "results.npz"
        posed_result_path = base_dir / "posed" / "exports" / "mini_npz" / "results.npz"
        gt_meta = np.load(base_dir / "unposed" / "exports" / "gt_meta.npz", allow_pickle=True)
        gt_square_pcd = _build_square_gt_point_cloud(dataset, scene, unposed_result_path)

        pose_results[scene] = compute_pose_with_gt(
            result_path=unposed_result_path,
            gt_extrinsics=gt_meta["extrinsics"],
        )
        if aligned_pose_diagnostics:
            aligned_pose_results[scene] = compute_umeyama_aligned_pose_with_gt(
                result_path=unposed_result_path,
                gt_extrinsics=gt_meta["extrinsics"],
            )

        unposed_fuse_path = base_dir / "unposed" / "exports" / "fuse" / "pcd.ply"
        recon_unposed_results[scene] = _evaluate_square_reconstruction(
            dataset,
            scene,
            unposed_result_path,
            unposed_fuse_path,
            "recon_unposed",
            gt_square_pcd=gt_square_pcd,
            depth_scale_shift_alignment=depth_scale_shift_alignment,
        )

        posed_fuse_path = base_dir / "posed" / "exports" / "fuse" / "pcd.ply"
        recon_posed_results[scene] = _evaluate_square_reconstruction(
            dataset,
            scene,
            posed_result_path,
            posed_fuse_path,
            "recon_posed",
            gt_square_pcd=gt_square_pcd,
            depth_scale_shift_alignment=depth_scale_shift_alignment,
        )

    pose_results["mean"] = mean_of_dicts(list(pose_results.values()))
    recon_unposed_results["mean"] = mean_of_dicts(list(recon_unposed_results.values()))
    recon_posed_results["mean"] = mean_of_dicts(list(recon_posed_results.values()))
    if aligned_pose_diagnostics:
        aligned_pose_results["mean"] = mean_of_dicts(list(aligned_pose_results.values()))

    metric_dir = work_dir / "metric_results"
    dump_json(metric_dir / "eth3d_pose.json", pose_results)
    dump_json(metric_dir / "eth3d_recon_unposed.json", recon_unposed_results)
    dump_json(metric_dir / "eth3d_recon_posed.json", recon_posed_results)
    if aligned_pose_diagnostics:
        dump_json(metric_dir / "aligned_pose.json", aligned_pose_results)

    metrics = {
        "pose": pose_results,
        "recon_unposed": recon_unposed_results,
        "recon_posed": recon_posed_results,
    }
    if aligned_pose_diagnostics:
        metrics["aligned_pose"] = aligned_pose_results
    return metrics


def load_anyview_api(args: argparse.Namespace):
    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    api = DepthAnything3.from_pretrained(args.da3_model_path).to(device)
    api.eval()
    return api


def run_da3_anyview_scene(
    api,
    scene_data,
    process_res: int,
    ref_view_strategy: str,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    input_images: list[Any] | None = None,
) -> dict[str, np.ndarray]:
    prediction, raw_camera_params = _run_anyview_inference_with_raw_camera(
        api,
        scene_data.image_files if input_images is None else input_images,
        process_res=process_res,
        ref_view_strategy=ref_view_strategy,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 anyview inference did not return ray-derived extrinsics/intrinsics.")
    return {
        "depth": prediction.depth.astype(np.float32),
        "conf": None if prediction.conf is None else prediction.conf.astype(np.float32),
        "extrinsics": prediction.extrinsics.astype(np.float32),
        "intrinsics": prediction.intrinsics.astype(np.float32),
        "raw_camera_params": raw_camera_params,
    }


def run_da3_anyview_square_scene(
    api,
    scene_data,
    process_res: int,
    ref_view_strategy: str,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    square_images, transform = _build_square_rgb_inputs(
        scene_data.image_files,
        target_size=process_res,
    )
    square_intrinsics = None
    if extrinsics is not None or intrinsics is not None:
        if extrinsics is None or intrinsics is None:
            raise ValueError("extrinsics and intrinsics must be provided together")
        square_intrinsics = np.stack(
            [apply_square_resize_crop_to_intrinsics(item, transform) for item in intrinsics],
            axis=0,
        ).astype(np.float32)
    prediction, raw_camera_params = _run_anyview_inference_with_raw_camera(
        api,
        square_images,
        process_res=process_res,
        ref_view_strategy=ref_view_strategy,
        extrinsics=None if extrinsics is None else np.asarray(extrinsics, dtype=np.float32),
        intrinsics=square_intrinsics,
    )
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 anyview inference did not return ray-derived extrinsics/intrinsics.")
    preprocess_meta = transform.to_export_dict()
    preprocess_meta["num_frames"] = np.array(len(scene_data.image_files), dtype=np.int32)
    return (
        {
            "depth": prediction.depth.astype(np.float32),
            "conf": None if prediction.conf is None else prediction.conf.astype(np.float32),
            "extrinsics": prediction.extrinsics.astype(np.float32),
            "intrinsics": prediction.intrinsics.astype(np.float32),
            "raw_camera_params": raw_camera_params,
        },
        preprocess_meta,
    )


def build_stitched_loader_args(
    args: argparse.Namespace,
    *,
    da3_process_res_method: str | None = None,
) -> argparse.Namespace:
    from models.stitching_layer_builder import ConvSpec, parse_conv_spec
    from inference_stitched_decode import apply_feedforward_model_inference_defaults

    stitching_layer_config = args.stitching_layer_config
    if not isinstance(stitching_layer_config, ConvSpec):
        stitching_layer_config = parse_conv_spec(stitching_layer_config)

    stitched_args = argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        feedforward_model="da3",
        video_model="wan",
        da3_model_path=args.da3_model_path,
        da3_config_name=args.da3_config_name,
        da3_checkpoint_name=args.da3_checkpoint_name,
        da3_enable_scene_flow_head=getattr(args, "da3_enable_scene_flow_head", False),
        da3_scene_flow_head_checkpoint_path=getattr(
            args,
            "da3_scene_flow_head_checkpoint_path",
            None,
        ),
        da3_scene_flow_preset=getattr(args, "da3_scene_flow_preset", "da3-giant-sceneflow"),
        da3_ref_view_strategy=args.da3_ref_view_strategy,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=stitching_layer_config,
        lora_config=args.lora_config,
        full_finetune_stitched_parts=args.full_finetune_stitched_parts,
        resolution=args.input_resolution,
        feedforward_resolution=args.input_resolution,
        max_frames=args.max_frames,
        device=args.device,
        wan_vae_path=args.wan_vae_path,
        hf_endpoint=args.hf_endpoint,
        anyplat_model_path=None,
        vggt_model_path=None,
        force_fp32_inference=True,
        da3_process_res_method=(
            da3_process_res_method
            if da3_process_res_method is not None
            else getattr(args, "da3_process_res_method", "upper_bound_resize")
        ),
    )
    apply_feedforward_model_inference_defaults(stitched_args)
    return stitched_args


def load_stitched_runner(
    args: argparse.Namespace,
    *,
    da3_process_res_method: str | None = None,
):
    from inference_stitched_decode import load_stitched_decoder

    stitched_args = build_stitched_loader_args(
        args,
        da3_process_res_method=da3_process_res_method,
    )
    return load_stitched_decoder(
        stitched_args,
        preserve_explicit_fields=(
            "da3_ref_view_strategy",
            "resolution",
            "feedforward_resolution",
        ),
    )


@torch.no_grad()
def _run_da3_stitched_decode_for_eth3d(
    args: argparse.Namespace,
    runner,
    prepared: dict[str, Any],
    *,
    camera_extrinsics: torch.Tensor | None = None,
    camera_intrinsics: torch.Tensor | None = None,
) -> dict[str, Any]:
    camera_param_source = getattr(args, "stitched_da3_camera_param_source", "ray_map")
    if (
        getattr(args, "stitched_da3_sky_postprocess", True)
        and camera_param_source == "ray_map"
    ):
        from inference_stitched_decode import run_stitched_decode

        if camera_extrinsics is None:
            return run_stitched_decode(
                args,
                runner,
                prepared,
            )
        return run_stitched_decode(
            args,
            runner,
            prepared,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
        )

    from inference_stitched_decode import decoder_autocast
    da3_stitched_module = importlib.import_module("models.da3_stitched")

    DA3Stitched = da3_stitched_module.DA3Stitched
    canonicalize_da3_geometry_outputs = da3_stitched_module.canonicalize_da3_geometry_outputs
    decode_official_da3_camera_params = da3_stitched_module.decode_official_da3_camera_params
    apply_official_da3_mono_sky_postprocess = getattr(
        da3_stitched_module,
        "apply_official_da3_mono_sky_postprocess",
        None,
    )

    if not isinstance(runner.stitched_3d_model, DA3Stitched):
        raise TypeError("ETH3D stitched sky toggle only supports DA3 stitched models.")

    device = runner.device
    vae_images = prepared["vae_images"].to(device)
    feedforward_images = prepared["feedforward_images"].to(device)
    force_fp32_inference = bool(getattr(args, "force_fp32_inference", True))

    with decoder_autocast(device, force_fp32_inference=force_fp32_inference):
        latents, _ = runner.vae_encoder_forward(vae_images, decode=False)
        decoder_kwargs = {
            "feedforward_image": feedforward_images,
            "train": True,
            "camera_extrinsics": None if camera_extrinsics is None else camera_extrinsics.to(device),
            "camera_intrinsics": None if camera_intrinsics is None else camera_intrinsics.to(device),
        }
        if prepared.get("image_hw") is not None:
            decoder_kwargs["image_hw"] = prepared["image_hw"]
        stitched_outputs = runner.forward_with_latent(
            latents,
            **decoder_kwargs,
        )

    outputs = canonicalize_da3_geometry_outputs(stitched_outputs)
    if getattr(args, "stitched_da3_sky_postprocess", True):
        if apply_official_da3_mono_sky_postprocess is None:
            raise ImportError(
                "apply_official_da3_mono_sky_postprocess is required when stitched_da3_sky_postprocess=True"
            )
        outputs = apply_official_da3_mono_sky_postprocess(outputs)
    if camera_param_source == "ray_map":
        camera_params = decode_official_da3_camera_params(
            outputs["ray"],
            outputs["ray_conf"],
            image_height=feedforward_images.shape[-2],
            image_width=feedforward_images.shape[-1],
        )
    elif camera_param_source == "camera_token":
        camera_params = _build_camera_params_from_token_outputs(
            stitched_outputs,
            image_height=feedforward_images.shape[-2],
            image_width=feedforward_images.shape[-1],
        )
    else:
        raise ValueError(
            f"Unsupported stitched_da3_camera_param_source: {camera_param_source}"
        )
    results = {
        "latents": latents.detach().cpu(),
        "depth": outputs["depth"].detach().cpu(),
        "depth_conf": outputs["depth_conf"].detach().cpu(),
        "ray": outputs["ray"].detach().cpu(),
        "ray_conf": outputs["ray_conf"].detach().cpu(),
        "camera_params": {
            key: value.detach().cpu()
            for key, value in camera_params.items()
        },
        "camera_param_source": camera_param_source,
    }
    if "sky" in outputs:
        results["sky"] = outputs["sky"].detach().cpu()
    if "scene_flow" in outputs:
        results["scene_flow"] = outputs["scene_flow"].detach().cpu()
    if "scene_flow_cam" in outputs:
        results["scene_flow_cam"] = outputs["scene_flow_cam"].detach().cpu()
    return results


def _extract_scene_flow_output_np(
    outputs: dict[str, Any],
    key: str,
    *,
    num_frames: int,
) -> np.ndarray | None:
    flow = outputs.get(key)
    if flow is None:
        return None
    if not torch.is_tensor(flow):
        flow = torch.as_tensor(flow)
    flow = flow.squeeze(0)[:num_frames].detach().cpu()
    if flow.ndim != 4:
        raise ValueError(f"Expected {key} scene-flow output with 4 dims after batch squeeze, got {tuple(flow.shape)}")
    if flow.shape[1] == 3:
        return flow.numpy().astype(np.float32)
    if flow.shape[-1] == 3:
        return flow.permute(0, 3, 1, 2).contiguous().numpy().astype(np.float32)
    raise ValueError(f"Expected {key} scene-flow output channel dimension of 3, got {tuple(flow.shape)}")


def run_da3_stitched_scene(
    runner,
    scene_data,
    args: argparse.Namespace,
    *,
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    process_res_method: str | None = None,
    input_images: list[Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from inference_stitched_decode import prepare_model_inputs_from_frames
    from models.da3_stitched import normalize_extrinsics_like_official_da3_api

    process_res_method = process_res_method or getattr(
        args,
        "da3_process_res_method",
        "upper_bound_resize",
    )
    frames = load_rgb_frames(scene_data.image_files if input_images is None else input_images)
    frames, temporal_pad = pad_frames_for_stitched_inference(frames)
    original_num_frames = temporal_pad.original_frames
    original_height, original_width = frames.shape[-2:]
    if process_res_method == "upper_bound_resize":
        transform = build_upper_bound_resize_transform(
            height=original_height,
            width=original_width,
            target_size=args.input_resolution,
        )
    elif process_res_method == "square_center_crop":
        transform = build_square_resize_crop_transform(
            height=original_height,
            width=original_width,
            target_size=args.input_resolution,
        )
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {process_res_method}")
    prepared = prepare_model_inputs_from_frames(
        frames,
        vae_resolution=args.input_resolution,
        feedforward_resolution=args.input_resolution,
        da3_process_res_method=process_res_method,
    )
    stitched_args = SimpleNamespace(
        config=None,
        feedforward_resolution=args.input_resolution,
        force_fp32_inference=True,
        stitched_da3_sky_postprocess=getattr(args, "stitched_da3_sky_postprocess", True),
    )
    camera_extrinsics = None
    camera_intrinsics = None
    input_camera_extrinsics = None
    input_camera_intrinsics = None
    if extrinsics is not None or intrinsics is not None:
        if extrinsics is None or intrinsics is None:
            raise ValueError("extrinsics and intrinsics must be provided together")
        if process_res_method == "upper_bound_resize":
            resized_height, resized_width = prepared.get(
                "image_hw",
                (transform.resized_height, transform.resized_width),
            )
            processed_intrinsics = np.stack(
                [
                    apply_resize_only_to_intrinsics(
                        item,
                        original_height=original_height,
                        original_width=original_width,
                        resized_height=resized_height,
                        resized_width=resized_width,
                    )
                    for item in intrinsics
                ],
                axis=0,
            ).astype(np.float32)
        else:
            processed_intrinsics = np.stack(
                [apply_square_resize_crop_to_intrinsics(item, transform) for item in intrinsics],
                axis=0,
            ).astype(np.float32)
        input_camera_extrinsics = torch.from_numpy(np.asarray(extrinsics, dtype=np.float32)).unsqueeze(0)
        input_camera_intrinsics = torch.from_numpy(processed_intrinsics).unsqueeze(0)
        normalized_camera_extrinsics = (
            normalize_extrinsics_like_official_da3_api(input_camera_extrinsics.clone())
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        camera_extrinsics = torch.from_numpy(
            _pad_camera_sequence_for_stitched_inference(
                normalized_camera_extrinsics,
                temporal_pad.padded_frames,
                pad_source_indices=temporal_pad.pad_source_indices,
            )
        ).unsqueeze(0)
        camera_intrinsics = torch.from_numpy(
            _pad_camera_sequence_for_stitched_inference(
                processed_intrinsics,
                temporal_pad.padded_frames,
                pad_source_indices=temporal_pad.pad_source_indices,
            )
        ).unsqueeze(0)

    outputs = _run_da3_stitched_decode_for_eth3d(
        stitched_args,
        runner,
        prepared,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
    )
    train_style_eval_alignment_enabled = bool(
        getattr(args, "stitched_da3_train_style_eval_alignment", False)
    )
    if train_style_eval_alignment_enabled:
        outputs = _apply_train_style_gt_alignment_to_stitched_outputs(
            outputs,
            scene_data,
            args=args,
            process_res_method=process_res_method,
            transform=transform,
            original_num_frames=original_num_frames,
        )
    camera_params = outputs.get("camera_params")
    camera_param_source = outputs.get(
        "camera_param_source",
        getattr(args, "stitched_da3_camera_param_source", "ray_map"),
    )
    if camera_params is None:
        raise RuntimeError(
            f"Stitched inference did not return {camera_param_source}-derived camera_params."
        )
    raw_camera_params = _camera_params_to_numpy(
        camera_params,
        num_frames=original_num_frames,
    )

    prediction_depth = outputs["depth"].squeeze(0)[:original_num_frames]
    if prediction_depth.dim() == 4 and prediction_depth.shape[1] == 1:
        prediction_depth = prediction_depth.squeeze(1)
    prediction_depth = prediction_depth.cpu().numpy().astype(np.float32)
    prediction_conf = outputs.get("depth_conf")
    if prediction_conf is not None:
        prediction_conf = prediction_conf.squeeze(0)[:original_num_frames]
        if prediction_conf.dim() == 4 and prediction_conf.shape[1] == 1:
            prediction_conf = prediction_conf.squeeze(1)
        prediction_conf = prediction_conf.cpu().numpy().astype(np.float32)
    prediction_scene_flow = _extract_scene_flow_output_np(
        outputs,
        "scene_flow",
        num_frames=original_num_frames,
    )
    prediction_scene_flow_cam = _extract_scene_flow_output_np(
        outputs,
        "scene_flow_cam",
        num_frames=original_num_frames,
    )

    if input_camera_extrinsics is not None and input_camera_intrinsics is not None:
        if train_style_eval_alignment_enabled:
            prediction = {
                "depth": prediction_depth,
                "conf": prediction_conf,
                "extrinsics": input_camera_extrinsics.squeeze(0)[:original_num_frames, :3, :].detach().cpu().numpy().astype(np.float32),
                "intrinsics": input_camera_intrinsics.squeeze(0)[:original_num_frames].detach().cpu().numpy().astype(np.float32),
                "raw_camera_params": raw_camera_params,
            }
        else:
            aligned = _align_prediction_to_input_camera_scale(
                depth=outputs["depth"][:, :original_num_frames],
                pred_extrinsics=camera_params["extrinsics"][:, :original_num_frames],
                input_extrinsics=input_camera_extrinsics[:, :original_num_frames],
                input_intrinsics=input_camera_intrinsics[:, :original_num_frames],
            )
            prediction = {
                "depth": aligned["depth"][:, 0] if aligned["depth"].ndim == 4 and aligned["depth"].shape[1] == 1 else aligned["depth"],
                "conf": prediction_conf,
                "extrinsics": aligned["extrinsics"],
                "intrinsics": aligned["intrinsics"],
                "raw_camera_params": raw_camera_params,
            }
    else:
        prediction = {
            "depth": prediction_depth,
            "conf": prediction_conf,
            "extrinsics": camera_params["extrinsics"].squeeze(0)[:original_num_frames].detach().cpu().numpy().astype(np.float32),
            "intrinsics": camera_params["intrinsics"].squeeze(0)[:original_num_frames].detach().cpu().numpy().astype(np.float32),
            "raw_camera_params": raw_camera_params,
        }
    if prediction_scene_flow is not None:
        prediction["scene_flow"] = prediction_scene_flow
    if prediction_scene_flow_cam is not None:
        prediction["scene_flow_cam"] = prediction_scene_flow_cam
    preprocess_meta = transform.to_export_dict()
    preprocess_meta["num_frames"] = np.array(original_num_frames, dtype=np.int32)
    preprocess_meta.update(temporal_pad.to_export_dict())
    return prediction, preprocess_meta


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    scenes: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    from depth_anything_3.bench.datasets.eth3d import ETH3D

    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)

    effective_args = SimpleNamespace(**vars(args))
    if effective_args.input_resolution is None:
        if model_kind in {"da3_stitched", "da3_stitched_square"}:
            effective_args.input_resolution = DEFAULT_STITCHED_INPUT_RESOLUTION
        else:
            effective_args.input_resolution = DEFAULT_INPUT_RESOLUTION

    dataset = ETH3D()
    dataset.data_root = effective_args.eth3d_root

    if model_kind == "da3_anyview":
        runner = load_anyview_api(effective_args)
        run_scene = lambda scene_data: (  # noqa: E731
            run_da3_anyview_scene(
                runner,
                scene_data=scene_data,
                process_res=effective_args.input_resolution,
                ref_view_strategy=effective_args.da3_ref_view_strategy,
            ),
            run_da3_anyview_scene(
                runner,
                scene_data=scene_data,
                process_res=effective_args.input_resolution,
                ref_view_strategy=effective_args.da3_ref_view_strategy,
                extrinsics=scene_data.extrinsics,
                intrinsics=scene_data.intrinsics,
            ),
            None,
        )
    elif model_kind == "da3_anyview_square":
        runner = load_anyview_api(effective_args)

        def run_scene(scene_data):
            unposed_prediction, preprocess_meta = run_da3_anyview_square_scene(
                runner,
                scene_data=scene_data,
                process_res=effective_args.input_resolution,
                ref_view_strategy=effective_args.da3_ref_view_strategy,
            )
            posed_prediction, _ = run_da3_anyview_square_scene(
                runner,
                scene_data=scene_data,
                process_res=effective_args.input_resolution,
                ref_view_strategy=effective_args.da3_ref_view_strategy,
                extrinsics=scene_data.extrinsics,
                intrinsics=scene_data.intrinsics,
            )
            return unposed_prediction, posed_prediction, preprocess_meta
    elif model_kind == "da3_stitched_square":
        runner = load_stitched_runner(
            effective_args,
            da3_process_res_method="square_center_crop",
        )

        def run_scene(scene_data):
            unposed_prediction, preprocess_meta = run_da3_stitched_scene(
                runner,
                scene_data=scene_data,
                args=effective_args,
                process_res_method="square_center_crop",
            )
            if getattr(effective_args, "stitched_da3_posed_rerun_with_gt_camera", True):
                posed_prediction, _ = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=effective_args,
                    extrinsics=scene_data.extrinsics,
                    intrinsics=scene_data.intrinsics,
                    process_res_method="square_center_crop",
                )
            else:
                posed_prediction = clone_prediction_dict(unposed_prediction)
            return unposed_prediction, posed_prediction, preprocess_meta
    else:
        runner = load_stitched_runner(
            effective_args,
            da3_process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
        )

        def run_scene(scene_data):
            unposed_prediction, preprocess_meta = run_da3_stitched_scene(
                runner,
                scene_data=scene_data,
                args=effective_args,
                process_res_method=getattr(effective_args, "da3_process_res_method", "upper_bound_resize"),
            )
            if getattr(effective_args, "stitched_da3_posed_rerun_with_gt_camera", True):
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
            return unposed_prediction, posed_prediction, preprocess_meta

    for scene in scenes:
        full_scene_data = dataset.get_data(scene)
        scene_max_frames = effective_args.max_frames
        input_order_strategy = "dataset"
        if model_kind in {"da3_stitched", "da3_stitched_square"}:
            scene_max_frames = resolve_stitched_eval_max_frames(
                total_frames=len(full_scene_data.image_files),
                requested_max_frames=effective_args.max_frames,
            )
            input_order_strategy = getattr(
                effective_args,
                "stitched_da3_input_order_strategy",
                DEFAULT_STITCHED_INPUT_ORDER_STRATEGY,
            )
        scene_data = sample_scene_data(
            full_scene_data,
            scene_max_frames,
            order_strategy=input_order_strategy,
            disable_middle_frame_filter=getattr(
                effective_args,
                "eth3d_disable_middle_frame_filter",
                False,
            ),
        )
        logger.info(
            "Evaluating {} scene {} with {} frames using {} input order; middle-frame filter disabled={}",
            model_kind,
            scene,
            len(scene_data.image_files),
            input_order_strategy,
            getattr(effective_args, "eth3d_disable_middle_frame_filter", False),
        )
        unposed_prediction, posed_prediction, preprocess_meta = run_scene(scene_data)
        write_eth3d_scene_exports(
            work_dir=output_root,
            scene_name=scene,
            unposed_prediction=unposed_prediction,
            posed_prediction=posed_prediction,
            scene_data=scene_data,
            preprocess_meta=preprocess_meta,
        )

    if "runner" in locals():
        del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    square_input_eval = model_kind in {"da3_anyview_square", "da3_stitched_square"}
    depth_scale_shift_alignment = bool(
        getattr(effective_args, "eth3d_depth_scale_shift_alignment", False)
    )
    if square_input_eval:
        if depth_scale_shift_alignment:
            metrics = evaluate_stitched_square_eth3d_work_dir(
                output_root,
                dataset,
                scenes,
                depth_scale_shift_alignment=True,
                aligned_pose_diagnostics=True,
            )
        else:
            metrics = evaluate_stitched_square_eth3d_work_dir(output_root, dataset, scenes)
    else:
        if depth_scale_shift_alignment:
            metrics = evaluate_eth3d_work_dir(
                output_root,
                dataset,
                scenes,
                depth_scale_shift_alignment=True,
                aligned_pose_diagnostics=True,
            )
        else:
            metrics = evaluate_eth3d_work_dir(output_root, dataset, scenes)
    if model_kind in {"da3_anyview_square", "da3_stitched", "da3_stitched_square"}:
        metrics.update(evaluate_raw_camera_diagnostics_work_dir(output_root, scenes))
    if model_kind == "da3_stitched_square":
        effective_process_res_method = "square_center_crop"
    else:
        effective_process_res_method = getattr(effective_args, "da3_process_res_method", "upper_bound_resize")
    dump_json(
        output_root / "run_meta.json",
        {
            "model_kind": model_kind,
            "eth3d_root": str(Path(effective_args.eth3d_root).resolve()),
            "input_resolution": effective_args.input_resolution,
            "da3_ref_view_strategy": effective_args.da3_ref_view_strategy,
            "da3_process_res_method": effective_process_res_method,
            "stitched_da3_sky_postprocess": getattr(effective_args, "stitched_da3_sky_postprocess", True),
            "stitched_da3_posed_rerun_with_gt_camera": getattr(
                effective_args,
                "stitched_da3_posed_rerun_with_gt_camera",
                True,
            ),
            "stitched_da3_train_style_eval_alignment": getattr(
                effective_args,
                "stitched_da3_train_style_eval_alignment",
                False,
            ),
            "eth3d_depth_scale_shift_alignment": depth_scale_shift_alignment,
            "eth3d_disable_middle_frame_filter": getattr(
                effective_args,
                "eth3d_disable_middle_frame_filter",
                False,
            ),
            "stitched_da3_camera_param_source": getattr(
                effective_args,
                "stitched_da3_camera_param_source",
                "ray_map",
            ),
            "stitched_da3_input_order_strategy": getattr(
                effective_args,
                "stitched_da3_input_order_strategy",
                DEFAULT_STITCHED_INPUT_ORDER_STRATEGY,
            ),
            "effective_stitched_da3_camera_param_source": (
                "ray_map_train_style_eval_alignment"
                if getattr(effective_args, "stitched_da3_train_style_eval_alignment", False)
                else getattr(effective_args, "stitched_da3_camera_param_source", "ray_map")
            ),
            "max_frames": effective_args.max_frames,
            "device": effective_args.device,
            "ray_pose_only": (
                model_kind in {"da3_anyview", "da3_anyview_square"}
                or getattr(effective_args, "stitched_da3_train_style_eval_alignment", False)
                or getattr(effective_args, "stitched_da3_camera_param_source", "ray_map") == "ray_map"
            ),
            "force_fp32_inference": True,
            "square_input_eval": square_input_eval,
            "scenes": scenes,
        },
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)

    available_scenes = list_eth3d_scenes(args.eth3d_root)
    if args.scenes:
        scenes = [scene for scene in available_scenes if scene in set(args.scenes)]
    else:
        scenes = available_scenes
    if not scenes:
        raise ValueError(f"No ETH3D scenes selected under {args.eth3d_root}")

    model_kinds = ["da3_anyview", "da3_stitched"] if args.model_kind == "both" else [args.model_kind]
    for model_kind in model_kinds:
        logger.info("Running ETH3D benchmark for {} on {} scenes", model_kind, len(scenes))
        evaluate_model_kind(args, model_kind=model_kind, scenes=scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
