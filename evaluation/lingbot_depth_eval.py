
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT

for import_path in (PROJECT_ROOT, PLAN4D_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils.da3_resize import resize_video_frames_upper_bound


RGB_SUFFIX = "_rgb.left.jpg"
GT_DEPTH_SUFFIX = "_depth_left.png"
DEFAULT_INPUT_RESOLUTION = 532


@dataclass(frozen=True)
class LingBotUnit:
    scene_id: str
    cam_id: str
    unit_dir: Path
    intrinsics_path: Path | None
    rgb_paths: tuple[Path, ...]
    depth_paths: tuple[Path, ...]


@dataclass(frozen=True)
class SquareROI:
    top: int
    left: int
    size: int


def default_dataset_root() -> str:
    return str(
        (
            CODE_ROOT
            / "dataset"
            / "LingBot-Depth-Dataset"
            / "extracted"
            / "lingbot-depth_pub"
            / "RobbySimVal"
            / "val_view"
        ).resolve()
    )


def default_da3_model_path() -> str:
    return str((PLAN4D_ROOT / "checkpoints" / "da3").resolve())


def default_wan_vae_path() -> str:
    return str(
        (PLAN4D_ROOT / "checkpoints" / "longcat" / "vae").resolve()
    )


def _to_serializable(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dim() == 0:
            return float(value.item())
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_serializable(payload), ensure_ascii=False) + "\n")
        handle.flush()


def resize_shorter_crop_square_batch(
    images: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] images, got {tuple(images.shape)}")
    _, _, height, width = images.shape
    scale = target_size / min(height, width)
    resized_h = round(height * scale)
    resized_w = round(width * scale)
    resized = F.interpolate(
        images,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )

    top = max(0, (resized_h - target_size) // 2)
    left = max(0, (resized_w - target_size) // 2)
    return resized[:, :, top : top + target_size, left : left + target_size]


def normalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[2] == 1:
        depth = depth.squeeze(2)
    if depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.dim() != 4:
        raise ValueError(f"Expected [B,T,H,W] or [B,T,H,W,1], got {tuple(depth.shape)}")
    return depth.to(dtype=torch.float32)


def align_depth_scale_shift(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    pred_depth = normalize_depth_tensor(pred_depth)
    gt_depth = normalize_depth_tensor(gt_depth)
    valid_mask = normalize_depth_tensor(valid_mask)

    gt_height, gt_width = gt_depth.shape[-2:]
    if pred_depth.shape[-2:] != (gt_height, gt_width):
        pred_depth = F.interpolate(
            pred_depth.reshape(-1, 1, pred_depth.shape[-2], pred_depth.shape[-1]),
            size=(gt_height, gt_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(pred_depth.shape[0], pred_depth.shape[1], gt_height, gt_width)

    if valid_mask.shape[-2:] != (gt_height, gt_width):
        valid_mask = F.interpolate(
            valid_mask.reshape(-1, 1, valid_mask.shape[-2], valid_mask.shape[-1]),
            size=(gt_height, gt_width),
            mode="nearest",
        ).reshape(valid_mask.shape[0], valid_mask.shape[1], gt_height, gt_width)

    aligned = pred_depth.clone()
    for batch_idx in range(pred_depth.shape[0]):
        for frame_idx in range(pred_depth.shape[1]):
            valid = valid_mask[batch_idx, frame_idx] > 0
            if valid.sum() < 2:
                continue
            pred = pred_depth[batch_idx, frame_idx][valid]
            gt = gt_depth[batch_idx, frame_idx][valid]
            pred_mean = pred.mean()
            gt_mean = gt.mean()
            pred_centered = pred - pred_mean
            gt_centered = gt - gt_mean
            scale = ((pred_centered * gt_centered).sum() / ((pred_centered**2).sum() + 1e-8)).clamp(
                min=0.01,
                max=100.0,
            )
            shift = gt_mean - scale * pred_mean
            aligned[batch_idx, frame_idx] = scale * pred_depth[batch_idx, frame_idx] + shift
    return aligned


def compute_depth_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    pred_depth = normalize_depth_tensor(pred_depth)
    gt_depth = normalize_depth_tensor(gt_depth)
    valid_mask = normalize_depth_tensor(valid_mask)

    gt_height, gt_width = gt_depth.shape[-2:]
    if pred_depth.shape[-2:] != (gt_height, gt_width):
        pred_depth = F.interpolate(
            pred_depth.reshape(-1, 1, pred_depth.shape[-2], pred_depth.shape[-1]),
            size=(gt_height, gt_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(pred_depth.shape[0], pred_depth.shape[1], gt_height, gt_width)

    if valid_mask.shape[-2:] != (gt_height, gt_width):
        valid_mask = F.interpolate(
            valid_mask.reshape(-1, 1, valid_mask.shape[-2], valid_mask.shape[-1]),
            size=(gt_height, gt_width),
            mode="nearest",
        ).reshape(valid_mask.shape[0], valid_mask.shape[1], gt_height, gt_width)

    valid = valid_mask.bool().flatten()
    pred = pred_depth.flatten()[valid].clamp(min=1e-6)
    gt = gt_depth.flatten()[valid].clamp(min=1e-6)
    if pred.numel() == 0:
        return {
            "depth_abs_rel": 0.0,
            "depth_sq_rel": 0.0,
            "depth_rmse": 0.0,
            "depth_rmse_log": 0.0,
            "depth_a1": 0.0,
            "depth_a2": 0.0,
            "depth_a3": 0.0,
        }

    thresh = torch.max(pred / gt, gt / pred)
    return {
        "depth_abs_rel": (torch.abs(pred - gt) / gt).mean().item(),
        "depth_sq_rel": (((pred - gt) ** 2) / gt).mean().item(),
        "depth_rmse": torch.sqrt(((pred - gt) ** 2).mean()).item(),
        "depth_rmse_log": torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean()).item(),
        "depth_a1": (thresh < 1.25).float().mean().item(),
        "depth_a2": (thresh < 1.25**2).float().mean().item(),
        "depth_a3": (thresh < 1.25**3).float().mean().item(),
    }


def aggregate_metrics(metric_history: dict[str, list[float]]) -> dict[str, float]:
    return {
        metric_name: float(sum(values) / len(values)) if values else 0.0
        for metric_name, values in metric_history.items()
    }


def _tensor_to_uint8_video(frames: torch.Tensor) -> list[Any]:
    frames = frames.detach().cpu().to(dtype=torch.float32)
    if frames.ndim == 3:
        frames = frames.unsqueeze(1)
    if frames.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] or [T,H,W], got {tuple(frames.shape)}")
    if frames.shape[1] == 1:
        frames = frames.repeat(1, 3, 1, 1)
    frames = frames.clamp(0.0, 1.0)
    return [
        (frame.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
        for frame in frames
    ]


def _safe_quantile(values: torch.Tensor, q: float) -> float:
    try:
        quantile = torch.quantile(values, q)
    except RuntimeError as exc:
        if "quantile() input tensor is too large" not in str(exc):
            raise
        quantile = float(np.quantile(values.detach().cpu().numpy(), q))
    if hasattr(quantile, "item"):
        return float(quantile.item())
    return float(quantile)


def _colorize_depth_sequence(depth: torch.Tensor) -> torch.Tensor:
    from matplotlib import pyplot as plt

    depth = depth.detach().cpu().to(dtype=torch.float32)
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape [T,H,W] or [T,H,W,1], got {tuple(depth.shape)}")

    lo = _safe_quantile(depth, 0.01)
    hi = _safe_quantile(depth, 0.99)
    scale = hi - lo
    depth_norm = (depth - lo) / max(scale, 1e-6)
    depth_norm = depth_norm.clamp(0, 1).numpy()
    colored = plt.cm.turbo(depth_norm)[..., :3]
    return torch.from_numpy(colored).permute(0, 3, 1, 2).float()


def _colorize_error_sequence(error: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    from matplotlib import pyplot as plt

    error = error.detach().cpu().to(dtype=torch.float32)
    valid_mask = valid_mask.detach().cpu()
    if error.ndim == 4 and error.shape[1] == 1:
        error = error[:, 0]
    if error.ndim == 4 and error.shape[-1] == 1:
        error = error.squeeze(-1)
    if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask[:, 0]
    if valid_mask.ndim == 4 and valid_mask.shape[-1] == 1:
        valid_mask = valid_mask.squeeze(-1)
    if error.ndim != 3 or valid_mask.ndim != 3:
        raise ValueError(
            f"Expected error/mask shapes [T,H,W], got {tuple(error.shape)} and {tuple(valid_mask.shape)}"
        )
    valid_mask = valid_mask > 0
    if error.shape != valid_mask.shape:
        raise ValueError(f"Error and valid_mask must match, got {tuple(error.shape)} and {tuple(valid_mask.shape)}")

    error = torch.where(valid_mask, error, torch.zeros_like(error))
    colored = torch.zeros((*error.shape, 3), dtype=torch.float32)
    if valid_mask.any():
        max_error = _safe_quantile(error[valid_mask], 0.99)
        error_norm = (error / max(max_error, 1e-6)).clamp(0, 1).numpy()
        colored = torch.from_numpy(plt.cm.inferno(error_norm)[..., :3]).float()
        colored[~valid_mask] = 0.0
    return colored.permute(0, 3, 1, 2)


def save_depth_evaluation_visuals(
    output_dir: Path,
    pred_depth_aligned: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        output_dir / "aligned_pred_depth.mp4",
        _tensor_to_uint8_video(_colorize_depth_sequence(pred_depth_aligned)),
        fps=10,
    )
    imageio.mimwrite(
        output_dir / "gt_depth.mp4",
        _tensor_to_uint8_video(_colorize_depth_sequence(gt_depth)),
        fps=10,
    )
    imageio.mimwrite(
        output_dir / "valid_mask.mp4",
        _tensor_to_uint8_video(valid_mask.unsqueeze(1).float()),
        fps=10,
    )
    depth_error = torch.abs(pred_depth_aligned - gt_depth)
    imageio.mimwrite(
        output_dir / "depth_error.mp4",
        _tensor_to_uint8_video(_colorize_error_sequence(depth_error, valid_mask)),
        fps=10,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw DA3 anyview-only and/or DA3 stitched checkpoints on "
            "LingBot RobbySimVal depth metrics."
        )
    )
    parser.add_argument(
        "--model_kind",
        type=str,
        default="both",
        choices=["da3_streaming", "da3_stitched", "both"],
        help="Which model path to evaluate.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=default_dataset_root(),
        help="LingBot RobbySimVal val_view root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save aggregated metrics, per-unit metrics, and visuals.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional stitched checkpoint path for the full stitched-model load path.",
    )
    parser.add_argument(
        "--initialization_weight_path",
        type=str,
        default=None,
        help=(
            "Optional transfer-layer init weight path. When set without --checkpoint_path, "
            "stitched eval loads raw base DA3 + this transfer layer only."
        ),
    )
    parser.add_argument(
        "--input_resolution",
        type=int,
        default=DEFAULT_INPUT_RESOLUTION,
        help="Target input resolution shared by both model paths.",
    )
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help="Preprocess semantics for both raw anyview and stitched DA3 evaluation.",
    )
    parser.add_argument(
        "--num_frames_per_unit",
        type=int,
        default=81,
        help="Maximum frames per scene/cam unit after stride sampling.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Frame stride applied before the max-frame clamp.",
    )
    parser.add_argument(
        "--max_units",
        type=int,
        default=None,
        help="Optional cap on the number of scene/cam units to evaluate.",
    )
    parser.add_argument(
        "--visualize_every_n",
        type=int,
        default=0,
        help="Save qualitative videos for every N evaluated units. 0 disables visuals.",
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
        help="Local Wan VAE directory used by the stitched path.",
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
        default="saddle_balanced",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view strategy for both raw anyview and stitched DA3.",
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
        help="Set when the stitched checkpoint stores full fine-tuned stitched parts.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint passed through to stitched loader when local paths are missing.",
    )
    parser.add_argument(
        "--force_fp32_inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep stitched DA3 eval in float32 by default to avoid bf16 failures in "
            "official DA3 camera-parameter decoding. Use --no-force_fp32_inference "
            "to restore the previous bf16 inference path."
        ),
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.num_frames_per_unit <= 0:
        raise ValueError("--num_frames_per_unit must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive.")
    if args.visualize_every_n < 0:
        raise ValueError("--visualize_every_n must be non-negative.")
    if args.max_units is not None and args.max_units <= 0:
        raise ValueError("--max_units must be positive when provided.")
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.initialization_weight_path and not Path(args.initialization_weight_path).exists():
        raise FileNotFoundError(
            f"Initialization weight path does not exist: {args.initialization_weight_path}"
        )
    if args.model_kind in {"da3_stitched", "both"}:
        if not args.checkpoint_path and not args.initialization_weight_path:
            raise ValueError(
                "Either --checkpoint_path or --initialization_weight_path is required for "
                "da3_stitched or both."
            )


def _strip_suffix(name: str, suffix: str) -> str | None:
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def enumerate_lingbot_units(dataset_root: str | Path) -> list[LingBotUnit]:
    root = Path(dataset_root)
    units: list[LingBotUnit] = []

    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for cam_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir()):
            intrinsics_path = cam_dir / "intrinsics.txt"

            rgb_map = {
                key: path
                for path in sorted(cam_dir.glob(f"*{RGB_SUFFIX}"))
                if (key := _strip_suffix(path.name, RGB_SUFFIX)) is not None
            }
            depth_map = {
                key: path
                for path in sorted(cam_dir.glob(f"*{GT_DEPTH_SUFFIX}"))
                if (key := _strip_suffix(path.name, GT_DEPTH_SUFFIX)) is not None
            }

            shared_keys = sorted(set(rgb_map) & set(depth_map))
            if not shared_keys:
                logger.warning("Skip empty LingBot unit with no RGB/GT pairs: {}", cam_dir)
                continue

            if len(shared_keys) != len(rgb_map) or len(shared_keys) != len(depth_map):
                logger.warning(
                    "LingBot unit {} has unmatched frames (rgb={}, depth={}, paired={})",
                    cam_dir,
                    len(rgb_map),
                    len(depth_map),
                    len(shared_keys),
                )

            units.append(
                LingBotUnit(
                    scene_id=scene_dir.name,
                    cam_id=cam_dir.name,
                    unit_dir=cam_dir,
                    intrinsics_path=intrinsics_path if intrinsics_path.exists() else None,
                    rgb_paths=tuple(rgb_map[key] for key in shared_keys),
                    depth_paths=tuple(depth_map[key] for key in shared_keys),
                )
            )

    return units


def sample_lingbot_unit(
    unit: LingBotUnit,
    num_frames_per_unit: int,
    frame_stride: int,
) -> LingBotUnit:
    indices = list(range(0, len(unit.rgb_paths), frame_stride))
    indices = indices[:num_frames_per_unit]
    if not indices:
        raise ValueError(f"Frame sampling produced an empty selection for {unit.unit_dir}")
    return replace(
        unit,
        rgb_paths=tuple(unit.rgb_paths[idx] for idx in indices),
        depth_paths=tuple(unit.depth_paths[idx] for idx in indices),
    )


def load_rgb_frames(paths: Iterable[Path]) -> torch.Tensor:
    frames = [torch.from_numpy(imageio.imread(path)[..., :3]) for path in paths]
    return rearrange(torch.stack(frames, dim=0), "t h w c -> t c h w").contiguous()


def load_depth_png_meters(path: str | Path) -> torch.Tensor:
    depth_mm = torch.from_numpy(imageio.imread(path).astype("int32"))
    if depth_mm.ndim == 3:
        depth_mm = depth_mm[..., 0]
    return depth_mm.to(dtype=torch.float32) / 1000.0


def load_depth_frames(paths: Iterable[Path]) -> torch.Tensor:
    return torch.stack([load_depth_png_meters(path) for path in paths], dim=0)


def build_center_square_roi(height: int, width: int) -> SquareROI:
    size = min(height, width)
    return SquareROI(
        top=(height - size) // 2,
        left=(width - size) // 2,
        size=size,
    )


def crop_tensor_to_roi(tensor: torch.Tensor, roi: SquareROI) -> torch.Tensor:
    bottom = roi.top + roi.size
    right = roi.left + roi.size
    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims, got {tuple(tensor.shape)}")
    return tensor[..., roi.top:bottom, roi.left:right]


def prepare_square_model_inputs_from_frames(
    frames: torch.Tensor,
    input_resolution: int,
) -> dict[str, torch.Tensor]:
    return prepare_model_inputs_from_frames(
        frames,
        input_resolution=input_resolution,
        process_res_method="square_center_crop",
    )


def prepare_model_inputs_from_frames(
    frames: torch.Tensor,
    input_resolution: int,
    process_res_method: str = "upper_bound_resize",
) -> dict[str, torch.Tensor]:
    if frames.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W] frames, got {tuple(frames.shape)}")

    frames_01 = frames.to(torch.float32) / 255.0
    if process_res_method == "upper_bound_resize":
        processed_frames_01, transform = resize_video_frames_upper_bound(
            frames_01,
            target_size=input_resolution,
            mode="bilinear",
        )
        image_hw = (
            int(transform.resized_height),
            int(transform.resized_width),
        )
    elif process_res_method == "square_center_crop":
        processed_frames_01 = resize_shorter_crop_square_batch(frames_01, target_size=input_resolution)
        image_hw = (input_resolution, input_resolution)
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {process_res_method}")
    processed_frames_signed = processed_frames_01 * 2.0 - 1.0

    return {
        "streaming_images": rearrange(processed_frames_01, "t c h w -> 1 t c h w"),
        "vae_images": rearrange(processed_frames_signed, "t c h w -> 1 c t h w"),
        "feedforward_images": rearrange(processed_frames_signed, "t c h w -> 1 c t h w"),
        "input_images_01": processed_frames_01,
        "image_hw": image_hw,
        "square_frames_01": processed_frames_01,
    }


def canonicalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 4:
        depth = depth.unsqueeze(2)
    elif depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth.permute(0, 1, 4, 2, 3)
    if depth.dim() != 5 or depth.shape[2] != 1:
        raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")
    return depth.to(dtype=torch.float32)


def _resize_bt_hw(
    tensor: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    if tensor.shape[-2:] == size:
        return tensor
    kwargs: dict[str, Any] = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(
        tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1]),
        size=size,
        mode=mode,
        **kwargs,
    ).reshape(tensor.shape[0], tensor.shape[1], size[0], size[1])


def trim_depth_eval_tensors_to_shared_frames(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int | bool]]:
    pred_depth = canonicalize_depth_tensor(pred_depth)
    pred_num_frames = int(pred_depth.shape[1])
    gt_num_frames = int(gt_depth.shape[0])
    mask_num_frames = int(valid_mask.shape[0])
    eval_num_frames = min(pred_num_frames, gt_num_frames, mask_num_frames)
    if eval_num_frames <= 0:
        raise ValueError(
            "Temporal trimming produced zero frames "
            f"(pred={pred_num_frames}, gt={gt_num_frames}, mask={mask_num_frames})."
        )

    frame_info: dict[str, int | bool] = {
        "pred_num_frames": pred_num_frames,
        "gt_num_frames": gt_num_frames,
        "mask_num_frames": mask_num_frames,
        "eval_num_frames": eval_num_frames,
        "frame_trimmed": (
            pred_num_frames != eval_num_frames
            or gt_num_frames != eval_num_frames
            or mask_num_frames != eval_num_frames
        ),
    }
    return (
        pred_depth[:, :eval_num_frames],
        gt_depth[:eval_num_frames],
        valid_mask[:eval_num_frames],
        frame_info,
    )


def evaluate_depth_prediction_on_center_square_roi(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    SquareROI,
    dict[str, int | bool],
]:
    if gt_depth.ndim != 3:
        raise ValueError(f"Expected gt_depth [T,H,W], got {tuple(gt_depth.shape)}")
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [T,H,W], got {tuple(valid_mask.shape)}")

    pred_depth, gt_depth, valid_mask, frame_info = trim_depth_eval_tensors_to_shared_frames(
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        valid_mask=valid_mask,
    )
    roi = build_center_square_roi(gt_depth.shape[-2], gt_depth.shape[-1])
    gt_depth_roi = crop_tensor_to_roi(gt_depth, roi).unsqueeze(0)
    valid_mask_roi = crop_tensor_to_roi(valid_mask, roi).unsqueeze(0)
    aligned_depth = align_depth_scale_shift(pred_depth, gt_depth_roi, valid_mask_roi)
    metrics = compute_depth_metrics(aligned_depth, gt_depth_roi, valid_mask_roi)
    return aligned_depth, gt_depth_roi, valid_mask_roi, metrics, roi, frame_info


def evaluate_depth_prediction_on_valid_pixels(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    dict[str, int],
    dict[str, int | bool],
]:
    if gt_depth.ndim != 3:
        raise ValueError(f"Expected gt_depth [T,H,W], got {tuple(gt_depth.shape)}")
    if valid_mask.ndim != 3:
        raise ValueError(f"Expected valid_mask [T,H,W], got {tuple(valid_mask.shape)}")

    pred_depth, gt_depth, valid_mask, frame_info = trim_depth_eval_tensors_to_shared_frames(
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        valid_mask=valid_mask,
    )
    pred_depth_bt = normalize_depth_tensor(pred_depth)
    gt_depth_bt = gt_depth.unsqueeze(0)
    valid_mask_bt = valid_mask.unsqueeze(0) > 0

    gt_size = (int(gt_depth.shape[-2]), int(gt_depth.shape[-1]))
    pred_valid_mask = torch.isfinite(pred_depth_bt) & (pred_depth_bt > 0)
    pred_depth_bt = torch.nan_to_num(pred_depth_bt, nan=0.0, posinf=0.0, neginf=0.0)
    pred_valid_mask = _resize_bt_hw(pred_valid_mask.float(), gt_size, mode="nearest") > 0.5
    valid_mask_bt = (
        valid_mask_bt
        & pred_valid_mask
        & torch.isfinite(gt_depth_bt)
        & (gt_depth_bt > 0)
    )
    valid_mask_f = valid_mask_bt.float()

    aligned_depth = align_depth_scale_shift(pred_depth_bt, gt_depth_bt, valid_mask_f)
    metrics = compute_depth_metrics(aligned_depth, gt_depth_bt, valid_mask_f)
    eval_info = {
        "eval_top": 0,
        "eval_left": 0,
        "eval_height": gt_size[0],
        "eval_width": gt_size[1],
    }
    return aligned_depth, gt_depth_bt, valid_mask_f, metrics, eval_info, frame_info


def build_run_metadata(
    *,
    model_kind: str,
    input_resolution: int,
    num_units: int,
    num_frames_per_unit: int,
    frame_stride: int,
    da3_process_res_method: str,
    dataset_root: str | None = None,
    checkpoint_path: str | None = None,
    initialization_weight_path: str | None = None,
) -> dict[str, Any]:
    if da3_process_res_method == "upper_bound_resize":
        input_protocol = "upper_bound_resize"
        metric_protocol = "ssi_align_on_valid_gt_depth_pixels"
    elif da3_process_res_method == "square_center_crop":
        input_protocol = "resize_shorter_crop_square"
        metric_protocol = "center_square_roi_at_original_resolution_then_scale_shift_align"
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {da3_process_res_method}")
    metadata = {
        "model_kind": model_kind,
        "dataset_root": dataset_root,
        "checkpoint_path": checkpoint_path,
        "initialization_weight_path": initialization_weight_path,
        "input_resolution": input_resolution,
        "da3_process_res_method": da3_process_res_method,
        "input_protocol": input_protocol,
        "metric_protocol": metric_protocol,
        "gt_depth_only": True,
        "num_units": num_units,
        "num_frames_per_unit": num_frames_per_unit,
        "frame_stride": frame_stride,
    }
    if model_kind == "da3_streaming":
        metadata["raw_semantics"] = "anyview_only_no_metric_postprocess"
    if model_kind == "da3_stitched":
        if checkpoint_path:
            metadata["stitched_weight_source"] = "checkpoint"
        elif initialization_weight_path:
            metadata["stitched_weight_source"] = "initialization_weight_only"
    return metadata


def _build_metric_history() -> dict[str, list[float]]:
    return {
        "depth_abs_rel": [],
        "depth_sq_rel": [],
        "depth_rmse": [],
        "depth_rmse_log": [],
        "depth_a1": [],
        "depth_a2": [],
        "depth_a3": [],
    }


def aggregate_per_scene_metrics(per_unit_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_to_records: dict[str, list[dict[str, Any]]] = {}
    for record in per_unit_metrics:
        scene_to_records.setdefault(record["scene_id"], []).append(record)

    per_scene_metrics: list[dict[str, Any]] = []
    for scene_id, records in sorted(scene_to_records.items()):
        metric_history = _build_metric_history()
        for record in records:
            for metric_name in metric_history:
                metric_history[metric_name].append(float(record[metric_name]))
        per_scene_metrics.append(
            {
                "scene_id": scene_id,
                "num_cams": len(records),
                **aggregate_metrics(metric_history),
            }
        )
    return per_scene_metrics


def load_da3_streaming_runner(args: argparse.Namespace):
    from plan4d.da3_interface import DA3StreamingInterface
    from utils.utils_for_thirdparty import load_da3_anyview_model

    device = torch.device(args.device)
    model = load_da3_anyview_model(args, device)
    runner = DA3StreamingInterface(model).to(device)
    return runner.eval()


@torch.no_grad()
def run_da3_streaming_depth(
    runner,
    prepared: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    output = runner.inference_full(
        prepared["streaming_images"].to(args.device),
        ref_view_strategy=args.da3_ref_view_strategy,
    )
    return {"depth": canonicalize_depth_tensor(output["depth"]).cpu()}


def _build_stitched_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    from models.stitching_layer_builder import parse_conv_spec

    return argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        initialization_weight_path=args.initialization_weight_path,
        feedforward_model="da3",
        video_model="wan",
        da3_model_path=args.da3_model_path,
        da3_config_name=args.da3_config_name,
        da3_checkpoint_name=args.da3_checkpoint_name,
        da3_ref_view_strategy=args.da3_ref_view_strategy,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=parse_conv_spec(args.stitching_layer_config),
        lora_config=args.lora_config,
        full_finetune_stitched_parts=args.full_finetune_stitched_parts,
        resolution=args.input_resolution,
        feedforward_resolution=args.input_resolution,
        max_frames=args.num_frames_per_unit,
        device=args.device,
        wan_vae_path=args.wan_vae_path,
        hf_endpoint=args.hf_endpoint,
        force_fp32_inference=args.force_fp32_inference,
        vggt_model_path=None,
    )


def _load_init_only_stitched_runner(args: argparse.Namespace):
    from inference_stitched_decode import apply_feedforward_model_inference_defaults
    from models.stitched_model import StitchVAE3D
    from utils.utils_for_thirdparty import (
        cast_to_bfloat16,
        configure_hf_runtime,
        load_feedforward_model,
        load_vae,
    )

    apply_feedforward_model_inference_defaults(args)
    configure_hf_runtime(args)
    device = torch.device(args.device)
    feed_forward_model = load_feedforward_model(args, device)
    video_vae = load_vae(args, device)
    video_vae.eval()
    stitched_model = StitchVAE3D(
        diffusion_vae=video_vae,
        feedforward_model=feed_forward_model,
        device=device,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=args.stitching_layer_config,
        resolution=args.resolution,
        feedforward_resolution=args.feedforward_resolution,
        stitching_layer_init_path=args.initialization_weight_path,
        stitching_module_type=getattr(args, "stitching_module_type", "conv3d"),
        stitching_hidden_channels=getattr(args, "stitching_hidden_channels", None),
        stitching_norm=getattr(args, "stitching_norm", "rms"),
        stitching_aux_kernel=getattr(args, "stitching_aux_kernel", None),
        stitching_attention_heads=getattr(args, "stitching_attention_heads", 4),
        num_frames=args.max_frames or 49,
        da3_ref_view_strategy=args.da3_ref_view_strategy,
    )
    if not bool(getattr(args, "force_fp32_inference", False)):
        cast_to_bfloat16(stitched_model)
    stitched_model = stitched_model.to(device)
    return stitched_model.eval()


def load_stitched_runner(args: argparse.Namespace):
    from inference_stitched_decode import (
        apply_feedforward_model_inference_defaults,
        load_stitched_decoder,
    )

    stitched_args = _build_stitched_loader_args(args)
    apply_feedforward_model_inference_defaults(stitched_args)
    if stitched_args.checkpoint_path:
        return load_stitched_decoder(stitched_args)
    if stitched_args.initialization_weight_path:
        return _load_init_only_stitched_runner(stitched_args)
    raise ValueError("Stitched evaluation requires either checkpoint_path or initialization_weight_path.")


@torch.no_grad()
def run_da3_stitched_depth(
    runner,
    prepared: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    from inference_stitched_decode import run_stitched_decode

    stitched_args = argparse.Namespace(
        config=None,
        feedforward_resolution=args.input_resolution,
        force_fp32_inference=bool(getattr(args, "force_fp32_inference", False)),
    )
    output = run_stitched_decode(stitched_args, runner, prepared)
    return {"depth": canonicalize_depth_tensor(output["depth"]).cpu()}


def evaluate_model_kind(
    args: argparse.Namespace,
    *,
    model_kind: str,
    units: list[LingBotUnit],
) -> None:
    output_root = Path(args.output_dir) / model_kind
    output_root.mkdir(parents=True, exist_ok=True)
    per_unit_jsonl_path = output_root / "per_unit_metrics.jsonl"
    per_unit_jsonl_path.write_text("", encoding="utf-8")

    load_runner = load_da3_streaming_runner if model_kind == "da3_streaming" else load_stitched_runner
    run_depth = run_da3_streaming_depth if model_kind == "da3_streaming" else run_da3_stitched_depth
    runner = load_runner(args)

    metric_history = _build_metric_history()
    per_unit_metrics: list[dict[str, Any]] = []

    for unit_index, unit in enumerate(units):
        sampled_unit = sample_lingbot_unit(unit, args.num_frames_per_unit, args.frame_stride)
        rgb_frames = load_rgb_frames(sampled_unit.rgb_paths)
        gt_depth = load_depth_frames(sampled_unit.depth_paths)
        valid_mask = (gt_depth > 0).to(dtype=torch.float32)
        prepared = prepare_model_inputs_from_frames(
            rgb_frames,
            args.input_resolution,
            process_res_method=args.da3_process_res_method,
        )
        outputs = run_depth(runner, prepared, args)
        if args.da3_process_res_method == "upper_bound_resize":
            aligned_depth, gt_depth_eval, valid_mask_eval, metrics, eval_info, frame_info = (
                evaluate_depth_prediction_on_valid_pixels(
                    pred_depth=outputs["depth"],
                    gt_depth=gt_depth,
                    valid_mask=valid_mask,
                )
            )
            roi_top = None
            roi_left = None
            roi_size = None
        else:
            aligned_depth, gt_depth_eval, valid_mask_eval, metrics, roi, frame_info = (
                evaluate_depth_prediction_on_center_square_roi(
                    pred_depth=outputs["depth"],
                    gt_depth=gt_depth,
                    valid_mask=valid_mask,
                )
            )
            eval_info = {
                "eval_top": roi.top,
                "eval_left": roi.left,
                "eval_height": roi.size,
                "eval_width": roi.size,
            }
            roi_top = roi.top
            roi_left = roi.left
            roi_size = roi.size

        for metric_name, metric_value in metrics.items():
            metric_history[metric_name].append(float(metric_value))

        record = {
            "scene_id": sampled_unit.scene_id,
            "cam_id": sampled_unit.cam_id,
            "unit_dir": sampled_unit.unit_dir,
            "num_frames": len(sampled_unit.rgb_paths),
            "roi_top": roi_top,
            "roi_left": roi_left,
            "roi_size": roi_size,
            **eval_info,
            **frame_info,
            **metrics,
        }
        per_unit_metrics.append(record)
        append_jsonl(per_unit_jsonl_path, record)
        if frame_info["frame_trimmed"]:
            logger.warning(
                "Temporal frame mismatch on {} {}: sampled={}, pred={}, gt={}, eval={}",
                sampled_unit.scene_id,
                sampled_unit.cam_id,
                len(sampled_unit.rgb_paths),
                frame_info["pred_num_frames"],
                frame_info["gt_num_frames"],
                frame_info["eval_num_frames"],
            )
        logger.info(
            "[{}/{}] {} {} frames={} abs_rel={:.4f} rmse={:.4f} a1={:.4f}",
            unit_index + 1,
            len(units),
            sampled_unit.scene_id,
            sampled_unit.cam_id,
            frame_info["eval_num_frames"],
            metrics["depth_abs_rel"],
            metrics["depth_rmse"],
            metrics["depth_a1"],
        )

        if args.visualize_every_n > 0 and unit_index % args.visualize_every_n == 0:
            vis_dir = output_root / "visuals" / sampled_unit.scene_id / sampled_unit.cam_id
            save_depth_evaluation_visuals(
                vis_dir,
                aligned_depth[0],
                gt_depth_eval[0],
                valid_mask_eval[0],
            )

    aggregated_metrics = aggregate_metrics(metric_history)
    metadata = build_run_metadata(
        model_kind=model_kind,
        input_resolution=args.input_resolution,
        num_units=len(units),
        num_frames_per_unit=args.num_frames_per_unit,
        frame_stride=args.frame_stride,
        da3_process_res_method=args.da3_process_res_method,
        dataset_root=args.dataset_root,
        checkpoint_path=args.checkpoint_path if model_kind == "da3_stitched" else None,
        initialization_weight_path=(
            args.initialization_weight_path if model_kind == "da3_stitched" else None
        ),
    )
    per_scene_metrics = aggregate_per_scene_metrics(per_unit_metrics)

    save_json(
        output_root / "metrics.json",
        {
            "meta": metadata,
            "metrics": aggregated_metrics,
        },
    )
    save_json(output_root / "per_unit_metrics.json", per_unit_metrics)
    save_json(output_root / "per_scene_metrics.json", per_scene_metrics)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    units = enumerate_lingbot_units(args.dataset_root)
    if args.max_units is not None:
        units = units[: args.max_units]
    if not units:
        raise ValueError(f"No LingBot scene/cam units found under {args.dataset_root}")

    model_kinds = (
        ["da3_streaming", "da3_stitched"]
        if args.model_kind == "both"
        else [args.model_kind]
    )
    for model_kind in model_kinds:
        logger.info("Evaluating {} on {} LingBot units", model_kind, len(units))
        evaluate_model_kind(args, model_kind=model_kind, units=units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
