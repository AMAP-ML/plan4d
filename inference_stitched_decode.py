import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from loguru import logger

from data.image_preprocessing import resize_shorter_crop_square_batch
from models.stitching_layer_builder import parse_conv_spec
from utils.da3_resize import resize_video_frames_upper_bound
from depth_anything_3.utils.scene_flow_postprocess import (
    filter_scene_flow_periodic,
    scene_flow_filter_period,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext:  # type: ignore[override]
        def __init__(self, enter_result=None):
            self.enter_result = enter_result

        def __enter__(self):
            return self.enter_result

        def __exit__(self, *excinfo):
            return False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stitched decode inference from an RGB video / frame folder, or "
            "evaluate depth metrics on PointOdyssey samples defined by a config."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help=(
            "Path to an input video file or a directory of RGB frames. Required unless "
            "--config is provided for dataset depth evaluation."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="PointOdyssey dataset config YAML, e.g. training/config/geometry_adapter_v3.yaml.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional override for data.data_root in --config.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="Optional override for data.val_split/test split in --config.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Maximum number of PointOdyssey samples to evaluate.",
    )
    parser.add_argument(
        "--visualize_every_n",
        type=int,
        default=10,
        help="Save qualitative outputs for every N evaluated samples.",
    )
    parser.add_argument(
        "--depth_eval_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only keep predicted depth and depth-evaluation artifacts. Disables gs, "
            "reconstruction, ply, tensors, input RGB, and confidence videos."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="stitched_decode_results",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/stitched_decoder.pth",
        help="Path to a stitched lightweight .pth, an epoch directory, an accelerate_state directory, or a full-state pytorch_model_fsdp.bin/pytorch_model.bin file.",
    )
    parser.add_argument(
        "--feedforward_model",
        type=str,
        default="da3",
        choices=["da3"],
    )
    parser.add_argument(
        "--video_model",
        type=str,
        default="wan",
        choices=["wan"],
    )
    parser.add_argument(
        "--da3_model_path",
        type=str,
        default=None,
        help="Optional local DA3 pretrained directory containing config.json and model.safetensors.",
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
        default=None,
        help="Enable the DA3 scene-flow head for stitched geometry inference.",
    )
    parser.add_argument(
        "--da3_scene_flow_head_checkpoint_path",
        type=str,
        default=None,
        help="Optional trained head override; otherwise use the stitched checkpoint's head.",
    )
    parser.add_argument(
        "--da3_scene_flow_preset",
        type=str,
        default=None,
        help="DA3 architecture preset for the scene-flow head and adapter.",
    )
    parser.add_argument(
        "--da3_ref_view_strategy",
        type=str,
        default="saddle_balanced",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view strategy for DA3 anyview backbone.",
    )
    parser.add_argument(
        "--wan_vae_path",
        type=str,
        default=None,
        help=(
            "Optional local Wan VAE directory. If omitted, auto-detects "
            "checkpoints/wan_vae or ../Gen3R/checkpoints/wan_vae."
        ),
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help=(
            "Hugging Face endpoint used when local model directories are unavailable. "
            "Defaults to HF_ENDPOINT env var or hf-mirror."
        ),
    )
    parser.add_argument(
        "--wan_vae_repo_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        help="Remote Wan VAE repo id used when --wan_vae_path is not provided.",
    )
    parser.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_2",
        help="Location of stitching layer in DA3 patch embed blocks.",
    )
    parser.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default=parse_conv_spec("conv3d_k5x3x3_o1024_s1x2x2_p2x1x1"),
        metavar="CONV_SPEC",
        help="Conv spec for the transfer / stitching layer.",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="r512,a256,d0.05,f0",
        help="LoRA config used by the stitched checkpoint.",
    )
    parser.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, stitched parts stop using LoRA and are loaded as full "
            "fine-tuned weights."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Input resolution for Wan VAE branch.",
    )
    parser.add_argument(
        "--feedforward_resolution",
        type=int,
        default=448,
        help="Input resolution for DA3 branch.",
    )
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help="DA3 stitched preprocessing semantics.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help=(
            "Maximum number of frames to use after stride sampling. Defaults to 21 in "
            "single-input mode and to the full dataset sample in --config eval mode."
        ),
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Stride when reading frames from the source.",
    )
    parser.add_argument(
        "--save_tensors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw tensors (.pt) for depth, conf, pose, and point maps.",
    )
    parser.add_argument(
        "--save_input_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save input RGB video alongside predictions.",
    )
    parser.add_argument(
        "--save_depth_conf_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save predicted depth confidence video.",
    )
    parser.add_argument(
        "--force_fp32_inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable CUDA autocast and keep stitched weights in float32 during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. cuda:0 or cpu. Default is auto.",
    )
    parser.add_argument(
        "--scene_flow_degrid_period", "--scene-flow-degrid-period",
        type=scene_flow_filter_period, default=0.0, help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    scene_flow_defaults = {
        "da3_enable_scene_flow_head": False,
        "da3_scene_flow_preset": "da3-giant-sceneflow",
    }
    explicit_fields = ["da3_scene_flow_head_checkpoint_path"]
    for field_name, default_value in scene_flow_defaults.items():
        if getattr(args, field_name) is None:
            setattr(args, field_name, default_value)
        else:
            explicit_fields.append(field_name)
    args.explicit_scene_flow_fields = tuple(explicit_fields)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.config is None and not args.input_path:
        raise ValueError("--input_path is required unless --config is provided.")
    num_samples = getattr(args, "num_samples", None)
    visualize_every_n = getattr(args, "visualize_every_n", 10)
    if num_samples is not None and num_samples <= 0:
        raise ValueError("--num_samples must be a positive integer.")
    if visualize_every_n <= 0:
        raise ValueError("--visualize_every_n must be a positive integer.")


def apply_feedforward_model_inference_defaults(args: argparse.Namespace) -> None:
    if args.feedforward_model != "da3":
        return
    if args.stitching_layer_config.out_channels != 1536:
        args.stitching_layer_config = parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1")
    if getattr(args, "resolution", None) == 512:
        args.resolution = 532
    if getattr(args, "feedforward_resolution", None) == 448:
        args.feedforward_resolution = 532
    if getattr(args, "da3_process_res_method", None) is None:
        args.da3_process_res_method = "upper_bound_resize"


def apply_output_mode(args: argparse.Namespace) -> None:
    if not getattr(args, "depth_eval_only", False):
        return

    args.save_input_rgb = False
    args.save_depth_conf_video = False
    args.save_tensors = False


def should_force_fp32_inference(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "force_fp32_inference", False))


FULL_STATE_CHECKPOINT_FILENAMES = ("pytorch_model_fsdp.bin", "pytorch_model.bin")
_OPTIONAL_WEIGHT_LOAD_UNEXPECTED_KEY_PREFIXES = (
    "model.scene_flow_head.",
    "model.scene_flow_adapter.",
)


@dataclass(frozen=True)
class CheckpointLoadPlan:
    checkpoint_path: Path
    checkpoint_dir: Optional[Path]
    lightweight_checkpoint_path: Optional[Path]
    full_state_checkpoint_path: Optional[Path]
    saved_args: Optional[argparse.Namespace]
    saved_epoch: Optional[int]


def _torch_load_checkpoint(path: Path, *, map_location: str = "cpu", mmap: bool = False):
    load_kwargs = {"weights_only": False, "map_location": map_location}
    if mmap:
        try:
            return torch.load(path, mmap=True, **load_kwargs)
        except TypeError:
            pass
    return torch.load(path, **load_kwargs)


def _filter_optional_unexpected_state_dict_keys(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    module_keys = set(module.state_dict().keys())
    ignored_unexpected_keys = [
        key
        for key in state_dict.keys()
        if key not in module_keys
        and any(
            key.startswith(prefix)
            for prefix in _OPTIONAL_WEIGHT_LOAD_UNEXPECTED_KEY_PREFIXES
        )
    ]
    if not ignored_unexpected_keys:
        return state_dict, []

    filtered_state_dict = state_dict.copy()
    for key in ignored_unexpected_keys:
        filtered_state_dict.pop(key, None)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        filtered_state_dict._metadata = metadata  # type: ignore[attr-defined]
    return filtered_state_dict, ignored_unexpected_keys


def _load_state_dict_with_diagnostics(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    component_name: str,
    strict: bool,
    expected_missing_keys: Optional[set[str] | list[str] | tuple[str, ...]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], list[str]]:
    filtered_state_dict, ignored_unexpected_keys = _filter_optional_unexpected_state_dict_keys(
        module,
        state_dict,
    )
    incompatible = module.load_state_dict(filtered_state_dict, strict=False)
    missing_keys = list(getattr(incompatible, "missing_keys", []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))
    base_model_missing_keys: list[str] = []
    if expected_missing_keys is not None:
        expected_missing_key_set = set(expected_missing_keys)
        filtered_missing_keys = [
            key for key in missing_keys if key in expected_missing_key_set
        ]
        base_model_missing_keys = [
            key for key in missing_keys if key not in expected_missing_key_set
        ]
        missing_keys = filtered_missing_keys
    emit = logger.info if log_fn is None else log_fn
    message = (
        f"[weight-load] {component_name}: "
        f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
    )
    if base_model_missing_keys:
        message += (
            f", base_model_missing_from_payload_count={len(base_model_missing_keys)}"
            f", base_model_missing_from_payload_head={base_model_missing_keys[:20]}"
        )
    emit(message)
    if ignored_unexpected_keys:
        logger.warning(
            "[weight-load] {}: ignoring optional unexpected scene-flow keys_count={} keys_head={}",
            component_name,
            len(ignored_unexpected_keys),
            ignored_unexpected_keys[:20],
        )
    if strict and (missing_keys or unexpected_keys or base_model_missing_keys):
        raise RuntimeError(
            "Failed to load {} strictly (missing_keys={}, unexpected_keys={})".format(
                component_name,
                base_model_missing_keys + missing_keys,
                unexpected_keys,
            )
        )
    return missing_keys, unexpected_keys


def _copy_checkpoint_tensor_to_parameter(
    parameter: torch.nn.Parameter,
    checkpoint_tensor: torch.Tensor,
) -> None:
    if tuple(parameter.shape) != tuple(checkpoint_tensor.shape):
        raise RuntimeError(
            "Checkpoint tensor shape {} does not match parameter shape {}".format(
                tuple(checkpoint_tensor.shape),
                tuple(parameter.shape),
            )
        )
    with torch.no_grad():
        parameter.copy_(
            checkpoint_tensor.to(device=parameter.device, dtype=parameter.dtype)
        )


def _collect_lora_expected_keys(module: torch.nn.Module) -> list[str]:
    from utils.lora_util.layers import LoRALayer

    expected_keys: list[str] = []
    for module_name, submodule in module.named_modules():
        if not isinstance(submodule, LoRALayer):
            continue
        prefix = f"{module_name}." if module_name else ""
        if hasattr(submodule, "lora_A"):
            expected_keys.append(f"{prefix}lora_A")
        if hasattr(submodule, "lora_B"):
            expected_keys.append(f"{prefix}lora_B")
        if getattr(submodule, "bias", None) is not None:
            expected_keys.append(f"{prefix}bias")
    return expected_keys


def _log_direct_weight_status(
    *,
    component_name: str,
    expected_keys: list[str],
    loaded_keys: list[str],
) -> None:
    missing_keys = [key for key in expected_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in expected_keys]
    logger.info(
        "[weight-load] {}: missing_keys={}, unexpected_keys={}",
        component_name,
        missing_keys,
        unexpected_keys,
    )


def _resolve_lightweight_checkpoint_path(
    checkpoint_dir: Path,
    saved_epoch: Optional[int],
    saved_step: Optional[int] = None,
) -> Optional[Path]:
    if saved_step is not None:
        exact_match = checkpoint_dir / f"stitched_model_step_{saved_step}.pth"
        if exact_match.exists():
            return exact_match.resolve()
    if saved_epoch is not None:
        exact_match = checkpoint_dir / f"stitched_model_epoch_{saved_epoch}.pth"
        if exact_match.exists():
            return exact_match.resolve()

    candidates = sorted(
        checkpoint_dir.glob("stitched_model_step_*.pth"),
        key=lambda candidate: int(candidate.stem.rsplit("_", 1)[-1]) if candidate.stem.rsplit("_", 1)[-1].isdigit() else -1,
    )
    if candidates:
        return candidates[-1].resolve()

    candidates = sorted(
        checkpoint_dir.glob("stitched_model_epoch_*.pth"),
        key=lambda candidate: int(candidate.stem.rsplit("_", 1)[-1]) if candidate.stem.rsplit("_", 1)[-1].isdigit() else -1,
    )
    if not candidates:
        return None
    return candidates[-1].resolve()


def _load_saved_resume_metadata(
    checkpoint_dir: Optional[Path],
) -> tuple[Optional[argparse.Namespace], Optional[int], Optional[int]]:
    if checkpoint_dir is None:
        return None, None, None

    resume_checkpoint_path = checkpoint_dir / "resume_checkpoint.pth"
    if not resume_checkpoint_path.exists():
        return None, None, None

    resume_payload = _torch_load_checkpoint(resume_checkpoint_path, map_location="cpu")
    if not isinstance(resume_payload, dict):
        return None, None, None

    saved_args = resume_payload.get("args")
    saved_epoch = None
    saved_step = None
    if resume_payload.get("epoch") is not None:
        saved_epoch = int(resume_payload["epoch"])
    if resume_payload.get("global_step") is not None:
        saved_step = int(resume_payload["global_step"])
    if not isinstance(saved_args, argparse.Namespace):
        saved_args = None
    return saved_args, saved_epoch, saved_step


def _resolve_full_state_checkpoint_path(accelerate_state_dir: Path) -> Optional[Path]:
    for file_name in FULL_STATE_CHECKPOINT_FILENAMES:
        candidate = accelerate_state_dir / file_name
        if candidate.exists():
            return candidate.resolve()
    return None


def build_checkpoint_load_plan(checkpoint_path: str) -> CheckpointLoadPlan:
    resolved = Path(checkpoint_path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {resolved}")

    if resolved.is_file():
        if resolved.name in FULL_STATE_CHECKPOINT_FILENAMES:
            checkpoint_dir = resolved.parent.parent.resolve() if resolved.parent.name == "accelerate_state" else resolved.parent.resolve()
            saved_args, saved_epoch, saved_step = _load_saved_resume_metadata(checkpoint_dir)
            lightweight_checkpoint_path = _resolve_lightweight_checkpoint_path(checkpoint_dir, saved_epoch, saved_step)
            return CheckpointLoadPlan(
                checkpoint_path=resolved,
                checkpoint_dir=checkpoint_dir,
                lightweight_checkpoint_path=lightweight_checkpoint_path,
                full_state_checkpoint_path=resolved,
                saved_args=saved_args,
                saved_epoch=saved_epoch,
            )

        checkpoint_dir = resolved.parent.resolve() if resolved.parent.is_dir() else None
        saved_args, saved_epoch, _ = _load_saved_resume_metadata(checkpoint_dir)
        return CheckpointLoadPlan(
            checkpoint_path=resolved,
            checkpoint_dir=checkpoint_dir if saved_args is not None or saved_epoch is not None else None,
            lightweight_checkpoint_path=resolved,
            full_state_checkpoint_path=None,
            saved_args=saved_args,
            saved_epoch=saved_epoch,
        )

    if resolved.name == "accelerate_state":
        checkpoint_dir = resolved.parent.resolve()
        accelerate_state_dir = resolved
        lightweight_checkpoint_path = None
    else:
        checkpoint_dir = resolved
        accelerate_state_dir = resolved / "accelerate_state"
        lightweight_checkpoint_path = None

    saved_args, saved_epoch, saved_step = _load_saved_resume_metadata(checkpoint_dir)
    full_state_checkpoint_path = _resolve_full_state_checkpoint_path(accelerate_state_dir)
    if lightweight_checkpoint_path is None and resolved.name != "accelerate_state":
        lightweight_checkpoint_path = _resolve_lightweight_checkpoint_path(checkpoint_dir, saved_epoch, saved_step)

    if full_state_checkpoint_path is None and lightweight_checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not find a supported checkpoint under directory: {resolved}"
        )

    return CheckpointLoadPlan(
        checkpoint_path=resolved,
        checkpoint_dir=checkpoint_dir,
        lightweight_checkpoint_path=lightweight_checkpoint_path,
        full_state_checkpoint_path=full_state_checkpoint_path,
        saved_args=saved_args,
        saved_epoch=saved_epoch,
    )


def apply_saved_checkpoint_args(
    args: argparse.Namespace,
    checkpoint_plan: CheckpointLoadPlan,
    *,
    preserve_fields: tuple[str, ...] = (),
) -> None:
    saved_args = checkpoint_plan.saved_args
    if saved_args is None:
        return
    preserved = set(preserve_fields)

    for field_name in (
        "feedforward_model",
        "video_model",
        "stitching_layer_location",
        "stitching_module_type",
        "stitching_hidden_channels",
        "stitching_norm",
        "stitching_aux_kernel",
        "stitching_attention_heads",
        "lora_config",
        "full_finetune_stitched_parts",
        "da3_ref_view_strategy",
        "da3_enable_scene_flow_head",
        "da3_scene_flow_head_checkpoint_path",
        "da3_scene_flow_preset",
        "resolution",
        "feedforward_resolution",
    ):
        if field_name in preserved and hasattr(args, field_name):
            continue
        if hasattr(saved_args, field_name):
            setattr(args, field_name, getattr(saved_args, field_name))

    if hasattr(saved_args, "stitching_layer_config"):
        stitching_layer_config = getattr(saved_args, "stitching_layer_config")
        if isinstance(stitching_layer_config, str):
            stitching_layer_config = parse_conv_spec(stitching_layer_config)
        args.stitching_layer_config = stitching_layer_config

    if getattr(args, "max_frames", None) is None and hasattr(saved_args, "num_frames_per_unit_scene"):
        args.max_frames = int(getattr(saved_args, "num_frames_per_unit_scene"))


def load_yaml_config(config_path: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(config_path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config path does not exist: {resolved}")
    with open(resolved, "r") as f:
        return resolved, yaml.safe_load(f)


def find_plan4d_root(config_path: Path) -> Path:
    for candidate in [config_path.parent] + list(config_path.parents):
        if (candidate / "plan4d" / "point_odyssey.py").exists():
            return candidate

    default_root = Path(".")
    if (default_root / "plan4d" / "point_odyssey.py").exists():
        return default_root

    raise FileNotFoundError(
        "Could not locate plan4d root containing plan4d/point_odyssey.py"
    )


def import_point_odyssey_dataset(config_path: Path):
    plan4d_root = find_plan4d_root(config_path)
    if str(plan4d_root) not in sys.path:
        sys.path.insert(0, str(plan4d_root))
    module = importlib.import_module("plan4d.point_odyssey")
    return module.PointOdysseyDataset


def build_point_odyssey_eval_dataset_kwargs(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    data_config = config.get("data", {})
    data_root = args.data_root or data_config.get("data_root")
    if not data_root:
        raise ValueError("PointOdyssey evaluation requires data.data_root or --data_root.")

    split = args.eval_split or data_config.get("val_split") or "test"
    image_size = data_config.get("image_size")

    return {
        "data_root": data_root,
        "split": split,
        "chunk_size": data_config.get("chunk_size", 21),
        "overlap": data_config.get("overlap", 0),
        "image_size": tuple(image_size) if image_size is not None else None,
        "target_height": data_config.get("target_height"),
        "target_width": data_config.get("target_width"),
        "load_depth": True,
        "load_ray": False,
        "max_scenes": None,
    }


def sample_frame_indices(
    total_frames: int,
    max_frames: int | None,
    frame_stride: int,
) -> list[int]:
    if total_frames <= 0:
        raise ValueError("No frames available in the input.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")

    indices = list(range(0, total_frames, frame_stride))
    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]
    if not indices:
        raise ValueError("Frame sampling produced an empty index list.")
    return indices


def _is_image_file(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))


def load_input_frames(
    input_path: str,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> torch.Tensor:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    frames: list[torch.Tensor] = []
    if source.is_dir():
        frame_paths = sorted(p for p in source.iterdir() if p.is_file() and _is_image_file(p.name))
        indices = sample_frame_indices(len(frame_paths), max_frames, frame_stride)
        for idx in indices:
            frame = imageio.v2.imread(frame_paths[idx])
            frames.append(torch.from_numpy(np.asarray(frame)[..., :3]))
    else:
        video_frames = []
        for frame in imageio.v3.imiter(source):
            video_frames.append(torch.from_numpy(np.asarray(frame)[..., :3]))
        indices = sample_frame_indices(len(video_frames), max_frames, frame_stride)
        frames = [video_frames[idx] for idx in indices]

    frame_tensor = torch.stack(frames, dim=0)
    return rearrange(frame_tensor, "t h w c -> t c h w").contiguous()


def prepare_model_inputs_from_frames(
    frames: torch.Tensor,
    vae_resolution: int,
    feedforward_resolution: int,
    *,
    da3_process_res_method: str = "square_center_crop",
) -> dict[str, torch.Tensor]:
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape [T,C,H,W], got {tuple(frames.shape)}")

    frames_01 = frames.to(torch.float32) / 255.0
    if da3_process_res_method == "upper_bound_resize":
        vae_frames_01, _ = resize_video_frames_upper_bound(
            frames_01,
            target_size=vae_resolution,
            mode="bilinear",
        )
        feedforward_frames_01, feedforward_transform = resize_video_frames_upper_bound(
            frames_01,
            target_size=feedforward_resolution,
            mode="bilinear",
        )
        image_hw = (
            int(feedforward_transform.resized_height),
            int(feedforward_transform.resized_width),
        )
    elif da3_process_res_method == "square_center_crop":
        vae_frames_01 = resize_shorter_crop_square_batch(frames_01, target_size=vae_resolution)
        feedforward_frames_01 = F.interpolate(
            vae_frames_01,
            size=(feedforward_resolution, feedforward_resolution),
            mode="bilinear",
            align_corners=False,
        )
        image_hw = (feedforward_resolution, feedforward_resolution)
    else:
        raise ValueError(f"Unsupported da3_process_res_method: {da3_process_res_method}")

    return {
        "vae_images": rearrange(vae_frames_01 * 2.0 - 1.0, "t c h w -> 1 c t h w"),
        "feedforward_images": rearrange(
            feedforward_frames_01 * 2.0 - 1.0, "t c h w -> 1 c t h w"
        ),
        "input_images_01": rearrange(feedforward_frames_01, "t c h w -> 1 c t h w"),
        "vae_frames_01": vae_frames_01,
        "feedforward_frames_01": feedforward_frames_01,
        "image_hw": image_hw,
    }


def normalize_depth_tensor(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5 and depth.shape[2] == 1:
        depth = depth.squeeze(2)
    if depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.dim() != 4:
        raise ValueError(f"Expected depth shape [B,T,H,W] or [B,T,H,W,1], got {tuple(depth.shape)}")
    return depth.float()


def align_depth_scale_shift(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    pred_depth = normalize_depth_tensor(pred_depth)
    gt_depth = normalize_depth_tensor(gt_depth)
    valid_mask = normalize_depth_tensor(valid_mask)

    h_gt, w_gt = gt_depth.shape[-2:]
    if pred_depth.shape[-2:] != (h_gt, w_gt):
        pred_depth = F.interpolate(
            pred_depth.reshape(-1, 1, pred_depth.shape[-2], pred_depth.shape[-1]),
            size=(h_gt, w_gt),
            mode="bilinear",
            align_corners=False,
        ).reshape(pred_depth.shape[0], pred_depth.shape[1], h_gt, w_gt)

    if valid_mask.shape[-2:] != (h_gt, w_gt):
        valid_mask = F.interpolate(
            valid_mask.reshape(-1, 1, valid_mask.shape[-2], valid_mask.shape[-1]),
            size=(h_gt, w_gt),
            mode="nearest",
        ).reshape(valid_mask.shape[0], valid_mask.shape[1], h_gt, w_gt)

    aligned = pred_depth.clone()
    batch_size, num_frames = pred_depth.shape[:2]

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            valid = valid_mask[batch_idx, frame_idx] > 0
            if valid.sum() < 2:
                continue

            pred = pred_depth[batch_idx, frame_idx][valid]
            gt = gt_depth[batch_idx, frame_idx][valid]

            pred_mean = pred.mean()
            gt_mean = gt.mean()
            pred_centered = pred - pred_mean
            gt_centered = gt - gt_mean

            var_pred = (pred_centered ** 2).sum() + 1e-8
            cov = (pred_centered * gt_centered).sum()

            scale = (cov / var_pred).clamp(min=0.01, max=100.0)
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

    h_gt, w_gt = gt_depth.shape[-2:]

    if pred_depth.shape[-2:] != (h_gt, w_gt):
        pred_depth = F.interpolate(
            pred_depth.reshape(-1, 1, pred_depth.shape[-2], pred_depth.shape[-1]),
            size=(h_gt, w_gt),
            mode="bilinear",
            align_corners=False,
        ).reshape(pred_depth.shape[0], pred_depth.shape[1], h_gt, w_gt)

    if valid_mask.shape[-2:] != (h_gt, w_gt):
        valid_mask = F.interpolate(
            valid_mask.reshape(-1, 1, valid_mask.shape[-2], valid_mask.shape[-1]),
            size=(h_gt, w_gt),
            mode="nearest",
        ).reshape(valid_mask.shape[0], valid_mask.shape[1], h_gt, w_gt)

    valid = valid_mask.bool().flatten()
    pred = pred_depth.flatten()
    gt = gt_depth.flatten()

    if valid.shape[0] != pred.shape[0]:
        raise ValueError(
            f"Depth metric shape mismatch: valid={tuple(valid.shape)}, pred={tuple(pred.shape)}"
        )

    pred = pred[valid]
    gt = gt[valid]

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

    gt = gt.clamp(min=1e-6)
    pred = pred.clamp(min=1e-6)

    abs_rel = (torch.abs(pred - gt) / gt).mean().item()
    sq_rel = (((pred - gt) ** 2) / gt).mean().item()
    rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
    rmse_log = torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean()).item()
    thresh = torch.max(pred / gt, gt / pred)
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25**2).float().mean().item()
    a3 = (thresh < 1.25**3).float().mean().item()

    return {
        "depth_abs_rel": abs_rel,
        "depth_sq_rel": sq_rel,
        "depth_rmse": rmse,
        "depth_rmse_log": rmse_log,
        "depth_a1": a1,
        "depth_a2": a2,
        "depth_a3": a3,
    }


def colorize_depth_sequence(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape [T,H,W] or [T,H,W,1], got {tuple(depth.shape)}")

    depth_cpu = depth.detach().float().cpu()
    depth_flat = depth_cpu.reshape(-1)
    max_quantile_samples = 2**20
    if depth_flat.numel() > max_quantile_samples:
        indices = torch.randperm(depth_flat.numel())[:max_quantile_samples]
        sampled = depth_flat[indices]
    else:
        sampled = depth_flat
    lo = torch.quantile(sampled, 0.01)
    hi = torch.quantile(sampled, 0.99)
    depth_norm = (depth_cpu - lo) / max((hi - lo).item(), 1e-6)
    depth_norm = depth_norm.clamp(0, 1).numpy()
    colored = plt.cm.turbo(depth_norm)[..., :3]
    return torch.from_numpy(colored).permute(0, 3, 1, 2).float()


def colorize_confidence_sequence(conf: torch.Tensor) -> torch.Tensor:
    if conf.ndim == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
    conf_cpu = conf.detach().float().cpu()
    lo = conf_cpu.min()
    hi = conf_cpu.max()
    conf_norm = (conf_cpu - lo) / max((hi - lo).item(), 1e-6)
    colored = plt.cm.viridis(conf_norm.numpy())[..., :3]
    return torch.from_numpy(colored).permute(0, 3, 1, 2).float()


def denormalize_intrinsics(
    intrinsics_norm: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    intrinsics_px = intrinsics_norm.clone()
    intrinsics_px[..., 0, :] *= width
    intrinsics_px[..., 1, :] *= height
    return intrinsics_px


def decoder_autocast(device: torch.device, force_fp32_inference: bool = False):
    if force_fp32_inference or device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)


def should_cast_stitched_model_bfloat16(
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    return device.type == "cuda" and not should_force_fp32_inference(args)


def _apply_stitched_model_inference_dtype_policy(
    stitched_model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if should_force_fp32_inference(args):
        stitched_model.float()
        return
    if should_cast_stitched_model_bfloat16(args, device):
        from utils.utils_for_thirdparty import cast_to_bfloat16

        cast_to_bfloat16(stitched_model)


def load_stitched_decoder(
    args: argparse.Namespace,
    checkpoint_plan: Optional[CheckpointLoadPlan] = None,
    *,
    preserve_explicit_fields: tuple[str, ...] = (),
    diffusion_vae_override: torch.nn.Module | None = None,
    ignore_diffusion_vae_state: bool = False,
) -> torch.nn.Module:
    from utils.utils_for_thirdparty import configure_hf_runtime

    checkpoint_plan = build_checkpoint_load_plan(args.checkpoint_path) if checkpoint_plan is None else checkpoint_plan
    independent_scene_flow_head_checkpoint_path = getattr(
        args,
        "da3_scene_flow_head_checkpoint_path",
        None,
    )
    effective_preserve_fields = tuple(preserve_explicit_fields)
    if independent_scene_flow_head_checkpoint_path:
        effective_preserve_fields = tuple(
            dict.fromkeys(
                (*effective_preserve_fields, "da3_scene_flow_head_checkpoint_path")
            )
        )
    apply_saved_checkpoint_args(
        args,
        checkpoint_plan,
        preserve_fields=effective_preserve_fields,
    )
    configure_hf_runtime(args)
    from models.da3_stitched import DA3Stitched
    from models.stitched_model import StitchVAE3D
    from utils.lora_util.utils import parse_lora_mode
    from utils.stitched_parts import (
        checkpoint_uses_full_finetune_stitched_parts,
        prepare_stitched_parts_for_loading,
        resolve_full_finetune_stitched_parts,
    )
    from utils.utils_for_thirdparty import load_feedforward_model, load_vae

    device = torch.device(args.device)
    load_independent_scene_flow_head = bool(independent_scene_flow_head_checkpoint_path)
    feed_forward_model = load_feedforward_model(
        args,
        device,
        load_scene_flow_head=load_independent_scene_flow_head,
    )
    video_vae = diffusion_vae_override if diffusion_vae_override is not None else load_vae(args, device)
    video_vae.eval()
    stitched_model = StitchVAE3D(
        diffusion_vae=video_vae,
        feedforward_model=feed_forward_model,
        device=device,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=args.stitching_layer_config,
        resolution=args.resolution,
        feedforward_resolution=args.feedforward_resolution,
        stitching_layer_init_path=None,
        stitching_module_type=getattr(args, "stitching_module_type", "conv3d"),
        stitching_hidden_channels=getattr(args, "stitching_hidden_channels", None),
        stitching_norm=getattr(args, "stitching_norm", "rms"),
        stitching_aux_kernel=getattr(args, "stitching_aux_kernel", None),
        stitching_attention_heads=getattr(args, "stitching_attention_heads", 4),
        num_frames=getattr(args, "max_frames", None) or 49,
        da3_ref_view_strategy=args.da3_ref_view_strategy,
    )

    lora_cfg = parse_lora_mode(args.lora_config)
    state_dict = None
    full_finetune_stitched_parts = bool(getattr(args, "full_finetune_stitched_parts", False))
    if checkpoint_plan.full_state_checkpoint_path is None:
        checkpoint_file = checkpoint_plan.lightweight_checkpoint_path
        if checkpoint_file is None:
            raise FileNotFoundError(
                f"Could not resolve a lightweight checkpoint file from {checkpoint_plan.checkpoint_path}"
            )
        logger.info("Loading stitched lightweight checkpoint from {}", checkpoint_file)
        state_dict = _torch_load_checkpoint(checkpoint_file, map_location="cpu")
        full_finetune_stitched_parts = resolve_full_finetune_stitched_parts(
            cli_value=full_finetune_stitched_parts,
            saved_args=checkpoint_plan.saved_args,
            state_dict=state_dict,
        )

    prepare_stitched_parts_for_loading(
        stitched_model.stitched_3d_model,
        lora_cfg,
        full_finetune_stitched_parts=full_finetune_stitched_parts,
    )

    if checkpoint_plan.full_state_checkpoint_path is not None:
        logger.info(
            "Loading stitched full-state checkpoint from {}",
            checkpoint_plan.full_state_checkpoint_path,
        )
        full_state_dict = _torch_load_checkpoint(
            checkpoint_plan.full_state_checkpoint_path,
            map_location="cpu",
            mmap=True,
        )
        if ignore_diffusion_vae_state:
            full_state_dict = {
                key: value for key, value in full_state_dict.items()
                if not key.startswith("diffusion_vae.")
            }
        missing_keys, unexpected_keys = _load_state_dict_with_diagnostics(
            stitched_model,
            full_state_dict,
            component_name="stitched_model(full_state)",
            strict=False,
        )
        if ignore_diffusion_vae_state:
            missing_keys = [key for key in missing_keys if not key.startswith("diffusion_vae.")]
            unexpected_keys = [key for key in unexpected_keys if not key.startswith("diffusion_vae.")]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Failed to load full-state checkpoint from {} (missing_keys={}, unexpected_keys={})".format(
                    checkpoint_plan.full_state_checkpoint_path,
                    missing_keys,
                    unexpected_keys,
                )
            )
    else:
        if state_dict is None:
            raise RuntimeError("Lightweight checkpoint state_dict was not loaded.")
        stitched_parts_mode = "full" if checkpoint_uses_full_finetune_stitched_parts(state_dict) else "lora"
        stitched_parts_key = "stitched_parts" if stitched_parts_mode == "full" else "lora"
        stitched_parts_payload = state_dict.get(stitched_parts_key)
        if stitched_parts_payload is None:
            raise KeyError(f"Checkpoint missing '{stitched_parts_key}' for stitched parts load.")
        expected_missing_keys = None
        if stitched_parts_mode == "lora":
            expected_missing_keys = _collect_lora_expected_keys(
                stitched_model.stitched_3d_model
            )
        _load_state_dict_with_diagnostics(
            stitched_model.stitched_3d_model,
            stitched_parts_payload,
            component_name=f"stitched_3d_model({stitched_parts_mode})",
            strict=stitched_parts_mode == "full",
            expected_missing_keys=expected_missing_keys,
        )
        _load_state_dict_with_diagnostics(
            stitched_model.stitching_layer,
            state_dict["stitching_layer"],
            component_name="stitching_layer",
            strict=True,
        )
        if isinstance(stitched_model.stitched_3d_model, DA3Stitched):
            pretrained = stitched_model.stitched_3d_model.backbone.pretrained
            loaded_token_keys = []
            if "cls_token" in state_dict:
                _copy_checkpoint_tensor_to_parameter(
                    pretrained.cls_token,
                    state_dict["cls_token"],
                )
                loaded_token_keys.append("cls_token")
            if "register_tokens" in state_dict and pretrained.register_tokens is not None:
                _copy_checkpoint_tensor_to_parameter(
                    pretrained.register_tokens,
                    state_dict["register_tokens"],
                )
                loaded_token_keys.append("register_tokens")
            if "camera_token" in state_dict and hasattr(pretrained, "camera_token"):
                _copy_checkpoint_tensor_to_parameter(
                    pretrained.camera_token,
                    state_dict["camera_token"],
                )
                loaded_token_keys.append("camera_token")
            expected_token_keys = ["cls_token"]
            if pretrained.register_tokens is not None:
                expected_token_keys.append("register_tokens")
            if hasattr(pretrained, "camera_token"):
                expected_token_keys.append("camera_token")
            _log_direct_weight_status(
                component_name="stitched_3d_model.special_tokens",
                expected_keys=expected_token_keys,
                loaded_keys=loaded_token_keys,
            )
    if load_independent_scene_flow_head and isinstance(stitched_model.stitched_3d_model, DA3Stitched):
        from utils.utils_for_thirdparty import _load_scene_flow_head_checkpoint

        _load_scene_flow_head_checkpoint(
            stitched_model.stitched_3d_model.model.scene_flow_head,
            Path(independent_scene_flow_head_checkpoint_path).expanduser(),
        )
    _apply_stitched_model_inference_dtype_policy(stitched_model, args, device)
    stitched_model = stitched_model.to(device)
    return stitched_model.eval()


@torch.no_grad()
def run_stitched_decode(
    args: argparse.Namespace,
    stitched_decoder: torch.nn.Module,
    prepared: dict[str, torch.Tensor],
    camera_extrinsics: torch.Tensor | None = None,
    camera_intrinsics: torch.Tensor | None = None,
) -> dict[str, Any]:
    from models.da3_stitched import (
        DA3Stitched,
        apply_official_da3_mono_sky_postprocess,
        canonicalize_da3_geometry_outputs,
        decode_official_da3_camera_params,
    )

    device = stitched_decoder.device
    vae_images = prepared["vae_images"].to(device)
    feedforward_images = prepared["feedforward_images"].to(device)
    force_fp32_inference = should_force_fp32_inference(args)
    if force_fp32_inference:
        vae_images = vae_images.float()
        feedforward_images = feedforward_images.float()

    def _move_runtime_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        value = value.to(device)
        return value.float() if force_fp32_inference else value

    with decoder_autocast(device, force_fp32_inference=force_fp32_inference):
        latents, _ = stitched_decoder.vae_encoder_forward(vae_images, decode=False)
        if force_fp32_inference:
            latents = latents.float()
        decoder_kwargs = {
            "feedforward_image": feedforward_images,
            "train": True,
            "camera_extrinsics": _move_runtime_tensor(camera_extrinsics),
            "camera_intrinsics": _move_runtime_tensor(camera_intrinsics),
        }
        if prepared.get("image_hw") is not None:
            decoder_kwargs["image_hw"] = prepared["image_hw"]
        stitched_outputs = stitched_decoder.forward_with_latent(
            latents,
            **decoder_kwargs,
        )

    if isinstance(stitched_decoder.stitched_3d_model, DA3Stitched):
        outputs = canonicalize_da3_geometry_outputs(stitched_outputs)
        outputs = apply_official_da3_mono_sky_postprocess(outputs)
        camera_params = decode_official_da3_camera_params(
            outputs["ray"],
            outputs["ray_conf"],
            image_height=feedforward_images.shape[-2],
            image_width=feedforward_images.shape[-1],
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
        }
        if "sky" in outputs:
            results["sky"] = outputs["sky"].detach().cpu()
        if "scene_flow" in outputs:
            results["scene_flow"] = outputs["scene_flow"].detach().cpu()
        if "scene_flow_cam" in outputs:
            results["scene_flow_cam"] = outputs["scene_flow_cam"].detach().cpu()

        period = getattr(args, "scene_flow_degrid_period", 0.0)
        if period:
            for key in ("scene_flow", "scene_flow_cam"):
                if key in results:
                    results[key] = filter_scene_flow_periodic(results[key], period)
            results["scene_flow_degrid_period"] = period
        return results

    raise TypeError("Plan4D requires a DA3 stitched decoder.")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_tensor_bundle(output_dir: Path, results: dict[str, Any]) -> None:
    tensor_bundle = {}
    for key in (
        "latents",
        "depth",
        "depth_conf",
        "ray",
        "ray_conf",
        "scene_flow",
        "scene_flow_cam",
        "scene_flow_degrid_period",
        "extrinsic_c2w",
        "extrinsic_w2c",
        "intrinsics_norm",
        "intrinsics_px",
        "point_maps",
        "camera_params",
    ):
        if key in results:
            tensor_bundle[key] = results[key]
    torch.save(tensor_bundle, output_dir / "decode_outputs.pt")


def save_video(frames: torch.Tensor, path: str, fps: int = 10) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("Expected video frames with shape (T, 3, H, W).")
    pixels = frames.detach().float().cpu().clamp(0, 1)
    pixels = pixels.permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).numpy()
    imageio.mimwrite(path, pixels, fps=fps, macro_block_size=1)


def save_visual_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    prepared: dict[str, torch.Tensor],
    results: dict[str, Any],
    stitched_decoder: torch.nn.Module,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_input_rgb:
        input_rgb = prepared["feedforward_frames_01"]
        save_video(input_rgb, str(output_dir / "input_rgb.mp4"), fps=10)

    depth_vis = colorize_depth_sequence(results["depth"][0])
    save_video(depth_vis, str(output_dir / "pred_depth.mp4"), fps=10)

    if args.save_depth_conf_video:
        depth_conf_vis = colorize_confidence_sequence(results["depth_conf"][0])
        save_video(depth_conf_vis, str(output_dir / "pred_depth_conf.mp4"), fps=10)

def save_inference_artifacts(
    args: argparse.Namespace,
    output_dir: Path,
    prepared: dict[str, torch.Tensor],
    results: dict[str, Any],
    stitched_decoder: torch.nn.Module,
    meta: dict[str, Any],
) -> None:
    if "scene_flow_degrid_period" in results:
        meta = {**meta, "scene_flow_degrid_period": results["scene_flow_degrid_period"]}
    save_json(
        output_dir / "meta.json",
        meta,
    )

    if args.save_tensors:
        save_tensor_bundle(output_dir, results)

    save_visual_outputs(args, output_dir, prepared, results, stitched_decoder)


def save_depth_evaluation_visuals(
    output_dir: Path,
    pred_depth_aligned: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    save_video(colorize_depth_sequence(pred_depth_aligned), str(output_dir / "aligned_pred_depth.mp4"), fps=10)
    save_video(colorize_depth_sequence(gt_depth), str(output_dir / "gt_depth.mp4"), fps=10)
    mask_vis = valid_mask.float().unsqueeze(1).repeat(1, 3, 1, 1)
    save_video(mask_vis, str(output_dir / "valid_mask.mp4"), fps=10)


def aggregate_metrics(metric_history: dict[str, list[float]]) -> dict[str, float]:
    aggregated = {}
    for metric_name, values in metric_history.items():
        aggregated[metric_name] = float(np.mean(values)) if values else 0.0
    return aggregated


def print_depth_metrics(metrics: dict[str, float]) -> None:
    print("\n" + "=" * 60)
    print("Depth Evaluation Results")
    print("=" * 60)
    print(f"  Abs Rel:             {metrics.get('depth_abs_rel', 0.0):.4f}")
    print(f"  Sq Rel:              {metrics.get('depth_sq_rel', 0.0):.4f}")
    print(f"  RMSE:                {metrics.get('depth_rmse', 0.0):.4f}")
    print(f"  RMSE log:            {metrics.get('depth_rmse_log', 0.0):.4f}")
    print(f"  δ < 1.25:            {metrics.get('depth_a1', 0.0):.4f}")
    print(f"  δ < 1.25²:           {metrics.get('depth_a2', 0.0):.4f}")
    print(f"  δ < 1.25³:           {metrics.get('depth_a3', 0.0):.4f}")
    print("=" * 60)


def run_single_input_inference(
    args: argparse.Namespace,
    stitched_decoder: torch.nn.Module,
) -> None:
    output_dir = Path(args.output_dir) / Path(args.input_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_max_frames = args.max_frames if args.max_frames is not None else 21

    logger.info(f"Loading input frames from {args.input_path}")
    frames = load_input_frames(
        args.input_path,
        max_frames=effective_max_frames,
        frame_stride=args.frame_stride,
    )
    logger.info(f"Loaded frames with shape {tuple(frames.shape)}")

    prepared = prepare_model_inputs_from_frames(
        frames,
        vae_resolution=args.resolution,
        feedforward_resolution=args.feedforward_resolution,
        da3_process_res_method=(
            args.da3_process_res_method if args.feedforward_model == "da3" else "square_center_crop"
        ),
    )

    logger.info("Running stitched decode inference")
    results = run_stitched_decode(args, stitched_decoder, prepared)

    save_inference_artifacts(
        args,
        output_dir,
        prepared,
        results,
        stitched_decoder,
        {
            "input_path": args.input_path,
            "checkpoint_path": args.checkpoint_path,
            "num_frames": int(prepared["feedforward_frames_01"].shape[0]),
            "max_frames": effective_max_frames,
            "resolution": args.resolution,
            "feedforward_resolution": args.feedforward_resolution,
            "da3_process_res_method": getattr(args, "da3_process_res_method", None),
            "stitching_layer_location": args.stitching_layer_location,
            "stitching_layer_config": str(args.stitching_layer_config),
        },
    )
    logger.info(f"Finished. Outputs saved to {output_dir}")


def run_point_odyssey_depth_evaluation(
    args: argparse.Namespace,
    stitched_decoder: torch.nn.Module,
) -> None:
    config_path, config = load_yaml_config(args.config)
    dataset_kwargs = build_point_odyssey_eval_dataset_kwargs(config, args)
    dataset_cls = import_point_odyssey_dataset(config_path)
    dataset = dataset_cls(**dataset_kwargs)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Created PointOdyssey dataset with {} samples from split '{}'".format(
            len(dataset), dataset_kwargs["split"]
        )
    )

    total_samples = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    metric_history = {
        "depth_abs_rel": [],
        "depth_sq_rel": [],
        "depth_rmse": [],
        "depth_rmse_log": [],
        "depth_a1": [],
        "depth_a2": [],
        "depth_a3": [],
    }
    per_sample_metrics = []

    for sample_idx in range(total_samples):
        sample = dataset[sample_idx]
        selected_indices = sample_frame_indices(
            total_frames=int(sample["rgbs"].shape[0]),
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )

        frames = rearrange(sample["rgbs"][selected_indices], "t h w c -> t c h w").contiguous()
        prepared = prepare_model_inputs_from_frames(
            frames,
            vae_resolution=args.resolution,
            feedforward_resolution=args.feedforward_resolution,
            da3_process_res_method=(
                args.da3_process_res_method
                if args.feedforward_model == "da3"
                else "square_center_crop"
            ),
        )
        results = run_stitched_decode(args, stitched_decoder, prepared)

        gt_depth = sample["depth"][selected_indices].unsqueeze(0).float()
        valid_mask = sample.get("valid_mask", (sample["depth"] > 0).float())[selected_indices].unsqueeze(0).float()
        pred_depth = normalize_depth_tensor(results["depth"])
        pred_depth_aligned = align_depth_scale_shift(pred_depth, gt_depth, valid_mask)
        depth_metrics = compute_depth_metrics(pred_depth_aligned, gt_depth, valid_mask)

        for metric_name, metric_value in depth_metrics.items():
            metric_history[metric_name].append(metric_value)

        absolute_start_idx = int(sample["start_idx"]) + selected_indices[0]
        absolute_end_idx = int(sample["start_idx"]) + selected_indices[-1]
        sample_record = {
            "sample_index": sample_idx,
            "scene_name": sample["scene_name"],
            "split": dataset_kwargs["split"],
            "start_idx": absolute_start_idx,
            "end_idx": absolute_end_idx,
            "num_frames": len(selected_indices),
            **depth_metrics,
        }
        per_sample_metrics.append(sample_record)
        logger.info(
            "[{}/{}] scene={} frames={} abs_rel={:.4f} rmse={:.4f} a1={:.4f}".format(
                sample_idx + 1,
                total_samples,
                sample["scene_name"],
                len(selected_indices),
                depth_metrics["depth_abs_rel"],
                depth_metrics["depth_rmse"],
                depth_metrics["depth_a1"],
            )
        )

        if sample_idx % args.visualize_every_n == 0:
            safe_scene_name = str(sample["scene_name"]).replace(os.sep, "_").replace(" ", "_")
            sample_output_dir = output_root / f"sample_{sample_idx:04d}_{safe_scene_name}"
            save_inference_artifacts(
                args,
                sample_output_dir,
                prepared,
                results,
                stitched_decoder,
                {
                    "mode": "point_odyssey_depth_eval",
                    "config_path": str(config_path),
                    "checkpoint_path": args.checkpoint_path,
                    "scene_name": sample["scene_name"],
                    "split": dataset_kwargs["split"],
                    "start_idx": absolute_start_idx,
                    "end_idx": absolute_end_idx,
                    "da3_process_res_method": getattr(args, "da3_process_res_method", None),
                    "num_frames": len(selected_indices),
                    "resolution": args.resolution,
                    "feedforward_resolution": args.feedforward_resolution,
                    "stitching_layer_location": args.stitching_layer_location,
                    "stitching_layer_config": str(args.stitching_layer_config),
                    "depth_metrics": depth_metrics,
                },
            )
            save_depth_evaluation_visuals(
                sample_output_dir,
                pred_depth_aligned[0],
                gt_depth[0],
                valid_mask[0],
            )

    aggregated_metrics = aggregate_metrics(metric_history)
    summary = {
        "mode": "point_odyssey_depth_eval",
        "config_path": str(config_path),
        "checkpoint_path": args.checkpoint_path,
        "dataset": dataset_kwargs,
        "num_evaluated_samples": total_samples,
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
        "metrics": aggregated_metrics,
    }
    save_json(output_root / "metrics.json", summary)
    save_json(output_root / "per_sample_metrics.json", per_sample_metrics)
    print_depth_metrics(aggregated_metrics)
    logger.info(f"Depth evaluation complete. Metrics saved to {output_root / 'metrics.json'}")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    apply_output_mode(args)
    checkpoint_plan = build_checkpoint_load_plan(args.checkpoint_path)
    preserve_fields = getattr(
        args, "explicit_scene_flow_fields", ("da3_scene_flow_head_checkpoint_path",)
    )
    apply_saved_checkpoint_args(args, checkpoint_plan, preserve_fields=preserve_fields)
    apply_feedforward_model_inference_defaults(args)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_plan.checkpoint_dir is not None:
        logger.info(
            "Resolved epoch checkpoint dir {} (full_state={}, lightweight={})",
            checkpoint_plan.checkpoint_dir,
            checkpoint_plan.full_state_checkpoint_path,
            checkpoint_plan.lightweight_checkpoint_path,
        )

    logger.info("Loading stitched decoder")
    stitched_decoder = load_stitched_decoder(
        args, checkpoint_plan=checkpoint_plan, preserve_explicit_fields=preserve_fields
    )
    logger.info(f"Stitched decoder loaded on {stitched_decoder.device}")

    if args.config is not None:
        run_point_odyssey_depth_evaluation(args, stitched_decoder)
    else:
        run_single_input_inference(args, stitched_decoder)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(parse_args())
