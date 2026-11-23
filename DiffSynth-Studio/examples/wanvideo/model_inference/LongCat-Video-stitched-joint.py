import argparse
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
DIFFSYNTH_ROOT = THIS_FILE.parents[3]
PROJECT_ROOT = THIS_FILE.parents[4]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for candidate in (DIFFSYNTH_ROOT, PROJECT_ROOT, DA3_ROOT, DA3_SRC_ROOT, PLAN4D_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from inference_stitched_decode import (
    apply_saved_checkpoint_args,
    build_checkpoint_load_plan,
    colorize_depth_sequence,
    decoder_autocast,
    load_stitched_decoder,
)
from models.da3_stitched import (
    canonicalize_da3_geometry_outputs,
    decode_official_da3_camera_params,
)
from models.stitching_layer_builder import parse_conv_spec
from utils.da3_resize import build_upper_bound_resize_transform
from depth_anything_3.utils.scene_flow_postprocess import (
    filter_scene_flow_periodic,
    scene_flow_filter_period,
)


WAN_SPATIAL_DIVISION_FACTOR = 16
DEFAULT_LONGCAT_WEIGHT_ROOT = str(PROJECT_ROOT / "checkpoints" / "longcat")
DEFAULT_LONGCAT_VAE_FILE = str(PROJECT_ROOT / "checkpoints" / "Wan2.1_VAE.safetensors")
DEFAULT_TOKENIZER_PATH = f"{DEFAULT_LONGCAT_WEIGHT_ROOT}/tokenizer"
DEFAULT_WAN_VAE_PATH = f"{DEFAULT_LONGCAT_WEIGHT_ROOT}/vae"
DEFAULT_DA3_MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "da3")
DEFAULT_DIT_CHECKPOINT_PATH = None
#DEFAULT_DIT_CHECKPOINT_PATH = ""
DEFAULT_DIT_LORA_TARGET_MODULES = (
    "adaLN_modulation.1,attn.qkv,attn.proj,cross_attn.q_linear,"
    "cross_attn.kv_linear,cross_attn.proj,ffn.w1,ffn.w2,ffn.w3"
)
DEFAULT_DIT_LORA_RANK = 32
DEFAULT_DIT_LORA_ALPHA = 32
DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints" / "stitched_decoder.pth")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "stitched_joint_inference_outputs"
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)
_INDEX_FILE_NAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_SINGLE_FILE_NAMES = (
    "diffusion_pytorch_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",
)
STITCHED_GEOMETRY_PASSTHROUGH_KEYS = (
    "camera_tokens",
    "camera_pose_encoding",
    "camera_extrinsics",
    "camera_intrinsics",
    "camera_image_hw",
    "camera_focal_lengths",
    "camera_principal_points",
    "scene_flow",
    "scene_flow_cam",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stitched joint LongCat inference and export video, depth, ray, and point cloud artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, required=True, help="Positive text prompt.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--task_type",
        type=str,
        default="ti2v",
        choices=["ti2v", "continuation", "t2v"],
        help="LongCat generation mode.",
    )
    parser.add_argument(
        "--condition_path",
        type=str,
        default=None,
        help="Condition image, video, or frame directory. Required for ti2v and continuation.",
    )
    parser.add_argument(
        "--continuation_cond_frames",
        type=int,
        default=17,
        help="Requested continuation frame count before LongCat 4n+1 normalization.",
    )
    parser.add_argument("--height", type=int, default=504)
    parser.add_argument("--width", type=int, default=504)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--max_frames", type=int, default=49, help="Stitched decoder temporal size.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma_shift", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--switch_dit_boundary", type=float, default=0.875)
    parser.add_argument("--rand_device", type=str, default="cpu")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--tiled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Wan VAE tiling during decode.",
    )
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26))
    parser.add_argument(
        "--save_latents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save diffusion latents.",
    )
    parser.add_argument(
        "--save_point_map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save assembled world point maps.",
    )
    parser.add_argument(
        "--save_point_cloud_ply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export sampled per-frame and merged point clouds as PLY.",
    )
    parser.add_argument(
        "--save_merged_point_cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export one merged point cloud in addition to per-frame PLY files.",
    )
    parser.add_argument("--max_points_per_frame", type=int, default=200000)
    parser.add_argument("--max_points_merged", type=int, default=1000000)
    parser.add_argument("--longcat_weight_root", type=str, default=DEFAULT_LONGCAT_WEIGHT_ROOT)
    parser.add_argument("--longcat_vae_file", type=str, default=DEFAULT_LONGCAT_VAE_FILE)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--checkpoint_path",
        "--dit_checkpoint_path",
        dest="dit_checkpoint_path",
        type=str,
        default=DEFAULT_DIT_CHECKPOINT_PATH,
        help="Joint-trained LongCat DiT checkpoint that overrides the base LongCat DiT weights.",
    )
    parser.add_argument(
        "--dit_lora_path",
        type=str,
        default=None,
        help="Optional DiT LoRA checkpoint to apply on top of a frozen base LongCat DiT.",
    )
    parser.add_argument(
        "--dit_lora_base_checkpoint_path",
        type=str,
        default=None,
        help="Optional base DiT checkpoint for LoRA inference. Defaults to the pretrained LongCat DiT under longcat_weight_root/dit.",
    )
    parser.add_argument(
        "--dit_lora_target_modules",
        type=str,
        default=DEFAULT_DIT_LORA_TARGET_MODULES,
        help="Comma-separated DiT module names used when injecting the LoRA adapter.",
    )
    parser.add_argument("--dit_lora_rank", type=int, default=DEFAULT_DIT_LORA_RANK)
    parser.add_argument("--dit_lora_alpha", type=int, default=DEFAULT_DIT_LORA_ALPHA)
    parser.add_argument(
        "--stitched_decoder_checkpoint_path",
        type=str,
        default=DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH,
        help="Frozen stitched decoder checkpoint used for geometry decode.",
    )
    parser.add_argument("--feedforward_model", type=str, default="da3", choices=["da3"])
    parser.add_argument("--video_model", type=str, default="wan", choices=["wan"])
    parser.add_argument("--da3_model_path", type=str, default=DEFAULT_DA3_MODEL_PATH)
    parser.add_argument("--da3_config_name", type=str, default="config.json")
    parser.add_argument("--da3_checkpoint_name", type=str, default="model.safetensors")
    parser.add_argument("--da3_ref_view_strategy", type=str, default="saddle_balanced")
    parser.add_argument(
        "--da3_enable_scene_flow_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach the DA3 scene-flow head to the stitched DA3 decoder.",
    )
    parser.add_argument(
        "--da3_scene_flow_head_checkpoint_path",
        type=str,
        default=None,
        help="Optional standalone DA3 scene-flow head checkpoint. Omit it to use weights from the stitched decoder checkpoint.",
    )
    parser.add_argument(
        "--da3_scene_flow_preset",
        type=str,
        default="da3-giant-sceneflow",
        help="DA3 registry preset used to build the scene-flow head and adapter.",
    )
    parser.add_argument("--wan_vae_path", type=str, default=DEFAULT_WAN_VAE_PATH)
    parser.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--wan_vae_repo_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--stitching_layer_location", type=str, default="enc_blocks_2")
    parser.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default=parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1"),
        metavar="CONV_SPEC",
    )
    parser.add_argument("--lora_config", type=str, default="r64,a32,d0.0,f0")
    parser.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resolution", type=int, default=504)
    parser.add_argument("--feedforward_resolution", type=int, default=504)
    parser.add_argument(
        "--da3_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize"],
        help="DA3/LongCat preprocessing protocol. Inference follows manifest upper_bound_resize semantics.",
    )
    parser.add_argument(
        "--scene_flow_degrid_period", "--scene-flow-degrid-period",
        type=scene_flow_filter_period, default=0.0, help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.task_type in {"ti2v", "continuation"} and not args.condition_path:
        raise ValueError("condition_path is required for ti2v and continuation inference.")
    if args.num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if args.continuation_cond_frames <= 0:
        raise ValueError("continuation_cond_frames must be positive.")
    if args.max_points_per_frame <= 0 or args.max_points_merged <= 0:
        raise ValueError("Point cloud sampling limits must be positive.")
    if args.dit_lora_rank <= 0 or args.dit_lora_alpha <= 0:
        raise ValueError("dit_lora_rank and dit_lora_alpha must be positive.")
    if args.dit_lora_base_checkpoint_path and not args.dit_lora_path:
        raise ValueError("dit_lora_base_checkpoint_path requires dit_lora_path.")


def compute_effective_generation_shape(
    height: int,
    width: int,
    num_frames: int,
    *,
    height_division_factor: int = 16,
    width_division_factor: int = 16,
    time_division_factor: int = 4,
    time_division_remainder: int = 1,
) -> tuple[int, int, int]:
    if height % height_division_factor != 0:
        height = (height + height_division_factor - 1) // height_division_factor * height_division_factor
    if width % width_division_factor != 0:
        width = (width + width_division_factor - 1) // width_division_factor * width_division_factor
    if num_frames % time_division_factor != time_division_remainder:
        num_frames = (num_frames + time_division_factor - 1) // time_division_factor * time_division_factor + time_division_remainder
    return int(height), int(width), int(num_frames)


def normalize_condition_frame_count(
    task_type: str,
    requested_frames: int,
    available_frames: int,
) -> int:
    if available_frames <= 0:
        raise ValueError("available_frames must be positive.")
    if task_type == "ti2v":
        return 1
    effective = min(max(int(requested_frames), 1), int(available_frames))
    return max(1, effective - ((effective - 1) % 4))


def build_generation_shape_notice(args: argparse.Namespace) -> str | None:
    effective_height, effective_width, _ = compute_effective_generation_shape(
        args.height,
        args.width,
        args.num_frames,
    )
    if effective_height == args.height and effective_width == args.width:
        return None
    if args.condition_path is not None:
        return (
            f"requested={args.height}x{args.width}, "
            f"effective={effective_height}x{effective_width}, "
            "condition frames resized with upper_bound_resize."
        )
    return (
        f"requested={args.height}x{args.width}, "
        f"effective={effective_height}x{effective_width}."
    )


def _natural_sort_key(path: Path) -> list[Any]:
    return [int(item) if item.isdigit() else item.lower() for item in re.split(r"(\d+)", path.name)]


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _resize_frame_upper_bound(frame: Image.Image, target_size: int, *, patch_size: int) -> Image.Image:
    transform = build_upper_bound_resize_transform(
        height=frame.height,
        width=frame.width,
        target_size=target_size,
        patch_size=patch_size,
    )
    return frame.resize((transform.resized_width, transform.resized_height), Image.BICUBIC)


def _load_frames_from_source(source: Path) -> list[Image.Image]:
    if not source.exists():
        raise FileNotFoundError(f"Condition source does not exist: {source}")

    frames: list[Image.Image] = []
    if source.is_dir():
        frame_paths = sorted([item for item in source.iterdir() if item.is_file() and _is_image_file(item)], key=_natural_sort_key)
        for frame_path in frame_paths:
            frames.append(Image.open(frame_path).convert("RGB"))
        return frames

    if _is_image_file(source):
        return [Image.open(source).convert("RGB")]

    for frame in iio.imiter(source):
        pil_frame = Image.fromarray(np.asarray(frame)[..., :3]).convert("RGB")
        frames.append(pil_frame)
    return frames


def load_condition_frames(args: argparse.Namespace) -> list[Image.Image] | None:
    if args.condition_path is None:
        return None

    source_frames = _load_frames_from_source(Path(args.condition_path))
    if len(source_frames) == 0:
        raise ValueError(f"No frames found under condition_path={args.condition_path}")

    da3_frames = [
        _resize_frame_upper_bound(frame, args.feedforward_resolution, patch_size=14)
        for frame in source_frames
    ]
    wan_frames = [
        _resize_frame_upper_bound(frame, args.feedforward_resolution, patch_size=WAN_SPATIAL_DIVISION_FACTOR)
        for frame in source_frames
    ]
    da3_height, da3_width = da3_frames[0].height, da3_frames[0].width
    wan_height, wan_width = wan_frames[0].height, wan_frames[0].width
    if any(frame.height != da3_height or frame.width != da3_width for frame in da3_frames):
        raise ValueError("All condition frames must resolve to the same DA3 upper_bound_resize size.")
    if any(frame.height != wan_height or frame.width != wan_width for frame in wan_frames):
        raise ValueError("All condition frames must resolve to the same Wan upper_bound_resize size.")
    args.feedforward_image_hw = (da3_height, da3_width)
    args.longcat_condition_hw = (wan_height, wan_width)
    args.height = wan_height
    args.width = wan_width

    if args.task_type == "ti2v":
        return [wan_frames[0]]
    if args.task_type == "continuation":
        cond_frames = normalize_condition_frame_count(
            "continuation",
            requested_frames=args.continuation_cond_frames,
            available_frames=len(wan_frames),
        )
        return wan_frames[-cond_frames:]
    return None


def _resolve_checkpoint_path(path_like: str) -> str | list[str]:
    path = Path(path_like)
    if path.is_file():
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    for index_name in _INDEX_FILE_NAMES:
        index_path = path / index_name
        if index_path.is_file():
            index_data = json.loads(index_path.read_text())
            weight_map = index_data.get("weight_map", {})
            shard_names = list(dict.fromkeys(weight_map.values()))
            if len(shard_names) == 0:
                raise ValueError(f"Shard index has no weight_map entries: {index_path}")
            return [str(path / shard_name) for shard_name in shard_names]

    for file_name in _SINGLE_FILE_NAMES:
        file_path = path / file_name
        if file_path.is_file():
            return str(file_path)

    safetensors_files = sorted(str(file_path) for file_path in path.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]
    if len(safetensors_files) > 1:
        return safetensors_files

    bin_files = sorted(str(file_path) for file_path in path.glob("*.bin"))
    if len(bin_files) == 1:
        return bin_files[0]
    if len(bin_files) > 1:
        return bin_files

    raise ValueError(f"Unsupported checkpoint directory layout: {path}")


def build_longcat_model_paths(
    weight_root: str,
    vae_file: str,
    *,
    dit_checkpoint_path: str | None = None,
) -> list[str | list[str]]:
    weight_root_path = Path(weight_root)
    dit_path = (
        _resolve_checkpoint_path(dit_checkpoint_path)
        if dit_checkpoint_path
        else _resolve_checkpoint_path(str(weight_root_path / "dit"))
    )
    return [
        dit_path,
        _resolve_checkpoint_path(str(weight_root_path / "text_encoder")),
        _resolve_checkpoint_path(vae_file),
    ]


def resolve_dit_checkpoint_override(args: argparse.Namespace) -> str | None:
    if args.dit_lora_path:
        return args.dit_lora_base_checkpoint_path
    return args.dit_checkpoint_path


def _split_csv_items(items: str) -> list[str]:
    return [item.strip() for item in items.split(",") if item.strip()]


def _load_state_dict(path: str, *, torch_dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    from diffsynth.core import load_state_dict

    return load_state_dict(path, torch_dtype=torch_dtype, device=device)


def _inject_dit_lora_adapter(
    model: torch.nn.Module,
    *,
    target_modules: list[str],
    lora_rank: int,
    lora_alpha: int,
) -> torch.nn.Module:
    from peft import LoraConfig, inject_adapter_in_model

    peft_target_modules: str | list[str]
    peft_target_modules = target_modules[0] if len(target_modules) == 1 else target_modules
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=peft_target_modules,
    )
    return inject_adapter_in_model(lora_config, model)


def _normalize_dit_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key or "lora_B.weight" in key:
            key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
        normalized[key] = value
    return normalized


def summarize_dit_lora_state_dict_keys(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    expected_lora_keys = sorted(key for key in model.state_dict().keys() if "lora_" in key)
    provided_lora_keys = sorted(key for key in state_dict.keys() if "lora_" in key)
    non_lora_keys = sorted(key for key in state_dict.keys() if "lora_" not in key)

    expected_lora_key_set = set(expected_lora_keys)
    provided_lora_key_set = set(provided_lora_keys)
    missing_keys = sorted(expected_lora_key_set - provided_lora_key_set)
    unexpected_keys = sorted(provided_lora_key_set - expected_lora_key_set)
    return missing_keys, unexpected_keys, non_lora_keys


def apply_dit_lora_to_pipeline(pipe, args: argparse.Namespace) -> None:
    if not args.dit_lora_path:
        return

    target_modules = _split_csv_items(args.dit_lora_target_modules)
    if len(target_modules) == 0:
        raise ValueError("dit_lora_target_modules must contain at least one module name.")

    pipe.dit = _inject_dit_lora_adapter(
        pipe.dit,
        target_modules=target_modules,
        lora_rank=args.dit_lora_rank,
        lora_alpha=args.dit_lora_alpha,
    )
    state_dict = _load_state_dict(
        args.dit_lora_path,
        torch_dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    state_dict = _normalize_dit_lora_state_dict(state_dict)
    missing_keys, unexpected_keys, non_lora_keys = summarize_dit_lora_state_dict_keys(pipe.dit, state_dict)
    pipe.dit.load_state_dict(state_dict, strict=False)
    print(
        "[INFO] Loaded DiT LoRA from "
        f"{args.dit_lora_path} "
        f"(missing_keys={missing_keys}, unexpected_keys={unexpected_keys}, non_lora_keys={non_lora_keys})"
    )
    if missing_keys or unexpected_keys or non_lora_keys:
        raise RuntimeError(
            "Failed to load DiT LoRA checkpoint from "
            f"{args.dit_lora_path} "
            f"(missing_keys={missing_keys}, unexpected_keys={unexpected_keys}, non_lora_keys={non_lora_keys})"
        )
    pipe.dit.to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.eval()
    pipe.dit.requires_grad_(False)


def _select_pipeline_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_longcat_pipeline(args: argparse.Namespace):
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    model_paths = build_longcat_model_paths(
        args.longcat_weight_root,
        args.longcat_vae_file,
        dit_checkpoint_path=resolve_dit_checkpoint_override(args),
    )
    model_configs = [ModelConfig(path=path, skip_download=True) for path in model_paths]
    tokenizer_config = ModelConfig(path=args.tokenizer_path, skip_download=True)
    device = torch.device(args.device)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=_select_pipeline_dtype(device),
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    apply_dit_lora_to_pipeline(pipe, args)
    return pipe


def apply_longcat_stitched_decoder_inference_defaults(args: argparse.Namespace) -> None:
    if args.feedforward_model != "da3":
        return
    if args.stitching_layer_config.out_channels != 1536:
        args.stitching_layer_config = parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1")
    if getattr(args, "da3_process_res_method", None) is None:
        args.da3_process_res_method = "upper_bound_resize"


def build_stitched_decoder_checkpoint_plan(args: argparse.Namespace):
    checkpoint_plan = build_checkpoint_load_plan(args.stitched_decoder_checkpoint_path)
    apply_saved_checkpoint_args(
        args,
        checkpoint_plan,
        preserve_fields=(
            "feedforward_resolution",
            "resolution",
            "da3_enable_scene_flow_head",
            "da3_scene_flow_head_checkpoint_path",
            "da3_scene_flow_preset",
        ),
    )
    apply_longcat_stitched_decoder_inference_defaults(args)
    return checkpoint_plan


def generate_video_and_latents(
    pipe,
    args: argparse.Namespace,
    condition_frames: list[Image.Image] | None,
) -> dict[str, Any]:
    pipe.scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=args.sigma_shift)
    inputs_posi = {
        "prompt": args.prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": args.num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": args.negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": args.num_inference_steps,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": args.seed,
        "rand_device": args.rand_device,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "cfg_scale": args.cfg_scale,
        "cfg_merge": False,
        "sigma_shift": args.sigma_shift,
        "motion_bucket_id": None,
        "longcat_video": condition_frames,
        "tiled": args.tiled,
        "tile_size": tuple(args.tile_size),
        "tile_stride": tuple(args.tile_stride),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    effective_height = int(inputs_shared["height"])
    effective_width = int(inputs_shared["width"])
    effective_frames = int(inputs_shared["num_frames"])

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="Generating")):
        if (
            timestep.item() < args.switch_dit_boundary * 1000
            and pipe.dit2 is not None
            and models["dit"] is not pipe.dit2
        ):
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2

        timestep_tensor = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep_tensor)
        if args.cfg_scale != 1.0:
            noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep_tensor)
            noise_pred = noise_pred_nega + args.cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred,
            pipe.scheduler.timesteps[progress_id],
            inputs_shared["latents"],
        )
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    for unit in pipe.post_units:
        inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    latents = inputs_shared["latents"].detach()
    pipe.load_models_to_device(["vae"])
    decoded_video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=args.tiled,
        tile_size=tuple(args.tile_size),
        tile_stride=tuple(args.tile_stride),
    )
    quantized_frames = pipe.vae_output_to_video(decoded_video)
    pipe.load_models_to_device([])
    return {
        "latents": latents,
        "decoded_video": decoded_video.detach(),
        "video_frames": quantized_frames,
        "effective_height": effective_height,
        "effective_width": effective_width,
        "effective_frames": effective_frames,
    }


def run_stitched_geometry(
    args: argparse.Namespace,
    stitched_decoder: torch.nn.Module,
    latents: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device(args.device)
    image_hw = getattr(args, "feedforward_image_hw", None) or (args.feedforward_resolution, args.feedforward_resolution)
    with decoder_autocast(device):
        outputs = stitched_decoder.forward_with_latent(
            latents.to(device=device, dtype=stitched_decoder.stitching_layer.weight.dtype),
            feedforward_image=None,
            image_hw=image_hw,
            train=False,
        )
    geometry = canonicalize_da3_geometry_outputs(outputs)
    for key in STITCHED_GEOMETRY_PASSTHROUGH_KEYS:
        if key in outputs:
            geometry[key] = outputs[key]

    if "camera_extrinsics" not in geometry or "camera_intrinsics" not in geometry:
        decoded_camera = decode_official_da3_camera_params(
            geometry["ray"],
            geometry.get("ray_conf"),
            image_height=int(image_hw[0]),
            image_width=int(image_hw[1]),
        )
        geometry.setdefault("camera_extrinsics", decoded_camera["extrinsics"])
        geometry.setdefault("camera_intrinsics", decoded_camera["intrinsics"])
        geometry.setdefault("camera_focal_lengths", decoded_camera["focal_lengths"])
        geometry.setdefault("camera_principal_points", decoded_camera["principal_points"])

    geometry_cpu: dict[str, Any] = {}
    for key, value in geometry.items():
        if isinstance(value, torch.Tensor):
            geometry_cpu[key] = value.detach().cpu()
        else:
            geometry_cpu[key] = value
    period = getattr(args, "scene_flow_degrid_period", 0.0)
    if period:
        for key in ("scene_flow", "scene_flow_cam"):
            if key in geometry_cpu:
                geometry_cpu[key] = filter_scene_flow_periodic(geometry_cpu[key], period)
    return geometry_cpu


def _to_uint8_video_from_tensor(
    video_bcthw: torch.Tensor,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    video = video_bcthw.detach().cpu().float()
    if video.dim() != 5:
        raise ValueError(f"Expected video tensor [B,C,T,H,W], got {tuple(video.shape)}")
    if target_hw is not None and tuple(video.shape[-2:]) != tuple(target_hw):
        video = F.interpolate(
            video,
            size=(video.shape[2], target_hw[0], target_hw[1]),
            mode="trilinear",
            align_corners=False,
        )
    video = ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return video[0].permute(1, 2, 3, 0).contiguous().numpy()


def _to_uint8_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.float()
        if tensor.max().item() > 1.0 or tensor.min().item() < 0.0:
            if tensor.min().item() >= -1.0 and tensor.max().item() <= 1.0:
                tensor = (tensor + 1.0) * 0.5
            else:
                tensor = tensor / 255.0
        array = (tensor.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return array
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    return array


def _visualize_ray_direction_frame(ray_6hw: np.ndarray) -> np.ndarray:
    direction = ray_6hw[:3].astype(np.float32)
    norm = np.linalg.norm(direction, axis=0, keepdims=True)
    direction = direction / np.clip(norm, 1e-6, None)
    rgb = ((direction.transpose(1, 2, 0) + 1.0) * 0.5 * 255.0).clip(0, 255)
    return rgb.astype(np.uint8)


def visualize_ray_direction_sequence(ray: torch.Tensor) -> list[np.ndarray]:
    ray_np = ray.detach().float().cpu().numpy()
    return [_visualize_ray_direction_frame(ray_np[0, frame_idx]) for frame_idx in range(ray_np.shape[1])]


def save_video_frames(frames: Sequence[Any] | np.ndarray | torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(frames, torch.Tensor):
        iterable = [frames[index] for index in range(frames.shape[0])]
    elif isinstance(frames, np.ndarray) and frames.ndim == 4:
        iterable = [frames[index] for index in range(frames.shape[0])]
    else:
        iterable = list(frames)

    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in iterable:
            writer.append_data(_to_uint8_frame(frame))


def save_camera_artifacts(output_dir: Path, geometry: dict[str, Any]) -> None:
    camera_artifacts = {
        "camera_pose_encoding": "pred_camera_pose_encoding_raw.pt",
        "camera_extrinsics": "pred_camera_extrinsics_raw.pt",
        "camera_intrinsics": "pred_camera_intrinsics_raw.pt",
        "camera_focal_lengths": "pred_camera_focal_lengths_raw.pt",
        "camera_principal_points": "pred_camera_principal_points_raw.pt",
        "camera_tokens": "pred_camera_tokens_raw.pt",
    }
    for key, filename in camera_artifacts.items():
        if key in geometry and isinstance(geometry[key], torch.Tensor):
            torch.save(geometry[key], output_dir / filename)

    payload = {}
    if "camera_image_hw" in geometry:
        camera_image_hw = geometry["camera_image_hw"]
        if isinstance(camera_image_hw, (tuple, list)) and len(camera_image_hw) == 2:
            payload["camera_image_hw"] = [int(camera_image_hw[0]), int(camera_image_hw[1])]
    if "camera_extrinsics" in geometry and isinstance(geometry["camera_extrinsics"], torch.Tensor):
        payload["camera_extrinsics_shape"] = list(geometry["camera_extrinsics"].shape)
    if "camera_intrinsics" in geometry and isinstance(geometry["camera_intrinsics"], torch.Tensor):
        payload["camera_intrinsics_shape"] = list(geometry["camera_intrinsics"].shape)
    if "camera_focal_lengths" in geometry and isinstance(geometry["camera_focal_lengths"], torch.Tensor):
        payload["camera_focal_lengths_shape"] = list(geometry["camera_focal_lengths"].shape)
    if "camera_principal_points" in geometry and isinstance(geometry["camera_principal_points"], torch.Tensor):
        payload["camera_principal_points_shape"] = list(geometry["camera_principal_points"].shape)
    if payload:
        with open(output_dir / "pred_camera_meta.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)


def _ensure_da3_eval_imports() -> None:
    preferred = [str(DA3_ROOT), str(DA3_SRC_ROOT), str(PLAN4D_ROOT)]
    remaining = [entry for entry in sys.path if entry not in preferred]
    sys.path[:] = preferred + remaining


def _load_da3_eval_module():
    _ensure_da3_eval_imports()
    module_name = "plan4d_da3_eval_scene_flow_head"
    if module_name in sys.modules:
        return sys.modules[module_name]

    eval_path = DA3_ROOT / "training" / "eval_scene_flow_head.py"
    spec = importlib.util.spec_from_file_location(module_name, eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DA3 eval module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def align_depth_and_ray_for_point_cloud(
    depth: torch.Tensor,
    ray: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_da3_eval_module()
    aligned_depth = module.standardize_depth_prediction(depth)
    aligned_ray = module.standardize_ray_prediction(ray, target_size=aligned_depth.shape[-2:])
    return aligned_depth, aligned_ray


def build_point_map_from_geometry(depth: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    module = _load_da3_eval_module()
    return module.build_world_point_map_from_depth_and_ray(depth, ray)


def write_binary_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    payload = np.empty(
        vertex_count,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    payload["x"] = points[:, 0]
    payload["y"] = points[:, 1]
    payload["z"] = points[:, 2]
    payload["red"] = colors[:, 0]
    payload["green"] = colors[:, 1]
    payload["blue"] = colors[:, 2]
    with open(path, "wb") as file:
        file.write(header.encode("ascii"))
        payload.tofile(file)


def _sample_point_cloud(
    point_map: np.ndarray,
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(valid_mask, dtype=bool)
    valid &= np.isfinite(point_map).all(axis=-1)
    valid &= np.isfinite(np.linalg.norm(point_map, axis=-1))
    valid &= np.linalg.norm(point_map, axis=-1) > 1e-8

    points = point_map[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3)
    if points.shape[0] == 0:
        return points, colors
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    return points.astype(np.float32), colors.astype(np.uint8)


def export_point_clouds(
    output_dir: Path,
    point_map_world: torch.Tensor,
    depth_aligned: torch.Tensor,
    rgb_video_hwc: np.ndarray,
    args: argparse.Namespace,
) -> None:
    point_cloud_dir = output_dir / "point_cloud"
    point_map_np = point_map_world[0].cpu().numpy()
    depth_np = depth_aligned[0, :, 0].cpu().numpy()

    merged_points: list[np.ndarray] = []
    merged_colors: list[np.ndarray] = []
    for frame_idx in range(point_map_np.shape[0]):
        valid_mask = depth_np[frame_idx] > 0
        points, colors = _sample_point_cloud(
            point_map=point_map_np[frame_idx],
            rgb=rgb_video_hwc[frame_idx],
            valid_mask=valid_mask,
            max_points=args.max_points_per_frame,
        )
        if points.shape[0] == 0:
            continue
        write_binary_point_cloud(point_cloud_dir / f"frame_{frame_idx:04d}.ply", points, colors)
        if args.save_merged_point_cloud:
            merged_points.append(points)
            merged_colors.append(colors)

    if args.save_merged_point_cloud and len(merged_points) > 0:
        merged_points_np = np.concatenate(merged_points, axis=0)
        merged_colors_np = np.concatenate(merged_colors, axis=0)
        if merged_points_np.shape[0] > args.max_points_merged:
            rng = np.random.default_rng(0)
            indices = rng.choice(merged_points_np.shape[0], size=args.max_points_merged, replace=False)
            merged_points_np = merged_points_np[indices]
            merged_colors_np = merged_colors_np[indices]
        write_binary_point_cloud(point_cloud_dir / "merged.ply", merged_points_np, merged_colors_np)


def build_output_dir(args: argparse.Namespace) -> Path:
    base_dir = Path(args.output_dir)
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_condition_artifacts(output_dir: Path, condition_frames: list[Image.Image] | None, fps: int) -> None:
    if condition_frames is None:
        return
    if len(condition_frames) == 1:
        condition_frames[0].save(output_dir / "condition_image.png")
    else:
        save_video_frames(condition_frames, output_dir / "condition_video.mp4", fps=fps)


def save_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    generation: dict[str, Any],
    geometry: dict[str, Any],
    condition_frames: list[Image.Image] | None,
) -> None:
    raw_video_hwc = np.stack([_to_uint8_frame(frame) for frame in generation["video_frames"]], axis=0)
    image_hw = getattr(args, "feedforward_image_hw", None) or (args.feedforward_resolution, args.feedforward_resolution)
    aligned_video_hwc = _to_uint8_video_from_tensor(
        generation["decoded_video"],
        target_hw=(int(image_hw[0]), int(image_hw[1])),
    )

    save_video_frames(raw_video_hwc, output_dir / "generated_video.mp4", fps=args.fps)
    if raw_video_hwc.shape[1:3] != aligned_video_hwc.shape[1:3]:
        save_video_frames(aligned_video_hwc, output_dir / "generated_video_geometry.mp4", fps=args.fps)

    save_condition_artifacts(output_dir, condition_frames, fps=args.fps)

    if args.save_latents:
        torch.save(generation["latents"].cpu(), output_dir / "latents.pt")

    torch.save(geometry["depth"], output_dir / "pred_depth_raw.pt")
    torch.save(geometry["ray"], output_dir / "pred_ray_raw.pt")
    torch.save(geometry["depth_conf"], output_dir / "pred_depth_conf_raw.pt")
    torch.save(geometry["ray_conf"], output_dir / "pred_ray_conf_raw.pt")
    if "scene_flow" in geometry and isinstance(geometry["scene_flow"], torch.Tensor):
        torch.save(geometry["scene_flow"], output_dir / "pred_scene_flow_raw.pt")
    if "scene_flow_cam" in geometry and isinstance(geometry["scene_flow_cam"], torch.Tensor):
        torch.save(geometry["scene_flow_cam"], output_dir / "pred_scene_flow_cam_raw.pt")
    save_camera_artifacts(output_dir, geometry)

    depth_vis = colorize_depth_sequence(geometry["depth"][0])
    save_video_frames(depth_vis, output_dir / "pred_depth_vis.mp4", fps=args.fps)
    ray_vis = visualize_ray_direction_sequence(geometry["ray"])
    save_video_frames(ray_vis, output_dir / "pred_ray_vis.mp4", fps=args.fps)

    depth_aligned, ray_aligned = align_depth_and_ray_for_point_cloud(geometry["depth"], geometry["ray"])
    point_map_world = build_point_map_from_geometry(depth_aligned, ray_aligned)
    if args.save_point_map:
        torch.save(point_map_world, output_dir / "point_map_world.pt")
        torch.save(depth_aligned, output_dir / "pred_depth_for_point_cloud.pt")
        torch.save(ray_aligned, output_dir / "pred_ray_for_point_cloud.pt")

    if args.save_point_cloud_ply:
        export_point_clouds(
            output_dir=output_dir,
            point_map_world=point_map_world,
            depth_aligned=depth_aligned,
            rgb_video_hwc=aligned_video_hwc,
            args=args,
        )

    meta = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "task_type": args.task_type,
        "condition_path": args.condition_path,
        "num_condition_frames": 0 if condition_frames is None else len(condition_frames),
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "sigma_shift": args.sigma_shift,
        "num_inference_steps": args.num_inference_steps,
        "dit_checkpoint_path": args.dit_checkpoint_path,
        "effective_dit_checkpoint_path": resolve_dit_checkpoint_override(args),
        "dit_lora_path": args.dit_lora_path,
        "dit_lora_base_checkpoint_path": args.dit_lora_base_checkpoint_path,
        "dit_lora_target_modules": args.dit_lora_target_modules,
        "dit_lora_rank": args.dit_lora_rank,
        "dit_lora_alpha": args.dit_lora_alpha,
        "stitched_decoder_checkpoint_path": args.stitched_decoder_checkpoint_path,
        "da3_model_path": args.da3_model_path,
        "da3_enable_scene_flow_head": bool(getattr(args, "da3_enable_scene_flow_head", False)),
        "da3_scene_flow_head_checkpoint_path": getattr(args, "da3_scene_flow_head_checkpoint_path", None),
        "da3_scene_flow_preset": getattr(args, "da3_scene_flow_preset", "da3-giant-sceneflow"),
        "longcat_weight_root": args.longcat_weight_root,
        "longcat_vae_file": args.longcat_vae_file,
        "tokenizer_path": args.tokenizer_path,
        "effective_video_height": generation["effective_height"],
        "effective_video_width": generation["effective_width"],
        "effective_num_frames": generation["effective_frames"],
        "feedforward_resolution": args.feedforward_resolution,
        "da3_process_res_method": args.da3_process_res_method,
        "feedforward_image_hw": [int(image_hw[0]), int(image_hw[1])],
        "longcat_condition_hw": (
            list(getattr(args, "longcat_condition_hw"))
            if getattr(args, "longcat_condition_hw", None) is not None
            else None
        ),
        "depth_raw_shape": list(geometry["depth"].shape),
        "ray_raw_shape": list(geometry["ray"].shape),
        "point_map_shape": list(point_map_world.shape),
    }
    if getattr(args, "scene_flow_degrid_period", 0.0):
        meta["scene_flow_degrid_period"] = args.scene_flow_degrid_period
    if "scene_flow" in geometry and isinstance(geometry["scene_flow"], torch.Tensor):
        meta["scene_flow_raw_shape"] = list(geometry["scene_flow"].shape)
    if "scene_flow_cam" in geometry and isinstance(geometry["scene_flow_cam"], torch.Tensor):
        meta["scene_flow_cam_raw_shape"] = list(geometry["scene_flow_cam"].shape)
    if "camera_pose_encoding" in geometry and isinstance(geometry["camera_pose_encoding"], torch.Tensor):
        meta["camera_pose_encoding_shape"] = list(geometry["camera_pose_encoding"].shape)
    if "camera_extrinsics" in geometry and isinstance(geometry["camera_extrinsics"], torch.Tensor):
        meta["camera_extrinsics_shape"] = list(geometry["camera_extrinsics"].shape)
    if "camera_intrinsics" in geometry and isinstance(geometry["camera_intrinsics"], torch.Tensor):
        meta["camera_intrinsics_shape"] = list(geometry["camera_intrinsics"].shape)
    if "camera_focal_lengths" in geometry and isinstance(geometry["camera_focal_lengths"], torch.Tensor):
        meta["camera_focal_lengths_shape"] = list(geometry["camera_focal_lengths"].shape)
    if "camera_principal_points" in geometry and isinstance(geometry["camera_principal_points"], torch.Tensor):
        meta["camera_principal_points_shape"] = list(geometry["camera_principal_points"].shape)
    if "camera_image_hw" in geometry and isinstance(geometry["camera_image_hw"], (tuple, list)):
        meta["camera_image_hw"] = [int(geometry["camera_image_hw"][0]), int(geometry["camera_image_hw"][1])]
    with open(output_dir / "meta.json", "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_plan = build_stitched_decoder_checkpoint_plan(args)

    print(f"[INFO] Using device: {args.device}")
    print(f"[INFO] Loading LongCat pipeline from {args.longcat_weight_root}")
    if args.dit_lora_path:
        base_dit = resolve_dit_checkpoint_override(args) or str(Path(args.longcat_weight_root) / "dit")
        print(f"[INFO] Using base DiT for LoRA inference from {base_dit}")
        print(
            "[INFO] Applying DiT LoRA "
            f"from {args.dit_lora_path} "
            f"(rank={args.dit_lora_rank}, alpha={args.dit_lora_alpha}, "
            f"target_modules={args.dit_lora_target_modules})"
        )
    pipe = load_longcat_pipeline(args)

    print(f"[INFO] Loading stitched decoder from {args.stitched_decoder_checkpoint_path}")
    decoder_args = argparse.Namespace(**vars(args))
    decoder_args.checkpoint_path = args.stitched_decoder_checkpoint_path
    stitched_decoder = load_stitched_decoder(
        decoder_args,
        checkpoint_plan=checkpoint_plan,
        preserve_explicit_fields=(
            "feedforward_resolution",
            "resolution",
            "da3_enable_scene_flow_head",
            "da3_scene_flow_head_checkpoint_path",
            "da3_scene_flow_preset",
        ),
    )

    condition_frames = load_condition_frames(args)
    if condition_frames is not None:
        print(f"[INFO] Loaded {len(condition_frames)} condition frames")
    shape_notice = build_generation_shape_notice(args)
    if shape_notice is not None:
        print(f"[INFO] {shape_notice}")

    generation = generate_video_and_latents(pipe, args, condition_frames)
    geometry = run_stitched_geometry(args, stitched_decoder, generation["latents"])

    output_dir = build_output_dir(args)
    save_outputs(args, output_dir, generation, geometry, condition_frames)
    print(f"[INFO] Saved outputs to {output_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
