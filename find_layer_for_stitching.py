import argparse
import csv
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from data.dataset_util import create_stitching_dataloader
from models.da3_stitched import (
    DA3_STITCH_CANDIDATE_LAYER_NAMES,
    DA3_PATCH_SIZE,
    compute_da3_stitching_input_spatial_size,
    compute_da3_token_grid_size,
    extract_da3_candidate_features,
    is_da3_model,
)
from models.stitching_layer_builder import ConvSpec
from utils.argument import apply_feedforward_model_defaults, find_layer_stitching_argument
from utils.da3_resize import coerce_image_hw
from utils.dist_util import torch_device_setup
from utils.utils_for_thirdparty import load_feedforward_model, load_vae


STREAMING_FEATURE_STORAGE_MODE = "streaming"


def feature_shard_file_path(
    feature_file_path: str | Path,
    *,
    rank: int,
    world_size: int,
) -> Path:
    feature_file_path = Path(feature_file_path)
    suffix = feature_file_path.suffix
    stem = feature_file_path.stem if suffix else feature_file_path.name
    shard_name = f"{stem}.rank{int(rank)}of{int(world_size)}{suffix}"
    return feature_file_path.with_name(shard_name)


def merge_feature_shards(
    feature_file_path: str | Path,
    *,
    world_size: int,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    feature_file_path = Path(feature_file_path)
    merged_vae_features = []
    merged_feedforward_features: dict[str, list[torch.Tensor]] = {}

    for rank in range(int(world_size)):
        shard_path = feature_shard_file_path(
            feature_file_path,
            rank=rank,
            world_size=world_size,
        )
        shard = torch_load_cpu(shard_path, mmap=True)
        merged_vae_features.append(shard["vae_features"])
        for layer_key, features in shard["feedforward_features"].items():
            merged_feedforward_features.setdefault(layer_key, []).append(features)

    merged = {
        "vae_features": torch.cat(merged_vae_features, dim=0),
        "feedforward_features": {
            layer_key: torch.cat(layer_features, dim=0)
            for layer_key, layer_features in merged_feedforward_features.items()
        },
    }
    torch.save(merged, feature_file_path)
    return merged


def merge_streaming_feature_shards(
    feature_file_path: str | Path,
    *,
    world_size: int,
) -> dict[str, object]:
    feature_file_path = Path(feature_file_path)
    merged_layer_keys: list[str] | None = None
    merged_feedforward_features: dict[str, list[str]] = {}
    merged_vae_features: list[str] = []
    merged_token_grid_shapes: list[tuple[int, int]] = []
    merged_sample_metadata: list[dict[str, object]] = []

    for rank in range(int(world_size)):
        shard_path = feature_shard_file_path(
            feature_file_path,
            rank=rank,
            world_size=world_size,
        )
        shard = torch_load_cpu(shard_path, mmap=True)
        if shard.get("storage_mode") != STREAMING_FEATURE_STORAGE_MODE:
            raise ValueError(
                f"Expected streaming feature shard at {shard_path}, "
                f"got storage_mode={shard.get('storage_mode')!r}."
            )
        layer_keys = list(shard.get("layer_keys") or shard["feedforward_feature_paths"].keys())
        if merged_layer_keys is None:
            merged_layer_keys = layer_keys
        elif layer_keys != merged_layer_keys:
            raise ValueError(
                f"Inconsistent streaming layer keys between shards: "
                f"{merged_layer_keys} vs {layer_keys}"
            )

        merged_vae_features.extend(shard["vae_feature_paths"])
        merged_sample_metadata.extend(list(shard.get("sample_metadata", [])))
        merged_token_grid_shapes.extend(
            [
                tuple(shape)
                for shape in shard.get("feedforward_token_grid_shapes", [])
            ]
        )
        for layer_key in layer_keys:
            merged_feedforward_features.setdefault(layer_key, []).extend(
                shard["feedforward_feature_paths"][layer_key]
            )

    merged = {
        "storage_mode": STREAMING_FEATURE_STORAGE_MODE,
        "layer_keys": merged_layer_keys or [],
        "vae_feature_paths": merged_vae_features,
        "feedforward_feature_paths": merged_feedforward_features,
    }
    if merged_token_grid_shapes:
        merged["feedforward_token_grid_shapes"] = merged_token_grid_shapes
    if merged_sample_metadata:
        merged["sample_metadata"] = merged_sample_metadata
    torch.save(merged, feature_file_path)
    return merged


def distributed_feature_state() -> tuple[int, int, bool]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, rank == 0
    return 0, 1, True


def streaming_feature_dir(
    feature_save_path: str | Path,
    *,
    rank: int,
    world_size: int,
) -> Path:
    return Path(feature_save_path) / "feature_chunks" / f"rank{int(rank)}of{int(world_size)}"


def streaming_feature_tensor_file_path(
    feature_root: str | Path,
    *,
    sample_index: int,
    tensor_kind: str,
    layer_key: str | None = None,
) -> Path:
    feature_root = Path(feature_root)
    if layer_key is None:
        file_name = f"{tensor_kind}_{int(sample_index):06d}.pt"
    else:
        file_name = f"{tensor_kind}_{layer_key}_{int(sample_index):06d}.pt"
    return feature_root / file_name


def is_streaming_feature_index(features: dict[str, object]) -> bool:
    return features.get("storage_mode") == STREAMING_FEATURE_STORAGE_MODE


def torch_load_cpu(
    load_target,
    *,
    mmap: bool = False,
):
    load_kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    if mmap and isinstance(load_target, (str, os.PathLike, Path)):
        try:
            return torch.load(load_target, mmap=True, **load_kwargs)
        except TypeError:
            pass
    return torch.load(load_target, **load_kwargs)


def load_feature_batch(batch_or_path):
    if isinstance(batch_or_path, (str, os.PathLike, Path)):
        return torch_load_cpu(batch_or_path, mmap=True)
    return batch_or_path


def select_feature_batch(feature_batches, index: int):
    if isinstance(feature_batches, torch.Tensor):
        return feature_batches[int(index)]
    return load_feature_batch(feature_batches[int(index)])


def feature_batch_count(feature_batches) -> int | None:
    if isinstance(feature_batches, torch.Tensor):
        return int(feature_batches.shape[0])
    if hasattr(feature_batches, "__len__"):
        return len(feature_batches)
    return None


def first_feature_batch(feature_batches) -> torch.Tensor:
    if isinstance(feature_batches, torch.Tensor):
        if feature_batches.shape[0] == 0:
            raise ValueError("Feature tensor is empty.")
        return feature_batches[0]
    if hasattr(feature_batches, "__len__") and len(feature_batches) > 0:
        return load_feature_batch(feature_batches[0])
    raise ValueError("Feature batch sequence is empty.")


def maybe_cuda_autocast(device: torch.device | str):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _resolve_search_device(device_name: str) -> torch.device:
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for decoder search but not available; falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "cuda":
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            logger.warning("CUDA requested for decoder search but no visible CUDA devices were found; falling back to CPU.")
            return torch.device("cpu")
        if requested.index is not None and requested.index >= device_count:
            logger.warning(
                "Requested decoder search device {} is out of range for {} visible CUDA device(s); falling back to cuda:0.",
                requested,
                device_count,
            )
            return torch.device("cuda:0")
    return requested


def _get_module_floating_dtype(module: torch.nn.Module) -> torch.dtype:
    explicit_dtype = getattr(module, "dtype", None)
    if isinstance(explicit_dtype, torch.dtype):
        return explicit_dtype
    for parameter in module.parameters(recurse=True):
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers(recurse=True):
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.float32


def _module_exists(root: torch.nn.Module, module_name: str) -> bool:
    try:
        root.get_submodule(module_name)
        return True
    except AttributeError:
        return False


def _extract_tensor_from_module_output(output) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _extract_tensor_from_module_output(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor_from_module_output(item)
            if tensor is not None:
                return tensor
    return None


def _safe_layer_key(layer_name: str) -> str:
    safe_chars = []
    for char in str(layer_name):
        if char.isalnum() or char in ("_", "-"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "layer"


def _parse_layer_name_list(raw_names: str | None) -> list[str]:
    if raw_names is None:
        return []
    return [item.strip() for item in str(raw_names).split(",") if item.strip()]


def select_da3_layer_keys(
    layer_keys: list[str],
    *,
    args: argparse.Namespace,
) -> list[str]:
    selected = list(layer_keys)
    layer_count = getattr(args, "da3_search_layer_count", None)
    if layer_count is not None:
        layer_count = int(layer_count)
        if layer_count <= 0:
            raise ValueError("--da3_search_layer_count must be a positive integer.")
        selected = selected[:layer_count]

    layer_index = getattr(args, "da3_search_layer_index", None)
    if layer_index is not None:
        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= len(selected):
            raise ValueError(
                f"--da3_search_layer_index must be in [0, {len(selected) - 1}], "
                f"got {layer_index}."
            )
        selected = [selected[layer_index]]

    return selected


def shard_layer_names(
    layer_names: list[str],
    *,
    shard_count: int,
    shard_index: int,
) -> list[str]:
    names = list(layer_names)
    shard_count = int(shard_count or 1)
    shard_index = int(shard_index or 0)
    if shard_count <= 0:
        raise ValueError("--vae_decoder_layer_shard_count must be a positive integer.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "--vae_decoder_layer_shard_index must be in "
            f"[0, {shard_count - 1}], got {shard_index}."
        )
    if shard_count == 1:
        return names
    return [
        layer_name
        for layer_idx, layer_name in enumerate(names)
        if layer_idx % shard_count == shard_index
    ]


def denormalize_wan_vae_latents_for_decode(
    vae_model: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    config = getattr(vae_model, "config", None)
    if config is None or not hasattr(config, "latents_mean") or not hasattr(config, "latents_std"):
        raise ValueError("normalized VAE latents require vae_model.config.latents_mean and latents_std.")
    z_dim = int(getattr(config, "z_dim", latents.shape[1]))
    latents_mean = (
        torch.tensor(config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(device=latents.device, dtype=latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(config.latents_std).view(1, z_dim, 1, 1, 1).to(
        device=latents.device,
        dtype=latents.dtype,
    )
    return latents / latents_std + latents_mean


def prepare_vae_decode_latents(
    latent_batch: torch.Tensor,
    *,
    vae_model: torch.nn.Module,
    latent_space_mode: str,
    device: torch.device,
) -> torch.Tensor:
    if latent_batch.dim() == 4:
        latent_batch = latent_batch.unsqueeze(0)
    if latent_batch.dim() != 5:
        raise ValueError(
            "Expected cached VAE latent with shape [C, T, H, W] or [B, C, T, H, W], "
            f"got {tuple(latent_batch.shape)}"
        )
    dtype = _get_module_floating_dtype(vae_model)
    latents = latent_batch.to(device=device, dtype=dtype, non_blocking=True)
    if latent_space_mode == "raw":
        return latents
    if latent_space_mode == "normalized":
        return denormalize_wan_vae_latents_for_decode(vae_model, latents)
    raise ValueError(f"Unsupported VAE latent space mode: {latent_space_mode}")


def _concat_decoder_hook_features(features: list[torch.Tensor], *, layer_name: str) -> torch.Tensor:
    if not features:
        raise ValueError(f"No tensor features captured from VAE decoder layer {layer_name!r}.")
    if len(features) == 1:
        return features[0]
    base_shape = list(features[0].shape)
    for feature in features[1:]:
        comparable_shape = list(feature.shape)
        if comparable_shape[:2] != base_shape[:2] or comparable_shape[3:] != base_shape[3:]:
            raise ValueError(
                f"Cannot concatenate chunked decoder features for {layer_name}: "
                f"{tuple(features[0].shape)} vs {tuple(feature.shape)}"
            )
    return torch.cat(features, dim=2)


class DecoderFeatureBatches:
    def __init__(
        self,
        latent_batches,
        *,
        vae_model: torch.nn.Module,
        layer_name: str,
        latent_space_mode: str,
        device: torch.device,
        feature_dtype: torch.dtype = torch.float16,
    ):
        self.latent_batches = latent_batches
        self.vae_model = vae_model
        self.layer_name = str(layer_name)
        self.latent_space_mode = str(latent_space_mode)
        self.device = torch.device(device)
        self.feature_dtype = feature_dtype

    def __len__(self):
        count = feature_batch_count(self.latent_batches)
        if count is None:
            raise TypeError("DecoderFeatureBatches requires a sized latent source.")
        return count

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        module = self.vae_model.get_submodule(self.layer_name)
        captured_features: list[torch.Tensor] = []

        def hook(_module, _inputs, output):
            feature = _extract_tensor_from_module_output(output)
            if feature is None or feature.dim() != 5:
                return
            captured_features.append(
                feature.detach().to(device="cpu", dtype=self.feature_dtype)
            )

        handle = module.register_forward_hook(hook)
        try:
            latent = select_feature_batch(self.latent_batches, int(index))
            latents = prepare_vae_decode_latents(
                latent,
                vae_model=self.vae_model,
                latent_space_mode=self.latent_space_mode,
                device=self.device,
            )
            with torch.no_grad(), maybe_cuda_autocast(self.device):
                _ = self.vae_model.decode(latents, return_dict=False)
        finally:
            handle.remove()

        feature = _concat_decoder_hook_features(captured_features, layer_name=self.layer_name)
        if feature.shape[0] != 1:
            raise ValueError(
                f"Decoder feature batches expect one sample at a time, got batch={feature.shape[0]}"
            )
        return feature[0].contiguous()


def default_vae_decoder_layer_names(vae_model: torch.nn.Module) -> list[str]:
    candidates = [
        "post_quant_conv",
        "decoder.conv_in",
        "decoder.mid_block",
    ]
    decoder = getattr(vae_model, "decoder", None)
    up_blocks = getattr(decoder, "up_blocks", None)
    if up_blocks is not None:
        candidates.extend(f"decoder.up_blocks.{idx}" for idx, _ in enumerate(up_blocks))
    candidates.extend(
        [
            "decoder.norm_out",
            "decoder.conv_out",
            "decoder",
        ]
    )
    return [name for name in candidates if _module_exists(vae_model, name)]


def discover_vae_decoder_layer_names(
    vae_model: torch.nn.Module,
    sample_latent: torch.Tensor,
    *,
    latent_space_mode: str,
    device: torch.device,
    max_layers: int | None = None,
) -> list[str]:
    captured: dict[str, tuple[int, ...]] = {}
    handles = []

    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            if module_name in captured:
                return
            feature = _extract_tensor_from_module_output(output)
            if feature is not None and feature.dim() == 5:
                captured[module_name] = tuple(feature.shape)

        return hook

    for module_name, module in vae_model.named_modules():
        if module_name == "post_quant_conv" or module_name == "decoder" or module_name.startswith("decoder."):
            handles.append(module.register_forward_hook(make_hook(module_name)))

    try:
        latents = prepare_vae_decode_latents(
            sample_latent,
            vae_model=vae_model,
            latent_space_mode=latent_space_mode,
            device=device,
        )
        with torch.no_grad(), maybe_cuda_autocast(device):
            _ = vae_model.decode(latents, return_dict=False)
    finally:
        for handle in handles:
            handle.remove()

    layer_names = list(captured.keys())
    if max_layers is not None:
        layer_names = layer_names[: int(max_layers)]
    return layer_names


def resolve_vae_decoder_layer_names(
    args: argparse.Namespace,
    *,
    vae_model: torch.nn.Module,
    sample_latent: torch.Tensor,
    device: torch.device,
) -> list[str]:
    explicit_names = _parse_layer_name_list(getattr(args, "vae_decoder_layer_names", None))
    if explicit_names:
        missing = [name for name in explicit_names if not _module_exists(vae_model, name)]
        if missing:
            raise ValueError(f"VAE decoder layer(s) not found: {missing}")
        return explicit_names

    max_layers = getattr(args, "vae_decoder_max_layers", None)
    mode = getattr(args, "vae_decoder_layer_mode", "block")
    if mode == "block":
        names = default_vae_decoder_layer_names(vae_model)
        if max_layers is not None:
            names = names[: int(max_layers)]
        return names
    if mode == "all":
        return discover_vae_decoder_layer_names(
            vae_model,
            sample_latent,
            latent_space_mode=getattr(args, "vae_latent_space", "raw"),
            device=device,
            max_layers=max_layers,
        )
    raise ValueError(f"Unsupported VAE decoder layer mode: {mode}")


def infer_feature_temporal_size(feature: torch.Tensor) -> int:
    if feature.dim() == 3:
        return int(feature.shape[0])
    if feature.dim() == 4:
        return int(feature.shape[1])
    if feature.dim() == 5:
        return int(feature.shape[2])
    raise ValueError(f"Cannot infer temporal size from feature shape {tuple(feature.shape)}")


def _conv_spec_tuple(value, *, dim: int) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        if len(value) != dim:
            raise ValueError(f"Expected ConvSpec tuple length {dim}, got {value}")
        return tuple(int(item) for item in value)
    return tuple(int(value) for _ in range(dim))


def _choose_dynamic_stride_axis(
    *,
    source_size: int,
    target_size: int,
    kernel_size: int,
    padding: int,
    dilation: int = 1,
) -> tuple[int, int]:
    if source_size <= 0 or target_size <= 0:
        raise ValueError(f"Invalid source/target axis sizes: {source_size}, {target_size}")
    if source_size <= target_size:
        stride = 1
        input_size = target_size * stride - 2 * padding + dilation * (kernel_size - 1)
        return stride, int(input_size)

    ideal_stride = max(1.0, float(source_size) / float(target_size))
    stride_candidates = {
        max(1, int(math.floor(ideal_stride))),
        max(1, int(round(ideal_stride))),
        max(1, int(math.ceil(ideal_stride))),
    }
    best_stride = min(
        stride_candidates,
        key=lambda stride: (
            abs(
                (
                    target_size * stride
                    - 2 * padding
                    + dilation * (kernel_size - 1)
                )
                - source_size
            ),
            abs(stride - ideal_stride),
            stride,
        ),
    )
    input_size = target_size * best_stride - 2 * padding + dilation * (kernel_size - 1)
    return int(best_stride), int(input_size)


def build_dynamic_decoder_transfer_layer(
    conv_spec: ConvSpec,
    *,
    in_channels: int,
    source_feature_shape: tuple[int, ...],
    target_grid_shape: tuple[int, int],
) -> tuple[torch.nn.Conv3d, tuple[int, int]]:
    if len(source_feature_shape) != 4:
        raise ValueError(
            "Expected source decoder feature shape [C, T, H, W], "
            f"got {source_feature_shape}"
        )
    kernel_size = _conv_spec_tuple(conv_spec.kernel_size, dim=3)
    padding = _conv_spec_tuple(conv_spec.padding, dim=3)
    dilation = _conv_spec_tuple(conv_spec.dilation, dim=3)

    _, _, source_h, source_w = source_feature_shape
    target_h, target_w = int(target_grid_shape[0]), int(target_grid_shape[1])
    stride_h, input_h = _choose_dynamic_stride_axis(
        source_size=int(source_h),
        target_size=target_h,
        kernel_size=kernel_size[1],
        padding=padding[1],
        dilation=dilation[1],
    )
    stride_w, input_w = _choose_dynamic_stride_axis(
        source_size=int(source_w),
        target_size=target_w,
        kernel_size=kernel_size[2],
        padding=padding[2],
        dilation=dilation[2],
    )
    dynamic_spec = ConvSpec(
        dim=3,
        out_channels=int(conv_spec.out_channels),
        kernel_size=kernel_size,
        stride=(1, stride_h, stride_w),
        padding=padding,
        dilation=dilation,
    )
    return dynamic_spec.build(in_channels=in_channels), (input_h, input_w)


def resize_source_feature_for_conv(
    source_feature: torch.Tensor,
    *,
    target_temporal_size: int | None,
    target_spatial_shape: tuple[int, int],
) -> torch.Tensor:
    if source_feature.dim() != 5:
        raise ValueError(f"Expected source feature [B, C, T, H, W], got {tuple(source_feature.shape)}")
    if target_temporal_size is None:
        target_temporal_size = (int(source_feature.shape[2]) - 1) * 4 + 1
    return nn.functional.interpolate(
        source_feature,
        size=[
            int(target_temporal_size),
            int(target_spatial_shape[0]),
            int(target_spatial_shape[1]),
        ],
        mode="trilinear",
        align_corners=True,
    )


def resolve_feature_file_path(feature_path: str | Path) -> Path:
    feature_path = Path(feature_path).expanduser()
    if feature_path.suffix == ".pt":
        return feature_path
    return feature_path / "features.pt"


def resolve_feature_output_dir(feature_path: str | Path) -> Path:
    feature_path = Path(feature_path).expanduser()
    if feature_path.suffix == ".pt":
        return feature_path.parent
    return feature_path


def extract_sample_metadata_from_batch(
    batch: dict[str, object],
    *,
    batch_idx: int,
    sample_index: int,
) -> dict[str, object]:
    metadata: dict[str, object] = {"sample_index": int(sample_index)}
    for key in ("dataset_name", "source_split", "scene_name", "clip_uid"):
        value = batch.get(key)
        if value is None:
            continue
        item = value
        if isinstance(value, (list, tuple)):
            item = value[batch_idx]
        elif torch.is_tensor(value):
            if value.dim() == 0:
                item = value.item()
            elif value.dim() == 1 and value.shape[0] > batch_idx:
                item = value[batch_idx].item()
            else:
                item = None
        if item is None:
            continue
        metadata[key] = str(item)
    return metadata


def infer_da3_token_grid_shape(
    num_tokens: int,
    *,
    feedforward_resolution: int,
    image_hw: tuple[int, int] | None = None,
    patch_size: int = DA3_PATCH_SIZE,
) -> tuple[int, int]:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    if image_hw is not None:
        height = int(image_hw[0]) // patch_size
        width = int(image_hw[1]) // patch_size
        if height * width != int(num_tokens):
            raise ValueError(
                f"image_hw={image_hw} implies token grid {(height, width)}, "
                f"but num_tokens={num_tokens}"
            )
        return height, width

    expected_longest_grid = compute_da3_token_grid_size(
        int(feedforward_resolution),
        patch_size=patch_size,
    )
    if expected_longest_grid > 0 and num_tokens % expected_longest_grid == 0:
        return num_tokens // expected_longest_grid, expected_longest_grid

    candidate_pairs: list[tuple[int, int]] = []
    upper = int(math.sqrt(num_tokens))
    for factor in range(1, upper + 1):
        if num_tokens % factor != 0:
            continue
        other = num_tokens // factor
        candidate_pairs.append((factor, other))

    if not candidate_pairs:
        raise ValueError(f"Failed to factorize num_tokens={num_tokens}")

    ranked_pairs = sorted(
        candidate_pairs,
        key=lambda pair: (
            abs(max(pair) - expected_longest_grid),
            abs(min(pair) - expected_longest_grid),
        ),
    )
    return ranked_pairs[0]


def resolve_token_grid_shape(
    feature_tokens: torch.Tensor,
    *,
    token_grid_shape: tuple[int, int] | None = None,
    feedforward_resolution: int | None = None,
) -> tuple[int, int]:
    if token_grid_shape is not None:
        return int(token_grid_shape[0]), int(token_grid_shape[1])

    num_tokens = int(feature_tokens.shape[1])
    square_size = int(math.sqrt(num_tokens))
    if square_size * square_size == num_tokens:
        return square_size, square_size
    if feedforward_resolution is None:
        raise ValueError(
            f"Cannot infer non-square DA3 token grid for num_tokens={num_tokens} "
            "without feedforward_resolution."
        )
    return infer_da3_token_grid_shape(
        num_tokens=num_tokens,
        feedforward_resolution=feedforward_resolution,
    )


def reshape_feature_tokens_to_grid(
    feature_tokens: torch.Tensor,
    *,
    token_grid_shape: tuple[int, int] | None = None,
    feedforward_resolution: int | None = None,
) -> torch.Tensor:
    if feature_tokens.dim() == 4:
        return feature_tokens
    if feature_tokens.dim() != 3:
        raise ValueError(
            "Expected feature tokens with shape [T, N, D] or [D, T, H, W], "
            f"got {tuple(feature_tokens.shape)}"
        )
    height, width = resolve_token_grid_shape(
        feature_tokens,
        token_grid_shape=token_grid_shape,
        feedforward_resolution=feedforward_resolution,
    )
    return rearrange(
        feature_tokens,
        "T (H W) D -> D T H W",
        H=height,
        W=width,
    )


def normalize_spatial_size(
    spatial_size: int | tuple[int, int] | None,
) -> tuple[int, int] | None:
    if spatial_size is None:
        return None
    if isinstance(spatial_size, tuple):
        return int(spatial_size[0]), int(spatial_size[1])
    return int(spatial_size), int(spatial_size)


def infer_token_grid_shape_from_batch(
    feature_tokens: torch.Tensor,
    *,
    batch_idx: int,
    token_grid_shapes: list[tuple[int, int]] | None,
    feedforward_resolution: int | None,
) -> tuple[int, int] | None:
    if token_grid_shapes is not None:
        return tuple(token_grid_shapes[batch_idx])
    if feature_tokens.dim() == 3:
        return resolve_token_grid_shape(
            feature_tokens,
            feedforward_resolution=feedforward_resolution,
        )
    return None


def compute_conv_input_spatial_shape(
    *,
    output_grid_shape: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int]:
    output_h, output_w = output_grid_shape
    input_h = output_h * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1)
    input_w = output_w * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1)
    return int(input_h), int(input_w)


@torch.no_grad()
def unfold3d(x, kernel_size, stride, padding, dilation=(1, 1, 1)):
    """Return a view of every sliding kT×kH×kW block as a separate axis.

    x             : (N, C, T, H, W)  – contiguous or channels-first stride
    kernel_size   : (kT, kH, kW)
    stride        : (sT, sH, sW)
    padding       : (pT, pH, pW)
    dilation      : (dT, dH, dW)
    result shape  : (N, C, kT, kH, kW, T_out, H_out, W_out)
    """
    # This avoids explicit im2col allocation by using as_strided views.
    kT, kH, kW = kernel_size
    sT, sH, sW = stride
    pT, pH, pW = padding
    dT, dH, dW = dilation

    x = torch.nn.functional.pad(x, (pW, pW, pH, pH, pT, pT))

    N, C, T, H, W = x.shape
    To = (T - (dT * (kT - 1) + 1)) // sT + 1
    Ho = (H - (dH * (kH - 1) + 1)) // sH + 1
    Wo = (W - (dW * (kW - 1) + 1)) // sW + 1

    sN, sC, sT0, sH0, sW0 = x.stride()

    return x.as_strided(
        size=(N, C, kT, kH, kW, To, Ho, Wo),
        stride=(sN, sC, sT0 * dT, sH0 * dH, sW0 * dW, sT0 * sT, sH0 * sH, sW0 * sW),
    )


@torch.no_grad()
def fit_conv3d_streaming(
    conv,
    Z_batches,  # tensors (B, C_in, F, H, W)
    Y_batches,  # tensors (B, C_out, F_y, H_y, W_y)
    target_spatial_size=None,
    ridge=1e-4,
    dtype=torch.float64,
    token_grid_shapes=None,
    feedforward_resolution: int | None = None,
    match_target_temporal: bool = False,
):
    """
    Solve (XᵀX + λI)W = XᵀY  without ever instantiating X.
    Result is written directly into `conv.weight` (+ bias if present).
    """
    # Streaming ridge regression: accumulate XtX/XtY across batches to avoid O(N*d) memory.
    device = conv.weight.device
    kT, kH, kW = conv.kernel_size
    C_in = conv.in_channels
    C_out = conv.out_channels
    d = C_in * kT * kH * kW  # cols of X

    # Accumulate XᵀX and XᵀY
    XtX = torch.zeros(d, d, device=device, dtype=dtype)
    XtY = torch.zeros(d, C_out, device=device, dtype=dtype)
    n_rows = 0  # for bias later
    first_vae_feature = first_feature_batch(Z_batches)
    total_batches = feature_batch_count(Z_batches)
    default_target_spatial_shape = normalize_spatial_size(target_spatial_size)
    if default_target_spatial_shape is None:
        default_target_spatial_shape = (
            int(first_vae_feature.shape[2]),
            int(first_vae_feature.shape[3]),
        )

    for batch_idx, (Z_b, Y_b) in enumerate(
        tqdm(zip(Z_batches, Y_batches), total=total_batches)
    ):
        Z_b = load_feature_batch(Z_b)
        Y_b = load_feature_batch(Y_b)
        target_temporal_size = (
            infer_feature_temporal_size(Y_b) if match_target_temporal else None
        )
        token_grid_shape = infer_token_grid_shape_from_batch(
            Y_b,
            batch_idx=batch_idx,
            token_grid_shapes=token_grid_shapes,
            feedforward_resolution=feedforward_resolution,
        )
        if token_grid_shape is not None:
            current_target_spatial_shape = compute_conv_input_spatial_shape(
                output_grid_shape=token_grid_shape,
                kernel_size=(kH, kW),
                stride=(conv.stride[1], conv.stride[2]),
                padding=(conv.padding[1], conv.padding[2]),
                dilation=(conv.dilation[1], conv.dilation[2]),
            )
        else:
            current_target_spatial_shape = default_target_spatial_shape
        Z_b = Z_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
        Y_b = Y_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
        Z_b = Z_b.unsqueeze(0)
        Z_b = resize_source_feature_for_conv(
            Z_b,
            target_temporal_size=target_temporal_size,
            target_spatial_shape=current_target_spatial_shape,
        )  # 1, C, T, H, W
        Y_b = reshape_feature_tokens_to_grid(
            Y_b,
            token_grid_shape=token_grid_shape,
            feedforward_resolution=feedforward_resolution,
        ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out

        patches = unfold3d(
            Z_b,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
        )
        # X_b : (B·T_out·H_out·W_out, d)
        X_b = patches.permute(0, 5, 6, 7, 1, 2, 3, 4).reshape(-1, d).to(dtype)
        # Y_b_vec : (same rows, C_out)
        Y_b_vec = Y_b.permute(0, 2, 3, 4, 1).reshape(-1, C_out).to(dtype)

        XtX += X_b.T @ X_b
        XtY += X_b.T @ Y_b_vec
        n_rows += X_b.shape[0]

        del Z_b, Y_b, patches, X_b, Y_b_vec
        torch.cuda.empty_cache()

    # Ridge ⇒ add λI
    XtX.diagonal().add_(ridge)
    W = torch.linalg.solve(XtX, XtY)  # (d, C_out)
    conv.weight.copy_(W.T.reshape(C_out, C_in, kT, kH, kW).to(conv.weight.dtype))

    # bias = mean residual
    if conv.bias is not None:
        sum_residual = torch.zeros(C_out, device=device, dtype=dtype)

        # second pass cheaper: residual uses matrix–vector product only
        for batch_idx, (Z_b, Y_b) in enumerate(
            tqdm(zip(Z_batches, Y_batches), total=total_batches)
        ):
            Z_b = load_feature_batch(Z_b)
            Y_b = load_feature_batch(Y_b)
            target_temporal_size = (
                infer_feature_temporal_size(Y_b) if match_target_temporal else None
            )
            token_grid_shape = infer_token_grid_shape_from_batch(
                Y_b,
                batch_idx=batch_idx,
                token_grid_shapes=token_grid_shapes,
                feedforward_resolution=feedforward_resolution,
            )
            if token_grid_shape is not None:
                current_target_spatial_shape = compute_conv_input_spatial_shape(
                    output_grid_shape=token_grid_shape,
                    kernel_size=(kH, kW),
                    stride=(conv.stride[1], conv.stride[2]),
                    padding=(conv.padding[1], conv.padding[2]),
                    dilation=(conv.dilation[1], conv.dilation[2]),
                )
            else:
                current_target_spatial_shape = default_target_spatial_shape
            Z_b = Z_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
            Y_b = Y_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
            Z_b = Z_b.unsqueeze(0)
            Z_b = resize_source_feature_for_conv(
                Z_b,
                target_temporal_size=target_temporal_size,
                target_spatial_shape=current_target_spatial_shape,
            )  # 1, C, T, H, W
            Y_b = reshape_feature_tokens_to_grid(
                Y_b,
                token_grid_shape=token_grid_shape,
                feedforward_resolution=feedforward_resolution,
            ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out

            patches = unfold3d(
                Z_b,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
            )
            X_b = patches.permute(0, 5, 6, 7, 1, 2, 3, 4).reshape(-1, d).to(dtype)
            Y_b_vec = Y_b.permute(0, 2, 3, 4, 1).reshape(-1, C_out).to(dtype)

            sum_residual += (Y_b_vec - X_b @ W).sum(0)
            del Z_b, Y_b, patches, X_b, Y_b_vec
            torch.cuda.empty_cache()

        conv.bias.copy_((sum_residual / n_rows).to(conv.weight.dtype))

    return conv


def _feature_sources_from_cache(
    cached_features: dict[str, object],
) -> tuple[object, list[str], dict[str, object], list[tuple[int, int]] | None]:
    if is_streaming_feature_index(cached_features):
        vae_feature_source = cached_features["vae_feature_paths"]
        layer_keys = list(cached_features["layer_keys"])
        feedforward_sources = dict(cached_features["feedforward_feature_paths"])
        token_grid_shapes = [
            tuple(shape)
            for shape in cached_features.get("feedforward_token_grid_shapes", [])
        ] or None
        return vae_feature_source, layer_keys, feedforward_sources, token_grid_shapes

    vae_feature_source = cached_features["vae_features"]
    feedforward_sources = dict(cached_features["feedforward_features"])
    layer_keys = list(feedforward_sources.keys())
    return vae_feature_source, layer_keys, feedforward_sources, None


def evaluate_stitching_layer_mse(
    stitching_layer: torch.nn.Module,
    source_batches,
    target_batches,
    *,
    target_spatial_size: int | tuple[int, int] | None,
    token_grid_shapes: list[tuple[int, int]] | None = None,
    feedforward_resolution: int | None = None,
    match_target_temporal: bool = False,
) -> float:
    device = stitching_layer.weight.device
    mse, vox = 0.0, 0
    total_batches = feature_batch_count(source_batches)
    with torch.no_grad():
        for batch_idx, (Z_b, Y_b) in enumerate(
            tqdm(
                zip(source_batches, target_batches),
                total=total_batches,
            )
        ):
            Z_b = load_feature_batch(Z_b)
            Y_b = load_feature_batch(Y_b)
            target_temporal_size = (
                infer_feature_temporal_size(Y_b) if match_target_temporal else None
            )
            token_grid_shape = infer_token_grid_shape_from_batch(
                Y_b,
                batch_idx=batch_idx,
                token_grid_shapes=token_grid_shapes,
                feedforward_resolution=feedforward_resolution,
            )
            if token_grid_shape is not None:
                current_target_spatial_shape = compute_conv_input_spatial_shape(
                    output_grid_shape=token_grid_shape,
                    kernel_size=(stitching_layer.kernel_size[1], stitching_layer.kernel_size[2]),
                    stride=(stitching_layer.stride[1], stitching_layer.stride[2]),
                    padding=(stitching_layer.padding[1], stitching_layer.padding[2]),
                    dilation=(stitching_layer.dilation[1], stitching_layer.dilation[2]),
                )
            else:
                current_target_spatial_shape = normalize_spatial_size(target_spatial_size)
                if current_target_spatial_shape is None:
                    current_target_spatial_shape = (
                        int(Z_b.shape[2]),
                        int(Z_b.shape[3]),
                    )

            Z_b = Z_b.to(device, non_blocking=True).unsqueeze(0)
            Z_b = resize_source_feature_for_conv(
                Z_b,
                target_temporal_size=target_temporal_size,
                target_spatial_shape=current_target_spatial_shape,
            )
            Y_b = reshape_feature_tokens_to_grid(
                Y_b.to(device, non_blocking=True),
                token_grid_shape=token_grid_shape,
                feedforward_resolution=feedforward_resolution,
            ).unsqueeze(0)
            pred = stitching_layer(Z_b.float())
            mse += (pred - Y_b).square().sum().item()
            vox += pred.numel()
    return mse / max(vox, 1)


def _write_decoder_search_leaderboard(
    output_dir: Path,
    results: list[dict[str, object]],
) -> None:
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["mse"]),
            str(item["vae_decoder_layer"]),
            str(item["da3_layer"]),
        ),
    )
    (output_dir / "decoder_layer_search_leaderboard.json").write_text(
        json.dumps(ranked, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "decoder_layer_search_leaderboard.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "vae_decoder_layer",
                "da3_layer",
                "mse",
                "state_dict_path",
            ],
        )
        writer.writeheader()
        for item in ranked:
            writer.writerow(
                {
                    "vae_decoder_layer": item["vae_decoder_layer"],
                    "da3_layer": item["da3_layer"],
                    "mse": item["mse"],
                    "state_dict_path": item["state_dict_path"],
                }
            )


def run_vae_decoder_layer_search(
    *,
    args: argparse.Namespace,
    feature_output_dir: str | Path,
    cached_features: dict[str, object],
    vae_model: torch.nn.Module,
) -> list[dict[str, object]]:
    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    search_device = _resolve_search_device(getattr(args, "decoder_search_device", "cuda"))
    vae_model.to(search_device)
    vae_model.eval()

    vae_feature_source, da3_layer_keys, feedforward_sources, token_grid_shapes = _feature_sources_from_cache(
        cached_features
    )
    da3_layer_keys = select_da3_layer_keys(da3_layer_keys, args=args)
    if not da3_layer_keys:
        raise ValueError("No DA3 target layers were selected for search.")

    sample_latent = first_feature_batch(vae_feature_source)
    all_source_layer_names = resolve_vae_decoder_layer_names(
        args,
        vae_model=vae_model,
        sample_latent=sample_latent,
        device=search_device,
    )
    if not all_source_layer_names:
        raise ValueError("No VAE decoder source layers were selected for search.")
    shard_count = int(getattr(args, "vae_decoder_layer_shard_count", 1) or 1)
    shard_index = int(getattr(args, "vae_decoder_layer_shard_index", 0) or 0)
    source_layer_names = shard_layer_names(
        all_source_layer_names,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    if not source_layer_names:
        logger.warning(
            "Skipping VAE decoder layer search shard {}/{} because no source layers were assigned.",
            shard_index,
            shard_count,
        )
        _write_decoder_search_leaderboard(feature_output_dir, [])
        return []

    logger.info(
        "Running VAE decoder layer search with {} source layer(s) from {} total, "
        "{} DA3 layer(s), shard={}/{}, latent_space={}, device={}.",
        len(source_layer_names),
        len(all_source_layer_names),
        len(da3_layer_keys),
        shard_index,
        shard_count,
        getattr(args, "vae_latent_space", "raw"),
        search_device,
    )

    source_infos = []
    source_results_by_name: dict[str, list[dict[str, object]]] = {}
    for source_layer_name in source_layer_names:
        source_batches = DecoderFeatureBatches(
            vae_feature_source,
            vae_model=vae_model,
            layer_name=source_layer_name,
            latent_space_mode=getattr(args, "vae_latent_space", "raw"),
            device=search_device,
        )
        first_source_feature = first_feature_batch(source_batches)
        in_channels = int(first_source_feature.shape[0])
        source_safe_name = _safe_layer_key(source_layer_name)
        source_infos.append(
            {
                "source_layer_name": source_layer_name,
                "source_batches": source_batches,
                "first_source_feature": first_source_feature,
                "in_channels": in_channels,
                "source_safe_name": source_safe_name,
            }
        )
        source_results_by_name[source_layer_name] = []

    results: list[dict[str, object]] = []
    for da3_layer_key in da3_layer_keys:
        layer_feature_source = feedforward_sources[da3_layer_key]
        first_target_feature = first_feature_batch(layer_feature_source)
        target_grid_shape = resolve_token_grid_shape(
            first_target_feature,
            feedforward_resolution=args.feedforward_resolution,
        )

        for source_info in source_infos:
            source_layer_name = str(source_info["source_layer_name"])
            source_batches = source_info["source_batches"]
            first_source_feature = source_info["first_source_feature"]
            in_channels = int(source_info["in_channels"])
            source_safe_name = str(source_info["source_safe_name"])
            stitching_layer, target_spatial_size = build_dynamic_decoder_transfer_layer(
                args.stitching_layer_config,
                in_channels=in_channels,
                source_feature_shape=tuple(first_source_feature.shape),
                target_grid_shape=target_grid_shape,
            )
            stitching_layer = stitching_layer.to(search_device)
            target_state_path = feature_output_dir / (
                f"state_dict_{source_safe_name}__{da3_layer_key}.pt"
            )
            if target_state_path.exists():
                logger.info("Loading existing decoder search state dict from {}.", target_state_path)
                state_dict = torch.load(
                    target_state_path,
                    map_location=search_device,
                    weights_only=False,
                )
                stitching_layer.load_state_dict(state_dict, strict=True)
            else:
                stitching_layer = fit_conv3d_streaming(
                    stitching_layer,
                    source_batches,
                    layer_feature_source,
                    target_spatial_size=target_spatial_size,
                    ridge=1e-4,
                    token_grid_shapes=token_grid_shapes,
                    feedforward_resolution=args.feedforward_resolution,
                    match_target_temporal=True,
                )
                torch.save(
                    {
                        key: value.detach().cpu()
                        for key, value in stitching_layer.state_dict().items()
                    },
                    target_state_path,
                )

            mse = evaluate_stitching_layer_mse(
                stitching_layer,
                source_batches,
                layer_feature_source,
                target_spatial_size=target_spatial_size,
                token_grid_shapes=token_grid_shapes,
                feedforward_resolution=args.feedforward_resolution,
                match_target_temporal=True,
            )
            metric = {
                "vae_decoder_layer": source_layer_name,
                "da3_layer": da3_layer_key,
                "mse": float(mse),
                "state_dict_path": str(target_state_path),
                "source_feature_shape": list(first_source_feature.shape),
                "target_grid_shape": list(target_grid_shape),
                "transfer_kernel_size": list(stitching_layer.kernel_size),
                "transfer_stride": list(stitching_layer.stride),
                "transfer_padding": list(stitching_layer.padding),
                "transfer_input_spatial_size": list(target_spatial_size),
            }
            (feature_output_dir / f"metrics_{source_safe_name}__{da3_layer_key}.json").write_text(
                json.dumps(metric, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (feature_output_dir / f"mse_{source_safe_name}__{da3_layer_key}.txt").write_text(
                f"{float(mse)}\n",
                encoding="utf-8",
            )
            source_results_by_name[source_layer_name].append(metric)
            results.append(metric)

    for source_info in source_infos:
        source_layer_name = str(source_info["source_layer_name"])
        source_safe_name = str(source_info["source_safe_name"])
        source_results = source_results_by_name[source_layer_name]
        best_for_source = min(source_results, key=lambda item: float(item["mse"]))
        (feature_output_dir / f"best_da3_for_{source_safe_name}.json").write_text(
            json.dumps(best_for_source, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Best DA3 layer for VAE decoder layer {}: {} mse={}",
            source_layer_name,
            best_for_source["da3_layer"],
            best_for_source["mse"],
        )

    _write_decoder_search_leaderboard(feature_output_dir, results)
    global_best = min(results, key=lambda item: float(item["mse"]))
    logger.info(
        "Global best VAE-decoder stitching pair: vae_layer={} da3_layer={} mse={}",
        global_best["vae_decoder_layer"],
        global_best["da3_layer"],
        global_best["mse"],
    )
    return results


@torch.no_grad()
def extract_da3_features_to_disk(
    feedforward_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_root: str | Path,
    *,
    iterations_for_feature_extraction: int,
) -> dict[str, object]:
    if not is_da3_model(feedforward_model):
        raise ValueError("extract_da3_features_to_disk() requires a DA3 feedforward model.")

    feature_root = Path(feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)

    layer_keys: list[str] | None = None
    feature_index: dict[str, object] = {
        "storage_mode": STREAMING_FEATURE_STORAGE_MODE,
        "layer_keys": [],
        "vae_feature_paths": [],
        "feedforward_feature_paths": {},
        "feedforward_token_grid_shapes": [],
        "sample_metadata": [],
    }
    sample_index = 0

    for step, data in tqdm(
        enumerate(data_loader), total=iterations_for_feature_extraction
    ):
        if step >= iterations_for_feature_extraction:
            break

        vae_images = data["vae_image_tensor"].to(vae_model.device)
        feedforward_img = data["feedforward_image_tensor"].to(vae_model.device)
        runtime_image_hw = coerce_image_hw(data.get("image_hw"))
        if runtime_image_hw is None:
            runtime_image_hw = (
                int(feedforward_img.shape[-2]),
                int(feedforward_img.shape[-1]),
            )
        batch_token_grid_shape = (
            int(runtime_image_hw[0]) // DA3_PATCH_SIZE,
            int(runtime_image_hw[1]) // DA3_PATCH_SIZE,
        )

        with maybe_cuda_autocast(vae_model.device):
            da3_features = extract_da3_candidate_features(
                feedforward_model,
                feedforward_img,
                candidate_layers=DA3_STITCH_CANDIDATE_LAYER_NAMES,
            )
            if hasattr(vae_model, "encode"):
                latents = vae_model.encode(vae_images).latent_dist.sample()
            else:
                raise NotImplementedError(
                    f"VAE encoding not implemented for {type(vae_model)}"
                )

        if layer_keys is None:
            layer_keys = list(da3_features.keys())
            feature_index["layer_keys"] = layer_keys
            feature_index["feedforward_feature_paths"] = {
                layer_key: [] for layer_key in layer_keys
            }

        latents_cpu = latents.detach().cpu().to(dtype=torch.float16)
        da3_features_cpu = {
            name: feature.detach().cpu().to(dtype=torch.float16)
            for name, feature in da3_features.items()
        }

        for batch_idx in range(int(latents_cpu.shape[0])):
            vae_path = streaming_feature_tensor_file_path(
                feature_root,
                sample_index=sample_index,
                tensor_kind="vae",
            )
            torch.save(latents_cpu[batch_idx], vae_path)
            feature_index["vae_feature_paths"].append(str(vae_path))
            feature_index["feedforward_token_grid_shapes"].append(batch_token_grid_shape)
            feature_index["sample_metadata"].append(
                extract_sample_metadata_from_batch(
                    data,
                    batch_idx=batch_idx,
                    sample_index=sample_index,
                )
            )

            for layer_key in layer_keys:
                layer_path = streaming_feature_tensor_file_path(
                    feature_root,
                    sample_index=sample_index,
                    tensor_kind="feedforward",
                    layer_key=layer_key,
                )
                torch.save(da3_features_cpu[layer_key][batch_idx], layer_path)
                feature_index["feedforward_feature_paths"][layer_key].append(str(layer_path))

            sample_index += 1

    return feature_index


@torch.no_grad()
def extract_features(
    feedforward_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
):
    # Extract latent features from VAE and intermediate DA3 encoder blocks.
    # Features are stored in float16 on CPU to keep memory usage manageable.
    batch_size = data_loader.batch_size
    vae_features = []
    feedforward_features = {}


    handles = []
    if not is_da3_model(feedforward_model):
        raise NotImplementedError(
            f"Hook registration not implemented for {type(feedforward_model)}"
        )

    for step, data in tqdm(
        enumerate(data_loader), total=args.iterations_for_feature_extraction
    ):
        if step >= args.iterations_for_feature_extraction:
            break
        vae_images = data["vae_image_tensor"].to(vae_model.device)
        feedforward_img = data["feedforward_image_tensor"].to(vae_model.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if is_da3_model(feedforward_model):
                    da3_features = extract_da3_candidate_features(
                        feedforward_model,
                        feedforward_img,
                        candidate_layers=DA3_STITCH_CANDIDATE_LAYER_NAMES,
                    )
                    for name, feature in da3_features.items():
                        feature_cpu = feature.detach().cpu().clone().to(dtype=torch.float16)
                        if name not in feedforward_features:
                            feedforward_features[name] = []
                        feedforward_features[name].append(feature_cpu)
                else:
                    raise NotImplementedError(
                        f"Feature extraction not implemented for {type(feedforward_model)}"
                    )

                if isinstance(vae_model, AutoencoderKLWan):
                    latents = vae_model.encode(vae_images).latent_dist.sample()
                else:
                    raise NotImplementedError(
                        f"VAE encoding not implemented for {type(vae_model)}"
                    )
        # store VAE features by quantizing to float16 to save memory
        vae_features.append(latents.detach().cpu().to(dtype=torch.float16))

    for handle in handles:
        handle.remove()

    for name, feature in feedforward_features.items():
        feedforward_features[name] = torch.cat(feature, dim=0)
    vae_features = torch.cat(vae_features, dim=0)
    return vae_features, feedforward_features


def main(args: argparse.Namespace):
    args = apply_feedforward_model_defaults(args)
    if getattr(args, "training_stage", None) is None:
        args.training_stage = "distill"
    device = torch_device_setup()
    rank, world_size, is_main_rank = distributed_feature_state()
    is_da3_feedforward = getattr(args, "feedforward_model", None) == "da3"

    feature_output_dir = resolve_feature_output_dir(args.feature_save_path)
    os.makedirs(feature_output_dir, exist_ok=True)
    feature_file_path = resolve_feature_file_path(args.feature_save_path)
    feature_load_path = resolve_feature_file_path(
        getattr(args, "feature_load_path", None) or args.feature_save_path
    )
    feature_index = None
    streaming_storage_enabled = is_da3_feedforward
    if os.path.exists(feature_load_path):
        if is_main_rank:
            logger.info(f"Loading cached features from {feature_load_path}.")
            features = torch_load_cpu(feature_load_path, mmap=True)
            if is_streaming_feature_index(features):
                feature_index = features
            else:
                vae_features = features["vae_features"]
                feedforward_features = features["feedforward_features"]
    else:
        if getattr(args, "feature_load_path", None):
            raise FileNotFoundError(f"Cached feature file not found: {feature_load_path}")
        # First step: extract features and save them
        # Pipeline:
        # 1) Extract VAE + feedforward features.
        # 2) Fit a Conv3D stitching layer per encoder block.
        # 3) Rank blocks by reconstruction MSE.
        feed_forward_model = load_feedforward_model(args, device)
        video_vae = load_vae(args, device)
        train_dataloader, _ = create_stitching_dataloader(
            args.dataset, args, augmentation=False
        )
        logger.info(
            f"Extracting features on rank {rank}/{world_size} and preparing cache for {feature_file_path}."
        )
        if streaming_storage_enabled:
            extracted_features = extract_da3_features_to_disk(
                feedforward_model=feed_forward_model,
                vae_model=video_vae,
                data_loader=train_dataloader,
                feature_root=streaming_feature_dir(
                    feature_output_dir,
                    rank=rank,
                    world_size=world_size,
                ),
                iterations_for_feature_extraction=args.iterations_for_feature_extraction,
            )
        else:
            vae_features, feedforward_features = extract_features(
                feedforward_model=feed_forward_model,
                vae_model=video_vae,
                data_loader=train_dataloader,
            )
            extracted_features = {
                "vae_features": vae_features,
                "feedforward_features": feedforward_features,
            }
        if world_size > 1:
            shard_path = feature_shard_file_path(
                feature_file_path,
                rank=rank,
                world_size=world_size,
            )
            torch.save(extracted_features, shard_path)
            logger.info(f"Saved rank-{rank} feature shard to {shard_path}.")
            dist.barrier()
            if is_main_rank:
                logger.info(
                    f"Merging {world_size} feature shards into {feature_file_path}."
                )
                if streaming_storage_enabled:
                    extracted_features = merge_streaming_feature_shards(
                        feature_file_path,
                        world_size=world_size,
                    )
                else:
                    extracted_features = merge_feature_shards(
                        feature_file_path,
                        world_size=world_size,
                    )
                logger.info(f"Extracted features saved at {feature_file_path}.")
            dist.barrier()
        else:
            torch.save(extracted_features, feature_file_path)
            logger.info(f"Extracted features saved at {feature_file_path}.")
        if is_main_rank:
            if is_streaming_feature_index(extracted_features):
                feature_index = extracted_features
            else:
                vae_features = extracted_features["vae_features"]
                feedforward_features = extracted_features["feedforward_features"]

    if world_size > 1 and not is_main_rank:
        logger.info(
            f"Rank {rank} completed feature extraction shard; waiting for rank 0 to finish fitting."
        )
        dist.barrier()
        return

    cached_features_for_search = (
        feature_index
        if feature_index is not None
        else {
            "vae_features": vae_features,
            "feedforward_features": feedforward_features,
        }
    )
    if getattr(args, "search_input_space", "vae_latent") == "vae_decoder":
        decoder_vae_model = locals().get("video_vae")
        if decoder_vae_model is None:
            decoder_vae_model = load_vae(args, device)
        run_vae_decoder_layer_search(
            args=args,
            feature_output_dir=feature_output_dir,
            cached_features=cached_features_for_search,
            vae_model=decoder_vae_model,
        )
        if world_size > 1:
            dist.barrier()
        return

    # second step: MSE optimization
    if feature_index is not None:
        vae_feature_source = feature_index["vae_feature_paths"]
        layer_keys = feature_index["layer_keys"]
        first_vae_feature = first_feature_batch(vae_feature_source)
        in_channels = int(first_vae_feature.shape[0])
        default_target_spatial_size = int(first_vae_feature.shape[2])
    else:
        vae_feature_source = vae_features
        layer_keys = list(feedforward_features.keys())
        in_channels = int(vae_features.shape[1])
        default_target_spatial_size = int(vae_features.shape[3])

    if is_da3_feedforward:
        layer_keys = select_da3_layer_keys(layer_keys, args=args)

    total_batches = feature_batch_count(vae_feature_source)
    token_grid_shapes = None
    if feature_index is not None:
        token_grid_shapes = [
            tuple(shape)
            for shape in feature_index.get("feedforward_token_grid_shapes", [])
        ] or None

    for layer_key in layer_keys:
        layer_feature_source = (
            feature_index["feedforward_feature_paths"][layer_key]
            if feature_index is not None
            else feedforward_features[layer_key]
        )
        stitching_layer = args.stitching_layer_config.build(
            in_channels=in_channels
        )
        target_spatial_size = default_target_spatial_size
        if is_da3_feedforward:
            target_spatial_size = compute_da3_stitching_input_spatial_size(
                feedforward_resolution=args.feedforward_resolution,
                kernel_size=stitching_layer.kernel_size[1],
                stride=stitching_layer.stride[1],
                padding=stitching_layer.padding[1],
                dilation=stitching_layer.dilation[1],
            )
        state_dict_file_path = os.path.join(
            feature_output_dir, f"state_dict_{layer_key}.pt"
        )
        if os.path.exists(state_dict_file_path):
            logger.info(
                f"Stitching layer state dict already exists at {state_dict_file_path}."
            )
            logger.info(f"Loading stitching layer state dict for layer {layer_key}.")
            state_dict = torch.load(
                state_dict_file_path,
                map_location="cpu",
                weights_only=False,
            )
            stitching_layer.load_state_dict(state_dict, strict=True)
        else:
            # fitting stitching layer
            stitching_layer = fit_conv3d_streaming(
                stitching_layer,
                vae_feature_source,
                layer_feature_source,
                target_spatial_size=target_spatial_size,
                ridge=1e-4,
                token_grid_shapes=token_grid_shapes,
                feedforward_resolution=args.feedforward_resolution,
            )
            torch.save(
                stitching_layer.state_dict(),
                state_dict_file_path,
            )
            logger.info(
                f"Fitted stitching layer state dict saved at {state_dict_file_path}."
            )
        with torch.no_grad():
            mse, vox = 0.0, 0
            for batch_idx, (Z_b, Y_b) in enumerate(
                tqdm(
                    zip(vae_feature_source, layer_feature_source),
                    total=total_batches,
                )
            ):
                Z_b = load_feature_batch(Z_b)
                Y_b = load_feature_batch(Y_b)

                def upsampling_layer(x):
                    T_vae = x.shape[2]
                    T_original = (T_vae - 1) * 4 + 1
                    token_grid_shape = infer_token_grid_shape_from_batch(
                        Y_b,
                        batch_idx=batch_idx,
                        token_grid_shapes=token_grid_shapes,
                        feedforward_resolution=args.feedforward_resolution if is_da3_feedforward else None,
                    )
                    if token_grid_shape is not None:
                        current_target_spatial_shape = compute_conv_input_spatial_shape(
                            output_grid_shape=token_grid_shape,
                            kernel_size=(stitching_layer.kernel_size[1], stitching_layer.kernel_size[2]),
                            stride=(stitching_layer.stride[1], stitching_layer.stride[2]),
                            padding=(stitching_layer.padding[1], stitching_layer.padding[2]),
                            dilation=(stitching_layer.dilation[1], stitching_layer.dilation[2]),
                        )
                    else:
                        current_target_spatial_shape = normalize_spatial_size(target_spatial_size)
                        if current_target_spatial_shape is None:
                            current_target_spatial_shape = (
                                int(x.shape[3]),
                                int(x.shape[4]),
                            )
                    x = nn.functional.interpolate(
                        x,
                        size=[
                            T_original,
                            current_target_spatial_shape[0],
                            current_target_spatial_shape[1],
                        ],
                        mode="trilinear",
                        align_corners=True,
                    )
                    return x

                token_grid_shape = infer_token_grid_shape_from_batch(
                    Y_b,
                    batch_idx=batch_idx,
                    token_grid_shapes=token_grid_shapes,
                    feedforward_resolution=args.feedforward_resolution if is_da3_feedforward else None,
                )
                Z_b = Z_b.unsqueeze(0)
                Z_b = upsampling_layer(Z_b)
                Y_b = reshape_feature_tokens_to_grid(
                    Y_b,
                    token_grid_shape=token_grid_shape,
                    feedforward_resolution=args.feedforward_resolution if is_da3_feedforward else None,
                ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out
                pred = stitching_layer(Z_b.float())
                mse += (pred - Y_b).square().sum().item()
                vox += pred.numel()
            print("Training MSE:", mse / vox)
            with open(
                os.path.join(feature_output_dir, f"mse_{layer_key}.txt"), "w"
            ) as f:
                f.write(f"{mse / vox}\n")

    layers = list(layer_keys)
    mses = []
    for layer_key in layer_keys:
        with open(
            os.path.join(feature_output_dir, f"mse_{layer_key}.txt"), "r"
        ) as f:
            mse = float(f.readline().strip())
            mses.append(mse)

    # give recommendation
    best_layer = layers[mses.index(min(mses))]
    logger.info(f"Best stitching layer: {best_layer} with MSE: {min(mses)}")
    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # argument: iterations
    args = find_layer_stitching_argument().parse_args()
    main(args)
