from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
DA3_ROOT = THIS_DIR.parent
PLAN4D_ROOT = DA3_ROOT.parent


def sanitize_import_paths() -> None:
    if str(THIS_DIR) in sys.path:
        sys.path.remove(str(THIS_DIR))
    resolved_script_dir = THIS_DIR.resolve()
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() != resolved_script_dir
    ]

    preferred_paths = [str(DA3_ROOT), str(DA3_ROOT / "src"), str(PLAN4D_ROOT)]
    remaining_paths = [entry for entry in sys.path if entry not in preferred_paths]
    sys.path[:] = preferred_paths + remaining_paths


sanitize_import_paths()

from plan4d.da3_interface import DA3StreamingInterface
from depth_anything_3.utils.geometry import unproject_depth
from training.datasets.multi_scene_flow import build_scene_flow_val_loaders
from training.datasets.point_odyssey_scene_flow import PointOdysseySceneFlowDataset, _load_point_odyssey_depth, resize_and_center_crop_image
from training.losses.scene_flow_loss import SceneFlowLoss
from training.utils import (
    get_scene_flow_head,
    load_checkpoint_state,
    load_da3_with_scene_flow_head,
    strip_model_prefix,
)


LOGGER = logging.getLogger("eval_scene_flow_head")
ALLOWED_SCENE_FLOW_HEAD_MISSING_KEYS = frozenset({"norm.weight", "norm.bias"})
DEFAULT_INFERENCE_PROCESS_RES = 504
DEFAULT_INFERENCE_PROCESS_RES_METHOD = "upper_bound_resize"

GEOMETRY_VIEW_ELEV = 24.0
GEOMETRY_VIEW_AZIM = -58.0
GEOMETRY_MAX_POINTS = 50000
GEOMETRY_MAX_SEGMENTS = 2000
GEOMETRY_DEFORM_DARKEN = 0.6
GEOMETRY_MOTION_BASE_DARKEN = 0.8
SCENE_FLOW_PLY_MAX_POINTS = 200000
GEOMETRY_OVERLAY_IMAGE_SIZE = 552
GEOMETRY_MOTION_OVERLAY_VIEWS: tuple[tuple[str, float, float], ...] = (
    ("Front", 0.0, -90.0),
    ("Back", 0.0, 90.0),
    ("Left", 0.0, 180.0),
    ("Right", 0.0, 0.0),
    ("Top", 90.0, -90.0),
    ("Oblique", GEOMETRY_VIEW_ELEV, GEOMETRY_VIEW_AZIM),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DA3 scene-flow head checkpoint and export visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Training output root, checkpoint dir, or checkpoint file. Required unless using offline overlay export.",
    )
    parser.add_argument("--config", type=str, default="../configs/scene_flow.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None, help="Select a dataset-specific validation loader when multiple are available")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None, help="Override evaluation clip length when using direct split-based loading")
    parser.add_argument("--overlap", type=int, default=None, help="Override evaluation clip overlap when using direct split-based loading")
    parser.add_argument("--decoder-chunk-size", type=int, default=None)
    parser.add_argument(
        "--process-res",
        type=int,
        default=DEFAULT_INFERENCE_PROCESS_RES,
        help="Inference-time preprocessing max resolution before DA3 forward.",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default=DEFAULT_INFERENCE_PROCESS_RES_METHOD,
        help="Inference-time preprocessing strategy passed to scene-flow datasets.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=DEFAULT_INFERENCE_PROCESS_RES,
        help="Inference-time dataloader target height.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=DEFAULT_INFERENCE_PROCESS_RES,
        help="Inference-time dataloader target width.",
    )
    parser.add_argument("--save-vis-every-n", type=int, default=10)
    parser.add_argument(
        "--config-overrides-saved-datasets",
        action="store_true",
        help="Replace checkpoint-saved data.datasets with the datasets declared in --config while keeping other checkpoint data settings.",
    )
    parser.add_argument("--quiver-step", type=int, default=24)
    parser.add_argument("--vis-scale", type=float, default=1.0)
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Legacy flag name. Export a per-sample preview video (.mp4) instead of a GIF.",
    )
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument(
        "--save-geometry-motion-video",
        action="store_true",
        help=(
            "Use world-space point maps reconstructed from pred depth + pred ray for visualization "
            "and export a geometry-motion MP4 for each selected sample."
        ),
    )
    parser.add_argument(
        "--export-geometry-motion-overlays-from-dir",
        type=str,
        default=None,
        help=(
            "Offline mode: read an eval output directory containing predictions/*.npz and export only the "
            "geometry-motion overlay column as separate prediction and GT MP4s."
        ),
    )
    parser.add_argument(
        "--geometry-motion-overlay-output-dir",
        type=str,
        default=None,
        help="Output directory for offline geometry-motion overlay MP4s. Defaults to <input>/geometry_motion_overlays.",
    )
    parser.add_argument(
        "--geometry-motion-overlay-fps",
        type=int,
        default=6,
        help="FPS for offline geometry-motion overlay MP4s.",
    )
    parser.add_argument(
        "--geometry-motion-overlay-max-samples",
        type=int,
        default=None,
        help="Maximum saved prediction NPZ files to export in offline overlay mode.",
    )
    parser.add_argument(
        "--geometry-motion-overlay-renderer",
        type=str,
        default="numpy",
        choices=["numpy", "matplotlib"],
        help="Renderer backend for offline geometry-motion overlay images and videos.",
    )
    parser.add_argument(
        "--save-scene-flow-ply",
        action="store_true",
        help="Export ASCII PLY point clouds with per-vertex scene-flow attributes for selected samples.",
    )
    parser.add_argument(
        "--scene-flow-ply-every-n",
        type=int,
        default=1,
        help="Save scene-flow PLY for every N evaluated samples when --save-scene-flow-ply is set.",
    )
    parser.add_argument(
        "--scene-flow-ply-max-points",
        type=int,
        default=SCENE_FLOW_PLY_MAX_POINTS,
        help="Maximum vertices written per scene-flow PLY.",
    )
    parser.add_argument(
        "--metrics-on-pred-point-map",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Forces metric-space evaluation and geometry decode, "
            "but no longer changes the core loss/EPE definition."
        ),
    )
    parser.add_argument(
        "--eval-space",
        type=str,
        default="auto",
        choices=["auto", "native", "metric"],
        help="Evaluation space: native target domain, restored metric domain, or auto-select based on target type",
    )
    parser.add_argument("--ref-view-strategy", type=str, default="first")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logger(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("plan4d.da3_interface").setLevel(logging.WARNING)


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_data_roots(data_cfg: Dict[str, Any]) -> None:
    if "data_root" in data_cfg and data_cfg["data_root"] and not Path(data_cfg["data_root"]).is_absolute():
        data_cfg["data_root"] = str((DA3_ROOT / data_cfg["data_root"]).resolve())

    for spec in data_cfg.get("datasets", []):
        if "data_root" in spec and spec["data_root"] and not Path(spec["data_root"]).is_absolute():
            spec["data_root"] = str((DA3_ROOT / spec["data_root"]).resolve())
        if "builder_dir" in spec and spec["builder_dir"] and not Path(spec["builder_dir"]).is_absolute():
            spec["builder_dir"] = str((DA3_ROOT / spec["builder_dir"]).resolve())
        for manifest_key in ("train_manifest", "val_manifest"):
            manifest_path = spec.get(manifest_key)
            if manifest_path and not Path(manifest_path).is_absolute():
                spec[manifest_key] = str((DA3_ROOT / manifest_path).resolve())
        for manifest_map_key in ("train_manifests", "val_manifests"):
            manifest_map = spec.get(manifest_map_key)
            if not manifest_map:
                continue
            for split_name, manifest_path in list(manifest_map.items()):
                if manifest_path and not Path(manifest_path).is_absolute():
                    manifest_map[split_name] = str((DA3_ROOT / manifest_path).resolve())


def resolve_scene_flow_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path

    candidates: List[Path] = []
    if (path / "pytorch_model_fsdp.bin").is_file():
        candidates.append(path / "pytorch_model_fsdp.bin")
    for name in ("best.pt", "last.pt"):
        candidate = path / name
        if candidate.is_file():
            candidates.append(candidate)

    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        for rel in (
            ("best", "pytorch_model_fsdp.bin"),
            ("best.pt",),
            ("last", "pytorch_model_fsdp.bin"),
            ("last.pt",),
        ):
            candidate = checkpoints_dir.joinpath(*rel)
            if candidate.is_file():
                candidates.append(candidate)

        epoch_dirs = sorted(
            [item for item in checkpoints_dir.iterdir() if item.is_dir() and item.name.startswith("epoch_")],
            reverse=True,
        )
        for epoch_dir in epoch_dirs:
            candidate = epoch_dir / "pytorch_model_fsdp.bin"
            if candidate.is_file():
                candidates.append(candidate)
                break

        epoch_files = sorted(
            [item for item in checkpoints_dir.iterdir() if item.is_file() and item.name.startswith("epoch_")],
            reverse=True,
        )
        if epoch_files:
            candidates.append(epoch_files[0])

    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Cannot resolve a scene-flow checkpoint from: {path}")


def find_training_output_root(checkpoint_path: Path) -> Optional[Path]:
    checkpoint_path = checkpoint_path.resolve()
    search_roots: Iterable[Path]
    if checkpoint_path.is_dir():
        search_roots = (checkpoint_path, *checkpoint_path.parents)
    else:
        search_roots = (checkpoint_path.parent, *checkpoint_path.parents)

    for root in search_roots:
        if (root / "resolved_config.json").is_file():
            return root
        checkpoints_dir = root / "checkpoints"
        if checkpoints_dir.is_dir():
            return root
    return None


def find_saved_config(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    candidates: List[Path] = []

    if checkpoint_path.is_file():
        candidates.append(checkpoint_path.parent / "meta.json")
        candidates.extend(parent / "resolved_config.json" for parent in checkpoint_path.parents)
    else:
        candidates.append(checkpoint_path / "meta.json")
        candidates.extend(parent / "resolved_config.json" for parent in checkpoint_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        payload = load_json_file(candidate)
        if candidate.name == "meta.json" and isinstance(payload.get("config"), dict):
            return payload["config"]
        if isinstance(payload, dict):
            return payload
    return None


def resolve_eval_config(config_path: Path, checkpoint_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    file_config = load_yaml_config(str(config_path))
    config = copy.deepcopy(file_config)
    saved_config = find_saved_config(checkpoint_path)
    if saved_config:
        config = deep_update(config, saved_config)
        if args.config_overrides_saved_datasets:
            file_data_cfg = file_config.get("data") if isinstance(file_config, Mapping) else None
            if isinstance(file_data_cfg, Mapping) and "datasets" in file_data_cfg:
                config.setdefault("data", {})
                config["data"]["datasets"] = copy.deepcopy(file_data_cfg["datasets"])

    split = args.split or config["data"].get("val_split") or config["data"].get("split", "test")
    config["data"]["split"] = split
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    else:
        config["data"]["batch_size"] = int(config["data"].get("batch_size", 1))
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.max_scenes is not None:
        config["data"]["max_scenes"] = args.max_scenes
    if args.chunk_size is not None:
        config["data"]["chunk_size"] = int(args.chunk_size)
    if args.overlap is not None:
        config["data"]["overlap"] = int(args.overlap)
    if args.decoder_chunk_size is not None:
        config["model"]["decoder_chunk_size"] = args.decoder_chunk_size
    if args.process_res is not None:
        config["data"]["process_res"] = int(args.process_res)
    if args.process_res_method is not None:
        config["data"]["process_res_method"] = str(args.process_res_method)
    if args.target_height is not None:
        config["data"]["target_height"] = int(args.target_height)
    if args.target_width is not None:
        config["data"]["target_width"] = int(args.target_width)

    if not Path(config["model"]["weights_dir"]).is_absolute():
        config["model"]["weights_dir"] = str((DA3_ROOT / config["model"]["weights_dir"]).resolve())
    resolve_data_roots(config["data"])
    return config


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_scene_flow_head_checkpoint(scene_flow_head: torch.nn.Module, checkpoint_path: str | Path) -> None:
    state_dict = load_checkpoint_state(str(checkpoint_path))
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)}")

    if "scene_flow_head" in state_dict and isinstance(state_dict["scene_flow_head"], dict):
        state_dict = state_dict["scene_flow_head"]

    state_dict = strip_model_prefix(state_dict)
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "scene_flow_head.")

    missing, unexpected = scene_flow_head.load_state_dict(state_dict, strict=False)
    missing_keys = set(missing)
    unexpected_keys = set(unexpected)
    if missing_keys.issubset(ALLOWED_SCENE_FLOW_HEAD_MISSING_KEYS) and not unexpected_keys:
        return
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load scene_flow_head cleanly from {checkpoint_path}: "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}"
        )


def _select_eval_loader(
    val_loaders: Mapping[str, DataLoader],
    split: str,
    dataset_name: Optional[str] = None,
) -> DataLoader:
    if not val_loaders:
        raise RuntimeError("No validation dataloaders were built for evaluation")

    if dataset_name is not None:
        exact_match = val_loaders.get(dataset_name)
        if exact_match is not None:
            return exact_match

        prefix_matches = [key for key in val_loaders.keys() if key.split(":")[0] == dataset_name]
        if len(prefix_matches) == 1:
            return val_loaders[prefix_matches[0]]
        if len(prefix_matches) > 1:
            raise ValueError(
                f"Multiple validation loaders match dataset_name={dataset_name}: {sorted(prefix_matches)}"
            )
        raise ValueError(
            f"dataset_name={dataset_name} did not match any validation loader: {sorted(val_loaders.keys())}"
        )

    split_matches = [key for key in val_loaders.keys() if key == split or key.endswith(f":{split}")]
    if len(split_matches) == 1:
        return val_loaders[split_matches[0]]
    if len(split_matches) > 1:
        raise ValueError(
            f"split={split} matched multiple validation loaders; pass --dataset-name to disambiguate: {sorted(split_matches)}"
        )
    if len(val_loaders) == 1:
        return next(iter(val_loaders.values()))

    raise ValueError(
        f"Could not infer which validation loader to use for split={split}; available: {sorted(val_loaders.keys())}"
    )


def _maybe_create_direct_point_odyssey_dataloader(data_cfg: Dict[str, Any]) -> Optional[DataLoader]:
    data_root = data_cfg.get("data_root")
    if data_root:
        dataset = PointOdysseySceneFlowDataset(
            data_root=data_root,
            split=data_cfg["split"],
            chunk_size=int(data_cfg["chunk_size"]),
            overlap=int(data_cfg["overlap"]),
            target_height=int(data_cfg["target_height"]),
            target_width=int(data_cfg["target_width"]),
            max_scenes=data_cfg.get("max_scenes"),
            flow_target_space=data_cfg.get("flow_target_space", "metric"),
            relative_scale_source=data_cfg.get("relative_scale_source", "flow_p95"),
            geometry_scale_stat=data_cfg.get("geometry_scale_stat", "mean_radius"),
            flow_scale_stat=data_cfg.get("flow_scale_stat", "p95"),
            flow_scale_eps=float(data_cfg.get("flow_scale_eps", 1e-6)),
            flow_target_amplification=float(data_cfg.get("flow_target_amplification", 1.0)),
        )
        return DataLoader(
            dataset,
            batch_size=int(data_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=False,
        )

    point_odyssey_specs = [
        spec
        for spec in data_cfg.get("datasets", [])
        if str(spec.get("name")) == "point_odyssey" and bool(spec.get("enabled_for_val", False))
    ]
    if len(point_odyssey_specs) != 1:
        return None

    spec = point_odyssey_specs[0]
    dataset = PointOdysseySceneFlowDataset(
        data_root=spec["data_root"],
        split=data_cfg["split"],
        chunk_size=int(data_cfg["chunk_size"]),
        overlap=int(data_cfg["overlap"]),
        target_height=int(data_cfg["target_height"]),
        target_width=int(data_cfg["target_width"]),
        max_scenes=spec.get("max_val_scenes", data_cfg.get("max_scenes")),
        flow_target_space=data_cfg.get("flow_target_space", "metric"),
        relative_scale_source=data_cfg.get("relative_scale_source", "flow_p95"),
        geometry_scale_stat=data_cfg.get("geometry_scale_stat", "mean_radius"),
        flow_scale_stat=data_cfg.get("flow_scale_stat", "p95"),
        flow_scale_eps=float(data_cfg.get("flow_scale_eps", 1e-6)),
        flow_target_amplification=float(data_cfg.get("flow_target_amplification", 1.0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )


def create_dataloader(
    data_cfg: Dict[str, Any],
    dataset_name: Optional[str] = None,
    prefer_direct_point_odyssey: bool = False,
) -> DataLoader:
    if prefer_direct_point_odyssey:
        direct_loader = _maybe_create_direct_point_odyssey_dataloader(data_cfg)
        if direct_loader is not None:
            return direct_loader

    if "datasets" in data_cfg:
        val_loaders = build_scene_flow_val_loaders(data_cfg, num_workers_override=data_cfg.get("num_workers"))
        return _select_eval_loader(val_loaders, split=str(data_cfg["split"]), dataset_name=dataset_name)

    direct_loader = _maybe_create_direct_point_odyssey_dataloader(data_cfg)
    if direct_loader is None:
        raise RuntimeError("Failed to create evaluation dataloader")
    return direct_loader


def metric_dict_to_float(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in loss_dict.items()}


def resolve_eval_space(
    target_space: str,
    eval_space: str,
    metrics_on_pred_point_map: bool = False,
) -> str:
    if eval_space not in {"auto", "native", "metric"}:
        raise ValueError(f"Unsupported eval_space: {eval_space}")

    resolved = eval_space
    if resolved == "auto":
        resolved = "metric" if target_space == "relative" else "native"
    if metrics_on_pred_point_map and resolved == "native":
        resolved = "metric"
    return resolved


def convert_flow_to_eval_space(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    target_space: str,
    eval_space: str,
    scene_flow_scale: Optional[float] = None,
    flow_target_amplification: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eval_space == "metric" and target_space == "relative":
        if scene_flow_scale is None:
            raise ValueError("metric evaluation for relative scene flow requires scene_flow_scale")
        scale = float(scene_flow_scale) / float(flow_target_amplification)
        return pred_flow * scale, target_flow * scale
    return pred_flow, target_flow


def resize_btchw(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"Expected tensor shape [B, T, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[-2:] == target_size:
        return tensor
    batch_size, sequence_length, channels, height, width = tensor.shape
    resized = F.interpolate(
        tensor.reshape(batch_size * sequence_length, channels, height, width),
        size=target_size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return resized.reshape(batch_size, sequence_length, channels, target_size[0], target_size[1])


def resize_bthw_mask(mask: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape [B, T, H, W], got {tuple(mask.shape)}")
    if mask.shape[-2:] == target_size:
        return mask
    batch_size, sequence_length, height, width = mask.shape
    resized = F.interpolate(
        mask.reshape(batch_size * sequence_length, 1, height, width).float(),
        size=target_size,
        mode="nearest",
    )
    return resized.reshape(batch_size, sequence_length, target_size[0], target_size[1])


def resize_thwc_tensor(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Expected tensor shape [T, H, W, C], got {tuple(tensor.shape)}")
    if tensor.shape[1:3] == target_size:
        return tensor
    sequence_length, height, width, channels = tensor.shape
    resized = F.interpolate(
        tensor.permute(0, 3, 1, 2).float(),
        size=target_size,
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
    )
    return resized.permute(0, 2, 3, 1).reshape(sequence_length, target_size[0], target_size[1], channels)


def scalar_to_rgb(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cmap_name: str = "inferno",
    clip_percentile: float = 99.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask) > 0.5

    rgb = np.zeros(values.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return rgb

    scale = float(np.percentile(values[valid], clip_percentile))
    scale = max(scale, 1e-6)
    normalized = np.clip(values / scale, 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgb = (cmap(normalized)[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def scene_flow_to_rgb(
    flow: np.ndarray,
    mask: Optional[np.ndarray] = None,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    magnitude = np.linalg.norm(flow, axis=-1)
    valid = np.isfinite(magnitude)
    if mask is not None:
        valid &= np.asarray(mask) > 0.5

    rgb = np.zeros(flow.shape, dtype=np.uint8)
    if not valid.any():
        return rgb

    scale = float(np.percentile(magnitude[valid], clip_percentile))
    scale = max(scale, 1e-6)

    direction = flow / np.maximum(magnitude[..., None], 1e-6)
    direction_rgb = (direction + 1.0) * 0.5
    intensity = np.clip(magnitude / scale, 0.0, 1.0)[..., None]
    rgb_float = np.clip(direction_rgb * intensity, 0.0, 1.0)
    rgb = (rgb_float * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def compute_point_map_valid_mask(point_map_world: np.ndarray) -> np.ndarray:
    points = np.asarray(point_map_world, dtype=np.float32)
    finite = np.isfinite(points).all(axis=-1)
    point_norm = np.linalg.norm(points, axis=-1)
    return finite & np.isfinite(point_norm) & (point_norm > 1e-8)


def convert_flow_frame0_to_world(flow: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 5:
        raise ValueError(f"Expected flow shape [B, T, 3, H, W], got {tuple(flow.shape)}")
    if extrinsics.ndim != 4 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics shape [B, T, 4, 4], got {tuple(extrinsics.shape)}")

    rot_w2c_frame0 = extrinsics[:, 0, :3, :3]
    rot_c2w_frame0 = rot_w2c_frame0.transpose(-1, -2).to(dtype=flow.dtype)
    return torch.einsum("bij,btjhw->btihw", rot_c2w_frame0, flow)


def build_world_point_map_from_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth.permute(0, 1, 3, 4, 2).contiguous()
    elif depth.ndim == 4:
        depth = depth.unsqueeze(-1)
    else:
        raise ValueError(f"Expected depth shape [B, T, 1, H, W] or [B, T, H, W], got {tuple(depth.shape)}")

    depth = depth.float()
    intrinsics = intrinsics.float()
    extrinsics = extrinsics.float()
    c2w = torch.inverse(extrinsics)
    return unproject_depth(depth, intrinsics, c2w=c2w)


def upsample_ray_with_renormalization(
    pred_ray: torch.Tensor,
    target_size: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    if pred_ray.ndim != 5 or pred_ray.shape[2] < 6:
        raise ValueError(f"Expected pred_ray shape [B, T, 6, H, W], got {tuple(pred_ray.shape)}")

    batch_size, sequence_length, channels, height, width = pred_ray.shape
    ray_flat = pred_ray.reshape(batch_size * sequence_length, channels, height, width)
    direction = ray_flat[:, :3]
    origin = ray_flat[:, 3:6]

    direction_up = F.interpolate(
        direction,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    origin_up = F.interpolate(
        origin,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    ray_up = torch.cat([direction_up, origin_up], dim=1)
    return ray_up.reshape(batch_size, sequence_length, 6, target_size[0], target_size[1])


def standardize_depth_prediction(pred_depth: torch.Tensor) -> torch.Tensor:
    if pred_depth.ndim == 4:
        return pred_depth.unsqueeze(2).float()
    if pred_depth.ndim == 5 and pred_depth.shape[2] == 1:
        return pred_depth.float()
    if pred_depth.ndim == 5 and pred_depth.shape[-1] == 1:
        return pred_depth.permute(0, 1, 4, 2, 3).float()
    raise ValueError(f"Unsupported pred_depth shape: {tuple(pred_depth.shape)}")


def standardize_ray_prediction(
    pred_ray: torch.Tensor,
    target_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    if pred_ray.ndim != 5:
        raise ValueError(f"Expected pred_ray shape [B, T, 6/7, H, W] or [B, T, H, W, 6/7], got {tuple(pred_ray.shape)}")

    if pred_ray.shape[-1] in (6, 7) and pred_ray.shape[2] not in (6, 7):
        pred_ray = pred_ray.permute(0, 1, 4, 2, 3).contiguous()
    elif pred_ray.shape[2] not in (6, 7):
        raise ValueError(f"Cannot infer pred_ray channel axis from shape {tuple(pred_ray.shape)}")

    if pred_ray.shape[2] > 6:
        pred_ray = pred_ray[:, :, :6]

    pred_ray = pred_ray.float()
    if target_size is not None and pred_ray.shape[-2:] != target_size:
        pred_ray = upsample_ray_with_renormalization(pred_ray, target_size=target_size, mode="bilinear")
    return pred_ray


def align_depth_least_squares(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    sequence_level: bool = True,
) -> torch.Tensor:
    orig_shape = pred.shape
    mask_flat = None

    if pred.dim() == 5:
        if sequence_level:
            batch_size = pred.shape[0]
            pred_flat = pred.reshape(batch_size, -1)
            target_flat = target.reshape(batch_size, -1)
            if valid_mask is not None:
                mask_flat = valid_mask.reshape(batch_size, -1) > 0.5
        else:
            pred_flat = pred.flatten(0, 2).flatten(1)
            target_flat = target.flatten(0, 2).flatten(1)
            if valid_mask is not None:
                mask_flat = valid_mask.flatten(0, 2).flatten(1) > 0.5
    elif pred.dim() == 4:
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)
        if valid_mask is not None:
            mask_flat = valid_mask.flatten(1) > 0.5
    else:
        raise ValueError(f"Unsupported pred shape for depth alignment: {tuple(pred.shape)}")

    if mask_flat is None:
        mask_flat = (target_flat > 0) & torch.isfinite(target_flat) & torch.isfinite(pred_flat)

    pred_masked = pred_flat.masked_fill(~mask_flat, 0)
    target_masked = target_flat.masked_fill(~mask_flat, 0)
    count = mask_flat.sum(dim=1, keepdim=True)
    valid_samples = count >= 10
    safe_count = count.clamp(min=1)

    pred_mean = pred_masked.sum(dim=1, keepdim=True) / safe_count
    target_mean = target_masked.sum(dim=1, keepdim=True) / safe_count

    pred_centered = (pred_flat - pred_mean) * mask_flat
    target_centered = (target_flat - target_mean) * mask_flat

    pred_var = (pred_centered ** 2).sum(dim=1, keepdim=True) + eps
    cov = (pred_centered * target_centered).sum(dim=1, keepdim=True)

    scale = (cov / pred_var).clamp(0.01, 100.0).detach()
    shift = (target_mean - scale * pred_mean).detach()
    aligned_flat = scale * pred_flat + shift

    if not valid_samples.all():
        aligned_flat = torch.where(valid_samples, aligned_flat, pred_flat)
    return aligned_flat.reshape(orig_shape)


def _find_depth_path(scene_dir: Path, frame_idx: int) -> Path:
    depth_path = scene_dir / "depths" / f"depth_{frame_idx:05d}.png"
    if depth_path.exists():
        return depth_path
    raise FileNotFoundError(f"Cannot find depth for frame {frame_idx} under {scene_dir}")


def load_gt_depth_batch(
    batch: Mapping[str, Any],
    data_root: Path,
    split: str,
    target_height: int,
    target_width: int,
    target_length: int,
) -> torch.Tensor:
    batch_gt_depth = batch.get("gt_depth")
    if batch_gt_depth is not None:
        depth_batch = torch.as_tensor(batch_gt_depth).float()
        if depth_batch.ndim == 5 and depth_batch.shape[2] == 1:
            depth_batch = depth_batch.squeeze(2)
        elif depth_batch.ndim != 4:
            raise ValueError(f"Expected gt_depth with shape [B, T, H, W] or [B, T, 1, H, W], got {tuple(depth_batch.shape)}")

        batch_size, sequence_length, height, width = depth_batch.shape
        if (height, width) != (target_height, target_width):
            depth_batch = F.interpolate(
                depth_batch.reshape(batch_size * sequence_length, 1, height, width),
                size=(target_height, target_width),
                mode="nearest",
            ).reshape(batch_size, sequence_length, target_height, target_width)
        if sequence_length < target_length:
            pad = target_length - sequence_length
            depth_batch = torch.cat([depth_batch, depth_batch[:, -1:].repeat(1, pad, 1, 1)], dim=1)
        elif sequence_length > target_length:
            depth_batch = depth_batch[:, :target_length]
        return depth_batch.contiguous()

    depth_sequences: list[np.ndarray] = []
    batch_size = int(batch["rgbs"].shape[0])

    for batch_index in range(batch_size):
        scene_name = str(extract_batch_item(batch["scene_name"], batch_index))
        start_idx = int(extract_batch_item(batch["start_idx"], batch_index).item())
        sequence_length = int(extract_batch_item(batch["sequence_length"], batch_index).item())
        scene_dir_item = batch.get("scene_dir")
        if scene_dir_item is not None:
            scene_dir = Path(str(extract_batch_item(scene_dir_item, batch_index)))
        else:
            scene_dir = Path(data_root) / split / scene_name

        frames: list[np.ndarray] = []
        for frame_idx in range(start_idx, start_idx + sequence_length):
            depth_path = _find_depth_path(scene_dir, frame_idx)
            depth = _load_point_odyssey_depth(depth_path)
            depth, _ = resize_and_center_crop_image(
                depth,
                target_height,
                target_width,
                interpolation=cv2.INTER_NEAREST,
            )
            frames.append(depth)

        depth_arr = np.stack(frames, axis=0).astype(np.float32)
        if depth_arr.shape[0] < target_length:
            pad = target_length - depth_arr.shape[0]
            depth_arr = np.concatenate([depth_arr, np.repeat(depth_arr[-1:], pad, axis=0)], axis=0)
        depth_sequences.append(depth_arr)

    return torch.from_numpy(np.stack(depth_sequences, axis=0)).float()


def build_world_point_map_from_depth_and_ray(
    depth: torch.Tensor,
    ray: torch.Tensor,
) -> torch.Tensor:
    depth = standardize_depth_prediction(depth)
    ray = standardize_ray_prediction(ray, target_size=depth.shape[-2:])
    depth = depth.squeeze(2)
    direction = ray[:, :, :3]
    origin = ray[:, :, 3:6]

    point_map = origin + depth.unsqueeze(2).float() * direction
    return point_map.permute(0, 1, 3, 4, 2).contiguous()


def compute_apd_at_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    deform_mask: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(f"Expected pred/target shape [B, T, 3, H, W], got {tuple(pred.shape)} and {tuple(target.shape)}")
    if deform_mask.ndim != 4:
        raise ValueError(f"Expected deform_mask shape [B, T, H, W], got {tuple(deform_mask.shape)}")

    valid = deform_mask > 0.5
    if not valid.any():
        return pred.new_zeros(())

    per_pixel_epe = torch.linalg.norm(pred - target, dim=2)
    return (per_pixel_epe[valid] < threshold).float().mean()


def compute_metrics_on_pred_point_map(
    pred_flow: torch.Tensor,
    target_flow: torch.Tensor,
    point_map_world: torch.Tensor,
    deform_mask: torch.Tensor,
    loss_type: str,
    zero_last_weight: float,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if point_map_world.ndim != 5 or point_map_world.shape[-1] != 3:
        raise ValueError(
            f"Expected point_map_world shape [B, T, H, W, 3], got {tuple(point_map_world.shape)}"
        )

    point_map_world = point_map_world.permute(0, 1, 4, 2, 3).to(dtype=pred_flow.dtype)
    pred_points = point_map_world + pred_flow
    target_points = point_map_world + target_flow

    valid = deform_mask > 0.5
    valid_ratio = valid.float().mean()
    diff = pred_points - target_points
    per_pixel_l1 = diff.abs().sum(dim=2)
    per_pixel_l2 = diff.square().sum(dim=2)
    per_pixel_epe = torch.linalg.norm(diff, dim=2)

    if valid.any():
        if loss_type == "l1":
            base_loss = per_pixel_l1[valid].mean()
        elif loss_type == "l2":
            base_loss = per_pixel_l2[valid].mean()
        elif loss_type == "epe":
            base_loss = per_pixel_epe[valid].mean()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        epe = per_pixel_epe[valid].mean()
    else:
        base_loss = pred_flow.new_zeros(())
        epe = pred_flow.new_zeros(())

    zero_last = torch.linalg.norm(pred_flow[:, -1], dim=1).mean()
    total = base_loss + zero_last_weight * zero_last

    return {
        "loss": total,
        "loss_sf": base_loss,
        "loss_zero_last": zero_last_weight * zero_last,
        "epe": epe,
        "APD0.05": compute_apd_at_threshold(pred_points, target_points, deform_mask, threshold=0.05),
        "valid_ratio": valid_ratio.clamp_min(eps),
        "last_frame_zero_epe": zero_last,
    }


def render_scene_flow_quiver(
    rgb: np.ndarray,
    flow: np.ndarray,
    mask: Optional[np.ndarray] = None,
    step: int = 24,
    clip_percentile: float = 95.0,
) -> np.ndarray:
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    flow = np.asarray(flow, dtype=np.float32)
    height, width = canvas.shape[:2]

    valid = np.ones((height, width), dtype=bool)
    if mask is not None:
        valid &= np.asarray(mask) > 0.5

    ys = np.arange(step // 2, height, max(step, 1))
    xs = np.arange(step // 2, width, max(step, 1))
    if ys.size == 0 or xs.size == 0:
        return canvas

    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    sample_valid = valid[grid_y, grid_x]
    if not sample_valid.any():
        return canvas

    sample_flow = flow[grid_y, grid_x]
    sample_u = sample_flow[..., 0][sample_valid]
    sample_v = sample_flow[..., 1][sample_valid]
    sample_x = grid_x[sample_valid]
    sample_y = grid_y[sample_valid]

    sample_mag = np.sqrt(sample_u ** 2 + sample_v ** 2)
    scale = float(np.percentile(sample_mag, clip_percentile)) if sample_mag.size > 0 else 1.0
    scale = max(scale, 1e-6)
    colors = scalar_to_rgb(sample_mag, np.ones_like(sample_mag, dtype=np.float32), cmap_name="turbo", clip_percentile=95.0)

    for x, y, u, v, color in zip(sample_x, sample_y, sample_u, sample_v, colors):
        dx = int(np.clip(u / scale, -1.0, 1.0) * step * 0.75)
        dy = int(np.clip(-v / scale, -1.0, 1.0) * step * 0.75)
        end_point = (int(np.clip(x + dx, 0, width - 1)), int(np.clip(y + dy, 0, height - 1)))
        color_bgr = tuple(int(channel) for channel in color[::-1])
        cv2.arrowedLine(canvas, (int(x), int(y)), end_point, color_bgr, 1, tipLength=0.25)
    return canvas


def add_tile_label(image: np.ndarray, text: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def maybe_resize(image: np.ndarray, vis_scale: float) -> np.ndarray:
    if abs(vis_scale - 1.0) < 1e-6:
        return image
    height, width = image.shape[:2]
    target_size = (max(int(width * vis_scale), 1), max(int(height * vis_scale), 1))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def make_geometry_colors(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    darken: float = 1.0,
) -> np.ndarray:
    colors = np.clip(np.asarray(rgb, dtype=np.float32) * float(darken), 0.0, 255.0).astype(np.uint8)
    colors[~np.asarray(valid_mask, dtype=bool)] = 0
    return colors


def _extract_valid_points(
    point_map_world: np.ndarray,
    deform_mask: Optional[np.ndarray],
    flow_world: Optional[np.ndarray] = None,
    use_deformed_points: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(point_map_world, dtype=np.float32)
    if use_deformed_points:
        if flow_world is None:
            raise ValueError("use_deformed_points=True requires flow_world")
        points = points + np.asarray(flow_world, dtype=np.float32)

    valid = np.isfinite(points).all(axis=-1)
    point_norm = np.linalg.norm(points, axis=-1)
    valid &= np.isfinite(point_norm) & (point_norm > 1e-8)
    if deform_mask is not None:
        valid &= np.asarray(deform_mask) > 0.5
    return points, valid


def make_motion_mask(
    flow_world: np.ndarray,
    deform_mask: np.ndarray,
    scene_scale: float,
    motion_quantile: float = 90.0,
    min_ratio: float = 0.01,
) -> np.ndarray:
    flow = np.asarray(flow_world, dtype=np.float32)
    mag = np.linalg.norm(flow, axis=-1)
    valid = (np.asarray(deform_mask) > 0.5) & np.isfinite(mag)
    if not valid.any():
        return valid

    safe_scene_scale = max(float(scene_scale), 1e-6)
    thr_q = float(np.percentile(mag[valid], motion_quantile))
    thr_abs = float(min_ratio) * safe_scene_scale
    threshold = max(thr_q, thr_abs)
    return valid & (mag >= threshold)


def flatten_point_map_for_projection(
    point_map_world: np.ndarray,
    rgb: np.ndarray,
    deform_mask: Optional[np.ndarray],
    use_deformed_points: bool,
    flow_world: Optional[np.ndarray] = None,
    darken: float = 1.0,
    max_points: int = 50000,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    points, valid = _extract_valid_points(
        point_map_world=point_map_world,
        deform_mask=deform_mask,
        flow_world=flow_world,
        use_deformed_points=use_deformed_points,
    )
    colors = make_geometry_colors(rgb, valid, darken=darken)

    flat_points = points[valid].reshape(-1, 3)
    flat_colors = colors[valid].reshape(-1, 3)

    if flat_points.shape[0] > max_points:
        if rng is None:
            rng = np.random.default_rng(0)
        indices = rng.choice(flat_points.shape[0], size=max_points, replace=False)
        flat_points = flat_points[indices]
        flat_colors = flat_colors[indices]

    return flat_points, flat_colors


def write_scene_flow_ply(
    output_path: Path,
    *,
    point_map_world: np.ndarray,
    flow_world: np.ndarray,
    rgbs: Optional[np.ndarray] = None,
    deform_mask: Optional[np.ndarray] = None,
    frame_stride: int = 1,
    max_points: int = SCENE_FLOW_PLY_MAX_POINTS,
) -> None:
    points = np.asarray(point_map_world, dtype=np.float32)
    flow = np.asarray(flow_world, dtype=np.float32)
    if points.shape != flow.shape or points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(
            f"point_map_world and flow_world must both have shape [T,H,W,3], got {points.shape} and {flow.shape}"
        )

    stride = max(int(frame_stride), 1)
    points = points[::stride]
    flow = flow[::stride]
    if rgbs is None:
        colors = np.full(points.shape, 255, dtype=np.uint8)
    else:
        colors = np.asarray(rgbs, dtype=np.uint8)[::stride]
        if colors.shape != points.shape:
            raise ValueError(f"rgbs must have shape [T,H,W,3], got {colors.shape}; expected {points.shape}")

    valid = np.isfinite(points).all(axis=-1) & np.isfinite(flow).all(axis=-1)
    point_norm = np.linalg.norm(points, axis=-1)
    valid &= np.isfinite(point_norm) & (point_norm > 1e-8)
    if deform_mask is not None:
        mask = np.asarray(deform_mask)[::stride]
        if mask.shape != points.shape[:-1]:
            raise ValueError(f"deform_mask must have shape [T,H,W], got {mask.shape}; expected {points.shape[:-1]}")
        valid &= mask > 0.5

    flat_points = points[valid].reshape(-1, 3)
    flat_flow = flow[valid].reshape(-1, 3)
    flat_colors = colors[valid].reshape(-1, 3)
    if flat_points.shape[0] > int(max_points):
        indices = np.linspace(0, flat_points.shape[0] - 1, int(max_points), dtype=np.int64)
        flat_points = flat_points[indices]
        flat_flow = flat_flow[indices]
        flat_colors = flat_colors[indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {flat_points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property float flow_x\n")
        file.write("property float flow_y\n")
        file.write("property float flow_z\n")
        file.write("end_header\n")
        for point, color, flow_vec in zip(flat_points, flat_colors, flat_flow):
            file.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{flow_vec[0]:.8g} {flow_vec[1]:.8g} {flow_vec[2]:.8g}\n"
            )


def compute_axis_limits(*point_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_sets = [points for points in point_sets if points.size > 0]
    if not valid_sets:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)

    stacked = np.concatenate(valid_sets, axis=0)
    lower = np.percentile(stacked, 2.0, axis=0)
    upper = np.percentile(stacked, 98.0, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-3)
    min_xyz = center - radius
    max_xyz = center + radius
    return min_xyz.astype(np.float32), max_xyz.astype(np.float32)


def compute_sequence_geometry_axis_limits(
    point_map_world: np.ndarray,
    pred_flow_world: np.ndarray,
    deform_mask: np.ndarray,
    max_points_per_frame: int = 25000,
) -> tuple[np.ndarray, np.ndarray]:
    point_sets: list[np.ndarray] = []
    rng = np.random.default_rng(0)

    for frame_idx in range(int(point_map_world.shape[0])):
        base_points, _ = flatten_point_map_for_projection(
            point_map_world=point_map_world[frame_idx],
            rgb=np.zeros_like(point_map_world[frame_idx], dtype=np.uint8),
            deform_mask=deform_mask[frame_idx],
            use_deformed_points=False,
            max_points=max_points_per_frame,
            rng=rng,
        )
        deformed_points, _ = flatten_point_map_for_projection(
            point_map_world=point_map_world[frame_idx],
            rgb=np.zeros_like(point_map_world[frame_idx], dtype=np.uint8),
            deform_mask=deform_mask[frame_idx],
            use_deformed_points=True,
            flow_world=pred_flow_world[frame_idx],
            max_points=max_points_per_frame,
            rng=rng,
        )
        if base_points.size > 0:
            point_sets.append(base_points)
        if deformed_points.size > 0:
            point_sets.append(deformed_points)

    return compute_axis_limits(*point_sets)


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    try:
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    except AttributeError:
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    finally:
        plt.close(fig)
    return image


def sample_motion_segments(
    point_map_world: np.ndarray,
    flow_world: np.ndarray,
    deform_mask: np.ndarray,
    scene_scale: float,
    max_segments: int = GEOMETRY_MAX_SEGMENTS,
) -> tuple[np.ndarray, np.ndarray]:
    points0, valid_points = _extract_valid_points(
        point_map_world=point_map_world,
        deform_mask=deform_mask,
        use_deformed_points=False,
    )
    flow = np.asarray(flow_world, dtype=np.float32)
    motion_mask = make_motion_mask(
        flow_world=flow,
        deform_mask=deform_mask,
        scene_scale=scene_scale,
        motion_quantile=92.0,
        min_ratio=0.01,
    )
    valid = motion_mask & valid_points & np.isfinite(flow).all(axis=-1)
    indices = np.argwhere(valid)
    if indices.shape[0] == 0:
        return np.empty((0, 2, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    rng = np.random.default_rng(0)
    if indices.shape[0] > max_segments:
        indices = indices[rng.choice(indices.shape[0], size=max_segments, replace=False)]

    y, x = indices[:, 0], indices[:, 1]
    start_points = points0[y, x]
    end_points = start_points + flow[y, x]
    finite = np.isfinite(start_points).all(axis=-1) & np.isfinite(end_points).all(axis=-1)
    segments = np.stack([start_points[finite], end_points[finite]], axis=1).astype(np.float32)
    segment_colors = scene_flow_to_rgb(flow, motion_mask)[y[finite], x[finite]].astype(np.uint8)
    return segments, segment_colors


def render_world_point_view(
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    axis_limits: tuple[np.ndarray, np.ndarray],
    motion_segments: Optional[np.ndarray] = None,
    motion_colors: Optional[np.ndarray] = None,
    elev: float = GEOMETRY_VIEW_ELEV,
    azim: float = GEOMETRY_VIEW_AZIM,
) -> np.ndarray:
    fig = plt.figure(figsize=(4.6, 4.6), dpi=120)
    axis = fig.add_subplot(111, projection="3d")
    min_xyz, max_xyz = axis_limits
    point_colors = colors.astype(np.float32) / 255.0 if colors.size > 0 else colors

    if points.size == 0:
        axis.text2D(0.5, 0.5, "No valid points", transform=axis.transAxes, ha="center", va="center", fontsize=11)
    else:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=point_colors,
            s=1.2,
            alpha=0.95,
            linewidths=0.0,
            depthshade=False,
        )

    if motion_segments is not None and motion_segments.size > 0:
        segment_rgb = motion_colors.astype(np.float32) / 255.0 if motion_colors is not None else None
        axis.add_collection3d(
            Line3DCollection(
                motion_segments,
                colors=segment_rgb,
                linewidths=1.2,
                alpha=0.95,
            )
        )

    axis.set_xlim(float(min_xyz[0]), float(max_xyz[0]))
    axis.set_ylim(float(min_xyz[1]), float(max_xyz[1]))
    axis.set_zlim(float(min_xyz[2]), float(max_xyz[2]))
    axis.set_box_aspect(np.maximum(max_xyz - min_xyz, 1e-3))
    axis.view_init(elev=float(elev), azim=float(azim))
    axis.set_axis_off()
    fig.tight_layout(pad=0.05)
    image = figure_to_rgb(fig)
    return add_tile_label(image, title) if title else image


def _view_rotation_matrix(elev: float, azim: float) -> np.ndarray:
    elev_rad = np.deg2rad(float(elev))
    azim_rad = np.deg2rad(float(azim))
    cos_azim, sin_azim = np.cos(azim_rad), np.sin(azim_rad)
    cos_elev, sin_elev = np.cos(elev_rad), np.sin(elev_rad)
    rot_z = np.array(
        [
            [cos_azim, -sin_azim, 0.0],
            [sin_azim, cos_azim, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_elev, -sin_elev],
            [0.0, sin_elev, cos_elev],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_z


def _project_points_to_canvas(
    points: np.ndarray,
    axis_limits: tuple[np.ndarray, np.ndarray],
    elev: float,
    azim: float,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    min_xyz, max_xyz = axis_limits
    center = (min_xyz + max_xyz).astype(np.float32) * 0.5
    radius = max(float(np.max(max_xyz - min_xyz)) * 0.5, 1e-6)
    rotated = (np.asarray(points, dtype=np.float32) - center) @ _view_rotation_matrix(elev, azim).T
    normalized = rotated / radius
    margin = 0.08
    scale = (float(image_size) - 1.0) * (0.5 - margin)
    xs = np.rint((normalized[:, 0] * scale) + (float(image_size) - 1.0) * 0.5).astype(np.int32)
    ys = np.rint((-normalized[:, 1] * scale) + (float(image_size) - 1.0) * 0.5).astype(np.int32)
    depth = normalized[:, 2].astype(np.float32)
    valid = (xs >= 0) & (xs < image_size) & (ys >= 0) & (ys < image_size) & np.isfinite(depth)
    return xs[valid], ys[valid], depth[valid], np.nonzero(valid)[0].astype(np.int64)


def render_world_point_view_numpy(
    points: np.ndarray,
    colors: np.ndarray,
    title: str,
    axis_limits: tuple[np.ndarray, np.ndarray],
    motion_segments: Optional[np.ndarray] = None,
    motion_colors: Optional[np.ndarray] = None,
    elev: float = GEOMETRY_VIEW_ELEV,
    azim: float = GEOMETRY_VIEW_AZIM,
    image_size: int = GEOMETRY_OVERLAY_IMAGE_SIZE,
) -> np.ndarray:
    _ = title
    canvas = np.full((int(image_size), int(image_size), 3), 255, dtype=np.uint8)
    depth_buffer = np.full((int(image_size), int(image_size)), -np.inf, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    xs, ys, depth, point_indices = _project_points_to_canvas(points, axis_limits, elev, azim, int(image_size))
    if xs.size > 0:
        projected_colors = colors[point_indices]
        flat_pixels = ys.astype(np.int64) * int(image_size) + xs.astype(np.int64)
        order = np.lexsort((depth, flat_pixels))
        sorted_pixels = flat_pixels[order]
        keep = np.r_[sorted_pixels[1:] != sorted_pixels[:-1], True]
        visible = order[keep]
        visible_xs = xs[visible]
        visible_ys = ys[visible]
        visible_colors = projected_colors[visible]
        for dy in (-1, 0, 1):
            draw_y = visible_ys + dy
            valid_y = (draw_y >= 0) & (draw_y < int(image_size))
            for dx in (-1, 0, 1):
                draw_x = visible_xs + dx
                valid = valid_y & (draw_x >= 0) & (draw_x < int(image_size))
                canvas[draw_y[valid], draw_x[valid]] = visible_colors[valid]

    if motion_segments is not None and np.asarray(motion_segments).size > 0:
        segments = np.asarray(motion_segments, dtype=np.float32)
        segment_colors = (
            np.asarray(motion_colors, dtype=np.uint8)
            if motion_colors is not None and np.asarray(motion_colors).size > 0
            else np.full((segments.shape[0], 3), 255, dtype=np.uint8)
        )
        for segment, color in zip(segments, segment_colors):
            endpoints = segment.reshape(2, 3)
            seg_x, seg_y, _seg_depth, _seg_indices = _project_points_to_canvas(
                endpoints,
                axis_limits,
                elev,
                azim,
                int(image_size),
            )
            if seg_x.shape[0] == 2:
                cv2.line(
                    canvas,
                    (int(seg_x[0]), int(seg_y[0])),
                    (int(seg_x[1]), int(seg_y[1])),
                    tuple(int(channel) for channel in color.tolist()),
                    1,
                    cv2.LINE_AA,
                )
    return canvas


def build_labeled_thumbnail(rgb: np.ndarray, target_height: int, target_width: int, label: str = "RGB Frame") -> np.ndarray:
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    rgb_height, rgb_width = rgb.shape[:2]
    scale = min(target_width / max(rgb_width, 1), target_height / max(rgb_height, 1))
    resized = cv2.resize(
        rgb,
        (max(int(rgb_width * scale), 1), max(int(rgb_height * scale), 1)),
        interpolation=cv2.INTER_AREA,
    )
    top = (target_height - resized.shape[0]) // 2
    left = (target_width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return add_tile_label(canvas, label)


def build_geometry_motion_panel(
    rgb: np.ndarray,
    point_map_world: np.ndarray,
    pred_flow_world: np.ndarray,
    gt_flow_world: np.ndarray,
    deform_mask: np.ndarray,
    frame_idx: int,
    axis_limits: Optional[tuple[np.ndarray, np.ndarray]] = None,
    vis_scale: float = 1.0,
) -> np.ndarray:
    _ = gt_flow_world
    base_points_pred, base_colors_pred = flatten_point_map_for_projection(
        point_map_world=point_map_world,
        rgb=rgb,
        deform_mask=deform_mask,
        use_deformed_points=False,
        max_points=GEOMETRY_MAX_POINTS,
    )
    deform_points_pred, deform_colors_pred = flatten_point_map_for_projection(
        point_map_world=point_map_world,
        rgb=rgb,
        deform_mask=deform_mask,
        use_deformed_points=True,
        flow_world=pred_flow_world,
        darken=GEOMETRY_DEFORM_DARKEN,
        max_points=GEOMETRY_MAX_POINTS,
    )
    motion_points_pred, motion_colors_pred = flatten_point_map_for_projection(
        point_map_world=point_map_world,
        rgb=rgb,
        deform_mask=deform_mask,
        use_deformed_points=False,
        darken=GEOMETRY_MOTION_BASE_DARKEN,
        max_points=GEOMETRY_MAX_POINTS,
    )
    if axis_limits is None:
        axis_limits = compute_axis_limits(base_points_pred, deform_points_pred)
    scene_scale = float(np.linalg.norm(axis_limits[1] - axis_limits[0]))
    motion_segments, motion_segment_colors = sample_motion_segments(
        point_map_world=point_map_world,
        flow_world=pred_flow_world,
        deform_mask=deform_mask,
        scene_scale=scene_scale,
    )

    pred_static_view = render_world_point_view(
        base_points_pred,
        base_colors_pred,
        f"Point Map X_t (t={frame_idx})",
        axis_limits,
    )
    pred_motion_view = render_world_point_view(
        motion_points_pred,
        motion_colors_pred,
        "Motion Overlay",
        axis_limits,
        motion_segments=motion_segments,
        motion_colors=motion_segment_colors,
    )
    pred_deformed_view = render_world_point_view(
        deform_points_pred,
        deform_colors_pred,
        "Deformed X_t + V_t",
        axis_limits,
    )
    rgb_thumb = build_labeled_thumbnail(
        rgb,
        target_height=pred_static_view.shape[0],
        target_width=pred_static_view.shape[1],
        label="RGB I_t",
    )
    panel = np.concatenate([rgb_thumb, pred_static_view, pred_motion_view, pred_deformed_view], axis=1)
    return maybe_resize(panel, vis_scale)


def build_geometry_motion_overlay_frame(
    *,
    rgb: np.ndarray,
    point_map_world: np.ndarray,
    flow_world: np.ndarray,
    geometry_mask: Optional[np.ndarray],
    motion_mask: np.ndarray,
    frame_idx: int,
    axis_limits: Optional[tuple[np.ndarray, np.ndarray]] = None,
    vis_scale: float = 1.0,
    label: str = "Motion Overlay",
    views: tuple[tuple[str, float, float], ...] = GEOMETRY_MOTION_OVERLAY_VIEWS,
    renderer: str = "numpy",
) -> np.ndarray:
    tiles = [
        image
        for _view_name, image in build_geometry_motion_overlay_view_images(
            rgb=rgb,
            point_map_world=point_map_world,
            flow_world=flow_world,
            geometry_mask=geometry_mask,
            motion_mask=motion_mask,
            frame_idx=frame_idx,
            axis_limits=axis_limits,
            label=label,
            views=views,
            add_labels=True,
            renderer=renderer,
        )
    ]
    rows = [np.concatenate(tiles[index:index + 3], axis=1) for index in range(0, len(tiles), 3)]
    return maybe_resize(np.concatenate(rows, axis=0), vis_scale)


def build_geometry_motion_overlay_view_images(
    *,
    rgb: np.ndarray,
    point_map_world: np.ndarray,
    flow_world: np.ndarray,
    geometry_mask: Optional[np.ndarray],
    motion_mask: np.ndarray,
    frame_idx: int,
    axis_limits: Optional[tuple[np.ndarray, np.ndarray]] = None,
    label: str = "Motion Overlay",
    flow_label: str = "",
    views: tuple[tuple[str, float, float], ...] = GEOMETRY_MOTION_OVERLAY_VIEWS,
    add_labels: bool = False,
    renderer: str = "numpy",
) -> list[tuple[str, np.ndarray]]:
    motion_points, motion_colors = flatten_point_map_for_projection(
        point_map_world=point_map_world,
        rgb=rgb,
        deform_mask=geometry_mask,
        use_deformed_points=False,
        darken=GEOMETRY_MOTION_BASE_DARKEN,
        max_points=GEOMETRY_MAX_POINTS,
    )
    if axis_limits is None:
        axis_limits = compute_axis_limits(motion_points)
    scene_scale = float(np.linalg.norm(axis_limits[1] - axis_limits[0]))
    motion_segments, motion_segment_colors = sample_motion_segments(
        point_map_world=point_map_world,
        flow_world=flow_world,
        deform_mask=motion_mask,
        scene_scale=scene_scale,
    )
    images: list[tuple[str, np.ndarray]] = []
    render_fn = render_world_point_view_numpy if renderer == "numpy" else render_world_point_view
    for view_name, elev, azim in views:
        title = f"{label} {view_name} t={frame_idx}" if add_labels else ""
        image = render_fn(
            motion_points,
            motion_colors,
            title,
            axis_limits,
            motion_segments=motion_segments,
            motion_colors=motion_segment_colors,
            elev=elev,
            azim=azim,
        )
        images.append((view_name, image))
    return images


def build_scene_flow_panel(
    rgb: np.ndarray,
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    deform_mask: np.ndarray,
    frame_idx: int,
    quiver_step: int,
    vis_scale: float,
) -> np.ndarray:
    mask_vis = np.repeat((deform_mask > 0.5)[..., None], 3, axis=-1).astype(np.uint8) * 255
    epe = np.linalg.norm(pred_flow - gt_flow, axis=-1)

    tiles = [
        add_tile_label(rgb, f"RGB t={frame_idx}"),
        add_tile_label(scene_flow_to_rgb(pred_flow, deform_mask), "Pred SceneFlow"),
        add_tile_label(scene_flow_to_rgb(gt_flow, deform_mask), "GT SceneFlow"),
        add_tile_label(scalar_to_rgb(epe, deform_mask, cmap_name="inferno"), "EPE"),
        add_tile_label(render_scene_flow_quiver(rgb, pred_flow, deform_mask, step=quiver_step), "Pred Quiver XY"),
        add_tile_label(mask_vis, "Deform Mask"),
    ]
    panel = np.concatenate(tiles, axis=1)
    return maybe_resize(panel, vis_scale)


def build_error_overlay(
    rgb: np.ndarray,
    error_rgb: np.ndarray,
    deform_mask: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    error_rgb = np.asarray(error_rgb, dtype=np.float32)
    valid = (np.asarray(deform_mask) > 0.5)[..., None].astype(np.float32)
    overlay = rgb * (1.0 - alpha * valid) + error_rgb * (alpha * valid)
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def summarize_epe(epe: np.ndarray, deform_mask: np.ndarray) -> Dict[str, float]:
    epe = np.asarray(epe, dtype=np.float32)
    valid = (np.asarray(deform_mask) > 0.5) & np.isfinite(epe)
    if not np.any(valid):
        return {
            "mean_epe": 0.0,
            "median_epe": 0.0,
            "p95_epe": 0.0,
            "max_epe": 0.0,
            "num_valid_pixels": 0,
        }

    values = epe[valid]
    return {
        "mean_epe": float(values.mean()),
        "median_epe": float(np.median(values)),
        "p95_epe": float(np.percentile(values, 95.0)),
        "max_epe": float(values.max()),
        "num_valid_pixels": int(valid.sum()),
    }


def extract_batch_item(batch_value: Any, index: int) -> Any:
    if isinstance(batch_value, torch.Tensor):
        return batch_value[index]
    if isinstance(batch_value, np.ndarray):
        return batch_value[index]
    if isinstance(batch_value, (list, tuple)):
        return batch_value[index]
    if isinstance(batch_value, str):
        return batch_value
    return batch_value


def make_sample_id(batch: Mapping[str, Any], batch_index: int) -> str:
    clip_uid = batch.get("clip_uid")
    if clip_uid is not None:
        return str(extract_batch_item(clip_uid, batch_index)).replace("/", "_")

    scene_name = str(extract_batch_item(batch["scene_name"], batch_index))
    start_idx = int(extract_batch_item(batch["start_idx"], batch_index).item())
    end_idx = int(extract_batch_item(batch["end_idx"], batch_index).item())
    safe_scene = scene_name.replace("/", "_")
    return f"{safe_scene}_{start_idx:05d}_{end_idx:05d}"


def save_sample_outputs(
    output_dir: Path,
    sample_id: str,
    rgbs: np.ndarray,
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    deform_mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    sequence_length: int,
    quiver_step: int,
    vis_scale: float,
    save_gif: bool,
    save_npz: bool,
    save_geometry_motion_video: bool = False,
    save_scene_flow_ply: bool = False,
    scene_flow_ply_every_n: int = 1,
    scene_flow_ply_max_points: int = SCENE_FLOW_PLY_MAX_POINTS,
    point_map_world: Optional[np.ndarray] = None,
    pred_flow_world: Optional[np.ndarray] = None,
    gt_flow_world: Optional[np.ndarray] = None,
    scene_flow_scale: Optional[float] = None,
    flow_p95_scale: Optional[float] = None,
    dataset_name: Optional[str] = None,
    source_split: Optional[str] = None,
    clip_uid: Optional[str] = None,
    frame_ids: Optional[np.ndarray] = None,
    eval_space: Optional[str] = None,
    target_space: Optional[str] = None,
) -> None:
    sample_vis_dir = output_dir / "visualizations" / sample_id
    sample_vis_dir.mkdir(parents=True, exist_ok=True)
    error_stats: Dict[str, Any] = {"per_frame": []}

    geometry_axis_limits = None
    if save_geometry_motion_video:
        if point_map_world is None or pred_flow_world is None or gt_flow_world is None:
            raise ValueError("geometry-motion video requires point_map_world, pred_flow_world and gt_flow_world")
        geometry_axis_limits = compute_sequence_geometry_axis_limits(
            point_map_world=point_map_world[:sequence_length],
            pred_flow_world=pred_flow_world[:sequence_length],
            deform_mask=deform_mask[:sequence_length],
        )

    preview_frames: List[np.ndarray] = []
    video_frames: List[np.ndarray] = []
    for frame_idx in range(sequence_length):
        error_map = np.linalg.norm(pred_flow[frame_idx] - gt_flow[frame_idx], axis=-1).astype(np.float32)
        error_heatmap = scalar_to_rgb(error_map, deform_mask[frame_idx], cmap_name="inferno")
        error_overlay = build_error_overlay(rgbs[frame_idx], error_heatmap, deform_mask[frame_idx])
        imageio.imwrite(sample_vis_dir / f"frame_{frame_idx:04d}_error_epe.png", error_heatmap)
        imageio.imwrite(sample_vis_dir / f"frame_{frame_idx:04d}_error_overlay.png", error_overlay)
        error_stats["per_frame"].append({"frame_idx": frame_idx, **summarize_epe(error_map, deform_mask[frame_idx])})

        if save_geometry_motion_video:
            panel = build_geometry_motion_panel(
                rgb=rgbs[frame_idx],
                point_map_world=point_map_world[frame_idx],
                pred_flow_world=pred_flow_world[frame_idx],
                gt_flow_world=gt_flow_world[frame_idx],
                deform_mask=deform_mask[frame_idx],
                frame_idx=frame_idx,
                axis_limits=geometry_axis_limits,
                vis_scale=vis_scale,
            )
            imageio.imwrite(sample_vis_dir / f"geometry_motion_{frame_idx:04d}.png", panel)
            video_frames.append(panel)
            if save_gif:
                preview_frames.append(panel)
        else:
            panel = build_scene_flow_panel(
                rgb=rgbs[frame_idx],
                pred_flow=pred_flow[frame_idx],
                gt_flow=gt_flow[frame_idx],
                deform_mask=deform_mask[frame_idx],
                frame_idx=frame_idx,
                quiver_step=quiver_step,
                vis_scale=vis_scale,
            )
            imageio.imwrite(sample_vis_dir / f"frame_{frame_idx:04d}.png", panel)
            if save_gif:
                preview_frames.append(panel)

    error_stats.update(summarize_epe(np.linalg.norm(pred_flow[:sequence_length] - gt_flow[:sequence_length], axis=-1), deform_mask[:sequence_length]))
    with open(sample_vis_dir / "error_stats.json", "w", encoding="utf-8") as file:
        json.dump(error_stats, file, ensure_ascii=False, indent=2)

    if save_geometry_motion_video and video_frames:
        with imageio.get_writer(sample_vis_dir / "geometry_motion.mp4", fps=6) as writer:
            for frame in video_frames:
                writer.append_data(frame)

    if save_gif and preview_frames and not save_geometry_motion_video:
        with imageio.get_writer(sample_vis_dir / "scene_flow.mp4", fps=6) as writer:
            for frame in preview_frames:
                writer.append_data(frame)

    if save_scene_flow_ply:
        if point_map_world is None:
            raise ValueError("scene-flow PLY export requires point_map_world")
        pred_ply_flow = pred_flow_world if pred_flow_world is not None else pred_flow
        gt_ply_flow = gt_flow_world if gt_flow_world is not None else gt_flow
        write_scene_flow_ply(
            sample_vis_dir / "pred_scene_flow.ply",
            point_map_world=point_map_world[:sequence_length],
            flow_world=pred_ply_flow[:sequence_length],
            rgbs=rgbs[:sequence_length],
            deform_mask=deform_mask[:sequence_length],
            frame_stride=int(scene_flow_ply_every_n),
            max_points=int(scene_flow_ply_max_points),
        )
        write_scene_flow_ply(
            sample_vis_dir / "gt_scene_flow.ply",
            point_map_world=point_map_world[:sequence_length],
            flow_world=gt_ply_flow[:sequence_length],
            rgbs=rgbs[:sequence_length],
            deform_mask=deform_mask[:sequence_length],
            frame_stride=int(scene_flow_ply_every_n),
            max_points=int(scene_flow_ply_max_points),
        )

    if save_npz:
        payload = {
            "rgbs": rgbs[:sequence_length],
            "pred_flow": pred_flow[:sequence_length],
            "gt_flow": gt_flow[:sequence_length],
            "deform_mask": deform_mask[:sequence_length],
            "intrinsics": intrinsics[:sequence_length],
            "extrinsics": extrinsics[:sequence_length],
            "sequence_length": np.array(sequence_length, dtype=np.int32),
        }
        if scene_flow_scale is not None:
            payload["scene_flow_scale"] = np.array(scene_flow_scale, dtype=np.float32)
        if flow_p95_scale is not None:
            payload["flow_p95_scale"] = np.array(flow_p95_scale, dtype=np.float32)
        if dataset_name is not None:
            payload["dataset_name"] = np.array(dataset_name)
        if source_split is not None:
            payload["source_split"] = np.array(source_split)
        if clip_uid is not None:
            payload["clip_uid"] = np.array(clip_uid)
        if frame_ids is not None:
            payload["frame_ids"] = np.asarray(frame_ids[:sequence_length], dtype=np.int64)
        if eval_space is not None:
            payload["eval_space"] = np.array(eval_space)
        if target_space is not None:
            payload["target_space"] = np.array(target_space)
        if point_map_world is not None:
            payload["point_map_world"] = point_map_world[:sequence_length]
            payload["point_map"] = point_map_world[:sequence_length]
            payload["valid_mask"] = compute_point_map_valid_mask(point_map_world[:sequence_length]).astype(np.bool_)
        if pred_flow_world is not None:
            payload["pred_flow_world"] = pred_flow_world[:sequence_length]
            payload["scene_flow"] = pred_flow_world[:sequence_length]
        if gt_flow_world is not None:
            payload["gt_flow_world"] = gt_flow_world[:sequence_length]
        np.savez_compressed(output_dir / "predictions" / f"{sample_id}.npz", **payload)


def _read_npz_string(payload: Mapping[str, np.ndarray], key: str) -> Optional[str]:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.shape == ():
        return str(value.item())
    return str(value)


def _load_required_npz_array(payload: Mapping[str, np.ndarray], key: str, path: Path) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"{path} is missing required key '{key}'")
    return np.asarray(payload[key])


def _resolve_sequence_length(payload: Mapping[str, np.ndarray], rgbs: np.ndarray, path: Path) -> int:
    if "sequence_length" not in payload:
        return int(rgbs.shape[0])
    sequence_length = int(np.asarray(payload["sequence_length"]).item())
    if sequence_length < 1:
        raise ValueError(f"{path} has invalid sequence_length={sequence_length}")
    return min(sequence_length, int(rgbs.shape[0]))


def _load_geometry_mask(payload: Mapping[str, np.ndarray], point_map_world: np.ndarray) -> np.ndarray:
    if "valid_mask" in payload:
        return np.asarray(payload["valid_mask"], dtype=bool)
    return compute_point_map_valid_mask(point_map_world).astype(np.bool_)


def _flow_arrays_are_metric(payload: Mapping[str, np.ndarray]) -> bool:
    target_space = _read_npz_string(payload, "target_space")
    eval_space = _read_npz_string(payload, "eval_space")
    if target_space == "relative":
        return eval_space == "metric"
    return True


def _resolve_metric_flow_array(
    payload: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> tuple[np.ndarray, bool, Optional[float]]:
    flow = _load_required_npz_array(payload, key, path).astype(np.float32)
    if _flow_arrays_are_metric(payload):
        return flow, False, None

    target_space = _read_npz_string(payload, "target_space")
    if target_space != "relative":
        return flow, False, None
    if "scene_flow_scale" not in payload:
        raise ValueError(f"{path} contains relative flow but is missing scene_flow_scale")
    scale = float(np.asarray(payload["scene_flow_scale"]).item())
    return flow * scale, True, scale


def _compute_overlay_axis_limits(
    point_map_world: np.ndarray,
    flow_world: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    geometry_mask: Optional[np.ndarray],
    motion_mask: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    flow_worlds = list(flow_world) if isinstance(flow_world, (list, tuple)) else [flow_world]
    point_sets: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for frame_idx in range(sequence_length):
        base_points, _ = flatten_point_map_for_projection(
            point_map_world=point_map_world[frame_idx],
            rgb=np.zeros_like(point_map_world[frame_idx], dtype=np.uint8),
            deform_mask=geometry_mask[frame_idx] if geometry_mask is not None else None,
            use_deformed_points=False,
            max_points=25000,
            rng=rng,
        )
        if base_points.size > 0:
            point_sets.append(base_points)
        for flow in flow_worlds:
            deformed_points, _ = flatten_point_map_for_projection(
                point_map_world=point_map_world[frame_idx],
                rgb=np.zeros_like(point_map_world[frame_idx], dtype=np.uint8),
                deform_mask=motion_mask[frame_idx],
                use_deformed_points=True,
                flow_world=flow[frame_idx],
                max_points=25000,
                rng=rng,
            )
            if deformed_points.size > 0:
                point_sets.append(deformed_points)

    return compute_axis_limits(*point_sets)


def _safe_view_name(view_name: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in view_name)
    return "_".join(part for part in safe.split("_") if part)


def _write_geometry_motion_overlay_outputs(
    output_path: Path,
    *,
    sample_output_dir: Path,
    rgbs: np.ndarray,
    point_map_world: np.ndarray,
    flow_world: np.ndarray,
    geometry_mask: Optional[np.ndarray],
    motion_mask: np.ndarray,
    sequence_length: int,
    axis_limits: tuple[np.ndarray, np.ndarray],
    label: str,
    flow_label: str,
    fps: int,
    vis_scale: float,
    views: tuple[tuple[str, float, float], ...],
    renderer: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = sample_output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    saved_images = 0
    with imageio.get_writer(output_path, fps=int(fps)) as writer:
        for frame_idx in range(sequence_length):
            view_images = build_geometry_motion_overlay_view_images(
                rgb=rgbs[frame_idx],
                point_map_world=point_map_world[frame_idx],
                flow_world=flow_world[frame_idx],
                geometry_mask=geometry_mask[frame_idx] if geometry_mask is not None else None,
                motion_mask=motion_mask[frame_idx],
                frame_idx=frame_idx,
                axis_limits=axis_limits,
                label=label,
                flow_label=flow_label,
                views=views,
                add_labels=False,
                renderer=renderer,
            )
            if frame_idx < sequence_length - 1:
                for view_name, image in view_images:
                    imageio.imwrite(images_dir / f"{flow_label}_t{frame_idx:04d}_{_safe_view_name(view_name)}.png", image)
                    saved_images += 1
            tiles = [add_tile_label(image, f"{label} {view_name} t={frame_idx}") for view_name, image in view_images]
            rows = [np.concatenate(tiles[index:index + 3], axis=1) for index in range(0, len(tiles), 3)]
            frame = maybe_resize(np.concatenate(rows, axis=0), vis_scale)
            writer.append_data(frame)
    return saved_images


def export_geometry_motion_overlays_from_npz(
    prediction_path: Path,
    output_dir: Path,
    *,
    fps: int = 6,
    vis_scale: float = 1.0,
    views: tuple[tuple[str, float, float], ...] = GEOMETRY_MOTION_OVERLAY_VIEWS,
    renderer: str = "numpy",
) -> int:
    with np.load(prediction_path, allow_pickle=False) as payload:
        rgbs = _load_required_npz_array(payload, "rgbs", prediction_path).astype(np.uint8)
        point_map_world = _load_required_npz_array(payload, "point_map_world", prediction_path).astype(np.float32)
        deform_mask = _load_required_npz_array(payload, "deform_mask", prediction_path).astype(np.float32)
        geometry_mask = _load_geometry_mask(payload, point_map_world)
        pred_flow_world, pred_scaled_to_metric, pred_scene_flow_scale = _resolve_metric_flow_array(
            payload,
            "pred_flow_world",
            prediction_path,
        )
        gt_flow_world, gt_scaled_to_metric, gt_scene_flow_scale = _resolve_metric_flow_array(
            payload,
            "gt_flow_world",
            prediction_path,
        )
        sequence_length = _resolve_sequence_length(payload, rgbs, prediction_path)
        sample_id = prediction_path.stem
        clip_uid = _read_npz_string(payload, "clip_uid")
        target_space = _read_npz_string(payload, "target_space")
        eval_space = _read_npz_string(payload, "eval_space")

    expected_shape = point_map_world.shape
    for name, value in {
        "pred_flow_world": pred_flow_world,
        "gt_flow_world": gt_flow_world,
    }.items():
        if value.shape != expected_shape:
            raise ValueError(f"{prediction_path} key '{name}' shape {value.shape} does not match point_map_world {expected_shape}")
    if rgbs.shape[:3] != point_map_world.shape[:3]:
        raise ValueError(f"{prediction_path} rgbs shape {rgbs.shape} is incompatible with point_map_world {point_map_world.shape}")
    if deform_mask.shape != point_map_world.shape[:3]:
        raise ValueError(f"{prediction_path} deform_mask shape {deform_mask.shape} is incompatible with point_map_world {point_map_world.shape}")
    if geometry_mask.shape != point_map_world.shape[:3]:
        raise ValueError(f"{prediction_path} valid_mask shape {geometry_mask.shape} is incompatible with point_map_world {point_map_world.shape}")

    sample_output_dir = output_dir / sample_id
    axis_limits = _compute_overlay_axis_limits(
        point_map_world,
        [pred_flow_world, gt_flow_world],
        geometry_mask,
        deform_mask,
        sequence_length,
    )
    pred_images = _write_geometry_motion_overlay_outputs(
        sample_output_dir / "geometry_motion_overlay_pred.mp4",
        sample_output_dir=sample_output_dir,
        rgbs=rgbs,
        point_map_world=point_map_world,
        flow_world=pred_flow_world,
        geometry_mask=geometry_mask,
        motion_mask=deform_mask,
        sequence_length=sequence_length,
        axis_limits=axis_limits,
        label="Prediction Overlay",
        flow_label="pred",
        fps=fps,
        vis_scale=vis_scale,
        views=views,
        renderer=renderer,
    )
    gt_images = _write_geometry_motion_overlay_outputs(
        sample_output_dir / "geometry_motion_overlay_gt.mp4",
        sample_output_dir=sample_output_dir,
        rgbs=rgbs,
        point_map_world=point_map_world,
        flow_world=gt_flow_world,
        geometry_mask=geometry_mask,
        motion_mask=deform_mask,
        sequence_length=sequence_length,
        axis_limits=axis_limits,
        label="GT Overlay",
        flow_label="gt",
        fps=fps,
        vis_scale=vis_scale,
        views=views,
        renderer=renderer,
    )
    metadata = {
        "sample_id": sample_id,
        "source_npz": str(prediction_path),
        "clip_uid": clip_uid,
        "sequence_length": int(sequence_length),
        "source_target_space": target_space,
        "source_eval_space": eval_space,
        "output_flow_space": "metric",
        "geometry_mask_source": "valid_mask" if geometry_mask is not None else "point_map_finite",
        "motion_mask_source": "deform_mask",
        "num_views": len(views),
        "renderer": renderer,
        "saved_view_images": {
            "prediction": pred_images,
            "gt": gt_images,
            "directory": "images",
            "last_frame_skipped": True,
        },
        "views": [{"name": name, "elev": elev, "azim": azim} for name, elev, azim in views],
        "pred_scaled_to_metric": bool(pred_scaled_to_metric),
        "gt_scaled_to_metric": bool(gt_scaled_to_metric),
        "scene_flow_scale": pred_scene_flow_scale if pred_scene_flow_scale is not None else gt_scene_flow_scale,
        "outputs": {
            "prediction": "geometry_motion_overlay_pred.mp4",
            "gt": "geometry_motion_overlay_gt.mp4",
        },
    }
    with open(sample_output_dir / "geometry_motion_overlay_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return 2


def export_geometry_motion_overlays_from_eval_dir(
    eval_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    fps: int = 6,
    vis_scale: float = 1.0,
    max_samples: Optional[int] = None,
    renderer: str = "numpy",
) -> int:
    eval_dir = Path(eval_dir).expanduser().resolve()
    predictions_dir = eval_dir / "predictions"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Expected predictions directory at {predictions_dir}")
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else eval_dir / "geometry_motion_overlays"
    prediction_paths = sorted(predictions_dir.glob("*.npz"))
    if max_samples is not None:
        prediction_paths = prediction_paths[: max(int(max_samples), 0)]
    if not prediction_paths:
        raise FileNotFoundError(f"No .npz prediction files found in {predictions_dir}")

    exported_videos = 0
    for prediction_path in tqdm(prediction_paths, desc="Exporting overlay videos", dynamic_ncols=True):
        exported_videos += export_geometry_motion_overlays_from_npz(
            prediction_path=prediction_path,
            output_dir=resolved_output_dir,
            fps=fps,
            vis_scale=vis_scale,
            renderer=renderer,
        )
    return exported_videos


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(
    da3_interface: DA3StreamingInterface,
    dataloader: DataLoader,
    criterion: SceneFlowLoss,
    device: torch.device,
    output_dir: Path,
    decoder_chunk_size: int,
    max_samples: Optional[int],
    save_vis_every_n: int,
    quiver_step: int,
    vis_scale: float,
    save_gif: bool,
    save_npz: bool,
    save_geometry_motion_video: bool,
    metrics_on_pred_point_map: bool,
    target_space: str,
    eval_space: str,
    ref_view_strategy: str,
    save_scene_flow_ply: bool = False,
    scene_flow_ply_every_n: int = 1,
    scene_flow_ply_max_points: int = SCENE_FLOW_PLY_MAX_POINTS,
) -> Dict[str, float]:
    if target_space not in {"metric", "relative"}:
        raise ValueError(f"Unsupported target_space: {target_space}")
    resolved_eval_space = resolve_eval_space(
        target_space=target_space,
        eval_space=eval_space,
        metrics_on_pred_point_map=metrics_on_pred_point_map,
    )
    if metrics_on_pred_point_map:
        LOGGER.warning(
            "--metrics-on-pred-point-map is deprecated and now behaves like metric-space evaluation with geometry decode"
        )
    if target_space == "relative" and resolved_eval_space == "native" and save_geometry_motion_video:
        raise ValueError("relative scene flow cannot drive geometry-motion video export in native eval space")
    if target_space == "relative" and resolved_eval_space == "native" and save_scene_flow_ply:
        raise ValueError("relative scene flow cannot drive scene-flow PLY export in native eval space")

    da3_interface.eval()
    all_metrics: Dict[str, List[float]] = {}
    per_sample_path = output_dir / "metrics_per_sample.jsonl"
    with open(per_sample_path, "w", encoding="utf-8"):
        pass

    if save_npz:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    if save_vis_every_n > 0 or save_scene_flow_ply:
        (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    evaluated_samples = 0
    progress = tqdm(dataloader, desc="Evaluating", dynamic_ncols=True)

    for batch in progress:
        images = batch["rgbs"].permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        intrinsics = batch["intrinsics"].to(device, non_blocking=True)
        extrinsics = batch["extrinsics"].to(device, non_blocking=True)
        target_flow = batch["scene_flow"].permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        deform_mask = batch["deform_mask"].to(device, non_blocking=True)
        height, width = images.shape[-2:]

        features = da3_interface.extract_multilayer_features(
            images,
            normalize=True,
            ref_view_strategy=ref_view_strategy,
        )
        head_dtype = next(get_scene_flow_head(da3_interface.da3_model).parameters()).dtype
        features = features.to(dtype=head_dtype)
        pred_flow = da3_interface.decode_features_to_scene_flow(
            features=features,
            image_size=(height, width),
            extrinsics=extrinsics,
            use_gradient=False,
            chunk_size=decoder_chunk_size,
            output_basis="frame0",
        )

        batch_size = images.shape[0]
        prospective_eval_indices = list(
            range(
                evaluated_samples,
                min(
                    evaluated_samples + batch_size,
                    max_samples if max_samples is not None else evaluated_samples + batch_size,
                ),
            )
        )
        any_geometry_motion_vis = save_geometry_motion_video and any(
            save_vis_every_n > 0 and sample_idx % save_vis_every_n == 0
            for sample_idx in prospective_eval_indices
        )
        any_scene_flow_ply = save_scene_flow_ply and any(
            scene_flow_ply_every_n > 0 and sample_idx % scene_flow_ply_every_n == 0
            for sample_idx in prospective_eval_indices
        )

        point_maps_world = None
        need_pred_point_map = metrics_on_pred_point_map or any_geometry_motion_vis or save_npz or any_scene_flow_ply
        if need_pred_point_map:
            geometry_output = da3_interface.decode_features_to_geometry(
                features=features,
                image_size=(height, width),
                use_gradient=False,
                chunk_size=decoder_chunk_size,
            )
            pred_depth = geometry_output["depth"]
            pred_ray = geometry_output["ray"]
            if pred_depth is None or pred_ray is None:
                raise RuntimeError(
                    "Pred point-map evaluation/visualization requires DA3 depth and ray outputs, "
                    "but decode_features_to_geometry returned incomplete geometry"
                )
            raw_pred_depth_shape = tuple(pred_depth.shape)
            raw_pred_ray_shape = tuple(pred_ray.shape)
            pred_depth = standardize_depth_prediction(pred_depth)
            pred_ray = standardize_ray_prediction(pred_ray, target_size=pred_depth.shape[-2:])
            if evaluated_samples == 0:
                LOGGER.info(
                    "Pred geometry shapes: raw_depth=%s raw_ray=%s aligned_depth=%s aligned_ray=%s",
                    raw_pred_depth_shape,
                    raw_pred_ray_shape,
                    tuple(pred_depth.shape),
                    tuple(pred_ray.shape),
                )

            gt_depth = None
            try:
                gt_depth = load_gt_depth_batch(
                    batch=batch,
                    data_root=Path(getattr(dataloader.dataset, "data_root")),
                    split=str(getattr(dataloader.dataset, "split")),
                    target_height=height,
                    target_width=width,
                    target_length=pred_depth.shape[1],
                ).to(device=device, dtype=pred_depth.dtype)
            except Exception as exc:
                LOGGER.warning("Failed to load GT depth for pred-point-map alignment, falling back to raw pred depth: %s", exc)

            if gt_depth is not None:
                gt_depth = gt_depth.unsqueeze(2)
                gt_depth_valid = (gt_depth > 1e-6) & torch.isfinite(gt_depth)
                pred_depth = align_depth_least_squares(
                    pred=pred_depth,
                    target=gt_depth,
                    valid_mask=gt_depth_valid,
                    sequence_level=True,
                )

            point_maps_world = build_world_point_map_from_depth_and_ray(pred_depth, pred_ray)

        for batch_index in range(batch_size):
            if max_samples is not None and evaluated_samples >= max_samples:
                break

            pred_item = pred_flow[batch_index : batch_index + 1]
            target_item = target_flow[batch_index : batch_index + 1]
            mask_item = deform_mask[batch_index : batch_index + 1]
            eval_rgbs_item = extract_batch_item(batch["rgbs"], batch_index)
            eval_intrinsics_item = extract_batch_item(batch["intrinsics"], batch_index)
            eval_extrinsics_item = extract_batch_item(batch["extrinsics"], batch_index)
            if "eval_scene_flow" in batch and "eval_deform_mask" in batch:
                eval_target_item = extract_batch_item(batch["eval_scene_flow"], batch_index).permute(0, 3, 1, 2).unsqueeze(0).to(
                    device=device,
                    dtype=target_item.dtype,
                )
                eval_mask_item = extract_batch_item(batch["eval_deform_mask"], batch_index).unsqueeze(0).to(
                    device=device,
                    dtype=mask_item.dtype,
                )
                native_size = tuple(int(dim) for dim in eval_target_item.shape[-2:])
                if pred_item.shape[-2:] != native_size:
                    pred_item = resize_btchw(pred_item, native_size, mode="bilinear")
                if mask_item.shape[-2:] != native_size:
                    mask_item = resize_bthw_mask(mask_item, native_size)
                target_item = eval_target_item
                mask_item = eval_mask_item
                if "eval_rgbs" in batch:
                    eval_rgbs_item = extract_batch_item(batch["eval_rgbs"], batch_index)
                if "eval_intrinsics" in batch:
                    eval_intrinsics_item = extract_batch_item(batch["eval_intrinsics"], batch_index)
            scene_flow_scale_item = None
            if "scene_flow_scale" in batch:
                scene_flow_scale_item = float(extract_batch_item(batch["scene_flow_scale"], batch_index).item())
            flow_p95_scale_item = None
            if "flow_p95_scale" in batch:
                flow_p95_scale_item = float(extract_batch_item(batch["flow_p95_scale"], batch_index).item())
            flow_target_amplification_item = 1.0
            if "flow_target_amplification" in batch:
                flow_target_amplification_item = float(extract_batch_item(batch["flow_target_amplification"], batch_index).item())

            pred_eval_item, target_eval_item = convert_flow_to_eval_space(
                pred_flow=pred_item,
                target_flow=target_item,
                target_space=target_space,
                eval_space=resolved_eval_space,
                scene_flow_scale=scene_flow_scale_item,
                flow_target_amplification=flow_target_amplification_item,
            )
            metric_tensors = criterion(pred_eval_item, target_eval_item, mask_item)
            metric_tensors["APD0.05"] = compute_apd_at_threshold(
                pred=pred_eval_item,
                target=target_eval_item,
                deform_mask=mask_item,
                threshold=0.05,
            )
            loss_dict = metric_dict_to_float(metric_tensors)

            valid = mask_item > 0.5
            loss_dict["valid_ratio"] = float(valid.float().mean().detach().cpu().item())
            pred_mag = torch.linalg.norm(pred_eval_item, dim=2)
            gt_mag = torch.linalg.norm(target_eval_item, dim=2)
            if valid.any():
                loss_dict["pred_mag_mean"] = float(pred_mag[valid].mean().detach().cpu().item())
                loss_dict["gt_mag_mean"] = float(gt_mag[valid].mean().detach().cpu().item())
                loss_dict["valid_pixels"] = float(valid.sum().detach().cpu().item())
            else:
                loss_dict["pred_mag_mean"] = 0.0
                loss_dict["gt_mag_mean"] = 0.0
                loss_dict["valid_pixels"] = 0.0

            sample_id = make_sample_id(batch, batch_index)
            dataset_name_item = batch.get("dataset_name")
            dataset_name_value = str(extract_batch_item(dataset_name_item, batch_index)) if dataset_name_item is not None else None
            source_split_item = batch.get("source_split")
            source_split_value = str(extract_batch_item(source_split_item, batch_index)) if source_split_item is not None else None
            clip_uid_item = batch.get("clip_uid")
            clip_uid_value = str(extract_batch_item(clip_uid_item, batch_index)) if clip_uid_item is not None else None
            frame_ids_value = None
            if "frame_ids" in batch:
                frame_ids_value = extract_batch_item(batch["frame_ids"], batch_index).detach().cpu().numpy()

            if scene_flow_scale_item is not None:
                loss_dict["scene_flow_scale"] = scene_flow_scale_item
            if flow_p95_scale_item is not None:
                loss_dict["flow_p95_scale"] = flow_p95_scale_item
            record = {
                "sample_id": sample_id,
                "dataset_name": dataset_name_value,
                "source_split": source_split_value,
                "clip_uid": clip_uid_value,
                "eval_space": resolved_eval_space,
                **loss_dict,
            }
            append_jsonl(per_sample_path, record)

            for key, value in loss_dict.items():
                all_metrics.setdefault(key, []).append(value)

            sequence_length = int(extract_batch_item(batch["sequence_length"], batch_index).item())
            should_save_vis = save_vis_every_n > 0 and evaluated_samples % save_vis_every_n == 0
            should_save_ply = save_scene_flow_ply and scene_flow_ply_every_n > 0 and evaluated_samples % scene_flow_ply_every_n == 0
            if should_save_vis or save_npz or should_save_ply:
                rgbs_np = eval_rgbs_item.cpu().numpy()
                intrinsics_np = eval_intrinsics_item.cpu().numpy()
                extrinsics_np = eval_extrinsics_item.cpu().numpy()
                pred_flow_np = pred_eval_item.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
                gt_flow_np = target_eval_item.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
                deform_mask_np = mask_item.squeeze(0).detach().cpu().numpy()
                point_map_world_np = None
                pred_flow_world_np = None
                gt_flow_world_np = None
                if point_maps_world is not None:
                    point_map_world_item = point_maps_world[batch_index]
                    if tuple(int(dim) for dim in point_map_world_item.shape[1:3]) != tuple(int(dim) for dim in pred_flow_np.shape[1:3]):
                        point_map_world_item = resize_thwc_tensor(
                            point_map_world_item,
                            target_size=(int(pred_flow_np.shape[1]), int(pred_flow_np.shape[2])),
                            mode="bilinear",
                        )
                    point_map_world_np = point_map_world_item.detach().cpu().numpy()
                if point_map_world_np is not None and (resolved_eval_space == "metric" or target_space == "metric"):
                    pred_flow_world_np = pred_eval_item.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
                    gt_flow_world_np = target_eval_item.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
                save_sample_outputs(
                    output_dir=output_dir,
                    sample_id=sample_id,
                    rgbs=rgbs_np,
                    pred_flow=pred_flow_np,
                    gt_flow=gt_flow_np,
                    deform_mask=deform_mask_np,
                    intrinsics=intrinsics_np,
                    extrinsics=extrinsics_np,
                    sequence_length=sequence_length,
                    quiver_step=quiver_step,
                    vis_scale=vis_scale,
                    save_gif=save_gif and should_save_vis,
                    save_npz=save_npz,
                    save_geometry_motion_video=save_geometry_motion_video and should_save_vis,
                    save_scene_flow_ply=should_save_ply,
                    scene_flow_ply_every_n=1,
                    scene_flow_ply_max_points=scene_flow_ply_max_points,
                    point_map_world=point_map_world_np,
                    pred_flow_world=pred_flow_world_np,
                    gt_flow_world=gt_flow_world_np,
                    scene_flow_scale=scene_flow_scale_item,
                    flow_p95_scale=flow_p95_scale_item,
                    dataset_name=dataset_name_value,
                    source_split=source_split_value,
                    clip_uid=clip_uid_value,
                    frame_ids=frame_ids_value,
                    eval_space=resolved_eval_space,
                    target_space=target_space,
                )

            evaluated_samples += 1
            progress.set_postfix(
                loss=f"{loss_dict['loss']:.4f}",
                epe=f"{loss_dict['epe']:.4f}",
                valid=f"{loss_dict['valid_ratio']:.4f}",
            )

        if max_samples is not None and evaluated_samples >= max_samples:
            break

    summary: Dict[str, float] = {"num_samples": float(evaluated_samples)}
    for key, values in all_metrics.items():
        summary[key] = float(np.mean(values)) if values else 0.0
    return summary


def resolve_output_dir(args: argparse.Namespace, checkpoint_path: Path, split: str) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        return output_dir if output_dir.is_absolute() else (DA3_ROOT / output_dir).resolve()

    training_output_root = find_training_output_root(checkpoint_path)
    if training_output_root is not None:
        return training_output_root / f"eval_scene_flow_{split}"
    return (DA3_ROOT / "training" / "outputs" / f"scene_flow_eval_{split}").resolve()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    if args.export_geometry_motion_overlays_from_dir is not None:
        exported_videos = export_geometry_motion_overlays_from_eval_dir(
            eval_dir=args.export_geometry_motion_overlays_from_dir,
            output_dir=args.geometry_motion_overlay_output_dir,
            fps=args.geometry_motion_overlay_fps,
            vis_scale=args.vis_scale,
            max_samples=args.geometry_motion_overlay_max_samples,
            renderer=args.geometry_motion_overlay_renderer,
        )
        LOGGER.info("Exported %d geometry-motion overlay videos", exported_videos)
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --export-geometry-motion-overlays-from-dir is used")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (DA3_ROOT / config_path).resolve()

    checkpoint_path = resolve_scene_flow_checkpoint_path(args.checkpoint)
    config = resolve_eval_config(config_path, checkpoint_path, args)
    output_dir = resolve_output_dir(args, checkpoint_path, config["data"]["split"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "resolved_eval_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    device = torch.device(args.device or ("cuda:1" if torch.cuda.is_available() else "cpu"))
    resolved_eval_space = resolve_eval_space(
        target_space=str(config["data"].get("flow_target_space", "metric")),
        eval_space=args.eval_space,
        metrics_on_pred_point_map=args.metrics_on_pred_point_map,
    )
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Resolved checkpoint: %s", checkpoint_path)
    LOGGER.info("Evaluation split: %s", config["data"]["split"])
    LOGGER.info("Target space: %s", config["data"].get("flow_target_space", "metric"))
    LOGGER.info("Eval space: %s", resolved_eval_space)
    LOGGER.info("Output dir: %s", output_dir)

    dataloader = create_dataloader(
        config["data"],
        dataset_name=args.dataset_name,
        prefer_direct_point_odyssey=args.chunk_size is not None or args.overlap is not None,
    )
    criterion = SceneFlowLoss(
        loss_type=config["loss"].get("loss_type", "l1"),
        zero_last_weight=float(config["loss"].get("zero_last_weight", 0.0)),
    )

    da3_model, load_info = load_da3_with_scene_flow_head(
        repo_root=DA3_ROOT,
        weights_dir=config["model"]["weights_dir"],
        scene_flow_preset=config["model"].get("scene_flow_preset", "da3-giant-sceneflow"),
        checkpoint_name=config["model"].get("checkpoint_name", "model.safetensors"),
        config_name=config["model"].get("config_name", "config.json"),
        device=device,
    )
    LOGGER.info("Loaded DA3 base config from %s", load_info["config_path"])
    LOGGER.info("Loaded DA3 base checkpoint from %s", load_info["checkpoint_path"])
    LOGGER.info(
        "DA3 base load_state_dict missing=%d unexpected=%d",
        len(load_info["missing_keys"]),
        len(load_info["unexpected_keys"]),
    )

    scene_flow_head = get_scene_flow_head(da3_model)
    load_scene_flow_head_checkpoint(scene_flow_head, checkpoint_path)
    da3_model.eval()
    scene_flow_head.eval()

    da3_interface = DA3StreamingInterface(
        da3_model=da3_model,
        freeze_encoder=True,
        freeze_decoder=True,
    )
    decoder_chunk_size = int(config["model"].get("decoder_chunk_size", 8))

    summary = evaluate(
        da3_interface=da3_interface,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        decoder_chunk_size=decoder_chunk_size,
        max_samples=args.max_samples,
        save_vis_every_n=args.save_vis_every_n,
        quiver_step=args.quiver_step,
        vis_scale=args.vis_scale,
        save_gif=args.save_gif,
        save_npz=args.save_npz,
        save_geometry_motion_video=args.save_geometry_motion_video,
        metrics_on_pred_point_map=args.metrics_on_pred_point_map,
        target_space=str(config["data"].get("flow_target_space", "metric")),
        eval_space=args.eval_space,
        ref_view_strategy=args.ref_view_strategy,
        save_scene_flow_ply=args.save_scene_flow_ply,
        scene_flow_ply_every_n=args.scene_flow_ply_every_n,
        scene_flow_ply_max_points=args.scene_flow_ply_max_points,
    )

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    LOGGER.info("Saved metrics summary to %s", summary_path)
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
