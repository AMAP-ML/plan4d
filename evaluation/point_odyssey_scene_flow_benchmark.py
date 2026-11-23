'''
Run the stitched DA3 PointOdyssey test split scene-flow evaluation:

PYTHONPATH=. \
python \
./evaluation/point_odyssey_scene_flow_benchmark.py \
  --checkpoint_path ./checkpoints/stitched_decoder.pth \
  --point_odyssey_root ./datasets/point_odyssey/dataset \
  --output_dir ./outputs/evaluation/point_odyssey \
  --split test \
  --clip_length 81 \
  --overlap 1 \
  --input_resolution 532 \
  --da3_enable_scene_flow_head \
  --scene_flow_target_space relative \
  --scene_flow_eval_space metric \
  --eval_modes posed \
  --device cuda:0
'''

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"
DA3_SRC_ROOT = DA3_ROOT / "src"

for import_path in reversed((PROJECT_ROOT, DA3_ROOT, DA3_SRC_ROOT, PLAN4D_ROOT)):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from evaluation.dynamic_replica_ray_benchmark import (  # noqa: E402
    DEFAULT_DA3_SCENE_FLOW_PRESET,
    DEFAULT_SCENE_FLOW_EVAL_SPACE,
    DEFAULT_SCENE_FLOW_TARGET_SPACE,
    DEFAULT_SCENE_FLOW_THRESHOLD,
    _aggregate_metric_records,
    compute_dynamic_replica_scene_flow_metrics,
    default_da3_scene_flow_head_checkpoint_path,
    dump_json,
    _write_raw_camera_params_npz,
    _write_results_npz,
)
from evaluation.eth3d_ray_benchmark import (  # noqa: E402
    default_wan_vae_path,
    load_stitched_runner,
    run_da3_stitched_scene,
)
from training.datasets.point_odyssey_scene_flow import (  # noqa: E402
    PointOdysseySceneFlowDataset,
    SampleInfo,
)


DEFAULT_INPUT_RESOLUTION = 504
DEFAULT_CLIP_LENGTH = 81
DEFAULT_OVERLAP = 1
DEFAULT_SPLIT = "test"
DEFAULT_TARGET_SPACE = DEFAULT_SCENE_FLOW_TARGET_SPACE
DEFAULT_EVAL_SPACE = DEFAULT_SCENE_FLOW_EVAL_SPACE
DEFAULT_RELATIVE_SCALE_SOURCE = "geometry_radius"
DEFAULT_GEOMETRY_SCALE_STAT = "mean_radius"
DEFAULT_FLOW_SCALE_STAT = "p95"
DEFAULT_CHECKPOINT_PATH = str((PROJECT_ROOT / "checkpoints" / "stitched_decoder.pth").resolve())


@dataclass(frozen=True)
class PointOdysseyClip:
    split: str
    scene_name: str
    start_idx: int
    end_idx: int

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(range(int(self.start_idx), int(self.end_idx)))

    @property
    def num_frames(self) -> int:
        return int(self.end_idx) - int(self.start_idx)

    @property
    def clip_uid(self) -> str:
        return f"{self.scene_name}:{int(self.start_idx):05d}:{int(self.end_idx):05d}"

    @property
    def clip_dirname(self) -> str:
        return f"{int(self.start_idx):05d}_{int(self.end_idx):05d}"

    @property
    def clip_key(self) -> str:
        return f"{self.scene_name}/{self.clip_dirname}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "scene_name": self.scene_name,
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "frame_indices": [int(frame_id) for frame_id in self.frame_indices],
            "clip_uid": self.clip_uid,
            "clip_dirname": self.clip_dirname,
            "num_frames": self.num_frames,
        }


def default_point_odyssey_root() -> str:
    return str((PROJECT_ROOT / "datasets" / "point_odyssey" / "dataset").resolve())


def default_da3_model_path() -> str:
    return str((DA3_ROOT.parent / "checkpoints" / "da3").resolve())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def point_odyssey_clip_from_sample(sample: Any, split: str) -> PointOdysseyClip:
    return PointOdysseyClip(
        split=str(split),
        scene_name=str(sample.scene_name),
        start_idx=int(sample.start_idx),
        end_idx=int(sample.end_idx),
    )


def select_point_odyssey_samples(
    samples: Sequence[Any],
    *,
    scenes: Sequence[str] | None = None,
    lazy_mode: bool = False,
    max_clips: int = -1,
) -> list[Any]:
    requested_scenes = {str(scene_name) for scene_name in scenes} if scenes else None
    selected = [
        sample
        for sample in samples
        if requested_scenes is None or str(sample.scene_name) in requested_scenes
    ]
    if lazy_mode:
        first_by_scene: dict[str, Any] = {}
        for sample in selected:
            first_by_scene.setdefault(str(sample.scene_name), sample)
        selected = list(first_by_scene.values())
    if max_clips > 0:
        selected = selected[: int(max_clips)]
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PLAN4D stitched DA3 scene flow on the PointOdyssey test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--point_odyssey_root", type=str, default=default_point_odyssey_root())
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--clip_length", type=int, default=DEFAULT_CLIP_LENGTH)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--input_resolution", type=int, default=DEFAULT_INPUT_RESOLUTION)
    parser.add_argument("--target_height", type=int, default=DEFAULT_INPUT_RESOLUTION)
    parser.add_argument("--target_width", type=int, default=DEFAULT_INPUT_RESOLUTION)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max_clips", type=int, default=-1)
    parser.add_argument("--lazy_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_modes", nargs="+", choices=["unposed", "posed"], default=["posed"])
    parser.add_argument("--da3_model_path", type=str, default=default_da3_model_path())
    parser.add_argument("--wan_vae_path", type=str, default=default_wan_vae_path())
    parser.add_argument("--da3_config_name", type=str, default="config.json")
    parser.add_argument("--da3_checkpoint_name", type=str, default="model.safetensors")
    parser.add_argument("--da3_enable_scene_flow_head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--da3_scene_flow_head_checkpoint_path", type=str, default=default_da3_scene_flow_head_checkpoint_path())
    parser.add_argument("--da3_scene_flow_preset", type=str, default=DEFAULT_DA3_SCENE_FLOW_PRESET)
    parser.add_argument("--da3_ref_view_strategy", type=str, default="first", choices=["saddle_balanced", "first", "middle", "last"])
    parser.add_argument("--da3_process_res_method", type=str, default="upper_bound_resize", choices=["upper_bound_resize", "square_center_crop"])
    parser.add_argument("--stitching_layer_location", type=str, default="enc_blocks_1")
    parser.add_argument("--stitching_layer_config", type=str, default="conv3d_k5x3x3_o1536_s1x2x2_p2x1x1")
    parser.add_argument("--lora_config", type=str, default="r512,a256,d0.05,f0")
    parser.add_argument("--full_finetune_stitched_parts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--force_fp32_inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stitched_da3_sky_postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stitched_da3_camera_param_source", type=str, default="ray_map", choices=["ray_map", "camera_token"])
    parser.add_argument("--scene_flow_target_space", type=str, default=DEFAULT_TARGET_SPACE, choices=["relative", "metric"])
    parser.add_argument("--scene_flow_eval_space", type=str, default=DEFAULT_EVAL_SPACE, choices=["auto", "native", "metric"])
    parser.add_argument("--scene_flow_threshold", type=float, default=DEFAULT_SCENE_FLOW_THRESHOLD)
    parser.add_argument("--relative_scale_source", type=str, default=DEFAULT_RELATIVE_SCALE_SOURCE, choices=["flow_p95", "geometry_radius"])
    parser.add_argument("--geometry_scale_stat", type=str, default=DEFAULT_GEOMETRY_SCALE_STAT)
    parser.add_argument("--flow_scale_stat", type=str, default=DEFAULT_FLOW_SCALE_STAT)
    parser.add_argument("--flow_scale_eps", type=float, default=1e-6)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.point_odyssey_root).expanduser()
    split_dir = dataset_root / str(args.split)
    if not split_dir.exists():
        raise FileNotFoundError(f"PointOdyssey split directory does not exist: {split_dir}")
    if not args.eval_only and not Path(args.checkpoint_path).expanduser().exists():
        raise FileNotFoundError(f"Stitched checkpoint does not exist: {args.checkpoint_path}")
    if args.clip_length <= 0:
        raise ValueError("--clip_length must be positive.")
    if args.overlap < 0 or args.overlap >= args.clip_length:
        raise ValueError("--overlap must be non-negative and smaller than --clip_length.")
    if args.input_resolution <= 0:
        raise ValueError("--input_resolution must be positive.")
    if args.target_height <= 0 or args.target_width <= 0:
        raise ValueError("--target_height and --target_width must be positive.")
    if args.max_clips == 0:
        raise ValueError("--max_clips must be positive or -1.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be positive when provided.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.scene_flow_threshold <= 0:
        raise ValueError("--scene_flow_threshold must be positive.")


def create_point_odyssey_dataset(args: argparse.Namespace) -> PointOdysseySceneFlowDataset:
    dataset = PointOdysseySceneFlowDataset(
        data_root=args.point_odyssey_root,
        split=args.split,
        chunk_size=int(args.clip_length),
        overlap=int(args.overlap),
        target_height=int(args.target_height),
        target_width=int(args.target_width),
        process_res=int(args.input_resolution),
        process_res_method=str(args.da3_process_res_method),
        max_scenes=args.max_scenes,
        pad_to_chunk_size=True,
        flow_target_space=str(args.scene_flow_target_space),
        relative_scale_source=str(args.relative_scale_source),
        geometry_scale_stat=str(args.geometry_scale_stat),
        flow_scale_stat=str(args.flow_scale_stat),
        flow_scale_eps=float(args.flow_scale_eps),
    )
    dataset.samples = select_point_odyssey_samples(
        dataset.samples,
        scenes=args.scenes,
        lazy_mode=bool(args.lazy_mode),
        max_clips=int(args.max_clips),
    )
    if not dataset.samples:
        raise ValueError("No PointOdyssey clips selected for evaluation.")
    return dataset


def create_point_odyssey_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = create_point_odyssey_dataset(args)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
    )


def resolve_runtime_max_frames(args: argparse.Namespace, clips: Sequence[PointOdysseyClip]) -> int:
    if not clips:
        raise ValueError("No PointOdyssey clips selected for evaluation.")
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


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_batch_item(batch_value: Any, index: int) -> Any:
    if torch.is_tensor(batch_value):
        return batch_value[index]
    if isinstance(batch_value, np.ndarray):
        return batch_value[index]
    if isinstance(batch_value, (list, tuple)):
        return batch_value[index]
    if isinstance(batch_value, str):
        return batch_value
    return batch_value


def point_odyssey_clip_from_batch(batch: Mapping[str, Any], batch_index: int, split: str) -> PointOdysseyClip:
    scene_name = str(_extract_batch_item(batch["scene_name"], batch_index))
    start_idx = int(_extract_batch_item(batch["start_idx"], batch_index).item())
    end_idx = int(_extract_batch_item(batch["end_idx"], batch_index).item())
    return PointOdysseyClip(split=str(split), scene_name=scene_name, start_idx=start_idx, end_idx=end_idx)


def _sequence_length_from_batch(batch: Mapping[str, Any], batch_index: int) -> int:
    return int(_extract_batch_item(batch["sequence_length"], batch_index).item())


def point_odyssey_gt_payload_from_batch(batch: Mapping[str, Any], batch_index: int) -> dict[str, Any]:
    sequence_length = _sequence_length_from_batch(batch, batch_index)
    payload = {
        "rgbs": _to_numpy(_extract_batch_item(batch["rgbs"], batch_index))[:sequence_length].astype(np.uint8),
        "intrinsics": _to_numpy(_extract_batch_item(batch["intrinsics"], batch_index))[:sequence_length].astype(np.float32),
        "extrinsics": _to_numpy(_extract_batch_item(batch["extrinsics"], batch_index))[:sequence_length].astype(np.float32),
        "scene_flow": _to_numpy(_extract_batch_item(batch["scene_flow"], batch_index))[:sequence_length].astype(np.float32),
        "deform_mask": _to_numpy(_extract_batch_item(batch["deform_mask"], batch_index))[:sequence_length].astype(np.float32),
        "sequence_length": sequence_length,
    }
    if "scene_flow_scale" in batch:
        payload["scene_flow_scale"] = float(_extract_batch_item(batch["scene_flow_scale"], batch_index).item())
    if "flow_p95_scale" in batch:
        payload["flow_p95_scale"] = float(_extract_batch_item(batch["flow_p95_scale"], batch_index).item())
    return payload


def scene_data_from_gt_payload(clip: PointOdysseyClip, payload: Mapping[str, Any]) -> SimpleNamespace:
    image_files = [
        f"{clip.scene_name}/rgb_{frame_idx:05d}.jpg"
        for frame_idx in clip.frame_indices[: int(payload["sequence_length"])]
    ]
    return SimpleNamespace(
        scene_name=clip.scene_name,
        image_files=image_files,
        extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
        intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
    )


def prediction_export_dir(work_dir: Path, clip: PointOdysseyClip, mode: str) -> Path:
    return work_dir / "point_odyssey" / clip.scene_name / clip.clip_dirname / str(mode)


def prediction_result_path(work_dir: str | Path, clip: PointOdysseyClip, mode: str) -> Path:
    return prediction_export_dir(Path(work_dir), clip, mode) / "exports" / "mini_npz" / "results.npz"


def _result_has_scene_flow(result_path: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        with np.load(result_path) as payload:
            return "scene_flow" in payload
    except Exception as exc:
        logger.warning("Ignoring unreadable existing result {}: {}", result_path, exc)
        return False


def _write_preprocess_meta_npz(export_dir: Path, preprocess_meta: Mapping[str, Any] | None) -> None:
    if not preprocess_meta:
        return
    save_dict = {
        key: np.asarray(value)
        for key, value in preprocess_meta.items()
        if value is not None
    }
    if not save_dict:
        return
    path = export_dir / "exports" / "preprocess_meta.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **save_dict)


def _write_gt_meta_npz(export_dir: Path, clip: PointOdysseyClip, payload: Mapping[str, Any]) -> None:
    path = export_dir / "exports" / "gt_meta.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "scene_name": np.array(clip.scene_name, dtype=object),
        "clip_uid": np.array(clip.clip_uid, dtype=object),
        "split": np.array(clip.split, dtype=object),
        "frame_indices": np.asarray(clip.frame_indices[: int(payload["sequence_length"])], dtype=np.int64),
        "extrinsics": np.asarray(payload["extrinsics"], dtype=np.float32),
        "intrinsics": np.asarray(payload["intrinsics"], dtype=np.float32),
        "scene_flow_scale": np.array(float(payload.get("scene_flow_scale", 1.0)), dtype=np.float32),
        "flow_p95_scale": np.array(float(payload.get("flow_p95_scale", 1.0)), dtype=np.float32),
    }
    np.savez_compressed(path, **save_dict)


def write_point_odyssey_clip_exports(
    work_dir: Path,
    clip: PointOdysseyClip,
    mode: str,
    prediction: Mapping[str, Any],
    gt_payload: Mapping[str, Any],
    preprocess_meta: Mapping[str, Any] | None,
) -> None:
    export_dir = prediction_export_dir(work_dir, clip, mode)
    _write_results_npz(export_dir, prediction)
    _write_raw_camera_params_npz(export_dir, prediction)
    _write_preprocess_meta_npz(export_dir, preprocess_meta)
    _write_gt_meta_npz(export_dir, clip, gt_payload)


def compute_point_odyssey_scene_flow_metrics(
    pred_flow: np.ndarray,
    target_flow: np.ndarray,
    deform_mask: np.ndarray,
    *,
    scene_flow_scale: float | None,
    target_space: str = DEFAULT_TARGET_SPACE,
    eval_space: str = DEFAULT_EVAL_SPACE,
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


def _make_single_clip_dataset(args: argparse.Namespace, clip: PointOdysseyClip) -> PointOdysseySceneFlowDataset:
    dataset = object.__new__(PointOdysseySceneFlowDataset)
    dataset.data_root = Path(args.point_odyssey_root)
    dataset.split = str(clip.split)
    dataset.chunk_size = int(args.clip_length)
    dataset.overlap = int(args.overlap)
    dataset.target_height = int(args.target_height)
    dataset.target_width = int(args.target_width)
    dataset.process_res = int(args.input_resolution)
    dataset.process_res_method = str(args.da3_process_res_method)
    dataset.max_scenes = None
    dataset.pad_to_chunk_size = True
    dataset.flow_target_space = str(args.scene_flow_target_space)
    dataset.relative_scale_source = str(args.relative_scale_source)
    dataset.geometry_scale_stat = str(args.geometry_scale_stat)
    dataset.flow_scale_stat = str(args.flow_scale_stat)
    dataset.flow_scale_eps = float(args.flow_scale_eps)
    dataset.samples = [
        SampleInfo(
            scene_name=clip.scene_name,
            scene_dir=Path(args.point_odyssey_root) / clip.split / clip.scene_name,
            start_idx=int(clip.start_idx),
            end_idx=int(clip.end_idx),
        )
    ]
    dataset._skipped_scene_names = set()
    return dataset


def load_point_odyssey_clip_scene_flow_gt(
    clip: PointOdysseyClip,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = _make_single_clip_dataset(args, clip)
    sample = dataset[0]
    return {
        "scene_flow": _to_numpy(sample["scene_flow"]).astype(np.float32)[: clip.num_frames],
        "deform_mask": _to_numpy(sample["deform_mask"]).astype(np.float32)[: clip.num_frames],
        "scene_flow_scale": float(sample.get("scene_flow_scale", 1.0)),
        "flow_p95_scale": float(sample.get("flow_p95_scale", 1.0)),
    }


def evaluate_point_odyssey_scene_flow_work_dir(
    *,
    work_dir: str | Path,
    clips: Sequence[PointOdysseyClip],
    modes: Sequence[str],
    scene_flow_gt_loader: Callable[[PointOdysseyClip], Mapping[str, Any]],
    target_space: str,
    eval_space: str,
    threshold: float,
) -> dict[str, Any]:
    work_dir = Path(work_dir)
    metric_dir = work_dir / "metric_results"
    metric_dir.mkdir(parents=True, exist_ok=True)
    per_mode_clip_metrics: dict[str, dict[str, dict[str, float]]] = {
        str(mode): {} for mode in modes
    }

    for clip in clips:
        gt_payload = scene_flow_gt_loader(clip)
        for mode in modes:
            result_path = prediction_result_path(work_dir, clip, mode)
            if not result_path.exists():
                logger.warning("Skipping PointOdyssey scene-flow {} metrics for missing export {}", mode, result_path)
                continue
            with np.load(result_path) as pred_npz:
                if "scene_flow" not in pred_npz:
                    logger.warning("Skipping PointOdyssey scene-flow {} metrics for {} because scene_flow is missing.", mode, clip.clip_uid)
                    continue
                pred_scene_flow = np.asarray(pred_npz["scene_flow"], dtype=np.float32)
            metrics = compute_point_odyssey_scene_flow_metrics(
                pred_scene_flow,
                np.asarray(gt_payload["scene_flow"], dtype=np.float32),
                np.asarray(gt_payload["deform_mask"], dtype=np.float32),
                scene_flow_scale=float(gt_payload.get("scene_flow_scale", 1.0)),
                target_space=str(target_space),
                eval_space=str(eval_space),
                threshold=float(threshold),
            )
            metrics["num_frames"] = float(clip.num_frames)
            per_mode_clip_metrics[str(mode)][clip.clip_key] = metrics

    if not any(per_mode_clip_metrics.values()):
        raise RuntimeError(
            f"No scene_flow predictions were found under {work_dir}. "
            "Run inference first with --da3_enable_scene_flow_head or pass --eval_only to an existing export directory."
        )

    summary: dict[str, Any] = {}
    for mode, per_clip in per_mode_clip_metrics.items():
        if not per_clip:
            continue
        payload = _aggregate_metric_records(per_clip)
        dump_json(metric_dir / f"point_odyssey_scene_flow_{mode}.json", payload)
        summary[f"scene_flow_{mode}"] = payload
    dump_json(metric_dir / "point_odyssey_scene_flow_summary.json", summary)
    return summary


def _build_run_metadata(
    args: argparse.Namespace,
    *,
    clips: Sequence[PointOdysseyClip],
) -> dict[str, Any]:
    return {
        "dataset": "point_odyssey",
        "split": str(args.split),
        "point_odyssey_root": str(args.point_odyssey_root),
        "checkpoint_path": str(args.checkpoint_path),
        "clip_length": int(args.clip_length),
        "overlap": int(args.overlap),
        "input_resolution": int(args.input_resolution),
        "process_res_method": str(args.da3_process_res_method),
        "clip_count": len(clips),
        "scenes": sorted({clip.scene_name for clip in clips}),
        "eval_modes": list(args.eval_modes),
        "da3_enable_scene_flow_head": bool(args.da3_enable_scene_flow_head),
        "da3_scene_flow_head_checkpoint_path": str(args.da3_scene_flow_head_checkpoint_path),
        "da3_scene_flow_preset": str(args.da3_scene_flow_preset),
        "scene_flow_target_space": str(args.scene_flow_target_space),
        "scene_flow_eval_space": str(args.scene_flow_eval_space),
        "relative_scale_source": str(args.relative_scale_source),
        "geometry_scale_stat": str(args.geometry_scale_stat),
        "flow_scale_stat": str(args.flow_scale_stat),
        "clips": [clip.to_dict() for clip in clips],
    }


def _collect_gt_cache_from_batch(
    batch: Mapping[str, Any],
    *,
    batch_size: int,
    split: str,
    gt_cache: dict[str, dict[str, Any]],
) -> list[tuple[PointOdysseyClip, dict[str, Any]]]:
    collected = []
    for batch_index in range(batch_size):
        clip = point_odyssey_clip_from_batch(batch, batch_index, split)
        payload = point_odyssey_gt_payload_from_batch(batch, batch_index)
        gt_cache[clip.clip_key] = payload
        collected.append((clip, payload))
    return collected


@torch.no_grad()
def run_point_odyssey_inference(
    *,
    runner,
    dataloader: DataLoader,
    args: argparse.Namespace,
    work_dir: Path,
    gt_cache: dict[str, dict[str, Any]],
) -> None:
    for batch in dataloader:
        batch_size = int(_to_numpy(batch["rgbs"]).shape[0])
        for clip, gt_payload in _collect_gt_cache_from_batch(
            batch,
            batch_size=batch_size,
            split=str(args.split),
            gt_cache=gt_cache,
        ):
            scene_data = scene_data_from_gt_payload(clip, gt_payload)
            input_images = [frame for frame in np.asarray(gt_payload["rgbs"], dtype=np.uint8)]
            for mode in args.eval_modes:
                result_path = prediction_result_path(work_dir, clip, mode)
                if args.resume_existing and _result_has_scene_flow(result_path):
                    logger.info("Reusing existing PointOdyssey {} export for {}", mode, clip.clip_uid)
                    continue
                logger.info("Running PointOdyssey stitched DA3 {} inference for {}", mode, clip.clip_uid)
                if mode == "posed":
                    extrinsics = np.asarray(gt_payload["extrinsics"], dtype=np.float32)
                    intrinsics = np.asarray(gt_payload["intrinsics"], dtype=np.float32)
                else:
                    extrinsics = None
                    intrinsics = None
                prediction, preprocess_meta = run_da3_stitched_scene(
                    runner,
                    scene_data=scene_data,
                    args=args,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    process_res_method=str(args.da3_process_res_method),
                    input_images=input_images,
                )
                if prediction.get("scene_flow") is None:
                    raise RuntimeError(
                        "Stitched inference did not produce scene_flow. "
                        "Confirm --da3_enable_scene_flow_head and --da3_scene_flow_head_checkpoint_path."
                    )
                write_point_odyssey_clip_exports(
                    work_dir,
                    clip,
                    str(mode),
                    prediction,
                    gt_payload,
                    preprocess_meta,
                )
                gc.collect()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(None if argv is None else list(argv))
    validate_args(args)

    dataloader = create_point_odyssey_dataloader(args)
    clips = [
        point_odyssey_clip_from_sample(sample, str(args.split))
        for sample in dataloader.dataset.samples
    ]
    resolve_runtime_max_frames(args, clips)

    output_root = Path(args.output_dir).expanduser().resolve()
    work_dir = output_root / "da3_stitched"
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(work_dir / "run_meta.json", _build_run_metadata(args, clips=clips))

    gt_cache: dict[str, dict[str, Any]] = {}
    if args.eval_only:
        for batch in dataloader:
            batch_size = int(_to_numpy(batch["rgbs"]).shape[0])
            _collect_gt_cache_from_batch(
                batch,
                batch_size=batch_size,
                split=str(args.split),
                gt_cache=gt_cache,
            )
    else:
        runner = load_stitched_runner(args, da3_process_res_method=str(args.da3_process_res_method))
        run_point_odyssey_inference(
            runner=runner,
            dataloader=dataloader,
            args=args,
            work_dir=work_dir,
            gt_cache=gt_cache,
        )

    def gt_loader(clip: PointOdysseyClip) -> Mapping[str, Any]:
        cached = gt_cache.get(clip.clip_key)
        if cached is not None:
            return cached
        return load_point_odyssey_clip_scene_flow_gt(clip, args)

    metrics = evaluate_point_odyssey_scene_flow_work_dir(
        work_dir=work_dir,
        clips=clips,
        modes=args.eval_modes,
        scene_flow_gt_loader=gt_loader,
        target_space=str(args.scene_flow_target_space),
        eval_space=str(args.scene_flow_eval_space),
        threshold=float(args.scene_flow_threshold),
    )
    all_metrics = {"da3_stitched": metrics}
    dump_json(output_root / "point_odyssey_scene_flow_summary_all_models.json", all_metrics)
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
