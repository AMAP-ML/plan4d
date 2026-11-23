from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.relative_pose import compute_relative_pose_metrics


DEFAULT_OUTPUT_NAME = "relative_pose_metrics.json"
DEFAULT_ROOTS = (
    "eth3d_eval_outputs",
    "movi_f_eval_outputs",
    "scannetpp_eval_outputs",
    "dynamic_replica_eval_outputs",
)


def _mean_of_dicts(records: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    output: dict[str, float] = {}
    for key in keys:
        values = [
            float(record[key])
            for record in records
            if key in record and isinstance(record[key], (int, float, np.integer, np.floating))
        ]
        if values:
            output[key] = float(np.mean(values))
    return output


def _aggregate_records(per_clip: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    per_scene: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for clip_key, metrics in per_clip.items():
        parts = clip_key.split("/")
        scene_name = parts[1] if len(parts) > 1 else parts[0]
        per_scene[scene_name].append(metrics)
    per_scene_mean = {
        scene: _mean_of_dicts(records)
        for scene, records in sorted(per_scene.items())
    }
    return {
        "per_clip": {key: dict(value) for key, value in sorted(per_clip.items())},
        "per_scene": per_scene_mean,
        "mean": _mean_of_dicts(list(per_scene_mean.values())),
    }


def _relative_model_result_key(model_root: Path, result_path: Path) -> tuple[str, str]:
    relative = result_path.relative_to(model_root / "model_results")
    parts = relative.parts
    if len(parts) < 5:
        raise ValueError(f"Unexpected result path under model_results: {result_path}")
    mode = result_path.parents[2].name
    clip_parts = parts[:-4]
    clip_key = "/".join(clip_parts)
    return clip_key, mode


def _load_extrinsics_pair(result_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gt_meta_path = result_path.parents[1] / "gt_meta.npz"
    if not gt_meta_path.exists():
        raise FileNotFoundError(f"Missing GT metadata next to {result_path}: {gt_meta_path}")
    with np.load(result_path, allow_pickle=True) as pred_npz:
        pred_extrinsics = np.asarray(pred_npz["extrinsics"], dtype=np.float64)
    with np.load(gt_meta_path, allow_pickle=True) as gt_npz:
        gt_extrinsics = np.asarray(gt_npz["extrinsics"], dtype=np.float64)
    return pred_extrinsics, gt_extrinsics


def _model_root_from_result_path(result_path: Path) -> Path:
    for parent in result_path.parents:
        if parent.name == "model_results":
            return parent.parent
    raise ValueError(f"Could not locate model_results ancestor for {result_path}")


def discover_model_roots(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if (root / "model_results").is_dir():
        return [root]
    return sorted(
        {
            _model_root_from_result_path(path)
            for path in root.glob("**/model_results/**/exports/mini_npz/results.npz")
        }
    )


def evaluate_model_root(
    model_root: str | Path,
    *,
    threshold_deg: float = 5.0,
    write: bool = True,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    model_root = Path(model_root).expanduser().resolve()
    per_mode: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    errors: dict[str, str] = {}
    result_paths = sorted((model_root / "model_results").glob("**/exports/mini_npz/results.npz"))
    for result_path in result_paths:
        try:
            clip_key, mode = _relative_model_result_key(model_root, result_path)
            pred_extrinsics, gt_extrinsics = _load_extrinsics_pair(result_path)
            per_mode[mode][clip_key] = compute_relative_pose_metrics(
                pred_extrinsics,
                gt_extrinsics,
                threshold_deg=float(threshold_deg),
            )
        except Exception as exc:
            errors[str(result_path.relative_to(model_root))] = str(exc)

    by_mode = {
        mode: _aggregate_records(records)
        for mode, records in sorted(per_mode.items())
    }
    all_records = {
        f"{mode}/{clip_key}": metrics
        for mode, records in per_mode.items()
        for clip_key, metrics in records.items()
    }
    payload = {
        "metadata": {
            "model_root": str(model_root),
            "num_result_files": float(len(result_paths)),
            "threshold_deg": float(threshold_deg),
            "rra_key": f"rra_{int(threshold_deg) if float(threshold_deg).is_integer() else threshold_deg}",
            "rta_key": f"rta_{int(threshold_deg) if float(threshold_deg).is_integer() else threshold_deg}",
            "rpe_t_definition": "mean(||t_pred_rel - t_gt_rel|| / ||t_gt_rel||) over valid frame pairs",
            "rpe_t_scale_aligned_definition": "same as rpe_t after one optimal scalar on all predicted relative translations",
            "rpe_r_definition": "relative rotation angle in degrees over all frame pairs",
        },
        "by_mode": by_mode,
        "all": _aggregate_records(all_records),
        "errors": errors,
    }
    if write:
        output_path = model_root / "metric_results" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def evaluate_roots(
    roots: Iterable[str | Path],
    *,
    threshold_deg: float = 5.0,
    write: bool = True,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    outputs = {}
    for root in roots:
        for model_root in discover_model_roots(root):
            outputs[str(model_root)] = evaluate_model_root(
                model_root,
                threshold_deg=float(threshold_deg),
                write=bool(write),
                output_name=str(output_name),
            )
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Posthoc relative pose metrics from saved results.npz and gt_meta.npz exports."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=None,
        help="Output roots or model roots to scan. Defaults to common *_eval_outputs directories.",
    )
    parser.add_argument(
        "--threshold_deg",
        type=float,
        default=5.0,
        help="Angular threshold for RRA/RTA accuracy metrics.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Metric JSON filename written under each model_root/metric_results.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print summary without writing JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    roots = args.roots or [str(repo_root / item) for item in DEFAULT_ROOTS]
    payloads = evaluate_roots(
        roots,
        threshold_deg=float(args.threshold_deg),
        write=not bool(args.dry_run),
        output_name=str(args.output_name),
    )
    for model_root, payload in sorted(payloads.items()):
        modes = ", ".join(sorted(payload["by_mode"].keys())) or "none"
        error_count = len(payload.get("errors", {}))
        print(f"{model_root}: modes={modes}, errors={error_count}")
    if not payloads:
        print("No saved results.npz files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
