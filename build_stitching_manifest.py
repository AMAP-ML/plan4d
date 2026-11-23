from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml

from data.stitching_manifest import (
    build_blinkvision_manifest_entries,
    build_dynamic_replica_manifest_entries_from_scene_flow_manifest,
    build_mvs_synth_manifest_entries,
    build_omniworld_manifest_entries,
    build_point_odyssey_manifest_entries,
    build_scannetpp_manifest_entries,
    build_virtual_kitti2_manifest_entries,
    save_stitching_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _requested_dataset_names(datasets: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not datasets:
        return None
    return {str(name) for name in datasets}


def _parse_split_manifest_spec(spec: str) -> tuple[str, Path]:
    split, sep, raw_path = spec.partition(':')
    split = split.strip()
    raw_path = raw_path.strip()
    if not sep or not split or not raw_path:
        raise argparse.ArgumentTypeError(
            f'Expected SPLIT:PATH for Dynamic Replica source manifests, got {spec!r}'
        )
    return split, Path(raw_path)


def build_dynamic_replica_stitching_manifests(
    source_manifests: Sequence[tuple[str, str | Path]],
    *,
    data_root: str | Path,
    relative_to: str | Path,
    output_dir: str | Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: Dict[str, Path] = {}
    for requested_split, source_manifest_path in source_manifests:
        entries = build_dynamic_replica_manifest_entries_from_scene_flow_manifest(
            source_manifest_path,
            data_root=data_root,
            relative_to=relative_to,
        )
        if not entries:
            continue
        output_path = output_dir / f'dynamic_replica_{requested_split}.jsonl'
        save_stitching_manifest(entries, output_path)
        manifest_paths[f'dynamic_replica_{requested_split}'] = output_path
    return manifest_paths


def build_stitching_manifests(
    data_cfg: dict,
    *,
    output_dir: str | Path,
    clip_length: int = 81,
    overlap: int = 1,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_data_root = Path(data_cfg["manifest_data_root"]).resolve()
    requested = _requested_dataset_names(datasets)
    manifest_paths: Dict[str, Path] = {}

    for spec in data_cfg.get("datasets", []):
        name = str(spec["name"])
        if requested is not None and name not in requested:
            continue
        data_root = _resolve_path(spec["data_root"], base=manifest_data_root)

        if name == "point_odyssey":
            for split in spec.get("train_splits", ["train"]):
                entries = build_point_odyssey_manifest_entries(
                    data_root=data_root,
                    relative_to=manifest_data_root,
                    splits=[split],
                    chunk_size=clip_length,
                    overlap=overlap,
                    max_scenes=spec.get("max_scenes"),
                )
                if not entries:
                    continue
                path = output_dir / f"{name}_{split}.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_{split}"] = path
            continue

        if name == "blinkvision":
            for split in spec.get("train_splits", ["indoor_train", "outdoor_train"]):
                entries = build_blinkvision_manifest_entries(
                    data_root=data_root,
                    relative_to=manifest_data_root,
                    splits=[split],
                    chunk_size=clip_length,
                    overlap=overlap,
                    max_scenes_per_split=spec.get("max_scenes_per_split"),
                    rgb_dir=str(spec.get("rgb_dir", "clean_uint8")),
                )
                if not entries:
                    continue
                path = output_dir / f"{name}_{split}.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_{split}"] = path
            continue

        if name == "mvs_synth":
            entries = build_mvs_synth_manifest_entries(
                data_root=data_root,
                relative_to=manifest_data_root,
                chunk_size=clip_length,
                overlap=overlap,
                max_scenes=spec.get("max_scenes"),
            )
            if entries:
                path = output_dir / f"{name}_train.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_train"] = path
            continue

        if name == "omniworld":
            entries = build_omniworld_manifest_entries(
                data_root=data_root,
                relative_to=manifest_data_root,
                splits=spec.get("train_splits", []),
                chunk_size=clip_length,
                overlap=overlap,
                max_scenes=spec.get("max_scenes"),
                pose_source=str(spec.get("pose_source", "camera")),
            )
            if entries:
                path = output_dir / f"{name}_train.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_train"] = path
            continue

        if name == "scannetpp":
            for split in spec.get("train_splits", ["nvs_sem_train"]):
                entries = build_scannetpp_manifest_entries(
                    data_root=data_root,
                    relative_to=manifest_data_root,
                    split_files=[split],
                    chunk_size=clip_length,
                    overlap=overlap,
                    max_scenes_per_split=spec.get("max_scenes_per_split"),
                )
                if not entries:
                    continue
                path = output_dir / f"{name}_{split}.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_{split}"] = path
            continue

        if name == "virtual_kitti2":
            entries = build_virtual_kitti2_manifest_entries(
                data_root=data_root,
                relative_to=manifest_data_root,
                chunk_size=clip_length,
                overlap=overlap,
                scenes=spec.get("train_scenes"),
                variations=spec.get("variations"),
                cameras=spec.get("cameras"),
            )
            if entries:
                path = output_dir / f"{name}_train.jsonl"
                save_stitching_manifest(entries, path)
                manifest_paths[f"{name}_train"] = path
            continue

        raise ValueError(f"Unsupported dataset name in config: {name}")

    return manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSONL manifests for PLAN4D stitching pretraining")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stitching.yaml",
        help="Path to manifest build config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="manifests/stitching_81",
        help="Directory to save generated manifest jsonl files.",
    )
    parser.add_argument("--clip-length", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names to export")
    parser.add_argument(
        "--dynamic-replica-source-manifest",
        action="append",
        type=_parse_split_manifest_spec,
        default=None,
        metavar="SPLIT:PATH",
        help=(
            "Convert existing Dynamic Replica scene-flow manifests into PLAN4D stitching manifests. "
            "Pass one or more SPLIT:PATH values, for example train:/path/to/dynamic_replica_train.jsonl"
        ),
    )
    parser.add_argument(
        "--dynamic-replica-data-root",
        type=str,
        default="dynamic_stereo/dynamic_replica_data",
        help=(
            "Dynamic Replica dataset root, relative to manifest_data_root by default. "
            "Used together with --dynamic-replica-source-manifest."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(args.config, base=PROJECT_ROOT)
    output_dir = _resolve_path(args.output_dir, base=PROJECT_ROOT)
    config = load_yaml_config(config_path)

    manifest_data_root = _resolve_path(config["data"]["manifest_data_root"], base=PROJECT_ROOT)
    config["data"]["manifest_data_root"] = str(manifest_data_root)

    dataset_filter = [token.strip() for token in args.datasets.split(",") if token.strip()] if args.datasets else None
    manifest_paths = build_stitching_manifests(
        config["data"],
        output_dir=output_dir,
        clip_length=args.clip_length,
        overlap=args.overlap,
        datasets=dataset_filter,
    )

    if args.dynamic_replica_source_manifest:
        dynamic_replica_root = _resolve_path(args.dynamic_replica_data_root, base=manifest_data_root)
        manifest_paths.update(
            build_dynamic_replica_stitching_manifests(
                args.dynamic_replica_source_manifest,
                data_root=dynamic_replica_root,
                relative_to=manifest_data_root,
                output_dir=output_dir,
            )
        )

    if not manifest_paths:
        print("No manifests were generated.")
        return
    print("Generated manifests:")
    for name, path in manifest_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
