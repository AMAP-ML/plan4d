from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

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

from training.datasets.multi_scene_flow import (
    BlinkVisionSceneFlowAdapter,
    OmniWorldSceneFlowAdapter,
    VirtualKITTI2SceneFlowAdapter,
    build_dynamic_replica_manifest_entries,
    build_point_odyssey_manifest_entries,
    save_scene_flow_manifest,
)
from training.datasets.movi_f_scene_flow import build_movi_f_manifest_entries


def load_yaml_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_data_roots(data_cfg: Dict) -> None:
    if "data_root" in data_cfg and not Path(data_cfg["data_root"]).is_absolute():
        data_cfg["data_root"] = str((DA3_ROOT / data_cfg["data_root"]).resolve())
    for spec in data_cfg.get("datasets", []):
        if "data_root" in spec and not Path(spec["data_root"]).is_absolute():
            spec["data_root"] = str((DA3_ROOT / spec["data_root"]).resolve())
        if "builder_dir" in spec and not Path(spec["builder_dir"]).is_absolute():
            spec["builder_dir"] = str((DA3_ROOT / spec["builder_dir"]).resolve())


def _requested_dataset_names(datasets: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not datasets:
        return None
    return {str(name) for name in datasets}


def _build_entries_for_dataset(
    spec: Mapping[str, object],
    *,
    clip_length: int,
    overlap: int,
    split_kind: str,
    target_height: int,
    target_width: int,
) -> List:
    name = str(spec["name"])
    if name == "point_odyssey":
        return build_point_odyssey_manifest_entries(
            data_root=spec["data_root"],
            splits=spec.get("train_splits", [spec.get("split", "train")]) if split_kind == "train" else spec.get("val_splits", []),
            chunk_size=clip_length,
            overlap=overlap,
            max_scenes=spec.get("max_scenes") if split_kind == "train" else spec.get("max_val_scenes"),
        )
    if name == "blinkvision":
        return BlinkVisionSceneFlowAdapter.build_manifest_entries(
            data_root=spec["data_root"],
            splits=spec.get("train_splits", []) if split_kind == "train" else spec.get("val_splits", []),
            chunk_size=clip_length,
            overlap=overlap,
            target_height=target_height,
            target_width=target_width,
            max_scenes_per_split=spec.get("max_scenes_per_split") if split_kind == "train" else spec.get("max_val_scenes_per_split"),
            max_depth_m=float(spec.get("max_depth_m", 30.0)),
            rgb_dir=str(spec.get("rgb_dir", "clean_uint8")),
        )
    if name == "virtual_kitti2":
        return VirtualKITTI2SceneFlowAdapter.build_manifest_entries(
            data_root=spec["data_root"],
            chunk_size=clip_length,
            overlap=overlap,
            target_height=target_height,
            target_width=target_width,
            scenes=spec.get("train_scenes") if split_kind == "train" else spec.get("val_scenes"),
            variations=spec.get("variations"),
            cameras=spec.get("cameras"),
        )
    if name == "omniworld":
        return OmniWorldSceneFlowAdapter.build_manifest_entries(
            data_root=spec["data_root"],
            splits=spec.get("train_splits", []) if split_kind == "train" else spec.get("val_splits", []),
            chunk_size=clip_length,
            overlap=overlap,
            target_height=target_height,
            target_width=target_width,
            max_scenes=spec.get("max_scenes") if split_kind == "train" else spec.get("max_val_scenes"),
            pose_source=str(spec.get("pose_source", "camera")),
        )
    return []


def build_scene_flow_manifests(
    data_cfg: Mapping[str, object],
    *,
    output_dir: str | Path,
    clip_length: int = 81,
    overlap: int = 1,
    datasets: Optional[Sequence[str]] = None,
    point_odyssey_splits: Optional[Sequence[str]] = None,
    dynamic_replica_splits: Optional[Sequence[str]] = None,
    movi_f_splits: Optional[Sequence[str]] = None,
    point_odyssey_extra_splits: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = _requested_dataset_names(datasets)
    requested_point_odyssey_splits = [
        str(split)
        for split in (
            point_odyssey_splits
            if point_odyssey_splits is not None
            else point_odyssey_extra_splits or []
        )
    ]
    requested_dynamic_replica_splits = [str(split) for split in (dynamic_replica_splits or [])]
    requested_movi_f_splits = [str(split) for split in (movi_f_splits or [])]
    target_height = int(data_cfg.get("target_height", 0))
    target_width = int(data_cfg.get("target_width", 0))
    manifest_paths: Dict[str, Path] = {}

    for raw_spec in data_cfg.get("datasets", []):
        spec = dict(raw_spec)
        name = str(spec["name"])
        if requested is not None and name not in requested:
            continue

        if name == "point_odyssey":
            split_specs: List[tuple[str, Optional[int]]] = []
            seen_splits: set[str] = set()
            if requested_point_odyssey_splits:
                train_splits = {str(split) for split in spec.get("train_splits", [spec.get("split", "train")])}
                val_splits = {str(split) for split in spec.get("val_splits", [])}
                for split in requested_point_odyssey_splits:
                    if split in seen_splits:
                        continue
                    seen_splits.add(split)
                    max_scenes = spec.get("max_val_scenes") if split in val_splits and split not in train_splits else spec.get("max_scenes")
                    split_specs.append((split, max_scenes))
            else:
                if bool(spec.get("enabled_for_train", True)):
                    for split in spec.get("train_splits", [spec.get("split", "train")]):
                        split = str(split)
                        if split in seen_splits:
                            continue
                        seen_splits.add(split)
                        split_specs.append((split, spec.get("max_scenes")))
                if bool(spec.get("enabled_for_val", False)):
                    for split in spec.get("val_splits", []):
                        split = str(split)
                        if split in seen_splits:
                            continue
                        seen_splits.add(split)
                        split_specs.append((split, spec.get("max_val_scenes")))

            for split, max_scenes in split_specs:
                entries = build_point_odyssey_manifest_entries(
                    data_root=spec["data_root"],
                    splits=[split],
                    chunk_size=clip_length,
                    overlap=overlap,
                    max_scenes=max_scenes,
                )
                if not entries:
                    continue
                split_path = output_dir / f"{name}_{split}.jsonl"
                save_scene_flow_manifest(entries, split_path, data_root=spec["data_root"])
                manifest_paths[f"{name}_{split}"] = split_path
            continue

        if name == "dynamic_replica":
            split_specs: List[tuple[str, Optional[int]]] = []
            seen_splits: set[str] = set()
            if requested_dynamic_replica_splits:
                train_splits = {str(split) for split in spec.get("train_splits", [spec.get("split", "train")])}
                val_splits = {str(split) for split in spec.get("val_splits", [spec.get("split", "test")])}
                for split in requested_dynamic_replica_splits:
                    if split in seen_splits:
                        continue
                    seen_splits.add(split)
                    max_scenes = spec.get("max_val_scenes") if split in val_splits and split not in train_splits else spec.get("max_scenes")
                    split_specs.append((split, max_scenes))
            else:
                if bool(spec.get("enabled_for_train", True)):
                    for split in spec.get("train_splits", [spec.get("split", "train")]):
                        split = str(split)
                        if split in seen_splits:
                            continue
                        seen_splits.add(split)
                        split_specs.append((split, spec.get("max_scenes")))
                if bool(spec.get("enabled_for_val", False)):
                    for split in spec.get("val_splits", [spec.get("split", "test")]):
                        split = str(split)
                        if split in seen_splits:
                            continue
                        seen_splits.add(split)
                        split_specs.append((split, spec.get("max_val_scenes")))

            for split, max_scenes in split_specs:
                entries = build_dynamic_replica_manifest_entries(
                    data_root=spec["data_root"],
                    splits=[split],
                    chunk_size=clip_length,
                    overlap=overlap,
                    max_scenes=max_scenes,
                    camera_names=spec.get("camera_names"),
                )
                if not entries:
                    continue
                split_path = output_dir / f"{name}_{split}.jsonl"
                save_scene_flow_manifest(entries, split_path, data_root=spec["data_root"])
                manifest_paths[f"{name}_{split}"] = split_path
            continue

        if name == "movi_f":
            split_specs: List[tuple[str, Optional[int]]] = []
            seen_splits: set[str] = set()
            if requested_movi_f_splits:
                val_splits = {str(split) for split in spec.get("val_splits", [spec.get("split", "validation")])}
                for split in requested_movi_f_splits:
                    if split in seen_splits:
                        continue
                    seen_splits.add(split)
                    split_specs.append((split, spec.get("max_val_scenes") if split in val_splits else spec.get("max_scenes")))
            elif bool(spec.get("enabled_for_val", False)):
                for split in spec.get("val_splits", [spec.get("split", "validation")]):
                    split = str(split)
                    if split in seen_splits:
                        continue
                    seen_splits.add(split)
                    split_specs.append((split, spec.get("max_val_scenes")))

            for split, max_scenes in split_specs:
                entries = build_movi_f_manifest_entries(
                    builder_dir=spec["builder_dir"],
                    splits=[split],
                    chunk_size=clip_length,
                    max_scenes=max_scenes,
                    pad_to_chunk_size=bool(spec.get("pad_to_chunk_size", True)),
                )
                if not entries:
                    continue
                split_path = output_dir / f"{name}_{split}.jsonl"
                save_scene_flow_manifest(entries, split_path)
                manifest_paths[f"{name}_{split}"] = split_path
            continue

        if bool(spec.get("enabled_for_train", True)):
            train_entries = _build_entries_for_dataset(
                spec,
                clip_length=clip_length,
                overlap=overlap,
                split_kind="train",
                target_height=target_height,
                target_width=target_width,
            )
            if train_entries:
                train_path = output_dir / f"{name}_train.jsonl"
                save_scene_flow_manifest(train_entries, train_path, data_root=spec["data_root"])
                manifest_paths[f"{name}_train"] = train_path

        if bool(spec.get("enabled_for_val", False)):
            val_entries = _build_entries_for_dataset(
                spec,
                clip_length=clip_length,
                overlap=overlap,
                split_kind="val",
                target_height=target_height,
                target_width=target_width,
            )
            if val_entries:
                val_path = output_dir / f"{name}_val.jsonl"
                save_scene_flow_manifest(val_entries, val_path, data_root=spec["data_root"])
                manifest_paths[f"{name}_val"] = val_path

    return manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline JSONL manifests for scene-flow pretraining")
    parser.add_argument("--config", type=str, default="../configs/scene_flow.yaml")
    parser.add_argument("--output-dir", type=str, default="training/manifests/scene_flow_81")
    parser.add_argument("--clip-length", type=int, default=81)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument(
        "--point-odyssey-splits",
        dest="point_odyssey_splits",
        type=str,
        default=None,
        help="Comma-separated PointOdyssey splits to export as standalone manifests, e.g. sample,test",
    )
    parser.add_argument(
        "--point-odyssey-extra-splits",
        dest="point_odyssey_splits",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dynamic-replica-splits",
        type=str,
        default=None,
        help="Comma-separated Dynamic Replica splits to export as standalone manifests, e.g. train,valid",
    )
    parser.add_argument(
        "--movi-f-splits",
        type=str,
        default=None,
        help="Comma-separated MOVi-F splits to export as standalone manifests, e.g. validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (DA3_ROOT / config_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (DA3_ROOT / output_dir).resolve()

    config = load_yaml_config(config_path)
    resolve_data_roots(config["data"])
    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()] if args.datasets else None
    point_odyssey_splits = [
        token.strip() for token in args.point_odyssey_splits.split(",") if token.strip()
    ] if args.point_odyssey_splits else None
    dynamic_replica_splits = [
        token.strip() for token in args.dynamic_replica_splits.split(",") if token.strip()
    ] if args.dynamic_replica_splits else None
    movi_f_splits = [
        token.strip() for token in args.movi_f_splits.split(",") if token.strip()
    ] if args.movi_f_splits else None
    manifest_paths = build_scene_flow_manifests(
        config["data"],
        output_dir=output_dir,
        clip_length=int(args.clip_length),
        overlap=int(args.overlap),
        datasets=datasets,
        point_odyssey_splits=point_odyssey_splits,
        dynamic_replica_splits=dynamic_replica_splits,
        movi_f_splits=movi_f_splits,
    )
    for key, path in sorted(manifest_paths.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
