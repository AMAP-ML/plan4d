from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STITCHING_MANIFEST_DIR = PROJECT_ROOT / "manifests" / "stitching_81"
DEFAULT_SCENE_FLOW_MANIFEST_DIR = (
    PROJECT_ROOT.parent
    / "plan4d"
    / "Depth-Anything-3"
    / "training"
    / "manifests"
    / "scene_flow_81"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "dataset"


@dataclass(frozen=True)
class AnnotationSummary:
    files_seen: int = 0
    rows_seen: int = 0
    updated_rows: int = 0
    unchanged_available_rows: int = 0
    removed_stale_rows: int = 0
    unmatched_rows: int = 0
    available_scenes: int = 0


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp_path.replace(path)


def _scene_key(row: Mapping) -> tuple[str, str]:
    return str(row.get("dataset_name", "")), str(row.get("scene_id", ""))


def _dataset_root(dataset_name: str, row: Mapping, data_root: Path) -> Path:
    adapter_kwargs = row.get("adapter_kwargs") if isinstance(row.get("adapter_kwargs"), Mapping) else {}
    if dataset_name == "point_odyssey":
        return data_root / "point_odyssey" / "dataset"
    if dataset_name == "blinkvision":
        return data_root / "BlinkVision_train_extracted"
    if dataset_name == "omniworld":
        return data_root / "OmniWorld" / "OmniWorld-Game-extracted"
    if dataset_name == "virtual_kitti2":
        return data_root / "Virtual_KITTI2"
    if dataset_name == "dynamic_replica":
        explicit = adapter_kwargs.get("data_root")
        if explicit:
            return Path(str(explicit)).expanduser()
        return data_root / "dynamic_stereo" / "dynamic_replica_data"
    return data_root


def _has_any_file(path: Path) -> bool:
    return path.exists() and any(child.is_file() for child in path.iterdir())


def light_probe_scene_flow(row: Mapping, *, data_root: str | Path = DEFAULT_DATA_ROOT) -> bool:
    dataset_name = str(row.get("dataset_name", ""))
    scene_relpath = row.get("scene_relpath")
    if not scene_relpath:
        return False
    root = _dataset_root(dataset_name, row, Path(data_root).expanduser())
    scene_dir = root / str(scene_relpath)
    adapter_kwargs = row.get("adapter_kwargs") if isinstance(row.get("adapter_kwargs"), Mapping) else {}

    if dataset_name == "point_odyssey":
        return (
            (scene_dir / "anno.npz").exists()
            and (scene_dir / "rgbs").exists()
            and (scene_dir / "depths").exists()
        )
    if dataset_name == "blinkvision":
        rgb_dir = scene_dir / str(adapter_kwargs.get("rgb_dir", "clean_uint8"))
        return (
            (scene_dir / "poses.npz").exists()
            and rgb_dir.exists()
            and (scene_dir / "depth").exists()
            and (scene_dir / "forward_flow").exists()
        )
    if dataset_name == "omniworld":
        split_idx = int(adapter_kwargs.get("split_idx", row.get("source_split", "split_0").split("_")[-1]))
        pose_source = str(adapter_kwargs.get("pose_source", "camera"))
        pose_dir = scene_dir / pose_source
        return (
            (scene_dir / "color").exists()
            and (scene_dir / "depth").exists()
            and (scene_dir / "flow").exists()
            and (pose_dir / f"split_{split_idx}.json").exists()
        )
    if dataset_name == "virtual_kitti2":
        camera = str(adapter_kwargs.get("camera", "Camera_0"))
        return (
            (scene_dir / "intrinsic.txt").exists()
            and (scene_dir / "extrinsic.txt").exists()
            and (scene_dir / "frames" / "rgb" / camera).exists()
            and (scene_dir / "frames" / "depth" / camera).exists()
            and _has_any_file(scene_dir / "frames" / "forwardSceneFlow" / camera)
        )
    if dataset_name == "dynamic_replica":
        return scene_dir.exists() or (root / str(row.get("source_split", "")) / str(row.get("scene_id", ""))).exists()
    return False


def load_probeable_scene_keys(
    scene_flow_manifest_dir: str | Path,
    *,
    probe_fn: Callable[[Mapping], bool] | None = None,
) -> set[tuple[str, str]]:
    scene_flow_manifest_dir = Path(scene_flow_manifest_dir).expanduser()
    probe_fn = light_probe_scene_flow if probe_fn is None else probe_fn
    representative_rows: dict[tuple[str, str], Mapping] = {}
    for manifest_path in sorted(scene_flow_manifest_dir.glob("*.jsonl")):
        for row in _read_jsonl(manifest_path):
            key = _scene_key(row)
            if all(key):
                representative_rows.setdefault(key, row)
    return {key for key, row in representative_rows.items() if probe_fn(row)}


def annotate_manifests(
    *,
    stitching_manifest_dir: str | Path = DEFAULT_STITCHING_MANIFEST_DIR,
    scene_flow_manifest_dir: str | Path = DEFAULT_SCENE_FLOW_MANIFEST_DIR,
    probe_fn: Callable[[Mapping], bool] | None = None,
    dry_run: bool = False,
) -> AnnotationSummary:
    stitching_manifest_dir = Path(stitching_manifest_dir).expanduser()
    available_scene_keys = load_probeable_scene_keys(scene_flow_manifest_dir, probe_fn=probe_fn)

    files_seen = rows_seen = updated_rows = unchanged_available_rows = removed_stale_rows = unmatched_rows = 0
    for manifest_path in sorted(stitching_manifest_dir.glob("*.jsonl")):
        files_seen += 1
        rows = _read_jsonl(manifest_path)
        changed = False
        for row in rows:
            rows_seen += 1
            key = _scene_key(row)
            if key in available_scene_keys:
                if row.get("scene_flow_available") is True:
                    unchanged_available_rows += 1
                else:
                    row["scene_flow_available"] = True
                    updated_rows += 1
                    changed = True
            else:
                unmatched_rows += 1
                if "scene_flow_available" in row:
                    row.pop("scene_flow_available", None)
                    removed_stale_rows += 1
                    changed = True
        if changed and not dry_run:
            _write_jsonl(manifest_path, rows)

    return AnnotationSummary(
        files_seen=files_seen,
        rows_seen=rows_seen,
        updated_rows=updated_rows,
        unchanged_available_rows=unchanged_available_rows,
        removed_stale_rows=removed_stale_rows,
        unmatched_rows=unmatched_rows,
        available_scenes=len(available_scene_keys),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate PLAN4D stitching manifests with scene_flow_available.")
    parser.add_argument("--stitching-manifest-dir", type=Path, default=DEFAULT_STITCHING_MANIFEST_DIR)
    parser.add_argument("--scene-flow-manifest-dir", type=Path, default=DEFAULT_SCENE_FLOW_MANIFEST_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = annotate_manifests(
        stitching_manifest_dir=args.stitching_manifest_dir,
        scene_flow_manifest_dir=args.scene_flow_manifest_dir,
        probe_fn=lambda row: light_probe_scene_flow(row, data_root=args.data_root),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
