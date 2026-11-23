"""Annotate PLAN4D ti2v manifests with `has_gt_scene_flow`.

The ground-truth scene-flow availability is taken from the DA3 scene-flow
manifests under ``Depth-Anything-3/training/manifests/scene_flow_81``.
For every ti2v sample the scene-level key is matched against the SF manifest
sets; matches receive ``has_gt_scene_flow=true`` (plus ``scene_flow_source``
copied from the SF entry), non-matches receive ``false``.

By default the augmented manifests are written next to the originals with the
suffix ``.with_gt_sf.jsonl``. Pass ``--inplace`` to overwrite originals.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Repo roots
DA3_TRAIN = Path("./Depth-Anything-3/training")
DATA_ROOT = Path("./datasets")

# DA3 scene-flow manifests that define GT availability.
SF_MANIFESTS: Dict[str, Path] = {
    "blinkvision": DA3_TRAIN / "manifests/scene_flow_81/blinkvision_train.jsonl",
    "dynamic_replica": DA3_TRAIN / "manifests/scene_flow_81/dynamic_replica_train.jsonl",
    "omniworld": DA3_TRAIN / "manifests/scene_flow_81/omniworld_train.jsonl",
    "point_odyssey": DA3_TRAIN / "manifests/scene_flow_81/point_odyssey_train.jsonl",
    "virtual_kitti2": DA3_TRAIN / "manifests/scene_flow_81/virtual_kitti2_train.jsonl",
}

# PLAN4D ti2v manifests to be annotated.
TI2V_MANIFESTS: Dict[str, Path] = {
    "blinkvision": DATA_ROOT / "BlinkVision_train_extracted/BlinkVision_train_ti2v_caption_en.jsonl",
    "dynamic_replica": DATA_ROOT / "dynamic_stereo/dynamic_replica_data/dynamic_replica_train_valid_ti2v_caption_en_81f.jsonl",
    "omniworld": DATA_ROOT / "OmniWorld/manifests/omniworld_train_ti2v_81f.jsonl",
    "point_odyssey": DATA_ROOT / "point_odyssey/point_odyssey_49f_ti2v_caption_en.jsonl",
    "virtual_kitti2": DATA_ROOT / "Virtual_KITTI2/vkitti2_ti2v_caption_en.jsonl",
}


def _omniworld_scene_key(scene_id_or_path: str) -> str:
    """Strip the ``OmniWorld-Game-extracted/`` prefix if present and return the
    bare scene id (e.g., ``007b24c8269d``)."""
    raw = scene_id_or_path.strip().strip("/")
    parts = raw.split("/")
    return parts[-1]


def _ti2v_scene_key(dataset: str, entry: dict) -> Optional[str]:
    """Derive the key used to look up GT SF availability for a ti2v entry."""
    if dataset == "blinkvision":
        split = entry.get("split")
        scene = entry.get("scene_name")
        if split and scene:
            return f"{split}/{scene}"
        return None
    if dataset == "dynamic_replica":
        split = entry.get("split")
        scene = entry.get("scene_name")
        if split and scene:
            return f"{split}/{scene}"
        return None
    if dataset == "omniworld":
        sid = entry.get("scene_id") or entry.get("scene_relpath")
        if not sid:
            return None
        return _omniworld_scene_key(sid)
    if dataset == "point_odyssey":
        split = entry.get("split")
        scene = entry.get("scene_name")
        if split and scene:
            return f"{split}/{scene}"
        return None
    if dataset == "virtual_kitti2":
        scene = entry.get("scene_name")
        variation = entry.get("variation")
        camera = entry.get("camera_name")
        if scene and variation and camera:
            return f"{scene}/{variation}/{camera}"
        return None
    return None


def _sf_scene_key(dataset: str, entry: dict) -> Optional[str]:
    """Derive the matching key from an SF manifest entry."""
    if dataset == "omniworld":
        sid = entry.get("scene_id") or entry.get("scene_relpath")
        return _omniworld_scene_key(sid) if sid else None
    if dataset == "virtual_kitti2":
        # SF scene_id is "Scene/variation/Camera" (see DA3
        # ``VirtualKITTI2SceneFlowAdapter``: ``scene_id=f"{scene}/{variation}/{camera}"``)
        return entry.get("scene_id")
    # default: scene_relpath ("{split}/{scene_name}")
    return entry.get("scene_relpath")


def build_sf_lookup() -> Dict[str, Dict[str, dict]]:
    """For each dataset, build {scene_key: {has_gt_flow, flow_source, n_clips}}."""
    out: Dict[str, Dict[str, dict]] = {}
    for dataset, path in SF_MANIFESTS.items():
        scene_to_info: Dict[str, dict] = {}
        if not path.exists():
            print(f"[WARN] missing SF manifest: {path}")
            out[dataset] = scene_to_info
            continue
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if not entry.get("enabled_for_train", True):
                    continue
                if not entry.get("has_gt_flow", True):
                    continue
                key = _sf_scene_key(dataset, entry)
                if key is None:
                    continue
                slot = scene_to_info.setdefault(
                    key,
                    {"flow_source": entry.get("flow_source"), "n_clips": 0},
                )
                slot["n_clips"] += 1
        out[dataset] = scene_to_info
    return out


def annotate_manifest(
    dataset: str,
    src: Path,
    dst: Path,
    sf_lookup: Dict[str, dict],
) -> Tuple[int, int, int]:
    """Returns (total, n_with_gt, n_without_gt)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_pos = 0
    n_neg = 0
    n_no_key = 0
    with src.open() as fr, dst.open("w") as fw:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            entry = json.loads(line)
            key = _ti2v_scene_key(dataset, entry)
            if key is None:
                entry["has_gt_scene_flow"] = False
                entry["scene_flow_match_key"] = None
                n_no_key += 1
                n_neg += 1
            else:
                info = sf_lookup.get(key)
                entry["has_gt_scene_flow"] = info is not None
                entry["scene_flow_match_key"] = key
                if info is not None:
                    entry["scene_flow_source"] = info.get("flow_source")
                    n_pos += 1
                else:
                    n_neg += 1
            fw.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if n_no_key:
        print(f"  [{dataset}] {n_no_key} entries lacked the required key fields")
    return n_total, n_pos, n_neg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original ti2v manifests instead of writing to .with_gt_sf.jsonl",
    )
    ap.add_argument(
        "--suffix",
        default=".with_gt_sf.jsonl",
        help="Suffix (replaces .jsonl) for the augmented manifest. Ignored with --inplace.",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of datasets to process (default: all).",
    )
    args = ap.parse_args()

    requested = set(args.datasets) if args.datasets else set(TI2V_MANIFESTS)
    unknown = requested - set(TI2V_MANIFESTS)
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {sorted(unknown)}")

    print("[INFO] Building scene-flow availability lookup from DA3 manifests ...")
    sf_lookup = build_sf_lookup()
    for ds, scenes in sf_lookup.items():
        print(f"  - {ds}: {len(scenes)} GT-SF scenes")

    print("\n[INFO] Annotating ti2v manifests ...")
    summary = []
    for dataset, src in TI2V_MANIFESTS.items():
        if dataset not in requested:
            continue
        if not src.exists():
            print(f"  [WARN] missing ti2v manifest: {src}")
            continue
        if args.inplace:
            dst = src
        else:
            dst = src.with_suffix("")
            dst = src.parent / (src.stem + args.suffix)
        if args.inplace:
            tmp = src.with_suffix(src.suffix + ".tmp")
            total, pos, neg = annotate_manifest(dataset, src, tmp, sf_lookup[dataset])
            tmp.replace(src)
        else:
            total, pos, neg = annotate_manifest(dataset, src, dst, sf_lookup[dataset])
        rate = (pos / total * 100.0) if total else 0.0
        summary.append((dataset, total, pos, neg, rate, dst))
        print(f"  - {dataset}: {pos}/{total} ({rate:.1f}%) have GT SF -> {dst}")

    print("\n[INFO] Summary")
    print("dataset            total      gt_sf    no_gt_sf   coverage   output")
    for dataset, total, pos, neg, rate, dst in summary:
        print(f"  {dataset:<17} {total:>7}   {pos:>7}   {neg:>7}   {rate:>6.1f}%   {dst}")


if __name__ == "__main__":
    main()
