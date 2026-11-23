#!/usr/bin/env python3
"""Append native camera frustums and connected trajectories to final global clouds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import open3d as o3d
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plan4d_final_outputs"))
from export_gen3r_pcd_with_cameras import infer_image_sizes, load_gen3r_cameras
from export_pcd_with_cameras import load_camera_arrays, resolve_image_sizes

DA3_SRC = ROOT / "Depth-Anything-3" / "src"
sys.path.insert(0, str(DA3_SRC))
from depth_anything_3.bench.utils import (
    _camera_centers,
    _index_color_rgb,
    _sample_segments,
    build_point_cloud_with_cameras,
)


OUTPUT_NAME = "global_pcd_with_camera_trajectory.ply"
MANIFEST_NAME = "global_pcd_with_camera_trajectory_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_sequence(value, tail_shapes: tuple[tuple[int, int], ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-2:] not in tail_shapes:
        raise ValueError(f"Invalid camera shape: {array.shape}")
    return array


def npz_array_shape(path: Path, name: str) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive, archive.open(f"{name}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _, _ = reader(stream)
    return shape


def resolve_inputs(method: str, case_dir: Path) -> dict:
    if method == "gen3r":
        source = case_dir / "pcds.ply"
        camera_sources = [case_dir / "cameras.json"]
        intrinsics, extrinsics = load_gen3r_cameras(camera_sources[0])
        image_sizes = infer_image_sizes(intrinsics)
        coordinate_space = "Gen3R/VGGT world"
    elif method == "plan4d":
        source = case_dir / "point_cloud/merged.ply"
        camera_sources = [case_dir / "decode_outputs.pt"]
        bundle = torch.load(camera_sources[0], map_location="cpu", weights_only=True, mmap=True)
        intrinsics = camera_sequence(bundle["intrinsics_px"], ((3, 3),))
        extrinsics = camera_sequence(bundle["extrinsic_w2c"], ((3, 4), (4, 4)))
        height, width = bundle["depth"].shape[-3:-1]
        image_sizes = [(int(height), int(width))] * len(intrinsics)
        coordinate_space = "Native world"
    elif method == "best" or (case_dir / "point_cloud/merged.ply").is_file():
        source = case_dir / "point_cloud/merged.ply"
        camera_sources = [case_dir / f"pred_camera_{kind}_raw.pt" for kind in ("intrinsics", "extrinsics")]
        intrinsics, extrinsics = load_camera_arrays(case_dir)
        image_sizes = resolve_image_sizes(case_dir, len(intrinsics))
        if image_sizes is None:
            raise ValueError(f"Missing native camera image dimensions: {case_dir}")
        coordinate_space = "Plan4D native world" if method == "best" else "TesserAct pseudo-depth/PnP world"
    elif method == "tesseract":
        geometry_dir = case_dir / "da3_geometry"
        source = geometry_dir / "reconstruction_pcd.ply"
        camera_sources = [geometry_dir / "exports/raw_camera_params.npz"]
        with np.load(camera_sources[0]) as bundle:
            intrinsics = camera_sequence(bundle["intrinsics"], ((3, 3),))
            extrinsics = camera_sequence(bundle["extrinsics"], ((3, 4), (4, 4)))
        depth_shape = npz_array_shape(geometry_dir / "exports/mini_npz/results.npz", "depth")
        image_sizes = [tuple(depth_shape[-2:])] * len(intrinsics)
        coordinate_space = "DA3 postprocessed world (TesserAct RGB)"
    else:
        raise ValueError(f"Unsupported method: {method}")

    if not source.is_file():
        raise FileNotFoundError(source)
    if len(intrinsics) == 0 or len(intrinsics) != len(extrinsics):
        raise ValueError(f"Camera count mismatch in {case_dir}")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError(f"Nonfinite camera matrices in {case_dir}")
    if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
        raise ValueError(f"Invalid focal lengths in {case_dir}")
    return {
        "source": source,
        "camera_sources": camera_sources,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_sizes": image_sizes,
        "coordinate_space": coordinate_space,
    }


def trajectory_cloud(centers: np.ndarray, samples: int) -> o3d.geometry.PointCloud:
    if len(centers) > 1:
        segments = np.stack([centers[:-1], centers[1:]], axis=1)
        points = _sample_segments(segments, samples)
        colors = np.repeat(
            np.stack([_index_color_rgb(index, len(centers)) for index in range(len(centers) - 1)]),
            samples,
            axis=0,
        )
    else:
        points = centers.copy()
        colors = _index_color_rgb(0, 1)[None]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def choose_frustum_scale(points: np.ndarray, centers: np.ndarray) -> float:
    sample = points[::max(1, len(points) // 100000)]
    lower, upper = np.percentile(sample, [1, 99], axis=0)
    extent_scale = float(np.linalg.norm(upper - lower)) * 0.025
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    positive_steps = steps[steps > 1e-6]
    motion_scale = float(np.median(positive_steps)) * 0.25 if len(positive_steps) else 0.0
    return max(extent_scale, motion_scale, 1e-4)


def validate_cloud(path: Path, source, centers, samples: int, expected_points: int) -> dict:
    restored = o3d.io.read_point_cloud(str(path))
    points = np.asarray(restored.points)
    source_points = np.asarray(source.points)
    if len(points) != expected_points or not np.isfinite(points).all():
        raise ValueError(f"Invalid output geometry in {path}: {len(points)} vs {expected_points}")
    if not np.array_equal(points[:len(source_points)], source_points):
        raise ValueError(f"Source geometry changed in {path}")
    source_rgb = np.rint(np.asarray(source.colors) * 255).astype(np.uint8)
    output_rgb = np.rint(np.asarray(restored.colors)[:len(source_points)] * 255).astype(np.uint8)
    if not np.array_equal(source_rgb, output_rgb):
        raise ValueError(f"Source colors changed in {path}")
    frustum_centers = points[len(source_points):len(source_points) + len(centers) * 8 * samples:8 * samples]
    if not np.allclose(frustum_centers, centers, rtol=0, atol=1e-12):
        raise ValueError(f"Camera centers changed in {path}")
    trajectory_start = len(source_points) + len(centers) * 8 * samples
    if len(centers) > 1:
        segments = points[trajectory_start:].reshape(len(centers) - 1, samples, 3)
        np.testing.assert_allclose(segments[:, 0], centers[:-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(segments[:, -1], centers[1:], rtol=0, atol=1e-12)
    else:
        np.testing.assert_allclose(points[trajectory_start:], centers, rtol=0, atol=1e-12)
    return {"readable": True, "all_points_finite": True, "source_xyz_rgb_preserved": True,
            "camera_centers_match": True, "trajectory_endpoints_match": True}


def export_case(method: str, case_dir: Path, *, samples: int = 32, verify_only: bool = False) -> dict:
    if samples < 2:
        raise ValueError("samples must be at least 2")
    inputs = resolve_inputs(method, case_dir)
    source_path = inputs["source"]
    output_path = case_dir / "point_cloud" / OUTPUT_NAME
    manifest_path = output_path.with_name(MANIFEST_NAME)
    source_hash = sha256(source_path)
    source = o3d.io.read_point_cloud(str(source_path))
    source_points = np.asarray(source.points)
    if len(source_points) == 0 or not source.has_colors() or not np.isfinite(source_points).all():
        raise ValueError(f"Empty, uncolored or nonfinite source cloud: {source_path}")
    centers = _camera_centers(inputs["extrinsics"])
    frame_count = len(centers)
    trajectory_count = (frame_count - 1) * samples if frame_count > 1 else 1
    expected_points = len(source_points) + frame_count * 8 * samples + trajectory_count
    frustum_scale = choose_frustum_scale(source_points, centers)
    existing = output_path.exists()
    if verify_only and not existing:
        raise FileNotFoundError(output_path)
    if existing:
        if not manifest_path.is_file():
            raise FileExistsError(f"Existing output without provenance manifest: {output_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_pcd_sha256"] != source_hash or manifest["samples_per_segment"] != samples:
            raise ValueError(f"Existing export has different inputs/settings: {output_path}")
        for camera_path in inputs["camera_sources"]:
            if manifest["camera_sources_sha256"].get(str(camera_path)) != sha256(camera_path):
                raise ValueError(f"Camera source changed since export: {camera_path}")
        if manifest["output_sha256"] != sha256(output_path):
            raise ValueError(f"Existing export checksum mismatch: {output_path}")
    else:
        if manifest_path.exists():
            raise FileExistsError(manifest_path)
        source_hashes = {str(path): sha256(path) for path in inputs["camera_sources"]}
        visualization = build_point_cloud_with_cameras(
            source, intrinsics=inputs["intrinsics"], extrinsics=inputs["extrinsics"],
            image_sizes=inputs["image_sizes"], frustum_scale=frustum_scale,
            samples_per_segment=samples,
        )
        visualization += trajectory_cloud(centers, samples)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output_path), visualization, write_ascii=False):
            raise OSError(f"Could not write {output_path}")
        del visualization
        manifest = {
            "method": method, "case": case_dir.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_pcd": str(source_path), "source_pcd_sha256": source_hash,
            "camera_sources_sha256": source_hashes,
            "output_path": str(output_path), "output_sha256": sha256(output_path),
            "coordinate_space": inputs["coordinate_space"], "extrinsics_convention": "world_to_camera",
            "point_transform": "none; preserve the existing global cloud",
            "frame_count": frame_count, "image_sizes_hw": inputs["image_sizes"],
            "source_point_count": len(source_points), "frustum_point_count": frame_count * 8 * samples,
            "trajectory_point_count": trajectory_count, "output_point_count": expected_points,
            "samples_per_segment": samples, "frustum_scale": frustum_scale,
            "camera_centers_world": centers.tolist(),
            "trajectory_length": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
            "color_order": "time-ordered HSV colors, matching existing DA3 camera visualization",
            "usage": "visualization; camera vertices are excluded from geometry metrics",
        }
    validation = validate_cloud(output_path, source, centers, samples, expected_points)
    if not existing:
        if sha256(source_path) != source_hash:
            raise ValueError(f"Source cloud changed during export: {source_path}")
        manifest["validation"] = validation
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return {"method": method, "case": case_dir.name, "status": "verified_existing" if existing else "exported",
            "output_path": str(output_path), "source_pcd": str(source_path),
            "source_point_count": len(source_points), "output_point_count": expected_points,
            "frame_count": frame_count, "output_bytes": output_path.stat().st_size,
            "manifest_path": str(manifest_path), "validation": validation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-root", type=Path, default=ROOT / "plan4d_final_outputs-best")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "plan4d_final_outputs-baselines")
    parser.add_argument("--methods", nargs="+", choices=("best", "gen3r", "tesseract", "plan4d"),
                        default=["best", "gen3r", "tesseract", "plan4d"])
    parser.add_argument("--case", help="Process one exact case name in each selected method.")
    parser.add_argument("--samples-per-segment", type=int, default=32)
    parser.add_argument("--verify-only", action="store_true", help="Validate existing outputs without writing PLYs.")
    parser.add_argument("--report", type=Path, help="New JSON report path; existing reports are never overwritten.")
    args = parser.parse_args()
    if args.samples_per_segment < 2:
        parser.error("--samples-per-segment must be at least 2")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    records = []
    for method in args.methods:
        group_root = args.best_root if method == "best" else args.baseline_root / method
        cases = [path for path in sorted(group_root.iterdir()) if path.is_dir() and not path.name.startswith("_")]
        for case_dir in cases:
            if args.case and case_dir.name != args.case:
                continue
            try:
                record = export_case(method, case_dir, samples=args.samples_per_segment, verify_only=args.verify_only)
            except Exception as error:
                record = {"method": method, "case": case_dir.name, "status": "error",
                          "error": f"{type(error).__name__}: {error}"}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {"case_count": len(records), "status_counts": dict(Counter(row["status"] for row in records)),
               "method_counts": dict(Counter(row["method"] for row in records)),
               "total_output_bytes": sum(row.get("output_bytes", 0) for row in records)}
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"summary": summary, "cases": records}, indent=2, ensure_ascii=False) + "\n")
    return 1 if not records or any(row["status"] == "error" for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
