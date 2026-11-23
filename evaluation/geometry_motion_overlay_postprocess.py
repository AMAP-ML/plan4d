from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_ROOT = PLAN4D_ROOT / "Depth-Anything-3"

for import_path in (PROJECT_ROOT, DA3_ROOT, DA3_ROOT / "src", PLAN4D_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


DEFAULT_MODES = ("posed", "unposed")
GT_MODE_ALIASES = {"gt", "gt_fuse", "fuse_gt"}
SUPPORTED_SCENE_FLOW_OUTPUT_SPACES = ("auto", "native", "metric")
SUPPORTED_FLOW_KEYS = ("scene_flow", "scene_flow_cam")
_RGB_RESULT_CACHE: dict[Path, np.ndarray | None] = {}


@dataclass(frozen=True)
class GeometryMotionOptions:
    modes: tuple[str, ...] = DEFAULT_MODES
    flow_key: str = "scene_flow"
    scene_flow_output_space: str = "auto"
    flow_scale: float = 1.0
    visual_flow_scale: float = 1.0
    output_subdir: str = "geometry_motion_overlay"
    overwrite: bool = False
    edge_threshold: float = 0.08
    border_crop: int = 5
    motion_quantile: float = 85.0
    motion_min_abs: float = 0.0
    max_points_per_frame: int = 30_000
    max_segments_per_frame: int = 1_500
    static_darken: float = 0.75
    deformed_darken: float = 0.6
    seed: int = 42
    views: int = 6
    cols: int = 0
    dpi: int = 150
    figsize: tuple[float, float] = (6.0, 6.0)
    point_size: float = 0.4
    line_width: float = 0.8
    segment_interp_steps: int = 5
    include_deformed_points: bool = False


@dataclass(frozen=True)
class AggregatedGeometryMotion:
    static_pts: np.ndarray
    static_rgb: np.ndarray
    dynamic_pts: np.ndarray
    dynamic_rgb: np.ndarray
    dynamic_flow: np.ndarray
    deformed_pts: np.ndarray
    segments: np.ndarray
    seg_colors: np.ndarray
    static_frame_idx: np.ndarray
    dynamic_frame_idx: np.ndarray
    segment_frame_idx: np.ndarray
    center: np.ndarray
    axis_limits: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class GeometryMotionOutputRecord:
    mode: str
    source_path: Path
    image_path: Path
    ply_path: Path
    summary_path: Path


@dataclass(frozen=True)
class GeometryMotionProcessResult:
    records: tuple[GeometryMotionOutputRecord, ...]
    skipped_modes: dict[str, int]
    errors: tuple[str, ...]


def _normalize_mode_name(mode: str) -> str:
    return str(mode).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(_normalize_mode_name(item) for item in str(value).split(",") if item.strip())
    if not modes:
        raise argparse.ArgumentTypeError("modes cannot be empty")
    return modes


def _as_scalar_float(value: object, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _as_depth_thw(depth: np.ndarray) -> np.ndarray:
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Expected depth shape [T,H,W], [T,1,H,W], or [T,H,W,1], got {array.shape}")


def _as_flow_thwc(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"Expected scene flow shape [T,3,H,W] or [T,H,W,3], got {array.shape}")
    if array.shape[1] == 3:
        return np.transpose(array, (0, 2, 3, 1)).copy()
    if array.shape[-1] == 3:
        return array
    raise ValueError(f"Expected scene-flow channel dimension of 3, got {array.shape}")


def _as_frame_matrices(matrices: np.ndarray, num_frames: int, name: str) -> np.ndarray:
    array = np.asarray(matrices, dtype=np.float32)
    if array.ndim == 2:
        array = np.broadcast_to(array[None], (num_frames, *array.shape)).copy()
    if array.ndim != 3 or array.shape[0] != num_frames:
        raise ValueError(f"Expected {name} to have {num_frames} frames, got {array.shape}")
    return array


def _as_homogeneous_extrinsics(extrinsics: np.ndarray, num_frames: int) -> np.ndarray:
    array = _as_frame_matrices(extrinsics, num_frames, "extrinsics")
    if array.shape[1:] == (4, 4):
        return array
    if array.shape[1:] != (3, 4):
        raise ValueError(f"Expected extrinsics shape [T,3,4] or [T,4,4], got {array.shape}")
    homogeneous = np.broadcast_to(np.eye(4, dtype=np.float32), (num_frames, 4, 4)).copy()
    homogeneous[:, :3, :4] = array
    return homogeneous


def _resize_rgb_thwc(rgb: np.ndarray, *, height: int, width: int) -> np.ndarray:
    return np.stack(
        [
            cv2.resize(frame[..., :3], (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
            for frame in np.asarray(rgb)
        ],
        axis=0,
    )


def _as_rgb_thwc(rgb: np.ndarray | None, *, num_frames: int, height: int, width: int) -> np.ndarray:
    if rgb is None:
        return np.full((num_frames, height, width, 3), 200, dtype=np.uint8)
    array = np.asarray(rgb)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"Expected rgb shape [T,H,W,3] or [T,3,H,W], got {array.shape}")
    if array.shape[0] != num_frames:
        raise ValueError(f"Expected rgb to have {num_frames} frames, got {array.shape}")
    if array.shape[1] == 3:
        array = np.transpose(array, (0, 2, 3, 1))
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Expected RGB channel dimension of 3, got {array.shape}")
    if array.shape[1:3] != (height, width):
        array = _resize_rgb_thwc(array, height=height, width=width)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        scale = 255.0 if finite.size == 0 or float(np.nanmax(finite)) <= 1.0 else 1.0
        array = np.rint(array.astype(np.float32) * np.float32(scale))
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _as_mask_thw(mask: np.ndarray | None, *, num_frames: int, height: int, width: int) -> np.ndarray | None:
    if mask is None:
        return None
    array = _as_depth_thw(np.asarray(mask))
    if array.shape != (num_frames, height, width):
        raise ValueError(f"Expected mask shape {(num_frames, height, width)}, got {array.shape}")
    return array > 0.5


def _camera_points_from_z_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape [3,3], got {intrinsic.shape}")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("intrinsic focal length must be non-zero")
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    yy, xx = np.indices(depth.shape, dtype=np.float32)
    z = np.asarray(depth, dtype=np.float32)
    x = (xx - cx) * z / np.float32(fx)
    y = (yy - cy) * z / np.float32(fy)
    return np.stack((x, y, z), axis=-1)


def _transform_camera_to_world(points_cam: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(np.asarray(extrinsic_w2c, dtype=np.float32))
    flat_points = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate((flat_points, ones), axis=1)
    points_world = points_h @ c2w.T
    return points_world[:, :3].reshape(points_cam.shape)


def _transform_world_to_camera(points_world: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    flat_points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate((flat_points, ones), axis=1)
    points_cam = points_h @ np.asarray(extrinsic_w2c, dtype=np.float32).T
    return points_cam[:, :3].reshape(points_world.shape)


def build_point_map_world(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    depth_thw = _as_depth_thw(depth)
    num_frames = depth_thw.shape[0]
    intrinsics_t = _as_frame_matrices(intrinsics, num_frames, "intrinsics")
    extrinsics_t = _as_homogeneous_extrinsics(extrinsics, num_frames)
    point_maps: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        points_cam = _camera_points_from_z_depth(depth_thw[frame_idx], intrinsics_t[frame_idx])
        point_maps.append(_transform_camera_to_world(points_cam, extrinsics_t[frame_idx]))
    return np.stack(point_maps, axis=0).astype(np.float32)


def build_point_map_frame0(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    depth_thw = _as_depth_thw(depth)
    num_frames = depth_thw.shape[0]
    intrinsics_t = _as_frame_matrices(intrinsics, num_frames, "intrinsics")
    extrinsics_t = _as_homogeneous_extrinsics(extrinsics, num_frames)
    frame0_w2c = extrinsics_t[0]
    point_maps: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        points_cam = _camera_points_from_z_depth(depth_thw[frame_idx], intrinsics_t[frame_idx])
        points_world = _transform_camera_to_world(points_cam, extrinsics_t[frame_idx])
        point_maps.append(_transform_world_to_camera(points_world, frame0_w2c))
    return np.stack(point_maps, axis=0).astype(np.float32)


def _camera_flow_to_frame0(scene_flow_cam: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    flow = _as_flow_thwc(scene_flow_cam)
    extrinsics_t = _as_homogeneous_extrinsics(extrinsics, flow.shape[0])
    rot_frame0 = extrinsics_t[0, :3, :3].astype(np.float32)
    converted = np.empty_like(flow, dtype=np.float32)
    for frame_idx in range(flow.shape[0]):
        rot_current = extrinsics_t[frame_idx, :3, :3].astype(np.float32)
        rotation = rot_frame0 @ rot_current.T
        converted[frame_idx] = np.einsum("ij,hwj->hwi", rotation, flow[frame_idx]).astype(np.float32)
    return converted


def point_map_edge_mask(point_map: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    points = np.asarray(point_map, dtype=np.float32)
    dx = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=-1)
    dy = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode="edge")
    dy = np.pad(dy, ((0, 1), (0, 0)), mode="edge")
    return (dx > float(threshold)) | (dy > float(threshold))


def compute_valid_mask(
    point_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    edge_threshold: float = 0.08,
    border_crop: int = 5,
) -> np.ndarray:
    points = np.asarray(point_map, dtype=np.float32)
    mask = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points, axis=-1) > 1e-8)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    if edge_threshold > 0:
        mask &= ~point_map_edge_mask(points, threshold=float(edge_threshold))
    crop = int(border_crop)
    if crop > 0 and min(mask.shape) > crop * 2:
        mask[:crop, :] = False
        mask[-crop:, :] = False
        mask[:, :crop] = False
        mask[:, -crop:] = False
    elif crop > 0:
        mask[:, :] = False
    return mask


def compute_motion_mask(
    scene_flow: np.ndarray,
    valid_mask: np.ndarray,
    *,
    quantile: float = 85.0,
    min_abs: float = 0.0,
) -> np.ndarray:
    flow = np.asarray(scene_flow, dtype=np.float32)
    magnitudes = np.linalg.norm(flow, axis=-1)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(magnitudes)
    if not np.any(valid):
        return np.zeros_like(valid, dtype=bool)
    finite_magnitudes = magnitudes[valid]
    threshold = max(float(np.percentile(finite_magnitudes, float(quantile))), float(min_abs))
    return valid & (magnitudes >= threshold) & (magnitudes > 1e-12)


def scene_flow_to_rgb(flow: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
    vectors = np.asarray(flow, dtype=np.float32).reshape(-1, 3)
    magnitudes = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(magnitudes) & (magnitudes > 1e-12)
    if np.any(valid):
        scale = float(np.percentile(magnitudes[valid], float(clip_percentile)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.max(magnitudes[valid]))
    else:
        scale = 1.0
    scale = max(scale, 1e-6)
    direction = vectors / np.maximum(magnitudes[:, None], 1e-6)
    hue = (np.arctan2(direction[:, 1], direction[:, 0]) + np.pi) / (2.0 * np.pi)
    value = np.clip(magnitudes / scale, 0.0, 1.0)
    rgb = _hsv_to_rgb(hue, np.ones_like(value, dtype=np.float32), value)
    rgb[~valid] = 0.0
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray, value: np.ndarray) -> np.ndarray:
    hue = np.mod(np.asarray(hue, dtype=np.float32), 1.0)
    saturation = np.clip(np.asarray(saturation, dtype=np.float32), 0.0, 1.0)
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    sector = np.floor(hue * 6.0).astype(np.int32)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector_mod = sector % 6
    rgb = np.empty((hue.shape[0], 3), dtype=np.float32)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for idx, channels in enumerate(choices):
        mask = sector_mod == idx
        if np.any(mask):
            rgb[mask, 0] = channels[0][mask]
            rgb[mask, 1] = channels[1][mask]
            rgb[mask, 2] = channels[2][mask]
    return rgb


def _sample_rows(
    *arrays: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    if not arrays:
        return ()
    count = arrays[0].shape[0]
    if max_count <= 0 or count <= max_count:
        return arrays
    indices = rng.choice(count, size=int(max_count), replace=False)
    return tuple(array[indices] for array in arrays)


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float32)


def aggregate_all_frames(
    *,
    point_map: np.ndarray,
    scene_flow: np.ndarray | None,
    rgbs: np.ndarray | None,
    valid_mask_raw: np.ndarray | None,
    deform_mask_raw: np.ndarray | None,
    edge_threshold: float = 0.08,
    border_crop: int = 5,
    motion_quantile: float = 85.0,
    motion_min_abs: float = 0.0,
    visual_flow_scale: float = 1.0,
    max_points_per_frame: int = 30_000,
    max_segments_per_frame: int = 1_500,
    static_darken: float = 0.75,
    deformed_darken: float = 0.6,
    seed: int = 42,
) -> AggregatedGeometryMotion:
    points = np.asarray(point_map, dtype=np.float32)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected point_map shape [T,H,W,3], got {points.shape}")
    num_frames, height, width = points.shape[:3]
    flow = None if scene_flow is None else _as_flow_thwc(scene_flow)
    if flow is not None and flow.shape != points.shape:
        raise ValueError(f"scene_flow shape {flow.shape} does not match point_map shape {points.shape}")
    rgb_thwc = _as_rgb_thwc(rgbs, num_frames=num_frames, height=height, width=width)
    valid_mask_thw = _as_mask_thw(valid_mask_raw, num_frames=num_frames, height=height, width=width)
    deform_mask_thw = _as_mask_thw(deform_mask_raw, num_frames=num_frames, height=height, width=width)
    rng = np.random.default_rng(int(seed))

    valid_point_chunks: list[np.ndarray] = []
    per_frame_valid: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        raw_valid = None if valid_mask_thw is None else valid_mask_thw[frame_idx]
        valid = compute_valid_mask(points[frame_idx], raw_valid, edge_threshold, border_crop)
        if deform_mask_thw is not None:
            valid &= deform_mask_thw[frame_idx]
        per_frame_valid.append(valid)
        if np.any(valid):
            valid_point_chunks.append(points[frame_idx][valid])
    center = (
        np.concatenate(valid_point_chunks, axis=0).mean(axis=0).astype(np.float32)
        if valid_point_chunks
        else np.zeros(3, dtype=np.float32)
    )

    static_pts_list: list[np.ndarray] = []
    static_rgb_list: list[np.ndarray] = []
    dynamic_pts_list: list[np.ndarray] = []
    dynamic_rgb_list: list[np.ndarray] = []
    dynamic_flow_list: list[np.ndarray] = []
    deformed_pts_list: list[np.ndarray] = []
    segment_list: list[np.ndarray] = []
    seg_color_list: list[np.ndarray] = []
    static_frame_idx_list: list[np.ndarray] = []
    dynamic_frame_idx_list: list[np.ndarray] = []
    segment_frame_idx_list: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        valid = per_frame_valid[frame_idx]
        rgb = rgb_thwc[frame_idx]
        if flow is None:
            pts = points[frame_idx][valid] - center
            colors = rgb[valid]
            pts, colors = _sample_rows(pts, colors, max_count=int(max_points_per_frame), rng=rng)
            static_pts_list.append(pts.astype(np.float32, copy=False))
            static_rgb_list.append(colors.astype(np.uint8, copy=False))
            static_frame_idx_list.append(np.full(pts.shape[0], frame_idx, dtype=np.int32))
            continue

        motion = compute_motion_mask(
            flow[frame_idx],
            valid,
            quantile=float(motion_quantile),
            min_abs=float(motion_min_abs),
        )
        static_sel = valid & ~motion
        dynamic_sel = valid & motion

        static_pts = points[frame_idx][static_sel] - center
        static_rgb = np.clip(rgb[static_sel].astype(np.float32) * float(static_darken), 0, 255).astype(np.uint8)
        static_pts, static_rgb = _sample_rows(
            static_pts,
            static_rgb,
            max_count=int(max_points_per_frame),
            rng=rng,
        )
        static_pts_list.append(static_pts.astype(np.float32, copy=False))
        static_rgb_list.append(static_rgb.astype(np.uint8, copy=False))
        static_frame_idx_list.append(np.full(static_pts.shape[0], frame_idx, dtype=np.int32))

        if not np.any(dynamic_sel):
            continue
        dynamic_pts = points[frame_idx][dynamic_sel] - center
        dynamic_rgb = rgb[dynamic_sel].copy()
        dynamic_flow = flow[frame_idx][dynamic_sel].astype(np.float32, copy=False)
        deformed_pts = dynamic_pts + dynamic_flow * np.float32(visual_flow_scale)
        dynamic_pts, dynamic_rgb, dynamic_flow, deformed_pts = _sample_rows(
            dynamic_pts,
            dynamic_rgb,
            dynamic_flow,
            deformed_pts,
            max_count=int(max_points_per_frame),
            rng=rng,
        )
        dynamic_pts_list.append(dynamic_pts.astype(np.float32, copy=False))
        dynamic_rgb_list.append(dynamic_rgb.astype(np.uint8, copy=False))
        dynamic_flow_list.append(dynamic_flow.astype(np.float32, copy=False))
        deformed_pts_list.append(deformed_pts.astype(np.float32, copy=False))
        dynamic_frame_idx_list.append(np.full(dynamic_pts.shape[0], frame_idx, dtype=np.int32))

        seg_start, seg_end, seg_flow = _sample_rows(
            dynamic_pts,
            deformed_pts,
            dynamic_flow,
            max_count=int(max_segments_per_frame),
            rng=rng,
        )
        segment_list.append(np.stack([seg_start, seg_end], axis=1).astype(np.float32))
        seg_color_list.append(scene_flow_to_rgb(seg_flow))
        segment_frame_idx_list.append(np.full(seg_start.shape[0], frame_idx, dtype=np.int32))

    static_pts = np.concatenate(static_pts_list, axis=0) if static_pts_list else _empty_points()
    static_rgb = (
        np.concatenate(static_rgb_list, axis=0).astype(np.uint8)
        if static_rgb_list
        else np.empty((0, 3), dtype=np.uint8)
    )
    dynamic_pts = np.concatenate(dynamic_pts_list, axis=0) if dynamic_pts_list else _empty_points()
    dynamic_rgb = (
        np.concatenate(dynamic_rgb_list, axis=0).astype(np.uint8)
        if dynamic_rgb_list
        else np.empty((0, 3), dtype=np.uint8)
    )
    dynamic_flow = np.concatenate(dynamic_flow_list, axis=0) if dynamic_flow_list else _empty_points()
    deformed_pts = np.concatenate(deformed_pts_list, axis=0) if deformed_pts_list else _empty_points()
    segments = (
        np.concatenate(segment_list, axis=0).astype(np.float32)
        if segment_list
        else np.empty((0, 2, 3), dtype=np.float32)
    )
    seg_colors = (
        np.concatenate(seg_color_list, axis=0).astype(np.uint8)
        if seg_color_list
        else np.empty((0, 3), dtype=np.uint8)
    )
    static_frame_idx = (
        np.concatenate(static_frame_idx_list, axis=0).astype(np.int32)
        if static_frame_idx_list
        else np.empty((0,), dtype=np.int32)
    )
    dynamic_frame_idx = (
        np.concatenate(dynamic_frame_idx_list, axis=0).astype(np.int32)
        if dynamic_frame_idx_list
        else np.empty((0,), dtype=np.int32)
    )
    segment_frame_idx = (
        np.concatenate(segment_frame_idx_list, axis=0).astype(np.int32)
        if segment_frame_idx_list
        else np.empty((0,), dtype=np.int32)
    )
    axis_limits = compute_axis_limits(static_pts, dynamic_pts, deformed_pts)
    return AggregatedGeometryMotion(
        static_pts=static_pts.astype(np.float32, copy=False),
        static_rgb=static_rgb.astype(np.uint8, copy=False),
        dynamic_pts=dynamic_pts.astype(np.float32, copy=False),
        dynamic_rgb=dynamic_rgb.astype(np.uint8, copy=False),
        dynamic_flow=dynamic_flow.astype(np.float32, copy=False),
        deformed_pts=deformed_pts.astype(np.float32, copy=False),
        segments=segments,
        seg_colors=seg_colors,
        static_frame_idx=static_frame_idx,
        dynamic_frame_idx=dynamic_frame_idx,
        segment_frame_idx=segment_frame_idx,
        center=center,
        axis_limits=axis_limits,
    )


def compute_axis_limits(*point_sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_sets = [np.asarray(points, dtype=np.float32) for points in point_sets if np.asarray(points).size > 0]
    if not valid_sets:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
    stacked = np.concatenate(valid_sets, axis=0)
    lower = np.percentile(stacked, 2.0, axis=0)
    upper = np.percentile(stacked, 98.0, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-3)
    return (center - radius).astype(np.float32), (center + radius).astype(np.float32)


def render_single_view(
    agg: AggregatedGeometryMotion,
    *,
    elev: float = 24.0,
    azim: float = -58.0,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
    point_size: float = 0.4,
    line_width: float = 0.8,
    title: str = "",
    include_deformed_points: bool = False,
) -> np.ndarray:
    fig = plt.figure(figsize=figsize, dpi=int(dpi))
    axis = fig.add_subplot(111, projection="3d")
    min_xyz, max_xyz = agg.axis_limits
    if agg.static_pts.size > 0:
        axis.scatter(
            agg.static_pts[:, 0],
            agg.static_pts[:, 1],
            agg.static_pts[:, 2],
            c=agg.static_rgb.astype(np.float32) / 255.0,
            s=float(point_size),
            alpha=0.6,
            linewidths=0.0,
            depthshade=False,
        )
    if agg.dynamic_pts.size > 0:
        axis.scatter(
            agg.dynamic_pts[:, 0],
            agg.dynamic_pts[:, 1],
            agg.dynamic_pts[:, 2],
            c=agg.dynamic_rgb.astype(np.float32) / 255.0,
            s=float(point_size) * 1.5,
            alpha=0.85,
            linewidths=0.0,
            depthshade=False,
        )
    if include_deformed_points and agg.deformed_pts.size > 0:
        axis.scatter(
            agg.deformed_pts[:, 0],
            agg.deformed_pts[:, 1],
            agg.deformed_pts[:, 2],
            c=np.clip(agg.dynamic_rgb.astype(np.float32) * 0.6, 0, 255) / 255.0,
            s=float(point_size),
            alpha=0.45,
            linewidths=0.0,
            depthshade=False,
        )
    if agg.segments.size > 0:
        axis.add_collection3d(
            Line3DCollection(
                agg.segments,
                colors=agg.seg_colors.astype(np.float32) / 255.0,
                linewidths=float(line_width),
                alpha=0.9,
            )
        )
    axis.set_xlim(float(min_xyz[0]), float(max_xyz[0]))
    axis.set_ylim(float(min_xyz[1]), float(max_xyz[1]))
    axis.set_zlim(float(min_xyz[2]), float(max_xyz[2]))
    axis.set_box_aspect(np.maximum(max_xyz - min_xyz, 1e-3))
    axis.view_init(elev=float(elev), azim=float(azim))
    axis.set_axis_off()
    if title:
        axis.set_title(title, fontsize=10, pad=2)
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image.astype(np.uint8, copy=False)


def render_multi_view(
    agg: AggregatedGeometryMotion,
    *,
    num_views: int = 6,
    cols: int = 0,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
    point_size: float = 0.4,
    line_width: float = 0.8,
    include_deformed_points: bool = False,
) -> np.ndarray:
    view_count = max(int(num_views), 1)
    col_count = min(view_count, 6) if int(cols) <= 0 else max(int(cols), 1)
    row_count = (view_count + col_count - 1) // col_count
    elevations = [20.0, 30.0, 10.0, 25.0, 35.0, 15.0, 20.0, 30.0]
    azimuths = np.linspace(-180.0, 180.0, view_count, endpoint=False)
    views: list[np.ndarray] = []
    for view_idx in range(view_count):
        elev = elevations[view_idx % len(elevations)]
        azim = float(azimuths[view_idx])
        views.append(
            render_single_view(
                agg,
                elev=elev,
                azim=azim,
                figsize=figsize,
                dpi=dpi,
                point_size=point_size,
                line_width=line_width,
                title=f"view {view_idx} elev={elev:.0f} azim={azim:.0f}",
                include_deformed_points=include_deformed_points,
            )
        )
    while len(views) < row_count * col_count:
        views.append(np.zeros_like(views[0], dtype=np.uint8))
    rows = [
        np.concatenate(views[row_idx * col_count : (row_idx + 1) * col_count], axis=1)
        for row_idx in range(row_count)
    ]
    return np.concatenate(rows, axis=0)


def write_motion_ply(
    output_path: str | Path,
    agg: AggregatedGeometryMotion,
    *,
    segment_interp_steps: int = 5,
    include_deformed_points: bool = False,
) -> Path:
    path = Path(output_path)
    vertex_xyz: list[np.ndarray] = []
    vertex_rgb: list[np.ndarray] = []
    vertex_flow: list[np.ndarray] = []
    vertex_frame: list[np.ndarray] = []
    vertex_flag: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []

    if agg.static_pts.size > 0:
        count = agg.static_pts.shape[0]
        vertex_xyz.append(agg.static_pts)
        vertex_rgb.append(agg.static_rgb)
        vertex_flow.append(np.zeros((count, 3), dtype=np.float32))
        vertex_frame.append(agg.static_frame_idx.astype(np.int32, copy=False))
        vertex_flag.append(np.zeros(count, dtype=np.uint8))
    if agg.dynamic_pts.size > 0:
        count = agg.dynamic_pts.shape[0]
        vertex_xyz.append(agg.dynamic_pts)
        vertex_rgb.append(agg.dynamic_rgb)
        vertex_flow.append(agg.dynamic_flow)
        vertex_frame.append(agg.dynamic_frame_idx.astype(np.int32, copy=False))
        vertex_flag.append(np.zeros(count, dtype=np.uint8))
    if include_deformed_points and agg.deformed_pts.size > 0:
        count = agg.deformed_pts.shape[0]
        vertex_xyz.append(agg.deformed_pts)
        vertex_rgb.append(np.clip(agg.dynamic_rgb.astype(np.float32) * 0.6, 0, 255).astype(np.uint8))
        vertex_flow.append(agg.dynamic_flow)
        vertex_frame.append(agg.dynamic_frame_idx.astype(np.int32, copy=False))
        vertex_flag.append(np.zeros(count, dtype=np.uint8))

    current_offset = sum(part.shape[0] for part in vertex_xyz)
    if agg.segments.size > 0:
        steps = max(int(segment_interp_steps), 2)
        t_values = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        interp = agg.segments[:, 0:1, :] * (1.0 - t_values[None, :, None]) + agg.segments[:, 1:2, :] * t_values[None, :, None]
        interp_points = interp.reshape(-1, 3).astype(np.float32)
        seg_colors = np.repeat(agg.seg_colors, steps, axis=0).astype(np.uint8)
        seg_flow = np.repeat(agg.segments[:, 1] - agg.segments[:, 0], steps, axis=0).astype(np.float32)
        vertex_xyz.append(interp_points)
        vertex_rgb.append(seg_colors)
        vertex_flow.append(seg_flow)
        vertex_frame.append(np.repeat(agg.segment_frame_idx, steps, axis=0).astype(np.int32))
        vertex_flag.append(np.ones(interp_points.shape[0], dtype=np.uint8))
        for segment_idx in range(agg.segments.shape[0]):
            base = current_offset + segment_idx * steps
            edges.extend((base + step_idx, base + step_idx + 1) for step_idx in range(steps - 1))

    path.parent.mkdir(parents=True, exist_ok=True)
    if not vertex_xyz:
        all_xyz = np.empty((0, 3), dtype=np.float32)
        all_rgb = np.empty((0, 3), dtype=np.uint8)
        all_flow = np.empty((0, 3), dtype=np.float32)
        all_frame = np.empty((0,), dtype=np.int32)
        all_flag = np.empty((0,), dtype=np.uint8)
    else:
        all_xyz = np.concatenate(vertex_xyz, axis=0).astype(np.float32)
        all_rgb = np.concatenate(vertex_rgb, axis=0).astype(np.uint8)
        all_flow = np.concatenate(vertex_flow, axis=0).astype(np.float32)
        all_frame = np.concatenate(vertex_frame, axis=0).astype(np.int32)
        all_flag = np.concatenate(vertex_flag, axis=0).astype(np.uint8)

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("flow_x", "<f4"),
            ("flow_y", "<f4"),
            ("flow_z", "<f4"),
            ("frame_idx", "<i4"),
            ("is_motion_trail", "u1"),
        ]
    )
    vertices = np.empty(all_xyz.shape[0], dtype=vertex_dtype)
    vertices["x"] = all_xyz[:, 0]
    vertices["y"] = all_xyz[:, 1]
    vertices["z"] = all_xyz[:, 2]
    vertices["red"] = all_rgb[:, 0]
    vertices["green"] = all_rgb[:, 1]
    vertices["blue"] = all_rgb[:, 2]
    vertices["flow_x"] = all_flow[:, 0]
    vertices["flow_y"] = all_flow[:, 1]
    vertices["flow_z"] = all_flow[:, 2]
    vertices["frame_idx"] = all_frame
    vertices["is_motion_trail"] = all_flag

    edge_dtype = np.dtype([("vertex1", "<i4"), ("vertex2", "<i4")])
    edge_array = np.empty(len(edges), dtype=edge_dtype)
    if edges:
        edge_values = np.asarray(edges, dtype=np.int32)
        edge_array["vertex1"] = edge_values[:, 0]
        edge_array["vertex2"] = edge_values[:, 1]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by PLAN4D geometry_motion_overlay_postprocess.py\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float flow_x\n"
        "property float flow_y\n"
        "property float flow_z\n"
        "property int frame_idx\n"
        "property uchar is_motion_trail\n"
        f"element edge {edge_array.shape[0]}\n"
        "property int vertex1\n"
        "property int vertex2\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)
        edge_array.tofile(file)
    return path


def _result_mode(result_path: Path) -> str | None:
    if result_path.name != "results.npz":
        return None
    if len(result_path.parents) < 3:
        return None
    if result_path.parent.name != "mini_npz" or result_path.parents[1].name != "exports":
        return None
    return _normalize_mode_name(result_path.parents[2].name)


def iter_result_npzs(input_dir: str | Path, modes: Sequence[str] = DEFAULT_MODES) -> Iterable[Path]:
    root = Path(input_dir)
    mode_set = {
        _normalize_mode_name(mode)
        for mode in modes
        if _normalize_mode_name(mode) not in GT_MODE_ALIASES
    }
    if not mode_set:
        return
    for result_path in sorted(root.rglob("results.npz")):
        mode = _result_mode(result_path)
        if mode in mode_set:
            yield result_path


def _find_run_meta_path(input_dir: str | Path) -> Path | None:
    root = Path(input_dir).expanduser().resolve()
    for candidate in (root, *root.parents):
        meta_path = candidate / "run_meta.json"
        if meta_path.is_file():
            return meta_path
    child_candidates = [root / "da3_stitched" / "run_meta.json"]
    if root.is_dir():
        child_candidates.extend(sorted(root.glob("*/run_meta.json")))
    for meta_path in child_candidates:
        if meta_path.is_file():
            return meta_path
    return None


def _load_run_meta(input_dir: str | Path) -> tuple[Path | None, dict]:
    meta_path = _find_run_meta_path(input_dir)
    if meta_path is None:
        return None, {}
    with meta_path.open("r", encoding="utf-8") as file:
        return meta_path, json.load(file)


def _run_meta_dataset_name(run_meta: Mapping[str, object]) -> str | None:
    benchmark = str(run_meta.get("benchmark", ""))
    dataset = str(run_meta.get("dataset", ""))
    if benchmark == "movi_f" or "movi_f_root" in run_meta:
        return "movi_f"
    if "dynamic_replica_root" in run_meta:
        return "dynamic_replica"
    if dataset == "point_odyssey" or "point_odyssey_root" in run_meta:
        return "point_odyssey"
    return None


def _clip_dirname(clip: Mapping[str, object]) -> str:
    if clip.get("clip_dirname"):
        return str(clip["clip_dirname"])
    frame_indices = tuple(int(item) for item in clip.get("frame_indices", ()))
    if frame_indices:
        return f"{frame_indices[0]:05d}_{frame_indices[-1] + 1:05d}"
    if "start_idx" in clip and "end_idx" in clip:
        return f"{int(clip['start_idx']):05d}_{int(clip['end_idx']):05d}"
    raise ValueError(f"Clip metadata is missing frame range: {clip}")


def _clip_base_dir(run_root: Path, run_meta: Mapping[str, object], clip: Mapping[str, object]) -> Path:
    dataset_name = _run_meta_dataset_name(run_meta)
    if dataset_name == "point_odyssey":
        return run_root / "point_odyssey" / str(clip["scene_name"]) / _clip_dirname(clip)
    return run_root / "model_results" / str(dataset_name) / str(clip["scene_name"]) / _clip_dirname(clip)


def _result_clip_key(result_path: Path) -> str | None:
    if len(result_path.parents) < 5:
        return None
    return f"{result_path.parents[4].name}/{result_path.parents[3].name}"


def _resolved_scene_flow_output_space(run_meta: Mapping[str, object], options: GeometryMotionOptions) -> str:
    requested = str(options.scene_flow_output_space)
    if requested not in SUPPORTED_SCENE_FLOW_OUTPUT_SPACES:
        raise ValueError(
            f"Unsupported scene_flow_output_space={requested!r}; expected one of {SUPPORTED_SCENE_FLOW_OUTPUT_SPACES}"
        )
    if requested != "auto":
        return requested
    target_space = str(run_meta.get("scene_flow_target_space", "relative"))
    eval_space = str(run_meta.get("scene_flow_eval_space", "auto"))
    if eval_space == "auto":
        return "metric" if target_space == "relative" else "native"
    return eval_space


def _should_apply_scene_flow_scale(run_meta: Mapping[str, object], options: GeometryMotionOptions) -> bool:
    target_space = str(run_meta.get("scene_flow_target_space", "relative"))
    output_space = _resolved_scene_flow_output_space(run_meta, options)
    return target_space == "relative" and output_space == "metric"


def _load_scene_flow_scale_from_metric_results(meta_path: Path, run_meta: Mapping[str, object], result_path: Path) -> float | None:
    mode = _result_mode(result_path)
    clip_key = _result_clip_key(result_path)
    if mode is None or clip_key is None:
        return None
    metric_dir = meta_path.parent / "metric_results"
    if not metric_dir.is_dir():
        return None
    dataset_name = _run_meta_dataset_name(run_meta)
    candidates = [metric_dir / f"{dataset_name}_scene_flow_{mode}.json"]
    candidates.extend(sorted(metric_dir.glob(f"*scene_flow_{mode}.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        record = payload.get("per_clip", {}).get(clip_key)
        if isinstance(record, dict) and "scene_flow_scale" in record:
            return _as_scalar_float(record["scene_flow_scale"])
    return None


def _scene_flow_scale_for_result_npz(result_path: Path, options: GeometryMotionOptions, payload) -> float:
    meta_path, run_meta = _load_run_meta(result_path)
    if meta_path is None or not _should_apply_scene_flow_scale(run_meta, options):
        return 1.0
    if "scene_flow_scale" in payload:
        return _as_scalar_float(payload["scene_flow_scale"])
    metric_scale = _load_scene_flow_scale_from_metric_results(meta_path, run_meta, result_path)
    if metric_scale is not None:
        return float(metric_scale)
    return 1.0


def _clip_metadata_for_result(run_meta: Mapping[str, object], result_path: Path) -> Mapping[str, object] | None:
    if len(result_path.parents) < 5:
        return None
    scene_name = result_path.parents[4].name
    clip_dirname = result_path.parents[3].name
    for clip in run_meta.get("clips", []):
        if str(clip.get("scene_name")) == scene_name and _clip_dirname(clip) == clip_dirname:
            return clip
    return None


def _result_clip_base_dir(result_path: Path) -> Path:
    return result_path.parents[3] if len(result_path.parents) >= 4 else result_path.parent


def _load_rgb_from_gt_meta(result_path: Path) -> np.ndarray | None:
    export_dir = result_path.parents[1]
    gt_meta_path = export_dir / "gt_meta.npz"
    if not gt_meta_path.is_file():
        return None
    with np.load(gt_meta_path, allow_pickle=True) as gt_meta:
        if "image_files" not in gt_meta:
            return None
        image_files = [str(path) for path in gt_meta["image_files"]]
    try:
        from evaluation.dynamic_replica_ray_benchmark import _load_rgb_image
    except Exception:
        return None
    return np.stack([_load_rgb_image(path) for path in image_files], axis=0).astype(np.uint8)


def _load_rgb_from_gt_payload(result_path: Path) -> np.ndarray | None:
    cache_key = _result_clip_base_dir(result_path)
    if cache_key in _RGB_RESULT_CACHE:
        return _RGB_RESULT_CACHE[cache_key]
    meta_path, run_meta = _load_run_meta(result_path)
    if meta_path is None:
        _RGB_RESULT_CACHE[cache_key] = None
        return None
    clip = _clip_metadata_for_result(run_meta, result_path)
    if clip is None:
        _RGB_RESULT_CACHE[cache_key] = None
        return None
    try:
        payload = _load_gt_scene_payload(run_meta, clip)
    except Exception:
        _RGB_RESULT_CACHE[cache_key] = None
        return None
    rgb = payload.get("rgb")
    if rgb is None:
        rgb = payload.get("rgbs")
    rgb_array = None if rgb is None else np.asarray(rgb, dtype=np.uint8)
    _RGB_RESULT_CACHE[cache_key] = rgb_array
    return rgb_array


def _load_rgb_for_result_npz(result_path: Path) -> np.ndarray | None:
    rgb = _load_rgb_from_gt_meta(result_path)
    if rgb is not None:
        return rgb
    return _load_rgb_from_gt_payload(result_path)


def _load_prediction_payload(result_path: Path, options: GeometryMotionOptions) -> dict[str, np.ndarray]:
    with np.load(result_path, allow_pickle=True) as payload:
        missing = [key for key in ("depth", "intrinsics", "extrinsics", options.flow_key) if key not in payload]
        if missing:
            raise KeyError(f"{result_path} is missing required arrays: {', '.join(missing)}")
        depth = np.asarray(payload["depth"], dtype=np.float32)
        scene_flow = np.asarray(payload[options.flow_key], dtype=np.float32)
        scale = _scene_flow_scale_for_result_npz(result_path, options, payload)
        scene_flow = scene_flow * np.float32(scale) * np.float32(options.flow_scale)
        rgb = np.asarray(payload["rgb"]) if "rgb" in payload else _load_rgb_for_result_npz(result_path)
        return {
            "depth": depth,
            "intrinsics": np.asarray(payload["intrinsics"], dtype=np.float32),
            "extrinsics": np.asarray(payload["extrinsics"], dtype=np.float32),
            "scene_flow": scene_flow,
            "valid_mask": np.asarray(payload["valid_mask"]) if "valid_mask" in payload else None,
            "deform_mask": np.asarray(payload["deform_mask"]) if "deform_mask" in payload else None,
            "rgb": rgb,
            "scene_flow_scale": np.array(scale, dtype=np.float32),
        }


def _scene_flow_scale_for_gt_payload(run_meta: Mapping[str, object], options: GeometryMotionOptions, payload: Mapping[str, object]) -> float:
    if not _should_apply_scene_flow_scale(run_meta, options):
        return 1.0
    return _as_scalar_float(payload.get("scene_flow_scale"), default=1.0)


def _load_movi_f_gt_scene_payload(run_meta: Mapping[str, object], clip: Mapping[str, object], *, target_space: str) -> dict[str, np.ndarray]:
    from evaluation.scene_flow_ply_postprocess import _load_movi_f_gt_scene_payload as load_payload

    return load_payload(dict(run_meta), dict(clip), target_space=target_space)


def _load_dynamic_replica_gt_scene_payload(run_meta: Mapping[str, object], clip: Mapping[str, object]) -> dict[str, np.ndarray]:
    from evaluation.scene_flow_ply_postprocess import _load_gt_scene_payload as load_payload

    return load_payload(dict(run_meta), dict(clip))


def _load_point_odyssey_scene_flow_module():
    module_path = DA3_ROOT / "training" / "datasets" / "point_odyssey_scene_flow.py"
    module_name = "_plan4d_point_odyssey_scene_flow_overlay"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PointOdyssey scene-flow module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_point_odyssey_gt_scene_payload(run_meta: Mapping[str, object], clip: Mapping[str, object]) -> dict[str, np.ndarray]:
    point_odyssey_module = _load_point_odyssey_scene_flow_module()
    PointOdysseySceneFlowDataset = point_odyssey_module.PointOdysseySceneFlowDataset
    SampleInfo = point_odyssey_module.SampleInfo
    load_point_odyssey_depth = point_odyssey_module._load_point_odyssey_depth
    find_depth_path = point_odyssey_module._find_depth_path

    frame_indices = tuple(int(frame_id) for frame_id in clip.get("frame_indices", ()))
    if not frame_indices:
        start_idx = int(clip["start_idx"])
        end_idx = int(clip["end_idx"])
        frame_indices = tuple(range(start_idx, end_idx))
    data_root = Path(str(run_meta["point_odyssey_root"]))
    split = str(clip.get("split", run_meta.get("split", "test")))
    dataset = object.__new__(PointOdysseySceneFlowDataset)
    dataset.data_root = data_root
    dataset.split = split
    dataset.chunk_size = int(run_meta.get("clip_length", len(frame_indices)))
    dataset.overlap = int(run_meta.get("overlap", 0))
    dataset.target_height = int(run_meta.get("input_resolution", 504))
    dataset.target_width = int(run_meta.get("input_resolution", 504))
    dataset.process_res = int(run_meta.get("input_resolution", 504))
    dataset.process_res_method = str(run_meta.get("process_res_method", "upper_bound_resize"))
    dataset.max_scenes = None
    dataset.pad_to_chunk_size = False
    dataset.flow_target_space = str(run_meta.get("scene_flow_target_space", "relative"))
    dataset.relative_scale_source = str(run_meta.get("relative_scale_source", "geometry_radius"))
    dataset.geometry_scale_stat = str(run_meta.get("geometry_scale_stat", "mean_radius"))
    dataset.flow_scale_stat = str(run_meta.get("flow_scale_stat", "p95"))
    dataset.flow_scale_eps = float(run_meta.get("flow_scale_eps", 1e-6))
    dataset.samples = [
        SampleInfo(
            scene_name=str(clip["scene_name"]),
            scene_dir=data_root / split / str(clip["scene_name"]),
            start_idx=int(frame_indices[0]),
            end_idx=int(frame_indices[-1]) + 1,
        )
    ]
    dataset._skipped_scene_names = set()
    sample = dataset[0]
    sequence_length = int(sample["sequence_length"].item())
    depths: list[np.ndarray] = []
    for frame_idx in frame_indices[:sequence_length]:
        depth = load_point_odyssey_depth(find_depth_path(dataset.samples[0].scene_dir, frame_idx))
        depth_resized = cv2.resize(
            depth,
            (int(sample["rgbs"].shape[2]), int(sample["rgbs"].shape[1])),
            interpolation=cv2.INTER_NEAREST,
        )
        depths.append(depth_resized.astype(np.float32))
    return {
        "depth": np.stack(depths, axis=0).astype(np.float32),
        "intrinsics": sample["intrinsics"].detach().cpu().numpy()[:sequence_length].astype(np.float32),
        "extrinsics": sample["extrinsics"].detach().cpu().numpy()[:sequence_length].astype(np.float32),
        "scene_flow": sample["scene_flow"].detach().cpu().numpy()[:sequence_length].astype(np.float32),
        "valid_mask": np.ones(tuple(sample["rgbs"].shape[:3]), dtype=np.float32)[:sequence_length],
        "deform_mask": sample["deform_mask"].detach().cpu().numpy()[:sequence_length].astype(np.float32),
        "rgb": sample["rgbs"].detach().cpu().numpy()[:sequence_length].astype(np.uint8),
        "scene_flow_scale": np.array(float(sample.get("scene_flow_scale", 1.0)), dtype=np.float32),
    }


def _load_gt_scene_payload(run_meta: Mapping[str, object], clip: Mapping[str, object]) -> dict[str, np.ndarray]:
    dataset_name = _run_meta_dataset_name(run_meta)
    if dataset_name == "movi_f":
        return _load_movi_f_gt_scene_payload(
            run_meta,
            clip,
            target_space=str(run_meta.get("scene_flow_target_space", "relative")),
        )
    if dataset_name == "dynamic_replica":
        return _load_dynamic_replica_gt_scene_payload(run_meta, clip)
    if dataset_name == "point_odyssey":
        return _load_point_odyssey_gt_scene_payload(run_meta, clip)
    raise RuntimeError("GT mode supports point_odyssey, Dynamic Replica, and MOVi-F run_meta.json files only.")


def _gt_payload_for_clip(run_meta: Mapping[str, object], clip: Mapping[str, object], options: GeometryMotionOptions) -> dict[str, np.ndarray]:
    payload = _load_gt_scene_payload(run_meta, clip)
    flow = np.asarray(payload[options.flow_key if options.flow_key in payload else "scene_flow"], dtype=np.float32)
    scale = _scene_flow_scale_for_gt_payload(run_meta, options, payload)
    payload = dict(payload)
    payload["scene_flow"] = flow * np.float32(scale) * np.float32(options.flow_scale)
    payload["scene_flow_scale"] = np.array(scale, dtype=np.float32)
    return payload


def _default_prediction_output_dir(result_path: Path, options: GeometryMotionOptions) -> Path:
    return result_path.parents[1] / options.output_subdir


def _default_gt_output_dir(base_dir: Path, options: GeometryMotionOptions) -> Path:
    return base_dir / "gt_fuse" / options.output_subdir


def _mirrored_prediction_output_dir(
    result_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    options: GeometryMotionOptions,
) -> Path:
    export_dir = result_path.parents[1]
    try:
        relative_export_dir = export_dir.relative_to(input_dir)
    except ValueError:
        relative_export_dir = Path(*export_dir.parts[-6:])
    return output_dir / relative_export_dir / options.output_subdir


def _mirrored_gt_output_dir(
    base_dir: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    options: GeometryMotionOptions,
) -> Path:
    gt_dir = base_dir / "gt_fuse"
    try:
        relative_gt_dir = gt_dir.relative_to(input_dir)
    except ValueError:
        relative_gt_dir = Path(*gt_dir.parts[-5:])
    return output_dir / relative_gt_dir / options.output_subdir


def _write_summary(path: Path, *, agg: AggregatedGeometryMotion, mode: str, source_path: Path, center: np.ndarray) -> None:
    payload = {
        "mode": mode,
        "source_path": str(source_path),
        "static_points": int(agg.static_pts.shape[0]),
        "dynamic_points": int(agg.dynamic_pts.shape[0]),
        "motion_segments": int(agg.segments.shape[0]),
        "center": [float(item) for item in center.reshape(-1)],
        "point_coordinate_space": "frame0_camera",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_geometry_motion_payload(
    *,
    payload: Mapping[str, object],
    output_dir: str | Path,
    mode: str,
    source_path: str | Path,
    options: GeometryMotionOptions,
) -> GeometryMotionOutputRecord:
    output_root = Path(output_dir)
    image_path = output_root / "geometry_motion_overlay.png"
    ply_path = output_root / "geometry_motion_overlay.ply"
    summary_path = output_root / "summary.json"
    if image_path.exists() and ply_path.exists() and not options.overwrite:
        return GeometryMotionOutputRecord(
            mode=str(mode),
            source_path=Path(source_path),
            image_path=image_path,
            ply_path=ply_path,
            summary_path=summary_path,
        )

    depth = np.asarray(payload["depth"], dtype=np.float32)
    extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
    point_map = build_point_map_frame0(
        depth=depth,
        intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
        extrinsics=extrinsics,
    )
    scene_flow = np.asarray(payload["scene_flow"], dtype=np.float32) if payload.get("scene_flow") is not None else None
    if scene_flow is not None and str(options.flow_key) == "scene_flow_cam":
        scene_flow = _camera_flow_to_frame0(scene_flow, extrinsics)
    agg = aggregate_all_frames(
        point_map=point_map,
        scene_flow=scene_flow,
        rgbs=payload.get("rgb"),
        valid_mask_raw=payload.get("valid_mask"),
        deform_mask_raw=payload.get("deform_mask"),
        edge_threshold=float(options.edge_threshold),
        border_crop=int(options.border_crop),
        motion_quantile=float(options.motion_quantile),
        motion_min_abs=float(options.motion_min_abs),
        visual_flow_scale=float(options.visual_flow_scale),
        max_points_per_frame=int(options.max_points_per_frame),
        max_segments_per_frame=int(options.max_segments_per_frame),
        static_darken=float(options.static_darken),
        deformed_darken=float(options.deformed_darken),
        seed=int(options.seed),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    image = render_multi_view(
        agg,
        num_views=int(options.views),
        cols=int(options.cols),
        figsize=tuple(options.figsize),
        dpi=int(options.dpi),
        point_size=float(options.point_size),
        line_width=float(options.line_width),
        include_deformed_points=bool(options.include_deformed_points),
    )
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    write_motion_ply(
        ply_path,
        agg,
        segment_interp_steps=int(options.segment_interp_steps),
        include_deformed_points=bool(options.include_deformed_points),
    )
    _write_summary(summary_path, agg=agg, mode=str(mode), source_path=Path(source_path), center=agg.center)
    return GeometryMotionOutputRecord(
        mode=str(mode),
        source_path=Path(source_path),
        image_path=image_path,
        ply_path=ply_path,
        summary_path=summary_path,
    )


def process_result_npz(
    result_path: str | Path,
    options: GeometryMotionOptions,
    *,
    output_dir: str | Path | None = None,
) -> GeometryMotionOutputRecord:
    result_path = Path(result_path)
    mode = _result_mode(result_path) or "unknown"
    payload = _load_prediction_payload(result_path, options)
    target_dir = Path(output_dir) if output_dir is not None else _default_prediction_output_dir(result_path, options)
    return process_geometry_motion_payload(
        payload=payload,
        output_dir=target_dir,
        mode=mode,
        source_path=result_path,
        options=options,
    )


def _wants_gt_mode(modes: Sequence[str]) -> bool:
    return any(_normalize_mode_name(mode) in GT_MODE_ALIASES for mode in modes)


def process_gt_fuse_outputs(
    input_dir: str | Path,
    options: GeometryMotionOptions,
    *,
    output_dir: str | Path | None = None,
) -> tuple[list[GeometryMotionOutputRecord], list[str]]:
    input_root = Path(input_dir).expanduser().resolve()
    meta_path, run_meta = _load_run_meta(input_root)
    if meta_path is None:
        return [], [f"{input_root}: GT mode requires run_meta.json"]
    run_root = meta_path.parent
    output_root = None if output_dir is None else Path(output_dir)
    records: list[GeometryMotionOutputRecord] = []
    errors: list[str] = []
    for clip in run_meta.get("clips", []):
        base_dir = _clip_base_dir(run_root, run_meta, clip)
        if not _paths_overlap(input_root, base_dir):
            continue
        try:
            payload = _gt_payload_for_clip(run_meta, clip, options)
            target_dir = (
                _default_gt_output_dir(base_dir, options)
                if output_root is None
                else _mirrored_gt_output_dir(base_dir, input_dir=input_root, output_dir=output_root, options=options)
            )
            records.append(
                process_geometry_motion_payload(
                    payload=payload,
                    output_dir=target_dir,
                    mode="gt_fuse",
                    source_path=base_dir,
                    options=options,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other clips.
            errors.append(f"{base_dir}: {exc}")
    return records, errors


def _paths_overlap(root: Path, path: Path) -> bool:
    root = root.expanduser().resolve()
    path = path.expanduser().resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        pass
    try:
        root.relative_to(path)
        return True
    except ValueError:
        return False


def process_output_tree(
    input_dir: str | Path,
    options: GeometryMotionOptions,
    *,
    output_dir: str | Path | None = None,
) -> GeometryMotionProcessResult:
    input_root = Path(input_dir).expanduser().resolve()
    output_root = None if output_dir is None else Path(output_dir)
    requested_modes = tuple(_normalize_mode_name(mode) for mode in options.modes)
    result_paths = list(iter_result_npzs(input_root, requested_modes))
    found_modes = {_result_mode(path) for path in result_paths}
    skipped_modes = {
        mode: 1
        for mode in requested_modes
        if mode not in GT_MODE_ALIASES and mode not in found_modes
    }

    records: list[GeometryMotionOutputRecord] = []
    errors: list[str] = []
    for result_path in result_paths:
        target_dir = (
            None
            if output_root is None
            else _mirrored_prediction_output_dir(result_path, input_dir=input_root, output_dir=output_root, options=options)
        )
        try:
            records.append(process_result_npz(result_path, options, output_dir=target_dir))
        except Exception as exc:  # noqa: BLE001 - summarize failures after batch.
            errors.append(f"{result_path}: {exc}")

    if _wants_gt_mode(requested_modes):
        gt_records, gt_errors = process_gt_fuse_outputs(input_root, options, output_dir=output_dir)
        records.extend(gt_records)
        errors.extend(gt_errors)
        if not gt_records:
            skipped_modes["gt_fuse"] = 1

    if not records and errors:
        raise RuntimeError(f"No geometry-motion overlay outputs were written from {input_root}.\n" + "\n".join(errors[:10]))
    return GeometryMotionProcessResult(records=tuple(records), skipped_modes=skipped_modes, errors=tuple(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process PLAN4D scene-flow exports into geometry-motion overlay images and PLYs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", nargs="?", help="Evaluation output root or subtree.")
    parser.add_argument("--input_dir", dest="input_dir_flag", help="Evaluation output root; overrides positional input_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional mirror output root. Defaults to writing beside each result.")
    parser.add_argument("--modes", type=_parse_modes, default=DEFAULT_MODES, help="Comma-separated modes: posed,unposed,gt/fuse_gt.")
    parser.add_argument("--flow_key", type=str, default="scene_flow", choices=SUPPORTED_FLOW_KEYS)
    parser.add_argument("--scene_flow_output_space", type=str, default="auto", choices=SUPPORTED_SCENE_FLOW_OUTPUT_SPACES)
    parser.add_argument("--flow_scale", type=float, default=1.0, help="Extra multiplier applied to stored flow and PLY attributes.")
    parser.add_argument("--visual_flow_scale", type=float, default=1.0, help="Display-only multiplier for rendered segments/trails.")
    parser.add_argument("--output_subdir", type=str, default="geometry_motion_overlay")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite files under the geometry_motion_overlay output dir.")
    parser.add_argument("--edge_threshold", type=float, default=0.08)
    parser.add_argument("--border_crop", type=int, default=5)
    parser.add_argument("--motion_quantile", type=float, default=85.0)
    parser.add_argument("--motion_min_abs", type=float, default=0.0)
    parser.add_argument("--max_points_per_frame", type=int, default=30_000, help="0 or negative disables per-frame point subsampling.")
    parser.add_argument("--max_segments_per_frame", type=int, default=1_500, help="0 or negative disables per-frame segment subsampling.")
    parser.add_argument("--static_darken", type=float, default=0.75)
    parser.add_argument("--deformed_darken", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--cols", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.0, 6.0), metavar=("W", "H"))
    parser.add_argument("--point_size", type=float, default=0.4)
    parser.add_argument("--line_width", type=float, default=0.8)
    parser.add_argument("--segment_interp_steps", type=int, default=5)
    parser.add_argument("--include_deformed_points", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir_flag or args.input_dir
    if not input_dir:
        raise SystemExit("input_dir is required")
    options = GeometryMotionOptions(
        modes=tuple(args.modes),
        flow_key=str(args.flow_key),
        scene_flow_output_space=str(args.scene_flow_output_space),
        flow_scale=float(args.flow_scale),
        visual_flow_scale=float(args.visual_flow_scale),
        output_subdir=str(args.output_subdir),
        overwrite=bool(args.overwrite),
        edge_threshold=float(args.edge_threshold),
        border_crop=int(args.border_crop),
        motion_quantile=float(args.motion_quantile),
        motion_min_abs=float(args.motion_min_abs),
        max_points_per_frame=int(args.max_points_per_frame),
        max_segments_per_frame=int(args.max_segments_per_frame),
        static_darken=float(args.static_darken),
        deformed_darken=float(args.deformed_darken),
        seed=int(args.seed),
        views=int(args.views),
        cols=int(args.cols),
        dpi=int(args.dpi),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        point_size=float(args.point_size),
        line_width=float(args.line_width),
        segment_interp_steps=int(args.segment_interp_steps),
        include_deformed_points=bool(args.include_deformed_points),
    )
    result = process_output_tree(input_dir, options, output_dir=args.output_dir)
    print(f"Wrote or reused {len(result.records)} geometry-motion overlay clips.")
    for record in result.records[:20]:
        print(record.image_path)
        print(record.ply_path)
    if len(result.records) > 20:
        print(f"... {len(result.records) - 20} more")
    if result.skipped_modes:
        print("Skipped missing modes:", json.dumps(result.skipped_modes, ensure_ascii=False, sort_keys=True))
    if result.errors:
        print(f"Encountered {len(result.errors)} errors; first errors:")
        for error in result.errors[:10]:
            print(error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
