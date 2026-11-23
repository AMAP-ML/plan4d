from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SUPPORTED_COLOR_MODES = ("direction_magnitude", "direction", "magnitude")
SUPPORTED_VISUALIZATION_STYLES = ("flow", "tracking")
SUPPORTED_SCENE_FLOW_OUTPUT_SPACES = ("auto", "native", "metric")
DEFAULT_MODES = ("unposed", "posed")
GT_MODE_ALIASES = {"gt", "gt_fuse", "fuse_gt"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RGB_RESULT_CACHE: dict[Path, np.ndarray | None] = {}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class FlowPlyOptions:
    flow_key: str = "scene_flow"
    color_mode: str = "direction_magnitude"
    magnitude_percentile: float = 95.0
    flow_scale: float = 1.0
    max_points_per_clip: int = 300_000
    min_depth: float = 1e-6
    max_depth: float | None = None
    min_conf: float | None = None
    pixel_stride: int = 1
    seed: int = 0
    modes: tuple[str, ...] = DEFAULT_MODES
    output_subdir: str = "scene_flow_ply"
    overwrite: bool = False
    visualization_style: str = "flow"
    trail_points: int = 20
    trail_only_moving: bool = True
    moving_threshold: float = 0.01
    scene_flow_output_space: str = "auto"


def _normalize_mode_name(mode: str) -> str:
    return str(mode).strip().lower().replace("-", "_").replace(" ", "_")


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


def _as_flow_tchw(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"Expected scene flow shape [T,3,H,W] or [T,H,W,3], got {array.shape}")
    if array.shape[1] == 3:
        return array
    if array.shape[-1] == 3:
        return np.transpose(array, (0, 3, 1, 2)).copy()
    raise ValueError(f"Expected scene-flow channel dimension of 3, got {array.shape}")


def _resize_rgb_thwc(rgb: np.ndarray, *, height: int, width: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ValueError(
            "RGB frames do not match depth resolution and cv2 is unavailable for resizing."
        ) from exc
    frames = [
        cv2.resize(frame[..., :3], (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        for frame in np.asarray(rgb)
    ]
    return np.stack(frames, axis=0)


def _as_rgb_thwc(rgb: np.ndarray, *, num_frames: int, height: int, width: int) -> np.ndarray:
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


def _valid_depth_mask(depth: np.ndarray, conf: np.ndarray | None, options: FlowPlyOptions) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > float(options.min_depth))
    if options.max_depth is not None:
        mask &= depth <= float(options.max_depth)
    if conf is not None and options.min_conf is not None:
        conf_array = np.asarray(conf, dtype=np.float32)
        if conf_array.shape != depth.shape:
            raise ValueError(f"Expected conf shape {depth.shape}, got {conf_array.shape}")
        mask &= np.isfinite(conf_array) & (conf_array >= float(options.min_conf))
    stride = int(options.pixel_stride)
    if stride < 1:
        raise ValueError("--pixel_stride must be >= 1")
    if stride > 1:
        stride_mask = np.zeros_like(mask, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask
    return mask


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
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def _radial_depth_to_z_depth_sequence(radial_depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth_thw = _as_depth_thw(radial_depth)
    intrinsics_t = _as_frame_matrices(intrinsics, depth_thw.shape[0], "intrinsics")
    z_depth = np.empty_like(depth_thw, dtype=np.float32)
    for frame_idx, depth_frame in enumerate(depth_thw):
        intrinsic = intrinsics_t[frame_idx]
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        if abs(fx) < 1e-12 or abs(fy) < 1e-12:
            raise ValueError("intrinsic focal length must be non-zero")
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        yy, xx = np.indices(depth_frame.shape, dtype=np.float32)
        x_norm = (xx - cx) / np.float32(fx)
        y_norm = (yy - cy) / np.float32(fy)
        ray_norm = np.sqrt(x_norm * x_norm + y_norm * y_norm + np.float32(1.0))
        z_depth[frame_idx] = np.asarray(depth_frame, dtype=np.float32) / np.maximum(ray_norm, np.float32(1e-8))
    return z_depth


def _transform_camera_to_world(points_cam: np.ndarray, extrinsic_w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(np.asarray(extrinsic_w2c, dtype=np.float32))
    flat_points = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    ones = np.ones((flat_points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate((flat_points, ones), axis=1)
    points_world = points_h @ c2w.T
    return points_world[:, :3].reshape(points_cam.shape)


def _sample_indices(num_points: int, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(num_points, size=int(max_points), replace=False))


def _magnitude_values(magnitudes: np.ndarray, percentile: float) -> np.ndarray:
    finite = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    if finite.size == 0:
        return np.zeros_like(magnitudes, dtype=np.float32)
    scale = float(np.percentile(finite, float(percentile)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return np.clip(magnitudes / scale, 0.0, 1.0).astype(np.float32)


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


def flow_vectors_to_rgb(
    flow_vectors: np.ndarray,
    *,
    color_mode: str = "direction_magnitude",
    magnitude_percentile: float = 95.0,
) -> np.ndarray:
    flow = np.asarray(flow_vectors, dtype=np.float32).reshape(-1, 3)
    if color_mode not in SUPPORTED_COLOR_MODES:
        raise ValueError(f"Unsupported color_mode={color_mode!r}; expected one of {SUPPORTED_COLOR_MODES}")
    magnitudes = np.linalg.norm(flow, axis=1)
    values = _magnitude_values(magnitudes, magnitude_percentile)
    nonzero = magnitudes > 1e-12
    if color_mode == "magnitude":
        rgb = np.repeat(values[:, None], 3, axis=1)
    else:
        hue = (np.arctan2(flow[:, 1], flow[:, 0]) + np.pi) / (2.0 * np.pi)
        value = values if color_mode == "direction_magnitude" else nonzero.astype(np.float32)
        rgb = _hsv_to_rgb(hue, np.ones_like(value, dtype=np.float32), value)
    rgb[~np.isfinite(rgb)] = 0.0
    rgb[~nonzero] = 0.0
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _build_tracking_scene_flow_point_cloud(
    *,
    depth_thw: np.ndarray,
    flow_tchw: np.ndarray,
    intrinsics_t: np.ndarray,
    extrinsics_t: np.ndarray,
    conf_thw: np.ndarray | None,
    valid_mask_thw: np.ndarray | None,
    options: FlowPlyOptions,
    rgb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, height, width = depth_thw.shape
    trail_points = int(options.trail_points)
    if trail_points < 1:
        raise ValueError("--trail_points must be >= 1")
    rgb_thwc = None if rgb is None else _as_rgb_thwc(rgb, num_frames=num_frames, height=height, width=width)

    static_point_chunks: list[np.ndarray] = []
    static_color_chunks: list[np.ndarray] = []
    trail_point_chunks: list[np.ndarray] = []
    trail_color_chunks: list[np.ndarray] = []

    for frame_idx in range(num_frames):
        depth_frame = depth_thw[frame_idx]
        conf_frame = None if conf_thw is None else conf_thw[frame_idx]
        valid = _valid_depth_mask(depth_frame, conf_frame, options)
        if valid_mask_thw is not None:
            valid &= valid_mask_thw[frame_idx] > 0.5
        flow_frame = np.transpose(flow_tchw[frame_idx], (1, 2, 0))
        valid &= np.isfinite(flow_frame).all(axis=-1)
        if not np.any(valid):
            continue

        points_cam = _camera_points_from_z_depth(depth_frame, intrinsics_t[frame_idx])
        points_world = _transform_camera_to_world(points_cam, extrinsics_t[frame_idx])
        pts_valid = points_world[valid].astype(np.float32, copy=False)
        flow_valid = flow_frame[valid].astype(np.float32, copy=False)

        static_point_chunks.append(pts_valid)
        if rgb_thwc is None:
            static_color_chunks.append(np.full((pts_valid.shape[0], 3), 180, dtype=np.uint8))
        else:
            static_color_chunks.append(rgb_thwc[frame_idx][valid])

        magnitudes = np.linalg.norm(flow_valid, axis=1)
        if bool(options.trail_only_moving):
            trail_mask = magnitudes > float(options.moving_threshold)
        else:
            trail_mask = np.ones(pts_valid.shape[0], dtype=bool)
        if not np.any(trail_mask):
            continue

        origins = pts_valid[trail_mask]
        flows = flow_valid[trail_mask]
        trail_rgb = flow_vectors_to_rgb(
            flows,
            color_mode=options.color_mode,
            magnitude_percentile=float(options.magnitude_percentile),
        )
        for t_value in np.linspace(0.0, 1.0, trail_points, dtype=np.float32):
            trail_point_chunks.append((origins + t_value * flows).astype(np.float32, copy=False))
            trail_color_chunks.append(trail_rgb.copy())

    if not static_point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(static_point_chunks + trail_point_chunks, axis=0)
    colors = np.concatenate(static_color_chunks + trail_color_chunks, axis=0)
    indices = _sample_indices(points.shape[0], int(options.max_points_per_clip), int(options.seed))
    return points[indices].astype(np.float32, copy=False), colors[indices].astype(np.uint8, copy=False)


def build_scene_flow_point_cloud(
    *,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    scene_flow: np.ndarray,
    options: FlowPlyOptions,
    conf: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    style = str(options.visualization_style)
    if style not in SUPPORTED_VISUALIZATION_STYLES:
        raise ValueError(f"Unsupported visualization_style={style!r}; expected one of {SUPPORTED_VISUALIZATION_STYLES}")
    depth_thw = _as_depth_thw(depth)
    flow_tchw = _as_flow_tchw(scene_flow) * float(options.flow_scale)
    num_frames, height, width = depth_thw.shape
    if flow_tchw.shape[0] != num_frames or flow_tchw.shape[-2:] != (height, width):
        raise ValueError(f"scene_flow shape {flow_tchw.shape} does not match depth shape {depth_thw.shape}")
    intrinsics_t = _as_frame_matrices(intrinsics, num_frames, "intrinsics")
    if intrinsics_t.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape [T,3,3], got {intrinsics_t.shape}")
    extrinsics_t = _as_homogeneous_extrinsics(extrinsics, num_frames)
    conf_thw = None if conf is None else _as_depth_thw(conf)
    if conf_thw is not None and conf_thw.shape != depth_thw.shape:
        raise ValueError(f"conf shape {conf_thw.shape} does not match depth shape {depth_thw.shape}")
    valid_mask_thw = None if valid_mask is None else _as_depth_thw(valid_mask)
    if valid_mask_thw is not None and valid_mask_thw.shape != depth_thw.shape:
        raise ValueError(f"valid_mask shape {valid_mask_thw.shape} does not match depth shape {depth_thw.shape}")

    if style == "tracking":
        return _build_tracking_scene_flow_point_cloud(
            depth_thw=depth_thw,
            flow_tchw=flow_tchw,
            intrinsics_t=intrinsics_t,
            extrinsics_t=extrinsics_t,
            conf_thw=conf_thw,
            valid_mask_thw=valid_mask_thw,
            options=options,
            rgb=rgb,
        )

    point_chunks: list[np.ndarray] = []
    flow_chunks: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        depth_frame = depth_thw[frame_idx]
        conf_frame = None if conf_thw is None else conf_thw[frame_idx]
        valid = _valid_depth_mask(depth_frame, conf_frame, options)
        if valid_mask_thw is not None:
            valid &= valid_mask_thw[frame_idx] > 0.5
        flow_frame = np.transpose(flow_tchw[frame_idx], (1, 2, 0))
        valid &= np.isfinite(flow_frame).all(axis=-1)
        if not np.any(valid):
            continue
        points_cam = _camera_points_from_z_depth(depth_frame, intrinsics_t[frame_idx])
        points_world = _transform_camera_to_world(points_cam, extrinsics_t[frame_idx])
        point_chunks.append(points_world[valid].astype(np.float32, copy=False))
        flow_chunks.append(flow_frame[valid].astype(np.float32, copy=False))

    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(point_chunks, axis=0)
    flow_vectors = np.concatenate(flow_chunks, axis=0)
    indices = _sample_indices(points.shape[0], int(options.max_points_per_clip), int(options.seed))
    points = points[indices]
    flow_vectors = flow_vectors[indices]
    colors = flow_vectors_to_rgb(
        flow_vectors,
        color_mode=options.color_mode,
        magnitude_percentile=float(options.magnitude_percentile),
    )
    return points.astype(np.float32, copy=False), colors

def write_point_cloud_ply(path: str | Path, points: np.ndarray, colors: np.ndarray, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        return output_path
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"Expected one color per point, got {points.shape[0]} points and {colors.shape[0]} colors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by PLAN4D scene_flow_ply_postprocess.py\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("wb") as file:
        file.write(header.encode("ascii"))
        vertices.tofile(file)
    return output_path


def _result_mode(result_path: Path) -> str | None:
    if result_path.name != "results.npz":
        return None
    if len(result_path.parents) < 3:
        return None
    if result_path.parent.name != "mini_npz" or result_path.parents[1].name != "exports":
        return None
    return result_path.parents[2].name


def iter_result_npzs(input_dir: str | Path, modes: Sequence[str] = DEFAULT_MODES) -> Iterable[Path]:
    root = Path(input_dir)
    mode_set = {
        _normalize_mode_name(str(mode))
        for mode in modes
        if _normalize_mode_name(str(mode)) not in GT_MODE_ALIASES
    }
    if not mode_set:
        return
    for result_path in sorted(root.rglob("results.npz")):
        mode = _result_mode(result_path)
        if mode is not None and mode in mode_set:
            yield result_path


def _output_file_name(options: FlowPlyOptions) -> str:
    suffix = "" if str(options.visualization_style) == "flow" else f"_{options.visualization_style}"
    return f"{options.flow_key}_{options.color_mode}{suffix}.ply"


def _default_output_path(result_path: Path, options: FlowPlyOptions) -> Path:
    export_dir = result_path.parents[1]
    return export_dir / options.output_subdir / _output_file_name(options)


def _mirrored_output_path(
    result_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    options: FlowPlyOptions,
) -> Path:
    export_dir = result_path.parents[1]
    try:
        relative_export_dir = export_dir.relative_to(input_dir)
    except ValueError:
        relative_export_dir = Path(*export_dir.parts[-4:])
    return output_dir / relative_export_dir / options.output_subdir / _output_file_name(options)


def process_result_npz(
    result_path: str | Path,
    options: FlowPlyOptions,
    *,
    output_path: str | Path | None = None,
) -> Path:
    result_path = Path(result_path)
    with np.load(result_path, allow_pickle=False) as payload:
        missing = [key for key in ("depth", "intrinsics", "extrinsics", options.flow_key) if key not in payload]
        if missing:
            raise KeyError(f"{result_path} is missing required arrays: {', '.join(missing)}")
        conf = np.asarray(payload["conf"], dtype=np.float32) if "conf" in payload else None
        rgb = np.asarray(payload["rgb"]) if "rgb" in payload else None
        if rgb is None and str(options.visualization_style) == "tracking":
            rgb = _load_rgb_for_result_npz(result_path)
        scene_flow = np.asarray(payload[options.flow_key], dtype=np.float32)
        scale = _scene_flow_scale_for_result_npz(result_path, options, payload)
        if scale != 1.0:
            scene_flow = scene_flow * np.float32(scale)
        points, colors = build_scene_flow_point_cloud(
            depth=np.asarray(payload["depth"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            scene_flow=scene_flow,
            conf=conf,
            options=options,
            rgb=rgb,
        )
    if points.shape[0] == 0:
        raise RuntimeError(f"No valid scene-flow points found in {result_path}")
    target_path = Path(output_path) if output_path is not None else _default_output_path(result_path, options)
    return write_point_cloud_ply(target_path, points, colors, overwrite=bool(options.overwrite))


def _wants_gt_mode(modes: Sequence[str]) -> bool:
    return any(_normalize_mode_name(str(mode)) in GT_MODE_ALIASES for mode in modes)


def _find_run_meta_path(input_dir: str | Path) -> Path | None:
    root = Path(input_dir).expanduser().resolve()
    for candidate in (root, *root.parents):
        meta_path = candidate / "run_meta.json"
        if meta_path.is_file():
            return meta_path
    return None


def _load_run_meta(input_dir: str | Path) -> tuple[Path, dict]:
    meta_path = _find_run_meta_path(input_dir)
    if meta_path is None:
        raise RuntimeError(
            "GT scene-flow PLY mode requires a run_meta.json from the evaluation output root."
        )
    with meta_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return meta_path, payload


def _run_meta_dataset_name(run_meta: dict) -> str:
    benchmark = str(run_meta.get("benchmark", ""))
    if benchmark == "movi_f" or "movi_f_root" in run_meta:
        return "movi_f"
    if "dynamic_replica_root" in run_meta:
        return "dynamic_replica"
    raise RuntimeError("GT mode currently supports MOVi-F and Dynamic Replica run_meta.json files only.")


def _clip_dirname(clip: dict) -> str:
    if clip.get("clip_dirname"):
        return str(clip["clip_dirname"])
    frame_indices = tuple(int(item) for item in clip.get("frame_indices", ()))
    if not frame_indices:
        raise ValueError(f"Clip metadata is missing frame_indices: {clip}")
    return f"{frame_indices[0]:05d}_{frame_indices[-1] + 1:05d}"


def _clip_base_dir(run_root: Path, run_meta: dict, clip: dict) -> Path:
    dataset_name = _run_meta_dataset_name(run_meta)
    return run_root / "model_results" / dataset_name / str(clip["scene_name"]) / _clip_dirname(clip)


def _result_clip_base_dir(result_path: Path) -> Path | None:
    if len(result_path.parents) < 4:
        return None
    return result_path.parents[3]


def _load_rgb_from_gt_meta(result_path: Path) -> np.ndarray | None:
    export_dir = result_path.parents[1]
    gt_meta_path = export_dir / "gt_meta.npz"
    if not gt_meta_path.is_file():
        return None
    with np.load(gt_meta_path, allow_pickle=True) as gt_meta:
        if "image_files" not in gt_meta:
            return None
        image_files = [str(path) for path in gt_meta["image_files"]]
    from evaluation.dynamic_replica_ray_benchmark import _load_rgb_image

    return np.stack([_load_rgb_image(path) for path in image_files], axis=0).astype(np.uint8)


def _clip_metadata_for_result(run_meta: dict, result_path: Path) -> dict | None:
    if len(result_path.parents) < 6:
        return None
    scene_name = result_path.parents[4].name
    clip_dirname = result_path.parents[3].name
    for clip in run_meta.get("clips", []):
        if str(clip.get("scene_name")) == scene_name and _clip_dirname(clip) == clip_dirname:
            return clip
    return None


def _load_rgb_for_result_npz(result_path: str | Path) -> np.ndarray | None:
    result_path = Path(result_path)
    cache_key = _result_clip_base_dir(result_path) or result_path
    if cache_key in _RGB_RESULT_CACHE:
        return _RGB_RESULT_CACHE[cache_key]

    rgb = _load_rgb_from_gt_meta(result_path)
    if rgb is None:
        meta_path = _find_run_meta_path(result_path)
        if meta_path is not None:
            with meta_path.open("r", encoding="utf-8") as file:
                run_meta = json.load(file)
            clip = _clip_metadata_for_result(run_meta, result_path)
            if clip is not None:
                rgb = _load_gt_scene_payload(run_meta, clip).get("rgb")
    _RGB_RESULT_CACHE[cache_key] = rgb
    return rgb


def _resolved_scene_flow_output_space(run_meta: dict, options: FlowPlyOptions) -> str:
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


def _should_apply_scene_flow_scale(run_meta: dict, options: FlowPlyOptions) -> bool:
    target_space = str(run_meta.get("scene_flow_target_space", "relative"))
    output_space = _resolved_scene_flow_output_space(run_meta, options)
    return target_space == "relative" and output_space == "metric"


def _result_clip_key(result_path: Path) -> str | None:
    if len(result_path.parents) < 5:
        return None
    return f"{result_path.parents[4].name}/{result_path.parents[3].name}"


def _load_scene_flow_scale_from_metric_results(meta_path: Path, run_meta: dict, result_path: Path) -> float | None:
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


def _scene_flow_scale_for_result_npz(result_path: Path, options: FlowPlyOptions, payload) -> float:
    meta_path = _find_run_meta_path(result_path)
    if meta_path is None:
        return 1.0
    with meta_path.open("r", encoding="utf-8") as file:
        run_meta = json.load(file)
    if not _should_apply_scene_flow_scale(run_meta, options):
        return 1.0
    if "scene_flow_scale" in payload:
        return _as_scalar_float(payload["scene_flow_scale"])
    metric_scale = _load_scene_flow_scale_from_metric_results(meta_path, run_meta, result_path)
    if metric_scale is not None:
        return float(metric_scale)
    clip = _clip_metadata_for_result(run_meta, result_path)
    if clip is None:
        return 1.0
    gt_payload = _load_gt_scene_payload(run_meta, clip)
    return _as_scalar_float(gt_payload.get("scene_flow_scale"), default=1.0)


def _scene_flow_scale_for_gt_payload(run_meta: dict, options: FlowPlyOptions, payload: dict) -> float:
    if not _should_apply_scene_flow_scale(run_meta, options):
        return 1.0
    return _as_scalar_float(payload.get("scene_flow_scale"), default=1.0)


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


def _gt_output_path(base_dir: Path, options: FlowPlyOptions) -> Path:
    return base_dir / "gt_fuse" / options.output_subdir / _output_file_name(options)


def _mirrored_gt_output_path(
    base_dir: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    options: FlowPlyOptions,
) -> Path:
    gt_dir = base_dir / "gt_fuse"
    try:
        relative_gt_dir = gt_dir.relative_to(input_dir)
    except ValueError:
        relative_gt_dir = Path(*gt_dir.parts[-4:])
    return output_dir / relative_gt_dir / options.output_subdir / _output_file_name(options)


def _load_movi_f_clip_sample_direct(dataset_root: str | Path, clip, *, target_space: str) -> dict:
    from evaluation.movi_f_ray_benchmark import _load_movi_f_scene_flow_adapter_class

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow_datasets is required to read MOVi-F GT data for fuse_gt mode."
        ) from exc

    sample_index = int(clip.sample_index)
    if sample_index < 0:
        raise ValueError(f"MOVi-F sample_index must be non-negative, got {sample_index}")

    MOViFSceneFlowAdapter = _load_movi_f_scene_flow_adapter_class()
    adapter = MOViFSceneFlowAdapter(
        builder_dir=dataset_root,
        split=str(clip.split),
        flow_target_space=str(target_space),
        chunk_size=None,
        pad_to_chunk_size=False,
        target_height=None,
        target_width=None,
    )
    builder = tfds.builder_from_directory(str(adapter.builder_dir))
    dataset = builder.as_dataset(split=adapter.split).skip(sample_index).take(1)
    iterator = iter(tfds.as_numpy(dataset))
    try:
        example = next(iterator)
    except StopIteration as exc:
        raise IndexError(f"MOVi-F sample_index={sample_index} is not present in split={clip.split}") from exc
    return dict(adapter._convert_example(example))


def _load_movi_f_gt_scene_payload(run_meta: dict, clip: dict, *, target_space: str) -> dict[str, np.ndarray]:
    from evaluation.movi_f_ray_benchmark import (
        MOViFClip,
        _positions_for_frame_ids,
        _slice_sample_array,
    )

    frame_indices = tuple(int(frame_id) for frame_id in clip["frame_indices"])
    clip_obj = MOViFClip(
        split=str(clip.get("split", run_meta.get("split", "validation"))),
        scene_name=str(clip["scene_name"]),
        sample_index=int(clip["sample_index"]),
        frame_indices=frame_indices,
    )
    sample = _load_movi_f_clip_sample_direct(
        run_meta["movi_f_root"],
        clip_obj,
        target_space=target_space,
    )
    positions = _positions_for_frame_ids(sample, clip_obj.frame_indices)

    rgb_key = "eval_rgbs" if "eval_rgbs" in sample else "rgbs"
    depth_key = "eval_gt_depth" if "eval_gt_depth" in sample else "gt_depth"
    depth_valid_key = "eval_gt_depth_valid_mask" if "eval_gt_depth_valid_mask" in sample else "gt_depth_valid_mask"
    intrinsics_key = "eval_intrinsics" if "eval_intrinsics" in sample else "intrinsics"
    scene_flow_key = "eval_scene_flow" if "eval_scene_flow" in sample else "scene_flow"
    deform_mask_key = "eval_deform_mask" if "eval_deform_mask" in sample else "deform_mask"

    radial_depth = _slice_sample_array(sample, depth_key, positions).astype(np.float32)
    intrinsics = _slice_sample_array(sample, intrinsics_key, positions).astype(np.float32)
    depth_valid = _slice_sample_array(sample, depth_valid_key, positions).astype(np.float32) > 0.5
    deform_valid = _slice_sample_array(sample, deform_mask_key, positions).astype(np.float32) > 0.5
    return {
        "depth": _radial_depth_to_z_depth_sequence(radial_depth, intrinsics),
        "intrinsics": intrinsics,
        "extrinsics": _slice_sample_array(sample, "extrinsics", positions).astype(np.float32),
        "scene_flow": _slice_sample_array(sample, scene_flow_key, positions).astype(np.float32),
        "valid_mask": (depth_valid & deform_valid).astype(np.float32),
        "rgb": _slice_sample_array(sample, rgb_key, positions).astype(np.uint8),
        "scene_flow_scale": _as_scalar_float(sample.get("scene_flow_scale", 1.0)),
    }


def _load_gt_scene_payload(run_meta: dict, clip: dict) -> dict[str, np.ndarray]:
    dataset_name = _run_meta_dataset_name(run_meta)
    target_space = str(run_meta.get("scene_flow_target_space", "relative"))
    frame_indices = tuple(int(frame_id) for frame_id in clip["frame_indices"])
    if dataset_name == "movi_f":
        return _load_movi_f_gt_scene_payload(run_meta, clip, target_space=target_space)

    from evaluation.dynamic_replica_ray_benchmark import (
        DynamicReplicaClip,
        _load_gt_depth_and_valid_mask,
        _load_rgb_image,
        load_dynamic_replica_clip_scene_data,
        load_dynamic_replica_clip_scene_flow_gt,
    )

    clip_obj = DynamicReplicaClip(
        split=str(clip.get("split", run_meta.get("split", "test"))),
        scene_name=str(clip["scene_name"]),
        camera_name=str(clip.get("camera_name", "left")),
        frame_indices=frame_indices,
    )
    scene_data = load_dynamic_replica_clip_scene_data(run_meta["dynamic_replica_root"], clip_obj)
    depths: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for depth_file in scene_data.depth_files:
        depth, valid = _load_gt_depth_and_valid_mask(depth_file)
        depths.append(np.asarray(depth, dtype=np.float32))
        valid_masks.append(np.asarray(valid, dtype=np.float32))
    scene_flow_payload = load_dynamic_replica_clip_scene_flow_gt(
        run_meta["dynamic_replica_root"],
        clip_obj,
        clip_length=int(run_meta.get("clip_length", len(frame_indices))),
        target_space=target_space,
    )
    return {
        "depth": np.stack(depths, axis=0).astype(np.float32),
        "intrinsics": np.asarray(scene_data.intrinsics, dtype=np.float32),
        "extrinsics": np.asarray(scene_data.extrinsics, dtype=np.float32),
        "scene_flow": np.asarray(scene_flow_payload["scene_flow"], dtype=np.float32),
        "valid_mask": np.stack(valid_masks, axis=0).astype(np.float32),
        "rgb": np.stack([_load_rgb_image(path) for path in scene_data.image_files], axis=0).astype(np.uint8),
        "scene_flow_scale": _as_scalar_float(scene_flow_payload.get("scene_flow_scale", 1.0)),
    }


def process_gt_fuse_outputs(
    input_dir: str | Path,
    options: FlowPlyOptions,
    *,
    output_dir: str | Path | None = None,
) -> list[Path]:
    input_root = Path(input_dir).expanduser().resolve()
    meta_path, run_meta = _load_run_meta(input_root)
    run_root = meta_path.parent
    output_root = None if output_dir is None else Path(output_dir)
    outputs: list[Path] = []
    errors: list[str] = []
    for clip in run_meta.get("clips", []):
        base_dir = _clip_base_dir(run_root, run_meta, clip)
        if not _paths_overlap(input_root, base_dir):
            continue
        try:
            payload = _load_gt_scene_payload(run_meta, clip)
            scene_flow = np.asarray(payload[options.flow_key], dtype=np.float32)
            scale = _scene_flow_scale_for_gt_payload(run_meta, options, payload)
            if scale != 1.0:
                scene_flow = scene_flow * np.float32(scale)
            points, colors = build_scene_flow_point_cloud(
                depth=payload["depth"],
                intrinsics=payload["intrinsics"],
                extrinsics=payload["extrinsics"],
                scene_flow=scene_flow,
                valid_mask=payload.get("valid_mask"),
                options=options,
                rgb=payload.get("rgb"),
            )
            if points.shape[0] == 0:
                raise RuntimeError(f"No valid GT scene-flow points found for {base_dir}")
            target_path = (
                _gt_output_path(base_dir, options)
                if output_root is None
                else _mirrored_gt_output_path(base_dir, input_dir=input_root, output_dir=output_root, options=options)
            )
            outputs.append(write_point_cloud_ply(target_path, points, colors, overwrite=bool(options.overwrite)))
        except Exception as exc:  # noqa: BLE001 - continue other clips and summarize failures.
            errors.append(f"{base_dir}: {exc}")
    if not outputs and _wants_gt_mode(options.modes):
        details = "\n".join(errors[:10])
        raise RuntimeError(f"No GT scene-flow PLY files were written from {input_root}.\n{details}")
    return outputs


def process_output_tree(
    input_dir: str | Path,
    options: FlowPlyOptions,
    *,
    output_dir: str | Path | None = None,
) -> list[Path]:
    input_root = Path(input_dir)
    output_root = None if output_dir is None else Path(output_dir)
    result_paths = list(iter_result_npzs(input_root, options.modes))
    gt_outputs = process_gt_fuse_outputs(input_root, options, output_dir=output_dir) if _wants_gt_mode(options.modes) else []
    if not result_paths and not gt_outputs:
        raise RuntimeError(f"No posed/unposed mini_npz results were found under {input_root}")

    outputs: list[Path] = []
    errors: list[str] = []
    for result_path in result_paths:
        target_path = (
            None
            if output_root is None
            else _mirrored_output_path(result_path, input_dir=input_root, output_dir=output_root, options=options)
        )
        try:
            outputs.append(process_result_npz(result_path, options, output_path=target_path))
        except KeyError as exc:
            errors.append(str(exc))
        except RuntimeError as exc:
            errors.append(str(exc))

    outputs.extend(gt_outputs)
    if not outputs:
        details = "\n".join(errors[:10])
        raise RuntimeError(f"No scene-flow PLY files were written from {input_root}.\n{details}")
    return outputs


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(_normalize_mode_name(item) for item in value.split(",") if item.strip())
    if not modes:
        raise argparse.ArgumentTypeError("modes cannot be empty")
    return modes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process DA3 evaluation exports and write aggregated point-only PLYs "
            "colored by scene-flow vectors."
        )
    )
    parser.add_argument("input_dir", nargs="?", help="Evaluation output root, e.g. .../da3_stitched.")
    parser.add_argument("--input_dir", dest="input_dir_flag", help="Evaluation output root; overrides positional input_dir.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional mirror output root. Defaults to writing beside each results.npz.")
    parser.add_argument("--flow_key", type=str, default="scene_flow", choices=["scene_flow", "scene_flow_cam"])
    parser.add_argument("--color_mode", type=str, default="direction_magnitude", choices=SUPPORTED_COLOR_MODES)
    parser.add_argument("--magnitude_percentile", type=float, default=95.0)
    parser.add_argument("--flow_scale", type=float, default=1.0, help="Optional extra multiplier applied after scene-flow space conversion.")
    parser.add_argument("--scene_flow_output_space", type=str, default="auto", choices=SUPPORTED_SCENE_FLOW_OUTPUT_SPACES, help="Output flow vectors in native or metric space. auto follows run_meta scene_flow_eval_space.")
    parser.add_argument("--visualization_style", type=str, default="flow", choices=SUPPORTED_VISUALIZATION_STYLES)
    parser.add_argument("--trail_points", type=int, default=20, help="Interpolated points per flow trail for tracking visualization.")
    parser.add_argument("--moving_threshold", type=float, default=0.01, help="Minimum flow magnitude for drawing tracking trails.")
    parser.add_argument("--trail_all_points", action="store_true", help="Draw tracking trails for all valid points, including near-static points.")
    parser.add_argument("--max_points_per_clip", type=int, default=300_000, help="0 or negative disables point subsampling.")
    parser.add_argument("--pixel_stride", type=int, default=1, help="Use every Nth pixel in x/y before random subsampling.")
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--min_conf", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default=DEFAULT_MODES,
        help="Comma-separated export modes. Supports unposed, posed, and gt/gt_fuse/fuse_gt. Default: unposed,posed.",
    )
    parser.add_argument("--output_subdir", type=str, default="scene_flow_ply")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing PLY files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir_flag or args.input_dir
    if not input_dir:
        raise SystemExit("input_dir is required")
    options = FlowPlyOptions(
        flow_key=str(args.flow_key),
        color_mode=str(args.color_mode),
        magnitude_percentile=float(args.magnitude_percentile),
        flow_scale=float(args.flow_scale),
        max_points_per_clip=int(args.max_points_per_clip),
        min_depth=float(args.min_depth),
        max_depth=None if args.max_depth is None else float(args.max_depth),
        min_conf=None if args.min_conf is None else float(args.min_conf),
        pixel_stride=int(args.pixel_stride),
        seed=int(args.seed),
        modes=tuple(args.modes),
        output_subdir=str(args.output_subdir),
        overwrite=bool(args.overwrite),
        visualization_style=str(args.visualization_style),
        trail_points=int(args.trail_points),
        trail_only_moving=not bool(args.trail_all_points),
        moving_threshold=float(args.moving_threshold),
        scene_flow_output_space=str(args.scene_flow_output_space),
    )
    outputs = process_output_tree(input_dir, options, output_dir=args.output_dir)
    print(f"Wrote or reused {len(outputs)} scene-flow PLY files.")
    for path in outputs[:20]:
        print(path)
    if len(outputs) > 20:
        print(f"... {len(outputs) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
