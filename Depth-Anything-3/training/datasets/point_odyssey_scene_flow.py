from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PATCH_SIZE = 14


@dataclass
class SampleInfo:
    scene_name: str
    scene_dir: Path
    start_idx: int
    end_idx: int


def _is_valid_numpy_array(
    array: object,
    *,
    ndim: int,
    min_frames: int,
    last_dim: Optional[int] = None,
) -> bool:
    if not isinstance(array, np.ndarray):
        return False
    if array.ndim != ndim:
        return False
    if array.shape[0] < min_frames:
        return False
    if last_dim is not None and array.shape[-1] != last_dim:
        return False
    return True


def _validate_scene_flow_annotation(anno: np.lib.npyio.NpzFile, min_frames: int) -> Tuple[bool, str]:
    intrinsics = anno['intrinsics'] if 'intrinsics' in anno else None
    if not _is_valid_numpy_array(intrinsics, ndim=3, min_frames=min_frames) or intrinsics.shape[1:] != (3, 3):
        return False, 'intrinsics'

    extrinsics = anno['extrinsics'] if 'extrinsics' in anno else None
    if not _is_valid_numpy_array(extrinsics, ndim=3, min_frames=min_frames) or extrinsics.shape[1:] not in {(4, 4), (3, 4)}:
        return False, 'extrinsics'

    trajs_2d = anno['trajs_2d'] if 'trajs_2d' in anno else None
    if not _is_valid_numpy_array(trajs_2d, ndim=3, min_frames=min_frames, last_dim=2):
        return False, 'trajs_2d'

    trajs_3d = anno['trajs_3d'] if 'trajs_3d' in anno else None
    if not _is_valid_numpy_array(trajs_3d, ndim=3, min_frames=min_frames, last_dim=3):
        return False, 'trajs_3d'

    if trajs_2d.shape[1] != trajs_3d.shape[1]:
        return False, 'trajs_points_mismatch'

    valids = anno['valids'] if 'valids' in anno else None
    if not _is_valid_numpy_array(valids, ndim=2, min_frames=min_frames) or valids.shape[1] != trajs_3d.shape[1]:
        return False, 'valids'

    visibs = anno['visibs'] if 'visibs' in anno else None
    if visibs is not None:
        if not _is_valid_numpy_array(visibs, ndim=2, min_frames=min_frames) or visibs.shape[1] != trajs_3d.shape[1]:
            return False, 'visibs'

    return True, ''


def compute_resize_params(
    src_height: int,
    src_width: int,
    target_height: int,
    target_width: int,
    *,
    process_res: Optional[int] = None,
    process_res_method: Optional[str] = None,
    patch_size: int = PATCH_SIZE,
) -> Tuple[int, int, int, int, int, int]:
    method = None if process_res_method is None else str(process_res_method)
    if method == "upper_bound_resize":
        if process_res is None or int(process_res) <= 0:
            raise ValueError(f"process_res must be positive for upper_bound_resize, got {process_res}")

        target_size = int(process_res)
        longest_side = max(int(src_height), int(src_width))
        scale = target_size / float(max(longest_side, 1))
        resize_height = max(1, int(round(src_height * scale)))
        resize_width = max(1, int(round(src_width * scale)))
        output_height = max(1, _nearest_multiple(resize_height, patch_size))
        output_width = max(1, _nearest_multiple(resize_width, patch_size))
        return output_height, output_width, 0, 0, output_height, output_width

    scale_h = target_height / src_height
    scale_w = target_width / src_width
    scale = max(scale_h, scale_w)

    resize_height = max(int(np.ceil(src_height * scale)), target_height)
    resize_width = max(int(np.ceil(src_width * scale)), target_width)
    crop_top = (resize_height - target_height) // 2
    crop_left = (resize_width - target_width) // 2
    return resize_height, resize_width, crop_top, crop_left, target_height, target_width


def resize_and_center_crop_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    *,
    process_res: Optional[int] = None,
    process_res_method: Optional[str] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    src_height, src_width = image.shape[:2]
    params = compute_resize_params(
        src_height,
        src_width,
        target_height,
        target_width,
        process_res=process_res,
        process_res_method=process_res_method,
    )
    resize_height, resize_width, crop_top, crop_left, crop_h, crop_w = params
    resized = cv2.resize(image, (resize_width, resize_height), interpolation=interpolation)
    cropped = resized[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
    return cropped, params


def _nearest_multiple(value: int, divisor: int) -> int:
    down = max(divisor, (value // divisor) * divisor)
    up = max(divisor, down + divisor)
    return up if abs(up - value) <= abs(value - down) else down


def adjust_intrinsics_for_resize_and_crop(
    intrinsics: np.ndarray,
    src_height: int,
    src_width: int,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
) -> np.ndarray:
    adjusted = intrinsics.copy().astype(np.float32)
    scale_x = resize_width / src_width
    scale_y = resize_height / src_height
    adjusted[0, 0] *= scale_x
    adjusted[1, 1] *= scale_y
    adjusted[0, 2] *= scale_x
    adjusted[1, 2] *= scale_y
    adjusted[0, 2] -= crop_left
    adjusted[1, 2] -= crop_top
    return adjusted


def get_trunk_indices(
    total_frames: int,
    chunk_size: int,
    overlap: int,
) -> List[Tuple[int, int]]:
    if total_frames <= chunk_size:
        return [(0, total_frames)]

    step = chunk_size - overlap
    indices: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        indices.append((start, end))
        if end >= total_frames:
            break
        start += step

    if indices:
        last_start, last_end = indices[-1]
        if last_end == total_frames and last_end - last_start < chunk_size:
            indices[-1] = (max(0, total_frames - chunk_size), total_frames)
    return indices


def _ensure_extrinsic_4x4(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float32)
    if extrinsic.shape == (3, 4):
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :4] = extrinsic.astype(np.float32)
        return ext
    raise ValueError(f"Unsupported extrinsic shape: {extrinsic.shape}")


def _find_rgb_path(scene_dir: Path, frame_idx: int) -> Path:
    jpg_path = scene_dir / 'rgbs' / f'rgb_{frame_idx:05d}.jpg'
    if jpg_path.exists():
        return jpg_path
    png_path = scene_dir / 'rgbs' / f'rgb_{frame_idx:05d}.png'
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Cannot find RGB for frame {frame_idx} under {scene_dir}")


def _find_depth_path(scene_dir: Path, frame_idx: int) -> Path:
    png_path = scene_dir / 'depths' / f'depth_{frame_idx:05d}.png'
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Cannot find depth for frame {frame_idx} under {scene_dir}")


def _load_point_odyssey_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth from {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 65535.0 * 1000.0
    else:
        depth = depth.astype(np.float32)
    return depth


def _unproject_depth_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"depth should have shape [H, W], got {depth.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics should have shape [3, 3], got {intrinsics.shape}")

    height, width = depth.shape
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij',
    )
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x = (xs - np.float32(cx)) * depth.astype(np.float32) / np.float32(max(fx, 1e-8))
    y = (ys - np.float32(cy)) * depth.astype(np.float32) / np.float32(max(fy, 1e-8))
    return np.stack([x, y, depth.astype(np.float32)], axis=-1).astype(np.float32)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transform = _ensure_extrinsic_4x4(transform)
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    flat = points.reshape(-1, 3).astype(np.float32)
    transformed = flat @ rotation.T + translation[None]
    return transformed.reshape(points.shape)


def compute_reference_points_from_depth_sequence(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    valid_masks: np.ndarray,
    reference_w2c: Optional[np.ndarray] = None,
) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    valid_masks = np.asarray(valid_masks)
    if depths.ndim != 3:
        raise ValueError(f"depths should have shape [T, H, W], got {depths.shape}")
    if intrinsics.shape != (depths.shape[0], 3, 3):
        raise ValueError(f"intrinsics should have shape [T, 3, 3], got {intrinsics.shape}")
    if extrinsics.shape[0] != depths.shape[0]:
        raise ValueError(f"extrinsics should have shape [T, 4, 4] or [T, 3, 4], got {extrinsics.shape}")
    if valid_masks.shape != depths.shape:
        raise ValueError(f"valid_masks should have shape {depths.shape}, got {valid_masks.shape}")

    reference_w2c = _ensure_extrinsic_4x4(reference_w2c if reference_w2c is not None else extrinsics[0])
    points_ref = np.zeros(depths.shape + (3,), dtype=np.float32)
    valid = (valid_masks > 0.5) & np.isfinite(depths) & (depths > 1e-6)

    for frame_idx in range(depths.shape[0]):
        points_cam = _unproject_depth_map(depths[frame_idx], intrinsics[frame_idx])
        current_c2w = np.linalg.inv(_ensure_extrinsic_4x4(extrinsics[frame_idx])).astype(np.float32)
        points_world = _transform_points(points_cam, current_c2w)
        points_ref[frame_idx] = _transform_points(points_world, reference_w2c)

    points_ref[~valid] = 0.0
    return points_ref


def compute_geometry_scale_from_points(
    points_ref: np.ndarray,
    valid_masks: np.ndarray,
    scale_stat: str = 'mean_radius',
    scale_eps: float = 1e-6,
) -> float:
    points_ref = np.asarray(points_ref, dtype=np.float32)
    valid_masks = np.asarray(valid_masks)
    if points_ref.ndim != 4 or points_ref.shape[-1] != 3:
        raise ValueError(f"points_ref should have shape [T, H, W, 3], got {points_ref.shape}")
    if valid_masks.shape != points_ref.shape[:-1]:
        raise ValueError(f"valid_masks should have shape {points_ref.shape[:-1]}, got {valid_masks.shape}")

    valid = (valid_masks > 0.5) & np.isfinite(points_ref).all(axis=-1)
    if not np.any(valid):
        return float(max(scale_eps, 1.0))

    valid_points = points_ref[valid].reshape(-1, 3).astype(np.float32)
    center = np.mean(valid_points, axis=0, dtype=np.float64).astype(np.float32)
    radii = np.linalg.norm(valid_points - center[None], axis=-1).astype(np.float32)
    radii = radii[np.isfinite(radii)]
    if radii.size == 0:
        return float(max(scale_eps, 1.0))

    if scale_stat == 'mean_radius':
        scale = float(np.mean(radii))
    elif scale_stat in {'median_radius', 'median'}:
        scale = float(np.median(radii))
    elif scale_stat in {'p90_radius', 'p90'}:
        scale = float(np.percentile(radii, 90.0))
    elif scale_stat in {'max_radius', 'max'}:
        scale = float(np.max(radii))
    else:
        raise ValueError(f"Unsupported geometry scale_stat: {scale_stat}")

    return float(max(scale, scale_eps))


def _apply_2d_resize_crop(
    trajs_2d: np.ndarray,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
    src_height: int,
    src_width: int,
) -> np.ndarray:
    trajs_2d = trajs_2d.copy().astype(np.float32)
    scale_x = resize_width / src_width
    scale_y = resize_height / src_height
    trajs_2d[..., 0] *= scale_x
    trajs_2d[..., 1] *= scale_y
    trajs_2d[..., 0] -= crop_left
    trajs_2d[..., 1] -= crop_top
    invalid = ~np.isfinite(trajs_2d)
    trajs_2d[invalid] = 0.0
    return trajs_2d


def build_track_mask(
    valids: np.ndarray,
    visibs: Optional[np.ndarray],
    trajs_2d: np.ndarray,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    track_mask = np.zeros_like(valids, dtype=bool)
    if len(valids) == 0:
        return track_mask

    in_bounds = (
        (trajs_2d[..., 0] >= 0)
        & (trajs_2d[..., 0] < image_width)
        & (trajs_2d[..., 1] >= 0)
        & (trajs_2d[..., 1] < image_height)
    )

    if len(valids) > 1:
        current_valid = valids[:-1] > 0.5
        next_valid = valids[1:] > 0.5
        track_mask[:-1] = current_valid & next_valid & in_bounds[:-1]
        if visibs is not None:
            track_mask[:-1] &= visibs[:-1] > 0.5
    return track_mask


def rasterize_scene_flow_dense_map(
    trajs_2d: np.ndarray,
    trajs_3d_world: np.ndarray,
    track_mask: np.ndarray,
    frame0_w2c: np.ndarray,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_frames = trajs_2d.shape[0]
    scene_flow = np.zeros((num_frames, image_height, image_width, 3), dtype=np.float32)
    deform_mask = np.zeros((num_frames, image_height, image_width), dtype=np.float32)

    if num_frames <= 1:
        return scene_flow, deform_mask

    frame0_w2c = _ensure_extrinsic_4x4(frame0_w2c)
    rot_frame0 = frame0_w2c[:3, :3].astype(np.float32)

    delta_world = (trajs_3d_world[1:] - trajs_3d_world[:-1]).astype(np.float32)
    delta_frame0 = np.einsum('ij,tnj->tni', rot_frame0, delta_world).astype(np.float32)

    for frame_idx in range(num_frames - 1):
        valid = track_mask[frame_idx].astype(bool)
        if not np.any(valid):
            continue

        uv = trajs_2d[frame_idx, valid]
        flow = delta_frame0[frame_idx, valid]
        finite = np.isfinite(uv).all(axis=-1) & np.isfinite(flow).all(axis=-1)
        if not np.any(finite):
            continue

        uv = uv[finite]
        flow = flow[finite]
        xs = np.rint(uv[:, 0]).astype(np.int64)
        ys = np.rint(uv[:, 1]).astype(np.int64)
        in_bounds = (
            (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
        )
        if not np.any(in_bounds):
            continue

        xs = xs[in_bounds]
        ys = ys[in_bounds]
        flow = flow[in_bounds]

        flow_sum = np.zeros((image_height, image_width, 3), dtype=np.float32)
        count = np.zeros((image_height, image_width), dtype=np.float32)
        for channel in range(3):
            np.add.at(flow_sum[..., channel], (ys, xs), flow[:, channel])
        np.add.at(count, (ys, xs), 1.0)

        occupied = count > 0
        scene_flow[frame_idx, occupied] = flow_sum[occupied] / count[occupied, None]
        deform_mask[frame_idx, occupied] = 1.0

    return scene_flow, deform_mask


def compute_scene_flow_scale(
    scene_flow: np.ndarray,
    deform_mask: np.ndarray,
    scale_stat: str = "p95",
    scale_eps: float = 1e-6,
) -> float:
    if scene_flow.ndim != 4 or scene_flow.shape[-1] != 3:
        raise ValueError(f"scene_flow should have shape [T, H, W, 3], got {scene_flow.shape}")
    if deform_mask.shape != scene_flow.shape[:-1]:
        raise ValueError(
            f"deform_mask should have shape {scene_flow.shape[:-1]}, got {deform_mask.shape}"
        )

    valid = (deform_mask > 0.5) & np.isfinite(scene_flow).all(axis=-1)
    if not np.any(valid):
        return float(max(scale_eps, 1.0))

    magnitudes = np.linalg.norm(scene_flow[valid], axis=-1).astype(np.float32)
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0.0)]
    if magnitudes.size == 0:
        return float(max(scale_eps, 1.0))

    if scale_stat == "p95":
        scale = float(np.percentile(magnitudes, 95.0))
    elif scale_stat == "median":
        scale = float(np.median(magnitudes))
    elif scale_stat == "mean":
        scale = float(np.mean(magnitudes))
    elif scale_stat == "max":
        scale = float(np.max(magnitudes))
    else:
        raise ValueError(f"Unsupported scene flow scale_stat: {scale_stat}")

    return float(max(scale, scale_eps))


def normalize_scene_flow_relative(
    scene_flow: np.ndarray,
    deform_mask: np.ndarray,
    scale_stat: str = "p95",
    scale_eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    flow_scale = compute_scene_flow_scale(
        scene_flow=scene_flow,
        deform_mask=deform_mask,
        scale_stat=scale_stat,
        scale_eps=scale_eps,
    )
    return normalize_scene_flow_with_scale(scene_flow, flow_scale), flow_scale


def normalize_scene_flow_with_scale(scene_flow: np.ndarray, flow_scale: float) -> np.ndarray:
    return (scene_flow.astype(np.float32) / np.float32(flow_scale)).astype(np.float32)


def _pad_last_frame(array: np.ndarray, target_length: int) -> np.ndarray:
    pad = target_length - array.shape[0]
    if pad <= 0:
        return array
    last = array[-1:]
    repeats = np.repeat(last, pad, axis=0)
    return np.concatenate([array, repeats], axis=0)


def _pad_zero(array: np.ndarray, target_length: int) -> np.ndarray:
    pad = target_length - array.shape[0]
    if pad <= 0:
        return array
    pad_shape = (pad,) + array.shape[1:]
    zeros = np.zeros(pad_shape, dtype=array.dtype)
    return np.concatenate([array, zeros], axis=0)


class PointOdysseySceneFlowDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        chunk_size: int = 49,
        overlap: int = 24,
        target_height: int = 532,
        target_width: int = 532,
        process_res: Optional[int] = None,
        process_res_method: Optional[str] = None,
        max_scenes: Optional[int] = None,
        pad_to_chunk_size: bool = True,
        flow_target_space: str = 'metric',
        relative_scale_source: str = 'flow_p95',
        geometry_scale_stat: str = 'mean_radius',
        flow_scale_stat: str = 'p95',
        flow_scale_eps: float = 1e-6,
        flow_target_amplification: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.target_height = target_height
        self.target_width = target_width
        self.process_res = None if process_res is None else int(process_res)
        self.process_res_method = None if process_res_method is None else str(process_res_method)
        self.max_scenes = max_scenes
        self.pad_to_chunk_size = pad_to_chunk_size
        self.flow_target_space = flow_target_space
        self.relative_scale_source = str(relative_scale_source)
        self.geometry_scale_stat = str(geometry_scale_stat)
        self.flow_scale_stat = flow_scale_stat
        self.flow_scale_eps = float(flow_scale_eps)
        self.flow_target_amplification = float(flow_target_amplification)
        if self.flow_target_space not in {'metric', 'relative'}:
            raise ValueError(f"Unsupported flow_target_space: {self.flow_target_space}")
        if self.relative_scale_source not in {'flow_p95', 'geometry_radius'}:
            raise ValueError(f"Unsupported relative_scale_source: {self.relative_scale_source}")

        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        scene_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
        if max_scenes is not None:
            scene_dirs = scene_dirs[:max_scenes]

        self.samples: List[SampleInfo] = []
        self._skipped_scene_names: set[str] = set()
        for scene_dir in scene_dirs:
            anno_path = scene_dir / 'anno.npz'
            if not anno_path.exists():
                continue
            with np.load(anno_path) as anno:
                intrinsics = anno['intrinsics'] if 'intrinsics' in anno else None
                if not isinstance(intrinsics, np.ndarray) or intrinsics.ndim != 3:
                    warnings.warn(f"Skipping scene {scene_dir.name}: malformed intrinsics in {anno_path}", RuntimeWarning)
                    continue
                total_frames = int(intrinsics.shape[0])
                is_valid_scene, invalid_reason = _validate_scene_flow_annotation(anno, total_frames)
                if not is_valid_scene:
                    warnings.warn(
                        f"Skipping scene {scene_dir.name}: malformed {invalid_reason} in {anno_path}",
                        RuntimeWarning,
                    )
                    continue
            for start_idx, end_idx in get_trunk_indices(total_frames, chunk_size, overlap):
                self.samples.append(
                    SampleInfo(
                        scene_name=scene_dir.name,
                        scene_dir=scene_dir,
                        start_idx=start_idx,
                        end_idx=end_idx,
                    )
                )

        if not self.samples:
            raise RuntimeError(f"No valid PointOdyssey samples found under {split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _warn_skip(self, scene_name: str, reason: str) -> None:
        if scene_name in self._skipped_scene_names:
            return
        self._skipped_scene_names.add(scene_name)
        warnings.warn(f"Skipping malformed scene-flow sample {scene_name}: {reason}", RuntimeWarning)

    def _compute_scene_flow_targets(
        self,
        *,
        scene_flow: np.ndarray,
        deform_mask: np.ndarray,
        depths: Optional[np.ndarray],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        valid_masks: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float, float]:
        flow_p95_scale = compute_scene_flow_scale(
            scene_flow=scene_flow,
            deform_mask=deform_mask,
            scale_stat=self.flow_scale_stat,
            scale_eps=self.flow_scale_eps,
        )
        scene_flow_scale = flow_p95_scale
        normalized_scene_flow = scene_flow.astype(np.float32)

        if self.flow_target_space == 'relative':
            if self.relative_scale_source == 'geometry_radius':
                if depths is None or valid_masks is None:
                    raise ValueError('geometry_radius relative scene flow requires depth-derived reference points')
                points_ref = compute_reference_points_from_depth_sequence(
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    valid_masks=valid_masks,
                    reference_w2c=extrinsics[0],
                )
                scene_flow_scale = compute_geometry_scale_from_points(
                    points_ref=points_ref,
                    valid_masks=valid_masks,
                    scale_stat=self.geometry_scale_stat,
                    scale_eps=self.flow_scale_eps,
                )
                normalized_scene_flow = normalize_scene_flow_with_scale(scene_flow, scene_flow_scale)
            else:
                normalized_scene_flow = normalize_scene_flow_with_scale(scene_flow, flow_p95_scale)

        if self.flow_target_amplification != 1.0:
            normalized_scene_flow = normalized_scene_flow * np.float32(self.flow_target_amplification)

        return normalized_scene_flow, scene_flow_scale, flow_p95_scale

    def _load_sample(self, sample: SampleInfo) -> Dict[str, torch.Tensor]:
        anno_path = sample.scene_dir / 'anno.npz'
        with np.load(anno_path) as anno:
            is_valid, invalid_reason = _validate_scene_flow_annotation(anno, sample.end_idx)
            if not is_valid:
                raise ValueError(f"malformed {invalid_reason} in {anno_path}")
            intrinsics = anno['intrinsics'][sample.start_idx:sample.end_idx].astype(np.float32)
            extrinsics = anno['extrinsics'][sample.start_idx:sample.end_idx].astype(np.float32)
            trajs_2d = anno['trajs_2d'][sample.start_idx:sample.end_idx].astype(np.float32)
            trajs_3d = anno['trajs_3d'][sample.start_idx:sample.end_idx].astype(np.float32)
            valids = anno['valids'][sample.start_idx:sample.end_idx].astype(np.float32)
            visibs = anno['visibs'][sample.start_idx:sample.end_idx].astype(np.float32) if 'visibs' in anno else None

        first_rgb = cv2.imread(str(_find_rgb_path(sample.scene_dir, sample.start_idx)), cv2.IMREAD_COLOR)
        if first_rgb is None:
            raise RuntimeError(f"Failed to read first RGB for {sample.scene_name}")
        first_rgb = cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)
        src_height, src_width = first_rgb.shape[:2]
        resize_height, resize_width, crop_top, crop_left, output_height, output_width = compute_resize_params(
            src_height,
            src_width,
            self.target_height,
            self.target_width,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )

        adjusted_intrinsics = []
        rgbs = []
        depths = []
        depth_valid_masks = []
        need_geometry_scale = self.flow_target_space == 'relative' and self.relative_scale_source == 'geometry_radius'
        for frame_idx in range(sample.start_idx, sample.end_idx):
            rgb = cv2.imread(str(_find_rgb_path(sample.scene_dir, frame_idx)), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB frame {frame_idx} for {sample.scene_name}")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            rgb = rgb[crop_top:crop_top + output_height, crop_left:crop_left + output_width]
            rgbs.append(rgb)
            if need_geometry_scale:
                depth = _load_point_odyssey_depth(_find_depth_path(sample.scene_dir, frame_idx))
                depth = cv2.resize(depth, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)
                depth = depth[crop_top:crop_top + output_height, crop_left:crop_left + output_width]
                valid_mask = np.isfinite(depth) & (depth > 1e-6)
                depths.append(depth.astype(np.float32))
                depth_valid_masks.append(valid_mask.astype(np.float32))

        for intrinsic in intrinsics:
            adjusted_intrinsics.append(
                adjust_intrinsics_for_resize_and_crop(
                    intrinsic,
                    src_height,
                    src_width,
                    resize_height,
                    resize_width,
                    crop_top,
                    crop_left,
                )
            )

        trajs_2d = _apply_2d_resize_crop(
            trajs_2d,
            resize_height,
            resize_width,
            crop_top,
            crop_left,
            src_height,
            src_width,
        )
        track_mask = build_track_mask(
            valids=valids,
            visibs=visibs,
            trajs_2d=trajs_2d,
            image_height=output_height,
            image_width=output_width,
        )
        scene_flow, deform_mask = rasterize_scene_flow_dense_map(
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d,
            track_mask=track_mask,
            frame0_w2c=extrinsics[0],
            image_height=output_height,
            image_width=output_width,
        )

        rgbs_arr = np.stack(rgbs, axis=0).astype(np.uint8)
        intrinsics_arr = np.stack(adjusted_intrinsics, axis=0).astype(np.float32)
        extrinsics_arr = np.stack([_ensure_extrinsic_4x4(ext) for ext in extrinsics], axis=0).astype(np.float32)
        sequence_length = rgbs_arr.shape[0]
        depths_arr = np.stack(depths, axis=0).astype(np.float32) if depths else None
        depth_valid_masks_arr = np.stack(depth_valid_masks, axis=0).astype(np.float32) if depth_valid_masks else None

        scene_flow, scene_flow_scale, flow_p95_scale = self._compute_scene_flow_targets(
            scene_flow=scene_flow,
            deform_mask=deform_mask,
            depths=depths_arr,
            intrinsics=intrinsics_arr,
            extrinsics=extrinsics_arr,
            valid_masks=depth_valid_masks_arr,
        )

        if self.pad_to_chunk_size and sequence_length < self.chunk_size:
            rgbs_arr = _pad_last_frame(rgbs_arr, self.chunk_size)
            intrinsics_arr = _pad_last_frame(intrinsics_arr, self.chunk_size)
            extrinsics_arr = _pad_last_frame(extrinsics_arr, self.chunk_size)
            if depths_arr is not None:
                depths_arr = _pad_last_frame(depths_arr, self.chunk_size)
            if depth_valid_masks_arr is not None:
                depth_valid_masks_arr = _pad_last_frame(depth_valid_masks_arr, self.chunk_size)
            scene_flow = _pad_zero(scene_flow, self.chunk_size)
            deform_mask = _pad_zero(deform_mask, self.chunk_size)

        sample_dict = {
            'rgbs': torch.from_numpy(rgbs_arr),
            'intrinsics': torch.from_numpy(intrinsics_arr).float(),
            'extrinsics': torch.from_numpy(extrinsics_arr).float(),
            'scene_flow': torch.from_numpy(scene_flow).float(),
            'scene_flow_scale': torch.tensor(scene_flow_scale, dtype=torch.float32),
            'flow_p95_scale': torch.tensor(flow_p95_scale, dtype=torch.float32),
            'flow_target_amplification': torch.tensor(float(self.flow_target_amplification), dtype=torch.float32),
            'deform_mask': torch.from_numpy(deform_mask).float(),
            'sequence_length': torch.tensor(sequence_length, dtype=torch.long),
            'scene_name': sample.scene_name,
            'start_idx': torch.tensor(sample.start_idx, dtype=torch.long),
            'end_idx': torch.tensor(sample.end_idx, dtype=torch.long),
        }
        if depths_arr is not None and depth_valid_masks_arr is not None:
            sample_dict['gt_depth'] = torch.from_numpy(depths_arr).float()
            sample_dict['gt_depth_valid_mask'] = torch.from_numpy(depth_valid_masks_arr).float()
        return sample_dict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.samples:
            raise RuntimeError('No valid PointOdyssey scene-flow samples are available')

        for offset in range(len(self.samples)):
            sample = self.samples[(idx + offset) % len(self.samples)]
            try:
                return self._load_sample(sample)
            except (FileNotFoundError, ValueError, IndexError, RuntimeError) as exc:
                self._warn_skip(sample.scene_name, str(exc))
                continue

        raise RuntimeError('All PointOdyssey scene-flow samples are invalid or unreadable')


def create_point_odyssey_scene_flow_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 1,
    chunk_size: int = 49,
    overlap: int = 24,
    target_height: int = 532,
    target_width: int = 532,
    max_scenes: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    flow_target_space: str = 'metric',
    relative_scale_source: str = 'flow_p95',
    geometry_scale_stat: str = 'mean_radius',
    flow_scale_stat: str = 'p95',
    flow_scale_eps: float = 1e-6,
) -> DataLoader:
    dataset = PointOdysseySceneFlowDataset(
        data_root=data_root,
        split=split,
        chunk_size=chunk_size,
        overlap=overlap,
        target_height=target_height,
        target_width=target_width,
        max_scenes=max_scenes,
        flow_target_space=flow_target_space,
        relative_scale_source=relative_scale_source,
        geometry_scale_stat=geometry_scale_stat,
        flow_scale_stat=flow_scale_stat,
        flow_scale_eps=flow_scale_eps,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
