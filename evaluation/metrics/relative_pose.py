from __future__ import annotations

import numpy as np


def _as_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim != 3:
        raise ValueError(f"Expected extrinsics with shape [N,3/4,4], got {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros(extrinsics.shape[:-2] + (1, 4), dtype=np.float64)
        bottom[..., 0, 3] = 1.0
        return np.concatenate([extrinsics, bottom], axis=-2)
    raise ValueError(f"Expected extrinsics with shape [N,3/4,4], got {extrinsics.shape}")


def _rotation_angle_deg(rotation_errors: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation_errors, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def _translation_angle_deg(
    pred_translation: np.ndarray,
    gt_translation: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    pred_norm = np.linalg.norm(pred_translation, axis=-1)
    gt_norm = np.linalg.norm(gt_translation, axis=-1)
    valid = (pred_norm > eps) & (gt_norm > eps)
    angles = np.full(pred_norm.shape, np.nan, dtype=np.float64)
    if valid.any():
        dot = np.sum(pred_translation[valid] * gt_translation[valid], axis=-1)
        cos_angle = np.clip(dot / (pred_norm[valid] * gt_norm[valid]), -1.0, 1.0)
        angles[valid] = np.rad2deg(np.arccos(cos_angle))
    return angles, valid


def _relative_transforms(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = _as_homogeneous(extrinsics)
    num_frames = int(extrinsics.shape[0])
    pair_i, pair_j = np.triu_indices(num_frames, k=1)
    if pair_i.size == 0:
        return np.zeros((0, 4, 4), dtype=np.float64)
    return np.linalg.inv(extrinsics[pair_i]) @ extrinsics[pair_j]


def _align_to_first_camera(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = _as_homogeneous(extrinsics)
    return extrinsics @ np.linalg.inv(extrinsics[0])[None]


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def _safe_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else 0.0


def _safe_accuracy_below(values: np.ndarray, threshold: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values < float(threshold))) if values.size else 0.0


def _normalized_translation_errors(
    pred_translation: np.ndarray,
    gt_translation: np.ndarray,
    *,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    gt_norm = np.linalg.norm(gt_translation, axis=-1)
    valid = gt_norm > eps
    errors = np.full(gt_norm.shape, np.nan, dtype=np.float64)
    if valid.any():
        delta = float(scale) * pred_translation[valid] - gt_translation[valid]
        errors[valid] = np.linalg.norm(delta, axis=-1) / gt_norm[valid]
    return errors, valid


def _optimal_translation_scale(
    pred_translation: np.ndarray,
    gt_translation: np.ndarray,
    valid: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    if not np.any(valid):
        return 1.0
    pred = pred_translation[valid].reshape(-1, 3)
    gt = gt_translation[valid].reshape(-1, 3)
    denom = float(np.sum(pred * pred))
    if not np.isfinite(denom) or denom <= eps:
        return 1.0
    scale = float(np.sum(pred * gt) / denom)
    if not np.isfinite(scale) or scale <= eps:
        return 1.0
    return scale


def compute_relative_pose_metrics(
    pred_extrinsics: np.ndarray,
    gt_extrinsics: np.ndarray,
    *,
    threshold_deg: float = 5.0,
) -> dict[str, float]:
    pred = _as_homogeneous(pred_extrinsics)
    gt = _as_homogeneous(gt_extrinsics)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Prediction/GT frame count mismatch: {pred.shape[0]} vs {gt.shape[0]}")
    if pred.shape[0] < 2:
        return {
            "num_frames": float(pred.shape[0]),
            "num_pairs": 0.0,
            "translation_pair_count": 0.0,
            "rra_5": 0.0,
            "rta_5": 0.0,
            "rpe_r_mean_deg": 0.0,
            "rpe_r_median_deg": 0.0,
            "rpe_t_mean": 0.0,
            "rpe_t_median": 0.0,
            "rpe_t_scale_aligned_mean": 0.0,
            "rpe_t_scale_aligned_median": 0.0,
            "relative_translation_scale": 1.0,
        }

    pred_rel = _relative_transforms(_align_to_first_camera(pred))
    gt_rel = _relative_transforms(_align_to_first_camera(gt))
    rotation_errors = np.linalg.inv(gt_rel[:, :3, :3]) @ pred_rel[:, :3, :3]
    rpe_r = _rotation_angle_deg(rotation_errors)
    pred_t = pred_rel[:, :3, 3]
    gt_t = gt_rel[:, :3, 3]
    rpe_t_angle, translation_valid = _translation_angle_deg(pred_t, gt_t)
    rpe_t, rpe_t_valid = _normalized_translation_errors(pred_t, gt_t)
    scale = _optimal_translation_scale(pred_t, gt_t, rpe_t_valid)
    rpe_t_scaled, _ = _normalized_translation_errors(pred_t, gt_t, scale=scale)

    threshold_label = int(threshold_deg) if float(threshold_deg).is_integer() else float(threshold_deg)
    metrics = {
        "num_frames": float(pred.shape[0]),
        "num_pairs": float(pred_rel.shape[0]),
        "translation_pair_count": float(np.count_nonzero(translation_valid)),
        f"rra_{threshold_label}": _safe_accuracy_below(rpe_r, float(threshold_deg)),
        f"rta_{threshold_label}": _safe_accuracy_below(rpe_t_angle, float(threshold_deg)),
        "rpe_r_mean_deg": _safe_mean(rpe_r),
        "rpe_r_median_deg": _safe_median(rpe_r),
        "rpe_t_angle_mean_deg": _safe_mean(rpe_t_angle),
        "rpe_t_angle_median_deg": _safe_median(rpe_t_angle),
        "rpe_t_mean": _safe_mean(rpe_t),
        "rpe_t_median": _safe_median(rpe_t),
        "rpe_t_scale_aligned_mean": _safe_mean(rpe_t_scaled),
        "rpe_t_scale_aligned_median": _safe_median(rpe_t_scaled),
        "relative_translation_scale": float(scale),
    }
    if threshold_label != 5:
        metrics["rra_5"] = _safe_accuracy_below(rpe_r, 5.0)
        metrics["rta_5"] = _safe_accuracy_below(rpe_t_angle, 5.0)
    return metrics
