from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
DA3_ROOT = THIS_DIR.parent
PLAN4D_ROOT = DA3_ROOT.parent

LOGGER = logging.getLogger("analyze_da3_token_motion")
PATCH_SIZE = 14


def sanitize_import_paths() -> None:
    resolved_script_dir = THIS_DIR.resolve()
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != resolved_script_dir]
    preferred_paths = [str(DA3_ROOT), str(DA3_ROOT / "src"), str(PLAN4D_ROOT)]
    remaining_paths = [entry for entry in sys.path if entry not in preferred_paths]
    sys.path[:] = preferred_paths + remaining_paths


def points_to_token_indices(
    points_xy: torch.Tensor,
    *,
    patch_size: int,
    token_height: int,
    token_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if points_xy.numel() == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=points_xy.device)
        return empty, torch.empty((0,), dtype=torch.bool, device=points_xy.device)
    token_xy = torch.floor(points_xy.to(dtype=torch.float32) / float(patch_size)).to(dtype=torch.long)
    valid = (
        torch.isfinite(points_xy).all(dim=-1)
        & (token_xy[:, 0] >= 0)
        & (token_xy[:, 0] < int(token_width))
        & (token_xy[:, 1] >= 0)
        & (token_xy[:, 1] < int(token_height))
    )
    return token_xy, valid


def cosine_similarity_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=-1)
    right = F.normalize(right.float(), dim=-1)
    return (left * right).sum(dim=-1)


def compute_similarity_gap_from_scores(s_pos: torch.Tensor, s_neg_all: torch.Tensor) -> Dict[str, float]:
    if s_pos.numel() == 0:
        return {
            "num_pairs": 0,
            "mean_s_pos": float("nan"),
            "mean_s_neg": float("nan"),
            "mean_delta_s": float("nan"),
            "median_delta_s": float("nan"),
            "pos_gt_neg_rate": float("nan"),
        }
    if s_neg_all.ndim == 1:
        s_neg_all = s_neg_all[:, None]
    s_neg = s_neg_all.mean(dim=1)
    delta = s_pos - s_neg
    return {
        "num_pairs": int(s_pos.shape[0]),
        "mean_s_pos": float(s_pos.mean().item()),
        "mean_s_neg": float(s_neg.mean().item()),
        "mean_delta_s": float(delta.mean().item()),
        "median_delta_s": float(delta.median().item()),
        "pos_gt_neg_rate": float((s_pos[:, None] > s_neg_all).float().mean().item()),
    }


def compute_similarity_gap(
    query_tokens: torch.Tensor,
    positive_tokens: torch.Tensor,
    negative_tokens: torch.Tensor,
) -> Dict[str, float]:
    if query_tokens.numel() == 0:
        return compute_similarity_gap_from_scores(
            torch.empty((0,), device=query_tokens.device),
            torch.empty((0,), device=query_tokens.device),
        )
    s_pos = cosine_similarity_rows(query_tokens, positive_tokens)
    if negative_tokens.ndim == 3:
        query = F.normalize(query_tokens.float(), dim=-1)
        neg = F.normalize(negative_tokens.float(), dim=-1)
        s_neg_all = torch.einsum("nc,nkc->nk", query, neg)
    else:
        s_neg_all = cosine_similarity_rows(query_tokens, negative_tokens)
    return compute_similarity_gap_from_scores(s_pos, s_neg_all)


def compute_retrieval_metrics(
    query_tokens: torch.Tensor,
    target_grid: torch.Tensor,
    gt_token_xy: torch.Tensor,
    *,
    topk: Sequence[int] = (1, 3),
) -> Tuple[Dict[str, float], torch.Tensor]:
    if query_tokens.numel() == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=query_tokens.device)
        metrics = {"num_pairs": 0, "mean_token_error": float("nan"), "median_token_error": float("nan"), "top1_exact": float("nan")}
        for value in topk:
            metrics[f"pck@{int(value)}_token"] = float("nan")
        return metrics, empty

    token_height, token_width, channels = target_grid.shape
    target_flat = F.normalize(target_grid.reshape(token_height * token_width, channels).float(), dim=-1)
    query = F.normalize(query_tokens.float(), dim=-1)
    scores = query @ target_flat.T
    max_topk = max(max(int(value) for value in topk), 1)
    top_values, top_indices = torch.topk(scores, k=min(max_topk, target_flat.shape[0]), dim=1)
    del top_values
    pred_indices = top_indices[:, 0]
    pred_xy = torch.stack([pred_indices % token_width, pred_indices // token_width], dim=1).to(dtype=torch.long)
    gt_xy = gt_token_xy.to(device=query_tokens.device, dtype=torch.long)
    error = torch.linalg.norm((pred_xy - gt_xy).float(), dim=-1)
    exact = (pred_xy == gt_xy).all(dim=-1).float()
    metrics = {
        "num_pairs": int(query_tokens.shape[0]),
        "mean_token_error": float(error.mean().item()),
        "median_token_error": float(error.median().item()),
        "top1_exact": float(exact.mean().item()),
    }
    for value in topk:
        threshold = float(value)
        metrics[f"pck@{int(value)}_token"] = float((error <= threshold).float().mean().item())
    for value in topk:
        k = min(int(value), top_indices.shape[1])
        gt_flat = gt_xy[:, 1] * token_width + gt_xy[:, 0]
        metrics[f"top{int(value)}_contains_gt"] = float((top_indices[:, :k] == gt_flat[:, None]).any(dim=1).float().mean().item())
    return metrics, pred_xy.cpu()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left_valid = left[valid].astype(np.float64)
    right_valid = right[valid].astype(np.float64)
    left_std = float(left_valid.std())
    right_std = float(right_valid.std())
    if left_std <= 1e-12 or right_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def compute_pca_motion_correlations(
    token_delta: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    max_components: int = 3,
) -> Dict[str, Any]:
    token_delta = np.asarray(token_delta, dtype=np.float64)
    if token_delta.ndim != 2:
        raise ValueError(f"token_delta must be 2D [N, C], got {token_delta.shape}")
    finite = np.isfinite(token_delta).all(axis=1)
    for values in targets.values():
        finite &= np.isfinite(np.asarray(values)).reshape(-1)
    token_delta = token_delta[finite]
    filtered_targets = {key: np.asarray(value).reshape(-1)[finite].astype(np.float64) for key, value in targets.items()}
    if token_delta.shape[0] < 3:
        return {"num_points": int(token_delta.shape[0]), "explained_variance_ratio": [], "pearson": {}, "spearman": {}}

    centered = token_delta - token_delta.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    num_components = min(int(max_components), vt.shape[0])
    pcs = centered @ vt[:num_components].T
    variance = singular_values**2
    total_variance = float(variance.sum())
    explained = (variance[:num_components] / total_variance).tolist() if total_variance > 0 else [float("nan")] * num_components
    pearson: Dict[str, float] = {}
    spearman: Dict[str, float] = {}
    for component_idx in range(num_components):
        component = pcs[:, component_idx]
        component_rank = _rankdata(component)
        for target_name, target_values in filtered_targets.items():
            key = f"pc{component_idx + 1}:{target_name}"
            pearson[key] = _safe_corr(component, target_values)
            spearman[key] = _safe_corr(component_rank, _rankdata(target_values))
    return {
        "num_points": int(token_delta.shape[0]),
        "explained_variance_ratio": [float(value) for value in explained],
        "pearson": pearson,
        "spearman": spearman,
    }


def aggregate_metric_dicts(metric_dicts: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    grouped: Dict[str, list[float]] = defaultdict(list)
    for metrics in metric_dicts:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                grouped[key].append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items() if values}


def sample_negative_tokens(
    target_grid: torch.Tensor,
    gt_token_xy: torch.Tensor,
    *,
    negative_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    token_height, token_width, channels = target_grid.shape
    total_tokens = token_height * token_width
    gt_flat = gt_token_xy[:, 1].to(torch.long) * token_width + gt_token_xy[:, 0].to(torch.long)
    samples = torch.randint(0, total_tokens, (gt_flat.shape[0], int(negative_samples)), generator=generator, device=target_grid.device)
    if total_tokens > 1:
        collision = samples == gt_flat[:, None]
        samples = torch.where(collision, (samples + 1) % total_tokens, samples)
    flat_grid = target_grid.reshape(total_tokens, channels)
    return flat_grid[samples]


def compute_cross_frame_token_flow(
    frame_tokens: torch.Tensor,
    next_tokens: torch.Tensor,
) -> torch.Tensor:
    token_height, token_width, channels = frame_tokens.shape
    query = F.normalize(frame_tokens.reshape(token_height * token_width, channels).float(), dim=-1)
    target = F.normalize(next_tokens.reshape(token_height * token_width, channels).float(), dim=-1)
    indices = (query @ target.T).argmax(dim=1)
    pred_xy = torch.stack([indices % token_width, indices // token_width], dim=1).float()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(token_height, device=frame_tokens.device),
        torch.arange(token_width, device=frame_tokens.device),
        indexing="ij",
    )
    base_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()
    return (pred_xy - base_xy).reshape(token_height, token_width, 2).cpu()


def compute_token_flow_metrics(pred_disp: torch.Tensor, gt_disp: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    valid = valid_mask.bool()
    if not valid.any():
        return {"num_flow_tokens": 0, "token_flow_epe": float("nan"), "token_flow_dir_cos": float("nan"), "token_flow_pck@1": float("nan")}
    pred = pred_disp[valid].float()
    target = gt_disp[valid].float()
    error = torch.linalg.norm(pred - target, dim=-1)
    pred_norm = torch.linalg.norm(pred, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    dir_valid = (pred_norm > 1e-6) & (target_norm > 1e-6)
    if dir_valid.any():
        dir_cos = ((pred[dir_valid] * target[dir_valid]).sum(dim=-1) / (pred_norm[dir_valid] * target_norm[dir_valid]).clamp_min(1e-6)).mean()
        dir_cos_value = float(dir_cos.item())
    else:
        dir_cos_value = float("nan")
    return {
        "num_flow_tokens": int(valid.sum().item()),
        "token_flow_epe": float(error.mean().item()),
        "token_flow_dir_cos": dir_cos_value,
        "token_flow_pck@1": float((error <= 1.0).float().mean().item()),
    }


def build_gt_token_displacement(
    src_token_xy: torch.Tensor,
    dst_token_xy: torch.Tensor,
    *,
    token_height: int,
    token_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    disp_sum = torch.zeros(token_height, token_width, 2, dtype=torch.float32)
    counts = torch.zeros(token_height, token_width, dtype=torch.float32)
    disp = (dst_token_xy - src_token_xy).float()
    for idx in range(src_token_xy.shape[0]):
        x = int(src_token_xy[idx, 0].item())
        y = int(src_token_xy[idx, 1].item())
        if 0 <= x < token_width and 0 <= y < token_height:
            disp_sum[y, x] += disp[idx]
            counts[y, x] += 1.0
    valid = counts > 0
    averaged = torch.zeros_like(disp_sum)
    averaged[valid] = disp_sum[valid] / counts[valid].unsqueeze(-1)
    return averaged, valid


def _to_numpy_item(value: Any, batch_index: int = 0) -> Any:
    if isinstance(value, torch.Tensor):
        item = value[batch_index] if value.ndim > 0 else value
        return item.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return value[batch_index]
    return value


def format_sample_id(evaluated: int, batch: Mapping[str, Any]) -> str:
    scene_name = _batch_string(batch, "scene_name", "unknown")
    safe_scene_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in scene_name)
    return f"sample_{int(evaluated):04d}_{safe_scene_name}"


def _batch_scalar_int(batch: Mapping[str, Any], key: str, default: int = 0) -> int:
    if key not in batch:
        return default
    value = _to_numpy_item(batch[key], 0)
    return int(np.asarray(value).reshape(-1)[0])


def _batch_string(batch: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in batch:
        return default
    value = batch[key]
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def load_point_odyssey_track_pairs(
    dataset: Any,
    batch: Mapping[str, Any],
    *,
    max_pairs_per_frame: int,
    seed: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    from training.datasets.point_odyssey_scene_flow import (
        _apply_2d_resize_crop,
        _find_rgb_path,
        build_track_mask,
        compute_resize_params,
    )
    import cv2

    if not hasattr(dataset, "samples"):
        return {}
    scene_name = _batch_string(batch, "scene_name")
    start_idx = _batch_scalar_int(batch, "start_idx")
    end_idx = _batch_scalar_int(batch, "end_idx")
    sample = None
    for candidate in getattr(dataset, "samples"):
        if candidate.scene_name == scene_name and int(candidate.start_idx) == start_idx and int(candidate.end_idx) == end_idx:
            sample = candidate
            break
    if sample is None:
        return {}

    with np.load(sample.scene_dir / "anno.npz") as anno:
        trajs_2d = anno["trajs_2d"][sample.start_idx : sample.end_idx].astype(np.float32)
        valids = anno["valids"][sample.start_idx : sample.end_idx].astype(np.float32)
        visibs = anno["visibs"][sample.start_idx : sample.end_idx].astype(np.float32) if "visibs" in anno else None

    first_rgb = cv2.imread(str(_find_rgb_path(sample.scene_dir, sample.start_idx)), cv2.IMREAD_COLOR)
    if first_rgb is None:
        return {}
    src_height, src_width = first_rgb.shape[:2]
    resize_height, resize_width, crop_top, crop_left, output_height, output_width = compute_resize_params(
        src_height,
        src_width,
        int(getattr(dataset, "target_height")),
        int(getattr(dataset, "target_width")),
        process_res=getattr(dataset, "process_res", None),
        process_res_method=getattr(dataset, "process_res_method", None),
    )
    trajs_2d = _apply_2d_resize_crop(trajs_2d, resize_height, resize_width, crop_top, crop_left, src_height, src_width)
    track_mask = build_track_mask(valids, visibs, trajs_2d, output_height, output_width)
    rng = np.random.default_rng(seed)
    pairs: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for frame_idx in range(max(0, min(trajs_2d.shape[0] - 1, end_idx - start_idx - 1))):
        valid_indices = np.flatnonzero(track_mask[frame_idx].astype(bool))
        if valid_indices.size == 0:
            continue
        if max_pairs_per_frame > 0 and valid_indices.size > max_pairs_per_frame:
            valid_indices = rng.choice(valid_indices, size=max_pairs_per_frame, replace=False)
        pairs[frame_idx] = (trajs_2d[frame_idx, valid_indices], trajs_2d[frame_idx + 1, valid_indices])
    return pairs


def collect_pair_tokens(
    tokens_tchw: torch.Tensor,
    pairs_by_frame: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    *,
    patch_size: int,
    negative_samples: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    sequence_length, token_height, token_width, channels = tokens_tchw.shape
    del channels
    s_pos_chunks: list[torch.Tensor] = []
    s_neg_chunks: list[torch.Tensor] = []
    similarity_metrics: list[Dict[str, float]] = []
    retrieval_metrics: list[Dict[str, float]] = []
    flow_metrics: list[Dict[str, float]] = []
    generator = torch.Generator(device=tokens_tchw.device).manual_seed(seed)
    for frame_idx, (src_xy_np, dst_xy_np) in pairs_by_frame.items():
        if frame_idx < 0 or frame_idx + 1 >= sequence_length:
            continue
        src_xy = torch.from_numpy(src_xy_np).to(device=tokens_tchw.device, dtype=torch.float32)
        dst_xy = torch.from_numpy(dst_xy_np).to(device=tokens_tchw.device, dtype=torch.float32)
        src_token_xy, src_valid = points_to_token_indices(src_xy, patch_size=patch_size, token_height=token_height, token_width=token_width)
        dst_token_xy, dst_valid = points_to_token_indices(dst_xy, patch_size=patch_size, token_height=token_height, token_width=token_width)
        valid = src_valid & dst_valid
        if not valid.any():
            continue
        src_token_xy = src_token_xy[valid]
        dst_token_xy = dst_token_xy[valid]
        frame_tokens = tokens_tchw[frame_idx]
        next_tokens = tokens_tchw[frame_idx + 1]
        query = frame_tokens[src_token_xy[:, 1], src_token_xy[:, 0]]
        positive = next_tokens[dst_token_xy[:, 1], dst_token_xy[:, 0]]
        negative = sample_negative_tokens(next_tokens, dst_token_xy, negative_samples=negative_samples, generator=generator)
        s_pos = cosine_similarity_rows(query, positive)
        s_neg_all = torch.einsum(
            "nc,nkc->nk",
            F.normalize(query.float(), dim=-1),
            F.normalize(negative.float(), dim=-1),
        )
        s_neg = s_neg_all.mean(dim=1)
        s_pos_chunks.append(s_pos.detach().cpu())
        s_neg_chunks.append(s_neg.detach().cpu())
        similarity_metrics.append(compute_similarity_gap_from_scores(s_pos, s_neg_all))
        retrieval, _ = compute_retrieval_metrics(query, next_tokens, dst_token_xy, topk=(1, 3))
        retrieval_metrics.append(retrieval)
        pred_disp = compute_cross_frame_token_flow(frame_tokens, next_tokens)
        gt_disp, gt_valid = build_gt_token_displacement(src_token_xy.cpu(), dst_token_xy.cpu(), token_height=token_height, token_width=token_width)
        flow_metrics.append(compute_token_flow_metrics(pred_disp, gt_disp, gt_valid))
        del query, positive, negative, s_pos, s_neg_all, s_neg
    if not s_pos_chunks:
        empty = torch.empty((0,), dtype=torch.float32)
        return empty, empty, empty, {
            "similarity": {},
            "retrieval": {},
            "token_flow": {},
        }
    return torch.cat(s_pos_chunks, dim=0), torch.cat(s_neg_chunks, dim=0), torch.empty((0,), dtype=torch.float32), {
        "similarity": aggregate_metric_dicts(similarity_metrics),
        "retrieval": aggregate_metric_dicts(retrieval_metrics),
        "token_flow": aggregate_metric_dicts(flow_metrics),
    }


def average_scene_flow_to_tokens(scene_flow_thwc: torch.Tensor, deform_mask_thw: torch.Tensor, token_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    token_height, token_width = token_size
    flow = scene_flow_thwc.permute(0, 3, 1, 2).float()
    mask = deform_mask_thw[:, None].float()
    flow = F.interpolate(flow * mask, size=(token_height, token_width), mode="area")
    counts = F.interpolate(mask, size=(token_height, token_width), mode="area").clamp_min(1e-6)
    averaged = (flow / counts).permute(0, 2, 3, 1)
    valid = counts[:, 0] > 1e-4
    return averaged.cpu(), valid.cpu()


def build_pca_payload(tokens_tchw: torch.Tensor, scene_flow_thwc: torch.Tensor, deform_mask_thw: torch.Tensor, *, max_points: int, seed: int) -> Dict[str, Any]:
    if tokens_tchw.shape[0] < 2:
        return {"num_points": 0, "explained_variance_ratio": [], "pearson": {}, "spearman": {}}
    token_height, token_width = int(tokens_tchw.shape[1]), int(tokens_tchw.shape[2])
    flow_tokens, valid_tokens = average_scene_flow_to_tokens(scene_flow_thwc, deform_mask_thw, (token_height, token_width))
    delta = (tokens_tchw[1:] - tokens_tchw[:-1]).detach().cpu()
    valid = valid_tokens[:-1]
    flat_delta = delta[valid].float().numpy()
    flat_flow = flow_tokens[:-1][valid].float().numpy()
    if flat_delta.shape[0] > max_points > 0:
        rng = np.random.default_rng(seed)
        indices = rng.choice(flat_delta.shape[0], size=max_points, replace=False)
        flat_delta = flat_delta[indices]
        flat_flow = flat_flow[indices]
    targets = {
        "flow_x": flat_flow[:, 0] if flat_flow.size else np.empty((0,), dtype=np.float32),
        "flow_y": flat_flow[:, 1] if flat_flow.size else np.empty((0,), dtype=np.float32),
        "flow_z": flat_flow[:, 2] if flat_flow.size else np.empty((0,), dtype=np.float32),
        "flow_mag": np.linalg.norm(flat_flow, axis=1) if flat_flow.size else np.empty((0,), dtype=np.float32),
    }
    return compute_pca_motion_correlations(flat_delta, targets, max_components=3)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_plot_raw_data(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(path, **payload)


def _finite_stat(summary: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    if not summary:
        return None
    value = summary.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def save_similarity_histogram(
    path: Path,
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    title: str,
    summary: Optional[Mapping[str, Any]] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    _ = title
    s_pos = np.asarray(s_pos, dtype=np.float32).reshape(-1)
    s_neg = np.asarray(s_neg, dtype=np.float32).reshape(-1)
    s_pos = s_pos[np.isfinite(s_pos)]
    s_neg = s_neg[np.isfinite(s_neg)]
    if s_pos.size == 0 or s_neg.size == 0:
        return

    combined = np.concatenate([s_pos, s_neg], axis=0)
    bins = np.histogram_bin_edges(combined, bins=40)
    gt_color = "#004488"
    neg_color = "#DDAA33"
    fill_alpha = 0.28

    fig, axis = plt.subplots(figsize=(6.8, 4.2), dpi=180)
    axis.hist(
        s_pos,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=gt_color,
        alpha=fill_alpha,
    )
    axis.hist(
        s_neg,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=neg_color,
        alpha=fill_alpha,
    )
    axis.hist(s_pos, bins=bins, density=True, histtype="step", color=gt_color, linewidth=2.0)
    axis.hist(s_neg, bins=bins, density=True, histtype="step", color=neg_color, linewidth=2.0)

    gt_mean = _finite_stat(summary, "mean_s_pos")
    neg_mean = _finite_stat(summary, "mean_s_neg")
    gt_median = float(np.median(s_pos))
    neg_median = float(np.median(s_neg))
    if gt_mean is not None:
        axis.axvline(gt_mean, color=gt_color, linestyle="--", linewidth=1.7, alpha=0.95)
    if neg_mean is not None:
        axis.axvline(neg_mean, color=neg_color, linestyle="--", linewidth=1.7, alpha=0.95)
    axis.axvline(gt_median, color=gt_color, linestyle="-", linewidth=1.9, alpha=0.95)
    axis.axvline(neg_median, color=neg_color, linestyle="-", linewidth=1.9, alpha=0.95)

    value_span = float(combined.max() - combined.min())
    value_margin = max(0.01, value_span * 0.03)
    axis.set_xlim(float(combined.min() - value_margin), float(combined.max() + value_margin))
    axis.set_xlabel("Cosine similarity")
    axis.set_ylabel("Density")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=gt_color, edgecolor=gt_color, alpha=fill_alpha, label="GT correspondence"),
            Patch(facecolor=neg_color, edgecolor=neg_color, alpha=fill_alpha, label="Random negative"),
            Line2D([0], [0], color="#555555", linewidth=1.8, linestyle="-", label="Median"),
            Line2D([0], [0], color="#555555", linewidth=1.6, linestyle="--", label="Mean"),
        ],
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_pca_bar(
    path: Path,
    pca_result: Mapping[str, Any],
    title: str,
    summary: Optional[Mapping[str, Any]] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    explained = np.asarray(pca_result.get("explained_variance_ratio", []), dtype=np.float32)
    if explained.size == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = title
    explained = explained[np.isfinite(explained)]
    if explained.size == 0:
        return

    component_labels = [f"PC{index}" for index in range(1, explained.size + 1)]
    explained_pct = explained * 100.0
    cumulative_pct = np.cumsum(explained_pct)
    x = np.arange(explained.size, dtype=np.float32)
    bar_color = "#004488"
    line_color = "#DDAA33"

    fig, axis = plt.subplots(figsize=(5.8, 3.6), dpi=180)
    bars = axis.bar(
        x,
        explained_pct,
        width=0.62,
        color=bar_color,
        edgecolor=bar_color,
        alpha=0.88,
    )
    axis.plot(
        x,
        cumulative_pct,
        color=line_color,
        linewidth=2.0,
        marker="o",
        markersize=6,
    )

    for bar, value in zip(bars, explained_pct.tolist()):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.7,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=bar_color,
        )

    ymax = max(float(cumulative_pct.max()), float(explained_pct.max()))
    axis.set_ylim(0.0, ymax * 1.22)
    axis.set_xticks(x, component_labels)
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Explained variance (%)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Patch(facecolor=bar_color, edgecolor=bar_color, alpha=0.88, label="Per-component variance"),
            Line2D([0], [0], color=line_color, linewidth=2.0, marker="o", markersize=6, label="Cumulative variance"),
        ],
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether frozen DA3 encoder tokens encode motion correspondence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="../configs/scene_flow.yaml")
    parser.add_argument("--output-dir", type=str, default="training/outputs/da3_token_motion_analysis")
    parser.add_argument("--dataset-name", type=str, default="point_odyssey")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--sample-tracks", type=int, default=1024)
    parser.add_argument("--negative-samples", type=int, default=4)
    parser.add_argument("--pca-max-points", type=int, default=20000)
    parser.add_argument("--ref-view-strategy", type=str, default="first")
    parser.add_argument("--layer-mode", choices=("concat",), default="concat")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-flow-head-checkpoint", type=str, default=None)
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def _select_loader(loaders: Mapping[str, Any], dataset_name: Optional[str]) -> Any:
    if not loaders:
        raise RuntimeError("No validation dataloaders were built")
    if dataset_name and dataset_name in loaders:
        return loaders[dataset_name]
    if dataset_name:
        matches = [key for key in loaders if key.split(":")[0] == dataset_name]
        if len(matches) == 1:
            return loaders[matches[0]]
        if len(matches) > 1:
            raise ValueError(f"Multiple loaders match {dataset_name}: {matches}")
    return next(iter(loaders.values()))


def _prepare_config(args: argparse.Namespace) -> Dict[str, Any]:
    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (DA3_ROOT / config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = copy.deepcopy(config)
    data_cfg = config.setdefault("data", {})
    data_cfg["batch_size"] = 1
    data_cfg["num_workers"] = int(args.num_workers)
    data_cfg["flow_target_space"] = "metric"
    if args.split:
        data_cfg["val_split"] = args.split
        if "datasets" in data_cfg:
            for spec in data_cfg["datasets"]:
                if spec.get("name") == args.dataset_name:
                    spec["val_splits"] = [args.split]
    if not Path(config["model"]["weights_dir"]).is_absolute():
        config["model"]["weights_dir"] = str((DA3_ROOT / config["model"]["weights_dir"]).resolve())
    if "data_root" in data_cfg and data_cfg["data_root"] and not Path(data_cfg["data_root"]).is_absolute():
        data_cfg["data_root"] = str((DA3_ROOT / data_cfg["data_root"]).resolve())
    for spec in data_cfg.get("datasets", []):
        if "data_root" in spec and spec["data_root"] and not Path(spec["data_root"]).is_absolute():
            spec["data_root"] = str((DA3_ROOT / spec["data_root"]).resolve())
    return config


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sanitize_import_paths()

    from plan4d.da3_interface import DA3StreamingInterface
    from training.datasets.multi_scene_flow import build_scene_flow_val_loaders
    from training.utils import get_scene_flow_head, load_da3_with_scene_flow_head
    from training.eval_scene_flow_head import load_scene_flow_head_checkpoint
    from training.losses.scene_flow_loss import SceneFlowLoss

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (DA3_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = output_dir / "metrics_per_sample.jsonl"
    with open(per_sample_path, "w", encoding="utf-8"):
        pass

    config = _prepare_config(args)
    dataloader = _select_loader(build_scene_flow_val_loaders(config["data"], num_workers_override=args.num_workers), args.dataset_name)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    da3_model, load_info = load_da3_with_scene_flow_head(
        repo_root=DA3_ROOT,
        weights_dir=config["model"]["weights_dir"],
        scene_flow_preset=config["model"].get("scene_flow_preset", "da3-giant-sceneflow"),
        checkpoint_name=config["model"].get("checkpoint_name", "model.safetensors"),
        config_name=config["model"].get("config_name", "config.json"),
        device=device,
    )
    LOGGER.info("Loaded DA3: %s", load_info)
    if args.scene_flow_head_checkpoint:
        load_scene_flow_head_checkpoint(get_scene_flow_head(da3_model), args.scene_flow_head_checkpoint)
        LOGGER.info("Loaded optional scene-flow head checkpoint: %s", args.scene_flow_head_checkpoint)
    da3_interface = DA3StreamingInterface(da3_model=da3_model)
    da3_interface.eval()
    criterion = SceneFlowLoss(loss_type="l1")

    sample_metrics: list[Dict[str, Any]] = []
    all_gap_metrics: list[Dict[str, float]] = []
    all_retrieval_metrics: list[Dict[str, float]] = []
    all_token_flow_metrics: list[Dict[str, float]] = []
    all_pca_results: list[Dict[str, Any]] = []
    evaluated = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_samples is not None and evaluated >= args.max_samples:
            break
        images = batch["rgbs"].permute(0, 1, 4, 2, 3).to(device, non_blocking=True)
        target_flow = batch["scene_flow"].to(device, non_blocking=True)
        deform_mask = batch["deform_mask"].to(device, non_blocking=True)
        sequence_length = int(_to_numpy_item(batch.get("sequence_length", torch.tensor([images.shape[1]])), 0).reshape(-1)[0])
        height, width = images.shape[-2:]
        with torch.no_grad():
            features = da3_interface.extract_multilayer_features(images, normalize=True, ref_view_strategy=args.ref_view_strategy)
            tokens = features[0, :, :sequence_length].permute(1, 2, 3, 0).contiguous().float()
            tokens = F.normalize(tokens, dim=-1)
            if device.type == "cuda":
                tokens = tokens.to(dtype=torch.float16)
        if not args.scene_flow_head_checkpoint:
            del features
        pairs_by_frame = load_point_odyssey_track_pairs(
            dataloader.dataset,
            batch,
            max_pairs_per_frame=int(args.sample_tracks),
            seed=int(args.seed) + batch_idx,
        )
        plot_s_pos, plot_s_neg, _, pair_extra = collect_pair_tokens(
            tokens,
            pairs_by_frame,
            patch_size=PATCH_SIZE,
            negative_samples=int(args.negative_samples),
            seed=int(args.seed) + batch_idx,
        )
        gap_metrics = pair_extra.get("similarity", {})
        pca_result = build_pca_payload(
            tokens,
            target_flow[0, :sequence_length].detach().cpu(),
            deform_mask[0, :sequence_length].detach().cpu(),
            max_points=int(args.pca_max_points),
            seed=int(args.seed) + batch_idx,
        )
        head_metrics: Dict[str, float] = {}
        if args.scene_flow_head_checkpoint:
            with torch.no_grad():
                head_dtype = next(get_scene_flow_head(da3_interface.da3_model).parameters()).dtype
                pred_flow = da3_interface.decode_features_to_scene_flow(
                    features=features.to(dtype=head_dtype),
                    image_size=(height, width),
                    extrinsics=batch["extrinsics"].to(device, non_blocking=True),
                    use_gradient=False,
                    chunk_size=int(config["model"].get("decoder_chunk_size", config["data"].get("chunk_size", 49))),
                    output_basis="frame0",
                )
                loss_dict = criterion(pred_flow, target_flow.permute(0, 1, 4, 2, 3), deform_mask)
                head_metrics = {f"head_{key}": float(value.detach().float().mean().item()) for key, value in loss_dict.items() if torch.is_tensor(value)}

        sample_id = format_sample_id(evaluated, batch)
        record = {
            "sample_id": sample_id,
            "scene_name": _batch_string(batch, "scene_name", "unknown"),
            "clip_uid": _batch_string(batch, "clip_uid", ""),
            "sequence_length": sequence_length,
            "num_pair_frames": len(pairs_by_frame),
            "similarity": gap_metrics,
            "retrieval": pair_extra.get("retrieval", {}),
            "token_flow": pair_extra.get("token_flow", {}),
            "pca": pca_result,
            "head_control": head_metrics,
        }
        append_jsonl(per_sample_path, record)
        sample_metrics.append(record)
        all_gap_metrics.append(gap_metrics)
        if pair_extra.get("retrieval"):
            all_retrieval_metrics.append(pair_extra["retrieval"])
        if pair_extra.get("token_flow"):
            all_token_flow_metrics.append(pair_extra["token_flow"])
        all_pca_results.append(pca_result)

        if args.save_plots:
            if plot_s_pos.numel() > 0:
                s_pos = plot_s_pos.numpy()
                s_neg = plot_s_neg.numpy()
                save_similarity_histogram(
                    output_dir / "plots" / f"{sample_id}_similarity.png",
                    s_pos,
                    s_neg,
                    sample_id,
                    summary=gap_metrics,
                )
                save_plot_raw_data(
                    output_dir / "plots" / f"{sample_id}_similarity_data.npz",
                    s_pos=s_pos,
                    s_neg=s_neg,
                    delta_s=s_pos - s_neg,
                    title=np.array(sample_id),
                )
            save_pca_bar(
                output_dir / "plots" / f"{sample_id}_pca_variance.png",
                pca_result,
                sample_id,
                summary=pca_result,
            )
            save_plot_raw_data(
                output_dir / "plots" / f"{sample_id}_pca_variance_data.npz",
                explained_variance_ratio=np.asarray(pca_result.get("explained_variance_ratio", []), dtype=np.float32),
                pc_index=np.arange(1, len(pca_result.get("explained_variance_ratio", [])) + 1, dtype=np.int32),
                title=np.array(sample_id),
            )
        evaluated += 1
        del images, target_flow, deform_mask, tokens, plot_s_pos, plot_s_neg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pca_explained = [result.get("explained_variance_ratio", []) for result in all_pca_results if result.get("explained_variance_ratio")]
    summary = {
        "num_samples": evaluated,
        "config": {
            "split": args.split,
            "dataset_name": args.dataset_name,
            "ref_view_strategy": args.ref_view_strategy,
            "sample_tracks": args.sample_tracks,
            "negative_samples": args.negative_samples,
            "flow_target_space": "metric",
        },
        "similarity": aggregate_metric_dicts(all_gap_metrics),
        "retrieval": aggregate_metric_dicts(all_retrieval_metrics),
        "token_flow": aggregate_metric_dicts(all_token_flow_metrics),
        "pca_mean_explained_variance_ratio": np.asarray(pca_explained, dtype=np.float32).mean(axis=0).tolist() if pca_explained else [],
    }
    save_json(output_dir / "metrics_summary.json", summary)
    report = [
        "# DA3 Token Motion Analysis",
        "",
        f"- Samples: {evaluated}",
        f"- Similarity mean Δs: {summary['similarity'].get('mean_delta_s', float('nan')):.6f}",
        f"- Retrieval PCK@1 token: {summary['retrieval'].get('pck@1_token', float('nan')):.6f}",
        f"- Retrieval PCK@3 token: {summary['retrieval'].get('pck@3_token', float('nan')):.6f}",
        f"- Token-flow EPE: {summary['token_flow'].get('token_flow_epe', float('nan')):.6f}",
        "",
        "Core metrics are computed directly on frozen DA3 encoder tokens and do not require the scene-flow head.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    LOGGER.info("Wrote analysis outputs to %s", output_dir)


if __name__ == "__main__":
    main()
