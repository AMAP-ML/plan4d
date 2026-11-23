from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.da3_stitched import canonicalize_da3_geometry_outputs


def _resample_btchw(
    tensor: torch.Tensor | None,
    target_frames: int,
    mode: str = "trilinear",
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[1] == target_frames:
        return tensor

    tensor_bcthw = tensor.permute(0, 2, 1, 3, 4)
    if mode == "nearest":
        resized = F.interpolate(
            tensor_bcthw,
            size=(target_frames, tensor.shape[-2], tensor.shape[-1]),
            mode="nearest",
        )
    else:
        resized = F.interpolate(
            tensor_bcthw,
            size=(target_frames, tensor.shape[-2], tensor.shape[-1]),
            mode=mode,
            align_corners=False,
        )
    return resized.permute(0, 2, 1, 3, 4)


def _resize_spatial_btchw(
    tensor: torch.Tensor | None,
    target_hw: tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.shape[-2:] == target_hw:
        return tensor

    flat = tensor.reshape(-1, tensor.shape[2], tensor.shape[3], tensor.shape[4])
    kwargs = {"mode": mode, "size": target_hw}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(flat, **kwargs)
    return resized.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], target_hw[0], target_hw[1])


def _match_time_dimension(
    tensor: torch.Tensor | None,
    target_frames: int,
) -> torch.Tensor | None:
    if tensor is None or tensor.dim() < 2 or tensor.shape[1] == target_frames:
        return tensor
    timeline = torch.linspace(
        0,
        tensor.shape[1] - 1,
        target_frames,
        device=tensor.device,
    )
    timeline = timeline.round().long().clamp(0, tensor.shape[1] - 1)
    return tensor.index_select(1, timeline)


def _ensure_btchw(
    tensor: torch.Tensor | None,
    *,
    channels: int | None = None,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(2)
    elif tensor.dim() == 5 and channels is not None and tensor.shape[-1] == channels:
        tensor = tensor.permute(0, 1, 4, 2, 3)
    if tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(tensor.shape)}")
    if channels is not None and tensor.shape[2] != channels:
        raise ValueError(
            f"Expected tensor with channel dim {channels}, got {tuple(tensor.shape)}"
        )
    return tensor


def prepare_gt_geometry_supervision_inputs(
    *,
    geometry_outputs: dict[str, Any],
    depth_gt: torch.Tensor,
    ray_gt: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    scene_flow_gt: torch.Tensor | None = None,
    deform_mask: torch.Tensor | None = None,
    geo_supervise_generated_only: bool = False,
    cond_video_frames: int = 0,
    gt_points_3d: torch.Tensor | None = None,
    gt_points_2d: torch.Tensor | None = None,
    points_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    pred = canonicalize_da3_geometry_outputs(geometry_outputs)

    pred_depth = pred["depth"]
    pred_ray = pred["ray"]
    depth_conf = pred.get("depth_conf")
    ray_conf = pred.get("ray_conf")
    pred_scene_flow = _ensure_btchw(geometry_outputs.get("scene_flow"), channels=3)
    scene_flow_gt = _ensure_btchw(scene_flow_gt, channels=3)
    deform_mask = _ensure_btchw(deform_mask, channels=1)

    target_t = depth_gt.shape[1]
    pred_depth = _resample_btchw(pred_depth, target_t, mode="trilinear")
    pred_ray = _resample_btchw(pred_ray, target_t, mode="trilinear")
    depth_conf = _resample_btchw(depth_conf, target_t, mode="trilinear")
    ray_conf = _resample_btchw(ray_conf, target_t, mode="trilinear")
    pred_scene_flow = _resample_btchw(pred_scene_flow, target_t, mode="trilinear")
    scene_flow_gt = _resample_btchw(scene_flow_gt, target_t, mode="trilinear")
    deform_mask = _resample_btchw(deform_mask, target_t, mode="nearest")
    valid_mask = _resample_btchw(valid_mask, target_t, mode="nearest")
    ray_gt = _resample_btchw(ray_gt, target_t, mode="trilinear")

    target_hw = depth_gt.shape[-2:]
    pred_depth = _resize_spatial_btchw(pred_depth, target_hw, mode="bilinear")
    pred_ray = _resize_spatial_btchw(pred_ray, target_hw, mode="bilinear")
    depth_conf = _resize_spatial_btchw(depth_conf, target_hw, mode="bilinear")
    ray_conf = _resize_spatial_btchw(ray_conf, target_hw, mode="bilinear")
    pred_scene_flow = _resize_spatial_btchw(pred_scene_flow, target_hw, mode="bilinear")
    scene_flow_gt = _resize_spatial_btchw(scene_flow_gt, target_hw, mode="bilinear")
    deform_mask = _resize_spatial_btchw(deform_mask, target_hw, mode="nearest")
    valid_mask = _resize_spatial_btchw(valid_mask, target_hw, mode="nearest")

    intrinsics = _match_time_dimension(intrinsics, target_t)
    extrinsics = _match_time_dimension(extrinsics, target_t)

    if geo_supervise_generated_only and cond_video_frames > 0:
        valid_mask = valid_mask.clone()
        valid_mask[:, : min(cond_video_frames, valid_mask.shape[1])] = 0
        if deform_mask is not None:
            deform_mask = deform_mask.clone()
            deform_mask[:, : min(cond_video_frames, deform_mask.shape[1])] = 0

    result = {
        "pred_depth": pred_depth,
        "target_depth": depth_gt,
        "pred_ray": pred_ray,
        "target_ray": ray_gt,
        "depth_conf": depth_conf,
        "ray_conf": ray_conf,
        "valid_mask": valid_mask,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "gt_points_3d": gt_points_3d,
        "gt_points_2d": gt_points_2d,
        "points_valid_mask": points_valid_mask,
        "pred_camera_intrinsics": geometry_outputs.get("camera_intrinsics", geometry_outputs.get("intrinsics")),
        "pred_camera_extrinsics": geometry_outputs.get("camera_extrinsics", geometry_outputs.get("extrinsics")),
        "pred_camera_tokens": geometry_outputs.get("camera_tokens"),
        "pred_camera_image_hw": geometry_outputs.get("camera_image_hw"),
    }
    if pred_scene_flow is not None and scene_flow_gt is not None and deform_mask is not None:
        result["pred_scene_flow"] = pred_scene_flow
        result["target_scene_flow"] = scene_flow_gt
        result["deform_mask"] = deform_mask
    return result


def compute_gt_geometry_loss_from_outputs(
    *,
    geometry_outputs: dict[str, Any],
    geometry_loss_fn,
    depth_gt: torch.Tensor,
    ray_gt: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    valid_mask: torch.Tensor,
    scene_flow_gt: torch.Tensor | None = None,
    deform_mask: torch.Tensor | None = None,
    geo_supervise_generated_only: bool = False,
    cond_video_frames: int = 0,
    gt_points_3d: torch.Tensor | None = None,
    gt_points_2d: torch.Tensor | None = None,
    points_valid_mask: torch.Tensor | None = None,
):
    loss_inputs = prepare_gt_geometry_supervision_inputs(
        geometry_outputs=geometry_outputs,
        depth_gt=depth_gt,
        ray_gt=ray_gt,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=valid_mask,
        scene_flow_gt=scene_flow_gt,
        deform_mask=deform_mask,
        geo_supervise_generated_only=geo_supervise_generated_only,
        cond_video_frames=cond_video_frames,
        gt_points_3d=gt_points_3d,
        gt_points_2d=gt_points_2d,
        points_valid_mask=points_valid_mask,
    )
    if loss_inputs["depth_conf"] is None:
        loss_inputs["depth_conf"] = torch.ones_like(loss_inputs["pred_depth"])
    if loss_inputs["ray_conf"] is None:
        loss_inputs["ray_conf"] = torch.ones_like(loss_inputs["pred_depth"])
    return geometry_loss_fn(**loss_inputs)
