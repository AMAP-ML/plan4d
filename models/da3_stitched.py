from __future__ import annotations

import copy
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT
DA3_SRC_ROOT = PLAN4D_ROOT / "Depth-Anything-3" / "src"

for import_path in (PLAN4D_ROOT, DA3_SRC_ROOT):
    if import_path.exists() and str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from depth_anything_3.model.dinov2.vision_transformer import (  # noqa: E402
    reorder_by_reference,
    restore_original_order,
    select_reference_view,
)
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION  # noqa: E402


def _load_plan4d_training_losses():
    losses_path = PLAN4D_ROOT / "plan4d" / "geometry_losses.py"
    spec = importlib.util.spec_from_file_location("_plan4d_plan4d_training_losses", losses_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plan4d training losses from {losses_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_plan4d_training_losses = _load_plan4d_training_losses()
DecodedCameraLoss = _plan4d_training_losses.DecodedCameraLoss
build_temporal_gradient_matching_loss = _plan4d_training_losses.build_temporal_gradient_matching_loss


DA3_STITCH_CANDIDATE_LAYER_NAMES = tuple(f"enc_blocks_{idx}" for idx in range(1, 13))
DA3_DISTILL_FEATURE_LAYER = "enc_blocks_1"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
DA3_PATCH_SIZE = 14


def is_da3_model(model: nn.Module) -> bool:
    candidate = get_da3_anyview_model(model)
    return hasattr(candidate, "backbone") and hasattr(candidate, "head")


def get_da3_anyview_model(model: nn.Module) -> nn.Module:
    return model.da3 if hasattr(model, "da3") else model


def compute_da3_token_grid_size(
    feedforward_resolution: int,
    patch_size: int = DA3_PATCH_SIZE,
) -> int:
    return feedforward_resolution // patch_size


def compute_da3_stitching_input_spatial_size(
    *,
    feedforward_resolution: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
    patch_size: int = DA3_PATCH_SIZE,
) -> int:
    height, _ = compute_da3_stitching_input_spatial_shape(
        image_hw=(feedforward_resolution, feedforward_resolution),
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        dilation=(dilation, dilation),
        patch_size=patch_size,
    )
    return height


def compute_da3_stitching_input_spatial_shape(
    *,
    image_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
    patch_size: int = DA3_PATCH_SIZE,
) -> tuple[int, int]:
    height, width = image_hw
    target_grid_h = compute_da3_token_grid_size(
        feedforward_resolution=height,
        patch_size=patch_size,
    )
    target_grid_w = compute_da3_token_grid_size(
        feedforward_resolution=width,
        patch_size=patch_size,
    )
    input_h = target_grid_h * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1)
    input_w = target_grid_w * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1)
    return input_h, input_w


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = prediction - target
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1]).clamp(max=100)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :]).clamp(max=100)
    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = prediction.shape[0] * prediction.shape[1] * prediction.shape[2]
    return torch.sum(image_loss) / divisor


def _gradient_loss_multi_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: int = 4,
) -> torch.Tensor:
    total = 0.0
    for scale in range(scales):
        step = 2**scale
        total = total + _gradient_loss(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
        )
    return total / scales


def normalize_images_imagenet(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    elif images.min() < 0:
        images = (images + 1.0) * 0.5
    elif images.max() > 1.0:
        images = images / 255.0

    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype).view(1, 1, 3, 1, 1)
    return (images - mean) / std


def _prepare_da3_images(context_image: torch.Tensor) -> torch.Tensor:
    if context_image.dim() != 5:
        raise ValueError(f"Expected [B, C, T, H, W] images, got {tuple(context_image.shape)}")
    return rearrange(context_image, "b c t h w -> b t c h w")


def canonicalize_da3_geometry_outputs(outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    depth = outputs["depth"]
    depth_conf = outputs["depth_conf"]
    ray = outputs["ray"]
    ray_conf = outputs["ray_conf"]
    sky = outputs.get("sky")

    if depth.dim() == 4:
        depth = depth.unsqueeze(2)
    elif depth.dim() == 5 and depth.shape[-1] == 1:
        depth = depth.permute(0, 1, 4, 2, 3)

    if depth_conf.dim() == 4:
        depth_conf = depth_conf.unsqueeze(2)
    elif depth_conf.dim() == 5 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf.permute(0, 1, 4, 2, 3)

    if ray.dim() == 5 and ray.shape[-1] == 6:
        ray = ray.permute(0, 1, 4, 2, 3)
    elif ray.dim() != 5 or ray.shape[2] != 6:
        raise ValueError(f"Unsupported DA3 ray shape: {tuple(ray.shape)}")

    if ray_conf.dim() == 4:
        ray_conf = ray_conf.unsqueeze(2)
    elif ray_conf.dim() == 5 and ray_conf.shape[-1] == 1:
        ray_conf = ray_conf.permute(0, 1, 4, 2, 3)

    canonicalized = {
        "depth": depth,
        "depth_conf": depth_conf,
        "ray": ray,
        "ray_conf": ray_conf,
    }
    if sky is not None:
        if sky.dim() == 5 and sky.shape[2] == 1:
            sky = sky.squeeze(2)
        elif sky.dim() == 5 and sky.shape[-1] == 1:
            sky = sky.squeeze(-1)
        elif sky.dim() != 4:
            raise ValueError(f"Unsupported DA3 sky shape: {tuple(sky.shape)}")
        canonicalized["sky"] = sky
    for flow_key in ("scene_flow", "scene_flow_cam"):
        scene_flow = outputs.get(flow_key)
        if scene_flow is None:
            continue
        if scene_flow.dim() == 5 and scene_flow.shape[-1] == 3:
            scene_flow = scene_flow.permute(0, 1, 4, 2, 3)
        elif scene_flow.dim() != 5 or scene_flow.shape[2] != 3:
            raise ValueError(f"Unsupported DA3 {flow_key} shape: {tuple(scene_flow.shape)}")
        canonicalized[flow_key] = scene_flow
    return canonicalized


def convert_da3_output_to_plain_dict(outputs: Any) -> dict[str, Any]:
    # FSDP treats dict-like objects exposing `__dataclass_fields__` as dataclasses.
    # DA3 heads may return addict.Dict, so normalize to a plain dict before the
    # trainable wrapper returns.
    if isinstance(outputs, dict) and type(outputs) is dict:
        return outputs
    if hasattr(outputs, "items"):
        return dict(outputs)
    raise TypeError(f"Unsupported DA3 output type: {type(outputs)}")


def attach_da3_camera_decode_outputs(
    outputs: dict[str, Any],
    *,
    camera_tokens: torch.Tensor | None,
    camera_decoder: nn.Module | None,
    image_hw: tuple[int, int],
) -> dict[str, Any]:
    updated = dict(outputs)
    if camera_tokens is None:
        return updated

    updated["camera_tokens"] = camera_tokens
    updated["camera_image_hw"] = (int(image_hw[0]), int(image_hw[1]))

    if camera_decoder is None:
        return updated

    decoded_camera = _decode_da3_camera_from_token_with_module_dtype(
        camera_tokens,
        camera_decoder,
        image_height=int(image_hw[0]),
        image_width=int(image_hw[1]),
    )
    updated["camera_pose_encoding"] = decoded_camera["pose_encoding"]
    updated["camera_extrinsics"] = decoded_camera["extrinsics"]
    updated["camera_intrinsics"] = decoded_camera["intrinsics"]
    return updated


def _get_decoder_autocast_context(
    *,
    device_type: str,
    decoder_dtype: torch.dtype | None,
):
    if decoder_dtype not in (torch.bfloat16, torch.float16):
        return nullcontext()
    if device_type == "cpu" and decoder_dtype != torch.bfloat16:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=decoder_dtype)


def _decode_da3_camera_from_token_with_module_dtype(
    camera_tokens: torch.Tensor,
    camera_decoder: nn.Module,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse

    if camera_tokens.dim() != 3:
        raise ValueError(
            "Expected camera_tokens with shape [B, T, C], "
            f"got {tuple(camera_tokens.shape)}"
        )

    decoder_dtype = _get_module_floating_dtype(camera_decoder)
    decoder_tokens = camera_tokens
    if decoder_dtype is not None and decoder_tokens.is_floating_point() and decoder_tokens.dtype != decoder_dtype:
        decoder_tokens = decoder_tokens.to(dtype=decoder_dtype)

    with _get_decoder_autocast_context(
        device_type=decoder_tokens.device.type,
        decoder_dtype=decoder_dtype,
    ):
        pose_encoding = camera_decoder(decoder_tokens)

    c2w_extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding,
        (image_height, image_width),
    )
    extrinsics = affine_inverse(c2w_extrinsics)[..., :3, :]

    return {
        "pose_encoding": pose_encoding,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def flatten_da3_feature_grid_to_tokens(feature_grid: torch.Tensor) -> torch.Tensor:
    if feature_grid.dim() != 5:
        raise ValueError(
            "feature_grid must be [B, C, T, H, W], "
            f"got {tuple(feature_grid.shape)}"
        )
    return rearrange(feature_grid, "b c t h w -> b t (h w) c").contiguous()


def reshape_da3_feature_tokens(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    if tokens.dim() == 3:
        return tokens.reshape(batch_size, num_frames, tokens.shape[1], tokens.shape[2]).contiguous()
    if tokens.dim() == 4:
        return tokens.contiguous()
    raise ValueError(
        "distill feature tokens must be [B*T, N, C] or [B, T, N, C], "
        f"got {tuple(tokens.shape)}"
    )


def normalize_extrinsics_like_official_da3_api(extrinsics: torch.Tensor) -> torch.Tensor:
    from depth_anything_3.utils.geometry import affine_inverse

    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(
            "Expected extrinsics with shape [B, T, 4, 4], "
            f"got {tuple(extrinsics.shape)}"
        )

    transform = affine_inverse(extrinsics[:, :1])
    normalized = extrinsics @ transform
    c2ws = affine_inverse(normalized)
    translations = c2ws[..., :3, 3]
    dists = translations.norm(dim=-1)
    median_dist = torch.median(dists)
    median_dist = torch.clamp(median_dist, min=1e-1)
    normalized = normalized.clone()
    normalized[..., :3, 3] = normalized[..., :3, 3] / median_dist
    return normalized


def decode_official_da3_camera_params(
    ray: torch.Tensor,
    ray_conf: torch.Tensor | None,
    *,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    from depth_anything_3.model import da3 as official_da3
    from depth_anything_3.utils.geometry import affine_inverse

    if ray.dim() == 5 and ray.shape[2] == 6:
        ray_hw_last = ray.permute(0, 1, 3, 4, 2).contiguous()
    elif ray.dim() == 5 and ray.shape[-1] == 6:
        ray_hw_last = ray
    else:
        raise ValueError(f"Unsupported DA3 ray shape: {tuple(ray.shape)}")

    if ray_conf is None:
        ray_conf_hw = None
    elif ray_conf.dim() == 5 and ray_conf.shape[2] == 1:
        ray_conf_hw = ray_conf.squeeze(2)
    elif ray_conf.dim() == 5 and ray_conf.shape[-1] == 1:
        ray_conf_hw = ray_conf.squeeze(-1)
    elif ray_conf.dim() == 4:
        ray_conf_hw = ray_conf
    else:
        raise ValueError(f"Unsupported DA3 ray_conf shape: {tuple(ray_conf.shape)}")

    pred_extrinsic, pred_focal_lengths, pred_principal_points = official_da3.get_extrinsic_from_camray(
        ray_hw_last,
        ray_conf_hw,
        ray_hw_last.shape[-3],
        ray_hw_last.shape[-2],
    )
    pred_extrinsic = affine_inverse(pred_extrinsic)[:, :, :3, :]

    pred_intrinsic = torch.eye(
        3,
        device=pred_extrinsic.device,
        dtype=pred_extrinsic.dtype,
    )[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1)
    pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] * image_width * 0.5
    pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] * image_height * 0.5
    pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * image_width * 0.5
    pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * image_height * 0.5

    return {
        "extrinsics": pred_extrinsic,
        "intrinsics": pred_intrinsic,
        "focal_lengths": pred_focal_lengths,
        "principal_points": pred_principal_points,
    }


def apply_official_da3_mono_sky_postprocess(
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from depth_anything_3.utils.alignment import (
        compute_sky_mask,
        set_sky_regions_to_max_depth,
    )

    sky = outputs.get("sky")
    if sky is None:
        return outputs

    non_sky_mask = compute_sky_mask(sky, threshold=0.3)
    if non_sky_mask.sum() <= 10:
        return outputs
    if (~non_sky_mask).sum() <= 10:
        return outputs

    depth = outputs["depth"]
    squeeze_channel = depth.dim() == 5 and depth.shape[2] == 1
    if squeeze_channel:
        depth = depth.squeeze(2)

    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100000:
        idx = torch.randint(
            0,
            non_sky_depth.numel(),
            (100000,),
            device=non_sky_depth.device,
        )
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth
    non_sky_max = torch.quantile(sampled_depth, 0.99)

    depth, _ = set_sky_regions_to_max_depth(
        depth,
        None,
        non_sky_mask,
        max_depth=non_sky_max,
    )
    updated = dict(outputs)
    updated["depth"] = depth.unsqueeze(2) if squeeze_channel else depth
    return updated


def _get_module_floating_dtype(module: nn.Module) -> torch.dtype | None:
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def _cast_backbone_outputs_dtype(
    features: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    target_dtype: torch.dtype | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if target_dtype is None:
        return features

    casted = []
    for token_features, camera_tokens in features:
        if token_features.is_floating_point() and token_features.dtype != target_dtype:
            token_features = token_features.to(dtype=target_dtype)
        if camera_tokens.is_floating_point() and camera_tokens.dtype != target_dtype:
            camera_tokens = camera_tokens.to(dtype=target_dtype)
        casted.append((token_features, camera_tokens))
    return tuple(casted)


def _align_dense_btchw_to_target(
    value: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if value.shape[1] != target.shape[1]:
        timeline = torch.linspace(
            0,
            value.shape[1] - 1,
            target.shape[1],
            device=value.device,
        ).round().long().clamp(0, value.shape[1] - 1)
        value = value.index_select(1, timeline)
    if value.shape[-2:] != target.shape[-2:]:
        flat = value.reshape(-1, value.shape[2], value.shape[3], value.shape[4])
        flat = F.interpolate(
            flat,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        value = flat.reshape(value.shape[0], value.shape[1], value.shape[2], *target.shape[-2:])
    return value


def attach_da3_scene_flow_outputs(
    outputs: dict[str, Any],
    *,
    features: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    scene_flow_head: nn.Module | None,
    scene_flow_adapter: nn.Module | None,
) -> dict[str, Any]:
    if scene_flow_head is None:
        return outputs

    updated = dict(outputs)
    extrinsics = updated.get("camera_extrinsics", updated.get("extrinsics"))
    if extrinsics is None:
        return updated

    scene_flow_head_dtype = _get_module_floating_dtype(scene_flow_head)
    scene_flow_features = _cast_backbone_outputs_dtype(
        features,
        scene_flow_head_dtype,
    )
    with _get_decoder_autocast_context(
        device_type=scene_flow_features[0][0].device.type,
        decoder_dtype=scene_flow_head_dtype,
    ):
        scene_flow_outputs = scene_flow_head(
            feats=scene_flow_features,
            H=int(updated["depth"].shape[-2]),
            W=int(updated["depth"].shape[-1]),
            patch_start_idx=0,
        )
    scene_flow_outputs = convert_da3_output_to_plain_dict(scene_flow_outputs)
    scene_flow_cam = scene_flow_outputs.get("scene_flow_cam")
    if scene_flow_cam is None:
        return updated

    if scene_flow_cam.shape[1] > 0:
        scene_flow_cam = scene_flow_cam.clone()
        scene_flow_cam[:, -1] = 0
    updated["scene_flow_cam"] = scene_flow_cam

    if scene_flow_adapter is not None:
        scene_flow = scene_flow_adapter(
            scene_flow_cam=scene_flow_cam,
            extrinsics=extrinsics,
        )
        if scene_flow.shape[1] > 0:
            scene_flow = scene_flow.clone()
            scene_flow[:, -1] = 0
        updated["scene_flow"] = scene_flow
    return updated


def _should_use_reentrant_checkpoint(block: nn.Module) -> bool:
    """Use reentrant checkpointing when a block carries frozen params under ZeRO-3.

    LoRA-wrapped DA3 blocks keep the original large weights frozen while only the
    low-rank adapters remain trainable. Under ZeRO-3, non-reentrant checkpoint
    recomputation may observe those frozen base weights as empty local shards,
    which produces the metadata mismatch seen during backward. Fully trainable
    blocks can keep the cheaper non-reentrant path.
    """
    has_parameter = False
    for parameter in block.parameters():
        has_parameter = True
        if not parameter.requires_grad:
            return True
    return False if has_parameter else False


@torch.no_grad()
def extract_da3_teacher_outputs(
    model: nn.Module,
    context_image: torch.Tensor,
    ref_view_strategy: str = "first",
    *,
    include_distill_feature_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    da3_model = get_da3_anyview_model(model)
    images = normalize_images_imagenet(_prepare_da3_images(context_image))
    raw_outputs = da3_model.backbone.forward(images, ref_view_strategy=ref_view_strategy)
    features = raw_outputs[0] if isinstance(raw_outputs, tuple) else raw_outputs
    outputs = da3_model.head(features, images.shape[-2], images.shape[-1], patch_start_idx=0)
    outputs = convert_da3_output_to_plain_dict(outputs)
    outputs = attach_da3_camera_decode_outputs(
        outputs,
        camera_tokens=features[-1][1] if len(features) > 0 else None,
        camera_decoder=getattr(da3_model, "cam_dec", None),
        image_hw=(images.shape[-2], images.shape[-1]),
    )
    outputs = attach_da3_scene_flow_outputs(
        outputs,
        features=features,
        scene_flow_head=getattr(da3_model, "scene_flow_head", None),
        scene_flow_adapter=getattr(da3_model, "scene_flow_adapter", None),
    )
    if include_distill_feature_tokens:
        feature_map = extract_da3_candidate_features(
            model,
            context_image,
            candidate_layers=(DA3_DISTILL_FEATURE_LAYER,),
        )
        teacher_tokens = feature_map.get(DA3_DISTILL_FEATURE_LAYER)
        if teacher_tokens is not None:
            outputs["distill_feature_tokens"] = reshape_da3_feature_tokens(
                teacher_tokens,
                batch_size=context_image.shape[0],
                num_frames=context_image.shape[2],
            )
    return outputs


@torch.no_grad()
def extract_da3_candidate_features(
    model: nn.Module,
    context_image: torch.Tensor,
    candidate_layers: Iterable[str] = DA3_STITCH_CANDIDATE_LAYER_NAMES,
) -> dict[str, torch.Tensor]:
    da3_model = get_da3_anyview_model(model)
    vit = da3_model.backbone.pretrained
    requested_layers = sorted(int(layer_name.split("_")[-1]) for layer_name in candidate_layers)
    max_requested_layer = max(requested_layers)
    images = normalize_images_imagenet(_prepare_da3_images(context_image))
    tokens = vit.prepare_tokens_with_masks(images)

    feature_map: dict[str, torch.Tensor] = {}
    for block_idx, block in enumerate(vit.blocks[:max_requested_layer], start=1):
        tokens = vit.process_attention(tokens, block, "local", pos=None)
        if block_idx not in requested_layers:
            continue
        normalized = vit.norm(tokens)
        normalized = normalized[..., 1 + vit.num_register_tokens :, :]
        feature_map[f"enc_blocks_{block_idx}"] = normalized.detach()

    return feature_map


def _postprocess_backbone_outputs(vit: nn.Module, raw_outputs: list[tuple[torch.Tensor, torch.Tensor]]):
    camera_tokens = [out[0] for out in raw_outputs]
    token_outputs = [out[1] for out in raw_outputs]
    if token_outputs[0].shape[-1] == vit.embed_dim:
        token_outputs = [vit.norm(out) for out in token_outputs]
    elif token_outputs[0].shape[-1] == (vit.embed_dim * 2):
        token_outputs = [
            torch.cat(
                [
                    out[..., : vit.embed_dim],
                    vit.norm(out[..., vit.embed_dim :]),
                ],
                dim=-1,
            )
            for out in token_outputs
        ]
    else:
        raise ValueError(f"Invalid DA3 token output shape: {tuple(token_outputs[0].shape)}")

    token_outputs = [out[..., 1 + vit.num_register_tokens :, :] for out in token_outputs]
    return tuple(zip(token_outputs, camera_tokens))


class TaskLossDA3:
    def __init__(
        self,
        *,
        lambda_depth: float = 1.0,
        lambda_depth_grad: float = 1.0,
        lambda_ray: float = 1.0,
        lambda_depth_conf: float = 0.005,
        lambda_ray_conf: float = 0.005,
        lambda_camera: float = 0.0,
        lambda_scene_flow: float = 1.0,
        lambda_feature_distill: float = 0.0,
        lambda_temporal: float = 1.0,
        temporal_trim: float = 0.2,
        temporal_grad_scales: int = 1,
        temporal_grad_decay: float = 0.5,
        temporal_diff_depth_th: float = 0.05,
    ):
        self.lambda_depth = lambda_depth
        self.lambda_depth_grad = lambda_depth_grad
        self.lambda_ray = lambda_ray
        self.lambda_depth_conf = lambda_depth_conf
        self.lambda_ray_conf = lambda_ray_conf
        self.lambda_camera = lambda_camera
        self.lambda_scene_flow = lambda_scene_flow
        self.lambda_feature_distill = lambda_feature_distill
        self.lambda_temporal = lambda_temporal
        self.camera_loss = DecodedCameraLoss(reduction="mean")
        self.temporal_loss_fn = build_temporal_gradient_matching_loss(
            trim=temporal_trim,
            temporal_grad_scales=temporal_grad_scales,
            temporal_grad_decay=temporal_grad_decay,
            temporal_diff_depth_th=temporal_diff_depth_th,
            reduction="batch-based",
        )

    def __call__(self, stitched_out: dict[str, torch.Tensor], ff_out: dict[str, torch.Tensor]):
        student_feature_tokens = None
        teacher_feature_tokens = None
        if self.lambda_feature_distill > 0:
            student_feature_tokens = stitched_out.get("distill_feature_tokens")
            teacher_feature_tokens = ff_out.get("distill_feature_tokens")
        stitched = canonicalize_da3_geometry_outputs(stitched_out)
        teacher = canonicalize_da3_geometry_outputs(ff_out)
        student_camera_extrinsics = stitched_out.get("camera_extrinsics", stitched_out.get("extrinsics"))
        student_camera_intrinsics = stitched_out.get("camera_intrinsics", stitched_out.get("intrinsics"))
        teacher_camera_extrinsics = ff_out.get("camera_extrinsics", ff_out.get("extrinsics"))
        teacher_camera_intrinsics = ff_out.get("camera_intrinsics", ff_out.get("intrinsics"))
        student_scene_flow = stitched_out.get("scene_flow")
        if student_scene_flow is None:
            student_scene_flow = stitched_out.get("scene_flow_cam")
        teacher_scene_flow = ff_out.get("scene_flow")
        if teacher_scene_flow is None:
            teacher_scene_flow = ff_out.get("scene_flow_cam")

        depth_loss_raw = F.l1_loss(stitched["depth"], teacher["depth"])
        depth_grad_loss_raw = _gradient_loss_multi_scale(
            rearrange(stitched["depth"], "b t c h w -> (b t) h w c"),
            rearrange(teacher["depth"], "b t c h w -> (b t) h w c"),
        )
        ray_loss_raw = F.l1_loss(stitched["ray"], teacher["ray"])
        depth_conf_loss_raw = F.l1_loss(stitched["depth_conf"], teacher["depth_conf"])
        ray_conf_loss_raw = F.l1_loss(stitched["ray_conf"], teacher["ray_conf"])
        if student_scene_flow is not None and teacher_scene_flow is not None and self.lambda_scene_flow > 0:
            teacher_scene_flow = _align_dense_btchw_to_target(teacher_scene_flow, student_scene_flow)
            scene_flow_loss_raw = F.l1_loss(student_scene_flow, teacher_scene_flow)
        else:
            scene_flow_loss_raw = stitched["depth"].new_tensor(0.0)
        if (
            student_feature_tokens is not None
            and teacher_feature_tokens is not None
            and self.lambda_feature_distill > 0
        ):
            if student_feature_tokens.shape[1] != teacher_feature_tokens.shape[1]:
                timeline = torch.linspace(
                    0,
                    teacher_feature_tokens.shape[1] - 1,
                    student_feature_tokens.shape[1],
                    device=teacher_feature_tokens.device,
                ).round().long()
                teacher_feature_tokens = teacher_feature_tokens.index_select(1, timeline)
            if student_feature_tokens.shape != teacher_feature_tokens.shape:
                raise ValueError(
                    "distill_feature_tokens shape mismatch: "
                    f"student={tuple(student_feature_tokens.shape)} "
                    f"teacher={tuple(teacher_feature_tokens.shape)}"
                )
            feature_distill_loss_raw = F.mse_loss(
                student_feature_tokens,
                teacher_feature_tokens,
            )
        else:
            feature_distill_loss_raw = stitched["depth"].new_tensor(0.0)
        if stitched["depth"].shape[1] > 1 and self.lambda_temporal > 0:
            temporal_mask = (
                torch.isfinite(stitched["depth"].squeeze(2))
                & torch.isfinite(teacher["depth"].squeeze(2))
            )
            temporal_loss_raw = self.temporal_loss_fn(
                prediction=stitched["depth"].squeeze(2),
                target=teacher["depth"].squeeze(2),
                mask=temporal_mask,
            )
        else:
            temporal_loss_raw = stitched["depth"].new_tensor(0.0)

        if (
            self.lambda_camera > 0
            and student_camera_extrinsics is not None
            and student_camera_intrinsics is not None
            and teacher_camera_extrinsics is not None
            and teacher_camera_intrinsics is not None
        ):
            camera_image_hw = stitched_out.get(
                "camera_image_hw",
                ff_out.get("camera_image_hw", stitched["depth"].shape[-2:]),
            )
            camera_loss_dict_raw = self.camera_loss(
                target_intrinsics=teacher_camera_intrinsics,
                target_extrinsics=teacher_camera_extrinsics,
                pred_intrinsics=student_camera_intrinsics,
                pred_extrinsics=student_camera_extrinsics,
                image_height=int(camera_image_hw[0]),
                image_width=int(camera_image_hw[1]),
            )
            camera_loss_raw = camera_loss_dict_raw["camera_loss"]
        else:
            camera_loss_dict_raw = None
            camera_loss_raw = stitched["depth"].new_tensor(0.0)

        depth_loss = depth_loss_raw * self.lambda_depth
        depth_grad_loss = depth_grad_loss_raw * self.lambda_depth_grad
        ray_loss = ray_loss_raw * self.lambda_ray
        depth_conf_loss = depth_conf_loss_raw * self.lambda_depth_conf
        ray_conf_loss = ray_conf_loss_raw * self.lambda_ray_conf
        camera_loss = camera_loss_raw * self.lambda_camera
        scene_flow_loss = scene_flow_loss_raw * self.lambda_scene_flow
        feature_distill_loss = feature_distill_loss_raw * self.lambda_feature_distill
        temporal_loss = temporal_loss_raw * self.lambda_temporal

        loss = {
            "depth_loss": depth_loss,
            "depth_grad_loss": depth_grad_loss,
            "ray_loss": ray_loss,
            "depth_conf_loss": depth_conf_loss,
            "ray_conf_loss": ray_conf_loss,
            "camera_loss": camera_loss,
            "scene_flow_loss": scene_flow_loss,
            "feature_distill_loss": feature_distill_loss,
            "temporal_loss": temporal_loss,
        }
        if camera_loss_dict_raw is not None:
            loss["camera_rotation_loss"] = camera_loss_dict_raw["camera_rotation_loss"] * self.lambda_camera
            loss["camera_translation_loss"] = camera_loss_dict_raw["camera_translation_loss"] * self.lambda_camera
            loss["camera_intrinsics_loss"] = camera_loss_dict_raw["camera_intrinsics_loss"] * self.lambda_camera
            loss["camera_center_loss"] = camera_loss_dict_raw["camera_center_loss"] * self.lambda_camera
        loss["total_loss"] = (
            depth_loss
            + depth_grad_loss
            + ray_loss
            + depth_conf_loss
            + ray_conf_loss
            + camera_loss
            + scene_flow_loss
            + feature_distill_loss
            + temporal_loss
        )
        return loss


class DA3Stitched(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        stitching_layer: str,
        *,
        ref_view_strategy: str = "first",
    ):
        super().__init__()
        self.model = copy.deepcopy(get_da3_anyview_model(model))
        self.backbone = self.model.backbone
        self.head = self.model.head
        self.ref_view_strategy = ref_view_strategy
        self.stitching_layer = stitching_layer
        self.stitched_layer_index = int(stitching_layer.split("_")[-1])
        if not (1 <= self.stitched_layer_index <= 12):
            raise ValueError(
                "DA3 stitching layer must be one of enc_blocks_1..enc_blocks_12 "
                f"to preserve native anyview routing, got {stitching_layer}."
            )
        self.grad_checkpointing = True

    def _prepare_injected_tokens(
        self,
        context_latent: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        vit = self.backbone.pretrained
        batch_size, _, num_frames, _, _ = context_latent.shape
        tokens = rearrange(context_latent, "b c t h w -> (b t) (h w) c")
        cls_token = vit.prepare_cls_token(batch_size, num_frames)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + vit.interpolate_pos_encoding(tokens, height, width)
        if vit.register_tokens is not None:
            tokens = torch.cat(
                (
                    tokens[:, :1],
                    vit.register_tokens.expand(tokens.shape[0], -1, -1),
                    tokens[:, 1:],
                ),
                dim=1,
            )
        return rearrange(tokens, "(b t) n c -> b t n c", b=batch_size, t=num_frames)

    def _run_attention_block(
        self,
        vit: nn.Module,
        tokens: torch.Tensor,
        block: nn.Module,
        attn_type: str,
        pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        enable_checkpointing: bool = False,
        gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        if not (self.grad_checkpointing and enable_checkpointing):
            return vit.process_attention(tokens, block, attn_type, pos=pos, attn_mask=attn_mask)

        def _forward(x: torch.Tensor) -> torch.Tensor:
            return vit.process_attention(x, block, attn_type, pos=pos, attn_mask=attn_mask)

        use_reentrant = _should_use_reentrant_checkpoint(block)
        if gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                return checkpoint(_forward, tokens, use_reentrant=use_reentrant)
        return checkpoint(_forward, tokens, use_reentrant=use_reentrant)

    def _resolve_image_hw(
        self,
        context_image: torch.Tensor | None = None,
        image_hw: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        if image_hw is not None:
            return image_hw
        if context_image is None:
            raise ValueError("DA3 stitched forward requires either context_image or image_hw.")
        images = _prepare_da3_images(context_image)
        return images.shape[-2], images.shape[-1]

    def _get_cam_enc(self) -> nn.Module | None:
        model = getattr(self, "model", None)
        if model is None:
            return None
        return getattr(model, "cam_enc", None)

    def get_cam_dec(self) -> nn.Module | None:
        model = getattr(self, "model", None)
        if model is None:
            return None
        return getattr(model, "cam_dec", None)

    def _forward_backbone_from_latents(
        self,
        context_latent: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        cam_token: torch.Tensor | None = None,
        enable_checkpointing: bool = False,
        gradient_checkpointing_offload: bool = False,
    ):
        batch_size, _, num_frames, _, _ = context_latent.shape
        height, width = image_hw
        vit = self.backbone.pretrained
        tokens = self._prepare_injected_tokens(context_latent, height, width)

        pos, pos_nodiff = vit._prepare_rope(batch_size, num_frames, height, width, tokens.device)
        output_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        local_tokens = tokens
        reorder_indices = None

        for block_index, block in enumerate(vit.blocks):
            if block_index < self.stitched_layer_index:
                continue

            if block_index < vit.rope_start or vit.rope is None:
                global_pos = None
                local_pos = None
            else:
                global_pos = pos_nodiff
                local_pos = pos

            if (
                vit.alt_start != -1
                and block_index == vit.alt_start - 1
                and tokens.shape[1] >= THRESH_FOR_REF_SELECTION
                and cam_token is None
            ):
                reorder_indices = select_reference_view(tokens, strategy=self.ref_view_strategy)
                tokens = reorder_by_reference(tokens, reorder_indices)
                local_tokens = reorder_by_reference(local_tokens, reorder_indices)

            if vit.alt_start != -1 and block_index == vit.alt_start:
                if cam_token is not None:
                    camera_token = cam_token.to(device=tokens.device, dtype=tokens.dtype)
                else:
                    ref_token = vit.camera_token[:, :1].expand(batch_size, -1, -1)
                    src_token = vit.camera_token[:, 1:].expand(batch_size, num_frames - 1, -1)
                    camera_token = torch.cat([ref_token, src_token], dim=1)
                tokens = tokens.clone()
                tokens[:, :, 0] = camera_token

            if vit.alt_start != -1 and block_index >= vit.alt_start and block_index % 2 == 1:
                tokens = self._run_attention_block(
                    vit,
                    tokens,
                    block,
                    "global",
                    pos=global_pos,
                    attn_mask=None,
                    enable_checkpointing=enable_checkpointing,
                    gradient_checkpointing_offload=gradient_checkpointing_offload,
                )
            else:
                tokens = self._run_attention_block(
                    vit,
                    tokens,
                    block,
                    "local",
                    pos=local_pos,
                    attn_mask=None,
                    enable_checkpointing=enable_checkpointing,
                    gradient_checkpointing_offload=gradient_checkpointing_offload,
                )
                local_tokens = tokens

            if block_index in self.backbone.out_layers:
                output = (
                    torch.cat([local_tokens, tokens], dim=-1) if vit.cat_token else tokens
                )
                if reorder_indices is not None and vit.alt_start != -1:
                    output = restore_original_order(output, reorder_indices)
                output_entries.append((output[:, :, 0], output))

        return _postprocess_backbone_outputs(vit, output_entries)

    def forward(
        self,
        context_latent: torch.Tensor,
        context_image: torch.Tensor | None = None,
        train: bool = False,
        image_hw: tuple[int, int] | None = None,
        camera_extrinsics: torch.Tensor | None = None,
        camera_intrinsics: torch.Tensor | None = None,
        gradient_checkpointing_offload: bool = False,
    ) -> dict[str, torch.Tensor]:
        height, width = self._resolve_image_hw(context_image=context_image, image_hw=image_hw)
        cam_token = None
        if camera_extrinsics is not None or camera_intrinsics is not None:
            if camera_extrinsics is None or camera_intrinsics is None:
                raise ValueError("camera_extrinsics and camera_intrinsics must be provided together.")
            cam_enc = self._get_cam_enc()
            if cam_enc is None:
                raise ValueError("DA3 stitched model does not expose cam_enc for camera-conditioned inference.")
            with torch.autocast(device_type=context_latent.device.type, enabled=False):
                cam_token = cam_enc(
                    camera_extrinsics.float(),
                    camera_intrinsics.float(),
                    (height, width),
                )
        features = self._forward_backbone_from_latents(
            context_latent,
            image_hw=(height, width),
            cam_token=cam_token,
            enable_checkpointing=train,
            gradient_checkpointing_offload=gradient_checkpointing_offload,
        )
        features_for_head = _cast_backbone_outputs_dtype(features, _get_module_floating_dtype(self.head))
        outputs = self.head(features_for_head, height, width, patch_start_idx=0)
        outputs = convert_da3_output_to_plain_dict(outputs)
        outputs = attach_da3_camera_decode_outputs(
            outputs,
            camera_tokens=features[-1][1] if len(features) > 0 else None,
            camera_decoder=self.get_cam_dec(),
            image_hw=(height, width),
        )
        return attach_da3_scene_flow_outputs(
            outputs,
            features=features,
            scene_flow_head=getattr(self.model, "scene_flow_head", None),
            scene_flow_adapter=getattr(self.model, "scene_flow_adapter", None),
        )
