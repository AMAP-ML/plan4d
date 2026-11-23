import argparse
import os
import random
import sys
import warnings
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
DIFFSYNTH_ROOT = THIS_FILE.parents[3]
PROJECT_ROOT = THIS_FILE.parents[4]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT

if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import accelerate
import torch
import torch.nn.functional as F
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    ImageCropAndResize,
    LoadAudio,
    LoadPointOdysseyClip,
    LoadTorchPickle,
    LoadVideo,
    ToAbsolutePath,
)
from diffsynth.diffusion import *
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.da3_stitched import (
    DA3Stitched,
    TaskLossDA3,
    canonicalize_da3_geometry_outputs,
    extract_da3_teacher_outputs,
)
from data.stitching_manifest import build_manifest_training_dataset
from models.stitching_layer_builder import parse_conv_spec
from utils.da3_resize import coerce_image_hw
from utils.gt_loss_alignment import compute_gt_geometry_loss_from_outputs
from utils.geo_loss_schedule import compute_geo_loss_warmup_scale
from utils.scene_flow_supervision import GeometrySceneFlowLoss
from utils.argument import parse_existing_file, parse_manifest_spec
from utils.utils_for_thirdparty import configure_hf_runtime, load_feedforward_model


DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints" / "stitched_decoder.pth")


def _parse_int_csv(values: str | None, default_values=(9, 13, 17)):
    if values is None:
        return list(default_values)
    parsed = []
    for item in values.split(","):
        item = item.strip()
        if item == "":
            continue
        parsed.append(int(item))
    return parsed if len(parsed) > 0 else list(default_values)


def _broadcast_flowmatch_loss_weight_mask(mask, target):
    if mask is None:
        return torch.ones_like(target, dtype=target.dtype, device=target.device)
    mask = mask.to(device=target.device, dtype=target.dtype)
    if mask.dim() == target.dim() - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != target.shape:
        mask = mask.expand_as(target)
    return mask


def _get_scheduler_sigma_for_timestep(scheduler, timestep):
    timestep_cpu = timestep.detach().to(device=scheduler.timesteps.device).cpu()
    timestep_id = torch.argmin((scheduler.timesteps - timestep_cpu).abs())
    return scheduler.sigmas[timestep_id].to(device=timestep.device, dtype=timestep.dtype)


def FlowMatchSFTCFGLoss(pipe, return_details=False, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    # Sync timestep across ranks. Without this, per-rank torch RNG diverges and
    # any timestep-conditional path in DiT / scheduler breaks ZeRO-3 collective
    # alignment. NCCL broadcast needs the tensor on GPU; move back to CPU after
    # because pipe.scheduler.timesteps is a CPU tensor and requires a CPU index.
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gpu_buf = timestep_id.to(device=pipe.device)
        torch.distributed.broadcast(gpu_buf, src=0)
        timestep_id = gpu_buf.cpu()
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    # CFG removed: always use unconditional forward (cfg_scale=1)
    inputs.pop("negative_context", None)

    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)

    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]

    mse = (noise_pred.float() - training_target.float()) ** 2
    loss_weight_mask = _broadcast_flowmatch_loss_weight_mask(inputs.get("loss_weight_mask", None), mse)
    if "first_frame_latents" in inputs and inputs.get("loss_weight_mask", None) is not None:
        loss_weight_mask = loss_weight_mask[:, :, 1:]
    masked_mse = mse * loss_weight_mask
    denom = loss_weight_mask.sum().clamp(min=1e-8)
    loss = masked_mse.sum() / denom
    loss = loss * pipe.scheduler.training_weight(timestep)

    if not return_details:
        return loss

    latents_for_x = inputs["latents"]
    if "first_frame_latents" in inputs:
        latents_for_x = latents_for_x[:, :, 1:]
    sigma = _get_scheduler_sigma_for_timestep(pipe.scheduler, timestep)
    while sigma.dim() < latents_for_x.dim():
        sigma = sigma.unsqueeze(-1)
    x_pred = latents_for_x - sigma * noise_pred
    details = {
        "timestep": timestep,
        "noise_pred": noise_pred,
        "training_target": training_target,
        "mse": mse,
        "loss_weight_mask": loss_weight_mask,
        "masked_mse": masked_mse,
        "x_pred": x_pred,
    }
    return loss, details


def _to_tensor_if_possible(value):
    if isinstance(value, torch.Tensor):
        return value
    return None


def _ensure_bthw1(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 2:
        return depth.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    if depth.dim() == 3:
        return depth.unsqueeze(0).unsqueeze(2)
    if depth.dim() == 4:
        if depth.shape[-1] == 1:
            return depth.unsqueeze(0).permute(0, 1, 4, 2, 3)
        return depth.unsqueeze(2)
    if depth.dim() == 5:
        if depth.shape[2] == 1:
            return depth
        if depth.shape[-1] == 1:
            return depth.permute(0, 1, 4, 2, 3)
    raise ValueError(f"Unsupported depth shape: {tuple(depth.shape)}")


def _ensure_bt6hw(ray: torch.Tensor) -> torch.Tensor:
    if ray.dim() == 4 and ray.shape[0] in (6, 7):
        return ray.unsqueeze(0).unsqueeze(0)[:, :, :6]
    if ray.dim() == 4 and ray.shape[-1] in (6, 7):
        return ray.unsqueeze(0).permute(0, 1, 4, 2, 3)[:, :, :6]
    if ray.dim() == 5 and ray.shape[2] in (6, 7):
        return ray[:, :, :6]
    if ray.dim() == 5 and ray.shape[-1] in (6, 7):
        return ray.permute(0, 1, 4, 2, 3)[:, :, :6]
    raise ValueError(f"Unsupported ray shape: {tuple(ray.shape)}")


def _ensure_bcthw(video: torch.Tensor) -> torch.Tensor:
    if video.dim() == 4:
        return video.unsqueeze(0)
    if video.dim() == 5:
        return video
    raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")


def _resample_btchw(tensor: torch.Tensor | None, target_frames: int, mode: str = "trilinear"):
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


def _resize_spatial_btchw(tensor: torch.Tensor | None, target_hw: tuple[int, int], mode: str = "bilinear"):
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


def _match_time_dimension(tensor, target_frames: int):
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


def _resize_video_bcthw_to_hw(
    video: torch.Tensor,
    target_hw: tuple[int, int],
) -> torch.Tensor:
    target_height, target_width = int(target_hw[0]), int(target_hw[1])
    if video.shape[-2:] == (target_height, target_width):
        return video
    return F.interpolate(
        video,
        size=(video.shape[2], target_height, target_width),
        mode="trilinear",
        align_corners=False,
    )


def _cast_module_floating_tensors(module: torch.nn.Module | None, dtype: torch.dtype | None) -> None:
    if module is None or dtype is None or not isinstance(module, torch.nn.Module):
        return
    module.to(dtype=dtype)


def _uses_shape_only_da3_student(stitched_decoder) -> bool:
    candidate = stitched_decoder.module if hasattr(stitched_decoder, "module") else stitched_decoder
    stitched_model = getattr(candidate, "stitched_3d_model", None)
    return isinstance(stitched_model, DA3Stitched)


def _sync_stitched_checkpoint_runtime_overrides(module, loader_args) -> dict[str, object]:
    overrides = (
        ("resolution", "stitched_resolution"),
        ("feedforward_resolution", "feedforward_resolution"),
        ("da3_ref_view_strategy", "ref_view_strategy"),
        ("stitching_layer_location", "stitching_layer_location"),
        ("lora_config", "lora_config"),
        ("full_finetune_stitched_parts", "full_finetune_stitched_parts"),
    )
    applied: dict[str, object] = {}
    for source_attr, target_attr in overrides:
        if not hasattr(loader_args, source_attr):
            continue
        value = getattr(loader_args, source_attr)
        if value is None:
            continue
        if getattr(module, target_attr, None) == value:
            continue
        setattr(module, target_attr, value)
        applied[target_attr] = value
    return applied


def _align_da3_geometry_to_target(source_outputs, target_outputs):
    source = canonicalize_da3_geometry_outputs(source_outputs)
    target = canonicalize_da3_geometry_outputs(target_outputs)

    aligned = {}
    for key, value in source.items():
        target_value = target[key]
        target_frames = target_value.shape[1]
        target_hw = target_value.shape[-2:]

        if value.shape[1] != target_frames:
            value = _resample_btchw(value, target_frames, mode="trilinear")
        if key == "ray" and value.shape[2] > target_value.shape[2]:
            value = value[:, :, : target_value.shape[2]]
        if value.shape[-2:] != target_hw:
            value = _resize_spatial_btchw(value, target_hw, mode="bilinear")
        aligned[key] = value
    return aligned


def _align_da3_distill_outputs_to_target(source_outputs, target_outputs):
    return _align_da3_geometry_to_target(source_outputs, target_outputs)


def _resolve_manifest_process_res_method(args) -> str:
    return getattr(
        args,
        "manifest_process_res_method",
        getattr(args, "da3_process_res_method", "upper_bound_resize"),
    )


def _resolve_distill_manifest_specs(args) -> list[tuple[str, Path]]:
    manifest_specs = list(getattr(args, "train_manifest", None) or [])
    action100m_manifest = getattr(args, "action100m_manifest", None)
    if action100m_manifest:
        manifest_specs.append(("action100m", Path(action100m_manifest).expanduser()))
    return manifest_specs


def _resolve_joint_teacher_rgb_video(joint_inputs) -> torch.Tensor:
    teacher_rgb_video = joint_inputs.get("teacher_rgb_video")
    if teacher_rgb_video is not None:
        return teacher_rgb_video
    return joint_inputs["gt_rgb_video"]


def _resolve_joint_student_image_hw(
    joint_inputs,
    *,
    joint_training_stage: str,
) -> tuple[int, int]:
    del joint_training_stage
    raw_image_hw = joint_inputs.get("image_hw")
    if raw_image_hw is not None:
        return coerce_image_hw(raw_image_hw)
    teacher_rgb_video = _resolve_joint_teacher_rgb_video(joint_inputs)
    return int(teacher_rgb_video.shape[-2]), int(teacher_rgb_video.shape[-1])


def build_stitched_joint_manifest_datasets(args, supervision_mode: str | None = None):
    resolved_supervision_mode = str(
        supervision_mode or getattr(args, "geometry_supervision_mode", "distill")
    ).lower()
    manifest_specs = list(getattr(args, "train_manifest", None) or [])
    if not manifest_specs:
        raise ValueError("Stitched joint manifest mode requires at least one --train_manifest.")

    manifest_data_root = getattr(args, "manifest_data_root", None) or getattr(
        args, "dataset_base_path", None
    )
    if manifest_data_root is None:
        raise ValueError("Manifest mode requires --manifest_data_root or --dataset_base_path.")

    dataset_kwargs = {
        "manifest_data_root": manifest_data_root,
        "image_resolution": int(args.height),
        "feedforward_image_resolution": int(args.feedforward_resolution),
        "chunk_size": int(args.num_frames),
        "process_res_method": _resolve_manifest_process_res_method(args),
    }
    if resolved_supervision_mode == "hybrid":
        return {
            "gt": build_manifest_training_dataset(
                manifest_specs,
                training_stage="gt",
                **dataset_kwargs,
            ),
            "distill": build_manifest_training_dataset(
                _resolve_distill_manifest_specs(args),
                training_stage="distill",
                **dataset_kwargs,
            ),
        }
    if resolved_supervision_mode == "gt":
        return build_manifest_training_dataset(
            manifest_specs,
            training_stage="gt",
            **dataset_kwargs,
        )
    if resolved_supervision_mode == "distill":
        return build_manifest_training_dataset(
            _resolve_distill_manifest_specs(args),
            training_stage="distill",
            **dataset_kwargs,
        )
    if resolved_supervision_mode == "none":
        return build_manifest_training_dataset(
            _resolve_distill_manifest_specs(args),
            training_stage="distill",
            **dataset_kwargs,
        )
    raise ValueError(
        f"Unsupported supervision_mode={resolved_supervision_mode!r}. Expected distill, gt, hybrid, or none."
    )


class JointStageAnnotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, stage: str):
        self.dataset = dataset
        self.stage = str(stage)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if not isinstance(sample, dict):
            raise TypeError(f"Joint manifest samples must be dict-like, got {type(sample)}")
        annotated = dict(sample)
        annotated["joint_training_stage"] = self.stage
        return annotated


class HybridStageAlignedJointDataset(torch.utils.data.Dataset):
    def __init__(self, gt_dataset: torch.utils.data.Dataset, distill_dataset: torch.utils.data.Dataset):
        self.gt_dataset = JointStageAnnotatedDataset(gt_dataset, stage="gt")
        self.distill_dataset = JointStageAnnotatedDataset(distill_dataset, stage="distill")
        self._gt_size = len(self.gt_dataset)
        self._distill_size = len(self.distill_dataset)
        self._distill_offset = self._gt_size

    def __len__(self):
        return self._gt_size + self._distill_size

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if idx < self._distill_offset:
            return self.gt_dataset[idx]
        return self.distill_dataset[idx - self._distill_offset]

    def _build_padded_stage_indices(
        self,
        *,
        stage: str,
        num_processes: int,
        seed: int,
    ) -> list[int]:
        if stage == "gt":
            stage_size = self._gt_size
            offset = 0
        elif stage == "distill":
            stage_size = self._distill_size
            offset = self._distill_offset
        else:
            raise ValueError(f"Unsupported hybrid stage: {stage}")

        if stage_size <= 0:
            return []

        indices = list(range(offset, offset + stage_size))
        stage_rng = random.Random(int(seed))
        stage_rng.shuffle(indices)
        pad = (-len(indices)) % int(max(1, num_processes))
        if pad > 0:
            indices.extend(indices[:pad])
        return indices

    def build_stage_aligned_order(
        self,
        *,
        num_processes: int,
        epoch: int,
        seed: int,
    ) -> list[int]:
        num_processes = int(max(1, num_processes))
        base_seed = int(seed) + int(epoch) * 100_003
        gt_indices = self._build_padded_stage_indices(
            stage="gt",
            num_processes=num_processes,
            seed=base_seed + 17,
        )
        distill_indices = self._build_padded_stage_indices(
            stage="distill",
            num_processes=num_processes,
            seed=base_seed + 31,
        )
        stage_groups = {
            "gt": [
                gt_indices[start : start + num_processes]
                for start in range(0, len(gt_indices), num_processes)
            ],
            "distill": [
                distill_indices[start : start + num_processes]
                for start in range(0, len(distill_indices), num_processes)
            ],
        }
        schedule = ["gt"] * len(stage_groups["gt"]) + ["distill"] * len(stage_groups["distill"])
        schedule_rng = random.Random(base_seed + 53)
        schedule_rng.shuffle(schedule)

        stage_iterators = {stage: iter(groups) for stage, groups in stage_groups.items()}
        ordered_indices = []
        for stage in schedule:
            ordered_indices.extend(next(stage_iterators[stage]))
        return ordered_indices


def build_stitched_joint_training_dataset(args):
    if not getattr(args, "train_manifest", None):
        return None

    if str(getattr(args, "geometry_supervision_mode", "distill")).lower() == "hybrid":
        datasets = build_stitched_joint_manifest_datasets(args, supervision_mode="hybrid")
        return HybridStageAlignedJointDataset(
            datasets["gt"],
            datasets["distill"],
        )

    dataset = build_stitched_joint_manifest_datasets(args)
    return JointStageAnnotatedDataset(
        dataset,
        stage=str(args.geometry_supervision_mode).lower(),
    )


def compute_stitched_distill_loss(
    *,
    x_pred: torch.Tensor,
    gt_rgb_video: torch.Tensor,
    student_feedforward_image: torch.Tensor | None = None,
    student_image_hw: tuple[int, int] | None = None,
    stitched_decoder,
    da3_teacher,
    loss_criterion,
    ref_view_strategy: str = "saddle_balanced",
    train: bool = False,
    gradient_checkpointing_offload: bool = False,
):
    stitched_out_raw = stitched_decoder.forward_with_latent(
        x_pred,
        feedforward_image=student_feedforward_image,
        image_hw=student_image_hw,
        train=train,
        gradient_checkpointing_offload=gradient_checkpointing_offload,
    )
    include_feature_distill = getattr(loss_criterion, "lambda_feature_distill", 0.0) > 0
    passthrough_keys = [
        "camera_tokens",
        "camera_pose_encoding",
        "camera_extrinsics",
        "camera_intrinsics",
        "camera_image_hw",
        "scene_flow",
        "scene_flow_cam",
    ]
    if include_feature_distill:
        passthrough_keys.append("distill_feature_tokens")

    stitched_out = canonicalize_da3_geometry_outputs(stitched_out_raw)
    for key in passthrough_keys:
        if key in stitched_out_raw:
            stitched_out[key] = stitched_out_raw[key]
    with torch.no_grad():
        teacher_out = extract_da3_teacher_outputs(
            da3_teacher,
            gt_rgb_video,
            ref_view_strategy=ref_view_strategy,
            include_distill_feature_tokens=include_feature_distill,
        )
    teacher_out_aligned = _align_da3_distill_outputs_to_target(teacher_out, stitched_out)
    for key in passthrough_keys:
        if key in teacher_out:
            teacher_out_aligned[key] = teacher_out[key]
    teacher_out = teacher_out_aligned
    return loss_criterion(stitched_out, teacher_out)


def compute_stitched_gt_loss(
    *,
    x_pred: torch.Tensor,
    student_feedforward_image: torch.Tensor | None = None,
    student_image_hw: tuple[int, int] | None = None,
    stitched_decoder,
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
    train: bool = False,
    gradient_checkpointing_offload: bool = False,
):
    stitched_out = stitched_decoder.forward_with_latent(
        x_pred,
        feedforward_image=student_feedforward_image,
        image_hw=student_image_hw,
        train=train,
        gradient_checkpointing_offload=gradient_checkpointing_offload,
    )
    return compute_gt_geometry_loss_from_outputs(
        geometry_outputs=stitched_out,
        geometry_loss_fn=geometry_loss_fn,
        depth_gt=depth_gt,
        ray_gt=ray_gt,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        valid_mask=valid_mask,
        scene_flow_gt=scene_flow_gt,
        deform_mask=deform_mask,
        geo_supervise_generated_only=geo_supervise_generated_only,
        cond_video_frames=cond_video_frames,
        gt_points_3d=None,
        gt_points_2d=None,
        points_valid_mask=None,
    )


def _ensure_plan4d_training_imports() -> None:
    if str(PLAN4D_ROOT) not in sys.path:
        sys.path.insert(0, str(PLAN4D_ROOT))


def _load_geometry_reconstruction_loss():
    _ensure_plan4d_training_imports()
    from plan4d.geometry_losses import GeometryReconstructionLoss

    return GeometryReconstructionLoss


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_alpha=None,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        stitched_joint_training=False,
        stitched_decoder_checkpoint_path=DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH,
        geometry_supervision_mode="distill",
        geo_supervise_generated_only=True,
        da3_model_path=None,
        da3_enable_scene_flow_head=False,
        da3_scene_flow_head_checkpoint_path=None,
        da3_scene_flow_preset="da3-giant-sceneflow",
        wan_vae_path=None,
        stitching_layer_location="enc_blocks_2",
        stitching_layer_config=None,
        stitched_resolution=532,
        feedforward_resolution=532,
        lora_config="r64,a32,d0.0,f0",
        full_finetune_stitched_parts=False,
        da3_checkpoint_name="model.safetensors",
        da3_config_name="config.json",
        hf_endpoint=None,
        wan_vae_repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        task_type_key="task_type",
        cond_frames_key="cond_frames",
        continuation_cond_choices="9,13,17",
        condition_dropout=0.1,
        dual_task_ratio="6,4",
        mask_condition_loss=False,
        diffusion_loss_weight=1.0,
        rgb_key="rgbs",
        depth_key="depth",
        ray_key="ray",
        intrinsics_key="intrinsics",
        extrinsics_key="extrinsics",
        valid_mask_key="valid_mask",
        scene_flow_key="scene_flow",
        deform_mask_key="deform_mask",
        geo_loss_weight=1.0,
        geo_loss_weight_depth=1.0,
        geo_loss_weight_ray=1.0,
        geo_loss_weight_depth_grad=1.0,
        geo_loss_weight_point_cloud=1.0,
        geo_loss_weight_camera=1.0,
        geo_loss_weight_temporal=1.0,
        geo_loss_weight_scene_flow=1.0,
        distill_loss_weight_depth=1.0,
        distill_loss_weight_depth_grad=1.0,
        distill_loss_weight_ray=1.0,
        distill_loss_weight_depth_conf=0.001,
        distill_loss_weight_ray_conf=0.001,
        distill_loss_weight_camera=1.0,
        distill_loss_weight_scene_flow=1.0,
        distill_loss_weight_feature=0.0,
        geo_loss_warmup_start=0.001,
        geo_loss_warmup_steps=500,
        ref_view_strategy="saddle_balanced",
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. "
                "To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True
        if stitched_joint_training and task not in {"sft", "sft:train"}:
            raise ValueError("`stitched_joint_training` currently only supports `sft` / `sft:train`.")

        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            lora_alpha=lora_alpha,
            preset_lora_path=preset_lora_path,
            preset_lora_model=preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.stitched_joint_training = stitched_joint_training
        self.stitched_decoder_checkpoint_path = stitched_decoder_checkpoint_path
        self.geometry_supervision_mode = geometry_supervision_mode
        self.geo_supervise_generated_only = geo_supervise_generated_only
        self.da3_model_path = da3_model_path
        self.da3_enable_scene_flow_head = bool(da3_enable_scene_flow_head)
        self.da3_scene_flow_head_checkpoint_path = da3_scene_flow_head_checkpoint_path
        self.da3_scene_flow_preset = da3_scene_flow_preset
        self.wan_vae_path = wan_vae_path
        self.stitching_layer_location = stitching_layer_location
        self.stitching_layer_config = (
            parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1")
            if stitching_layer_config is None
            else stitching_layer_config
        )
        self.stitched_resolution = stitched_resolution
        self.feedforward_resolution = feedforward_resolution
        self.lora_config = lora_config
        self.full_finetune_stitched_parts = full_finetune_stitched_parts
        self.da3_checkpoint_name = da3_checkpoint_name
        self.da3_config_name = da3_config_name
        self.hf_endpoint = hf_endpoint
        self.wan_vae_repo_id = wan_vae_repo_id
        self.task_type_key = task_type_key
        self.cond_frames_key = cond_frames_key
        self.continuation_cond_choices = _parse_int_csv(continuation_cond_choices, default_values=(9, 13, 17))
        self.condition_dropout = condition_dropout
        self.dual_task_ratio = _parse_int_csv(dual_task_ratio, default_values=(6, 4))
        self.mask_condition_loss = mask_condition_loss
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.rgb_key = rgb_key
        self.depth_key = depth_key
        self.ray_key = ray_key
        self.intrinsics_key = intrinsics_key
        self.extrinsics_key = extrinsics_key
        self.valid_mask_key = valid_mask_key
        self.scene_flow_key = scene_flow_key
        self.deform_mask_key = deform_mask_key
        self.geo_loss_weight = geo_loss_weight
        self.geo_loss_weight_depth = geo_loss_weight_depth
        self.geo_loss_weight_ray = geo_loss_weight_ray
        self.geo_loss_weight_depth_grad = geo_loss_weight_depth_grad
        self.geo_loss_weight_point_cloud = geo_loss_weight_point_cloud
        self.geo_loss_weight_camera = geo_loss_weight_camera
        self.geo_loss_weight_temporal = geo_loss_weight_temporal
        self.geo_loss_weight_scene_flow = geo_loss_weight_scene_flow
        self.distill_loss_weight_depth = distill_loss_weight_depth
        self.distill_loss_weight_depth_grad = distill_loss_weight_depth_grad
        self.distill_loss_weight_ray = distill_loss_weight_ray
        self.distill_loss_weight_depth_conf = distill_loss_weight_depth_conf
        self.distill_loss_weight_ray_conf = distill_loss_weight_ray_conf
        self.distill_loss_weight_camera = distill_loss_weight_camera
        self.distill_loss_weight_scene_flow = distill_loss_weight_scene_flow
        self.distill_loss_weight_feature = distill_loss_weight_feature
        self.geo_loss_warmup_start = geo_loss_warmup_start
        self.geo_loss_warmup_steps = geo_loss_warmup_steps
        self.ref_view_strategy = ref_view_strategy
        self.stitched_decoder = None
        self.da3_teacher = None
        self.geometry_loss_fn = None
        self.latest_joint_metrics = {}
        self.current_global_step = 0

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

        if self.stitched_joint_training:
            self._setup_stitched_joint_modules()

    def _setup_stitched_joint_modules(self):
        resolved_supervision_mode = str(self.geometry_supervision_mode).lower()
        if resolved_supervision_mode == "none":
            self.stitched_decoder = None
            self.da3_teacher = None
            self.geometry_loss_fn = None
            return

        if self.stitched_decoder_checkpoint_path is None:
            raise ValueError("`stitched_decoder_checkpoint_path` is required when `stitched_joint_training` is enabled.")
        if self.da3_model_path is None:
            raise ValueError("`da3_model_path` is required when `stitched_joint_training` is enabled.")

        from inference_stitched_decode import load_stitched_decoder

        loader_args = argparse.Namespace(
            device=str(self.pipe.device),
            checkpoint_path=self.stitched_decoder_checkpoint_path,
            feedforward_model="da3",
            video_model="wan",
            da3_model_path=self.da3_model_path,
            da3_config_name=self.da3_config_name,
            da3_checkpoint_name=self.da3_checkpoint_name,
            da3_ref_view_strategy=self.ref_view_strategy,
            da3_enable_scene_flow_head=self.da3_enable_scene_flow_head,
            da3_scene_flow_head_checkpoint_path=self.da3_scene_flow_head_checkpoint_path,
            da3_scene_flow_preset=self.da3_scene_flow_preset,
            wan_vae_path=self.wan_vae_path,
            hf_endpoint=self.hf_endpoint,
            wan_vae_repo_id=self.wan_vae_repo_id,
            stitching_layer_location=self.stitching_layer_location,
            stitching_layer_config=self.stitching_layer_config,
            initialization_weight_path=None,
            resolution=self.stitched_resolution,
            feedforward_resolution=self.feedforward_resolution,
            lora_config=self.lora_config,
            full_finetune_stitched_parts=self.full_finetune_stitched_parts,
        )
        configure_hf_runtime(loader_args)
        target_dtype = getattr(self.pipe, "torch_dtype", None)
        self.stitched_decoder = load_stitched_decoder(
            loader_args,
            diffusion_vae_override=self.pipe.vae,
            ignore_diffusion_vae_state=True,
        ).to(self.pipe.device)
        _cast_module_floating_tensors(self.stitched_decoder, target_dtype)
        _sync_stitched_checkpoint_runtime_overrides(self, loader_args)
        self.stitched_decoder.eval()
        self.stitched_decoder.requires_grad_(False)

        def _build_distill_loss():
            return TaskLossDA3(
                lambda_depth=getattr(self, "distill_loss_weight_depth", 1.0),
                lambda_depth_grad=getattr(self, "distill_loss_weight_depth_grad", 0.01),
                lambda_ray=getattr(self, "distill_loss_weight_ray", 1.0),
                lambda_depth_conf=getattr(self, "distill_loss_weight_depth_conf", 0.005),
                lambda_ray_conf=getattr(self, "distill_loss_weight_ray_conf", 0.005),
                lambda_camera=getattr(self, "distill_loss_weight_camera", 1.0),
                lambda_scene_flow=(
                    getattr(self, "distill_loss_weight_scene_flow", 1.0)
                    if self.da3_enable_scene_flow_head
                    else 0.0
                ),
                lambda_feature_distill=getattr(self, "distill_loss_weight_feature", 0.0),
                lambda_temporal=getattr(self, "geo_loss_weight_temporal", 1.0),
            )

        def _build_gt_loss():
            geometry_loss_cls = _load_geometry_reconstruction_loss()
            gt_loss = geometry_loss_cls(
                lambda_depth=getattr(self, "geo_loss_weight_depth", 1.0),
                lambda_depth_grad=getattr(self, "geo_loss_weight_depth_grad", 0.1),
                lambda_ray=getattr(self, "geo_loss_weight_ray", 0.01),
                lambda_point_cloud=getattr(self, "geo_loss_weight_point_cloud", 0.01),
                lambda_camera=getattr(self, "geo_loss_weight_camera", 1.0),
                use_trimmed_loss=False,
                use_temporal_loss=getattr(self, "geo_loss_weight_temporal", 1.0) > 0,
                lambda_temporal=getattr(self, "geo_loss_weight_temporal", 1.0),
            ).to(device=self.pipe.device)
            if self.da3_enable_scene_flow_head:
                gt_loss = GeometrySceneFlowLoss(
                    base_geometry_loss=gt_loss,
                    lambda_scene_flow=getattr(self, "geo_loss_weight_scene_flow", 1.0),
                ).to(device=self.pipe.device)
            return gt_loss

        def _load_raw_da3_teacher():
            teacher_args = argparse.Namespace(**vars(loader_args))
            teacher_args.feedforward_model = "da3"
            teacher = load_feedforward_model(teacher_args, self.pipe.device).to(self.pipe.device)
            _cast_module_floating_tensors(teacher, target_dtype)
            teacher.eval()
            teacher.requires_grad_(False)
            return teacher

        self.da3_teacher = None
        if resolved_supervision_mode == "distill":
            self.da3_teacher = _load_raw_da3_teacher()
            self.geometry_loss_fn = _build_distill_loss()
        elif resolved_supervision_mode == "gt":
            self.geometry_loss_fn = _build_gt_loss()
        elif resolved_supervision_mode == "hybrid":
            self.da3_teacher = _load_raw_da3_teacher()
            self.geometry_loss_fn = {
                "distill": _build_distill_loss(),
                "gt": _build_gt_loss(),
            }
        else:
            raise ValueError(
                f"Unsupported geometry_supervision_mode={self.geometry_supervision_mode!r}. "
                "Expected `distill`, `gt`, or `hybrid`."
            )

        if isinstance(self.geometry_loss_fn, dict):
            for loss_fn in self.geometry_loss_fn.values():
                _cast_module_floating_tensors(loss_fn, target_dtype)
        else:
            _cast_module_floating_tensors(self.geometry_loss_fn, target_dtype)

    def set_training_progress(self, *, global_step: int | None = None, epoch_id: int | None = None, step_in_epoch: int | None = None):
        del epoch_id, step_in_epoch
        if global_step is not None:
            self.current_global_step = int(global_step)

    def _compute_effective_geo_loss_weight(self):
        current_global_step = int(getattr(self, "current_global_step", 0))
        warmup_scale = compute_geo_loss_warmup_scale(
            global_step=current_global_step,
            warmup_steps=self.geo_loss_warmup_steps,
            start=self.geo_loss_warmup_start,
        )
        return self.geo_loss_weight * warmup_scale, warmup_scale

    def _sample_task_type(self):
        weights = self.dual_task_ratio if len(self.dual_task_ratio) >= 2 else [6, 4]
        total = max(1, weights[0] + weights[1])
        probability_ti2v = weights[0] / total
        return "ti2v" if random.random() < probability_ti2v else "continuation"

    def _normalize_task_type(self, data):
        raw_task_type = data.get(self.task_type_key, None)
        if isinstance(raw_task_type, (list, tuple)) and len(raw_task_type) > 0:
            raw_task_type = raw_task_type[0]
        if isinstance(raw_task_type, torch.Tensor) and raw_task_type.numel() == 1:
            raw_task_type = raw_task_type.item()
        if raw_task_type is None:
            return self._sample_task_type()
        task_type = str(raw_task_type).strip().lower()
        if task_type in ("", "nan", "none"):
            return self._sample_task_type()
        if task_type in ("ti2v", "i2v", "image_to_video", "image2video"):
            return "ti2v"
        if task_type in ("continuation", "vc", "video_continuation", "longcat"):
            return "continuation"
        return self._sample_task_type()

    def _resolve_joint_training_stage(self, data, default_stage: str | None = None) -> str:
        if self.geometry_supervision_mode != "hybrid":
            return str(default_stage or self.geometry_supervision_mode).lower()

        raw_stage = data.get("joint_training_stage", default_stage or "distill")
        if isinstance(raw_stage, (list, tuple)) and len(raw_stage) > 0:
            raw_stage = raw_stage[0]
        if isinstance(raw_stage, torch.Tensor) and raw_stage.numel() == 1:
            raw_stage = raw_stage.item()
        resolved_stage = str(raw_stage).strip().lower()
        if resolved_stage not in {"gt", "distill"}:
            raise ValueError(
                f"Unsupported joint_training_stage={raw_stage!r} for hybrid supervision."
            )
        return resolved_stage

    def _extract_video_frames(self, data):
        if "video" in data and isinstance(data["video"], list) and len(data["video"]) > 0:
            return data["video"]

        vae_tensor = _to_tensor_if_possible(data.get("vae_image_tensor", None))
        if vae_tensor is not None:
            if vae_tensor.dim() == 5:
                vae_tensor = vae_tensor[0]
            if vae_tensor.dim() != 4:
                raise ValueError(
                    f"Expected VAE tensor [C,T,H,W] or [T,C,H,W], got {tuple(vae_tensor.shape)}"
                )
            if vae_tensor.shape[0] == 3:
                rgb_thwc = vae_tensor.permute(1, 2, 3, 0)
            elif vae_tensor.shape[1] == 3:
                rgb_thwc = vae_tensor.permute(0, 2, 3, 1)
            elif vae_tensor.shape[-1] == 3:
                rgb_thwc = vae_tensor
            else:
                raise ValueError(f"Unsupported VAE tensor layout: {tuple(vae_tensor.shape)}")

            if rgb_thwc.dtype != torch.uint8:
                rgb_float = rgb_thwc.float()
                if rgb_float.min() >= -1.0 and rgb_float.max() <= 1.0:
                    rgb_float = (rgb_float + 1.0) * 0.5
                elif rgb_float.max() > 1.0:
                    rgb_float = rgb_float / 255.0
                rgb_thwc = (rgb_float.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)

            from PIL import Image

            return [Image.fromarray(frame) for frame in rgb_thwc.cpu().numpy()]

        rgbs = _to_tensor_if_possible(data.get(self.rgb_key, None))
        if rgbs is None:
            raise ValueError(
                f"Cannot build video frames from batch. Available keys: {list(data.keys())}. "
                f"Expected `video` list, `vae_image_tensor`, or `{self.rgb_key}` tensor."
            )

        if rgbs.dim() == 5:
            rgbs = rgbs[0]
        if rgbs.dim() != 4:
            raise ValueError(f"Expected RGB tensor [T,H,W,3] or [T,3,H,W], got {tuple(rgbs.shape)}")

        if rgbs.shape[-1] == 3:
            rgb_thwc = rgbs
        elif rgbs.shape[1] == 3:
            rgb_thwc = rgbs.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Unsupported RGB tensor layout: {tuple(rgbs.shape)}")

        if rgb_thwc.dtype != torch.uint8:
            if rgb_thwc.max() <= 1.0:
                rgb_thwc = (rgb_thwc * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                rgb_thwc = rgb_thwc.clamp(0, 255).to(torch.uint8)

        from PIL import Image

        return [Image.fromarray(frame) for frame in rgb_thwc.cpu().numpy()]

    def _encode_prompt(self, prompt: str):
        self.pipe.load_models_to_device(("text_encoder",))
        ids, mask = self.pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.pipe.device)
        mask = mask.to(self.pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.pipe.text_encoder(ids, mask)
        for index, seq_len in enumerate(seq_lens):
            prompt_emb[index, seq_len:] = 0
        return prompt_emb.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

    def _extract_geo_targets(self, data):
        depth = _to_tensor_if_possible(data.get(self.depth_key, None))
        ray = _to_tensor_if_possible(data.get(self.ray_key, None))
        intrinsics = _to_tensor_if_possible(data.get(self.intrinsics_key, None))
        extrinsics = _to_tensor_if_possible(data.get(self.extrinsics_key, None))
        valid_mask = _to_tensor_if_possible(data.get(self.valid_mask_key, None))
        scene_flow = _to_tensor_if_possible(data.get(self.scene_flow_key, None))
        deform_mask = _to_tensor_if_possible(data.get(self.deform_mask_key, None))

        if depth is None or ray is None:
            return {}

        if depth.dim() == 4:
            depth = depth.unsqueeze(0)
        if ray.dim() == 4:
            ray = ray.unsqueeze(0)
        if intrinsics is not None and intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(0)
        if extrinsics is not None and extrinsics.dim() == 3:
            extrinsics = extrinsics.unsqueeze(0)
        if valid_mask is not None and valid_mask.dim() == 4:
            valid_mask = valid_mask.unsqueeze(0)
        if scene_flow is not None and scene_flow.dim() == 4:
            scene_flow = scene_flow.unsqueeze(0)
        if deform_mask is not None:
            if deform_mask.dim() == 3:
                deform_mask = deform_mask.unsqueeze(0)
            elif deform_mask.dim() == 4 and deform_mask.shape[1] in (1, 3):
                deform_mask = deform_mask.unsqueeze(0)

        depth = _ensure_bthw1(depth).to(device=self.pipe.device, dtype=torch.float32)
        ray = _ensure_bt6hw(ray).to(device=self.pipe.device, dtype=torch.float32)
        intrinsics = None if intrinsics is None else intrinsics.to(device=self.pipe.device, dtype=torch.float32)
        extrinsics = None if extrinsics is None else extrinsics.to(device=self.pipe.device, dtype=torch.float32)
        scene_flow = None if scene_flow is None else scene_flow.to(device=self.pipe.device, dtype=torch.float32)
        deform_mask = None if deform_mask is None else deform_mask.to(device=self.pipe.device, dtype=torch.float32)

        if valid_mask is None:
            valid_mask = (depth > 0).to(dtype=torch.float32)
        else:
            valid_mask = _ensure_bthw1(valid_mask).to(device=self.pipe.device, dtype=torch.float32)

        return {
            "depth_gt": depth,
            "ray_gt": ray,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "valid_mask": valid_mask,
            "scene_flow_gt": scene_flow,
            "deform_mask": deform_mask,
        }

    def _prepare_stitched_joint_inputs(self, data):
        video = self._extract_video_frames(data)
        if len(video) == 0:
            raise ValueError("Empty video frames encountered in batch.")

        self.pipe.load_models_to_device(("vae", "text_encoder"))
        video_tensor = self.pipe.preprocess_video(video).to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        with torch.no_grad():
            input_latents = self.pipe.vae.encode(video_tensor, device=self.pipe.device).to(
                dtype=self.pipe.torch_dtype,
                device=self.pipe.device,
            )

        task_type = self._normalize_task_type(data)
        if task_type == "ti2v":
            cond_video_frames = 1
        else:
            raw_cond_frames = data.get(self.cond_frames_key, None)
            if isinstance(raw_cond_frames, (list, tuple)) and len(raw_cond_frames) > 0:
                raw_cond_frames = raw_cond_frames[0]
            if isinstance(raw_cond_frames, torch.Tensor) and raw_cond_frames.numel() == 1:
                raw_cond_frames = raw_cond_frames.item()
            if raw_cond_frames is None or str(raw_cond_frames).strip().lower() in ("", "nan", "none"):
                cond_video_frames = random.choice(self.continuation_cond_choices)
            else:
                cond_video_frames = int(raw_cond_frames)
                if cond_video_frames % 4 != 1:
                    cond_video_frames = max(1, cond_video_frames - ((cond_video_frames - 1) % 4))

        num_video_frames = len(video)
        cond_video_frames = int(max(1, min(cond_video_frames, num_video_frames)))
        cond_latent_frames = (cond_video_frames - 1) // 4 + 1
        if random.random() < self.condition_dropout:
            cond_video_frames = 0
            cond_latent_frames = 0
        cond_latent_frames = min(cond_latent_frames, input_latents.shape[2])

        # Sync per-rank random decisions: task_type / cond_video_frames /
        # cond_latent_frames steer the DiT cross-attn branch (num_cond_latents>0
        # path in longcat_video_dit.py:309). Divergence across ranks deadlocks
        # ZeRO-3 parameter all-gather.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            pre_task = task_type
            pre_cond_video = cond_video_frames
            pre_cond_latent = cond_latent_frames
            pre_latent_T = int(input_latents.shape[2])
            pre_num_video = num_video_frames
            task_id = 0 if task_type == "ti2v" else 1
            sync_buf = torch.tensor(
                [task_id, cond_video_frames, cond_latent_frames],
                device=self.pipe.device,
                dtype=torch.long,
            )
            torch.distributed.broadcast(sync_buf, src=0)
            task_id, cond_video_frames, cond_latent_frames = (int(v) for v in sync_buf.tolist())
            task_type = "ti2v" if task_id == 0 else "continuation"
            cond_latent_frames = min(cond_latent_frames, input_latents.shape[2])
            # Diagnostic: print per-rank pre/post broadcast state + which
            # optional fields the sample carries. If post values all align
            # but training still hangs, the divergence is in deeper data
            # branches (per-manifest field presence / da3 / stitched).
            _manifest_src = data.get("manifest_source", data.get("source", data.get("dataset", "?")))
            _has_sf_gt = (data.get(self.scene_flow_key, None) is not None) if hasattr(self, "scene_flow_key") else False
            _has_deform = (data.get(self.deform_mask_key, None) is not None) if hasattr(self, "deform_mask_key") else False
            _has_image_hw = data.get("image_hw", None) is not None
            _has_teacher = data.get("teacher_feedforward_image_tensor", None) is not None
            _has_depth = data.get(self.depth_key, None) is not None
            _has_ray = data.get(self.ray_key, None) is not None
            _has_intr = data.get(self.intrinsics_key, None) is not None
            _has_extr = data.get(self.extrinsics_key, None) is not None
            sys.stderr.write(
                f"[zero3-sync rank={rank}] "
                f"pre task={pre_task} cond_v={pre_cond_video} cond_l={pre_cond_latent} "
                f"num_video={pre_num_video} latent_T={pre_latent_T} | "
                f"post task={task_type} cond_v={cond_video_frames} cond_l={cond_latent_frames} | "
                f"src={_manifest_src} sf={_has_sf_gt} deform={_has_deform} img_hw={_has_image_hw} "
                f"teacher={_has_teacher} depth={_has_depth} ray={_has_ray} intr={_has_intr} extr={_has_extr}\n"
            )
            sys.stderr.flush()

        longcat_latents = None if cond_latent_frames == 0 else input_latents[:, :, :cond_latent_frames].detach()

        loss_weight_mask = torch.ones_like(input_latents, dtype=self.pipe.torch_dtype, device=self.pipe.device)
        if self.mask_condition_loss and cond_latent_frames > 0:
            loss_weight_mask[:, :, :cond_latent_frames] = 0
        raw_sequence_length = data.get("sequence_length", None)
        if raw_sequence_length is not None:
            seq_len = int(raw_sequence_length.item()) if torch.is_tensor(raw_sequence_length) else int(raw_sequence_length)
            real_latent_t = (seq_len - 1) // 4 + 1
            if real_latent_t < input_latents.shape[2]:
                loss_weight_mask[:, :, real_latent_t:] = 0

        prompt = str(data.get("prompt", ""))
        joint_training_stage = self._resolve_joint_training_stage(data)
        inputs = {
            "input_latents": input_latents.detach(),
            "context": self._encode_prompt(prompt),
            "cfg_scale": 1,
            "longcat_latents": longcat_latents,
            "loss_weight_mask": loss_weight_mask,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "gt_rgb_video": video_tensor.detach(),
            "cond_video_frames": cond_video_frames,
            "task_type": task_type,
            "joint_training_stage": joint_training_stage,
        }
        teacher_rgb_video = _to_tensor_if_possible(data.get("teacher_feedforward_image_tensor", None))
        if teacher_rgb_video is not None:
            inputs["teacher_rgb_video"] = _ensure_bcthw(teacher_rgb_video).to(
                device=self.pipe.device,
                dtype=self.pipe.torch_dtype,
            )
        raw_image_hw = data.get("image_hw", None)
        if raw_image_hw is not None:
            inputs["image_hw"] = coerce_image_hw(raw_image_hw)

        if joint_training_stage == "gt":
            geo_targets = self._extract_geo_targets(data)
            missing = [
                key
                for key in ("depth_gt", "ray_gt", "intrinsics", "extrinsics")
                if key not in geo_targets or geo_targets[key] is None
            ]
            if missing:
                raise ValueError(
                    "Geometry GT supervision requires keys "
                    f"{missing}. Available keys: {list(data.keys())}"
                )
            inputs.update(geo_targets)
        return inputs

    def _new_zero_geo_metrics(self):
        zero = torch.tensor(0.0, device=self.pipe.device, dtype=torch.float32)
        return {
            "geo_loss": zero,
            "geo_loss_raw": zero,
            "depth_loss": zero,
            "ray_loss": zero,
            "depth_grad_loss": zero,
            "point_cloud_loss": zero,
            "camera_loss": zero,
            "camera_rotation_loss": zero,
            "camera_translation_loss": zero,
            "camera_intrinsics_loss": zero,
            "camera_center_loss": zero,
            "depth_conf_loss": zero,
            "ray_conf_loss": zero,
            "scene_flow_loss": zero,
            "feature_distill_loss": zero,
            "scene_flow_epe": zero,
            "scene_flow_valid_ratio": zero,
            "temporal_loss": zero,
        }

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                video = self._extract_video_frames(data)
                inputs_shared[extra_input] = video[0]
            elif extra_input == "end_image":
                video = self._extract_video_frames(data)
                inputs_shared[extra_input] = video[-1]
            elif extra_input in {"reference_image", "vace_reference_image"}:
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            inputs_shared["num_frames"] = 4 * (len(self._extract_video_frames(data)) - 1) + 1
        return inputs_shared

    def get_pipeline_inputs(self, data):
        video = self._extract_video_frames(data)
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            "input_video": video,
            "height": video[0].size[1],
            "width": video[0].size[0],
            "num_frames": len(video),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward_stitched_joint(self, data):
        joint_inputs = self._prepare_stitched_joint_inputs(data)
        self.pipe.load_models_to_device(tuple(self.pipe.in_iteration_models))
        diffusion_loss, diffusion_details = FlowMatchSFTCFGLoss(
            self.pipe,
            return_details=True,
            **joint_inputs,
        )
        diffusion_loss_weighted = diffusion_loss * self.diffusion_loss_weight
        diffusion_loss_weight_tensor = torch.tensor(
            self.diffusion_loss_weight,
            device=self.pipe.device,
            dtype=torch.float32,
        )
        x_pred = diffusion_details["x_pred"]
        joint_training_stage = self._resolve_joint_training_stage(
            joint_inputs,
            default_stage=joint_inputs.get("joint_training_stage", None),
        )
        if joint_training_stage == "none":
            zero_metrics = self._new_zero_geo_metrics()
            zero = torch.tensor(0.0, device=self.pipe.device, dtype=torch.float32)
            self.latest_joint_metrics = {
                "diffusion_loss": diffusion_loss.detach(),
                "diffusion_loss_weight": diffusion_loss_weight_tensor,
                "diffusion_loss_weighted": diffusion_loss_weighted.detach(),
                "geo_loss_weight_effective": zero,
                "geo_loss_warmup_scale": zero,
                **zero_metrics,
            }
            return diffusion_loss_weighted

        student_image_hw = _resolve_joint_student_image_hw(
            joint_inputs,
            joint_training_stage=joint_training_stage,
        )

        student_feedforward_image = None
        if _uses_shape_only_da3_student(self.stitched_decoder):
            student_feedforward_image = None
        else:
            self.pipe.load_models_to_device(("vae",))
            predicted_rgb = self.pipe.vae.decode(x_pred, device=self.pipe.device)
            student_feedforward_image = _resize_video_bcthw_to_hw(
                predicted_rgb,
                student_image_hw,
            )
        if joint_training_stage == "distill":
            distill_loss_fn = (
                self.geometry_loss_fn["distill"]
                if isinstance(self.geometry_loss_fn, dict)
                else self.geometry_loss_fn
            )
            geo_loss_dict = compute_stitched_distill_loss(
                x_pred=x_pred,
                gt_rgb_video=_resolve_joint_teacher_rgb_video(joint_inputs),
                student_feedforward_image=student_feedforward_image,
                student_image_hw=student_image_hw,
                stitched_decoder=self.stitched_decoder,
                da3_teacher=self.da3_teacher,
                loss_criterion=distill_loss_fn,
                ref_view_strategy=self.ref_view_strategy,
                train=self.use_gradient_checkpointing,
                gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )
        elif joint_training_stage == "gt":
            gt_loss_fn = (
                self.geometry_loss_fn["gt"]
                if isinstance(self.geometry_loss_fn, dict)
                else self.geometry_loss_fn
            )
            geo_loss_dict = compute_stitched_gt_loss(
                x_pred=x_pred,
                student_feedforward_image=student_feedforward_image,
                student_image_hw=student_image_hw,
                stitched_decoder=self.stitched_decoder,
                geometry_loss_fn=gt_loss_fn,
                depth_gt=joint_inputs["depth_gt"],
                ray_gt=joint_inputs["ray_gt"],
                intrinsics=joint_inputs["intrinsics"],
                extrinsics=joint_inputs["extrinsics"],
                valid_mask=joint_inputs["valid_mask"],
                scene_flow_gt=joint_inputs.get("scene_flow_gt"),
                deform_mask=joint_inputs.get("deform_mask"),
                geo_supervise_generated_only=self.geo_supervise_generated_only,
                cond_video_frames=int(joint_inputs.get("cond_video_frames", 0)),
                train=self.use_gradient_checkpointing,
                gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )
        else:
            raise ValueError(
                f"Unsupported joint training stage {joint_training_stage!r} in stitched joint forward."
            )

        geo_loss_raw = geo_loss_dict["total_loss"]
        effective_geo_loss_weight, geo_loss_warmup_scale = self._compute_effective_geo_loss_weight()
        geo_loss = effective_geo_loss_weight * geo_loss_raw
        total_loss = diffusion_loss_weighted + geo_loss
        geo_loss_weight_effective_tensor = torch.tensor(
            effective_geo_loss_weight,
            device=self.pipe.device,
            dtype=torch.float32,
        )
        geo_loss_warmup_scale_tensor = torch.tensor(
            geo_loss_warmup_scale,
            device=self.pipe.device,
            dtype=torch.float32,
        )
        self.latest_joint_metrics = {
            "diffusion_loss": diffusion_loss.detach(),
            "diffusion_loss_weight": diffusion_loss_weight_tensor,
            "diffusion_loss_weighted": diffusion_loss_weighted.detach(),
            "geo_loss": geo_loss.detach(),
            "geo_loss_raw": geo_loss_raw.detach(),
            "geo_loss_weight_effective": geo_loss_weight_effective_tensor,
            "geo_loss_warmup_scale": geo_loss_warmup_scale_tensor,
            "depth_loss": geo_loss_dict.get("depth_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "ray_loss": geo_loss_dict.get("ray_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "depth_grad_loss": geo_loss_dict.get("depth_grad_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "point_cloud_loss": geo_loss_dict.get("point_cloud_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "camera_loss": geo_loss_dict.get("camera_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "camera_rotation_loss": geo_loss_dict.get("camera_rotation_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "camera_translation_loss": geo_loss_dict.get("camera_translation_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "camera_intrinsics_loss": geo_loss_dict.get("camera_intrinsics_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "camera_center_loss": geo_loss_dict.get("camera_center_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "depth_conf_loss": geo_loss_dict.get("depth_conf_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "ray_conf_loss": geo_loss_dict.get("ray_conf_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "scene_flow_loss": geo_loss_dict.get("scene_flow_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "feature_distill_loss": geo_loss_dict.get("feature_distill_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "scene_flow_epe": geo_loss_dict.get("scene_flow_epe", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "scene_flow_valid_ratio": geo_loss_dict.get("scene_flow_valid_ratio", torch.tensor(0.0, device=self.pipe.device)).detach(),
            "temporal_loss": geo_loss_dict.get("temporal_loss", torch.tensor(0.0, device=self.pipe.device)).detach(),
        }
        return total_loss

    def forward(self, data, inputs=None):
        if self.stitched_joint_training:
            return self.forward_stitched_joint(data)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.set_defaults(height=532, width=532)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.",
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Whether to initialize models on CPU.",
    )
    parser.add_argument(
        "--framewise_decoding",
        default=False,
        action="store_true",
        help="Enable it if this model is a WanToDance global model.",
    )
    parser.add_argument(
        "--stitched_joint_training",
        default=False,
        action="store_true",
        help="Enable LongCat + stitched DA3 joint training with video v-loss and geometry x-loss.",
    )
    parser.add_argument(
        "--stitched_decoder_checkpoint_path",
        type=str,
        default=DEFAULT_STITCHED_DECODER_CHECKPOINT_PATH,
        help="Checkpoint path for the trained stitched decoder used during joint training. Supports both full-finetune and LoRA stitched checkpoints.",
    )
    parser.add_argument(
        "--geometry_supervision_mode",
        type=str,
        default="distill",
        choices=["distill", "gt", "hybrid", "none"],
        help="Geometry supervision source. distill=DA3 teacher, gt=dataset depth/ray supervision, hybrid=mixed gt+distill manifest batches, none=RGB-only SFT without stitched/DA3 geometry modules.",
    )
    parser.add_argument(
        "--train_manifest",
        action="append",
        type=parse_manifest_spec,
        default=None,
        help="Optional NAME:PATH manifest spec for stitched joint manifest training. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest_data_root",
        type=str,
        default=None,
        help="Common data root used to resolve relative scene_relpath values in --train_manifest.",
    )
    parser.add_argument(
        "--action100m_manifest",
        type=parse_existing_file,
        default=None,
        help="Optional Action100M distill-only manifest appended to the distill branch in manifest training.",
    )
    parser.add_argument(
        "--manifest_process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "square_center_crop"],
        help="Manifest preprocessing semantics for stitched joint training.",
    )
    parser.add_argument(
        "--geo_supervise_generated_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, geometry x-loss ignores the conditioned prefix frames.",
    )
    parser.add_argument("--da3_model_path", type=str, default=None, help="Path to the full DA3 teacher model directory.")
    parser.add_argument("--da3_checkpoint_name", type=str, default="model.safetensors", help="DA3 checkpoint filename.")
    parser.add_argument("--da3_config_name", type=str, default="config.json", help="DA3 config filename.")
    parser.add_argument(
        "--da3_enable_scene_flow_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach the DA3 scene-flow head to the raw teacher and stitched DA3 student.",
    )
    parser.add_argument(
        "--da3_scene_flow_head_checkpoint_path",
        type=str,
        default=None,
        help="Checkpoint path for the DA3 scene-flow head when --da3_enable_scene_flow_head is enabled.",
    )
    parser.add_argument(
        "--da3_scene_flow_preset",
        type=str,
        default="da3-giant-sceneflow",
        help="DA3 registry preset used to build the scene-flow head and adapter.",
    )
    parser.add_argument("--wan_vae_path", type=str, default=None, help="Optional local Wan VAE directory for stitched decoder loading.")
    parser.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_2",
        help="Location of the stitched latent injection layer inside DA3.",
    )
    parser.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default=parse_conv_spec("conv3d_k5x3x3_o1536_s1x2x2_p2x1x1"),
        metavar="CONV_SPEC",
        help="Conv spec for the stitched latent projection layer.",
    )
    parser.add_argument("--stitched_resolution", type=int, default=532, help="Wan VAE branch resolution used by the stitched decoder.")
    parser.add_argument("--feedforward_resolution", type=int, default=532, help="DA3 branch resolution used by the stitched decoder.")
    parser.add_argument("--lora_config", type=str, default="r64,a32,d0.0,f0", help="LoRA config used by the stitched checkpoint.")
    parser.add_argument(
        "--full_finetune_stitched_parts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether the stitched checkpoint stores full fine-tuned stitched parts instead of LoRA deltas.",
    )
    parser.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), help="HF endpoint for stitched decoder loading.")
    parser.add_argument("--wan_vae_repo_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="Fallback Wan VAE repo id for loader compatibility.")
    parser.add_argument("--task_type_key", type=str, default="task_type", help="Metadata key for task type (ti2v/continuation).")
    parser.add_argument("--cond_frames_key", type=str, default="cond_frames", help="Metadata key for continuation condition frames.")
    parser.add_argument("--continuation_cond_choices", type=str, default="9,13,17", help="Fallback continuation condition frames.")
    parser.add_argument("--condition_dropout", type=float, default=0.1, help="Probability of dropping LongCat condition latents.")
    parser.add_argument("--dual_task_ratio", type=str, default="6,4", help="Sampling ratio for ti2v vs continuation.")
    parser.add_argument("--mask_condition_loss", default=False, action="store_true", help="Mask v-loss on conditioned latent prefix.")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0, help="Global weight applied to RGB diffusion/v-pred loss in stitched joint training.")
    parser.add_argument("--rgb_key", type=str, default="rgbs", help="Optional RGB tensor key when `video` frames are absent.")
    parser.add_argument("--depth_key", type=str, default="depth", help="Depth GT tensor key for geometry supervision.")
    parser.add_argument("--ray_key", type=str, default="ray", help="Ray GT tensor key for geometry supervision.")
    parser.add_argument("--intrinsics_key", type=str, default="intrinsics", help="Intrinsics tensor key for geometry supervision.")
    parser.add_argument("--extrinsics_key", type=str, default="extrinsics", help="Extrinsics tensor key for geometry supervision.")
    parser.add_argument("--valid_mask_key", type=str, default="valid_mask", help="Valid-mask tensor key for geometry supervision.")
    parser.add_argument("--scene_flow_key", type=str, default="scene_flow", help="Scene-flow GT tensor key for geometry supervision.")
    parser.add_argument("--deform_mask_key", type=str, default="deform_mask", help="Scene-flow deform-mask tensor key for geometry supervision.")
    parser.add_argument("--geo_loss_weight", type=float, default=1.0, help="Global weight applied to geometry x-loss.")
    parser.add_argument("--geo_loss_weight_depth", type=float, default=1.0, help="Depth GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_ray", type=float, default=1.0, help="Ray GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_depth_grad", type=float, default=1.0, help="Depth-gradient GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_point_cloud", type=float, default=1.0, help="Point-cloud GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_camera", type=float, default=1.0, help="Decoded-camera GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_temporal", type=float, default=1.0, help="Temporal depth GT weight in GT supervision mode.")
    parser.add_argument("--geo_loss_weight_scene_flow", type=float, default=1.0, help="Scene-flow GT weight when the DA3 scene-flow head is enabled.")
    parser.add_argument("--distill_loss_weight_depth", type=float, default=1.0, help="Depth distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_depth_grad", type=float, default=1.0, help="Depth-gradient distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_ray", type=float, default=1.0, help="Ray distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_depth_conf", type=float, default=0.001, help="Depth-confidence distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_ray_conf", type=float, default=0.001, help="Ray-confidence distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_camera", type=float, default=1.0, help="Decoded-camera distill weight in distill supervision mode.")
    parser.add_argument("--distill_loss_weight_scene_flow", type=float, default=1.0, help="Scene-flow distill weight when the DA3 scene-flow head is enabled.")
    parser.add_argument("--distill_loss_weight_feature", type=float, default=0.0, help="Feature-token distill weight in distill supervision mode.")
    parser.add_argument("--geo_loss_warmup_start", type=float, default=0.001, help="Initial geometry-loss warmup scale before ramping to 1.0.")
    parser.add_argument("--geo_loss_warmup_steps", type=int, default=500, help="Linear warmup steps for the geometry-loss scale. <=0 disables warmup.")
    parser.add_argument(
        "--ref_view_strategy",
        type=str,
        default="middle",
        choices=["saddle_balanced", "first", "middle", "last"],
        help="Reference-view strategy used by the frozen DA3 teacher / stitched decoder.",
    )
    parser.add_argument("--geometry_tensor_keys", type=str, default="rgbs,depth,ray,intrinsics,extrinsics", help="Comma-separated geometry keys used in stitched joint training.")
    parser.add_argument("--online_point_odyssey_geometry", default=False, action="store_true", help="Load PointOdyssey geometry online from scene_dir/start_idx/end_idx instead of preprocessed .pth files.")
    parser.add_argument("--point_odyssey_scene_dir_key", type=str, default="scene_dir", help="Metadata key for PointOdyssey scene directory.")
    parser.add_argument("--point_odyssey_start_idx_key", type=str, default="start_idx", help="Metadata key for PointOdyssey clip start frame index.")
    parser.add_argument("--point_odyssey_end_idx_key", type=str, default="end_idx", help="Metadata key for PointOdyssey clip end frame index.")
    parser.add_argument("--point_odyssey_target_height", type=int, default=None, help="Optional resize+crop target height for online PointOdyssey loading. Defaults to --height.")
    parser.add_argument("--point_odyssey_target_width", type=int, default=None, help="Optional resize+crop target width for online PointOdyssey loading. Defaults to --width.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    if args.train_manifest:
        if not args.stitched_joint_training:
            raise ValueError("--train_manifest is only supported together with --stitched_joint_training.")
        if args.manifest_data_root is None:
            args.manifest_data_root = args.dataset_base_path
        dataset = build_stitched_joint_training_dataset(args)
    else:
        special_operator_map = {
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        }
        if args.stitched_joint_training:
            geometry_tensor_keys = [key.strip() for key in args.geometry_tensor_keys.split(",") if key.strip()]
            data_file_key_set = {key.strip() for key in args.data_file_keys.split(",") if key.strip()}
            if args.online_point_odyssey_geometry:
                point_odyssey_target_height = args.point_odyssey_target_height if args.point_odyssey_target_height is not None else args.height
                point_odyssey_target_width = args.point_odyssey_target_width if args.point_odyssey_target_width is not None else args.width
                point_odyssey_loader = LoadPointOdysseyClip(
                    dataset_base_path=args.dataset_base_path,
                    scene_dir_key=args.point_odyssey_scene_dir_key,
                    start_idx_key=args.point_odyssey_start_idx_key,
                    end_idx_key=args.point_odyssey_end_idx_key,
                    target_height=point_odyssey_target_height,
                    target_width=point_odyssey_target_width,
                )
                for key in geometry_tensor_keys:
                    if key in data_file_key_set:
                        special_operator_map[key] = point_odyssey_loader
            elif args.geometry_supervision_mode == "gt":
                for key in geometry_tensor_keys:
                    if key in data_file_key_set and key not in special_operator_map:
                        special_operator_map[key] = ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle()

        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4 if not args.framewise_decoding else 1,
                time_division_remainder=1 if not args.framewise_decoding else 0,
            ),
            special_operator_map=special_operator_map,
        )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        stitched_joint_training=args.stitched_joint_training,
        stitched_decoder_checkpoint_path=args.stitched_decoder_checkpoint_path,
        geometry_supervision_mode=args.geometry_supervision_mode,
        geo_supervise_generated_only=args.geo_supervise_generated_only,
        da3_model_path=args.da3_model_path,
        da3_enable_scene_flow_head=args.da3_enable_scene_flow_head,
        da3_scene_flow_head_checkpoint_path=args.da3_scene_flow_head_checkpoint_path,
        da3_scene_flow_preset=args.da3_scene_flow_preset,
        wan_vae_path=args.wan_vae_path,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=args.stitching_layer_config,
        stitched_resolution=args.stitched_resolution,
        feedforward_resolution=args.feedforward_resolution,
        lora_config=args.lora_config,
        full_finetune_stitched_parts=args.full_finetune_stitched_parts,
        da3_checkpoint_name=args.da3_checkpoint_name,
        da3_config_name=args.da3_config_name,
        hf_endpoint=args.hf_endpoint,
        wan_vae_repo_id=args.wan_vae_repo_id,
        task_type_key=args.task_type_key,
        cond_frames_key=args.cond_frames_key,
        continuation_cond_choices=args.continuation_cond_choices,
        condition_dropout=args.condition_dropout,
        dual_task_ratio=args.dual_task_ratio,
        mask_condition_loss=args.mask_condition_loss,
        diffusion_loss_weight=args.diffusion_loss_weight,
        rgb_key=args.rgb_key,
        depth_key=args.depth_key,
        ray_key=args.ray_key,
        intrinsics_key=args.intrinsics_key,
        extrinsics_key=args.extrinsics_key,
        valid_mask_key=args.valid_mask_key,
        scene_flow_key=args.scene_flow_key,
        deform_mask_key=args.deform_mask_key,
        geo_loss_weight=args.geo_loss_weight,
        geo_loss_weight_depth=args.geo_loss_weight_depth,
        geo_loss_weight_ray=args.geo_loss_weight_ray,
        geo_loss_weight_depth_grad=args.geo_loss_weight_depth_grad,
        geo_loss_weight_point_cloud=args.geo_loss_weight_point_cloud,
        geo_loss_weight_camera=args.geo_loss_weight_camera,
        geo_loss_weight_temporal=args.geo_loss_weight_temporal,
        geo_loss_weight_scene_flow=args.geo_loss_weight_scene_flow,
        distill_loss_weight_depth=args.distill_loss_weight_depth,
        distill_loss_weight_depth_grad=args.distill_loss_weight_depth_grad,
        distill_loss_weight_ray=args.distill_loss_weight_ray,
        distill_loss_weight_depth_conf=args.distill_loss_weight_depth_conf,
        distill_loss_weight_ray_conf=args.distill_loss_weight_ray_conf,
        distill_loss_weight_camera=args.distill_loss_weight_camera,
        distill_loss_weight_scene_flow=args.distill_loss_weight_scene_flow,
        distill_loss_weight_feature=args.distill_loss_weight_feature,
        geo_loss_warmup_start=args.geo_loss_warmup_start,
        geo_loss_warmup_steps=args.geo_loss_warmup_steps,
        ref_view_strategy=args.ref_view_strategy,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
