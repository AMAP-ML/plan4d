from contextlib import nullcontext
import math
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator
from accelerate.data_loader import skip_first_batches as accelerate_skip_first_batches
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    set_seed as accelerate_set_seed,
)
from einops import rearrange
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import utils.dist_util as dist_util
from data.dataset_util import create_hybrid_stitching_dataloaders, create_stitching_dataloader
from models.da3_stitched import (
    DA3Stitched,
    TaskLossDA3,
    canonicalize_da3_geometry_outputs,
    extract_da3_teacher_outputs,
)
from models.stitched_model import StitchVAE3D
from utils.argument import apply_feedforward_model_defaults, stitching_training_argument
from utils.dist_util import torch_device_setup
from utils.da3_resize import coerce_image_hw
from utils.lora_util.utils import parse_lora_mode
from utils.geo_loss_schedule import compute_geo_loss_warmup_scale
from utils.scene_flow_supervision import GeometrySceneFlowLoss
from utils.stitched_parts import (
    checkpoint_uses_full_finetune_stitched_parts,
    configure_stitched_parts_training,
    export_stitched_parts_state,
    resolve_full_finetune_stitched_parts,
)
from utils.utils_for_thirdparty import (
    cast_to_bfloat16,
    configure_hf_runtime,
    load_feedforward_model,
    load_vae,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT


def ensure_plan4d_training_imports() -> None:
    if str(PLAN4D_ROOT) not in sys.path:
        sys.path.insert(0, str(PLAN4D_ROOT))


def should_use_accelerate(
    args,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    env = os.environ if env is None else env
    if getattr(args, "use_accelerate", False):
        return True

    try:
        world_size = int(env.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    try:
        local_rank = int(env.get("LOCAL_RANK", "-1"))
    except ValueError:
        local_rank = -1
    return world_size > 1 or local_rank >= 0


def build_accelerator_kwargs(
    args,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    env = os.environ if env is None else env
    accelerator_kwargs: Dict[str, object] = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision if args.mixed_precision != "no" else None,
    }
    use_deepspeed = env.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
    use_fsdp = env.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
    if not use_deepspeed and not use_fsdp:
        accelerator_kwargs["kwargs_handlers"] = [
            DistributedDataParallelKwargs(
                find_unused_parameters=False,
                static_graph=True,
            )
        ]
    return accelerator_kwargs


def reduce_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, float]:
    reduced: Dict[str, float] = {}
    for key, value in loss_dict.items():
        if not torch.is_tensor(value) or value.dim() != 0:
            continue
        scalar = value.detach().float()
        if accelerator is not None:
            scalar = accelerator.reduce(scalar, reduction="mean")
        reduced[key] = float(scalar.item())
    return reduced


def build_step_log_dict(
    loss_dict: Dict[str, torch.Tensor],
    *,
    lr: float,
    grad_norm=None,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, float]:
    log_dict = reduce_loss_dict(loss_dict, accelerator)
    log_dict["lr"] = float(lr)
    if grad_norm is not None:
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.detach().float().item())
        else:
            grad_norm = float(grad_norm)
        log_dict["grad_norm"] = grad_norm
    return log_dict


STEP_LOG_EXCLUDED_METRICS = frozenset(
    {
        "scene_flow_epe",
        "scene_flow_valid_ratio",
    }
)


def filter_step_log_dict(log_dict: Dict[str, float]) -> Dict[str, float]:
    return {
        key: value
        for key, value in log_dict.items()
        if key not in STEP_LOG_EXCLUDED_METRICS
    }


def load_geometry_reconstruction_loss():
    ensure_plan4d_training_imports()
    from plan4d.geometry_losses import GeometryReconstructionLoss

    return GeometryReconstructionLoss


def build_base_da3_geometry_loss(args):
    GeometryReconstructionLoss = load_geometry_reconstruction_loss()
    return GeometryReconstructionLoss(
        lambda_depth=args.geo_loss_weight_depth,
        lambda_depth_grad=args.geo_loss_weight_depth_grad,
        lambda_ray=args.geo_loss_weight_ray,
        lambda_point_cloud=args.geo_loss_weight_point_cloud,
        lambda_camera=args.geo_loss_weight_camera,
        use_trimmed_loss=False,
        use_temporal_loss=args.geo_loss_weight_temporal > 0,
        lambda_temporal=args.geo_loss_weight_temporal,
    )


def build_da3_loss_criterion(args):
    enable_scene_flow_head = bool(getattr(args, "da3_enable_scene_flow_head", False))

    if args.training_stage in ("distill", "hybrid"):
        distill_loss = TaskLossDA3(
            lambda_depth=args.distill_loss_weight_depth,
            lambda_depth_grad=args.distill_loss_weight_depth_grad,
            lambda_ray=args.distill_loss_weight_ray,
            lambda_depth_conf=args.distill_loss_weight_depth_conf,
            lambda_ray_conf=args.distill_loss_weight_ray_conf,
            lambda_camera=args.distill_loss_weight_camera,
            lambda_feature_distill=getattr(args, "distill_loss_weight_feature", 0.0),
            lambda_scene_flow=(
                args.distill_loss_weight_scene_flow if enable_scene_flow_head else 0.0
            ),
            lambda_temporal=args.geo_loss_weight_temporal,
        )
        if args.training_stage == "distill":
            return distill_loss

    gt_loss = build_base_da3_geometry_loss(args)
    if enable_scene_flow_head:
        gt_loss = GeometrySceneFlowLoss(
            base_geometry_loss=gt_loss,
            lambda_scene_flow=args.geo_loss_weight_scene_flow,
        )

    if args.training_stage == "hybrid":
        return {
            "distill": distill_loss,
            "gt": gt_loss,
        }
    return gt_loss


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_student_train_scope(args) -> str:
    student_train_scope = getattr(args, "student_train_scope", "default")
    if student_train_scope == "default":
        return "full_model" if getattr(args, "full_finetune_stitched_parts", False) else "lora"
    return student_train_scope


def resolve_student_da3_load_options(args) -> dict[str, bool]:
    if getattr(args, "feedforward_model", None) != "da3":
        return {
            "load_pretrained": True,
            "load_scene_flow_head": bool(getattr(args, "da3_enable_scene_flow_head", False)),
        }

    if getattr(args, "student_da3_init_mode", "pretrained") == "random":
        return {
            "load_pretrained": False,
            "load_scene_flow_head": False,
        }

    return {
        "load_pretrained": True,
        "load_scene_flow_head": bool(getattr(args, "da3_enable_scene_flow_head", False)),
    }


def configure_student_model_training(
    stitched_model,
    *,
    lora_cfg,
    student_train_scope: str,
) -> None:
    for param in stitched_model.stitching_layer.parameters():
        param.requires_grad = True

    if student_train_scope == "full_model":
        for param in stitched_model.stitched_3d_model.parameters():
            param.requires_grad = True
        return

    if student_train_scope == "transfer_only":
        for param in stitched_model.stitched_3d_model.parameters():
            param.requires_grad = False
        return

    if student_train_scope == "lora":
        configure_stitched_parts_training(
            stitched_model.stitched_3d_model,
            lora_cfg,
            full_finetune_stitched_parts=False,
        )
        return

    raise ValueError(f"Unsupported student_train_scope: {student_train_scope}")


def apply_training_precision_policy(
    stitched_model,
    feed_forward_model,
    *,
    use_accelerate: bool,
) -> None:
    """
    FSDP requires uniform parameter dtypes before flattening.
    Keep the training graph in fp32 when Accelerate/FSDP is active and rely on
    autocast for bf16 execution. For non-FSDP paths we preserve the existing
    selective manual cast for memory/perf parity.
    """
    if use_accelerate:
        return

    cast_to_bfloat16(stitched_model)
    if feed_forward_model is not None:
        cast_to_bfloat16(feed_forward_model)


def get_module_floating_dtype(module) -> Optional[torch.dtype]:
    if module is None:
        return None
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor.dtype
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor.dtype
    return None


def get_accelerate_training_dtype(
    accelerator: Optional[Accelerator],
) -> Optional[torch.dtype]:
    if accelerator is None:
        return None

    mixed_precision = getattr(accelerator, "mixed_precision", None)
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def cast_frozen_teacher_to_training_dtype(
    feed_forward_model,
    *,
    accelerator: Optional[Accelerator] = None,
) -> None:
    if feed_forward_model is None or not isinstance(feed_forward_model, torch.nn.Module):
        return

    target_dtype = get_accelerate_training_dtype(accelerator)
    if target_dtype is None:
        return

    feed_forward_model.to(dtype=target_dtype)


def get_stitched_model_student_dtype(
    stitched_model,
    *,
    accelerator: Optional[Accelerator] = None,
) -> Optional[torch.dtype]:
    student_dtype = get_accelerate_training_dtype(accelerator)
    if student_dtype is not None:
        return student_dtype
    return get_module_floating_dtype(
        unwrap_stitched_model(stitched_model, accelerator=accelerator)
    )


def get_stitched_runtime_diffusion_vae(
    stitched_model,
    *,
    accelerator: Optional[Accelerator] = None,
):
    unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
    getter = getattr(unwrapped, "get_runtime_diffusion_vae", None)
    if callable(getter):
        return getter()
    return getattr(unwrapped, "diffusion_vae", None)


def should_encode_runtime_vae_outside_engine(
    stitched_model,
    *,
    accelerator: Optional[Accelerator] = None,
) -> bool:
    unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
    runtime_vae = get_stitched_runtime_diffusion_vae(
        stitched_model,
        accelerator=accelerator,
    )
    return runtime_vae is not None and getattr(unwrapped, "diffusion_vae", None) is None


def encode_stitching_latents(
    stitched_model,
    images: torch.Tensor,
    *,
    accelerator: Optional[Accelerator] = None,
) -> torch.Tensor:
    unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
    with torch.no_grad():
        latents, _ = unwrapped.vae_encoder_forward(images, decode=False)
    return latents


def move_tensor_to_device_dtype(
    tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dtype is not None and tensor.is_floating_point():
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device)


def get_training_autocast_context(accelerator: Optional[Accelerator]):
    if accelerator is not None:
        return accelerator.autocast()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def prepare_stitching_batch_inputs(
    data: dict[str, torch.Tensor],
    *,
    args,
    stitched_model,
    feed_forward_model,
    device: torch.device,
    accelerator: Optional[Accelerator] = None,
) -> dict[str, torch.Tensor | int | None]:
    student_feedforward = data.get("feedforward_image_tensor")
    teacher_feedforward = data.get("teacher_feedforward_image_tensor", student_feedforward)
    used_view_source = student_feedforward
    if used_view_source is None:
        used_view_source = teacher_feedforward
    if used_view_source is None:
        used_view_source = data["vae_image_tensor"]
    used_view_count = select_used_view_count(
        args,
        used_view_source.shape[2],
    )

    vae_images = data["vae_image_tensor"][:, :, :used_view_count]
    feedforward_img_student = (
        student_feedforward[:, :, :used_view_count]
        if student_feedforward is not None
        else None
    )
    feedforward_img_teacher = (
        teacher_feedforward[:, :, :used_view_count]
        if teacher_feedforward is not None
        else None
    )

    student_dtype = get_stitched_model_student_dtype(
        stitched_model,
        accelerator=accelerator,
    )
    vae_dtype = get_module_floating_dtype(
        get_stitched_runtime_diffusion_vae(
            stitched_model,
            accelerator=accelerator,
        )
    )
    if vae_dtype is None:
        vae_dtype = student_dtype
    teacher_dtype = get_module_floating_dtype(feed_forward_model)
    image_hw = coerce_image_hw(data.get("image_hw"))

    return {
        "used_view_count": used_view_count,
        "vae_images": move_tensor_to_device_dtype(
            vae_images,
            device,
            dtype=vae_dtype,
        ),
        "feedforward_img_student": move_tensor_to_device_dtype(
            feedforward_img_student,
            device,
            dtype=student_dtype,
        ) if feedforward_img_student is not None else None,
        "feedforward_img_teacher": move_tensor_to_device_dtype(
            feedforward_img_teacher,
            device,
            dtype=teacher_dtype,
        ) if feed_forward_model is not None and feedforward_img_teacher is not None else None,
        "image_hw": image_hw,
    }


def build_hybrid_step_schedule(
    *,
    num_gt_steps: int,
    num_distill_steps: int,
    epoch: int,
    global_seed: int,
) -> list[str]:
    schedule = ["gt"] * int(num_gt_steps) + ["distill"] * int(num_distill_steps)
    schedule_rng = random.Random(int(global_seed) + int(epoch) * 100_003)
    schedule_rng.shuffle(schedule)
    return schedule


def compute_hybrid_resume_batch_skips(
    *,
    schedule: list[str],
    step_in_epoch_start: int,
) -> Dict[str, int]:
    if step_in_epoch_start < 0:
        raise ValueError("step_in_epoch_start must be non-negative.")
    if step_in_epoch_start > len(schedule):
        raise ValueError("step_in_epoch_start cannot exceed schedule length.")

    schedule_prefix = schedule[:step_in_epoch_start]
    return {
        "gt": schedule_prefix.count("gt"),
        "distill": schedule_prefix.count("distill"),
    }


def maybe_skip_first_batches_for_resume(
    dataloader,
    *,
    skip_batches: int,
):
    if skip_batches < 0:
        raise ValueError("skip_batches must be non-negative.")
    if skip_batches == 0:
        return dataloader
    if skip_batches > len(dataloader):
        raise ValueError("skip_batches cannot exceed dataloader length.")
    return accelerate_skip_first_batches(dataloader, num_batches=skip_batches)


def _resolve_loss_criterion_for_stage(loss_criterion, stage: str):
    if isinstance(loss_criterion, Mapping):
        return loss_criterion[stage]
    return loss_criterion


def _compute_stitching_loss_for_batch(
    data,
    *,
    stage: str,
    stitched_model,
    feed_forward_model,
    loss_criterion,
    args,
    training_global_step: int,
    accelerator: Optional[Accelerator] = None,
):
    device = accelerator.device if accelerator is not None else stitched_model.device
    student_dtype = get_stitched_model_student_dtype(
        stitched_model,
        accelerator=accelerator,
    )
    use_external_vae_encoding = should_encode_runtime_vae_outside_engine(
        stitched_model,
        accelerator=accelerator,
    )
    batch_inputs = prepare_stitching_batch_inputs(
        data,
        args=args,
        stitched_model=stitched_model,
        feed_forward_model=feed_forward_model,
        device=device,
        accelerator=accelerator,
    )
    used_view_for_this_iter = batch_inputs["used_view_count"]
    vae_images = batch_inputs["vae_images"]
    feedforward_img = batch_inputs["feedforward_img_student"]
    teacher_feedforward_img = batch_inputs["feedforward_img_teacher"]

    student_latent = None
    if use_external_vae_encoding:
        student_latent = encode_stitching_latents(
            stitched_model,
            vae_images,
            accelerator=accelerator,
        )
        student_latent = move_tensor_to_device_dtype(
            student_latent,
            device,
            dtype=student_dtype,
        )

    ff_out = None
    autocast_context = get_training_autocast_context(accelerator)
    criterion = _resolve_loss_criterion_for_stage(loss_criterion, stage)
    include_distill_feature_tokens = (
        stage == "distill" and getattr(args, "distill_loss_weight_feature", 0.0) > 0
    )
    if stage == "distill":
        with torch.no_grad():
            ff_out = extract_da3_teacher_outputs(
                feed_forward_model,
                teacher_feedforward_img,
                ref_view_strategy=args.da3_ref_view_strategy,
                include_distill_feature_tokens=include_distill_feature_tokens,
            )

    with autocast_context:
        if student_latent is None:
            stitched_out = stitched_model(
                images=vae_images,
                feedforward_image=feedforward_img,
                train=True,
                image_hw=batch_inputs["image_hw"],
            )
        else:
            stitched_out = stitched_model(
                latent=student_latent,
                feedforward_image=feedforward_img,
                train=True,
                image_hw=batch_inputs["image_hw"],
            )
        if stage == "distill":
            loss = criterion(stitched_out, ff_out)
        else:
            loss_inputs = prepare_da3_gt_loss_inputs(
                stitched_out,
                data,
                used_view_count=used_view_for_this_iter,
                device=device,
            )
            loss = criterion(**loss_inputs)

    if stage == "gt":
        loss = apply_geo_loss_schedule(loss, args=args, global_step=training_global_step)
    return loss


def validate_da3_training_runtime_args(args) -> None:
    if getattr(args, "feedforward_model", None) != "da3":
        return
    if getattr(args, "da3_process_res_method", "upper_bound_resize") != "upper_bound_resize":
        return
    if getattr(args, "batch_size", 1) != 1:
        raise ValueError(
            "DA3 upper_bound_resize stitching training currently requires batch_size=1 "
            "because non-square samples are not padded into a shared batch canvas."
        )


def should_save_checkpoint_for_global_step(
    *,
    global_step: int,
    save_every_n_steps: int,
    is_last_training_step: bool = False,
) -> bool:
    if save_every_n_steps <= 0:
        raise ValueError("save_every_n_steps must be a positive integer.")
    if global_step < 0:
        raise ValueError("global_step must be non-negative.")
    if is_last_training_step:
        return True
    return global_step > 0 and global_step % save_every_n_steps == 0


def should_log_training_step(
    *,
    step: int,
    total_steps: int,
    log_every_n_steps: int,
) -> bool:
    if log_every_n_steps <= 0:
        raise ValueError("log_every_n_steps must be a positive integer.")
    return step == 0 or ((step + 1) % log_every_n_steps == 0) or (step == total_steps - 1)


def resolve_save_every_n_steps(
    *,
    save_every_n_steps: Optional[int],
    save_every_n_epochs: Optional[int],
    updates_per_epoch: int,
) -> int:
    if updates_per_epoch <= 0:
        raise ValueError("updates_per_epoch must be a positive integer.")
    if save_every_n_epochs is not None:
        if save_every_n_epochs <= 0:
            raise ValueError("--save_every_n_epochs must be a positive integer.")
        resolved = save_every_n_epochs * updates_per_epoch
        logger.warning(
            "--save_every_n_epochs is deprecated; converting {} epoch(s) to {} global step(s).",
            save_every_n_epochs,
            resolved,
        )
        return resolved
    if save_every_n_steps is None:
        return 100
    if save_every_n_steps <= 0:
        raise ValueError("--save_every_n_steps must be a positive integer.")
    return save_every_n_steps


def _sorted_lightweight_checkpoint_candidates(
    checkpoint_dir: Path,
    pattern: str,
    *,
    stem_prefix: str,
) -> list[Path]:
    return sorted(
        checkpoint_dir.glob(pattern),
        key=lambda candidate: int(candidate.stem[len(stem_prefix):])
        if candidate.stem[len(stem_prefix):].isdigit()
        else -1,
    )


def unwrap_stitched_model(
    stitched_model,
    accelerator: Optional[Accelerator] = None,
):
    if accelerator is not None:
        return accelerator.unwrap_model(stitched_model)
    if isinstance(stitched_model, DDP):
        return stitched_model.module
    return stitched_model


def collect_special_token_state(stitched_model) -> dict[str, torch.Tensor]:

    if isinstance(stitched_model.stitched_3d_model, DA3Stitched):
        pretrained = stitched_model.stitched_3d_model.backbone.pretrained
        token_state = {
            "cls_token": pretrained.cls_token.detach().cpu(),
        }
        if pretrained.register_tokens is not None:
            token_state["register_tokens"] = pretrained.register_tokens.detach().cpu()
        if hasattr(pretrained, "camera_token"):
            token_state["camera_token"] = pretrained.camera_token.detach().cpu()
        return token_state

    return {}


def load_special_token_state(
    stitched_model,
    state_dict: dict[str, torch.Tensor],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_keys: list[str] = []
    unexpected_keys: list[str] = []


    if isinstance(stitched_model.stitched_3d_model, DA3Stitched):
        pretrained = stitched_model.stitched_3d_model.backbone.pretrained
        expected_keys = ["cls_token"]
        if pretrained.register_tokens is not None:
            expected_keys.append("register_tokens")
        if hasattr(pretrained, "camera_token"):
            expected_keys.append("camera_token")
        if "cls_token" in state_dict:
            pretrained.cls_token.data = state_dict["cls_token"]
        if "register_tokens" in state_dict and pretrained.register_tokens is not None:
            pretrained.register_tokens.data = state_dict["register_tokens"]
        if "camera_token" in state_dict and hasattr(pretrained, "camera_token"):
            pretrained.camera_token.data = state_dict["camera_token"]
        missing_keys = tuple(key for key in expected_keys if key not in state_dict)
        unexpected_keys = [
            key
            for key in ("mask_token", "cls_token", "register_tokens", "camera_token")
            if key in state_dict and key not in expected_keys
        ]
        return missing_keys, tuple(unexpected_keys)

    return (), ()


def log_state_dict_key_diagnostics(
    component_name: str,
    *,
    missing_keys=(),
    unexpected_keys=(),
    limit: int = 20,
) -> None:
    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)
    logger.info(
        "{} load diagnostics: missing_keys_count={} missing_keys_head={} unexpected_keys_count={} unexpected_keys_head={}",
        component_name,
        len(missing_keys),
        missing_keys[:limit],
        len(unexpected_keys),
        unexpected_keys[:limit],
    )


def split_incompatible_keys(load_result) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if load_result is None:
        return (), ()
    return (
        tuple(getattr(load_result, "missing_keys", ()) or ()),
        tuple(getattr(load_result, "unexpected_keys", ()) or ()),
    )


_OPTIONAL_STITCHED_PARTS_KEY_MISMATCH_PREFIXES = ("model.scene_flow_head.",)


def get_state_dict_key_mismatch(module, payload) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_keys = tuple(module.state_dict().keys())
    missing_keys = tuple(key for key in expected_keys if key not in payload)
    unexpected_keys = tuple(key for key in payload if key not in expected_keys)
    return missing_keys, unexpected_keys


def is_optional_stitched_parts_key_mismatch_set(
    missing_keys: tuple[str, ...],
    unexpected_keys: tuple[str, ...],
) -> bool:
    mismatched_keys = missing_keys + unexpected_keys
    return bool(mismatched_keys) and all(
        any(key.startswith(prefix) for prefix in _OPTIONAL_STITCHED_PARTS_KEY_MISMATCH_PREFIXES)
        for key in mismatched_keys
    )


def load_stitched_parts_checkpoint(
    stitched_parts,
    state_dict: Mapping[str, object],
    *,
    component_name: str,
) -> str:
    if checkpoint_uses_full_finetune_stitched_parts(state_dict):
        payload = state_dict.get("stitched_parts")
        if payload is None:
            raise KeyError("Checkpoint missing 'stitched_parts' for full-finetune mode.")
        mode = "full"
        missing_keys, unexpected_keys = get_state_dict_key_mismatch(stitched_parts, payload)
        if is_optional_stitched_parts_key_mismatch_set(missing_keys, unexpected_keys):
            incompatible = stitched_parts.load_state_dict(payload, strict=False)
            logger.warning(
                "{} ({}) checkpoint has optional stitched parts key mismatches; keeping current initialization for missing keys and ignoring unexpected optional keys. missing_keys_count={} missing_keys_head={} unexpected_keys_count={} unexpected_keys_head={}",
                component_name,
                mode,
                len(missing_keys),
                list(missing_keys[:20]),
                len(unexpected_keys),
                list(unexpected_keys[:20]),
            )
        else:
            incompatible = stitched_parts.load_state_dict(payload, strict=True)
    else:
        payload = state_dict.get("lora")
        if payload is None:
            raise KeyError("Checkpoint missing 'lora' for LoRA mode.")
        mode = "lora"
        incompatible = stitched_parts.load_state_dict(payload, strict=False)

    log_state_dict_key_diagnostics(
        f"{component_name} ({mode})",
        missing_keys=incompatible.missing_keys,
        unexpected_keys=incompatible.unexpected_keys,
    )
    return mode


def clone_checkpoint_tensors_for_save(value):
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu().clone()
    if isinstance(value, dict):
        return {key: clone_checkpoint_tensors_for_save(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_checkpoint_tensors_for_save(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_checkpoint_tensors_for_save(item) for item in value)
    return value


def build_lightweight_checkpoint_state_from_unwrapped(
    stitched_model,
    *,
    full_finetune_stitched_parts: bool,
):
    state_dict = {
        "stitching_layer": clone_checkpoint_tensors_for_save(stitched_model.stitching_layer.state_dict()),
    }
    state_dict.update(
        clone_checkpoint_tensors_for_save(
            export_stitched_parts_state(
                stitched_model.stitched_3d_model,
                full_finetune_stitched_parts=full_finetune_stitched_parts,
            )
        )
    )
    state_dict.update(clone_checkpoint_tensors_for_save(collect_special_token_state(stitched_model)))
    return state_dict


def _strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: clone_checkpoint_tensors_for_save(value)
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _filter_lora_state_from_flat_state_dict(
    module_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for key, value in module_state_dict.items():
        if "lora_" not in key:
            continue
        filtered[key] = clone_checkpoint_tensors_for_save(value)
        bias_name = key.split("lora_")[0] + "bias"
        if bias_name in module_state_dict:
            filtered[bias_name] = clone_checkpoint_tensors_for_save(module_state_dict[bias_name])
    return filtered


def _extract_special_token_state_from_module_state(
    stitched_model,
    module_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    def _first_present(*candidate_keys: str) -> Optional[torch.Tensor]:
        for candidate_key in candidate_keys:
            value = module_state_dict.get(candidate_key)
            if value is not None:
                return clone_checkpoint_tensors_for_save(value)
        return None


    if isinstance(stitched_model.stitched_3d_model, DA3Stitched):
        token_state = {}
        for token_name, candidate_keys in (
            ("cls_token", ("model.backbone.pretrained.cls_token", "backbone.pretrained.cls_token")),
            (
                "register_tokens",
                ("model.backbone.pretrained.register_tokens", "backbone.pretrained.register_tokens"),
            ),
            ("camera_token", ("model.backbone.pretrained.camera_token", "backbone.pretrained.camera_token")),
        ):
            value = _first_present(*candidate_keys)
            if value is not None:
                token_state[token_name] = value
        return token_state

    return {}


def build_lightweight_checkpoint_state_from_flat_state_dict(
    stitched_model,
    flat_state_dict: Mapping[str, torch.Tensor],
    *,
    full_finetune_stitched_parts: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | str]:
    stitching_state = _strip_state_dict_prefix(flat_state_dict, "stitching_layer.")
    stitched_parts_state = _strip_state_dict_prefix(flat_state_dict, "stitched_3d_model.")

    state_dict: dict[str, torch.Tensor | dict[str, torch.Tensor] | str] = {
        "stitching_layer": stitching_state,
    }
    if full_finetune_stitched_parts:
        state_dict["stitched_parts_mode"] = "full"
        state_dict["stitched_parts"] = stitched_parts_state
    else:
        state_dict["stitched_parts_mode"] = "lora"
        state_dict["lora"] = _filter_lora_state_from_flat_state_dict(stitched_parts_state)
    state_dict.update(_extract_special_token_state_from_module_state(stitched_model, stitched_parts_state))
    return state_dict


def fsdp_full_param_context(stitched_model, *, accelerator: Optional[Accelerator]):
    if accelerator is None or getattr(accelerator, "distributed_type", None) != DistributedType.FSDP:
        return nullcontext()

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return FSDP.summon_full_params(
        stitched_model,
        writeback=False,
        offload_to_cpu=True,
        rank0_only=True,
    )


def build_lightweight_checkpoint_state(
    stitched_model,
    args,
    accelerator: Optional[Accelerator] = None,
):
    full_finetune_stitched_parts = getattr(args, "full_finetune_stitched_parts", False)

    if accelerator is not None and getattr(accelerator, "distributed_type", None) == DistributedType.DEEPSPEED:
        full_state_dict = accelerator.get_state_dict(stitched_model)
        if not accelerator.is_main_process:
            return None
        unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
        return build_lightweight_checkpoint_state_from_flat_state_dict(
            unwrapped,
            full_state_dict,
            full_finetune_stitched_parts=full_finetune_stitched_parts,
        )

    if accelerator is not None and getattr(accelerator, "distributed_type", None) == DistributedType.FSDP:
        with fsdp_full_param_context(stitched_model, accelerator=accelerator):
            if not accelerator.is_main_process:
                return None
            unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
            return build_lightweight_checkpoint_state_from_unwrapped(
                unwrapped,
                full_finetune_stitched_parts=full_finetune_stitched_parts,
            )

    if accelerator is not None and not accelerator.is_main_process:
        return None

    unwrapped = unwrap_stitched_model(stitched_model, accelerator=accelerator)
    return build_lightweight_checkpoint_state_from_unwrapped(
        unwrapped,
        full_finetune_stitched_parts=full_finetune_stitched_parts,
    )


def select_used_view_count(args, total_views: int) -> int:
    if args.feedforward_model == "da3":
        return total_views

    candidate_views = [view_count for view_count in (9, 13, 17, 21) if view_count <= total_views]
    if not candidate_views:
        return total_views
    return random.choice(candidate_views)


def prepare_da3_gt_loss_inputs(
    stitched_out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    used_view_count: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pred = canonicalize_da3_geometry_outputs(stitched_out)

    def _slice_to_device(key: str):
        value = batch[key][:, :used_view_count].to(device)
        return value

    loss_inputs = {
        "pred_depth": pred["depth"],
        "target_depth": _slice_to_device("depth"),
        "pred_ray": pred["ray"],
        "target_ray": _slice_to_device("ray"),
        "depth_conf": pred["depth_conf"],
        "ray_conf": pred["ray_conf"],
        "valid_mask": _slice_to_device("valid_mask"),
        "intrinsics": _slice_to_device("intrinsics"),
        "extrinsics": _slice_to_device("extrinsics"),
    }
    if stitched_out.get("scene_flow") is not None and batch.get("scene_flow") is not None:
        loss_inputs["pred_scene_flow"] = stitched_out["scene_flow"].to(device)
        loss_inputs["target_scene_flow"] = _slice_to_device("scene_flow")
        if batch.get("deform_mask") is not None:
            loss_inputs["deform_mask"] = _slice_to_device("deform_mask")
    if stitched_out.get("camera_intrinsics") is not None:
        loss_inputs["pred_camera_intrinsics"] = stitched_out["camera_intrinsics"].to(device)
    if stitched_out.get("camera_extrinsics") is not None:
        loss_inputs["pred_camera_extrinsics"] = stitched_out["camera_extrinsics"].to(device)
    if stitched_out.get("camera_tokens") is not None:
        loss_inputs["pred_camera_tokens"] = stitched_out["camera_tokens"].to(device)
    if stitched_out.get("camera_image_hw") is not None:
        loss_inputs["pred_camera_image_hw"] = stitched_out["camera_image_hw"]

    return loss_inputs


def apply_geo_loss_schedule(
    loss_dict: Dict[str, torch.Tensor],
    *,
    args,
    global_step: int,
) -> Dict[str, torch.Tensor]:
    if "total_loss" not in loss_dict or not torch.is_tensor(loss_dict["total_loss"]):
        return loss_dict
    if getattr(args, "feedforward_model", None) != "da3":
        return loss_dict

    raw_total_loss = loss_dict["total_loss"]
    warmup_scale = compute_geo_loss_warmup_scale(
        global_step=global_step,
        warmup_steps=getattr(args, "geo_loss_warmup_steps", 0),
        start=getattr(args, "geo_loss_warmup_start", 1.0),
    )
    effective_geo_weight = getattr(args, "geo_loss_weight", 1.0) * warmup_scale

    scheduled_loss = dict(loss_dict)
    scheduled_loss["geo_loss_raw"] = raw_total_loss
    scheduled_loss["geo_loss_warmup_scale"] = raw_total_loss.new_tensor(warmup_scale)
    scheduled_loss["geo_loss_weight_effective"] = raw_total_loss.new_tensor(effective_geo_weight)
    scheduled_loss["geo_loss"] = raw_total_loss * effective_geo_weight
    scheduled_loss["total_loss"] = scheduled_loss["geo_loss"]
    return scheduled_loss


def resolve_resume_lightweight_checkpoint_path(
    resume_dir: str,
    saved_step: Optional[int] = None,
    saved_epoch: Optional[int] = None,
) -> Optional[str]:
    if saved_step is not None:
        exact_match = os.path.join(resume_dir, f"stitched_model_step_{saved_step}.pth")
        if os.path.exists(exact_match):
            return exact_match
    if saved_epoch is not None:
        exact_match = os.path.join(resume_dir, f"stitched_model_epoch_{saved_epoch}.pth")
        if os.path.exists(exact_match):
            return exact_match

    candidate_paths = _sorted_lightweight_checkpoint_candidates(
        Path(resume_dir),
        "stitched_model_step_*.pth",
        stem_prefix="stitched_model_step_",
    )
    if candidate_paths:
        return str(candidate_paths[-1])

    candidate_paths = _sorted_lightweight_checkpoint_candidates(
        Path(resume_dir),
        "stitched_model_epoch_*.pth",
        stem_prefix="stitched_model_epoch_",
    )
    if candidate_paths:
        return str(candidate_paths[-1])
    return None


def infer_resume_full_finetune_stitched_parts(
    resume_dir: Optional[str],
    *,
    cli_value: bool,
) -> bool:
    if resume_dir is None or not os.path.isdir(resume_dir):
        return cli_value

    saved_args = None
    saved_epoch = None
    saved_step = None
    resume_checkpoint_path = os.path.join(resume_dir, "resume_checkpoint.pth")
    if os.path.exists(resume_checkpoint_path):
        resume_payload = torch.load(
            resume_checkpoint_path,
            weights_only=False,
            map_location="cpu",
        )
        if isinstance(resume_payload, dict):
            saved_args = resume_payload.get("args")
            if resume_payload.get("epoch") is not None:
                saved_epoch = int(resume_payload["epoch"])
            if resume_payload.get("global_step") is not None:
                saved_step = int(resume_payload["global_step"])

    state_dict = None
    checkpoint_path = resolve_resume_lightweight_checkpoint_path(
        resume_dir,
        saved_step=saved_step,
        saved_epoch=saved_epoch,
    )
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state_dict = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location="cpu",
        )

    return resolve_full_finetune_stitched_parts(
        cli_value=cli_value,
        saved_args=saved_args,
        state_dict=state_dict,
    )


def resolve_init_stitched_model_checkpoint_path(
    init_path: str,
) -> str:
    if not init_path:
        raise ValueError("--init_stitched_model_path must be a non-empty path.")

    expanded_path = os.path.abspath(os.path.expanduser(init_path))
    if os.path.isdir(expanded_path):
        checkpoint_path = resolve_resume_lightweight_checkpoint_path(
            expanded_path,
            saved_step=None,
            saved_epoch=None,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find stitched_model_step_*.pth or stitched_model_epoch_*.pth under init dir: {expanded_path}"
            )
        return checkpoint_path

    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Init stitched model checkpoint does not exist: {expanded_path}")
    return expanded_path


def infer_init_full_finetune_stitched_parts(
    init_checkpoint_path: Optional[str],
    *,
    cli_value: bool,
) -> bool:
    if init_checkpoint_path is None:
        return cli_value

    checkpoint_path = resolve_init_stitched_model_checkpoint_path(init_checkpoint_path)
    state_dict = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )
    return resolve_full_finetune_stitched_parts(
        cli_value=cli_value,
        state_dict=state_dict,
    )


def load_initial_stitched_model_checkpoint(
    init_checkpoint_path: str,
    stitched_model,
) -> str:
    checkpoint_path = resolve_init_stitched_model_checkpoint_path(init_checkpoint_path)
    model_state_dict = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )
    if "stitching_layer" not in model_state_dict:
        raise KeyError(
            f"Init stitched model checkpoint is missing 'stitching_layer': {checkpoint_path}"
        )

    stitched_parts_mode = load_stitched_parts_checkpoint(
        stitched_model.stitched_3d_model,
        model_state_dict,
        component_name="init checkpoint stitched parts",
    )
    incompatible = stitched_model.stitching_layer.load_state_dict(
        model_state_dict["stitching_layer"],
        strict=True,
    )
    missing_keys, unexpected_keys = split_incompatible_keys(incompatible)
    log_state_dict_key_diagnostics(
        "init checkpoint stitching_layer",
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    missing_keys, unexpected_keys = load_special_token_state(stitched_model, model_state_dict)
    log_state_dict_key_diagnostics(
        "init checkpoint special tokens",
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    return stitched_parts_mode


def load_resume_checkpoint_metadata(
    resume_dir: str,
) -> tuple[dict, int, int, int]:
    checkpoint = torch.load(
        os.path.join(resume_dir, "resume_checkpoint.pth"),
        weights_only=False,
        map_location="cpu",
    )
    saved_epoch = int(checkpoint["epoch"])
    saved_step_in_epoch = checkpoint.get("step_in_epoch", None)
    if saved_step_in_epoch is None:
        logger.warning(
            "Resume checkpoint {} is missing step_in_epoch; falling back to next-epoch resume semantics.",
            resume_dir,
        )
        return checkpoint, saved_epoch + 1, int(checkpoint.get("global_step", 0)), 0
    return checkpoint, saved_epoch, int(checkpoint.get("global_step", 0)), int(saved_step_in_epoch)


def load_resume_checkpoint_weights_only(
    resume_dir: str,
    stitched_model,
) -> tuple[str, int, int, int]:
    stitched_parts_mode = load_initial_stitched_model_checkpoint(resume_dir, stitched_model)
    _, epoch_start, global_step, step_in_epoch_start = load_resume_checkpoint_metadata(resume_dir)
    return stitched_parts_mode, epoch_start, global_step, step_in_epoch_start


def load_lightweight_checkpoint(
    resume_dir: str,
    stitched_model,
) -> tuple[dict, int, int, int]:
    checkpoint, epoch_start, global_step, step_in_epoch_start = load_resume_checkpoint_metadata(
        resume_dir
    )
    saved_epoch = int(checkpoint["epoch"])
    checkpoint_path = resolve_resume_lightweight_checkpoint_path(
        resume_dir,
        saved_step=int(checkpoint.get("global_step", 0)),
        saved_epoch=saved_epoch,
    )
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"Could not find stitched_model_step_*.pth or stitched_model_epoch_*.pth under resume dir: {resume_dir}"
        )
    model_state_dict = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location="cpu",
    )
    load_stitched_parts_checkpoint(
        stitched_model.stitched_3d_model,
        model_state_dict,
        component_name="resume checkpoint stitched parts",
    )
    incompatible = stitched_model.stitching_layer.load_state_dict(
        model_state_dict["stitching_layer"],
        strict=True,
    )
    missing_keys, unexpected_keys = split_incompatible_keys(incompatible)
    log_state_dict_key_diagnostics(
        "resume checkpoint stitching_layer",
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    missing_keys, unexpected_keys = load_special_token_state(stitched_model, model_state_dict)
    log_state_dict_key_diagnostics(
        "resume checkpoint special tokens",
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )
    return checkpoint, epoch_start, global_step, step_in_epoch_start


def save_checkpoint(
    path,
    epoch,
    stitched_model,
    optimizer,
    scheduler,
    args,
    accelerator: Optional[Accelerator] = None,
    global_step: int = 0,
    step_in_epoch: int = 0,
):
    if global_step <= 0:
        raise ValueError("global_step must be a positive integer when saving checkpoints.")
    if step_in_epoch < 0:
        raise ValueError("step_in_epoch must be non-negative when saving checkpoints.")
    logger.info("Saving resume checkpoint at global step {} (epoch {}).", global_step, epoch)
    save_folder_for_step = os.path.join(path, f"step_{global_step}")
    os.makedirs(save_folder_for_step, exist_ok=True)
    resume_checkpoint_path = os.path.join(
        save_folder_for_step, "resume_checkpoint.pth"
    )

    if accelerator is None:
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "step_in_epoch": step_in_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": args,
            },
            resume_checkpoint_path,
        )
    else:
        accelerate_state_dir = os.path.join(save_folder_for_step, "accelerate_state")
        accelerator.save_state(accelerate_state_dir)
        if accelerator.is_main_process:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": step_in_epoch,
                    "args": args,
                },
                resume_checkpoint_path,
            )
            with open(
                os.path.join(save_folder_for_step, "meta.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "step_in_epoch": step_in_epoch,
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
        accelerator.wait_for_everyone()

    save_model_path = os.path.join(
        save_folder_for_step, f"stitched_model_step_{global_step}.pth"
    )
    logger.info("Saving model at global step {} (epoch {}).", global_step, epoch)
    state_dict = build_lightweight_checkpoint_state(
        stitched_model,
        args,
        accelerator=accelerator,
    )
    if state_dict is not None:
        torch.save(state_dict, save_model_path)
    if accelerator is not None:
        accelerator.wait_for_everyone()


def training_one_epoch_loop(
    stitched_model,
    feed_forward_model,
    optimizer,
    scheduler,
    train_dataloader,
    loss_criterion,
    epoch,
    args,
    global_step_start: int = 0,
    step_in_epoch_start: int = 0,
    total_steps_override: Optional[int] = None,
    resume_loader_is_prefiltered: bool = False,
    accelerator: Optional[Accelerator] = None,
    wandb_logger=None,
):
    logger.info(f"Epoch {epoch} training started.")
    stitched_model.train()
    if feed_forward_model is not None:
        feed_forward_model.eval()
    if args.log_every_n_steps <= 0:
        raise ValueError("--log_every_n_steps must be a positive integer.")

    is_main_process = (
        accelerator.is_main_process if accelerator is not None else dist_util.is_main_process()
    )
    total_steps = len(train_dataloader) if total_steps_override is None else int(total_steps_override)
    if step_in_epoch_start < 0:
        raise ValueError("step_in_epoch_start must be non-negative.")
    if step_in_epoch_start > total_steps:
        raise ValueError("step_in_epoch_start cannot exceed total_steps.")
    is_prefiltered_resume = resume_loader_is_prefiltered or (
        total_steps_override is not None
        and len(train_dataloader) == total_steps - step_in_epoch_start
    )
    if is_prefiltered_resume and len(train_dataloader) != total_steps - step_in_epoch_start:
        raise ValueError(
            "Prefiltered resume train_dataloader length does not match total_steps_override."
        )
    iterable = (
        tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            total=total_steps,
            initial=step_in_epoch_start if is_prefiltered_resume else 0,
            dynamic_ncols=True,
            leave=False,
        )
        if is_main_process
        else train_dataloader
    )
    current_global_step = global_step_start

    step_iterator = (
        enumerate(iterable, start=step_in_epoch_start)
        if is_prefiltered_resume
        else enumerate(iterable)
    )
    for step, data in step_iterator:
        if not is_prefiltered_resume and step < step_in_epoch_start:
            continue
        training_global_step = current_global_step
        grad_norm = None
        if accelerator is None:
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_stitching_loss_for_batch(
                data,
                stage=args.training_stage,
                stitched_model=stitched_model,
                feed_forward_model=feed_forward_model,
                loss_criterion=loss_criterion,
                args=args,
                training_global_step=training_global_step,
                accelerator=accelerator,
            )

            total_loss = loss["total_loss"]
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in stitched_model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with accelerator.accumulate(stitched_model):
                loss = _compute_stitching_loss_for_batch(
                    data,
                    stage=args.training_stage,
                    stitched_model=stitched_model,
                    feed_forward_model=feed_forward_model,
                    loss_criterion=loss_criterion,
                    args=args,
                    training_global_step=training_global_step,
                    accelerator=accelerator,
                )

                total_loss = loss["total_loss"]
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        (p for p in stitched_model.parameters() if p.requires_grad),
                        1.0,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        should_log_step = should_log_training_step(
            step=step,
            total_steps=total_steps,
            log_every_n_steps=args.log_every_n_steps,
        )
        log_dict = None
        if should_log_step or wandb_logger is not None:
            log_dict = build_step_log_dict(
                loss,
                lr=scheduler.get_last_lr()[0],
                grad_norm=grad_norm,
                accelerator=accelerator,
            )
        if should_log_step and is_main_process and log_dict is not None:
            step_log_dict = filter_step_log_dict(log_dict)
            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(
                    {
                        key: f"{value:.4f}"
                        for key, value in step_log_dict.items()
                        if key in {"total_loss", "lr", "grad_norm"}
                    }
                )
            metric_message = ", ".join(f"{key}={value:.6f}" for key, value in step_log_dict.items())
            logger.info(
                "Epoch {} Step {}/{}: {}",
                epoch,
                step + 1,
                total_steps,
                metric_message,
            )
        if wandb_logger is not None and log_dict is not None:
            wandb_logger.log(log_dict)

        current_global_step += 1
        is_last_training_step = (
            epoch == args.num_epochs - 1 and step == total_steps - 1
        )
        if should_save_checkpoint_for_global_step(
            global_step=current_global_step,
            save_every_n_steps=args.save_every_n_steps,
            is_last_training_step=is_last_training_step,
        ):
            save_checkpoint(
                args.save_path,
                epoch,
                stitched_model,
                optimizer,
                scheduler,
                args,
                accelerator=accelerator,
                global_step=current_global_step,
                step_in_epoch=step + 1,
            )

    return current_global_step


def training_one_epoch_loop_hybrid(
    stitched_model,
    feed_forward_model,
    optimizer,
    scheduler,
    *,
    gt_dataloader,
    distill_dataloader,
    loss_criterion_by_stage,
    epoch,
    args,
    global_step_start: int = 0,
    step_in_epoch_start: int = 0,
    total_gt_steps: Optional[int] = None,
    total_distill_steps: Optional[int] = None,
    resume_loader_is_prefiltered: bool = False,
    accelerator: Optional[Accelerator] = None,
    wandb_logger=None,
):
    logger.info(f"Epoch {epoch} hybrid training started.")
    stitched_model.train()
    if feed_forward_model is not None:
        feed_forward_model.eval()
    if args.log_every_n_steps <= 0:
        raise ValueError("--log_every_n_steps must be a positive integer.")

    is_main_process = (
        accelerator.is_main_process if accelerator is not None else dist_util.is_main_process()
    )
    resolved_gt_steps = len(gt_dataloader) if total_gt_steps is None else int(total_gt_steps)
    resolved_distill_steps = (
        len(distill_dataloader) if total_distill_steps is None else int(total_distill_steps)
    )
    total_steps = resolved_gt_steps + resolved_distill_steps
    if step_in_epoch_start < 0:
        raise ValueError("step_in_epoch_start must be non-negative.")
    if step_in_epoch_start > total_steps:
        raise ValueError("step_in_epoch_start cannot exceed total_steps.")
    schedule = build_hybrid_step_schedule(
        num_gt_steps=resolved_gt_steps,
        num_distill_steps=resolved_distill_steps,
        epoch=epoch,
        global_seed=args.global_seed,
    )
    skip_counts = compute_hybrid_resume_batch_skips(
        schedule=schedule,
        step_in_epoch_start=step_in_epoch_start,
    )
    is_prefiltered_resume = resume_loader_is_prefiltered or (
        len(gt_dataloader) == resolved_gt_steps - skip_counts["gt"]
        and len(distill_dataloader) == resolved_distill_steps - skip_counts["distill"]
    )
    if is_prefiltered_resume:
        if len(gt_dataloader) != resolved_gt_steps - skip_counts["gt"]:
            raise ValueError("Prefiltered gt_dataloader length does not match resume metadata.")
        if len(distill_dataloader) != resolved_distill_steps - skip_counts["distill"]:
            raise ValueError(
                "Prefiltered distill_dataloader length does not match resume metadata."
            )
    iterable = (
        tqdm(
            range(step_in_epoch_start if is_prefiltered_resume else 0, total_steps),
            desc=f"Epoch {epoch}",
            total=total_steps,
            initial=step_in_epoch_start if is_prefiltered_resume else 0,
            dynamic_ncols=True,
            leave=False,
        )
        if is_main_process
        else range(step_in_epoch_start if is_prefiltered_resume else 0, total_steps)
    )
    dataloader_iterators = {
        "gt": iter(gt_dataloader),
        "distill": iter(distill_dataloader),
    }
    current_global_step = global_step_start

    for step in iterable:
        stage = schedule[step]
        data = next(dataloader_iterators[stage])
        if not is_prefiltered_resume and step < step_in_epoch_start:
            continue

        training_global_step = current_global_step
        grad_norm = None
        if accelerator is None:
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_stitching_loss_for_batch(
                data,
                stage=stage,
                stitched_model=stitched_model,
                feed_forward_model=feed_forward_model,
                loss_criterion=loss_criterion_by_stage,
                args=args,
                training_global_step=training_global_step,
                accelerator=accelerator,
            )
            total_loss = loss["total_loss"]
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in stitched_model.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            with accelerator.accumulate(stitched_model):
                loss = _compute_stitching_loss_for_batch(
                    data,
                    stage=stage,
                    stitched_model=stitched_model,
                    feed_forward_model=feed_forward_model,
                    loss_criterion=loss_criterion_by_stage,
                    args=args,
                    training_global_step=training_global_step,
                    accelerator=accelerator,
                )
                total_loss = loss["total_loss"]
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        (p for p in stitched_model.parameters() if p.requires_grad),
                        1.0,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        should_log_step = should_log_training_step(
            step=step,
            total_steps=total_steps,
            log_every_n_steps=args.log_every_n_steps,
        )
        log_dict = None
        if should_log_step or wandb_logger is not None:
            log_dict = build_step_log_dict(
                loss,
                lr=scheduler.get_last_lr()[0],
                grad_norm=grad_norm,
                accelerator=accelerator,
            )
        if should_log_step and is_main_process and log_dict is not None:
            step_log_dict = filter_step_log_dict(log_dict)
            if hasattr(iterable, "set_postfix"):
                iterable.set_postfix(
                    {
                        key: f"{value:.4f}"
                        for key, value in step_log_dict.items()
                        if key in {"total_loss", "lr", "grad_norm"}
                    }
                )
            metric_message = ", ".join(
                f"{key}={value:.6f}" for key, value in step_log_dict.items()
            )
            logger.info(
                "Epoch {} Step {}/{} [{}]: {}",
                epoch,
                step + 1,
                total_steps,
                stage,
                metric_message,
            )
        if wandb_logger is not None and log_dict is not None:
            wandb_logger.log(log_dict)

        current_global_step += 1
        is_last_training_step = epoch == args.num_epochs - 1 and step == total_steps - 1
        if should_save_checkpoint_for_global_step(
            global_step=current_global_step,
            save_every_n_steps=args.save_every_n_steps,
            is_last_training_step=is_last_training_step,
        ):
            save_checkpoint(
                args.save_path,
                epoch,
                stitched_model,
                optimizer,
                scheduler,
                args,
                accelerator=accelerator,
                global_step=current_global_step,
                step_in_epoch=step + 1,
            )

    return current_global_step


def main(args):
    if args.resume_checkpoint_path is not None and args.init_stitched_model_path is not None:
        raise ValueError(
            "--resume_checkpoint_path and --init_stitched_model_path are mutually exclusive. "
            "Use resume to restore optimizer/scheduler state, or init_stitched_model_path "
            "to initialize only model weights."
        )
    if args.resume_checkpoint_weights_only and args.resume_checkpoint_path is None:
        raise ValueError(
            "--resume_checkpoint_weights_only requires --resume_checkpoint_path so it can "
            "load stitched weights and resume metadata from that directory."
        )

    if getattr(args, "student_train_scope", "default") != "default":
        args.full_finetune_stitched_parts = (
            getattr(args, "student_train_scope", "default") in {"full_model", "transfer_only"}
        )

    cli_full_finetune_value = getattr(args, "full_finetune_stitched_parts", False)
    if args.resume_checkpoint_path is not None:
        resolved_full_finetune = infer_resume_full_finetune_stitched_parts(
            args.resume_checkpoint_path,
            cli_value=cli_full_finetune_value,
        )
    else:
        resolved_full_finetune = infer_init_full_finetune_stitched_parts(
            args.init_stitched_model_path,
            cli_value=cli_full_finetune_value,
        )
    if resolved_full_finetune != cli_full_finetune_value:
        source_path = args.resume_checkpoint_path or args.init_stitched_model_path
        logger.info(
            "Overriding full_finetune_stitched_parts to {} based on checkpoint {}",
            resolved_full_finetune,
            source_path,
        )
    args.full_finetune_stitched_parts = resolved_full_finetune

    args = apply_feedforward_model_defaults(args)
    args.student_train_scope = resolve_student_train_scope(args)
    configure_hf_runtime(args)
    if args.training_stage == "hybrid":
        if args.feedforward_model != "da3":
            raise ValueError("Hybrid stitching training is only implemented for DA3.")
        if not getattr(args, "train_manifest", None):
            raise ValueError("Hybrid stitching training requires manifest-mode datasets.")

    use_accelerate = should_use_accelerate(args)
    accelerator: Optional[Accelerator] = None
    if use_accelerate:
        accelerator = Accelerator(**build_accelerator_kwargs(args))
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        accelerate_set_seed(args.global_seed)
        logger.info(
            "Accelerate enabled: processes={} grad_accum={} mixed_precision={}",
            accelerator.num_processes,
            accelerator.gradient_accumulation_steps,
            accelerator.mixed_precision,
        )
    else:
        device = torch_device_setup()
        is_main_process = dist_util.is_main_process()
        set_random_seed(args.global_seed)

    logger.info("Loading models...")
    teacher_feed_forward_model = load_feedforward_model(args, device)
    if isinstance(teacher_feed_forward_model, torch.nn.Module):
        teacher_feed_forward_model.requires_grad_(False)
        teacher_feed_forward_model.eval()
    video_vae = load_vae(args, device)
    video_vae.eval()
    student_feed_forward_model = teacher_feed_forward_model
    student_da3_load_options = resolve_student_da3_load_options(args)
    if (
        args.feedforward_model == "da3"
        and (
            not student_da3_load_options["load_pretrained"]
            or student_da3_load_options["load_scene_flow_head"]
            != bool(getattr(args, "da3_enable_scene_flow_head", False))
        )
    ):
        student_feed_forward_model = load_feedforward_model(
            args,
            torch.device("cpu"),
            **student_da3_load_options,
        )
        logger.info(
            "Loaded separate DA3 student init model: init_mode={} load_pretrained={} load_scene_flow_head={}",
            args.student_da3_init_mode,
            student_da3_load_options["load_pretrained"],
            student_da3_load_options["load_scene_flow_head"],
        )
    logger.info("Initializing stitching model")
    stitched_model = StitchVAE3D(
        diffusion_vae=video_vae,
        feedforward_model=student_feed_forward_model,
        device=device,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=args.stitching_layer_config,
        resolution=args.resolution,
        feedforward_resolution=args.feedforward_resolution,
        stitching_layer_init_path=args.initialization_weight_path,
        stitching_module_type=getattr(args, "stitching_module_type", "conv3d"),
        stitching_hidden_channels=getattr(args, "stitching_hidden_channels", None),
        stitching_norm=getattr(args, "stitching_norm", "rms"),
        stitching_aux_kernel=getattr(args, "stitching_aux_kernel", None),
        stitching_attention_heads=getattr(args, "stitching_attention_heads", 4),
        num_frames=args.num_frames_per_unit_scene,
        da3_ref_view_strategy=args.da3_ref_view_strategy,
    )
    if student_feed_forward_model is not teacher_feed_forward_model:
        del student_feed_forward_model

    lora_cfg = parse_lora_mode(args.lora_config)
    runtime_diffusion_vae = get_stitched_runtime_diffusion_vae(stitched_model)
    for param in runtime_diffusion_vae.parameters():
        param.requires_grad = False
    configure_student_model_training(
        stitched_model,
        lora_cfg=lora_cfg,
        student_train_scope=args.student_train_scope,
    )
    logger.info(
        "Student train scope: {} (checkpoint_mode={})",
        args.student_train_scope,
        "full" if args.full_finetune_stitched_parts else "lora",
    )
    if args.init_stitched_model_path is not None:
        logger.info("Initializing stitched model weights from {}", args.init_stitched_model_path)
        init_mode = load_initial_stitched_model_checkpoint(
            args.init_stitched_model_path,
            stitched_model,
        )
        logger.info(
            "Initialized stitched model checkpoint with stitched parts mode: {}",
            init_mode,
        )

    resume_dir = args.resume_checkpoint_path
    resume_checkpoint = None
    epoch_start = 0
    global_step = 0
    step_in_epoch_start = 0
    resume_accelerate_state = (
        accelerator is not None
        and resume_dir is not None
        and not args.resume_checkpoint_weights_only
        and os.path.isdir(os.path.join(resume_dir, "accelerate_state"))
    )
    if resume_dir is not None and args.resume_checkpoint_weights_only:
        logger.info(
            "Initializing stitched model weights from resume checkpoint {} and inheriting only epoch/global_step/step_in_epoch.",
            resume_dir,
        )
        init_mode, epoch_start, global_step, step_in_epoch_start = (
            load_resume_checkpoint_weights_only(
                resume_dir,
                stitched_model,
            )
        )
        logger.info(
            "Initialized resume checkpoint weights with stitched parts mode: {}. Optimizer/scheduler/DeepSpeed state will not be restored.",
            init_mode,
        )
    elif resume_dir is not None and not resume_accelerate_state:
        logger.info("Resuming lightweight checkpoint from {}", resume_dir)
        resume_checkpoint, epoch_start, global_step, step_in_epoch_start = load_lightweight_checkpoint(
            resume_dir,
            stitched_model,
        )

    stitched_model.to(device)
    if accelerator is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
        detach_runtime_vae = getattr(stitched_model, "detach_runtime_diffusion_vae", None)
        if callable(detach_runtime_vae):
            detach_runtime_vae()
        logger.info("Detached runtime diffusion VAE from stitched model before DeepSpeed prepare.")
    apply_training_precision_policy(
        stitched_model,
        teacher_feed_forward_model,
        use_accelerate=accelerator is not None,
    )
    cast_frozen_teacher_to_training_dtype(
        teacher_feed_forward_model,
        accelerator=accelerator,
    )

    if args.feedforward_model == "da3" and args.training_stage == "gt":
        teacher_feed_forward_model = None

    validate_da3_training_runtime_args(args)

    if accelerator is None:
        stitched_model = DDP(
            stitched_model,
            find_unused_parameters=True,
        )

    train_dataloader = None
    distributed_sampler = None
    hybrid_dataloaders = None
    hybrid_samplers = None
    full_train_steps = None
    full_gt_steps = None
    full_distill_steps = None
    raw_train_steps = None
    raw_gt_steps = None
    raw_distill_steps = None
    updates_per_epoch = None
    if args.training_stage == "hybrid":
        hybrid_dataloaders, hybrid_samplers = create_hybrid_stitching_dataloaders(
            args,
            use_distributed_sampler=accelerator is None,
        )
        raw_gt_steps = len(hybrid_dataloaders["gt"])
        raw_distill_steps = len(hybrid_dataloaders["distill"])
    else:
        train_dataloader, distributed_sampler = create_stitching_dataloader(
            args.dataset,
            args,
            augmentation=True,
            use_distributed_sampler=accelerator is None,
        )
        raw_train_steps = len(train_dataloader)

    logger.info("Setting up optimizer.")
    logger.info("configuration:Optimizer: AdamW")
    logger.info(f"configuration:Learning rate: {args.learning_rate}")
    logger.info(f"configuration:Weight decay: {args.weight_decay}")
    logger.info(f"configuration:Warmup steps: {args.warmup_steps}")

    optimizer = torch.optim.AdamW(
        (p for p in stitched_model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if accelerator is not None:
        if args.training_stage == "hybrid":
            stitched_model, optimizer, hybrid_dataloaders["gt"], hybrid_dataloaders["distill"] = accelerator.prepare(
                stitched_model,
                optimizer,
                hybrid_dataloaders["gt"],
                hybrid_dataloaders["distill"],
            )
        else:
            stitched_model, optimizer, train_dataloader = accelerator.prepare(
                stitched_model,
                optimizer,
                train_dataloader,
            )
    if args.training_stage == "hybrid":
        full_gt_steps = len(hybrid_dataloaders["gt"])
        full_distill_steps = len(hybrid_dataloaders["distill"])
        updates_per_epoch = full_gt_steps + full_distill_steps
        if accelerator is not None and (
            raw_gt_steps != full_gt_steps or raw_distill_steps != full_distill_steps
        ):
            logger.info(
                "Adjusted hybrid step counts after Accelerate prepare: gt {} -> {}, distill {} -> {}.",
                raw_gt_steps,
                full_gt_steps,
                raw_distill_steps,
                full_distill_steps,
            )
    else:
        full_train_steps = len(train_dataloader)
        updates_per_epoch = full_train_steps
        if accelerator is not None and raw_train_steps != full_train_steps:
            logger.info(
                "Adjusted train step count after Accelerate prepare: {} -> {}.",
                raw_train_steps,
                full_train_steps,
            )

    total_updates = max(1, args.num_epochs * updates_per_epoch)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )

    if accelerator is not None:
        accelerator.register_for_checkpointing(scheduler)
        if resume_accelerate_state and resume_dir is not None:
            logger.info("Resuming accelerate state from {}", resume_dir)
            accelerator.load_state(os.path.join(resume_dir, "accelerate_state"))
            meta = torch.load(
                os.path.join(resume_dir, "resume_checkpoint.pth"),
                weights_only=False,
                map_location="cpu",
            )
            if meta.get("step_in_epoch", None) is None:
                logger.warning(
                    "Accelerate resume checkpoint {} is missing step_in_epoch; falling back to next-epoch resume semantics.",
                    resume_dir,
                )
                epoch_start = int(meta.get("epoch", 0)) + 1
                step_in_epoch_start = 0
            else:
                epoch_start = int(meta.get("epoch", 0))
                step_in_epoch_start = int(meta.get("step_in_epoch", 0))
            global_step = int(meta.get("global_step", 0))
    elif resume_checkpoint is not None:
        logger.info(
            "Resuming optimizer & scheduler state from {}", args.resume_checkpoint_path
        )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])

    main_rank = is_main_process if accelerator is None else accelerator.is_main_process
    if args.wandb_logging and main_rank:
        logger.info("Setting up wandb logger.")
        import wandb

        wandb_logger = wandb.init(
            project=args.wandb_project_name,
            name=args.exp_name,
            config=vars(args),
        )
    else:
        wandb_logger = None

    loss_criterion = build_da3_loss_criterion(args)

    os.makedirs(args.save_path, exist_ok=True)
    args.save_every_n_steps = resolve_save_every_n_steps(
        save_every_n_steps=getattr(args, "save_every_n_steps", None),
        save_every_n_epochs=getattr(args, "save_every_n_epochs", None),
        updates_per_epoch=updates_per_epoch,
    )

    resume_train_dataloader = None
    resume_hybrid_dataloaders = None
    if step_in_epoch_start > 0:
        if args.training_stage == "hybrid":
            resume_schedule = build_hybrid_step_schedule(
                num_gt_steps=full_gt_steps,
                num_distill_steps=full_distill_steps,
                epoch=epoch_start,
                global_seed=args.global_seed,
            )
            resume_skip_counts = compute_hybrid_resume_batch_skips(
                schedule=resume_schedule,
                step_in_epoch_start=step_in_epoch_start,
            )
            logger.info(
                "Fast resume for epoch {} from step {}: skip {} gt batches and {} distill batches.",
                epoch_start,
                step_in_epoch_start,
                resume_skip_counts["gt"],
                resume_skip_counts["distill"],
            )
            resume_hybrid_dataloaders = {
                "gt": maybe_skip_first_batches_for_resume(
                    hybrid_dataloaders["gt"],
                    skip_batches=resume_skip_counts["gt"],
                ),
                "distill": maybe_skip_first_batches_for_resume(
                    hybrid_dataloaders["distill"],
                    skip_batches=resume_skip_counts["distill"],
                ),
            }
        else:
            logger.info(
                "Fast resume for epoch {} from step {}: skip {} train batches.",
                epoch_start,
                step_in_epoch_start,
                step_in_epoch_start,
            )
            resume_train_dataloader = maybe_skip_first_batches_for_resume(
                train_dataloader,
                skip_batches=step_in_epoch_start,
            )

    for epoch in range(epoch_start, args.num_epochs):
        if args.training_stage == "hybrid":
            active_hybrid_dataloaders = (
                resume_hybrid_dataloaders
                if epoch == epoch_start and resume_hybrid_dataloaders is not None
                else hybrid_dataloaders
            )
            for sampler in (hybrid_samplers or {}).values():
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
            for loader in active_hybrid_dataloaders.values():
                if hasattr(loader, "set_epoch"):
                    loader.set_epoch(epoch)
                sampler = getattr(loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            global_step = training_one_epoch_loop_hybrid(
                stitched_model,
                teacher_feed_forward_model,
                optimizer,
                scheduler,
                gt_dataloader=active_hybrid_dataloaders["gt"],
                distill_dataloader=active_hybrid_dataloaders["distill"],
                loss_criterion_by_stage=loss_criterion,
                epoch=epoch,
                args=args,
                global_step_start=global_step,
                step_in_epoch_start=step_in_epoch_start if epoch == epoch_start else 0,
                total_gt_steps=full_gt_steps,
                total_distill_steps=full_distill_steps,
                resume_loader_is_prefiltered=(
                    epoch == epoch_start and resume_hybrid_dataloaders is not None
                ),
                accelerator=accelerator,
                wandb_logger=wandb_logger,
            )
        else:
            active_train_dataloader = (
                resume_train_dataloader
                if epoch == epoch_start and resume_train_dataloader is not None
                else train_dataloader
            )
            if distributed_sampler is not None and hasattr(distributed_sampler, "set_epoch"):
                distributed_sampler.set_epoch(epoch)
            if hasattr(active_train_dataloader, "set_epoch"):
                active_train_dataloader.set_epoch(epoch)
            sampler = getattr(active_train_dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            global_step = training_one_epoch_loop(
                stitched_model,
                teacher_feed_forward_model,
                optimizer,
                scheduler,
                active_train_dataloader,
                loss_criterion,
                epoch,
                args,
                global_step_start=global_step,
                step_in_epoch_start=step_in_epoch_start if epoch == epoch_start else 0,
                total_steps_override=full_train_steps,
                resume_loader_is_prefiltered=(
                    epoch == epoch_start and resume_train_dataloader is not None
                ),
                accelerator=accelerator,
                wandb_logger=wandb_logger,
            )
        step_in_epoch_start = 0


if __name__ == "__main__":
    args = stitching_training_argument().parse_args()
    main(args)

#update