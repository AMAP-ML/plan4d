"""Utility functions to load third-party models."""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import torch
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT

DEFAULT_WAN_VAE_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_DA3_SCENE_FLOW_PRESET = "da3-giant-sceneflow"
DEFAULT_DA3_SCENE_FLOW_HEAD_CHECKPOINT = (
    PLAN4D_ROOT
    / "checkpoints"
    / "scene_flow_head.bin"
)

WAN_VAE_REQUIRED_FILES = ("config.json", "diffusion_pytorch_model.safetensors")
DA3_REQUIRED_FILES = ("config.json", "model.safetensors")


def _normalize_dir(path: str | os.PathLike | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        return None
    return str(path_obj)


def _is_complete_model_dir(path: str | os.PathLike | None, required_files: tuple[str, ...]) -> bool:
    normalized = _normalize_dir(path)
    if normalized is None:
        return False
    path_obj = Path(normalized)
    return path_obj.is_dir() and all((path_obj / file_name).exists() for file_name in required_files)


def _pick_first_complete_dir(
    explicit_path: str | None,
    required_files: tuple[str, ...],
    candidate_dirs: tuple[Path, ...],
) -> str | None:
    if _is_complete_model_dir(explicit_path, required_files):
        return _normalize_dir(explicit_path)

    for candidate in candidate_dirs:
        if _is_complete_model_dir(candidate, required_files):
            return str(candidate.resolve())
    return None




def resolve_wan_vae_path(args: argparse.Namespace) -> str | None:
    return _pick_first_complete_dir(
        explicit_path=getattr(args, "wan_vae_path", None),
        required_files=WAN_VAE_REQUIRED_FILES,
        candidate_dirs=(
            PROJECT_ROOT / "checkpoints" / "wan_vae",
            PROJECT_ROOT / "checkpoints" / "longcat" / "vae",
        ),
    )


def resolve_da3_model_path(args: argparse.Namespace) -> str | None:
    return _pick_first_complete_dir(
        explicit_path=getattr(args, "da3_model_path", None),
        required_files=DA3_REQUIRED_FILES,
        candidate_dirs=(
            PROJECT_ROOT / "checkpoints" / "da3",
        ),
    )




def configure_hf_runtime(args: argparse.Namespace) -> None:
    """Apply mirror/local model configuration before importing HF-backed modules."""
    hf_endpoint = getattr(args, "hf_endpoint", None)
    if hf_endpoint:
        hf_endpoint = hf_endpoint.rstrip("/")
        os.environ["HF_ENDPOINT"] = hf_endpoint
    else:
        hf_endpoint = os.environ.get("HF_ENDPOINT")

    da3_model_path = resolve_da3_model_path(args)
    if da3_model_path is not None:
        os.environ["PLAN4D_DA3_MODEL_PATH"] = da3_model_path

    try:
        import huggingface_hub.constants as hf_constants

        if hf_endpoint:
            hf_constants.ENDPOINT = hf_endpoint
            hf_constants.HUGGINGFACE_CO_URL_HOME = f"{hf_endpoint}/"
            hf_constants.HUGGINGFACE_CO_URL_TEMPLATE = (
                hf_endpoint + "/{repo_id}/resolve/{revision}/{filename}"
            )
    except ImportError:
        pass

    try:
        import transformers.utils.hub as transformers_hub

        if hf_endpoint:
            transformers_hub.HUGGINGFACE_CO_RESOLVE_ENDPOINT = hf_endpoint
            transformers_hub.HUGGINGFACE_CO_PREFIX = (
                hf_endpoint + "/{model_id}/resolve/{revision}/{filename}"
            )
            transformers_hub.HUGGINGFACE_CO_EXAMPLES_TELEMETRY = (
                hf_endpoint + "/api/telemetry/examples"
            )
    except ImportError:
        pass


def _ensure_plan4d_da3_import_paths() -> Path:
    plan4d_root = PLAN4D_ROOT
    da3_src_root = plan4d_root / "Depth-Anything-3" / "src"

    for path in (plan4d_root, da3_src_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    if not da3_src_root.exists():
        raise FileNotFoundError(
            f"Could not locate DA3 source directory: {da3_src_root}"
        )
    return plan4d_root


def _load_state_dict_from_checkpoint(
    checkpoint_path: Path, *, mmap: bool = False,
) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(checkpoint_path))

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu",
        mmap=mmap and zipfile.is_zipfile(checkpoint_path),
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _strip_model_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_state_dict[key[6:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _load_scene_flow_head_checkpoint(
    scene_flow_head: torch.nn.Module | None,
    checkpoint_path: Path | None,
) -> None:
    if scene_flow_head is None or checkpoint_path is None:
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DA3 scene-flow head checkpoint not found: {checkpoint_path}")

    state_dict = _load_state_dict_from_checkpoint(checkpoint_path, mmap=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported scene-flow checkpoint payload type: {type(state_dict)}")
    if "scene_flow_head" in state_dict and isinstance(state_dict["scene_flow_head"], dict):
        state_dict = state_dict["scene_flow_head"]
    elif "stitched_parts" in state_dict and isinstance(state_dict["stitched_parts"], dict):
        prefix = "model.scene_flow_head."
        state_dict = {
            key[len(prefix):]: value
            for key, value in state_dict["stitched_parts"].items()
            if key.startswith(prefix)
        }
        if not state_dict:
            raise ValueError(f"Stitched checkpoint contains no scene-flow head: {checkpoint_path}")

    state_dict = _strip_model_prefix(state_dict)
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "scene_flow_head.")

    missing_keys, unexpected_keys = scene_flow_head.load_state_dict(state_dict, strict=False)
    allowed_missing = {"norm.weight", "norm.bias"}
    missing_keys = set(missing_keys)
    unexpected_keys = set(unexpected_keys)
    if missing_keys.issubset(allowed_missing) and not unexpected_keys:
        return
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load DA3 scene-flow head checkpoint cleanly from "
            f"{checkpoint_path}: missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}"
        )


def build_da3_anyview_model(
    args: argparse.Namespace,
    *,
    device: torch.device | None,
    load_pretrained: bool = True,
    load_scene_flow_head: bool | None = None,
    eval_mode: bool = True,
) -> torch.nn.Module:
    da3_model_path = resolve_da3_model_path(args)
    if da3_model_path is None:
        raise FileNotFoundError(
            "DA3 stitching requires a local model directory. "
            "Set --da3_model_path or place weights under "
            "checkpoints/da3."
        )

    _ensure_plan4d_da3_import_paths()

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from omegaconf import OmegaConf

    model_dir = Path(da3_model_path)
    config_path = model_dir / getattr(args, "da3_config_name", "config.json")
    checkpoint_path = model_dir / getattr(args, "da3_checkpoint_name", "model.safetensors")

    if not config_path.exists():
        raise FileNotFoundError(f"DA3 config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DA3 checkpoint not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_json = json.load(f)

    model_cfg = config_json.get("config", config_json)
    anyview_cfg = OmegaConf.create(model_cfg.get("anyview", model_cfg))
    enable_scene_flow_head = bool(getattr(args, "da3_enable_scene_flow_head", False))
    if enable_scene_flow_head:
        scene_flow_preset = getattr(args, "da3_scene_flow_preset", DEFAULT_DA3_SCENE_FLOW_PRESET)
        preset_cfg = load_config(MODEL_REGISTRY[scene_flow_preset])
        anyview_cfg.scene_flow_head = preset_cfg.scene_flow_head
        anyview_cfg.scene_flow_adapter = preset_cfg.scene_flow_adapter
    model = create_object(anyview_cfg)

    state_dict = _strip_model_prefix(_load_state_dict_from_checkpoint(checkpoint_path))
    anyview_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("da3_metric.") or key.startswith("metric."):
            continue
        if key.startswith("da3."):
            anyview_state_dict[key[4:]] = value
        elif key.startswith("anyview."):
            anyview_state_dict[key[8:]] = value
        else:
            anyview_state_dict[key] = value

    if load_pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(anyview_state_dict, strict=False)
        logger.info(
            "Loaded DA3 anyview model from {} (missing_keys={}, unexpected_keys={})",
            checkpoint_path,
            len(missing_keys),
            len(unexpected_keys),
        )
    else:
        logger.info("Initialized DA3 anyview model from config only without loading {}", checkpoint_path)

    should_load_scene_flow_head = (
        enable_scene_flow_head
        if load_scene_flow_head is None
        else bool(load_scene_flow_head)
    )
    if should_load_scene_flow_head:
        raw_checkpoint_path = getattr(
            args,
            "da3_scene_flow_head_checkpoint_path",
            DEFAULT_DA3_SCENE_FLOW_HEAD_CHECKPOINT,
        )
        # Caller may set the attribute to None to opt out of overriding the
        # scene-flow head (e.g. when the stitched decoder checkpoint already
        # carries a trained head). getattr() returns None instead of falling
        # back to the default in that case, so we have to filter it here.
        if raw_checkpoint_path is None:
            logger.info(
                "Skipping DA3 scene-flow head override: "
                "da3_scene_flow_head_checkpoint_path is None"
            )
        else:
            scene_flow_checkpoint_path = Path(raw_checkpoint_path).expanduser()
            da3_core = model.da3 if hasattr(model, "da3") else model
            _load_scene_flow_head_checkpoint(
                getattr(da3_core, "scene_flow_head", None),
                scene_flow_checkpoint_path,
            )
            logger.info(
                "Loaded DA3 scene-flow head checkpoint from {}",
                scene_flow_checkpoint_path,
            )
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model


def load_da3_anyview_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return build_da3_anyview_model(
        args,
        device=device,
        load_pretrained=True,
        load_scene_flow_head=bool(getattr(args, "da3_enable_scene_flow_head", False)),
        eval_mode=True,
    )


def load_feedforward_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    load_pretrained: bool = True,
    load_scene_flow_head: bool | None = None,
):
    """
    Load a feedforward 3D model based on the provided arguments.
    """

    if args.feedforward_model == "da3":
        logger.info("Loading DA3 anyview model with device {}", device)
        model = build_da3_anyview_model(
            args,
            device=device,
            load_pretrained=load_pretrained,
            load_scene_flow_head=load_scene_flow_head,
            eval_mode=True,
        )
    else:
        raise NotImplementedError(
            f"Feedforward model {args.feedforward_model} is not implemented."
        )
    return model


def load_vae(args: argparse.Namespace, device: torch.device):
    """
    Load a Video VAE based on the provided arguments.
    """
    from diffusers import AutoencoderKLWan

    if args.video_model == "wan":
        local_vae_path = resolve_wan_vae_path(args)
        if local_vae_path is not None:
            logger.info(f"Loading Wan VAE from local directory {local_vae_path} with device {device}")
            video_vae = AutoencoderKLWan.from_pretrained(
                local_vae_path,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        else:
            model_id = getattr(args, "wan_vae_repo_id", DEFAULT_WAN_VAE_REPO_ID)
            logger.info(
                "Loading Wan VAE from {} with device {} (HF_ENDPOINT={})",
                model_id,
                device,
                os.environ.get("HF_ENDPOINT"),
            )
            # Only the VAE subfolder is needed for stitching.
            video_vae = AutoencoderKLWan.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
        video_vae.to(device)
    else:
        raise NotImplementedError(
            f"Video model {args.video_model} is not implemented."
        )
    return video_vae


def cast_to_bfloat16(model: torch.nn.Module):
    """
    Cast the model's Conv2d and Linear layers to bfloat16, except for layers in 'head'.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            # Keep heads in fp32 for stability; cast the rest to bfloat16 for speed/memory.
            if "head" in str.lower(name):
                continue
            if isinstance(module, torch.nn.Conv2d):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
