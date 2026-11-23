"""Helpers for switching stitched parts between LoRA and full-parameter fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch.nn as nn
from loguru import logger

from models.da3_stitched import DA3Stitched
from utils.lora_util.utils import add_lora, lora_state_dict, mark_only_lora_as_trainable


_LORA_BIAS_MODE = "lora_only"
_OPTIONAL_KEY_MISMATCH_PREFIXES = ("model.scene_flow_head.",)


def _state_dict_key_mismatch(
    stitched_parts: nn.Module,
    payload: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    expected_keys = tuple(stitched_parts.state_dict().keys())
    missing_keys = tuple(key for key in expected_keys if key not in payload)
    unexpected_keys = tuple(key for key in payload if key not in expected_keys)
    return missing_keys, unexpected_keys


def _is_optional_key_mismatch_set(
    missing_keys: tuple[str, ...],
    unexpected_keys: tuple[str, ...],
) -> bool:
    mismatched_keys = missing_keys + unexpected_keys
    return bool(mismatched_keys) and all(
        any(key.startswith(prefix) for prefix in _OPTIONAL_KEY_MISMATCH_PREFIXES)
        for key in mismatched_keys
    )


def _log_stitched_parts_diagnostics(
    *,
    mode: str,
    missing_keys=(),
    unexpected_keys=(),
    limit: int = 20,
) -> None:
    missing_keys = list(missing_keys)
    unexpected_keys = list(unexpected_keys)
    logger.info(
        "stitched_parts ({}) load diagnostics: missing_keys_count={} missing_keys_head={} unexpected_keys_count={} unexpected_keys_head={}",
        mode,
        len(missing_keys),
        missing_keys[:limit],
        len(unexpected_keys),
        unexpected_keys[:limit],
    )


def checkpoint_uses_full_finetune_stitched_parts(state_dict: Mapping[str, Any]) -> bool:
    mode = state_dict.get("stitched_parts_mode")
    if mode is not None:
        if mode not in {"lora", "full"}:
            raise ValueError(f"Unsupported stitched_parts_mode: {mode}")
        return mode == "full"
    return "stitched_parts" in state_dict and "lora" not in state_dict


def resolve_full_finetune_stitched_parts(
    *,
    cli_value: bool = False,
    saved_args: Any = None,
    state_dict: Optional[Mapping[str, Any]] = None,
) -> bool:
    if state_dict is not None and any(
        key in state_dict for key in ("stitched_parts_mode", "stitched_parts", "lora")
    ):
        return checkpoint_uses_full_finetune_stitched_parts(state_dict)
    if saved_args is not None and hasattr(saved_args, "full_finetune_stitched_parts"):
        return bool(getattr(saved_args, "full_finetune_stitched_parts"))
    return bool(cli_value)


def prepare_stitched_parts_for_loading(
    stitched_parts: nn.Module,
    lora_cfg,
    *,
    full_finetune_stitched_parts: bool,
) -> None:
    if full_finetune_stitched_parts:
        return

    add_lora(
        stitched_parts,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        fan_in_fan_out=lora_cfg.fan_in_fan_out,
    )


def set_stitched_special_tokens_trainable(
    stitched_parts: nn.Module,
    *,
    requires_grad: bool,
) -> None:

    if isinstance(stitched_parts, DA3Stitched):
        pretrained = stitched_parts.backbone.pretrained
        pretrained.cls_token.requires_grad = requires_grad
        if pretrained.register_tokens is not None:
            pretrained.register_tokens.requires_grad = requires_grad
        if hasattr(pretrained, "camera_token"):
            pretrained.camera_token.requires_grad = requires_grad


def configure_stitched_parts_training(
    stitched_parts: nn.Module,
    lora_cfg,
    *,
    full_finetune_stitched_parts: bool,
) -> None:
    if full_finetune_stitched_parts:
        for param in stitched_parts.parameters():
            param.requires_grad = True
        set_stitched_special_tokens_trainable(stitched_parts, requires_grad=True)
        return

    for param in stitched_parts.parameters():
        param.requires_grad = False
    prepare_stitched_parts_for_loading(
        stitched_parts,
        lora_cfg,
        full_finetune_stitched_parts=False,
    )
    mark_only_lora_as_trainable(stitched_parts, bias=_LORA_BIAS_MODE)
    set_stitched_special_tokens_trainable(stitched_parts, requires_grad=True)


def export_stitched_parts_state(
    stitched_parts: nn.Module,
    *,
    full_finetune_stitched_parts: bool,
) -> dict[str, Any]:
    if full_finetune_stitched_parts:
        return {
            "stitched_parts_mode": "full",
            "stitched_parts": stitched_parts.state_dict(),
        }

    return {
        "stitched_parts_mode": "lora",
        "lora": lora_state_dict(stitched_parts, bias=_LORA_BIAS_MODE),
    }


def load_stitched_parts_state(
    stitched_parts: nn.Module,
    state_dict: Mapping[str, Any],
) -> str:
    if checkpoint_uses_full_finetune_stitched_parts(state_dict):
        payload = state_dict.get("stitched_parts")
        if payload is None:
            raise KeyError("Checkpoint missing 'stitched_parts' for full-finetune mode.")
        missing_keys, unexpected_keys = _state_dict_key_mismatch(stitched_parts, payload)
        _log_stitched_parts_diagnostics(
            mode="full",
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )
        if _is_optional_key_mismatch_set(missing_keys, unexpected_keys):
            stitched_parts.load_state_dict(payload, strict=False)
            logger.warning(
                "stitched_parts ({}) checkpoint has optional key mismatches; keeping current initialization for missing keys and ignoring unexpected optional keys. missing_keys_count={} missing_keys_head={} unexpected_keys_count={} unexpected_keys_head={}",
                "full",
                len(missing_keys),
                list(missing_keys[:20]),
                len(unexpected_keys),
                list(unexpected_keys[:20]),
            )
        else:
            stitched_parts.load_state_dict(payload, strict=True)
        return "full"

    payload = state_dict.get("lora")
    if payload is None:
        raise KeyError("Checkpoint missing 'lora' for LoRA mode.")
    incompatible = stitched_parts.load_state_dict(payload, strict=False)
    _log_stitched_parts_diagnostics(
        mode="lora",
        missing_keys=getattr(incompatible, "missing_keys", ()) or (),
        unexpected_keys=getattr(incompatible, "unexpected_keys", ()) or (),
    )
    return "lora"
