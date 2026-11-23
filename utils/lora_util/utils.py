#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .layers import Conv2d, Linear, LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    # Freeze all params except LoRA (and optionally biases).
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    # Export only LoRA (and optionally bias) weights for lightweight checkpoints.
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The configuration / injection helpers below are original Plan4D code.
# They are independent of loralib (the MIT-licensed `mark_only_lora_as_
# trainable` and `lora_state_dict` above come from microsoft/LoRA).
# ---------------------------------------------------------------------------

_BIAS_CHOICES = ("none", "all", "lora_only")


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "lora_only"
    target_modules: Optional[Tuple[str, ...]] = None
    fan_in_fan_out: bool = False
    finetune_encoder: bool = False
    freeze_head: bool = False


_FLAG_KEYS = {"enc": "finetune_encoder", "fix_head": "freeze_head", "fixhead": "freeze_head"}


def parse_lora_mode(spec: str) -> LoraConfig:
    """Parse a compact LoRA spec string.

    Comma-separated chunks:
      ``r<int>``      LoRA rank (e.g. ``r32``)
      ``a<int>``      LoRA alpha (e.g. ``a16``)
      ``d<float>``    dropout (e.g. ``d0.05``)
      ``b<mode>``     bias mode: none | all | lora_only
      ``t<names>``    target-module substrings joined by ``|``
      ``f<0/1>``      fan_in_fan_out flag
      ``enc``         finetune_encoder flag
      ``fix_head``/``fixhead``  freeze_head flag
    """
    cfg = LoraConfig()
    if not spec:
        return cfg

    numeric_key = {"r", "a", "d", "f"}
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        lowered = chunk.lower()

        flag_attr = _FLAG_KEYS.get(lowered)
        if flag_attr is not None:
            setattr(cfg, flag_attr, True)
            continue

        if not lowered or lowered[0] not in ("r", "a", "d", "b", "t", "f"):
            raise ValueError(f"Bad LoRA chunk: {chunk!r}")
        key, value = lowered[0], lowered[1:]
        if not value:
            raise ValueError(f"Bad LoRA chunk: {chunk!r}")

        if key in numeric_key:
            number = re.fullmatch(r"[\d.]+", value)
            if number is None:
                raise ValueError(f"Bad LoRA chunk: {chunk!r}")
            if key == "r":
                cfg.r = int(value)
            elif key == "a":
                cfg.alpha = int(value)
            elif key == "d":
                cfg.dropout = float(value)
            else:
                cfg.fan_in_fan_out = bool(int(float(value)))
        elif key == "b":
            if value not in _BIAS_CHOICES:
                raise ValueError("b chunk must be none|all|lora_only")
            cfg.bias = value
        else:  # key == "t"
            cfg.target_modules = tuple(value.split("|"))

    return cfg


def _resolve_module_path(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    """Split ``a.b.c.child`` into ``(root.a.b.c, "child")``.

    Walks the attribute chain so the caller can ``setattr`` the leaf on the
    returned parent. A name without dots resolves to the root itself.
    """
    if "." not in dotted_name:
        return root, dotted_name
    parent_path, _, leaf = dotted_name.rpartition(".")
    parent = root
    for attr in parent_path.split("."):
        parent = getattr(parent, attr)
    return parent, leaf


def add_lora(
    model: nn.Module,
    target_modules=None,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    fan_in_fan_out: bool = False,
):
    """Wrap matching ``nn.Linear`` / ``nn.Conv2d`` submodules with LoRA
    counterparts from ``.layers`` in place, copying the frozen weights so the
    wrapped model is functionally identical before any LoRA training."""
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if target_modules and not any(t in full_name for t in target_modules):
            continue

        if isinstance(module, nn.Linear):
            lora_layer = Linear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                fan_in_fan_out=fan_in_fan_out,
                bias=module.bias is not None,
            )
        else:
            if module.kernel_size[0] != module.kernel_size[1]:
                raise ValueError("Only square kernels are supported for Conv2d LoRA layers.")
            lora_layer = Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride,
                padding=module.padding[0],
                dilation=module.dilation[0],
                groups=module.groups,
                bias=module.bias is not None,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
            )

        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)
        module.weight = lora_layer.weight
        parent, leaf_name = _resolve_module_path(model, full_name)
        setattr(parent, leaf_name, lora_layer)
    return model
