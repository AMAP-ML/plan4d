import re
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

NumberOrTuple = Union[int, Tuple[int, ...]]


def _to_tuple(value: NumberOrTuple, *, dim: int) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        if len(value) != dim:
            raise ValueError(f"Expected tuple of length {dim}, got {value}")
        return tuple(int(item) for item in value)
    return tuple(int(value) for _ in range(dim))


def _resolve_group_count(num_channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2):
        if num_channels % candidate == 0:
            return candidate
    return 1


class ChannelAffine3d(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class ChannelRMSNorm3d(nn.Module):
    def __init__(self, num_channels: int, *, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.square().mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        normalized = x * scale
        if not self.affine:
            return normalized
        return normalized * self.weight + self.bias


def _build_norm_layer(norm_type: str, num_channels: int) -> nn.Module:
    normalized = str(norm_type).lower()
    if normalized in ("none", "identity"):
        return nn.Identity()
    if normalized == "rms":
        return ChannelRMSNorm3d(num_channels, affine=True)
    if normalized == "group":
        return nn.GroupNorm(_resolve_group_count(num_channels), num_channels)
    if normalized == "affine":
        return ChannelAffine3d(num_channels)
    raise ValueError(f"Unsupported stitching norm: {norm_type}")


class _StitchingModuleBase(nn.Module):
    geometry_conv: nn.Conv3d

    @property
    def kernel_size(self):
        return self.geometry_conv.kernel_size

    @property
    def stride(self):
        return self.geometry_conv.stride

    @property
    def padding(self):
        return self.geometry_conv.padding

    @property
    def dilation(self):
        return self.geometry_conv.dilation

    @property
    def in_channels(self):
        return self.geometry_conv.in_channels

    @property
    def out_channels(self):
        return self.geometry_conv.out_channels


# --------------------------------------------------------------------------- #
# 1.  Dataclass that *represents* a convolution layer                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvSpec:
    dim: int
    out_channels: int
    kernel_size: NumberOrTuple
    stride: NumberOrTuple = 1
    padding: NumberOrTuple = 0
    dilation: NumberOrTuple = 1

    # optional helper: build a real nn.Module
    def build(
        self,
        in_channels: int,
        bias: bool = True,
        groups: int = 1,
    ):
        cls_map: Dict[int, type] = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        conv_cls = cls_map[self.dim]

        return conv_cls(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            padding_mode="replicate",
            groups=groups,
            bias=bias,
        )


def _build_conv3d(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: NumberOrTuple,
    stride: NumberOrTuple = 1,
    padding: NumberOrTuple = 0,
    dilation: NumberOrTuple = 1,
    bias: bool = True,
) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        padding_mode="replicate",
    )


def _derive_aux_padding(conv_spec: ConvSpec, aux_kernel_size: NumberOrTuple) -> tuple[int, ...]:
    main_kernel = _to_tuple(conv_spec.kernel_size, dim=conv_spec.dim)
    main_padding = _to_tuple(conv_spec.padding, dim=conv_spec.dim)
    main_dilation = _to_tuple(conv_spec.dilation, dim=conv_spec.dim)
    aux_kernel = _to_tuple(aux_kernel_size, dim=conv_spec.dim)
    aux_padding = []
    for padding, kernel, dilation, aux_kernel_item in zip(
        main_padding,
        main_kernel,
        main_dilation,
        aux_kernel,
    ):
        numerator = 2 * padding - dilation * (kernel - 1) + dilation * (aux_kernel_item - 1)
        if numerator % 2 != 0:
            raise ValueError(
                "Auxiliary kernel cannot preserve output shape with the current ConvSpec: "
                f"kernel={main_kernel}, padding={main_padding}, dilation={main_dilation}, "
                f"aux_kernel={aux_kernel}"
            )
        aux_padding.append(numerator // 2)
    return tuple(aux_padding)


class ConvAffineStitchingModule(_StitchingModuleBase):
    def __init__(
        self,
        *,
        conv_spec: ConvSpec,
        in_channels: int,
        norm_type: str,
    ):
        super().__init__()
        self.conv = conv_spec.build(in_channels=in_channels)
        self.calibrator = _build_norm_layer(norm_type, self.conv.out_channels)
        self.geometry_conv = self.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calibrator(self.conv(x))


class ResidualBottleneckStitchingModule(_StitchingModuleBase):
    def __init__(
        self,
        *,
        conv_spec: ConvSpec,
        in_channels: int,
        hidden_channels: int,
        norm_type: str,
    ):
        super().__init__()
        self.skip_conv = conv_spec.build(in_channels=in_channels)
        self.pre_norm = _build_norm_layer(norm_type, in_channels)
        self.expand = _build_conv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=True,
        )
        self.hidden_conv = _build_conv3d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=conv_spec.kernel_size,
            stride=conv_spec.stride,
            padding=conv_spec.padding,
            dilation=conv_spec.dilation,
            bias=True,
        )
        self.project = _build_conv3d(
            in_channels=hidden_channels,
            out_channels=conv_spec.out_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        self.activation = nn.GELU()
        self.geometry_conv = self.skip_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_conv(x)
        branch = self.pre_norm(x)
        branch = self.activation(self.expand(branch))
        branch = self.activation(self.hidden_conv(branch))
        branch = self.project(branch)
        return residual + branch


class KernelMixtureStitchingModule(_StitchingModuleBase):
    def __init__(
        self,
        *,
        conv_spec: ConvSpec,
        in_channels: int,
        aux_kernel_size: NumberOrTuple,
        norm_type: str,
    ):
        super().__init__()
        aux_padding = _derive_aux_padding(conv_spec, aux_kernel_size)
        self.pre_norm = _build_norm_layer(norm_type, in_channels)
        self.branch_main = conv_spec.build(in_channels=in_channels)
        self.branch_aux = _build_conv3d(
            in_channels=in_channels,
            out_channels=conv_spec.out_channels,
            kernel_size=aux_kernel_size,
            stride=conv_spec.stride,
            padding=aux_padding,
            dilation=conv_spec.dilation,
            bias=True,
        )
        self.branch_logits = nn.Parameter(torch.zeros(2))
        self.geometry_conv = self.branch_main

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_input = self.pre_norm(x)
        weights = torch.softmax(self.branch_logits, dim=0)
        return weights[0] * self.branch_main(branch_input) + weights[1] * self.branch_aux(branch_input)


class AxialAttentionStitchingModule(_StitchingModuleBase):
    def __init__(
        self,
        *,
        conv_spec: ConvSpec,
        in_channels: int,
        hidden_channels: int,
        norm_type: str,
        attention_heads: int,
    ):
        super().__init__()
        if hidden_channels % int(attention_heads) != 0:
            raise ValueError(
                f"hidden_channels={hidden_channels} must be divisible by attention_heads={attention_heads}"
            )
        self.stem = _build_conv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=conv_spec.kernel_size,
            stride=conv_spec.stride,
            padding=conv_spec.padding,
            dilation=conv_spec.dilation,
            bias=True,
        )
        self.pre_norm = _build_norm_layer(norm_type, hidden_channels)
        self.temporal_attn = nn.MultiheadAttention(hidden_channels, attention_heads, batch_first=True)
        self.height_attn = nn.MultiheadAttention(hidden_channels, attention_heads, batch_first=True)
        self.width_attn = nn.MultiheadAttention(hidden_channels, attention_heads, batch_first=True)
        self.output_proj = _build_conv3d(
            in_channels=hidden_channels,
            out_channels=conv_spec.out_channels,
            kernel_size=1,
            bias=True,
        )
        self.geometry_conv = self.stem

    @staticmethod
    def _apply_axis_attention(
        x: torch.Tensor,
        *,
        axis: str,
        attention: nn.MultiheadAttention,
    ) -> torch.Tensor:
        batch_size, channels, frames, height, width = x.shape
        if axis == "t":
            sequence = x.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, frames, channels)
            attended, _ = attention(sequence, sequence, sequence, need_weights=False)
            return attended.reshape(batch_size, height, width, frames, channels).permute(0, 4, 3, 1, 2)
        if axis == "h":
            sequence = x.permute(0, 2, 4, 3, 1).reshape(batch_size * frames * width, height, channels)
            attended, _ = attention(sequence, sequence, sequence, need_weights=False)
            return attended.reshape(batch_size, frames, width, height, channels).permute(0, 4, 1, 3, 2)
        if axis == "w":
            sequence = x.permute(0, 2, 3, 4, 1).reshape(batch_size * frames * height, width, channels)
            attended, _ = attention(sequence, sequence, sequence, need_weights=False)
            return attended.reshape(batch_size, frames, height, width, channels).permute(0, 4, 1, 2, 3)
        raise ValueError(f"Unsupported axis: {axis}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        hidden = hidden + self._apply_axis_attention(self.pre_norm(hidden), axis="t", attention=self.temporal_attn)
        hidden = hidden + self._apply_axis_attention(self.pre_norm(hidden), axis="h", attention=self.height_attn)
        hidden = hidden + self._apply_axis_attention(self.pre_norm(hidden), axis="w", attention=self.width_attn)
        return self.output_proj(hidden)


def build_stitching_module(
    *,
    module_type: str,
    conv_spec: ConvSpec,
    in_channels: int,
    hidden_channels: int | None = None,
    norm_type: str = "rms",
    aux_kernel_size: NumberOrTuple | None = None,
    attention_heads: int = 4,
):
    normalized_module_type = str(module_type).lower()
    hidden_channels = int(hidden_channels or max(conv_spec.out_channels, in_channels))
    if normalized_module_type == "conv3d":
        return conv_spec.build(in_channels=in_channels)
    if normalized_module_type == "res_bottleneck":
        return ResidualBottleneckStitchingModule(
            conv_spec=conv_spec,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
        )
    if normalized_module_type == "conv_affine":
        return ConvAffineStitchingModule(
            conv_spec=conv_spec,
            in_channels=in_channels,
            norm_type=norm_type,
        )
    if normalized_module_type == "kernel_mixture":
        return KernelMixtureStitchingModule(
            conv_spec=conv_spec,
            in_channels=in_channels,
            aux_kernel_size=aux_kernel_size or conv_spec.kernel_size,
            norm_type=norm_type,
        )
    if normalized_module_type == "axial_attention":
        return AxialAttentionStitchingModule(
            conv_spec=conv_spec,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            attention_heads=attention_heads,
        )
    raise ValueError(f"Unsupported stitching module type: {module_type}")


def warm_start_stitching_module_from_conv_state(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> bool:
    if isinstance(module, nn.Conv3d):
        module.load_state_dict(state_dict, strict=True)
        return True

    target_module = None
    for attribute_name in ("skip_conv", "conv", "branch_main", "stem"):
        if hasattr(module, attribute_name):
            target_module = getattr(module, attribute_name)
            break
    if target_module is None:
        return False

    missing = {"weight"} - set(state_dict.keys())
    if missing:
        return False
    if target_module.weight.shape != state_dict["weight"].shape:
        return False

    target_module.weight.data.copy_(state_dict["weight"].to(dtype=target_module.weight.dtype))
    if target_module.bias is not None and state_dict.get("bias") is not None:
        target_module.bias.data.copy_(state_dict["bias"].to(dtype=target_module.bias.dtype))
    return True


# --------------------------------------------------------------------------- #
# 2.  Regex-based parser                                                      #
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(
    r"^conv(?P<dim>[123])d_"  # conv1d / conv2d / conv3d
    r"k(?P<k>[0-9x]+)_"  # kernel size
    r"o(?P<o>[0-9]+)"  # out channels (REQUIRED)
    r"(?:_s(?P<s>[0-9x]+))?"  # optional stride
    r"(?:_p(?P<p>[0-9x]+))?"  # optional padding
    r"(?:_d(?P<d>[0-9x]+))?"  # optional dilation
    r"$",
    re.IGNORECASE,
)


def _to_int_or_tuple(txt: str | None) -> NumberOrTuple:
    if not txt:
        # caller handles defaults
        return 0
    if "x" in txt:
        return tuple(int(n) for n in txt.split("x"))
    return int(txt)


def parse_conv_spec(spec: str) -> ConvSpec:
    """
    Convert 'conv3d_k3x3x3_o32_s2_p1' → ConvSpec(...)
    Raises ValueError if the string doesn’t follow the grammar.
    """
    m = _TOKEN_RE.fullmatch(spec)
    if not m:
        raise ValueError(
            f"Bad CONV_SPEC {spec!r}. Expected something like "
            "'conv2d_k3_o64', 'conv3d_k3x3x3_o32_s2_p1', …"
        )

    g = m.groupdict()
    return ConvSpec(
        dim=int(g["dim"]),
        out_channels=int(g["o"]),
        kernel_size=_to_int_or_tuple(g["k"]),
        stride=_to_int_or_tuple(g["s"]) if g["s"] else 1,
        padding=_to_int_or_tuple(g["p"]) if g["p"] else 0,
        dilation=_to_int_or_tuple(g["d"]) if g["d"] else 1,
    )
