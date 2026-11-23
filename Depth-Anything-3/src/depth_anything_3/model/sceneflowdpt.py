from typing import Any, Sequence

import torch

from depth_anything_3.model.dpt import DPT


class SceneFlowDPT(DPT):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 3,
        activation: str = "linear",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
    ) -> None:
        # For direction_expm1, the parent DPT uses linear activation;
        # the vector-level expm1 is applied in this class's forward().
        self._scene_flow_activation = activation
        parent_activation = "linear" if activation == "direction_expm1" else activation
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=parent_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            head_name="scene_flow_cam",
            use_sky_head=False,
            norm_type=norm_type,
            fusion_block_inplace=fusion_block_inplace,
        )
        self.has_conf = False

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        output = super().forward(*args, **kwargs)
        if self._scene_flow_activation == "direction_expm1":
            flow = output["scene_flow_cam"]
            d = flow.norm(dim=2, keepdim=True)
            direction = flow / d.clamp(min=1e-8)
            output["scene_flow_cam"] = direction * torch.expm1(d)
        # FSDP's output traversal treats objects exposing `__dataclass_fields__`
        # as dataclasses. `addict.Dict` exposes that attribute, so convert to a
        # plain dict before returning from the trainable module.
        return dict(output)
