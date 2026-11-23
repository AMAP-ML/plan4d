import torch
import torch.nn as nn

from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous


class SceneFlowAdapter(nn.Module):
    def forward(
        self,
        scene_flow_cam: torch.Tensor,
        extrinsics: torch.Tensor,
        zero_last: bool = True,
    ) -> torch.Tensor:
        if scene_flow_cam.ndim != 5:
            raise ValueError(
                f"scene_flow_cam should have shape [B, S, 3, H, W], got {scene_flow_cam.shape}"
            )
        if extrinsics is None:
            raise ValueError("extrinsics is required to convert scene flow to frame0 basis")

        extrinsics = as_homogeneous(extrinsics)
        rot_c2w = affine_inverse(extrinsics)[..., :3, :3]
        scene_flow = torch.einsum("bsij,bsjhw->bsihw", rot_c2w, scene_flow_cam)

        if zero_last and scene_flow.shape[1] > 0:
            scene_flow = scene_flow.clone()
            scene_flow[:, -1] = 0

        return scene_flow
