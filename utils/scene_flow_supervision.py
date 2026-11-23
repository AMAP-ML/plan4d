from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent
PLAN4D_ROOT = PROJECT_ROOT


def _ensure_plan4d_training_imports() -> None:
    da3_root = PLAN4D_ROOT / "Depth-Anything-3"
    da3_src_root = da3_root / "src"
    for path in (PLAN4D_ROOT, da3_src_root):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_scene_flow_loss_class():
    _ensure_plan4d_training_imports()
    scene_flow_loss_path = (
        PLAN4D_ROOT
        / "Depth-Anything-3"
        / "training"
        / "losses"
        / "scene_flow_loss.py"
    )
    if not scene_flow_loss_path.exists():
        raise FileNotFoundError(f"Scene-flow loss file not found: {scene_flow_loss_path}")

    module_name = "_plan4d_da3_scene_flow_loss"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, scene_flow_loss_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to build import spec for {scene_flow_loss_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    SceneFlowLoss = module.SceneFlowLoss

    return SceneFlowLoss


class GeometrySceneFlowLoss(nn.Module):
    def __init__(
        self,
        *,
        base_geometry_loss,
        lambda_scene_flow: float = 1.0,
        scene_flow_loss_type: str = "l1",
        scene_flow_zero_last_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_geometry_loss = base_geometry_loss
        self.lambda_scene_flow = float(lambda_scene_flow)
        self.scene_flow_loss = load_scene_flow_loss_class()(
            loss_type=scene_flow_loss_type,
            zero_last_weight=scene_flow_zero_last_weight,
        )

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        base_kwargs = dict(kwargs)
        pred_scene_flow = base_kwargs.pop("pred_scene_flow", None)
        target_scene_flow = base_kwargs.pop("target_scene_flow", None)
        deform_mask = base_kwargs.pop("deform_mask", None)

        loss_dict = dict(self.base_geometry_loss(**base_kwargs))
        total_loss = loss_dict["total_loss"]

        if (
            self.lambda_scene_flow <= 0
            or pred_scene_flow is None
            or target_scene_flow is None
            or deform_mask is None
        ):
            scene_flow_loss = (
                pred_scene_flow.float().sum() * 0.0
                if pred_scene_flow is not None
                else total_loss.new_zeros(())
            )
            loss_dict.setdefault("scene_flow_loss", scene_flow_loss)
            loss_dict["total_loss"] = total_loss + scene_flow_loss
            return loss_dict

        if deform_mask.dim() == 5 and deform_mask.shape[2] == 1:
            deform_mask = deform_mask.squeeze(2)
        scene_flow_metrics = self.scene_flow_loss(
            pred=pred_scene_flow,
            target=target_scene_flow,
            deform_mask=deform_mask,
        )
        scene_flow_loss = scene_flow_metrics["loss"] * self.lambda_scene_flow
        loss_dict["scene_flow_loss"] = scene_flow_loss
        loss_dict["scene_flow_epe"] = scene_flow_metrics["epe"]
        loss_dict["scene_flow_valid_ratio"] = scene_flow_metrics["valid_ratio"]
        loss_dict["total_loss"] = total_loss + scene_flow_loss
        return loss_dict
