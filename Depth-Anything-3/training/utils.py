from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from omegaconf import OmegaConf


def strip_model_prefix(state_dict: Dict[str, torch.Tensor], prefix: str = 'model.') -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[len(prefix):] if key.startswith(prefix) else key] = value
    return cleaned


def load_checkpoint_state(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def get_anyview_da3_net(model: torch.nn.Module) -> torch.nn.Module:
    da3_model = model
    if hasattr(da3_model, 'model'):
        da3_model = da3_model.model
    if hasattr(da3_model, 'da3'):
        return da3_model.da3
    return da3_model


def get_scene_flow_head(model: torch.nn.Module) -> torch.nn.Module:
    da3_net = get_anyview_da3_net(model)
    head = getattr(da3_net, 'scene_flow_head', None)
    if head is None:
        raise RuntimeError('scene_flow_head is not available in the loaded DA3 model')
    return head


def get_scene_flow_adapter(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    da3_net = get_anyview_da3_net(model)
    return getattr(da3_net, 'scene_flow_adapter', None)


def freeze_all_but_scene_flow_head(model: torch.nn.Module) -> torch.nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    scene_flow_head = get_scene_flow_head(model)
    for param in scene_flow_head.parameters():
        param.requires_grad = True
    return scene_flow_head


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def load_da3_with_scene_flow_head(
    repo_root: Path,
    weights_dir: str,
    scene_flow_preset: str = 'da3-giant-sceneflow',
    checkpoint_name: str = 'model.safetensors',
    config_name: str = 'config.json',
    device: str | torch.device = 'cpu',
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    repo_root = Path(repo_root)
    src_root = repo_root / 'src'

    import sys
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY

    weights_dir = Path(weights_dir)
    if not weights_dir.is_absolute():
        weights_dir = (repo_root / weights_dir).resolve()
    config_path = weights_dir / config_name
    checkpoint_path = weights_dir / checkpoint_name
    if not config_path.exists():
        raise FileNotFoundError(f'Missing DA3 config: {config_path}')
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Missing DA3 checkpoint: {checkpoint_path}')

    with open(config_path, 'r', encoding='utf-8') as file:
        loaded = json.load(file)
    model_cfg = OmegaConf.create(loaded.get('config', loaded))

    preset_cfg = load_config(MODEL_REGISTRY[scene_flow_preset])
    target_cfg = model_cfg.anyview if 'anyview' in model_cfg else model_cfg
    target_cfg.scene_flow_head = preset_cfg.scene_flow_head
    target_cfg.scene_flow_adapter = preset_cfg.scene_flow_adapter

    model = create_object(model_cfg)
    state_dict = strip_model_prefix(load_checkpoint_state(str(checkpoint_path)))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    return model, {
        'config_path': str(config_path),
        'checkpoint_path': str(checkpoint_path),
        'missing_keys': list(missing_keys),
        'unexpected_keys': list(unexpected_keys),
    }
