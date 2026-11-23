from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
DA3_ROOT = THIS_DIR.parent
PLAN4D_ROOT = DA3_ROOT.parent


def sanitize_import_paths() -> None:
    # When this file is executed as `training/train_scene_flow_head.py`,
    # Python puts `training/` at sys.path[0], which shadows the external
    # `datasets` package with our local `training/datasets`.
    if str(THIS_DIR) in sys.path:
        sys.path.remove(str(THIS_DIR))
    resolved_script_dir = THIS_DIR.resolve()
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or '.').resolve() != resolved_script_dir
    ]

    preferred_paths = [str(DA3_ROOT), str(DA3_ROOT / 'src'), str(PLAN4D_ROOT)]
    remaining_paths = [entry for entry in sys.path if entry not in preferred_paths]
    sys.path[:] = preferred_paths + remaining_paths


sanitize_import_paths()

from plan4d.da3_interface import DA3StreamingInterface
from training.datasets.multi_scene_flow import (
    MixedSceneFlowDataset,
    build_scene_flow_train_components,
    build_scene_flow_val_loaders,
)
from training.eval_scene_flow_head import load_scene_flow_head_checkpoint
from training.losses.scene_flow_loss import SceneFlowLoss
from training.samplers.mixture_sampler import DistributedMixtureBatchSampler
from training.utils import (
    count_parameters,
    freeze_all_but_scene_flow_head,
    get_scene_flow_head,
    load_da3_with_scene_flow_head,
)

DEFAULT_PROCESS_RES = 504
DEFAULT_PROCESS_RES_METHOD = "upper_bound_resize"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train DA3 scene flow DPT head with fixed DA3 backbone')
    parser.add_argument('--config', type=str, default='../configs/scene_flow.yaml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--init-scene-flow-head-ckpt', type=str, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--use-accelerate', action='store_true')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=None)
    parser.add_argument('--mixed-precision', type=str, default=None, choices=['no', 'fp16', 'bf16'])
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def should_save_step_checkpoint(global_step: int, training_cfg: Mapping[str, object]) -> bool:
    interval = int(training_cfg.get('save_every_n_steps') or 0)
    return interval > 0 and global_step > 0 and global_step % interval == 0


def build_periodic_checkpoint_path(
    output_dir: Path,
    global_step: int,
    accelerator: Optional[Accelerator] = None,
) -> Path:
    checkpoint_name = f'step_{global_step:06d}'
    return output_dir / 'checkpoints' / (checkpoint_name if accelerator is not None else f'{checkpoint_name}.pt')


def get_checkpoint_progress(
    *,
    epoch: int,
    completed_steps_in_epoch: int,
    epoch_completed: bool,
) -> Dict[str, int]:
    if epoch_completed:
        return {
            'epoch': int(epoch) + 1,
            'step_in_epoch': 0,
        }
    return {
        'epoch': int(epoch),
        'step_in_epoch': int(completed_steps_in_epoch),
    }


def compute_resume_batches_to_skip(step_in_epoch: int, grad_accumulation_steps: int) -> int:
    return max(0, int(step_in_epoch)) * max(1, int(grad_accumulation_steps))


def apply_data_preprocess_defaults(config: Dict) -> None:
    data_cfg = config.setdefault('data', {})
    process_res = int(data_cfg.get('process_res', DEFAULT_PROCESS_RES))
    process_res_method = str(data_cfg.get('process_res_method', DEFAULT_PROCESS_RES_METHOD))
    data_cfg['process_res'] = process_res
    data_cfg['process_res_method'] = process_res_method
    data_cfg.setdefault('target_height', process_res)
    data_cfg.setdefault('target_width', process_res)


def resolve_data_roots(data_cfg: Dict) -> None:
    if 'data_root' in data_cfg and not Path(data_cfg['data_root']).is_absolute():
        data_cfg['data_root'] = str((DA3_ROOT / data_cfg['data_root']).resolve())
    for spec in data_cfg.get('datasets', []):
        if 'data_root' in spec and not Path(spec['data_root']).is_absolute():
            spec['data_root'] = str((DA3_ROOT / spec['data_root']).resolve())
        if 'builder_dir' in spec and not Path(spec['builder_dir']).is_absolute():
            spec['builder_dir'] = str((DA3_ROOT / spec['builder_dir']).resolve())
        for manifest_key in ('train_manifest', 'val_manifest'):
            if manifest_key in spec and spec[manifest_key] and not Path(spec[manifest_key]).is_absolute():
                spec[manifest_key] = str((DA3_ROOT / spec[manifest_key]).resolve())
        for manifest_map_key in ('train_manifests', 'val_manifests'):
            manifest_map = spec.get(manifest_map_key)
            if not manifest_map:
                continue
            for split_name, manifest_path in list(manifest_map.items()):
                if manifest_path and not Path(manifest_path).is_absolute():
                    manifest_map[split_name] = str((DA3_ROOT / manifest_path).resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def should_use_accelerate(args: argparse.Namespace, env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    if getattr(args, 'use_accelerate', False):
        return True

    try:
        world_size = int(env.get('WORLD_SIZE', '1'))
    except ValueError:
        world_size = 1
    try:
        local_rank = int(env.get('LOCAL_RANK', '-1'))
    except ValueError:
        local_rank = -1

    return world_size > 1 or local_rank >= 0


def setup_logger(output_dir: Path, is_main_process: bool = True, process_index: int = 0) -> logging.Logger:
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f'train_scene_flow_head.rank{process_index}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if not is_main_process:
        stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)

    if is_main_process:
        file_handler = logging.FileHandler(output_dir / 'train.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_batch_dataset_name(batch: Dict[str, object]) -> str:
    dataset_name = batch.get('dataset_name', 'unknown')
    if isinstance(dataset_name, (list, tuple)):
        return str(dataset_name[0]) if dataset_name else 'unknown'
    return str(dataset_name)


def maybe_initialize_scene_flow_head_from_checkpoint(
    da3_model: torch.nn.Module,
    checkpoint_path: Optional[str],
    logger: logging.Logger,
    *,
    skip_if_missing: bool = True,
) -> bool:
    if not checkpoint_path:
        return False
    checkpoint = Path(checkpoint_path)
    if skip_if_missing and not checkpoint.exists():
        logger.warning('Skipping scene_flow_head init because checkpoint was not found: %s', checkpoint)
        return False
    load_scene_flow_head_checkpoint(get_scene_flow_head(da3_model), str(checkpoint))
    logger.info('Initialized scene_flow_head from %s', checkpoint)
    return True


def resolve_init_scene_flow_head_ckpt(
    cli_checkpoint: Optional[str],
    config: Dict,
    repo_root: Path,
) -> Optional[str]:
    checkpoint = cli_checkpoint or config.get('model', {}).get('init_scene_flow_head_ckpt')
    if checkpoint and not Path(checkpoint).is_absolute():
        checkpoint = str((repo_root / checkpoint).resolve())
    return checkpoint


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_scale: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_scale = min_lr_scale
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_id = 0

    def step(self) -> None:
        self.step_id += 1
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * self.get_scale(self.step_id)

    def state_dict(self) -> Dict[str, float]:
        return {'step_id': self.step_id}

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.step_id = int(state_dict.get('step_id', 0))
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * self.get_scale(self.step_id)

    def get_scale(self, step_id: int) -> float:
        if self.warmup_steps > 0 and step_id <= self.warmup_steps:
            return step_id / self.warmup_steps
        progress = (step_id - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine


def save_checkpoint(
    path: Path,
    da3_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    config: Dict,
    accelerator: Optional[Accelerator] = None,
) -> None:
    if accelerator is None:
        scene_flow_head = get_scene_flow_head(da3_model)
        checkpoint = {
            'scene_flow_head': scene_flow_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'step_in_epoch': step_in_epoch,
            'global_step': global_step,
            'config': config,
        }
        torch.save(checkpoint, path)
        return

    path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(path))
    if accelerator.is_main_process:
        with open(path / 'meta.json', 'w', encoding='utf-8') as file:
            json.dump(
                {
                    'epoch': epoch,
                    'step_in_epoch': step_in_epoch,
                    'global_step': global_step,
                    'config': config,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )
    accelerator.wait_for_everyone()


def set_scene_flow_head_module(model: torch.nn.Module, scene_flow_head: torch.nn.Module) -> None:
    da3_model = model
    if hasattr(da3_model, 'model'):
        da3_model = da3_model.model
    if hasattr(da3_model, 'da3'):
        da3_model = da3_model.da3
    if not hasattr(da3_model, 'scene_flow_head'):
        raise RuntimeError('scene_flow_head is not available in the loaded DA3 model')
    da3_model.scene_flow_head = scene_flow_head


def load_state_checkpoint(
    path: Path,
    accelerator: Accelerator,
    logger: logging.Logger,
) -> Dict[str, int]:
    accelerator.load_state(str(path))
    meta_path = path / 'meta.json'
    meta = {'epoch': 0, 'step_in_epoch': 0, 'global_step': 0}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as file:
            meta = json.load(file)
    logger.info('Resumed accelerate state from %s', path)
    return {
        'epoch': int(meta.get('epoch', 0)),
        'step_in_epoch': int(meta.get('step_in_epoch', 0)),
        'global_step': int(meta.get('global_step', 0)),
    }


def load_file_resume_checkpoint(
    path: str,
    da3_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    logger: logging.Logger,
    load_optimizer_state: bool = True,
) -> Dict[str, int]:
    checkpoint = torch.load(path, map_location='cpu')
    get_scene_flow_head(da3_model).load_state_dict(checkpoint['scene_flow_head'])
    if load_optimizer_state:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        logger.warning('Loaded scene_flow_head weights from %s without optimizer/scheduler state', path)
    logger.info('Resumed scene_flow_head from %s', path)
    return {
        'epoch': int(checkpoint.get('epoch', 0)),
        'step_in_epoch': int(checkpoint.get('step_in_epoch', 0)),
        'global_step': int(checkpoint.get('global_step', 0)),
    }


def reduce_loss_dict(loss_dict: Dict[str, torch.Tensor], accelerator: Optional[Accelerator] = None) -> Dict[str, float]:
    reduced: Dict[str, float] = {}
    for key, value in loss_dict.items():
        scalar = value.detach().float()
        if accelerator is not None:
            scalar = accelerator.reduce(scalar, reduction='mean')
        reduced[key] = float(scalar.item())
    return reduced


def reduce_batch_scalar(value: object, accelerator: Optional[Accelerator] = None) -> float:
    if value is None:
        return float("nan")
    scalar = torch.as_tensor(value, dtype=torch.float32)
    if scalar.ndim > 0:
        scalar = scalar.mean()
    if accelerator is not None:
        scalar = accelerator.reduce(scalar.to(accelerator.device), reduction='mean')
    return float(scalar.item())


def optional_batch_tensor(batch: Mapping[str, object], key: str, device: torch.device) -> Optional[torch.Tensor]:
    value = batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def build_scene_flow_loss_kwargs(
    batch: Mapping[str, object],
    device: torch.device,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> Dict[str, object]:
    return {
        'scene_flow_scale': optional_batch_tensor(batch, 'scene_flow_scale', device),
        'gt_depth': optional_batch_tensor(batch, 'gt_depth', device),
        'gt_depth_valid_mask': optional_batch_tensor(batch, 'gt_depth_valid_mask', device),
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'flow_target_amplification': optional_batch_tensor(batch, 'flow_target_amplification', device),
    }


@torch.no_grad()
def evaluate(
    da3_interface: DA3StreamingInterface,
    dataloader: DataLoader,
    criterion: SceneFlowLoss,
    device: torch.device,
    decoder_chunk_size: int,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, float]:
    da3_interface.eval()
    metric_names = (
        'loss',
        'epe',
        'valid_ratio',
        'last_frame_zero_epe',
        'loss_depth_warp',
        'depth_warp_abs',
        'depth_warp_valid_ratio',
    )
    metrics = {key: 0.0 for key in metric_names}
    num_batches = 0.0

    for batch in dataloader:
        images = batch['rgbs'].permute(0, 1, 4, 2, 3).to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        target_flow = batch['scene_flow'].permute(0, 1, 4, 2, 3).to(device)
        deform_mask = batch['deform_mask'].to(device)
        height, width = images.shape[-2:]

        features = da3_interface.extract_multilayer_features(images, normalize=True, ref_view_strategy='first')
        head_dtype = next(get_scene_flow_head(da3_interface.da3_model).parameters()).dtype
        features = features.to(dtype=head_dtype)
        pred_flow = da3_interface.decode_features_to_scene_flow(
            features=features,
            image_size=(height, width),
            extrinsics=extrinsics,
            use_gradient=False,
            chunk_size=decoder_chunk_size,
            output_basis='frame0',
        )
        loss_dict = criterion(
            pred_flow,
            target_flow,
            deform_mask,
            **build_scene_flow_loss_kwargs(batch, device, intrinsics, extrinsics),
        )
        reduced_loss_dict = reduce_loss_dict(loss_dict, accelerator)
        for key in metric_names:
            metrics[key] += reduced_loss_dict[key]
        num_batches += 1.0

    if num_batches == 0:
        return metrics
    return {key: value / num_batches for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (DA3_ROOT / config_path).resolve()
    config = load_yaml_config(str(config_path))
    apply_data_preprocess_defaults(config)
    if args.smoke:
        config['data']['batch_size'] = 1
        config['data']['chunk_size'] = 17
        config['data']['overlap'] = 8
        config['data']['process_res'] = 196
        config['data']['process_res_method'] = DEFAULT_PROCESS_RES_METHOD
        config['data']['target_height'] = 196
        config['data']['target_width'] = 196
        if 'datasets' in config['data']:
            for spec in config['data']['datasets']:
                if spec['name'] == 'point_odyssey':
                    spec['max_scenes'] = 1
                    spec['max_val_scenes'] = 1
                if spec['name'] == 'blinkvision':
                    spec['max_scenes_per_split'] = 1
                    spec['max_val_scenes_per_split'] = 1
            config['training']['steps_per_epoch'] = 1
        else:
            config['data']['split'] = config['data'].get('smoke_split', 'sample')
            config['data']['val_split'] = config['data'].get('smoke_split', 'sample')
            config['data']['max_scenes'] = 1
        config['training']['num_epochs'] = 1
        config['training']['max_steps'] = 1
        config['training']['save_every_n_steps'] = 1
        config['training']['eval_every_n_epochs'] = 1
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.num_workers is not None:
        config['data']['num_workers'] = args.num_workers

    if not Path(config['model']['weights_dir']).is_absolute():
        config['model']['weights_dir'] = str((DA3_ROOT / config['model']['weights_dir']).resolve())
    resolve_data_roots(config['data'])

    use_accelerate = should_use_accelerate(args)
    accelerator: Optional[Accelerator] = None
    if use_accelerate:
        mixed_precision = args.mixed_precision or config['training'].get('mixed_precision')
        gradient_accumulation_steps = int(
            args.gradient_accumulation_steps
            or config['training'].get('gradient_accumulation_steps', 1)
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision if mixed_precision and mixed_precision != 'no' else None,
        )
        device = accelerator.device
        is_main_process = accelerator.is_main_process
        process_index = accelerator.process_index
        accelerate_set_seed(int(config.get('seed', 42)))
    else:
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        is_main_process = True
        process_index = 0
        set_seed(int(config.get('seed', 42)))

    output_dir = Path(args.output_dir or config['training']['output_dir'])
    if not output_dir.is_absolute():
        output_dir = (DA3_ROOT / output_dir).resolve()
    logger = setup_logger(output_dir, is_main_process=is_main_process, process_index=process_index)
    logger.info('Using device: %s', device)
    if accelerator is not None:
        logger.info('Accelerate enabled: processes=%d grad_accum=%d mixed_precision=%s', accelerator.num_processes, accelerator.gradient_accumulation_steps, accelerator.mixed_precision)
        if args.device:
            logger.warning('Ignoring --device because accelerate controls device placement')

    da3_model, load_info = load_da3_with_scene_flow_head(
        repo_root=DA3_ROOT,
        weights_dir=config['model']['weights_dir'],
        scene_flow_preset=config['model'].get('scene_flow_preset', 'da3-giant-sceneflow'),
        checkpoint_name=config['model'].get('checkpoint_name', 'model.safetensors'),
        config_name=config['model'].get('config_name', 'config.json'),
        device=device,
    )
    logger.info('Loaded DA3 config from %s', load_info['config_path'])
    logger.info('Loaded DA3 checkpoint from %s', load_info['checkpoint_path'])
    logger.info('DA3 load_state_dict missing=%d unexpected=%d', len(load_info['missing_keys']), len(load_info['unexpected_keys']))

    trainable_head = freeze_all_but_scene_flow_head(da3_model)
    da3_interface = DA3StreamingInterface(da3_model=da3_model, freeze_encoder=True, freeze_decoder=False)
    da3_model.eval()
    trainable_head.train()

    init_scene_flow_head_ckpt = resolve_init_scene_flow_head_ckpt(
        args.init_scene_flow_head_ckpt,
        config,
        DA3_ROOT,
    )

    total_params = count_parameters(da3_model.parameters())
    trainable_params = count_parameters(param for param in da3_model.parameters() if param.requires_grad)
    logger.info('DA3 params: total=%s trainable=%s', f'{total_params:,}', f'{trainable_params:,}')

    train_dataset, train_mix_info = build_scene_flow_train_components(config['data'])
    batch_size = int(config['data'].get('batch_size', 1))
    num_workers = int(config['data'].get('num_workers', 0) if args.num_workers is None else args.num_workers)
    pin_memory = bool(config['data'].get('pin_memory', True))
    num_replicas = accelerator.num_processes if accelerator is not None else 1

    train_batch_sampler = None
    if isinstance(train_dataset, MixedSceneFlowDataset):
        steps_per_epoch = int(
            config['training'].get('steps_per_epoch')
            or max(1, len(train_dataset) // max(1, batch_size * num_replicas))
        )
        train_batch_sampler = DistributedMixtureBatchSampler(
            dataset_sizes=train_mix_info['dataset_sizes'],
            dataset_offsets=train_mix_info['dataset_offsets'],
            batch_size=batch_size,
            dataset_probs=train_mix_info['dataset_probs'],
            steps_per_epoch=steps_per_epoch,
            seed=int(config.get('seed', 42)),
            num_replicas=num_replicas,
            rank=process_index,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    val_loaders = build_scene_flow_val_loaders(config['data'], num_workers_override=args.num_workers)

    logger.info(
        'Train data mix: %s',
        ', '.join(
            f'{name}(n={size}, p={prob:.4f})'
            for name, size, prob in zip(
                train_mix_info['dataset_names'],
                train_mix_info['dataset_sizes'],
                train_mix_info['dataset_probs'],
            )
        ),
    )
    logger.info('Validation datasets: %s', ', '.join(val_loaders.keys()) if val_loaders else 'none')

    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is None:
        maybe_initialize_scene_flow_head_from_checkpoint(
            da3_model,
            init_scene_flow_head_ckpt,
            logger,
            skip_if_missing=True,
        )

    optimizer = torch.optim.AdamW(
        trainable_head.parameters(),
        lr=float(config['optimizer']['learning_rate']),
        weight_decay=float(config['optimizer'].get('weight_decay', 0.0)),
        betas=tuple(config['optimizer'].get('betas', [0.9, 0.999])),
    )

    max_steps = int(config['training'].get('max_steps') or (len(train_loader) * int(config['training']['num_epochs'])))
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=int(config['optimizer'].get('warmup_steps', 0)),
        total_steps=max_steps,
        min_lr_scale=float(config['optimizer'].get('min_lr_scale', 0.1)),
    )
    criterion = SceneFlowLoss(
        loss_type=config['loss'].get('loss_type', 'l1'),
        zero_last_weight=float(config['loss'].get('zero_last_weight', 0.0)),
        scale_invariant=bool(config['loss'].get('scale_invariant', False)),
        depth_warp_weight=float(config['loss'].get('depth_warp_weight', 0.0)),
        depth_warp_loss_type=config['loss'].get('depth_warp_loss_type', 'l1'),
        depth_warp_space=config['loss'].get('depth_warp_space', 'metric'),
    )

    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    resume_from_state_dir = accelerator is not None and resume_path is not None and resume_path.is_dir()
    if resume_path and not resume_from_state_dir:
        resume_info = load_file_resume_checkpoint(
            str(resume_path),
            da3_model,
            optimizer,
            scheduler,
            logger,
            load_optimizer_state=accelerator is None,
        )
        start_epoch = resume_info['epoch']
        start_step_in_epoch = resume_info.get('step_in_epoch', 0)
        global_step = resume_info['global_step']

    train_epoch_sampler = train_batch_sampler
    if accelerator is not None:
        accelerator.register_for_checkpointing(scheduler)
        val_loader_names = list(val_loaders.keys())
        val_loader_list = [val_loaders[name] for name in val_loader_names]
        if train_batch_sampler is None:
            prepared = accelerator.prepare(
                trainable_head,
                optimizer,
                train_loader,
                *val_loader_list,
            )
            trainable_head = prepared[0]
            optimizer = prepared[1]
            train_loader = prepared[2]
            prepared_val_loaders = prepared[3:]
        else:
            # DistributedMixtureBatchSampler already shards global batches by rank.
            # Passing this loader into accelerate.prepare would shard it again.
            prepared = accelerator.prepare(
                trainable_head,
                optimizer,
                *val_loader_list,
            )
            trainable_head = prepared[0]
            optimizer = prepared[1]
            prepared_val_loaders = prepared[2:]
            if is_main_process:
                logger.info('Skipping accelerate dataloader sharding for train_loader (DistributedMixtureBatchSampler is already rank-aware)')
        val_loaders = {
            name: loader
            for name, loader in zip(val_loader_names, prepared_val_loaders)
        }
        set_scene_flow_head_module(da3_model, trainable_head)
        if resume_from_state_dir and resume_path is not None:
            resume_info = load_state_checkpoint(resume_path, accelerator, logger)
            start_epoch = resume_info['epoch']
            start_step_in_epoch = resume_info.get('step_in_epoch', 0)
            global_step = resume_info['global_step']

    if is_main_process:
        (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'resolved_config.json', 'w', encoding='utf-8') as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    best_val = float('inf')
    stop_after_steps = int(config['training'].get('max_steps') or 0)
    decoder_chunk_size = int(config['model'].get('decoder_chunk_size', 8))
    grad_accumulation_steps = accelerator.gradient_accumulation_steps if accelerator is not None else 1
    last_progress = {
        'epoch': start_epoch,
        'step_in_epoch': start_step_in_epoch,
    }

    for epoch in range(start_epoch, int(config['training']['num_epochs'])):
        if train_epoch_sampler is not None:
            train_epoch_sampler.set_epoch(epoch)
        else:
            sampler = getattr(train_loader, 'sampler', None)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
        da3_model.eval()
        trainable_head.train()
        running = {
            'loss': 0.0,
            'epe': 0.0,
            'valid_ratio': 0.0,
            'last_frame_zero_epe': 0.0,
            'loss_depth_warp': 0.0,
            'depth_warp_abs': 0.0,
            'depth_warp_valid_ratio': 0.0,
            'scene_flow_scale': 0.0,
            'flow_p95_scale': 0.0,
        }
        running_by_dataset = defaultdict(
            lambda: {
                'loss': 0.0,
                'epe': 0.0,
                'valid_ratio': 0.0,
                'last_frame_zero_epe': 0.0,
                'loss_depth_warp': 0.0,
                'depth_warp_abs': 0.0,
                'depth_warp_valid_ratio': 0.0,
                'scene_flow_scale': 0.0,
                'flow_p95_scale': 0.0,
                'num_batches': 0,
            }
        )
        num_batches = 0
        completed_steps_in_epoch = start_step_in_epoch if epoch == start_epoch else 0
        resume_batches_to_skip = (
            compute_resume_batches_to_skip(start_step_in_epoch, grad_accumulation_steps)
            if epoch == start_epoch else 0
        )
        total_train_batches = len(train_loader)
        if resume_batches_to_skip:
            logger.info(
                'Resuming epoch=%d by skipping %d batches (%d completed optimizer steps in this epoch)',
                epoch,
                resume_batches_to_skip,
                start_step_in_epoch,
            )

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < resume_batches_to_skip:
                continue
            batch_dataset_name = get_batch_dataset_name(batch)
            images = batch['rgbs'].permute(0, 1, 4, 2, 3).to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            target_flow = batch['scene_flow'].permute(0, 1, 4, 2, 3).to(device)
            deform_mask = batch['deform_mask'].to(device)
            height, width = images.shape[-2:]
            loss_kwargs = build_scene_flow_loss_kwargs(batch, device, intrinsics, extrinsics)

            if accelerator is None:
                optimizer.zero_grad(set_to_none=True)
                features = da3_interface.extract_multilayer_features(images, normalize=True, ref_view_strategy='first')
                head_dtype = next(trainable_head.parameters()).dtype
                features = features.to(dtype=head_dtype)
                pred_flow = da3_interface.decode_features_to_scene_flow(
                    features=features,
                    image_size=(height, width),
                    extrinsics=extrinsics,
                    use_gradient=True,
                    chunk_size=decoder_chunk_size,
                    output_basis='frame0',
                )
                loss_dict = criterion(pred_flow, target_flow, deform_mask, **loss_kwargs)
                loss = loss_dict['loss']
                loss.backward()
                if config['optimizer'].get('max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(trainable_head.parameters(), float(config['optimizer']['max_grad_norm']))
                optimizer.step()
                scheduler.step()
                step_completed = True
            else:
                with accelerator.accumulate(trainable_head):
                    features = da3_interface.extract_multilayer_features(images, normalize=True, ref_view_strategy='first')
                    head_dtype = next(trainable_head.parameters()).dtype
                    features = features.to(dtype=head_dtype)
                    pred_flow = da3_interface.decode_features_to_scene_flow(
                        features=features,
                        image_size=(height, width),
                        extrinsics=extrinsics,
                        use_gradient=True,
                        chunk_size=decoder_chunk_size,
                        output_basis='frame0',
                    )
                    loss_dict = criterion(pred_flow, target_flow, deform_mask, **loss_kwargs)
                    loss = loss_dict['loss']
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and config['optimizer'].get('max_grad_norm'):
                        accelerator.clip_grad_norm_(trainable_head.parameters(), float(config['optimizer']['max_grad_norm']))
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                step_completed = accelerator.sync_gradients

            reduced_loss_dict = reduce_loss_dict(loss_dict, accelerator)
            reduced_loss_dict['scene_flow_scale'] = reduce_batch_scalar(batch.get('scene_flow_scale'), accelerator)
            reduced_loss_dict['flow_p95_scale'] = reduce_batch_scalar(
                batch.get('flow_p95_scale', batch.get('scene_flow_scale')),
                accelerator,
            )
            num_batches += 1
            for key in running:
                running[key] += reduced_loss_dict[key]
                running_by_dataset[batch_dataset_name][key] += reduced_loss_dict[key]
            running_by_dataset[batch_dataset_name]['num_batches'] += 1

            if step_completed:
                global_step += 1
                completed_steps_in_epoch += 1

                if should_save_step_checkpoint(global_step, config['training']):
                    progress = get_checkpoint_progress(
                        epoch=epoch,
                        completed_steps_in_epoch=completed_steps_in_epoch,
                        epoch_completed=(batch_idx + 1) >= total_train_batches,
                    )
                    save_checkpoint(
                        build_periodic_checkpoint_path(output_dir, global_step, accelerator=accelerator),
                        da3_model,
                        optimizer,
                        scheduler,
                        progress['epoch'],
                        progress['step_in_epoch'],
                        global_step,
                        config,
                        accelerator=accelerator,
                    )

            if step_completed and global_step % int(config['training'].get('log_every_n_steps', 10)) == 0:
                logger.info(
                    'epoch=%d step=%d dataset=%s loss=%.6f epe=%.6f valid_ratio=%.6f last_zero=%.6f depth_warp=%.6f depth_warp_abs=%.6f scale=%.6f flow_p95=%.6f lr=%.6e',
                    epoch,
                    global_step,
                    batch_dataset_name,
                    running['loss'] / num_batches,
                    running['epe'] / num_batches,
                    running['valid_ratio'] / num_batches,
                    running['last_frame_zero_epe'] / num_batches,
                    running['loss_depth_warp'] / num_batches,
                    running['depth_warp_abs'] / num_batches,
                    running['scene_flow_scale'] / num_batches,
                    running['flow_p95_scale'] / num_batches,
                    optimizer.param_groups[0]['lr'],
                )

            if stop_after_steps and global_step >= stop_after_steps:
                break

        if num_batches == 0:
            if resume_batches_to_skip >= total_train_batches and total_train_batches > 0:
                last_progress = get_checkpoint_progress(
                    epoch=epoch,
                    completed_steps_in_epoch=completed_steps_in_epoch,
                    epoch_completed=True,
                )
                continue
            logger.warning('No training batches were produced for epoch %d', epoch)
            continue

        epoch_completed = (resume_batches_to_skip + num_batches) >= total_train_batches
        progress = get_checkpoint_progress(
            epoch=epoch,
            completed_steps_in_epoch=completed_steps_in_epoch,
            epoch_completed=epoch_completed,
        )
        last_progress = progress

        for dataset_name, stats in sorted(running_by_dataset.items()):
            dataset_batches = max(1, int(stats['num_batches']))
            logger.info(
                'train epoch=%d dataset=%s loss=%.6f epe=%.6f valid_ratio=%.6f last_zero=%.6f depth_warp=%.6f depth_warp_abs=%.6f scale=%.6f flow_p95=%.6f',
                epoch,
                dataset_name,
                stats['loss'] / dataset_batches,
                stats['epe'] / dataset_batches,
                stats['valid_ratio'] / dataset_batches,
                stats['last_frame_zero_epe'] / dataset_batches,
                stats['loss_depth_warp'] / dataset_batches,
                stats['depth_warp_abs'] / dataset_batches,
                stats['scene_flow_scale'] / dataset_batches,
                stats['flow_p95_scale'] / dataset_batches,
            )

        if (epoch + 1) % int(config['training'].get('eval_every_n_epochs', 1)) == 0 and val_loaders:
            val_losses = []
            for val_name, val_loader in val_loaders.items():
                val_metrics = evaluate(
                    da3_interface=da3_interface,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=device,
                    decoder_chunk_size=decoder_chunk_size,
                    accelerator=accelerator,
                )
                val_losses.append(val_metrics['loss'])
                logger.info(
                    'val epoch=%d dataset=%s loss=%.6f epe=%.6f valid_ratio=%.6f last_zero=%.6f depth_warp=%.6f depth_warp_abs=%.6f',
                    epoch,
                    val_name,
                    val_metrics['loss'],
                    val_metrics['epe'],
                    val_metrics['valid_ratio'],
                    val_metrics['last_frame_zero_epe'],
                    val_metrics['loss_depth_warp'],
                    val_metrics['depth_warp_abs'],
                )
            mean_val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
            if mean_val_loss < best_val:
                best_val = mean_val_loss
                save_checkpoint(
                    output_dir / 'checkpoints' / ('best' if accelerator is not None else 'best.pt'),
                    da3_model,
                    optimizer,
                    scheduler,
                    progress['epoch'],
                    progress['step_in_epoch'],
                    global_step,
                    config,
                    accelerator=accelerator,
                )

        if stop_after_steps and global_step >= stop_after_steps:
            break

    save_checkpoint(
        output_dir / 'checkpoints' / ('last' if accelerator is not None else 'last.pt'),
        da3_model,
        optimizer,
        scheduler,
        last_progress['epoch'],
        last_progress['step_in_epoch'],
        global_step,
        config,
        accelerator=accelerator,
    )
    logger.info('Training finished at global_step=%d', global_step)


if __name__ == '__main__':
    main()
