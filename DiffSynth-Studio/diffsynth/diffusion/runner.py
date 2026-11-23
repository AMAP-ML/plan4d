import math
import json
import os
from pathlib import Path

import torch
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger, load_trainer_state


class StageAlignedSampler(torch.utils.data.Sampler[int]):
    def __init__(self, dataset: torch.utils.data.Dataset, *, num_processes: int, seed: int = 22):
        self.dataset = dataset
        self.num_processes = int(max(1, num_processes))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _ordered_indices(self):
        return self.dataset.build_stage_aligned_order(
            num_processes=self.num_processes,
            epoch=self.epoch,
            seed=self.seed,
        )

    def __iter__(self):
        return iter(self._ordered_indices())

    def __len__(self):
        return len(self._ordered_indices())


def _to_metric_tensor(value, device):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.detach().to(device=device, dtype=torch.float32)
    if isinstance(value, (float, int)):
        return torch.tensor(float(value), device=device, dtype=torch.float32)
    return None


def _collect_step_metrics(accelerator: Accelerator, model: torch.nn.Module, loss: torch.Tensor):
    metrics = {}
    loss_tensor = _to_metric_tensor(loss, accelerator.device)
    if loss_tensor is not None:
        metrics["loss"] = loss_tensor

    latest_joint_metrics = getattr(accelerator.unwrap_model(model), "latest_joint_metrics", None)
    if isinstance(latest_joint_metrics, dict):
        for name, value in latest_joint_metrics.items():
            metric_tensor = _to_metric_tensor(value, accelerator.device)
            if metric_tensor is not None and name not in metrics:
                metrics[name] = metric_tensor

    reduced_metrics = {}
    for name, metric_tensor in metrics.items():
        reduced_metrics[name] = accelerator.reduce(metric_tensor, reduction="mean").item()
    return reduced_metrics


def _compute_grad_norm(accelerator: Accelerator, model: torch.nn.Module):
    if not getattr(accelerator, "sync_gradients", True):
        return None

    deepspeed_engine = getattr(accelerator, "deepspeed_engine_wrapped", None)
    if deepspeed_engine is not None:
        grad_norm = deepspeed_engine.get_global_grad_norm()
        return _to_metric_tensor(grad_norm, accelerator.device)

    local_grad_sq = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    has_grad = False
    for parameter in accelerator.unwrap_model(model).parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        local_grad_sq = local_grad_sq + grad.float().pow(2).sum().to(accelerator.device)
        has_grad = True

    if not has_grad:
        return None

    total_grad_sq = accelerator.reduce(local_grad_sq, reduction="sum")
    return total_grad_sq.sqrt()


def _log_step_metrics(accelerator: Accelerator, metrics, epoch_id: int, num_epochs: int, global_step: int, step_in_epoch: int):
    if len(metrics) == 0:
        return
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    accelerator.print(
        f"[train] step={global_step} epoch={epoch_id + 1}/{num_epochs} step_in_epoch={step_in_epoch} {metric_text}"
    )


def _build_trainer_state(global_step: int, epoch_id: int, step_in_epoch: int, model_logger: ModelLogger):
    return {
        "global_step": int(global_step),
        "epoch": int(epoch_id),
        "step_in_epoch": int(step_in_epoch),
        "model_logger_num_steps": int(model_logger.num_steps),
    }


def _set_training_progress(accelerator: Accelerator, model: torch.nn.Module, global_step: int, epoch_id: int, step_in_epoch: int):
    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "set_training_progress"):
        unwrapped_model.set_training_progress(
            global_step=global_step,
            epoch_id=epoch_id,
            step_in_epoch=step_in_epoch,
        )
        return

    setattr(unwrapped_model, "current_global_step", int(global_step))
    setattr(unwrapped_model, "current_epoch_id", int(epoch_id))
    setattr(unwrapped_model, "current_step_in_epoch", int(step_in_epoch))


def _build_training_dataloader(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    *,
    num_workers: int,
):
    sampler = None
    shuffle = True
    if hasattr(dataset, "build_stage_aligned_order"):
        sampler = StageAlignedSampler(
            dataset,
            num_processes=getattr(accelerator, "num_processes", 1),
        )
        shuffle = False

    dataloader_kwargs = {
        "dataset": dataset,
        "shuffle": shuffle,
        "collate_fn": lambda x: x[0],
        "num_workers": num_workers,
    }
    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)
    return dataloader, sampler


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    *,
    lr_scheduler: str,
    warmup_steps: int,
    num_epochs: int,
):
    gradient_accumulation_steps = int(max(1, getattr(accelerator, "gradient_accumulation_steps", 1)))
    updates_per_epoch = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps))
    num_training_steps = max(1, int(num_epochs) * updates_per_epoch)
    return transformers.get_scheduler(
        name=lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=max(0, int(warmup_steps)),
        num_training_steps=num_training_steps,
    )


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args=None,
):
    resume_from_checkpoint = None
    warmup_steps = 0
    lr_scheduler = "cosine"
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
        warmup_steps = getattr(args, "warmup_steps", 0)
        lr_scheduler = getattr(args, "lr_scheduler", "cosine")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    dataloader, stage_aligned_sampler = _build_training_dataloader(
        accelerator,
        dataset,
        num_workers=num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    scheduler = _build_lr_scheduler(
        optimizer,
        dataloader,
        accelerator,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
    )
    if hasattr(accelerator, "register_for_checkpointing"):
        accelerator.register_for_checkpointing(scheduler)

    start_epoch = 0
    resume_step_in_epoch = 0
    global_step = 0
    if resume_from_checkpoint is not None:
        accelerator.load_state(resume_from_checkpoint)
        trainer_state = load_trainer_state(resume_from_checkpoint)
        global_step = int(trainer_state.get("global_step", 0))
        start_epoch = int(trainer_state.get("epoch", 0))
        resume_step_in_epoch = int(trainer_state.get("step_in_epoch", 0))
        model_logger.num_steps = int(trainer_state.get("model_logger_num_steps", global_step))
    else:
        model_logger.num_steps = 0

    initialize_deepspeed_gradient_checkpointing(accelerator, model)

    for epoch_id in range(start_epoch, num_epochs):
        if stage_aligned_sampler is not None:
            stage_aligned_sampler.set_epoch(epoch_id)
        step_offset = resume_step_in_epoch if epoch_id == start_epoch else 0
        epoch_dataloader = dataloader
        skipped_by_loader = 0
        manual_step_offset = step_offset
        if step_offset > 0 and hasattr(accelerator, "skip_first_batches"):
            epoch_dataloader = accelerator.skip_first_batches(dataloader, num_batches=step_offset)
            skipped_by_loader = step_offset
            manual_step_offset = 0

        progress_kwargs = {
            "disable": not getattr(accelerator, "is_local_main_process", True),
        }
        if skipped_by_loader > 0:
            progress_kwargs["initial"] = skipped_by_loader
            try:
                progress_kwargs["total"] = len(dataloader)
            except TypeError:
                pass

        progress = tqdm(epoch_dataloader, **progress_kwargs)
        for batch_id, data in enumerate(progress):
            if manual_step_offset > 0 and batch_id < manual_step_offset:
                continue
            step_in_epoch = skipped_by_loader + batch_id + 1
            _set_training_progress(accelerator, model, global_step, epoch_id, step_in_epoch)
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                grad_norm = _compute_grad_norm(accelerator, model)
                optimizer.step()
                if getattr(accelerator, "sync_gradients", True):
                    scheduler.step()

            global_step += 1
            model_logger.num_steps = global_step
            step_metrics = _collect_step_metrics(accelerator, model, loss)
            grad_norm_tensor = _to_metric_tensor(grad_norm, accelerator.device)
            if grad_norm_tensor is not None:
                step_metrics["grad_norm"] = accelerator.reduce(grad_norm_tensor, reduction="mean").item()
            _log_step_metrics(accelerator, step_metrics, epoch_id, num_epochs, global_step, step_in_epoch)
            step_trainer_state = _build_trainer_state(global_step, epoch_id, step_in_epoch, model_logger)
            model_logger.on_step_end(
                accelerator,
                model,
                save_steps,
                global_step=global_step,
                trainer_state=step_trainer_state,
                loss=loss,
            )
        resume_step_in_epoch = 0
        if save_steps is None:
            epoch_trainer_state = _build_trainer_state(global_step, epoch_id + 1, 0, model_logger)
            model_logger.on_epoch_end(accelerator, model, epoch_id, trainer_state=epoch_trainer_state)
    final_trainer_state = _build_trainer_state(global_step, num_epochs, 0, model_logger)
    model_logger.on_training_end(accelerator, model, save_steps, trainer_state=final_trainer_state)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)


def _resolve_activation_checkpointing_config(ds_config):
    if not isinstance(ds_config, dict):
        return None
    if "activation_checkpointing" in ds_config:
        return ds_config["activation_checkpointing"]

    file_config = _resolve_deepspeed_file_config(ds_config)
    if file_config is None:
        return None
    return file_config.get("activation_checkpointing")


def _resolve_deepspeed_file_config(ds_config):
    if not isinstance(ds_config, dict):
        return None

    config_file = ds_config.get("deepspeed_config_file")
    if not config_file:
        return None

    try:
        file_config = json.loads(Path(config_file).read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Failed to read DeepSpeed config file {config_file}: {error}")
        return None

    if not isinstance(file_config, dict):
        return None
    return file_config


def _resolve_zero_stage(ds_config):
    if not isinstance(ds_config, dict):
        return None

    zero_stage = ds_config.get("zero_stage")
    if zero_stage is not None:
        try:
            return int(zero_stage)
        except (TypeError, ValueError):
            return None

    zero_optimization = ds_config.get("zero_optimization")
    if isinstance(zero_optimization, dict):
        zero_stage = zero_optimization.get("stage")
        if zero_stage is not None:
            try:
                return int(zero_stage)
            except (TypeError, ValueError):
                return None

    file_config = _resolve_deepspeed_file_config(ds_config)
    if file_config is None:
        return None

    zero_optimization = file_config.get("zero_optimization")
    if not isinstance(zero_optimization, dict):
        return None

    zero_stage = zero_optimization.get("stage")
    if zero_stage is None:
        return None
    try:
        return int(zero_stage)
    except (TypeError, ValueError):
        return None


def _infer_activation_checkpointing_config(ds_config, model):
    if _resolve_zero_stage(ds_config) != 3 or model is None:
        return None

    if not getattr(model, "use_gradient_checkpointing", False):
        return None

    return {
        "partition_activations": True,
        "cpu_checkpointing": bool(getattr(model, "use_gradient_checkpointing_offload", False)),
        "contiguous_memory_optimization": False,
    }


def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator, model: torch.nn.Module = None):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        act_config = _resolve_activation_checkpointing_config(ds_config)
        if act_config is None and model is not None:
            act_config = _infer_activation_checkpointing_config(ds_config, accelerator.unwrap_model(model))
            if act_config is not None:
                print("Using inferred DeepSpeed activation checkpointing config for inline ZeRO-3 setup.")
        if act_config is not None:
            import deepspeed
            checkpointing = deepspeed.checkpointing
            if hasattr(checkpointing, "is_configured") and checkpointing.is_configured():
                return
            checkpointing.configure(
                mpu_=None,
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
