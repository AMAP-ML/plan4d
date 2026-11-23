import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from data.dl3dv_datasets import DL3DVStitchingDataset
from data.point_odyssey_stitching_dataset import PointOdysseyStitchingDataset
from data.scannet_unlabeled_dataset import ScannetUnlabeledDataaset
from data.stitching_manifest import build_manifest_training_dataset


def parse_point_odyssey_splits(split_spec: str | None) -> tuple[str, ...]:
    if split_spec is None:
        return ("train",)

    splits = tuple(split.strip() for split in split_spec.split(",") if split.strip())
    if not splits:
        raise ValueError("point_odyssey_splits must contain at least one non-empty split name.")
    return splits


def _build_single_stitching_dataset(
    name: str,
    root_path: str,
    num_images_from_unit_scene: int,
    num_frames_per_unit_scene: int,
    image_resolution: int,
    feedforward_image_resolution: int,
    augmentation: bool = False,
    point_odyssey_split: str = "train",
    da3_process_res_method: str = "upper_bound_resize",
):
    # Map dataset names to loader classes and expected directory layouts.
    # Add new datasets here and in the README.
    if name == "dl3dv":
        return DL3DVStitchingDataset(
            root_path=root_path,
            num_images_from_unit_scene=num_images_from_unit_scene,
            num_frames_per_unit_scene=num_frames_per_unit_scene,
            image_resolution=image_resolution,
            feedforward_image_resolution=feedforward_image_resolution,
            color_augmentation=augmentation,
        )
    elif name == "scannet":
        return ScannetUnlabeledDataaset(
            root_path=root_path,
            num_images_from_unit_scene=num_images_from_unit_scene,
            num_frames_per_unit_scene=num_frames_per_unit_scene,
            image_resolution=image_resolution,
            feedforward_image_resolution=feedforward_image_resolution,
            color_augmentation=augmentation,
        )
    elif name == "point_odyssey":
        return PointOdysseyStitchingDataset(
            root_path=root_path,
            split=point_odyssey_split,
            image_resolution=image_resolution,
            feedforward_image_resolution=feedforward_image_resolution,
            chunk_size=num_frames_per_unit_scene,
            overlap=1,
            process_res_method=da3_process_res_method,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def _describe_sample_feedforward_shape(dataset) -> str:
    if len(dataset) == 0:
        return "feedforward shape unavailable"
    sample = dataset[0]
    feedforward_tensor = None
    if isinstance(sample, dict):
        feedforward_tensor = sample.get("feedforward_image_tensor")
        if feedforward_tensor is None:
            feedforward_tensor = sample.get("teacher_feedforward_image_tensor")
    if torch.is_tensor(feedforward_tensor) and feedforward_tensor.ndim >= 2:
        height = int(feedforward_tensor.shape[-2])
        width = int(feedforward_tensor.shape[-1])
        return f"feedforward sample[0] {height}×{width}"
    return "feedforward shape unavailable"


def _create_loader_from_dataset(
    train_set,
    *,
    args: argparse.Namespace,
    use_distributed_sampler: bool,
):
    sampler = None
    if (
        use_distributed_sampler
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        sampler = DistributedSampler(
            train_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=22,
        )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    sample_shape_summary = _describe_sample_feedforward_shape(train_set)
    logger.info(
        f"DataLoader has {len(train_set)} samples; batch size {args.batch_size}, "
        f"{sample_shape_summary}"
    )
    return loader, sampler


def _resolve_distill_manifest_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    manifest_specs = list(getattr(args, "train_manifest", None) or [])
    action100m_manifest = getattr(args, "action100m_manifest", None)
    if action100m_manifest:
        manifest_specs.append(("action100m", Path(action100m_manifest).expanduser()))
    return manifest_specs


def _manifest_dataset_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "manifest_data_root": args.manifest_data_root,
        "image_resolution": args.resolution,
        "feedforward_image_resolution": args.feedforward_resolution,
        "chunk_size": args.num_frames_per_unit_scene,
        "process_res_method": getattr(args, "da3_process_res_method", "upper_bound_resize"),
        "samples_per_manifest": getattr(args, "samples_per_manifest", None),
        "scannetpp_local_root": getattr(args, "scannetpp_local_root", None),
    }


def create_hybrid_stitching_dataloaders(
    args: argparse.Namespace,
    *,
    use_distributed_sampler: bool = True,
):
    manifest_specs = getattr(args, "train_manifest", None) or []
    if not manifest_specs:
        raise ValueError("--training_stage hybrid requires --train_manifest inputs.")
    if getattr(args, "manifest_data_root", None) is None:
        raise ValueError("--manifest_data_root is required when --train_manifest is used.")

    gt_set = build_manifest_training_dataset(
        manifest_specs,
        training_stage="gt",
        **_manifest_dataset_kwargs(args),
    )
    distill_set = build_manifest_training_dataset(
        _resolve_distill_manifest_specs(args),
        training_stage="distill",
        **_manifest_dataset_kwargs(args),
    )
    gt_loader, gt_sampler = _create_loader_from_dataset(
        gt_set,
        args=args,
        use_distributed_sampler=use_distributed_sampler,
    )
    distill_loader, distill_sampler = _create_loader_from_dataset(
        distill_set,
        args=args,
        use_distributed_sampler=use_distributed_sampler,
    )
    return {"gt": gt_loader, "distill": distill_loader}, {"gt": gt_sampler, "distill": distill_sampler}


def create_stitching_dataloader(
    datasets: list[tuple[str, str]] | None,  # <── list of (name, root) pairs
    args: argparse.Namespace,
    augmentation: bool = False,
    use_distributed_sampler: bool = True,
):
    logger.info("Creating dataloader")
    manifest_specs = getattr(args, "train_manifest", None) or []
    if manifest_specs:
        training_stage = getattr(args, "training_stage", None) or "distill"
        logger.info("  Manifest mode enabled")
        for name, manifest_path in manifest_specs:
            logger.info(f"  Manifest: {name} at {manifest_path}")
        if getattr(args, "manifest_data_root", None) is None:
            raise ValueError("--manifest_data_root is required when --train_manifest is used.")
        if training_stage == "hybrid":
            raise ValueError("Hybrid mode must use create_hybrid_stitching_dataloaders().")

        train_set = build_manifest_training_dataset(
            _resolve_distill_manifest_specs(args)
            if training_stage == "distill"
            else manifest_specs,
            training_stage=training_stage,
            **_manifest_dataset_kwargs(args),
        )
        ds_objects = [train_set]
    else:
        if not datasets:
            raise ValueError("Either --dataset or --train_manifest must be provided.")
        for n, r in datasets:
            logger.info(f"  Dataset: {n} at {r}")

        ds_objects = []
        point_odyssey_splits = parse_point_odyssey_splits(
            getattr(args, "point_odyssey_splits", "train")
        )
        for n, r in datasets:
            if n == "point_odyssey":
                for split in point_odyssey_splits:
                    logger.info(f"  Point Odyssey split: {split}")
                    ds_objects.append(
                        _build_single_stitching_dataset(
                            name=n,
                            root_path=r,
                            num_images_from_unit_scene=args.num_images_from_unit_scene,
                            num_frames_per_unit_scene=args.num_frames_per_unit_scene,
                            image_resolution=args.resolution,
                            feedforward_image_resolution=args.feedforward_resolution,
                            augmentation=augmentation,
                            point_odyssey_split=split,
                            da3_process_res_method=getattr(
                                args, "da3_process_res_method", "upper_bound_resize"
                            ),
                        )
                    )
            else:
                ds_objects.append(
                    _build_single_stitching_dataset(
                        name=n,
                        root_path=r,
                        num_images_from_unit_scene=args.num_images_from_unit_scene,
                        num_frames_per_unit_scene=args.num_frames_per_unit_scene,
                        image_resolution=args.resolution,
                        feedforward_image_resolution=args.feedforward_resolution,
                        augmentation=augmentation,
                        da3_process_res_method=getattr(
                            args, "da3_process_res_method", "upper_bound_resize"
                        ),
                    )
                )

    train_set = ds_objects[0] if len(ds_objects) == 1 else ConcatDataset(ds_objects)
    loader, sampler = _create_loader_from_dataset(
        train_set,
        args=args,
        use_distributed_sampler=use_distributed_sampler,
    )
    logger.info(f"Combined {len(ds_objects)} dataset(s) into one training loader.")
    return loader, sampler


