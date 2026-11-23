from __future__ import annotations

from collections import Counter
from typing import Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class DistributedMixtureBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset_sizes: Sequence[int],
        dataset_offsets: Sequence[int],
        batch_size: int,
        dataset_probs: Sequence[float],
        steps_per_epoch: int,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if len(dataset_sizes) != len(dataset_offsets) or len(dataset_sizes) != len(dataset_probs):
            raise ValueError("dataset_sizes, dataset_offsets, and dataset_probs must have the same length")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if steps_per_epoch <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset_sizes = [int(size) for size in dataset_sizes]
        self.dataset_offsets = [int(offset) for offset in dataset_offsets]
        self.batch_size = int(batch_size)
        self.dataset_probs = np.asarray(dataset_probs, dtype=np.float64)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0

        if np.any(np.asarray(self.dataset_sizes) <= 0):
            raise ValueError(f"dataset_sizes must all be positive, got {dataset_sizes}")
        if np.any(self.dataset_probs < 0):
            raise ValueError(f"dataset_probs must be non-negative, got {dataset_probs}")
        probs_sum = float(self.dataset_probs.sum())
        if probs_sum <= 0.0:
            raise ValueError(f"dataset_probs must sum to a positive value, got {dataset_probs}")
        self.dataset_probs = self.dataset_probs / probs_sum

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        probs = torch.tensor(self.dataset_probs, dtype=torch.float64)
        global_batch = self.batch_size * self.num_replicas
        step_dataset_ids = torch.multinomial(
            probs,
            num_samples=self.steps_per_epoch,
            replacement=True,
            generator=generator,
        ).tolist()
        requested_batches = Counter(int(dataset_id) for dataset_id in step_dataset_ids)

        dataset_pools: dict[int, List[int]] = {}
        dataset_positions: dict[int, int] = {}
        for dataset_id, num_batches in requested_batches.items():
            needed = int(num_batches) * global_batch
            local_indices: List[int] = []
            while len(local_indices) < needed:
                local_indices.extend(
                    torch.randperm(self.dataset_sizes[dataset_id], generator=generator).tolist()
                )
            dataset_pools[dataset_id] = local_indices
            dataset_positions[dataset_id] = 0

        for dataset_id in step_dataset_ids:
            start_pos = dataset_positions[dataset_id]
            end_pos = start_pos + global_batch
            local_indices = dataset_pools[dataset_id][start_pos:end_pos]
            dataset_positions[dataset_id] = end_pos
            global_indices = [self.dataset_offsets[dataset_id] + local_idx for local_idx in local_indices]
            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield global_indices[start:end]
