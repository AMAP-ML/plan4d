"""
Utility functions for distributed training.
"""

import os
import socket

import torch
import torch.distributed as dist

DEVICE = None


def _find_free_port():
    # Bind to port 0 to let the OS choose a free port for single-node DDP.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def setup_dist():
    if dist.is_initialized():
        return

    if os.environ.get("MASTER_ADDR", None) is None:
        # Default to single-process DDP setup if not launched by torchrun.
        hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    global DEVICE
    DEVICE = device


def is_distributed_launch_requested() -> bool:
    if dist.is_initialized():
        return True

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        return False

    try:
        return int(world_size) > 1
    except ValueError:
        return True


def ensure_cuda_arch_supported(device_index: int) -> None:
    supported_arches = torch.cuda.get_arch_list()
    if not supported_arches:
        return

    major, minor = torch.cuda.get_device_capability(device_index)
    device_arch = f"sm_{major}{minor}"
    if device_arch in supported_arches:
        return

    device_name = torch.cuda.get_device_name(device_index)
    raise RuntimeError(
        "Current PyTorch CUDA build does not support "
        f"{device_name} ({device_arch}). Supported architectures: "
        f"{' '.join(supported_arches)}. "
        "Please switch to a supported GPU or install a PyTorch build compiled for this CUDA arch."
    )


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def device():
    if not dist.is_initialized():
        raise NameError
    return DEVICE


def torch_device_setup():
    assert torch.cuda.is_available(), "We basically require GPUs."
    if is_distributed_launch_requested():
        setup_dist()
        torch_device = torch.device("cuda", device())
        ensure_cuda_arch_supported(torch_device.index)
        dist.barrier(device_ids=[torch_device.index])
        return torch_device

    device_index = torch.cuda.current_device()
    ensure_cuda_arch_supported(device_index)
    global DEVICE
    DEVICE = device_index
    torch_device = torch.device("cuda", device_index)
    return torch_device


def is_main_process() -> bool:
    """Return True on rank‑0 (or non‑distributed run)."""
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )
