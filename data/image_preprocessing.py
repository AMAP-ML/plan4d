import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def load_image(image_path: str, transform: torchvision.transforms.Compose):
    with Image.open(image_path) as handle:
        rgb = handle.convert("RGB")
    return transform(rgb)


def resize_shorter_crop_square_batch(
    images: torch.Tensor,
    target_size: int = 448,
) -> torch.Tensor:
    """Turn a (T, C, H, W) batch into (T, C, S, S): first an isotropic
    bilinear resize that maps the shorter side to `target_size`, then a
    centre crop along the remaining axis."""
    height, width = images.shape[-2:]
    scale = target_size / min(height, width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)

    for axis, length in ((2, new_h), (3, new_w)):
        if length > target_size:
            start = (length - target_size) // 2
            images = images.narrow(axis, start, target_size)

    assert images.shape[-2:] == torch.Size((target_size, target_size)), "logic error"
    return images
