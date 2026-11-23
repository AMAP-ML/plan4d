"""
DL3DV Dataset Dataloader

Expected Folder Structure:

<root_dir>/
├── 1K/
│   └── <scene_name>/
│       ├── images_4/
│       │   ├── <image_000.png>
│       │   ├── <image_001.png>
│       │   └── ...
│       └── transforms.json
├── 2K/
├── 3K/
│   ...
└── 11K/

- <root_dir>: Root directory containing different resolution folders (1K to 11K)
- <scene_name>: Specific scene folder under each resolution
- images_4/: Directory containing 8 images per scene
- transforms.json: Contains camera parameters and metadata for the scene
"""

import json
import os
import random
import re
import traceback
from glob import glob
from pathlib import Path

import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from data.image_preprocessing import load_image, resize_shorter_crop_square_batch


class DL3DVStitchingDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        num_images_from_unit_scene: int,
        num_frames_per_unit_scene: int,
        image_resolution: int = 512,
        feedforward_image_resolution: int = 448,
        color_augmentation: bool = True,
    ):
        super(DL3DVStitchingDataset, self).__init__()
        self.root_path = root_path
        self.num_images_from_unit_scene = num_images_from_unit_scene
        self.num_frames_per_unit_scene = num_frames_per_unit_scene
        self.image_resolution = image_resolution
        self.feedforward_image_resolution = feedforward_image_resolution
        self.color_augmentation = color_augmentation
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if self.color_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
            )

        """crawl scene lists"""
        self.sequences = sorted(glob(str(root_path) + "/*/*/"))  # total 9286 sequences

    def __len__(self):
        return len(self.sequences)

    def get_scene_images(
        self,
        image_scene_path,
        scene_start_index,
        scene_end_index,
        image_index,
    ):
        all_images_paths = sorted(
            f
            for f in os.listdir(image_scene_path / "images_4")
            if re.match(r"^frame_\d+", f)
        )
        all_images_paths = all_images_paths[scene_start_index:scene_end_index]
        file_paths = [
            image_scene_path / "images_4" / f_info for f_info in all_images_paths
        ]

        # load image
        vae_imgs = []
        for i in image_index:
            vae_imgs.append(load_image(file_paths[i], self.transforms))
        vae_tensor = torch.stack(vae_imgs)

        # processing image (resize + color augmentation)
        vae_tensor = resize_shorter_crop_square_batch(
            vae_tensor, target_size=self.image_resolution
        )
        if self.color_augmentation:
            vae_tensor = self.color_jitter(vae_tensor)

        # We suppose that feedforward model and video vae have different image resolution
        feedforward_imgs = [
            transforms.Resize(
                (self.feedforward_image_resolution, self.feedforward_image_resolution)
            )(img)
            for img in vae_tensor
        ]

        # normalize to [-1, 1]
        vae_tensor = (
            rearrange(vae_tensor, "t c h w -> c t h w").unsqueeze(0).float() * 2 - 1
        )
        feedforward_tensor = (
            rearrange(torch.stack(feedforward_imgs), "t c h w -> c t h w")
            .unsqueeze(0)
            .float()
            * 2
            - 1
        )
        return vae_tensor, feedforward_tensor

    def get_item_sub(self, idx):
        sequence_path = Path(self.sequences[idx])

        try:
            num_frames = len(
                sorted(
                    f
                    for f in os.listdir(sequence_path / "images_4")
                    if re.match(r"^frame_\d+", f)
                )
            )

            # Sampling strategy:
            # 1) Choose a random window length in [num_images_from_unit_scene, num_frames_per_unit_scene].
            # 2) Randomly choose a start index for that window.
            # 3) Randomly pick num_images_from_unit_scene frames within the window.
            frames_per_scene = random.randint(
                self.num_images_from_unit_scene, self.num_frames_per_unit_scene
            )
            # 2. randomly select a start index according to frames_per_scene
            start = random.randint(0, num_frames - frames_per_scene - 1)
            # 3. determine end index
            end = start + frames_per_scene
            # 4. randomly sample num_images_from_unit_scene indices from the selected frames_per_scene
            image_index = random.sample(
                range(1, frames_per_scene), self.num_images_from_unit_scene - 1
            )
            # 5. add the first frame index and sort
            image_index.append(0)
            image_index.sort()
            vae_image_tensor, feedforward_image_tensor = self.get_scene_images(
                sequence_path, start, end, image_index
            )

        except Exception as e:
            print(f"Error reading sequence at {sequence_path}: {e}")
            raise

        return dict(
            # remove batch dim( c t h w)
            vae_image_tensor=vae_image_tensor[0],
            feedforward_image_tensor=feedforward_image_tensor[0],
            dataset_name="dl3dv",
            scene_name=sequence_path.name,
            clip_uid=f"{sequence_path.name}:{start}:{end}",
        )

    def __getitem__(self, idx):
        # since we found some sequences have issues, we try to load another sequence when error occurs
        try:
            return self.get_item_sub(idx)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            traceback.print_exc()
            while True:
                idx = random.randint(0, len(self.sequences) - 1)
                try:
                    return self.get_item_sub(idx)
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    traceback.print_exc()
                    continue


