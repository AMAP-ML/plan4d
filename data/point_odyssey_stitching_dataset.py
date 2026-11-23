import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.da3_resize import (
    apply_resize_only_to_intrinsics,
    resize_video_frames_upper_bound,
)


def _resolve_plan4d_root() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1],
    )
    for candidate in candidates:
        if (candidate / "plan4d" / "point_odyssey.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate plan4d root containing plan4d/point_odyssey.py"
    )


def _load_point_odyssey_dataset_components():
    plan4d_root = _resolve_plan4d_root()
    if str(plan4d_root) not in sys.path:
        sys.path.insert(0, str(plan4d_root))

    from plan4d.point_odyssey import PointOdysseyDataset, generate_ray_map

    return PointOdysseyDataset, generate_ray_map


def _resize_video_frames(
    frames_01: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    if frames_01.shape[-2:] == (resolution, resolution):
        return frames_01

    frames_resized = F.interpolate(
        frames_01,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )
    return frames_resized


class PointOdysseyStitchingDataset(Dataset):
    """
    Point Odyssey wrapper for PLAN4D stitching.

    - Feedforward branch uses Point Odyssey's geometry-aware resize/crop to the DA3 image size.
    - Wan VAE branch is derived from the same RGB frames and resized to the VAE image size.
    - GT depth/ray/intrinsics/extrinsics stay at feedforward resolution for DA3 GT loss.
    """

    def __init__(
        self,
        root_path: str | Path,
        *,
        split: str = "train",
        image_resolution: int = 532,
        feedforward_image_resolution: int = 532,
        chunk_size: int = 49,
        overlap: int = 1,
        max_scenes: int | None = None,
        process_res_method: str = "square_center_crop",
    ):
        super().__init__()
        point_odyssey_cls, self._generate_ray_map = _load_point_odyssey_dataset_components()
        self.image_resolution = image_resolution
        self.feedforward_image_resolution = feedforward_image_resolution
        self.process_res_method = process_res_method
        target_height = feedforward_image_resolution if process_res_method == "square_center_crop" else None
        target_width = feedforward_image_resolution if process_res_method == "square_center_crop" else None
        load_ray = process_res_method == "square_center_crop"
        self.base_dataset = point_odyssey_cls(
            data_root=str(Path(root_path).expanduser()),
            split=split,
            chunk_size=chunk_size,
            overlap=overlap,
            target_height=target_height,
            target_width=target_width,
            load_depth=True,
            load_ray=load_ray,
            max_scenes=max_scenes,
        )
        self._point_supervision_cache: dict[Path, dict[str, int | bool]] = {}
        self._valid_sample_indices = self._build_valid_sample_indices()

    def __len__(self) -> int:
        return len(self._valid_sample_indices)

    def _get_scene_point_supervision_meta(self, scene_dir: Path) -> dict[str, int | bool]:
        scene_dir = Path(scene_dir)
        cached = self._point_supervision_cache.get(scene_dir)
        if cached is not None:
            return cached

        anno_path = scene_dir / "anno.npz"
        with np.load(anno_path) as anno:
            raw_trajs_3d = anno["trajs_3d"] if "trajs_3d" in anno.files else None
            raw_trajs_2d = anno["trajs_2d"] if "trajs_2d" in anno.files else None
            cached = {
                "valid_trajs_3d": int(
                    isinstance(raw_trajs_3d, np.ndarray)
                    and raw_trajs_3d.ndim == 3
                    and raw_trajs_3d.shape[-1] == 3
                ),
                "valid_trajs_2d": int(
                    isinstance(raw_trajs_2d, np.ndarray)
                    and raw_trajs_2d.ndim == 3
                    and raw_trajs_2d.shape[-1] == 2
                ),
                "trajs_3d_frames": 0 if raw_trajs_3d is None else int(raw_trajs_3d.shape[0]),
                "trajs_2d_frames": 0 if raw_trajs_2d is None else int(raw_trajs_2d.shape[0]),
                "num_points_match": int(
                    isinstance(raw_trajs_3d, np.ndarray)
                    and isinstance(raw_trajs_2d, np.ndarray)
                    and raw_trajs_3d.ndim == 3
                    and raw_trajs_2d.ndim == 3
                    and raw_trajs_3d.shape[1] == raw_trajs_2d.shape[1]
                ),
            }

        self._point_supervision_cache[scene_dir] = cached
        return cached

    def _sample_has_sparse_points(self, sample_info: dict[str, Any]) -> bool:
        meta = self._get_scene_point_supervision_meta(Path(sample_info["scene_dir"]))
        return bool(
            meta["valid_trajs_3d"]
            and meta["valid_trajs_2d"]
            and meta["num_points_match"]
            and meta["trajs_3d_frames"] >= int(sample_info["end_idx"])
            and meta["trajs_2d_frames"] >= int(sample_info["end_idx"])
        )

    def _build_valid_sample_indices(self) -> list[int]:
        valid_indices: list[int] = []
        for sample_idx, sample_info in enumerate(self.base_dataset.samples):
            if self._sample_has_sparse_points(sample_info):
                valid_indices.append(sample_idx)
        return valid_indices

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base_dataset[self._valid_sample_indices[idx]]

        rgbs = sample["rgbs"].float() / 255.0  # [T, H, W, 3]
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()  # [T, 3, H, W]

        if self.process_res_method == "upper_bound_resize":
            feedforward_rgb, _ = resize_video_frames_upper_bound(
                rgbs,
                target_size=self.feedforward_image_resolution,
                mode="bilinear",
            )
            vae_rgb, _ = resize_video_frames_upper_bound(
                rgbs,
                target_size=self.image_resolution,
                mode="bilinear",
            )
            depth = F.interpolate(
                sample["depth"].unsqueeze(1).float(),
                size=feedforward_rgb.shape[-2:],
                mode="nearest",
            )
            valid_mask = F.interpolate(
                sample["valid_mask"].unsqueeze(1).float(),
                size=feedforward_rgb.shape[-2:],
                mode="nearest",
            )
            intrinsics = apply_resize_only_to_intrinsics(
                sample["intrinsics"].float(),
                original_height=int(rgbs.shape[-2]),
                original_width=int(rgbs.shape[-1]),
                resized_height=int(feedforward_rgb.shape[-2]),
                resized_width=int(feedforward_rgb.shape[-1]),
            ).contiguous()
            extrinsics = sample["extrinsics"].float().contiguous()
            rays = []
            for frame_intrinsics, frame_extrinsics in zip(intrinsics, extrinsics):
                rays.append(
                    self._generate_ray_map(
                        frame_intrinsics,
                        frame_extrinsics,
                        int(feedforward_rgb.shape[-2]),
                        int(feedforward_rgb.shape[-1]),
                    )
                )
            ray = torch.stack(rays, dim=0).contiguous()
            points_2d = sample.get("points_2d")
            if points_2d is not None:
                points_2d = points_2d.float().clone()
                points_2d[..., 0] *= feedforward_rgb.shape[-1] / float(rgbs.shape[-1])
                points_2d[..., 1] *= feedforward_rgb.shape[-2] / float(rgbs.shape[-2])
        else:
            feedforward_rgb = rgbs
            vae_rgb = _resize_video_frames(rgbs, self.image_resolution)
            depth = sample["depth"].unsqueeze(1).float().contiguous()
            valid_mask = sample["valid_mask"].unsqueeze(1).float().contiguous()
            intrinsics = sample["intrinsics"].float().contiguous()
            extrinsics = sample["extrinsics"].float().contiguous()
            ray = sample["ray"].float().contiguous()
            points_2d = sample.get("points_2d")

        result: dict[str, Any] = {
            "dataset_name": "point_odyssey",
            "vae_image_tensor": (vae_rgb * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous(),
            "feedforward_image_tensor": (feedforward_rgb * 2.0 - 1.0)
            .permute(1, 0, 2, 3)
            .contiguous(),
            "image_hw": (int(feedforward_rgb.shape[-2]), int(feedforward_rgb.shape[-1])),
            "depth": depth.contiguous(),
            "ray": ray.contiguous(),
            "valid_mask": valid_mask.contiguous(),
            "intrinsics": intrinsics.contiguous(),
            "extrinsics": extrinsics.contiguous(),
            "scene_name": sample["scene_name"],
            "start_idx": sample["start_idx"],
            "end_idx": sample["end_idx"],
            "clip_uid": f"{sample['scene_name']}:{sample['start_idx']}:{sample['end_idx']}",
        }

        for key in ("points_3d", "points_valid"):
            if key in sample:
                result[key] = sample[key].contiguous()
        if points_2d is not None:
            result["points_2d"] = points_2d.contiguous()

        return result
