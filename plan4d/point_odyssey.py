"""Point Odyssey Dataset for GeometryAdapter VAE training.

This module provides a dataset class for loading Point Odyssey data
for training the GeometryAdapter VAE with DA3-Streaming.

Features:
    - Online RGB image loading (no pre-extracted tokens)
    - Trunk-based sampling with configurable chunk_size and overlap
    - Dynamic sample count calculation (non-hardcoded)
    - Ground truth depth and camera parameters loading
    - Ray map generation from camera intrinsics and extrinsics
    - Resize + center crop with consistent geometry processing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def compute_resize_params(
    src_height: int,
    src_width: int,
    target_height: int,
    target_width: int,
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute resize and center crop parameters.
    
    Strategy: Resize so that the smaller dimension matches target,
    then center crop to get exact target size.
    
    Args:
        src_height: Source image height
        src_width: Source image width
        target_height: Target height after crop
        target_width: Target width after crop
        
    Returns:
        Tuple of (resize_height, resize_width, crop_top, crop_left, crop_height, crop_width)
    """
    # Calculate scale factors
    scale_h = target_height / src_height
    scale_w = target_width / src_width
    
    # Use the larger scale to ensure we can crop to target size
    scale = max(scale_h, scale_w)
    
    # Compute resize dimensions
    resize_height = int(np.ceil(src_height * scale))
    resize_width = int(np.ceil(src_width * scale))
    
    # Ensure resize dimensions are at least target size
    resize_height = max(resize_height, target_height)
    resize_width = max(resize_width, target_width)
    
    # Compute center crop offsets
    crop_top = (resize_height - target_height) // 2
    crop_left = (resize_width - target_width) // 2
    
    return resize_height, resize_width, crop_top, crop_left, target_height, target_width


def resize_and_center_crop_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, Tuple[int, int, int, int, int, int]]:
    """
    Resize and center crop an image to target size.
    
    Args:
        image: Input image [H, W, C] or [H, W]
        target_height: Target height
        target_width: Target width
        interpolation: OpenCV interpolation method
        
    Returns:
        Cropped image and resize parameters tuple
    """
    src_height, src_width = image.shape[:2]
    
    # Compute resize and crop parameters
    params = compute_resize_params(src_height, src_width, target_height, target_width)
    resize_height, resize_width, crop_top, crop_left, crop_h, crop_w = params
    
    # Resize
    resized = cv2.resize(image, (resize_width, resize_height), interpolation=interpolation)
    
    # Center crop
    cropped = resized[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w]
    
    return cropped, params


def adjust_intrinsics_for_resize_and_crop(
    intrinsics: np.ndarray,
    src_height: int,
    src_width: int,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_left: int,
) -> np.ndarray:
    """
    Adjust camera intrinsics for resize and center crop operations.
    
    The intrinsic matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    
    For resize: scale fx, fy, cx, cy by the resize scale factors
    For crop: subtract crop offsets from cx, cy
    
    Args:
        intrinsics: Camera intrinsic matrix [3, 3]
        src_height: Original image height
        src_width: Original image width
        resize_height: Height after resize
        resize_width: Width after resize
        crop_top: Top offset for center crop
        crop_left: Left offset for center crop
        
    Returns:
        Adjusted intrinsic matrix [3, 3]
    """
    adjusted = intrinsics.copy()
    
    # Compute scale factors
    scale_x = resize_width / src_width
    scale_y = resize_height / src_height
    
    # Scale focal lengths and principal point for resize
    adjusted[0, 0] *= scale_x  # fx
    adjusted[1, 1] *= scale_y  # fy
    adjusted[0, 2] *= scale_x  # cx
    adjusted[1, 2] *= scale_y  # cy
    
    # Adjust principal point for crop offset
    adjusted[0, 2] -= crop_left  # cx
    adjusted[1, 2] -= crop_top   # cy
    
    return adjusted


def get_trunk_indices(
    total_frames: int,
    chunk_size: int = 120,
    overlap: int = 60
) -> List[Tuple[int, int]]:
    """
    Calculate trunk indices for a scene, ensuring all frames are covered.
    
    The last trunk may have increased overlap to ensure all frames are used.
    
    Args:
        total_frames: Total number of frames in the scene
        chunk_size: Number of frames per trunk
        overlap: Default overlap between adjacent trunks
        
    Returns:
        List of (start_idx, end_idx) tuples for each trunk
    """
    if total_frames <= chunk_size:
        return [(0, total_frames)]
    
    step = chunk_size - overlap
    indices = []
    start = 0
    
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        indices.append((start, end))
        
        if end >= total_frames:
            break
        
        start += step
    
    # Ensure the last trunk covers chunk_size frames if possible
    # by extending backwards (increasing overlap)
    if len(indices) > 0:
        last_start, last_end = indices[-1]
        if last_end - last_start < chunk_size and last_end == total_frames:
            new_start = max(0, total_frames - chunk_size)
            indices[-1] = (new_start, total_frames)
    
    return indices


def generate_ray_map(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Generate ray map from camera intrinsics and extrinsics.
    
    Ray map format: [6, H, W] where:
        - [0:3, H, W]: ray direction (unnormalized, encodes intrinsic info)
        - [3:6, H, W]: ray origin (camera position in world coordinates)
    
    Direction is d_world = R^T @ K^{-1} @ [u, v, 1], intentionally NOT normalized.
    This preserves focal length information and ensures depth * direction = 3D point.
    
    Args:
        intrinsics: Camera intrinsic matrix [3, 3]
        extrinsics: Camera extrinsic matrix [4, 4] (world-to-camera)
        height: Image height
        width: Image width
        
    Returns:
        ray_map: [6, H, W] tensor
    """
    device = intrinsics.device
    dtype = intrinsics.dtype
    
    # Generate pixel grid
    u = torch.arange(width, dtype=dtype, device=device)
    v = torch.arange(height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')  # [H, W]
    
    # Homogeneous pixel coordinates [H, W, 3]
    ones = torch.ones_like(u)
    pixels = torch.stack([u, v, ones], dim=-1)  # [H, W, 3]
    
    # Camera coordinate ray directions: d_cam = K^{-1} @ p
    K_inv = torch.inverse(intrinsics)  # [3, 3]
    directions_cam = torch.einsum('ij,hwj->hwi', K_inv, pixels)  # [H, W, 3]
    
    # Extract rotation and translation from extrinsics
    R = extrinsics[:3, :3]  # [3, 3]
    t = extrinsics[:3, 3]   # [3]
    
    # World coordinate ray directions: d_world = R^T @ d_cam
    R_inv = R.T  # [3, 3]
    directions_world = torch.einsum('ij,hwj->hwi', R_inv, directions_cam)  # [H, W, 3]
    
    # NOTE: Direction vectors are intentionally NOT normalized.
    # Unnormalized direction d = R^T @ K^{-1} @ [u,v,1] encodes intrinsic info
    # (focal length in the vector magnitude) and ensures depth * direction
    # gives correct 3D points (z-buffer depth * unnormalized ray = camera-space point).
    
    # Camera origin in world coordinates: origin = -R^T @ t
    origin = -R_inv @ t  # [3]
    origins = origin.view(1, 1, 3).expand(height, width, 3)  # [H, W, 3]
    
    # Combine into ray map: [H, W, 6] -> [6, H, W]
    # Order: [direction(3), origin(3)] - This is GT format
    ray_map = torch.cat([directions_world, origins], dim=-1)  # [H, W, 6]
    ray_map = ray_map.permute(2, 0, 1)  # [6, H, W]
    
    return ray_map


class PointOdysseyDataset(Dataset):
    """
    Dataset for Point Odyssey data with trunk-based sampling.
    
    This dataset loads RGB images, depth maps, and camera parameters
    from Point Odyssey scenes. It uses trunk-based sampling where each
    sample consists of a fixed number of consecutive frames (chunk_size)
    with overlap between adjacent samples.
    
    Data format returned:
        - rgbs: [T, H, W, 3] uint8 RGB images
        - depth: [T, H, W] float32 depth maps
        - intrinsics: [T, 3, 3] camera intrinsic matrices
        - extrinsics: [T, 4, 4] camera extrinsic matrices
        - ray: [T, 6, H, W] ray maps (direction + origin)
    
    Resize + Center Crop:
        If target_height and target_width are specified, images will be
        resized and center cropped to the target size. The same geometric
        transformation is applied to depth maps and camera intrinsics.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        chunk_size: int = 120,
        overlap: int = 60,
        image_size: Optional[Tuple[int, int]] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        load_depth: bool = True,
        load_ray: bool = True,
        max_scenes: Optional[int] = None,
        additional_splits: Optional[List[str]] = None,
    ):
        """
        Initialize Point Odyssey dataset.
        
        Args:
            data_root: Root directory of Point Odyssey dataset
            split: Data split ('train', 'test', 'sample')
            chunk_size: Number of frames per sample (trunk)
            overlap: Overlap frames between adjacent trunks
            image_size: Optional target image size (H, W) for resizing (deprecated, use target_height/target_width)
            target_height: Target height after resize and center crop
            target_width: Target width after resize and center crop
            load_depth: Whether to load depth maps
            load_ray: Whether to generate ray maps
            max_scenes: Maximum number of scenes to load (for debugging)
            additional_splits: Additional splits to merge with the main split (e.g., ['sample', 'val'])
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.image_size = image_size
        self.target_height = target_height
        self.target_width = target_width
        self.load_depth = load_depth
        self.load_ray = load_ray
        self.additional_splits = additional_splits or []
        
        # Build sample index
        self.samples = self._build_sample_index(max_scenes)
        
    def _build_sample_index(
        self,
        max_scenes: Optional[int] = None
    ) -> List[Dict]:
        """
        Build index of all samples across all scenes.
        
        Each sample is a dictionary containing:
            - scene_dir: Path to scene directory
            - scene_name: Name of the scene
            - start_idx: Start frame index
            - end_idx: End frame index
            - num_frames: Number of frames in this sample
        
        This method also merges samples from additional_splits if specified.
        """
        samples = []
        
        # Collect all splits to process (main split + additional splits)
        splits_to_process = [self.split] + self.additional_splits
        
        for current_split in splits_to_process:
            split_dir = self.data_root / current_split
            
            if not split_dir.exists():
                # Skip non-existent additional splits silently
                if current_split != self.split:
                    continue
                raise ValueError(f"Split directory not found: {split_dir}")
            
            # Get all scene directories
            scene_dirs = sorted([
                d for d in split_dir.iterdir()
                if d.is_dir() and (d / 'rgbs').exists()
            ])
            
            # Apply max_scenes limit only to main split
            if max_scenes is not None and current_split == self.split:
                scene_dirs = scene_dirs[:max_scenes]
            
            for scene_dir in scene_dirs:
                # Count frames in this scene
                rgb_dir = scene_dir / 'rgbs'
                rgb_files = sorted([
                    f for f in rgb_dir.iterdir()
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                ])
                num_frames = len(rgb_files)
                
                if num_frames == 0:
                    continue
                
                # Get trunk indices for this scene
                trunk_indices = get_trunk_indices(
                    num_frames,
                    self.chunk_size,
                    self.overlap
                )
                
                # Add each trunk as a sample
                for start_idx, end_idx in trunk_indices:
                    samples.append({
                        'scene_dir': scene_dir,
                        'scene_name': scene_dir.name,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'num_frames': end_idx - start_idx,
                        'total_scene_frames': num_frames,
                        'split': current_split,  # Track which split this sample came from
                    })
        
        return samples
    
    def __len__(self) -> int:
        """Return total number of samples (dynamically calculated)."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index.
        
        Returns:
            Dictionary containing:
                - rgbs: [T, H, W, 3] RGB images (uint8)
                - depth: [T, H, W] depth maps (float32), if load_depth=True
                - intrinsics: [T, 3, 3] camera intrinsics (float32)
                - extrinsics: [T, 4, 4] camera extrinsics (float32)
                - ray: [T, 6, H, W] ray maps (float32), if load_ray=True
                - scene_name: scene name string
                - start_idx: start frame index
                - end_idx: end frame index
        """
        sample_info = self.samples[idx]
        scene_dir = sample_info['scene_dir']
        start_idx = sample_info['start_idx']
        end_idx = sample_info['end_idx']
        
        # Load annotation data
        anno_path = scene_dir / 'anno.npz'
        anno = np.load(anno_path)
        
        # Initialize lists
        rgbs = []
        depths = []
        intrinsics_list = []
        extrinsics_list = []
        rays = []
        
        # Determine if we need resize + center crop
        use_resize_crop = (self.target_height is not None and self.target_width is not None)
        
        # Load data for each frame
        for frame_idx in range(start_idx, end_idx):
            # Load RGB image
            rgb_path = scene_dir / 'rgbs' / f'rgb_{frame_idx:05d}.jpg'
            if not rgb_path.exists():
                rgb_path = scene_dir / 'rgbs' / f'rgb_{frame_idx:05d}.png'
            
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            
            # Get original image dimensions
            orig_height, orig_width = rgb.shape[:2]
            
            # Get camera parameters for this frame (original)
            intrinsics_orig = anno['intrinsics'][frame_idx].copy()  # [3, 3]
            extrinsics = anno['extrinsics'][frame_idx].copy()  # [4, 4]
            
            # Apply resize + center crop if target size is specified
            if use_resize_crop:
                # Resize and center crop RGB
                rgb, resize_params = resize_and_center_crop_image(
                    rgb, self.target_height, self.target_width, cv2.INTER_LINEAR
                )
                resize_height, resize_width, crop_top, crop_left, _, _ = resize_params
                
                # Adjust intrinsics for resize and crop
                intrinsics = adjust_intrinsics_for_resize_and_crop(
                    intrinsics_orig,
                    orig_height, orig_width,
                    resize_height, resize_width,
                    crop_top, crop_left
                )
            elif self.image_size is not None:
                # Legacy resize behavior (deprecated)
                rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
                # Scale intrinsics for simple resize
                scale_x = self.image_size[1] / orig_width
                scale_y = self.image_size[0] / orig_height
                intrinsics = intrinsics_orig.copy()
                intrinsics[0, 0] *= scale_x  # fx
                intrinsics[1, 1] *= scale_y  # fy
                intrinsics[0, 2] *= scale_x  # cx
                intrinsics[1, 2] *= scale_y  # cy
            else:
                intrinsics = intrinsics_orig
            
            rgbs.append(rgb)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            
            # Load depth if requested
            if self.load_depth:
                depth_path = scene_dir / 'depths' / f'depth_{frame_idx:05d}.png'
                if depth_path.exists():
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                    # Convert to float32 depth (assuming 16-bit PNG)
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) / 1000.0  # mm to meters
                    else:
                        depth = depth.astype(np.float32)
                    
                    # Apply same resize + center crop to depth
                    if use_resize_crop:
                        depth, _ = resize_and_center_crop_image(
                            depth, self.target_height, self.target_width, cv2.INTER_NEAREST
                        )
                    elif self.image_size is not None:
                        depth = cv2.resize(
                            depth,
                            (self.image_size[1], self.image_size[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    depths.append(depth)
                else:
                    # Create dummy depth if not available
                    h, w = rgb.shape[:2]
                    depths.append(np.zeros((h, w), dtype=np.float32))
            
            # Generate ray map if requested
            # Ray map is generated using the adjusted intrinsics
            if self.load_ray:
                h, w = rgb.shape[:2]
                intrinsics_tensor = torch.from_numpy(intrinsics).float()
                extrinsics_tensor = torch.from_numpy(extrinsics).float()
                
                ray_map = generate_ray_map(
                    intrinsics_tensor,
                    extrinsics_tensor,
                    h, w
                )
                rays.append(ray_map.numpy())
        
        # Stack arrays
        rgbs = np.stack(rgbs, axis=0)  # [T, H, W, 3]
        intrinsics_arr = np.stack(intrinsics_list, axis=0)  # [T, 3, 3]
        extrinsics_arr = np.stack(extrinsics_list, axis=0)  # [T, 4, 4]
        
        # Convert to tensors
        result = {
            'rgbs': torch.from_numpy(rgbs),  # [T, H, W, 3] uint8
            'intrinsics': torch.from_numpy(intrinsics_arr).float(),  # [T, 3, 3]
            'extrinsics': torch.from_numpy(extrinsics_arr).float(),  # [T, 4, 4]
            'scene_name': sample_info['scene_name'],
            'start_idx': start_idx,
            'end_idx': end_idx,
        }
        
        if self.load_depth:
            depths = np.stack(depths, axis=0)  # [T, H, W]
            result['depth'] = torch.from_numpy(depths).float()
            
            # Generate valid_mask: depth > 0 indicates valid pixels
            # This masks out invalid depth values (e.g., missing data, sky, etc.)
            valid_mask = (depths > 0).astype(np.float32)  # [T, H, W]
            result['valid_mask'] = torch.from_numpy(valid_mask).float()
        
        if self.load_ray:
            rays = np.stack(rays, axis=0)  # [T, 6, H, W]
            result['ray'] = torch.from_numpy(rays).float()
        
        # ===== Load sparse 3D trajectory points for Point Cloud Loss =====
        if 'trajs_3d' in anno and 'trajs_2d' in anno:
            raw_trajs_3d = anno['trajs_3d']
            raw_trajs_2d = anno['trajs_2d']

            valid_trajs_3d = (
                isinstance(raw_trajs_3d, np.ndarray)
                and raw_trajs_3d.ndim == 3
                and raw_trajs_3d.shape[-1] == 3
                and raw_trajs_3d.shape[0] >= end_idx
            )
            valid_trajs_2d = (
                isinstance(raw_trajs_2d, np.ndarray)
                and raw_trajs_2d.ndim == 3
                and raw_trajs_2d.shape[-1] == 2
                and raw_trajs_2d.shape[0] >= end_idx
            )
            valid_point_supervision = (
                valid_trajs_3d
                and valid_trajs_2d
                and raw_trajs_3d.shape[1] == raw_trajs_2d.shape[1]
            )

            if valid_point_supervision:
                # trajs_3d: [S_total, N_points, 3] world coordinates
                # trajs_2d: [S_total, N_points, 2] pixel coordinates (x, y)
                # valids: [S_total, N_points] whether point has valid 3D position
                # visibs: [S_total, N_points] whether point is visible in frame
                trajs_3d = raw_trajs_3d[start_idx:end_idx].copy()  # [T, N, 3]
                trajs_2d = raw_trajs_2d[start_idx:end_idx].copy()  # [T, N, 2]

                raw_valids = anno['valids'] if 'valids' in anno else None
                if (
                    isinstance(raw_valids, np.ndarray)
                    and raw_valids.ndim == 2
                    and raw_valids.shape[0] >= end_idx
                    and raw_valids.shape[1] == trajs_3d.shape[1]
                ):
                    point_valids = raw_valids[start_idx:end_idx].copy()
                else:
                    point_valids = np.ones(trajs_3d.shape[:2], dtype=np.float32)

                raw_visibs = anno['visibs'] if 'visibs' in anno else None
                if (
                    isinstance(raw_visibs, np.ndarray)
                    and raw_visibs.ndim == 2
                    and raw_visibs.shape[0] >= end_idx
                    and raw_visibs.shape[1] == trajs_3d.shape[1]
                ):
                    point_visibs = raw_visibs[start_idx:end_idx].copy()
                else:
                    point_visibs = np.ones(trajs_3d.shape[:2], dtype=np.float32)

                points_valid = (point_valids > 0.5) & (point_visibs > 0.5)  # [T, N]

                if use_resize_crop:
                    first_rgb_path = scene_dir / 'rgbs' / f'rgb_{start_idx:05d}.jpg'
                    if not first_rgb_path.exists():
                        first_rgb_path = scene_dir / 'rgbs' / f'rgb_{start_idx:05d}.png'
                    first_rgb = cv2.imread(str(first_rgb_path))
                    src_h, src_w = first_rgb.shape[:2]

                    resize_params = compute_resize_params(src_h, src_w, self.target_height, self.target_width)
                    resize_h, resize_w, crop_t, crop_l, crop_h, crop_w = resize_params

                    scale_x = resize_w / src_w
                    scale_y = resize_h / src_h
                    trajs_2d[..., 0] *= scale_x
                    trajs_2d[..., 1] *= scale_y

                    trajs_2d[..., 0] -= crop_l
                    trajs_2d[..., 1] -= crop_t

                    out_of_bounds = (
                        (trajs_2d[..., 0] < 0) |
                        (trajs_2d[..., 0] >= crop_w) |
                        (trajs_2d[..., 1] < 0) |
                        (trajs_2d[..., 1] >= crop_h)
                    )
                    points_valid = points_valid & (~out_of_bounds)

                elif self.image_size is not None:
                    first_rgb_path = scene_dir / 'rgbs' / f'rgb_{start_idx:05d}.jpg'
                    if not first_rgb_path.exists():
                        first_rgb_path = scene_dir / 'rgbs' / f'rgb_{start_idx:05d}.png'
                    first_rgb = cv2.imread(str(first_rgb_path))
                    src_h, src_w = first_rgb.shape[:2]

                    scale_x = self.image_size[1] / src_w
                    scale_y = self.image_size[0] / src_h
                    trajs_2d[..., 0] *= scale_x
                    trajs_2d[..., 1] *= scale_y

                invalid_2d = ~np.isfinite(trajs_2d)
                trajs_2d[invalid_2d] = 0.0

                result['points_3d'] = torch.from_numpy(trajs_3d).float()  # [T, N, 3]
                result['points_2d'] = torch.from_numpy(trajs_2d).float()  # [T, N, 2]
                result['points_valid'] = torch.from_numpy(points_valid.astype(np.float32))  # [T, N]
        
        return result
    
    def get_scene_info(self, idx: int) -> Dict:
        """Get scene information for a sample."""
        sample_info = self.samples[idx]
        scene_dir = sample_info['scene_dir']
        
        # Load scene_info.json if exists
        scene_info_path = scene_dir / 'scene_info.json'
        if scene_info_path.exists():
            with open(scene_info_path, 'r') as f:
                scene_info = json.load(f)
        else:
            scene_info = {}
        
        return {
            **sample_info,
            **scene_info,
        }


class PointOdysseyTokenDataset(Dataset):
    """
    Wrapper dataset that prepares Point Odyssey images for DA3 multi-layer extraction.
    
    IMPORTANT: This dataset does NOT extract tokens in __getitem__ because:
    1. DataLoader workers cannot use CUDA (fork multiprocessing limitation)
    2. Token extraction must happen in the main training loop
    
    This dataset returns preprocessed RGB images ready for DA3 extraction.
    The actual multi-layer feature extraction (5 layers × 1536 = 7680 channels)
    should be done in the training loop using DA3StreamingInterface.extract_multilayer_features().
    
    Data Flow:
        Dataset.__getitem__() -> returns 'rgbs' [T, H, W, 3] uint8
        Training loop -> da3_interface.extract_multilayer_features(rgbs)
                      -> tokens [B, 7680, T, H//14, W//14]
    """
    
    def __init__(
        self,
        base_dataset: PointOdysseyDataset,
        da3_interface: Optional['DA3StreamingInterface'] = None,
        patch_size: int = 14,
        embed_dim: int = 7680,
        device: str = 'cuda',
    ):
        """
        Initialize token dataset.
        
        Note: da3_interface is stored but NOT used in __getitem__ due to CUDA
        multiprocessing limitations. Token extraction happens in training loop.
        
        Args:
            base_dataset: PointOdysseyDataset instance
            da3_interface: DA3StreamingInterface (stored for reference, not used in __getitem__)
            patch_size: DinoV2 patch size (default: 14)
            embed_dim: Total embedding dimension after multi-layer concatenation
                      (default: 7680 = 5 layers × 1536)
            device: Device for DA3 encoder (used in training loop, not here)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.da3_interface = da3_interface  # Stored for reference only
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with RGB images (tokens extracted in training loop).
        
        IMPORTANT: Does NOT extract tokens here due to CUDA multiprocessing limitations.
        Token extraction must happen in the main training loop.
        
        Returns:
            Dictionary containing:
                - rgbs: [T, H, W, 3] RGB images (uint8) - for token extraction in training loop
                - depth: [T, H, W] depth maps (if available)
                - ray: [T, 6, H, W] ray maps (if available)
                - intrinsics: [T, 3, 3] camera intrinsics
                - extrinsics: [T, 4, 4] camera extrinsics
        """
        # Get base data (includes rgbs, depth, ray, etc.)
        data = self.base_dataset[idx]
        
        # Return data as-is - token extraction happens in training loop
        # The training loop will use da3_interface.extract_multilayer_features()
        # to extract multi-layer tokens from rgbs
        
        return data


def create_point_odyssey_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 1,
    chunk_size: int = 120,
    overlap: int = 60,
    image_size: Optional[Tuple[int, int]] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    da3_encoder: Optional[torch.nn.Module] = None,
    max_scenes: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for Point Odyssey dataset.
    
    Args:
        data_root: Root directory of Point Odyssey dataset
        split: Data split ('train', 'test', 'sample')
        batch_size: Batch size
        chunk_size: Number of frames per sample
        overlap: Overlap between adjacent samples
        image_size: Optional target image size (H, W)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        da3_encoder: Optional DA3 encoder for token extraction
        max_scenes: Maximum number of scenes (for debugging)
        
    Returns:
        DataLoader instance
    """
    # Create base dataset
    base_dataset = PointOdysseyDataset(
        data_root=data_root,
        split=split,
        chunk_size=chunk_size,
        overlap=overlap,
        image_size=image_size,
        load_depth=True,
        load_ray=True,
        max_scenes=max_scenes,
    )
    
    # Wrap with token extraction if encoder provided
    if da3_encoder is not None:
        dataset = PointOdysseyTokenDataset(
            base_dataset=base_dataset,
            da3_encoder=da3_encoder,
        )
    else:
        dataset = base_dataset
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return dataloader


if __name__ == '__main__':
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                        default='./datasets/point_odyssey/dataset')
    parser.add_argument('--split', type=str, default='sample')
    parser.add_argument('--chunk_size', type=int, default=120)
    parser.add_argument('--overlap', type=int, default=60)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Point Odyssey Dataset")
    print("=" * 60)
    
    # Create dataset
    dataset = PointOdysseyDataset(
        data_root=args.data_root,
        split=args.split,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_scenes=2,  # Limit for testing
    )
    
    print(f"\nDataset info:")
    print(f"  Split: {args.split}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Overlap: {args.overlap}")
    
    # Get first sample
    if len(dataset) > 0:
        print(f"\nLoading first sample...")
        sample = dataset[0]
        
        print(f"\nSample contents:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # Get scene info
        scene_info = dataset.get_scene_info(0)
        print(f"\nScene info:")
        for key, value in scene_info.items():
            if not isinstance(value, (Path, np.ndarray)):
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
