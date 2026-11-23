"""DA3-Streaming interface for GeometryAdapter VAE - FINAL FIXED VERSION.

Based on actual DA3 source code analysis:

1. DinoV2.forward() uses `out_layers` LIST, not `n` integer
2. DualDPT Head does NOT use camera_token (only feat[0])
3. patch_start_idx depends on backbone output format
4. Ray format: DA3 outputs (origin, direction), Dataset GT is (direction, origin)!

Key Fixes:
    - Use backbone.forward() or pass layer indices as list
    - Correctly handle ray channel order mismatch
    - Proper ImageNet normalization
"""

from typing import Dict, Optional, Tuple, List
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import logging

logger = logging.getLogger(__name__)

# DA3 uses these layer indices for 4-layer feature pyramid
DEFAULT_INTERMEDIATE_LAYER_IDX = [19, 27, 33, 39]

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def normalize_images_imagenet(images: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization to images.
    
    Args:
        images: [B, S, C, H, W] or [B, C, H, W], range [0, 1] or [0, 255]
        
    Returns:
        Normalized images in ImageNet format
    """
    # Ensure float
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    elif images.max() > 1.0:
        images = images / 255.0
    
    # Handle different input shapes
    if images.dim() == 5:  # [B, S, C, H, W]
        mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1).to(images.device)
        std = IMAGENET_STD.view(1, 1, 3, 1, 1).to(images.device)
    else:  # [B, C, H, W]
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(images.device)
        std = IMAGENET_STD.view(1, 3, 1, 1).to(images.device)
    
    return (images - mean) / std


def convert_ray_gt_to_da3_format(ray_gt: torch.Tensor) -> torch.Tensor:
    """
    Convert GT ray from Dataset format to DA3 format.
    
    Dataset GT format: [B, S, 6, H, W] where [0:3]=direction, [3:6]=origin
    DA3 output format: [B, S, 6, H, W] where [0:3]=origin, [3:6]=direction
    
    Args:
        ray_gt: [B, S, 6, H, W] in Dataset format
        
    Returns:
        ray_da3: [B, S, 6, H, W] in DA3 format (origin first, then direction)
    """
    direction = ray_gt[:, :, :3, :, :]  # [B, S, 3, H, W]
    origin = ray_gt[:, :, 3:6, :, :]    # [B, S, 3, H, W]
    
    # Swap to DA3 format: origin first
    return torch.cat([origin, direction], dim=2)


def convert_ray_da3_to_gt_format(ray_da3: torch.Tensor) -> torch.Tensor:
    """
    Convert DA3 ray output to Dataset GT format (for loss computation).
    
    CONFIRMED by user:
    - DA3 output format: [B, S, 6, H, W] where [0:3]=direction, [3:6]=origin
    - Dataset GT format: [B, S, 6, H, W] where [0:3]=direction, [3:6]=origin
    
    Both formats are THE SAME! This function is now a no-op but kept for API compatibility.
    
    NOTE: Input must be in [B, S, C, H, W] format. If your input is [B, S, H, W, C],
    permute it first: ray = ray.permute(0, 1, 4, 2, 3)
    
    Args:
        ray_da3: [B, S, 6, H, W] - DA3 format (direction first, then origin)
        
    Returns:
        ray_gt_format: [B, S, 6, H, W] - Same as input (formats are identical)
    """
    # DA3 and GT use the same [direction, origin] format; return it directly
    # Keep only the first six channels
    return ray_da3[:, :, :6, :, :]


def convert_ray_camera_to_world(
    pred_ray: torch.Tensor,
    extrinsics: torch.Tensor,
    ray_format: str = "gt",
) -> torch.Tensor:
    """
    Compatibility wrapper for v2 pipeline.

    In geometry_adapter_v2, DA3 ray and dataset GT ray are already aligned in
    world-space semantics (direction + origin). Therefore no extra coordinate
    transform is applied here; we only standardize channel count.
    """
    del extrinsics, ray_format
    return convert_ray_da3_to_gt_format(pred_ray)





class DA3StreamingInterface(nn.Module):
    """
    Final fixed DA3-Streaming interface based on actual source code analysis.
    
    Key differences from previous versions:
    1. Uses backbone.forward() which correctly passes out_layers as list
    2. Does not rely on camera_token (DualDPT ignores it anyway)
    3. Handles ray channel order conversion
    4. Includes proper ImageNet normalization
    """
    
    PATCH_SIZE = 14
    EMBED_DIM = 1536
    PER_LAYER_DIM = 3072  # With cat_token=True: local (1536) + global (1536)
    OUTPUT_DIM = 12288    # 4 layers × 3072
    NUM_LAYERS = 40
    NUM_REGISTER_TOKENS = 4
    NUM_INTERMEDIATE_LAYERS = 4

    def __init__(
        self,
        da3_model: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
        freeze_decoder: bool = True,
        intermediate_layer_idx: Optional[List[int]] = None,
    ):
        super().__init__()
        self.da3_model = da3_model
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.intermediate_layer_idx = intermediate_layer_idx or DEFAULT_INTERMEDIATE_LAYER_IDX
        
        if da3_model is not None:
            self._setup_model()

    def _setup_model(self) -> None:
        """Setup model freezing."""
        if self.freeze_encoder:
            backbone = self._get_backbone()
            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False

        if self.freeze_decoder:
            head = self._get_head()
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = False

            scene_flow_head = self._get_scene_flow_head()
            if scene_flow_head is not None:
                for param in scene_flow_head.parameters():
                    param.requires_grad = False

            scene_flow_adapter = self._get_scene_flow_adapter()
            if scene_flow_adapter is not None:
                for param in scene_flow_adapter.parameters():
                    param.requires_grad = False

    def set_da3_model(self, da3_model: nn.Module) -> None:
        self.da3_model = da3_model
        self._setup_model()

    def _get_da3_net(self):
        """Get the DepthAnything3Net (da3 branch for anyview)."""
        da3_model = self.da3_model
        if hasattr(da3_model, 'model'):
            da3_model = da3_model.model
        if hasattr(da3_model, 'da3'):
            return da3_model.da3
        return da3_model

    def _get_backbone(self):
        """Get DinoV2 backbone."""
        da3_net = self._get_da3_net()
        if hasattr(da3_net, 'backbone'):
            return da3_net.backbone
        return None
    
    def _get_head(self):
        """Get DualDPT Head."""
        da3_net = self._get_da3_net()
        if hasattr(da3_net, 'head'):
            return da3_net.head
        return None

    def _get_scene_flow_head(self):
        """Get scene flow token-only head if available."""
        da3_net = self._get_da3_net()
        if hasattr(da3_net, 'scene_flow_head'):
            return da3_net.scene_flow_head
        return None

    def _get_scene_flow_adapter(self):
        """Get scene flow adapter if available."""
        da3_net = self._get_da3_net()
        if hasattr(da3_net, 'scene_flow_adapter'):
            return da3_net.scene_flow_adapter
        return None

    def _output_get(self, output, key: str, default=None):
        """Read from dict-like/object-like DA3 outputs."""
        if output is None:
            return default
        if isinstance(output, dict):
            return output.get(key, default)
        return getattr(output, key, default)

    def _prepare_features_for_dpt(
        self,
        features: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Convert stacked spatial features to DPT-compatible token tuples."""
        if features.dim() != 5:
            raise ValueError(f"Expected 5D features [B, C, S, ph, pw], got {features.shape}")

        b, c, s, _, _ = features.shape
        if c != self.OUTPUT_DIM:
            raise ValueError(f"Expected {self.OUTPUT_DIM} channels, got {c}")

        feats_for_dpt = []
        for feat in torch.chunk(features, self.NUM_INTERMEDIATE_LAYERS, dim=1):
            feat_reshaped = rearrange(feat, 'b c s ph pw -> b s (ph pw) c')
            dummy_camera_token = torch.zeros(
                b,
                s,
                self.PER_LAYER_DIM,
                device=feat.device,
                dtype=feat.dtype,
            )
            feats_for_dpt.append((feat_reshaped, dummy_camera_token))

        return feats_for_dpt

    def _get_module_reference_tensor(self, module: nn.Module) -> Optional[torch.Tensor]:
        ref_tensor = next(module.parameters(), None)
        if ref_tensor is None:
            ref_tensor = next(module.buffers(), None)
        return ref_tensor

    def _align_dpt_inputs_to_module(
        self,
        feats_for_dpt: List[Tuple[torch.Tensor, torch.Tensor]],
        module: nn.Module,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Match DPT token dtype/device to the live decoder module."""
        ref_tensor = self._get_module_reference_tensor(module)
        if ref_tensor is None:
            return feats_for_dpt

        aligned = []
        target_device = ref_tensor.device
        target_dtype = ref_tensor.dtype
        for feat_tokens, camera_token in feats_for_dpt:
            feat_tokens = feat_tokens.to(device=target_device, dtype=target_dtype)
            camera_token = camera_token.to(device=target_device, dtype=target_dtype)
            aligned.append((feat_tokens, camera_token))
        return aligned

    def _autocast_for_module(self, module: nn.Module):
        """Mirror DA3 API autocast behavior for direct head calls."""
        ref_tensor = self._get_module_reference_tensor(module)
        if ref_tensor is None:
            return nullcontext()

        device_type = ref_tensor.device.type
        target_dtype = ref_tensor.dtype
        if device_type == "cuda" and target_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type=device_type, dtype=target_dtype)
        if device_type == "cpu" and target_dtype == torch.bfloat16:
            return torch.autocast(device_type=device_type, dtype=target_dtype)
        return nullcontext()

    def _extract_spatial_features_from_output(
        self,
        output,
        image_size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """Extract stacked spatial features from DA3 full-forward outputs."""
        h, w = image_size
        patch_h, patch_w = h // self.PATCH_SIZE, w // self.PATCH_SIZE

        features_list = self._output_get(output, 'feats')
        if features_list is not None:
            if isinstance(features_list, (list, tuple)):
                stacked_features = torch.cat(features_list, dim=-1)
            else:
                stacked_features = features_list
            return rearrange(
                stacked_features,
                'b s (ph pw) c -> b c s ph pw',
                ph=patch_h,
                pw=patch_w,
            )

        aux_features = self._output_get(output, 'aux')
        if aux_features is None:
            return None

        feature_tokens = []
        for layer_idx in self.intermediate_layer_idx:
            feat_key = f'feat_layer_{layer_idx}'
            if feat_key not in aux_features:
                logger.warning(f"[DA3Interface] Missing {feat_key} in DA3 aux output")
                return None
            feature_tokens.append(
                rearrange(aux_features[feat_key], 'b s ph pw c -> b s (ph pw) c')
            )

        stacked_features = torch.cat(feature_tokens, dim=-1)
        return rearrange(
            stacked_features,
            'b s (ph pw) c -> b c s ph pw',
            ph=patch_h,
            pw=patch_w,
        )

    @property
    def output_dim(self) -> int:
        return self.OUTPUT_DIM

    @torch.no_grad()
    def extract_multilayer_features(
        self, 
        images: torch.Tensor,
        normalize: bool = True,
        ref_view_strategy: str = "saddle_balanced",
    ) -> torch.Tensor:
        """
        Extract multi-layer features using DA3's backbone.forward().
        
        This method uses the correct DA3 API which passes out_layers as a list,
        not as n (integer).
        
        Args:
            images: [B, S, C, H, W] or [B, C, H, W], RGB images
                   Can be uint8 [0,255] or float [0,1] - will be normalized
            normalize: Whether to apply ImageNet normalization
            ref_view_strategy: Strategy for selecting reference view in DA3
                - "saddle_balanced": Default balanced strategy
                - "first": Use first frame as reference
                - "middle": Use middle frame as reference
                - "last": Use last frame as reference
            
        Returns:
            features: [B, 12288, S, H//14, W//14]
        """
        if self.da3_model is None:
            raise RuntimeError("DA3 model not set")
        
        # Handle input shape
        is_video = images.dim() == 5
        if not is_video:
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
        
        b, s, c, h, w = images.shape
        
        # Normalize images if needed
        if normalize:
            images = normalize_images_imagenet(images)
        
        # Get backbone
        backbone = self._get_backbone()
        if backbone is None:
            raise RuntimeError("Cannot find backbone in DA3 model")
        
        # Verify backbone configuration
        logger.info(f"[DA3Interface] Backbone type: {type(backbone).__name__}")
        logger.info(f"[DA3Interface] out_layers: {getattr(backbone, 'out_layers', 'N/A')}")
        logger.info(f"[DA3Interface] cat_token: {getattr(backbone, 'cat_token', 'N/A')}")
        
        # ===== KEY FIX: Use backbone.forward() which correctly uses out_layers =====
        # backbone.forward() internally calls:
        #   pretrained.get_intermediate_layers(x, self.out_layers)
        # This passes out_layers as a LIST, not n!
        
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
            # backbone.forward returns: [(features, camera_token), ...]
            raw_outputs = backbone.forward(images, ref_view_strategy=ref_view_strategy)
        
        # Handle output format
        if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2:
            outputs_with_cam, aux = raw_outputs
        else:
            outputs_with_cam = raw_outputs
        
        if not isinstance(outputs_with_cam, (list, tuple)):
            outputs_with_cam = list(outputs_with_cam)
        
        logger.info(f"[DA3Interface] Got {len(outputs_with_cam)} layer outputs")
        
        # Extract features (DualDPT only uses feat[0], ignores camera_token)
        features_list = []
        for i, out in enumerate(outputs_with_cam):
            if isinstance(out, tuple):
                feat = out[0]  # [B, S, num_patches, C]
                # camera_token = out[1]  # Not used by DualDPT
            else:
                feat = out
            
            logger.info(f"[DA3Interface] Layer {i}: feat shape {feat.shape}")
            
            # Verify dimension
            expected_dim = self.PER_LAYER_DIM
            if feat.shape[-1] != expected_dim:
                logger.warning(f"[DA3Interface] Layer {i}: expected {expected_dim} channels, got {feat.shape[-1]}")
            
            features_list.append(feat)
        
        # Concatenate all layers
        if len(features_list) == 1:
            stacked_features = features_list[0]
        else:
            stacked_features = torch.cat(features_list, dim=-1)  # [B, S, num_patches, 12288]
        
        logger.info(f"[DA3Interface] Stacked features: {stacked_features.shape}")
        
        # Reshape to spatial format
        patch_h, patch_w = h // self.PATCH_SIZE, w // self.PATCH_SIZE
        spatial_features = rearrange(
            stacked_features, 
            'b s (ph pw) c -> b c s ph pw', 
            ph=patch_h, pw=patch_w
        )
        
        if not is_video:
            spatial_features = spatial_features.squeeze(2)
        
        logger.info(f"[DA3Interface] Output features: {spatial_features.shape}")
        
        return spatial_features

    def decode_features_to_geometry(
        self,
        features: torch.Tensor,
        image_size: Tuple[int, int],
        use_gradient: bool = True,
        chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode features to depth and ray using DualDPT Head.
        
        Note: Based on DualDPT source code analysis:
        - DualDPT does NOT use camera_token (only feat[0])
        - patch_start_idx should be 0 if backbone already removed prefix tokens
        
        Args:
            features: [B, 12288, S, ph, pw]
            image_size: (H, W)
            
        Returns:
            Dict with:
                - 'depth': [B, S, 1, H, W]
                - 'ray': [B, S, 7, H, W] (origin[3], direction[3], conf[1])
        """
        if self.da3_model is None:
            raise RuntimeError("DA3 model not set")

        h, w = image_size
        feats_for_dpt = self._prepare_features_for_dpt(features)

        head = self._get_head()
        if head is None:
            raise RuntimeError("Cannot find DualDPT Head in DA3 model")
        feats_for_dpt = self._align_dpt_inputs_to_module(feats_for_dpt, head)

        patch_start_idx = 0
        logger.info(f"[DA3Interface] Calling DualDPT with patch_start_idx={patch_start_idx}")

        if use_gradient:
            with self._autocast_for_module(head):
                output = head.forward(
                    feats=feats_for_dpt,
                    H=h,
                    W=w,
                    patch_start_idx=patch_start_idx,
                    chunk_size=chunk_size,
                )
        else:
            with torch.no_grad():
                with self._autocast_for_module(head):
                    output = head.forward(
                        feats=feats_for_dpt,
                        H=h,
                        W=w,
                        patch_start_idx=patch_start_idx,
                        chunk_size=chunk_size,
                    )

        logger.info(f"[DA3Interface] DualDPT output keys: {list(output.keys())}")

        return {
            'depth': output.get('depth'),
            'depth_conf': output.get('depth_conf'),
            'ray': output.get('ray'),
            'ray_conf': output.get('ray_conf'),
        }

    def decode_features_to_scene_flow(
        self,
        features: torch.Tensor,
        image_size: Tuple[int, int],
        extrinsics: Optional[torch.Tensor] = None,
        use_gradient: bool = True,
        chunk_size: int = 8,
        output_basis: str = "frame0",
    ) -> torch.Tensor:
        """
        Decode stacked 4-layer DA3 tokens to scene flow.

        Args:
            features: [B, 12288, S, ph, pw]
            image_size: (H, W)
            extrinsics: [B, S, 4, 4] or [B, S, 3, 4] world-to-camera extrinsics.
                Required when ``output_basis='frame0'``.
            use_gradient: Whether to keep gradients for the decoder path.
            chunk_size: Temporal chunk size for DPT decoding.
            output_basis: ``'frame0'`` for first-frame camera basis, or ``'camera'`` for
                per-frame camera basis.

        Returns:
            Scene flow tensor with shape [B, S, 3, H, W].
        """
        if self.da3_model is None:
            raise RuntimeError("DA3 model not set")
        if output_basis not in {"frame0", "camera"}:
            raise ValueError(f"Unsupported output_basis: {output_basis}")
        if output_basis == "frame0" and extrinsics is None:
            raise ValueError("extrinsics is required when output_basis='frame0'")

        h, w = image_size
        feats_for_dpt = self._prepare_features_for_dpt(features)

        scene_flow_head = self._get_scene_flow_head()
        if scene_flow_head is None:
            raise RuntimeError("Cannot find scene flow head in DA3 model")
        feats_for_dpt = self._align_dpt_inputs_to_module(feats_for_dpt, scene_flow_head)

        patch_start_idx = 0
        logger.info(f"[DA3Interface] Calling SceneFlowDPT with patch_start_idx={patch_start_idx}")

        if use_gradient:
            with self._autocast_for_module(scene_flow_head):
                output = scene_flow_head.forward(
                    feats=feats_for_dpt,
                    H=h,
                    W=w,
                    patch_start_idx=patch_start_idx,
                    chunk_size=chunk_size,
                )
        else:
            with torch.no_grad():
                with self._autocast_for_module(scene_flow_head):
                    output = scene_flow_head.forward(
                        feats=feats_for_dpt,
                        H=h,
                        W=w,
                        patch_start_idx=patch_start_idx,
                        chunk_size=chunk_size,
                    )

        scene_flow_cam = output.get('scene_flow_cam')
        if scene_flow_cam is None:
            raise RuntimeError("scene flow head did not return 'scene_flow_cam'")

        if scene_flow_cam.shape[1] > 0:
            scene_flow_cam = scene_flow_cam.clone()
            scene_flow_cam[:, -1] = 0

        if output_basis == "camera":
            return scene_flow_cam

        scene_flow_adapter = self._get_scene_flow_adapter()
        if scene_flow_adapter is None:
            raise RuntimeError("Cannot find scene flow adapter in DA3 model")

        if use_gradient:
            return scene_flow_adapter(scene_flow_cam=scene_flow_cam, extrinsics=extrinsics)
        with torch.no_grad():
            return scene_flow_adapter(scene_flow_cam=scene_flow_cam, extrinsics=extrinsics)

    @torch.no_grad()
    def inference_full(
        self,
        images: torch.Tensor,
        ref_view_strategy: str = "saddle_balanced",
        infer_scene_flow: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference using DA3's internal forward (most reliable).
        
        This bypasses extract + decode and uses DA3's complete pipeline.
        
        Args:
            images: [B, S, C, H, W] RGB images (will be normalized)
            infer_scene_flow: Whether to enable DA3's integrated scene flow branch.
            
        Returns:
            Model output dict with depth, ray, pose and optional scene flow.
        """
        if self.da3_model is None:
            raise RuntimeError("DA3 model not set")

        if images.dim() == 4:
            images = images.unsqueeze(1)

        images = normalize_images_imagenet(images)
        da3_net = self._get_da3_net()
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
            output = da3_net(
                images,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=self.intermediate_layer_idx,
                infer_gs=False,
                infer_scene_flow=infer_scene_flow,
                use_ray_pose=False,
                ref_view_strategy=ref_view_strategy,
            )

        return output

    @torch.no_grad()
    def extract_features_with_geometry(
        self,
        images: torch.Tensor,
        ref_view_strategy: str = "saddle_balanced",
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract multi-layer features AND geometry outputs using DA3's full forward.
        
        This method uses DA3's complete pipeline to get both:
        1. Multi-layer features for GeometryAdapter training
        2. Geometry outputs (depth, ray) for supervision
        
        This is more reliable than separate extract + decode as it uses
        DA3's internal ref_view_strategy for consistent processing.
        
        Args:
            images: [B, S, C, H, W] or [B, C, H, W], RGB images
                   Can be uint8 [0,255] or float [0,1] - will be normalized
            ref_view_strategy: Strategy for selecting reference view in DA3
                - "saddle_balanced": Default balanced strategy
                - "first": Use first frame as reference
                - "middle": Use middle frame as reference
                - "last": Use last frame as reference
            normalize: Whether to apply ImageNet normalization
            
        Returns:
            Tuple of:
                - features: [B, 12288, S, H//14, W//14] multi-layer features
                - geometry: Dict with 'depth', 'ray', 'depth_conf', 'ray_conf'
        """
        if self.da3_model is None:
            raise RuntimeError("DA3 model not set")
        
        # Handle input shape
        is_video = images.dim() == 5
        if not is_video:
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
        
        b, s, c, h, w = images.shape
        
        # Normalize images if needed
        if normalize:
            images = normalize_images_imagenet(images)
        
        da3_net = self._get_da3_net()
        
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        with torch.autocast(device_type=images.device.type, dtype=autocast_dtype):
            # Use DA3's full forward with export_feat_layers to get both features and geometry
            output = da3_net(
                images,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=self.intermediate_layer_idx,
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy=ref_view_strategy,
            )
        
        spatial_features = self._extract_spatial_features_from_output(output, image_size=(h, w))
        if spatial_features is None:
            # Fallback: extract features separately using backbone
            logger.warning("[DA3Interface] Could not read exported aux features, falling back to backbone extraction")
            spatial_features = self.extract_multilayer_features(
                images, normalize=False, ref_view_strategy=ref_view_strategy
            )
        
        if not is_video:
            spatial_features = spatial_features.squeeze(2)
        
        # Extract geometry outputs
        geometry = {
            'depth': self._output_get(output, 'depth'),
            'depth_conf': self._output_get(output, 'depth_conf'),
            'ray': self._output_get(output, 'ray'),
            'ray_conf': self._output_get(output, 'ray_conf'),
            'extrinsics': self._output_get(output, 'extrinsics'),
            'intrinsics': self._output_get(output, 'intrinsics'),
        }
        
        logger.info(f"[DA3Interface] extract_features_with_geometry:")
        logger.info(f"  - features shape: {spatial_features.shape}")
        logger.info(f"  - depth shape: {geometry['depth'].shape if geometry['depth'] is not None else None}")
        logger.info(f"  - ray shape: {geometry['ray'].shape if geometry['ray'] is not None else None}")
        
        return spatial_features, geometry

# ============================================================
# Utility functions for loss computation
# ============================================================
def compute_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_type: str = 'l1',
) -> torch.Tensor:
    """
    Compute depth loss with valid mask.
    
    Args:
        pred_depth: [B, S, 1, H, W] predicted depth
        gt_depth: [B, S, H, W] ground truth depth
        valid_mask: [B, S, H, W] valid depth mask (1 = valid, 0 = invalid)
        loss_type: 'l1' or 'l2'
        
    Returns:
        Scalar loss value
    """
    # Squeeze pred depth if needed
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(2)  # [B, S, H, W]
    
    # Apply valid mask
    valid = valid_mask > 0
    
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    
    pred_valid = pred_depth[valid]
    gt_valid = gt_depth[valid]
    
    if loss_type == 'l1':
        return F.l1_loss(pred_valid, gt_valid)
    elif loss_type == 'l2':
        return F.mse_loss(pred_valid, gt_valid)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_ray_loss(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    convert_gt_format: bool = True,
    loss_type: str = 'l1',
) -> torch.Tensor:
    """
    Compute ray loss, handling channel order mismatch.
    
    IMPORTANT: 
        - DA3 output: [origin, direction] (channels 0:3 = origin, 3:6 = direction)
        - Dataset GT: [direction, origin] (channels 0:3 = direction, 3:6 = origin)
    
    Args:
        pred_ray: [B, S, 6/7, H, W] DA3 format (origin, direction)
        gt_ray: [B, S, 6, H, W] Dataset format (direction, origin)
        convert_gt_format: If True, convert GT to DA3 format before comparison
        loss_type: 'l1' or 'l2'
        
    Returns:
        Scalar loss value
    """
    # Extract pred components (DA3 format: origin first)
    pred_origin = pred_ray[:, :, :3, :, :]
    pred_dir = pred_ray[:, :, 3:6, :, :]
    
    if convert_gt_format:
        # GT format: direction first
        gt_dir = gt_ray[:, :, :3, :, :]
        gt_origin = gt_ray[:, :, 3:6, :, :]
    else:
        # Assume GT is already in DA3 format
        gt_origin = gt_ray[:, :, :3, :, :]
        gt_dir = gt_ray[:, :, 3:6, :, :]
    
    # Compute losses separately
    if loss_type == 'l1':
        origin_loss = F.l1_loss(pred_origin, gt_origin)
        dir_loss = F.l1_loss(pred_dir, gt_dir)
    elif loss_type == 'l2':
        origin_loss = F.mse_loss(pred_origin, gt_origin)
        dir_loss = F.mse_loss(pred_dir, gt_dir)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return origin_loss + dir_loss


# ============================================================
# Test and validation utilities
# ============================================================

def validate_interface(
    da3_model,
    test_images: torch.Tensor,
    device: str = 'cuda',
):
    """
    Validate that the interface produces reasonable outputs.
    
    Args:
        da3_model: Loaded DA3 model
        test_images: [B, S, H, W, 3] test images (uint8)
        device: Device
        
    Returns:
        Validation results dict
    """
    interface = DA3StreamingInterface(da3_model)
    
    # Convert images: [B, S, H, W, 3] -> [B, S, 3, H, W]
    if test_images.shape[-1] == 3:
        test_images = test_images.permute(0, 1, 4, 2, 3)
    
    test_images = test_images.to(device)
    
    b, s, c, h, w = test_images.shape
    
    results = {}
    
    # Test feature extraction
    logger.info("Testing feature extraction...")
    features = interface.extract_multilayer_features(test_images)
    results['features_shape'] = features.shape
    results['features_mean'] = features.mean().item()
    results['features_std'] = features.std().item()
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Features stats: mean={results['features_mean']:.4f}, std={results['features_std']:.4f}")
    
    # Test decoding
    logger.info("Testing decoding...")
    outputs = interface.decode_features_to_geometry(features, image_size=(h, w))
    
    if outputs['depth'] is not None:
        results['depth_shape'] = outputs['depth'].shape
        results['depth_mean'] = outputs['depth'].mean().item()
        results['depth_range'] = (outputs['depth'].min().item(), outputs['depth'].max().item())
        logger.info(f"Depth shape: {results['depth_shape']}")
        logger.info(f"Depth stats: mean={results['depth_mean']:.4f}, range={results['depth_range']}")
    
    if outputs['ray'] is not None:
        results['ray_shape'] = outputs['ray'].shape
        results['ray_mean'] = outputs['ray'].mean().item()
        logger.info(f"Ray shape: {results['ray_shape']}")
        logger.info(f"Ray stats: mean={results['ray_mean']:.4f}")
    
    # Test full inference
    logger.info("Testing full inference...")
    full_output = interface.inference_full(test_images)
    results['full_inference_keys'] = list(full_output.keys())
    logger.info(f"Full inference keys: {results['full_inference_keys']}")
    
    return results
