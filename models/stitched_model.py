import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from loguru import logger

from models.da3_stitched import (
    DA3Stitched,
    DA3_DISTILL_FEATURE_LAYER,
    compute_da3_stitching_input_spatial_shape,
    compute_da3_stitching_input_spatial_size,
    flatten_da3_feature_grid_to_tokens,
    is_da3_model,
)
from models.stitching_layer_builder import (
    build_stitching_module,
    warm_start_stitching_module_from_conv_state,
)
from utils.wan_utils import AutoencoderKLWan as AutoencoderKLWan_wan


_SUPPORTED_WAN_VAE_CLASS_NAMES = {"WanVideoVAE", "WanVideoVAE38", "AutoencoderKLWan"}


def _is_supported_wan_vae(diffusion_vae: nn.Module) -> bool:
    candidate = diffusion_vae.module if hasattr(diffusion_vae, "module") else diffusion_vae
    return isinstance(candidate, AutoencoderKLWan) or isinstance(candidate, AutoencoderKLWan_wan) or candidate.__class__.__name__ in _SUPPORTED_WAN_VAE_CLASS_NAMES


def _should_attach_da3_distill_tokens(stitched_3d_model: nn.Module) -> bool:
    return isinstance(stitched_3d_model, DA3Stitched) and getattr(
        stitched_3d_model,
        "stitching_layer",
        None,
    ) == DA3_DISTILL_FEATURE_LAYER


class StitchVAE3D(nn.Module):
    def __init__(
        self,
        diffusion_vae: nn.Module,
        feedforward_model: nn.Module,
        device: torch.device,
        stitching_layer_location: str,
        stitching_layer_config: str,
        resolution: int,
        feedforward_resolution: int,
        stitching_layer_init_path: str = None,
        stitching_module_type: str = "conv3d",
        stitching_hidden_channels: int | None = None,
        stitching_norm: str = "rms",
        stitching_aux_kernel: tuple[int, ...] | None = None,
        stitching_attention_heads: int = 4,
        num_frames: int = 13,
        da3_ref_view_strategy: str = "first",
    ):
        """
        The class to stitch a 3D feedforward model into a diffusion VAE.
        """

        super().__init__()
        # High-level flow:
        # 1) Encode video frames with the diffusion VAE.
        # 2) Upsample VAE latents to match the feedforward model's token grid.
        # 3) Map latents into the feedforward model via a learnable stitching layer.
        self.device = device
        self.num_frames = num_frames
        self.resolution = resolution
        self.feedforward_resolution = feedforward_resolution
        self.stitching_module_type = stitching_module_type
        self.stitching_hidden_channels = stitching_hidden_channels
        self.stitching_norm = stitching_norm
        self.stitching_aux_kernel = stitching_aux_kernel
        self.stitching_attention_heads = stitching_attention_heads
        self.diffusion_vae = diffusion_vae
        self.vae_latent_dimension = self.get_vae_latent_dimension(
            diffusion_vae, resolution
        )
        # feedforward 3d model
        self.stitched_3d_model, self.feedforward_dimension = (
            self.preprocess_feedforward_model(
                feedforward_model,
                stitching_layer_location,
                da3_ref_view_strategy=da3_ref_view_strategy,
            )
        )
        # stitching layer
        (
            self.stitching_layer,
            self.pre_upsample_layer,
        ) = self.get_stitching_layer(
            stitching_layer_config,
            stitching_layer_init_path,
            stitching_module_type=stitching_module_type,
            stitching_hidden_channels=stitching_hidden_channels,
            stitching_norm=stitching_norm,
            stitching_aux_kernel=stitching_aux_kernel,
            stitching_attention_heads=stitching_attention_heads,
        )

    def get_runtime_diffusion_vae(self) -> nn.Module | None:
        runtime_vae = self.__dict__.get("_runtime_diffusion_vae", None)
        if runtime_vae is not None:
            return runtime_vae
        return getattr(self, "diffusion_vae", None)

    def detach_runtime_diffusion_vae(self) -> nn.Module | None:
        runtime_vae = getattr(self, "diffusion_vae", None)
        if runtime_vae is None:
            return self.get_runtime_diffusion_vae()

        self._modules.pop("diffusion_vae", None)
        self.__dict__["diffusion_vae"] = None
        self.__dict__["_runtime_diffusion_vae"] = runtime_vae
        return runtime_vae

    def get_vae_latent_dimension(self, diffusion_vae: nn.Module, resolution: int):
        if _is_supported_wan_vae(diffusion_vae):
            # WAN VAE uses a fixed temporal length (T=13) and 8x spatial downsampling.
            # C=16 is WAN's latent channel dimension. If you swap VAEs, update these.
            T = self.num_frames
            H = resolution // 8
            W = resolution // 8
            C = 16
            vae_latent_dimension = torch.tensor([T, C, H, W])
        else:
            raise NotImplementedError(
                f"VAE latent dimension extraction not implemented for {type(diffusion_vae)}"
            )
        return vae_latent_dimension

    def preprocess_feedforward_model(
        self,
        feedforward_model: nn.Module,
        stitching_layer: str,
        *,
        da3_ref_view_strategy: str = "first",
    ):
        if is_da3_model(feedforward_model):
            model = DA3Stitched(
                feedforward_model,
                stitching_layer,
                ref_view_strategy=da3_ref_view_strategy,
            )
            T = self.num_frames
            C = 1536
            feedforward_dimension = torch.tensor(
                [T, C, self.feedforward_resolution, self.feedforward_resolution]
            )
        else:
            raise NotImplementedError(
                f"Feedforward model preprocessing not implemented for {type(feedforward_model)}"
            )
        return model, feedforward_dimension

    @staticmethod
    def _coerce_image_hw(image_hw) -> tuple[int, int] | None:
        if image_hw is None:
            return None
        if torch.is_tensor(image_hw):
            if image_hw.numel() != 2:
                raise ValueError(f"Expected image_hw tensor with 2 values, got {tuple(image_hw.shape)}")
            flat = image_hw.reshape(-1)
            return int(flat[0].item()), int(flat[1].item())
        if isinstance(image_hw, (tuple, list)) and len(image_hw) == 2:
            return int(image_hw[0]), int(image_hw[1])
        raise ValueError(f"Unsupported image_hw type: {type(image_hw)}")

    @staticmethod
    def _get_spatial_pair(value) -> tuple[int, int]:
        if isinstance(value, int):
            return value, value
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 3:
            return int(value[1]), int(value[2])
        raise ValueError(f"Unsupported convolution parameter: {value}")

    def _resolve_da3_runtime_image_hw(
        self,
        *,
        image_hw=None,
        feedforward_image: torch.Tensor | None = None,
    ) -> tuple[int, int]:
        runtime_image_hw = self._coerce_image_hw(image_hw)
        if runtime_image_hw is not None:
            return runtime_image_hw
        if feedforward_image is not None:
            return int(feedforward_image.shape[-2]), int(feedforward_image.shape[-1])
        return int(self.feedforward_resolution), int(self.feedforward_resolution)

    def get_stitching_layer(
        self,
        stitching_layer_config: str,
        stitching_layer_init_path: str = None,
        *,
        stitching_module_type: str = "conv3d",
        stitching_hidden_channels: int | None = None,
        stitching_norm: str = "rms",
        stitching_aux_kernel: tuple[int, ...] | None = None,
        stitching_attention_heads: int = 4,
    ):
        # The stitching layer maps VAE latents (C=16) to DA3 token channels (C=1024).
        stitching_layer = build_stitching_module(
            module_type=stitching_module_type,
            conv_spec=stitching_layer_config,
            in_channels=int(self.vae_latent_dimension[1]),
            hidden_channels=stitching_hidden_channels,
            norm_type=stitching_norm,
            aux_kernel_size=stitching_aux_kernel,
            attention_heads=stitching_attention_heads,
        )

        target_spatial_size = (
            self.vae_latent_dimension[2].item(),
            self.vae_latent_dimension[3].item(),
        )
        is_da3_stitched = is_da3_model(self.stitched_3d_model)
        kernel_size = stride = padding = dilation = None
        if is_da3_stitched:
            kernel_size = self._get_spatial_pair(stitching_layer.kernel_size)
            stride = self._get_spatial_pair(stitching_layer.stride)
            padding = self._get_spatial_pair(stitching_layer.padding)
            dilation = self._get_spatial_pair(stitching_layer.dilation)
            da3_stitching_input_size = compute_da3_stitching_input_spatial_size(
                feedforward_resolution=self.feedforward_resolution,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                dilation=dilation[0],
            )
            target_spatial_size = (
                da3_stitching_input_size,
                da3_stitching_input_size,
            )

        def upsampling_layer(x, *, image_hw=None, feedforward_image=None):
            # WAN VAE uses temporal stride 4; upsample time back to original length.
            # Spatial size is aligned to the VAE latent grid (resolution//8).
            T_vae = x.shape[2]
            T_original = (T_vae - 1) * 4 + 1
            runtime_target_spatial_size = target_spatial_size
            if is_da3_stitched:
                runtime_image_hw = self._resolve_da3_runtime_image_hw(
                    image_hw=image_hw,
                    feedforward_image=feedforward_image,
                )
                runtime_target_spatial_size = compute_da3_stitching_input_spatial_shape(
                    image_hw=runtime_image_hw,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            x = torch.nn.functional.interpolate(
                x,
                size=[
                    T_original,
                    runtime_target_spatial_size[0],
                    runtime_target_spatial_size[1],
                ],
                mode="trilinear",
                align_corners=True,
            )
            return x

        if stitching_layer_init_path is not None:
            logger.info(f"Loading the stitching layer from {stitching_layer_init_path}")
            # load the weight
            state_dict = torch.load(
                stitching_layer_init_path,
                map_location=self.device,
                weights_only=False,
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            missing_keys = ()
            unexpected_keys = ()
            try:
                incompatible = stitching_layer.load_state_dict(state_dict, strict=True)
                missing_keys = tuple(getattr(incompatible, "missing_keys", ()) or ())
                unexpected_keys = tuple(getattr(incompatible, "unexpected_keys", ()) or ())
            except RuntimeError:
                if not warm_start_stitching_module_from_conv_state(stitching_layer, state_dict):
                    raise
                logger.info(
                    "Loaded legacy conv stitching weights into {} via warm-start fallback.",
                    type(stitching_layer).__name__,
                )
            logger.info(
                "initialization_weight_path stitching_layer load diagnostics: missing_keys_count={} missing_keys_head={} unexpected_keys_count={} unexpected_keys_head={}",
                len(missing_keys),
                list(missing_keys[:20]),
                len(unexpected_keys),
                list(unexpected_keys[:20]),
            )
        return stitching_layer, upsampling_layer

    def vae_encoder_forward(self, images, decode=False):
        """
        The forward function for the VAE encoder
        Args:
            images (torch.Tensor): the input images, shape (B, C, T, H, W)
        Returns:
            latents (torch.Tensor): the latent variables, shape (B, C_out, T_out, H_out, W_out)
        """
        # images are expected in [-1, 1] range and channels-first.
        # we suppose that diffusion_vae follows diffuser's convention.
        runtime_vae = self.get_runtime_diffusion_vae()
        if runtime_vae is None:
            raise RuntimeError("Runtime diffusion VAE is not available for StitchVAE3D.")
        latents = runtime_vae.encode(images).latent_dist.sample()
        if decode:
            feedforward_image = runtime_vae.decode(latents)[0]
        else:
            feedforward_image = None
        return latents, feedforward_image

    def forward(
        self,
        images=None,
        feedforward_image=None,
        train=False,
        *,
        latent=None,
        image_hw=None,
        camera_extrinsics=None,
        camera_intrinsics=None,
        gradient_checkpointing_offload=False,
        output_mode=None,
    ):
        with torch.no_grad():
            if latent is None:
                if images is None:
                    raise ValueError("Either `images` or `latent` must be provided.")
                latent, feedforward_image_ = self.vae_encoder_forward(images, decode=False)
            else:
                feedforward_image_ = None

            if feedforward_image_ is not None:
                feedforward_image = F.interpolate(
                    feedforward_image_,
                    size=(
                        feedforward_image.shape[2],
                        self.feedforward_dimension[2],
                        self.feedforward_dimension[3],
                    ),
                    mode="trilinear",
                    align_corners=True,
                )
            latent = self.pre_upsample_layer(
                latent,
                image_hw=image_hw,
                feedforward_image=feedforward_image,
            )

        stitching_latent = self.stitching_layer(latent)
        final_output = self.stitched_3d_model_forward(
            stitching_latent,
            feedforward_image,
            train=train,
            image_hw=image_hw,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            gradient_checkpointing_offload=gradient_checkpointing_offload,
            output_mode=output_mode,
        )
        if _should_attach_da3_distill_tokens(self.stitched_3d_model):
            final_output = dict(final_output)
            final_output["distill_feature_tokens"] = flatten_da3_feature_grid_to_tokens(
                stitching_latent
            )
        return final_output

    def forward_with_latent(
        self,
        latent,
        feedforward_image=None,
        train=False,
        image_hw=None,
        camera_extrinsics=None,
        camera_intrinsics=None,
        gradient_checkpointing_offload=False,
        output_mode=None,
    ):
        latent = self.pre_upsample_layer(
            latent,
            image_hw=image_hw,
            feedforward_image=feedforward_image,
        )
        stitching_latent = self.stitching_layer(latent)
        final_output = self.stitched_3d_model_forward(
            stitching_latent,
            feedforward_image=feedforward_image,
            train=train,
            image_hw=image_hw,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            gradient_checkpointing_offload=gradient_checkpointing_offload,
            output_mode=output_mode,
        )
        if _should_attach_da3_distill_tokens(self.stitched_3d_model):
            final_output = dict(final_output)
            final_output["distill_feature_tokens"] = flatten_da3_feature_grid_to_tokens(
                stitching_latent
            )
        return final_output

    def stitched_3d_model_forward(
        self,
        latents,
        feedforward_image=None,
        train=False,
        image_hw=None,
        camera_extrinsics=None,
        camera_intrinsics=None,
        gradient_checkpointing_offload=False,
        output_mode=None,
    ):
        if isinstance(self.stitched_3d_model, DA3Stitched):
            output = self.stitched_3d_model(
                latents,
                feedforward_image,
                train=train,
                image_hw=image_hw,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                gradient_checkpointing_offload=gradient_checkpointing_offload,
            )
        else:
            raise NotImplementedError(
                f"Stitched 3D model forward not implemented for {type(self.stitched_3d_model)}"
            )
        return output
